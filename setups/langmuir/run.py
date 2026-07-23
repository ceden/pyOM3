"""
Coupled wave-ocean model.

PE layout (MPI.COMM_WORLD split by color):
  color 0 : wave model   (column,   from pyOM3.wim_simple)
  color 1 : ocean model  (langmuir, from pyOM3.nh_simple)
  color 2 : idle

Both models advance in time from t=0 and exchange Stokes drift /
surface stress through an intercommunicator. The run writes a restart
before the wallclock limit and continues from it on resubmission,
writing output into per-segment files.
"""
# add here pyOM3 code location, here it's $HOME/src/pyOM3/
import os,sys; sys.path.append(os.path.expanduser('~')+'/src/pyOM3/')
import numpy as np
import netCDF4
from time import perf_counter
from mpi4py import MPI

from pyOM3.wim_simple.snapshot import snap_nc4 as wave_base
from pyOM3.nh_simple.snapshot import snap_zarr_gather as ocean_base
from pyOM3.nh_simple.les import les
from coupled_io import (CouplingWriter, segname, read_restart_header,
                        write_wave_restart, load_wave_restart,
                        write_ocean_restart, load_ocean_restart)

wall_t0 = perf_counter()   # wallclock reference, includes setup time


# ===========================================================================
# CONFIGURATION  --  edit run settings here
# ===========================================================================
# PE distribution (wave model uses n_pes_wave PEs, ocean uses the grid below)
n_pes_x_ocean, n_pes_y_ocean = 2, 3  
n_pes_wave = 1

# run control
steps = 3000000          # number of coupled time steps
snapint_ocean = 3600. /4.   # ocean snapshot interval [s], hourly
snapint_wave  = 300.     # wave snapshot AND coupling record interval [s]
    
verbose = False

# profiling: every so many couplings, each side prints its average work
# and wait time per coupling (0 disables). The wave model SHOULD be the
# waiting side; if the ocean waits, 800 PEs idle on one wave column.
profile_interval = 100

# wallclock management: shortly before the SLURM time limit both models
# write a restart and exit cleanly; resubmitting the same script continues
# the run exactly (fields, AB tendency buffers, Stokes baseline, coupling
# products) and writes output into new files suffixed _<segment>.
wallclock_limit  = 8 * 3600.   # [s] SLURM time limit of the job
wallclock_margin = 20 * 60.    # [s] stop this early to write the restart

        
restart_interval = 0           # couplings between periodic restart writes
                               # (0: only at the wallclock exit)
restart_ocean = 'restart_ocean.zarr'
restart_wave  = 'restart_wave.npz'

# model time steps [s]. Ocean dt from the TAIL of the vertical advective
# CFL: the crash forensics located detonations at downwelling-jet cores
# (z ~ -6 m) whose local w-CFL crossed the AB3 stability limit (~0.72)
# while the global max was still ~0.3-0.5. dt = 0.25 keeps the sustained
# max near 0.15 so extreme jets stay sub-critical; 30/0.25 = 120 exact.
# One model day then needs ~10-11 h, i.e. a chain of two jobs via the
# restart machinery.
wave_dt_set = 30.
ocean_dt_set = 0.25

# ocean grid refinement factor
ocean_fac = 8

# initial-condition noise: random velocity perturbation [m/s] in the mixed
# layer, seeding the Langmuir instability; the wave stress is applied clean
ic_noise_amp = 1e-4

# buoyancy advection scheme (parameter switch in nh_simple/model.py:
# superbee flux limiter, else upwind3 if set, else centered 2nd order).
# With superbee, the limiter provides the SGS scalar mixing and the
# explicit Smagorinsky buoyancy diffusion is switched OFF automatically
# to avoid double mixing at the entrainment interface; with the centered
# scheme it stays ON.
b_advection_superbee = True

# subgrid closure:
#   False -> constant harmonic viscosity Ah/Av acting on the prognostic u_L
#   True  -> Smagorinsky closure acting on the EULERIAN velocity u = u_L - u_S
#            (les_mix_eulerian); Ah/Av are dropped, SGS buoyancy diffusion
#            kappa = visc/prandtl is added (there was none before)
use_les = True

# wave-to-ocean momentum transfer:
#   False -> surface wave stress applied only at the top ocean cell
#            (calc_stress, a 2D field taux/tauy)
#   True  -> depth-dependent wave stress work distributed over the water column
#            (calc_stress_work, a 3D body force like the Stokes velocity)
use_stress_work = False

# snapshot output files
wave_snap_file = 'wave.cdf'
ocean_snap_file = 'ocean.zarr'         # single zarr store, gathered and written by PE0
coupling_snap_file = 'coupling.cdf'    # Stokes vel. + wind stress exchanged each coupling
# ===========================================================================


# ---------------------------------------------------------------------------
# wave model
# ---------------------------------------------------------------------------
class column(wave_base):

    def set_initial(self, p):
        # as in the fig3 notebook: seed E in the interior only
        self.E[1:-1, 1:-1, 1:-1, 1:-1] = 1e-2
        self.A = self.E / self.omega_t[None, :, None, None]
        self.ustar = np.ones_like(self.E[0, 0, :, :]) * p['ustar0']
        self.udir = np.ones_like(self.E[0, 0, :, :]) * p['dir']
        return


def get_wave_parameter():
    p1 = {'nx': 1, 'ny': 1, 'nf': 62, 'nd': 80, 'dx': 1e3, 'dy': 1e3, 'dt': 30,
          'f0': 0.035, 'df0': 0.05, 'periodic_in_x': True, 'periodic_in_y': True,
          'mask_energy': True, 'limit_growth': True, 'tail_flux': 'input',
          'snonlin_backend': 'fortran', 'implicit_dissipation_only': True,
          'snapint': 30 * 200}
    p2 = {'ustar0': 0.35, 'dir': np.pi / 2}   # as in the fig3 notebook
    return p1, p2


# ---------------------------------------------------------------------------
# ocean model (3D model with vortex force and wind stress, McWilliams et al.)
# ---------------------------------------------------------------------------
class langmuir(les, ocean_base):

    def set_initial(self, N0):
        """setup initial stratification"""
        z = np.arange(self.nz) * self.dz
        z = z - (z[-1] + self.dz / 2)
        sl = slice(self.zs_pe - 1, self.ze_pe)
        #self.b[:, :, self.halo:-self.halo] = np.where(
        #    z[None, None, sl] < -self.Lz / 2., z[None, None, sl] * N0**2, 0)
        self.b[:, :, self.halo:-self.halo] = np.minimum( (z[None, None, sl]+self.Lz/2) * N0**2, 0)
        self.b = self.apply_bc(self.b)

        # random velocity noise in the mixed layer only (the stratified
        # interior stays unperturbed) to seed the Langmuir instability;
        # per-PE independent draws are fine for noise, apply_bc makes the
        # halos consistent, and the first pressure solve projects the
        # perturbation divergence-free. Replaces the former noise factor
        # on the surface stress, which is now purely the wave-model stress.
        mask = np.zeros(self.u.shape)
        mask[:, :, self.halo:-self.halo] = np.where(
            z[None, None, sl] < -self.Lz / 2., 0., 1.)
        self.u = self.apply_bc(self.u + ic_noise_amp * mask
                               * rng.standard_normal(self.u.shape))
        self.v = self.apply_bc(self.v + ic_noise_amp * mask
                               * rng.standard_normal(self.v.shape))
        return

    def set_forcing(self, tau):
        """apply wave stress (surface-only or depth-dependent), vortex and
        gradient force"""
        if self.use_stress_work:
            # depth-dependent wave stress work: a body force at every level,
            # already distributed over the column (no division by dz).
            # tau_work_x, tau_work_y come at tracer points, so average onto the
            # velocity points like u_S/v_S: x-component in x, y-component in y.
            self.du[tau, ...] += 0.5 * (self.tau_work_x + self.rollx(self.tau_work_x, -1))
            self.dv[tau, ...] += 0.5 * (self.tau_work_y + self.rolly(self.tau_work_y, -1))
        elif self.my_blk_z == self.n_pes_z:
            # surface wave stress applied to the top ocean cell only
            self.du[tau, :, :, -(self.halo + 1)] += self.taux / self.dz
            self.dv[tau, :, :, -(self.halo + 1)] += self.tauy / self.dz

        # vortex force in the u_L formulation. Prognostic velocity is
        # u_L = u + u_S; the CL momentum equation then reads
        #   dt u_L + (u_L.grad) u_L + f k x u_L = -grad pi'
        #           + dt u_S - u_L x (curl u_S)
        # (Coriolis-Stokes force absorbed by Coriolis acting on u_L; the
        # Eulerian vortex force u_S x (curl u) turns into the Stokes-shear
        # force -u_L x (curl u_S), and the Bernoulli terms move into pi').
        # With horizontally uniform u_S(z):  curl u_S = (-dz v_S, dz u_S, 0),
        #   F_x = +w dz(u_S),  F_y = +w dz(v_S),  F_z = -(u dz(u_S)+v dz(v_S)),
        # so F.u_L = 0: NO energy exchange through the vortex force, all
        # wave->mean energy enters via dt u_S (and the stress). The averaging
        # below pairs transposed operators (rollx -1 on w vs rollx +1 on u,
        # z-average W->T vs T->W), so the discrete work sums to zero exactly
        # in periodic x,y.
        dzus = (self.rollz(self.u_S, -1) - self.u_S) / self.dz   # at W faces
        dzvs = (self.rollz(self.v_S, -1) - self.v_S) / self.dz
        dzus = self.apply_vertical_flux_bc(dzus)
        dzvs = self.apply_vertical_flux_bc(dzvs)

        px = 0.5 * (self.w + self.rollx(self.w, -1)) * dzus      # at U-W edges
        py = 0.5 * (self.w + self.rolly(self.w, -1)) * dzvs      # at V-W edges
        self.du[tau, ...] += 0.5 * (px + self.rollz(px, 1))      # -> U points
        self.dv[tau, ...] += 0.5 * (py + self.rollz(py, 1))      # -> V points

        ut = 0.5 * (self.u + self.rollx(self.u, 1))              # U -> T
        vt = 0.5 * (self.v + self.rolly(self.v, 1))              # V -> T
        self.dw[tau, ...] += -(0.5 * (ut + self.rollz(ut, -1)) * dzus
                               + 0.5 * (vt + self.rollz(vt, -1)) * dzvs)
        self.dw[tau, ...] = self.apply_vertical_flux_bc(self.dw[tau, ...])

        # Smagorinsky mixing of the Eulerian velocity u = u_L - u_S and of b:
        # direct forward-Euler update of the prognostic fields (like the
        # les_closure slot in ps3D: after the AB tendencies are assembled,
        # before the AB step), so the vortex-force tendency above still uses
        # the same fields as the advective tendencies computed earlier.
        if self.enable_les:
            self.les_closure()
        return

    def set_forcing_direct(self):
        """Stokes increment of the u_L formulation: plain forward increment
        u_L += dt*dt_us OUTSIDE the AB weighting, constant over the ocean
        sub-cycles of one coupling interval, so the increments telescope to
        u_S(t2)-u_S(t1) exactly (see wim.interface.calc_dt_us)."""
        self.u += self.dt_us * self.dt
        self.v += self.dt_vs * self.dt
        return


def get_ocean_parameter(fac=1):
    """returns all parameters for setup of ocean model"""
    nx, ny, nz = 40 * fac, 40 * fac, 40 * fac
    Lx, Ly, Lz = 200., 200., 50.
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz   # fac=8: dx=dy=0.625, dz=0.15625 m

    # La^2 = Ustar/U_s, Ustar^2 = tau_0/rho, tau_0 = La^4 U_s^2 rho
    N0_loc = 0.0044

    p1 = {'nx': nx, 'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz, 'dt': 10. / fac,
          'f0': 1e-4, 'N0': 0e-4, 'periodic_in_z': False, 'Ro': 1., 'dsqr': 1.,
          'buoyancy_superbee_advection': b_advection_superbee,
          'snapint': 200 * 10. / fac}
    if use_les:
        # Smagorinsky on the Eulerian velocity replaces the constant Ah/Av
        # (which acted on u_L, mixing the Stokes shear as a side effect);
        # min_smag = 0: pure Smagorinsky, viscosity vanishes where the
        # resolved Eulerian strain does (quiescent stratified interior)
        # scale-selective horizontal biharmonic against grid-scale noise:
        # the pre-crash diagnostic showed ~70% of near-surface w-variance
        # below 4 dx (centered advection dispersion, Smagorinsky response
        # too weak at 2 dx) while the explicit stability budget was only
        # 6% used. Ahbi = dx^4/(tau_damp pi^4) damps 2 dx with e-folding
        # tau_damp while scales > 8 dx are barely touched. Horizontal-only
        # (Avbi = 0): with horizontally uniform u_S the operator acts
        # identically on u_L and u = u_L - u_S, so the friction-on-u
        # consistency of the LES is preserved exactly.
        tau_damp = 30.                        # [s] e-folding of 2 dx noise
        dx_o = 200. / (40 * ocean_fac)
        # Peclet bound: the crash forensics showed single-cell w spikes
        # fed by Langmuir convergence lines collapsing to grid scale;
        # the bound enforces nu >= |u_h| sqrt(dx dy)/Peclet_limit at the
        # convergence lines (cap-safe: re-capped inside les.py), limiting
        # their collapse and the extremes they inject.
        p1.update({'enable_les': True, 'enable_smagorinsky': True,
                   'les_mix_eulerian': True, 'smag_coeff': 0.148,
                   'prandtl': 0.47, 'min_smag': 0.0,
                   'enable_peclet_bound': True, 'Peclet_limit': 50.0,
                   'les_mix_b': not b_advection_superbee,})
                   #'Ahbi': dx_o**4 / (tau_damp * np.pi**4), 'Avbi': 0.0})
    else:
        p1.update({'enable_les': False,
                   'Ah': 0.001 / fac**2, 'Av': 0.001 / fac**2})
    p2 = {'N0_loc': N0_loc}
    return p1, p2


# ---------------------------------------------------------------------------
# coupling-parameter output (Stokes velocity + wind stress) to a single zarr
# ---------------------------------------------------------------------------
# PE distribution and communicators
# ---------------------------------------------------------------------------
n_pes_ocean = n_pes_x_ocean * n_pes_y_ocean
world = MPI.COMM_WORLD
my_pe, n_pes = world.Get_rank(), world.Get_size()

color = 0 if my_pe < n_pes_wave else (1 if my_pe < n_pes_wave + n_pes_ocean else 2)
comm = world.Split(color, key=0)
if color == 0:
    intercomm = comm.Create_intercomm(0, world, n_pes_wave, tag=42)
elif color == 1:
    intercomm = comm.Create_intercomm(0, world, 0, tag=42)
else:
    intercomm = None

ss = np.random.SeedSequence([12345, my_pe])   # unique RNG on each PE
rng = np.random.default_rng(ss)


# ---------------------------------------------------------------------------
# restart detection (world root reads the wave restart header) and
# per-segment output file names
# ---------------------------------------------------------------------------
restarting = os.path.exists(restart_wave) and os.path.exists(restart_ocean)
if my_pe == 0:
    hdr = read_restart_header(restart_wave) if restarting else None
    n0, seg = (hdr[0] + 1, hdr[1] + 1) if hdr is not None else (0, 0)
else:
    n0, seg = 0, 0
restarting, n0, seg = world.bcast((restarting, n0, seg), root=0)

wave_snap_file     = segname(wave_snap_file, seg)
ocean_snap_file    = segname(ocean_snap_file, seg)
coupling_snap_file = segname(coupling_snap_file, seg)
if my_pe == 0:
    print('restarting =', restarting, ' segment =', seg, ' first coupling n =', n0)


# ---------------------------------------------------------------------------
# setup the two models
# ---------------------------------------------------------------------------
if color == 0:
    p1, p2 = get_wave_parameter()
    p1['snapint'] = snapint_wave
    p1['dt'] = wave_dt_set
    M = column(p1, comm_in=comm)
    M.setup(snap_file_name=wave_snap_file, show_each_time_step=False)
    M.set_initial(p2)

elif color == 1:
    p1, p2 = get_ocean_parameter(fac=ocean_fac)
    p1['snapint'] = snapint_ocean
    p1['dt'] = ocean_dt_set
    M = langmuir(p1, n_pes_x=n_pes_x_ocean, n_pes_y=n_pes_y_ocean, comm_in=comm)
    M.use_stress_work = use_stress_work
    M.setup(snap_file_name=ocean_snap_file, show_each_time_step=True)
    M.set_initial(p2['N0_loc'])
    M.u_S = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
    M.v_S = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
    M.dt_us = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
    M.dt_vs = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
    if use_stress_work:
        M.tau_work_x = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
        M.tau_work_y = np.zeros((M.Nx, M.Ny, M.Nz), np.float64)
    else:
        M.taux = np.zeros((M.Nx, M.Ny), np.float64)
        M.tauy = np.zeros((M.Nx, M.Ny), np.float64)

world.Barrier()


# ---------------------------------------------------------------------------
# communicate time steps; ocean dt must be a fraction of wave dt
# ---------------------------------------------------------------------------
if color == 0:   # wave model
    if comm.Get_rank() == 0:
        intercomm.send(M.dt, dest=0, tag=1)
        ocean_dt = intercomm.recv(source=0, tag=2)
        ocean_dz = intercomm.recv(source=0, tag=3)
        ocean_nz = intercomm.recv(source=0, tag=4)
        coupling_writer = CouplingWriter(coupling_snap_file, ocean_nz, use_stress_work)

elif color == 1:   # ocean model
    if M.my_pe == 0:
        wave_dt = intercomm.recv(source=0, tag=1)
        intercomm.send(M.dt, dest=0, tag=2)
        intercomm.send(M.dz, dest=0, tag=3)
        intercomm.send(M.nz, dest=0, tag=4)

        if wave_dt < M.dt:
            raise RuntimeError('wave time step is not larger than ocean time step')
        sub_cycle = int(round(wave_dt / M.dt))
        if abs(sub_cycle * M.dt - wave_dt) > 1e-9 * wave_dt:
            raise RuntimeError('wave time step is not a multiple of ocean time step')
    else:
        sub_cycle = 0

    sub_cycle = comm.bcast(sub_cycle, root=0)
    if M.my_pe == 0:
        print('ocean model sub cycles', sub_cycle, 'times before coupling')

world.Barrier()


# ---------------------------------------------------------------------------
# distribution of the coupling products to the ocean PEs; also used to
# replay the last exchange when loading a restart
# ---------------------------------------------------------------------------
def apply_coupling_products(M, u_S, stress, dt_us):
    u_S = M.comm.bcast(u_S, root=0)
    zsl = slice(M.zs_pe - 1, M.ze_pe)
    M.u_S[:, :, M.halo:-M.halo] = u_S[0, zsl, 0, 0][None, None, :]
    M.v_S[:, :, M.halo:-M.halo] = u_S[1, zsl, 0, 0][None, None, :]
    M.u_S, M.v_S = M.apply_bc(M.u_S), M.apply_bc(M.v_S)
    dt_us = M.comm.bcast(dt_us, root=0)
    M.dt_us[:, :, M.halo:-M.halo] = dt_us[0, zsl, 0, 0][None, None, :]
    M.dt_vs[:, :, M.halo:-M.halo] = dt_us[1, zsl, 0, 0][None, None, :]
    M.dt_us, M.dt_vs = M.apply_bc(M.dt_us), M.apply_bc(M.dt_vs)
    stress = M.comm.bcast(stress, root=0)
    if use_stress_work:
        # depth-dependent: distribute over the column like u_S/v_S
        M.tau_work_x[:, :, M.halo:-M.halo] = stress[0, zsl, 0, 0][None, None, :]
        M.tau_work_y[:, :, M.halo:-M.halo] = stress[1, zsl, 0, 0][None, None, :]
        M.tau_work_x = M.apply_bc(M.tau_work_x)
        M.tau_work_y = M.apply_bc(M.tau_work_y)
    else:
        # surface stress: one clean value from the wave model per column
        M.taux[:, :] = stress[0, 0, 0]
        M.tauy[:, :] = stress[1, 0, 0]


# ---------------------------------------------------------------------------
# time stepping
# ---------------------------------------------------------------------------
def ab_coeffs(tA, tB, tC):
    """advance Adam-Bashforth coefficients (Euler -> AB2 -> AB3)"""
    if tA == 1. and tB == 0. and tC == 0.:
        return 1.5, -0.5, 0.
    if tA == 1.5 and tB == -0.5 and tC == 0.:
        return 23. / 12., -16. / 12., 5. / 12.
    return tA, tB, tC


if color == 0:   # wave model

    tA, tB, tC = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    next_snap = snapint_wave   # next threshold for coupling output
    t_work = t_wait = 0.0

    if restarting:
        tau, taum1, taum2, tA, tB, tC, next_snap = \
            load_wave_restart(M, restart_wave)
        print('wave : restart loaded, continuing at n=', n0)
    else:
        # no wave-only spinup (as in the fig3 notebook): the wave field
        # grows coupled from its small initial seed. Prime the Stokes-
        # tendency baseline at ZERO so the increments telescope to
        # u_S(t) - 0 exactly, i.e. u_L = u + u_S from the first coupling
        # on (u_S(0) is tiny).
        M.ust_last = 0.0 * M.calc_stokes(ocean_dz, ocean_nz)

    # coupled run
    for n in range(n0, steps):
        t0 = perf_counter()
        M.time_step(tA, tB, tC, tau, taum1, taum2)
        M.diagnose(n, tau)
        tau, taum1, taum2 = np.mod(tau + 1, 3), np.mod(taum1 + 1, 3), np.mod(taum2 + 1, 3)
        tA, tB, tC = ab_coeffs(tA, tB, tC)

        # exchange coupling data
        dt_us, u_S = M.calc_dt_us(ocean_dz, ocean_nz)
        if use_stress_work:
            stress = M.calc_stress_work(ocean_dz, ocean_nz)
        else:
            stress = M.calc_stress()
        t_work += perf_counter() - t0
        if comm.Get_rank() == 0:
            if verbose:
                print('wave: now coupling at n=', n)
            t0 = perf_counter()
            u = intercomm.recv(source=0, tag=1)   # blocks until the ocean
            t_wait += perf_counter() - t0         # finished its sub-cycles
            v = intercomm.recv(source=0, tag=2)
            intercomm.send(u_S, dest=0, tag=3)
            intercomm.send(stress, dest=0, tag=4)
            intercomm.send(dt_us, dest=0, tag=5)

            # write coupling parameters to zarr on snapint boundaries
            coupled_time = (n + 1) * M.dt
            if coupled_time >= next_snap:
                coupling_writer.write(coupled_time, u_S, stress, dt_us)
                next_snap += snapint_wave

            # ocean sends u_L directly (prognostic variable of the u_L
            # formulation): no u + u_S composition needed for the Doppler
            # velocities. propagation.py reads ocean_U/V and ocean_Udf/Vdf
            # (frequency derivative from calc_doppler_vel_df); the previous
            # manual time derivative ocean_dUdt was never read by WIM.
            u = np.reshape(u, (ocean_nz, M.ny, M.nx))
            v = np.reshape(v, (ocean_nz, M.ny, M.nx))
            M.ocean_U, M.ocean_V = M.calc_doppler_vel(ocean_dz, ocean_nz, u, v)
            M.ocean_Udf, M.ocean_Vdf = \
                M.calc_doppler_vel_df(ocean_dz, ocean_nz, u, v)

            if verbose:
                print('wave: coupling done')

        t0 = perf_counter()
        world.Barrier()   # one global barrier per wave model time step
        t_wait += perf_counter() - t0

        if profile_interval and (n + 1) % profile_interval == 0 \
                            and comm.Get_rank() == 0:
            print('profile wave : %6.3f s work  %6.3f s waiting  per coupling'
                  % (t_work / profile_interval, t_wait / profile_interval))
            t_work = t_wait = 0.0

        # stop code, agreed by ALL world ranks: 0 continue, 1 wallclock,
        # 2 ocean crash (no restart written -- keep any earlier good one)
        wc = perf_counter() - wall_t0 > wallclock_limit - wallclock_margin
        stop = world.allreduce(1 if wc else 0, op=MPI.MAX)
        if stop == 2:
            print('wave : ocean crashed, exiting at n=', n)
            break
        if stop == 1 or (restart_interval and (n + 1) % restart_interval == 0):
            if comm.Get_rank() == 0:   # restart assumes n_pes_wave == 1
                write_wave_restart(M, restart_wave, seg, n, tau, taum1,
                                   taum2, tA, tB, tC, next_snap)
        if stop == 1:
            print('wave : wallclock limit reached, exiting cleanly at n=', n)
            break
    print('wave done')

elif color == 1:   # ocean model

    tA, tB, tC = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    t_work = t_wait = 0.0

    if restarting:
        (tau, taum1, taum2, tA, tB, tC), cpl = \
            load_ocean_restart(M, restart_ocean, n0 - 1)
        # replay the distribution of the last exchanged coupling products,
        # so the first restarted interval is forced identically
        apply_coupling_products(M, *cpl)
        if M.my_pe == 0:
            print('ocean: restart loaded, continuing at n=', n0)

    _local_dumped = False
    for n in range(n0, steps):
        t0 = perf_counter()
        for m in range(sub_cycle):
            M.time_step(tA, tB, tC, tau, taum1, taum2)
            M.diagnose(m + n * sub_cycle, tau)
            tau, taum1, taum2 = np.mod(tau + 1, 3), np.mod(taum1 + 1, 3), np.mod(taum2 + 1, 3)
            tA, tB, tC = ab_coeffs(tA, tB, tC)

            # per-step LOCAL detonation catch: each PE checks only its own
            # slab (no communication); the first bad step dumps the local
            # state at most ONE step after onset, before the coupling-level
            # (collective) machinery stops the run. Which PEs dump plus the
            # slab content localizes the onset in space.
            if not _local_dumped:
                lmax = float(np.max(np.abs(M.w)))
                if not np.isfinite(lmax) or lmax * M.dt / M.dz > 0.7:
                    _local_dumped = True
                    np.savez('crash_local_pe%04d.npz' % M.my_pe,
                             n=n, m=m, my_blk_y=M.my_blk_y,
                             my_blk_z=M.my_blk_z,
                             ys=M.ys_pe, ye=M.ye_pe, zs=M.zs_pe, ze=M.ze_pe,
                             u=M.u, v=M.v, w=M.w, b=M.b)
                    print('ocean pe %d: LOCAL DUMP at n=%d m=%d '
                          '(blk_y=%d blk_z=%d, max|w| CFL=%.2f)'
                          % (M.my_pe, n, m, M.my_blk_y, M.my_blk_z,
                             lmax * M.dt / M.dz), flush=True)

        # CFL / health monitor: one elementwise-MAX allreduce per coupling.
        # Fourth entry: the vortex-force rotation number S*dt with
        # S = max |dz u_S| -- the vortex force is skew (exactly zero
        # discrete work) but under AB3 it is only stable for
        # S*dt <~ 0.72 (imaginary-axis limit); interior slab only, the
        # zeroed z-halos at the lids would fake a jump.
        h = M.halo
        S = max(np.max(np.abs(np.diff(M.u_S[h, h, h:-h]))),
                np.max(np.abs(np.diff(M.v_S[h, h, h:-h])))) / M.dz
        cfl_loc = np.array([np.max(np.abs(M.u)) * M.dt / M.dx,
                            np.max(np.abs(M.v)) * M.dt / M.dy,
                            np.max(np.abs(M.w)) * M.dt / M.dz,
                            S * M.dt])
        cfl = np.zeros(4)
        M.comm.Allreduce(cfl_loc, cfl, op=MPI.MAX)
        crash = bool(not np.all(np.isfinite(cfl)) or cfl[:3].max() > 0.9)
        if M.my_pe == 0 and (crash or cfl[:3].max() > 0.5 or cfl[3] > 0.6):
            print('OCEAN ALARM n=%d  CFL=(%.2f, %.2f, %.2f)  Sdt=%.2f'
                  % (n, cfl[0], cfl[1], cfl[2], cfl[3]), flush=True)

        # exchange coupling data
        gathered_u = M.gather_array_mpi(M.u)
        gathered_v = M.gather_array_mpi(M.v)
        t_work += perf_counter() - t0
        if M.my_pe == 0:
            ax = 0 if M.fortran_ordering else -1
            gathered_u = np.sum(np.sum(gathered_u, axis=ax), axis=ax) / (M.nx * M.ny)
            gathered_v = np.sum(np.sum(gathered_v, axis=ax), axis=ax) / (M.nx * M.ny)
            if verbose:
                print('ocean: now coupling at n=', n)
            intercomm.send(gathered_u, dest=0, tag=1)
            intercomm.send(gathered_v, dest=0, tag=2)
            t0 = perf_counter()
            u_S = intercomm.recv(source=0, tag=3)   # blocks while the wave
            t_wait += perf_counter() - t0           # model still computes
            stress = intercomm.recv(source=0, tag=4)
            dt_us = intercomm.recv(source=0, tag=5)
            if verbose:
                print('ocean: coupling done')
        else:
            u_S, stress, dt_us = 0, 0, 0

        # broadcast coupling data and transfer to local arrays
        apply_coupling_products(M, u_S, stress, dt_us)

        t0 = perf_counter()
        world.Barrier()   # one global barrier per wave model time step
        t_wait += perf_counter() - t0

        if profile_interval and (n + 1) % profile_interval == 0 \
                            and M.my_pe == 0:
            print('profile ocean: %6.3f s work  %6.3f s waiting  per coupling'
                  '  CFL=(%.3f, %.3f, %.3f)  Sdt=%.3f'
                  % (t_work / profile_interval, t_wait / profile_interval,
                     cfl[0], cfl[1], cfl[2], cfl[3]))
            t_work = t_wait = 0.0

        # stop code, agreed by ALL world ranks (LOR-style via MAX):
        # 0 continue, 1 wallclock (write restart), 2 crash (dump state,
        # do NOT write a NaN restart)
        stop = world.allreduce(2 if crash else 0, op=MPI.MAX)
        if stop == 2:
            # forensic dump of the full 3D state at the first bad coupling
            write_ocean_restart(M, segname('crash_state.zarr', seg), n,
                                tau, taum1, taum2, tA, tB, tC,
                                u_S, stress, dt_us)
            if M.my_pe == 0:
                print('ocean: CFL/NaN crash at n=%d, state dumped, exiting'
                      % n, flush=True)
            break
        if stop == 1 or (restart_interval and (n + 1) % restart_interval == 0):
            write_ocean_restart(M, restart_ocean, n, tau, taum1, taum2,
                                tA, tB, tC, u_S, stress, dt_us)
        if stop == 1:
            if M.my_pe == 0:
                print('ocean: wallclock limit reached, exiting cleanly at n=', n)
            break
    if M.my_pe == 0:
        print('ocean done')

else:   # idle PEs: must join every world-collective of the coupled loop
    for n in range(n0, steps):
        world.Barrier()
        stop = world.allreduce(0, op=MPI.MAX)
        if stop:
            break

world.Barrier()
if my_pe == 0:
    print('all done, bye')
