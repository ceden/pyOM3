"""
acc1 experiment physics: grid and forcing constants, the storm wind field,
and the wave / ocean model classes (topography, initial state, forcing with
the Craik-Leibovich terms, and the energy-diagnostic methods).
Swap this module to run a different configuration with the same coupler.
"""
import numpy as np
import pyOM3 as OM
from pyOM3.wim_simple.snapshot        import snap_nc4 as wave_base
from pyOM3.pe_simple.snapshot  import snap_nc4 as ocean_base
from pyOM3.pe_simple.tke       import tke as tke_mixin
from mpi4py import MPI


# acc1 resolution / physics (shared horizontal grid for BOTH models)
hRESOLVE = 1.0
vRESOLVE = 5.0
t_rest   = 30 * 86400.0
phi0     = -25.0 / 180.0 * np.pi
omega    = 7.292123516990375e-05
radius   = 6370.0e3


nx = int(128 * hRESOLVE)
ny = int(128 * hRESOLVE)
nz = int(18 * vRESOLVE)
dx = 20e3 / hRESOLVE
dy = 20e3 / hRESOLVE
dz = 50.0 / vRESOLVE
L_x, L_y = nx * dx, ny * dy

# wind (as in acc1: kinematic ustar amplitude, sinusoidal over southern half).
# The wave model is forced by ustar/udir. Storm-train forcing (identical to the
# standalone wim_acc1.py): eastward-propagating Southern-Ocean storms confined
# to the southern half of the channel.
ustar0    = 0.75             # storm-core friction velocity [m/s]
ustar_bg  = 0.05             # calm background [m/s]
z0_val    = 0.00022
n_storms  = 3                # storm cells around the channel
c_storm   = 12.0             # eastward storm propagation speed [m/s]
L_storm_x = 300e3            # zonal storm scale [m]
L_storm_y = 350e3            # meridional storm scale [m]
y_track   = 0.25             # storm-track latitude as fraction of L_y

# run control
# The WAVE model is the fast, CFL-limited model; the OCEAN model is slow.
# The wave sub-cycles ocean_dt/wave_dt times per ocean step and coupling
# happens once per ocean step. ocean_dt MUST be a positive integer multiple
# of wave_dt (checked below at the dt handshake).
wave_dt  = 200.0 / hRESOLVE   # wave time step [s]  (AB3-stability limited, CFL~0.22)
ocean_dt = 600.0 / hRESOLVE   # acc1 ocean time step [s] (= 2400 s) -> 6 wave sub-steps
snapint     = 86400.*3


# wave->ocean momentum transfer: 'stress' (2D surface) or 'stress_work' (3D)
transfer_mode = 'stress'       # 'stress' or 'stress_work'

# ocean numerics (acc1)
Ahbi  = 5e11 / hRESOLVE**4
kappa_T = 1e-4 #/ vRESOLVE
A_v     = 1e-4 #/ vRESOLVE
r_bot   = 1e-5 * vRESOLVE

wave_snap_file  = 'wave_acc1.cdf'
ocean_snap_file = 'ocean_acc1.cdf'
halo = 1

# ===========================================================================


use_stress_work = (transfer_mode == 'stress_work')


# ---------------------------------------------------------------------------
# helper: acc1 global fields (identical to the standalone acc1 config)
# ---------------------------------------------------------------------------
def yt_glob(): return (0.5 + np.arange(ny)) * dy
def xt_glob(): return (0.5 + np.arange(nx)) * dx


def acc1_bstar(y):
    """acc1 target surface buoyancy, in pe_simple's positive-buoyancy sign
    convention (db = +30e-3 -> densest at the channel edges)."""
    db = 30e-3
    bstar = np.full_like(y, db)
    south = y < L_y / 2.0
    bstar = np.where(south, db * y / (L_y / 2.0), bstar)
    y2 = L_y * 0.75
    north = y > y2
    bstar = np.where(north, db * (1.0 - (y - y2) / (L_y - y2)), bstar)
    return bstar


def storm_forcing(xg, yg, t):
    """
    Eastward-propagating Southern-Ocean storm train (smooth vector blend),
    identical to the standalone wim_acc1.py.

    xg, yg are THIS PE's slices of the GLOBAL cell-centre coordinates [m], so
    building the field with meshgrid from them is correct on any PE layout.
    A row of n_storms cyclonic lows on a storm track at y_track*L_y translates
    eastward at c_storm, wrapping periodically in x. The wind is a VECTOR blend
    of cyclonic rotation plus a solid eastward push (no pure spin centre -> no
    direction singularity), confined to the southern half of the channel.
    Returns ustar(ny_loc,nx_loc), udir(ny_loc,nx_loc).
    """
    X, Y = np.meshgrid(xg, yg)                       # (ny_loc, nx_loc)
    yc = y_track * L_y
    u_e = np.full_like(X, ustar_bg)                  # eastward (+x) component
    v_n = np.zeros_like(X)                           # northward (+y) component
    rot  = 0.6 * ustar0
    push = 0.6 * ustar0
    for s in range(n_storms):
        xc  = np.mod(L_x * (s + 0.5) / n_storms + c_storm * t, L_x)
        dxp = (X - xc + L_x / 2.0) % L_x - L_x / 2.0
        dyp = Y - yc
        amp = np.exp(-((dxp / L_storm_x) ** 2 + (dyp / L_storm_y) ** 2))
        # softened radius: no direction singularity at the core. 1.5*L_storm_x
        # keeps the wind-direction rotation below one direction bin per grid
        # cell (0.8*L gave ~19 deg/cell > the 15 deg bin, an under-resolved
        # forcing feature that seeds grid-scale structure).
        rr  = np.sqrt(dxp ** 2 + dyp ** 2 + (1.5 * L_storm_x) ** 2)
        u_e = u_e + amp * (rot * dyp / rr + push)
        v_n = v_n - amp * (rot * dxp / rr)
    ustar = np.sqrt(u_e ** 2 + v_n ** 2)
    udir  = np.arctan2(u_e, v_n)                      # theta=0->north, pi/2->east
    # wind confined to the southern half as in acc1.py, but with a SMOOTH
    # taper (~4 cells wide) instead of a hard step: the sharp y-cutoff is an
    # under-resolved forcing edge that seeds a 2dx mode in the wave field.
    taper = 0.5 * (1.0 - np.tanh((Y - L_y / 2.0) / (2 * dy)))
    ustar = ustar * taper
    return ustar, udir


# ---------------------------------------------------------------------------
# WAVE model : full acc1 grid, reflective BCs, wind from acc1 ustar profile
# ---------------------------------------------------------------------------
class wave(wave_base):

    def set_topography(self):
        """acc1 ridge land mask (same geometry as acc1.py), wim convention.
        Land where y > L_y/2 AND (x > 0.75 L_x or x < 0.25 L_x): two blocks in
        the northern half at the east and west quarters. wim uses kbot = 1 in
        ocean, 0 on land. Built per-PE from global coordinates via meshgrid."""
        xc_glob = (0.5 + np.arange(self.nx)) * self.dx
        yc_glob = (0.5 + np.arange(self.ny)) * self.dy
        sx = slice(self.xs_pe - 1, self.xe_pe)
        sy = slice(self.ys_pe - 1, self.ye_pe)
        X, Y = np.meshgrid(xc_glob[sx], yc_glob[sy])      # (ny_loc, nx_loc)

        land = (Y > L_y / 2.0) & ((X > L_x * 0.75) | (X < L_x * 0.25))
        ocean = np.where(land, 0.0, 1.0)                  # 1 ocean, 0 land

        kbot = np.zeros((self.Ny, self.Nx), np.float64)
        sl = slice(self.halo, -self.halo)
        kbot[sl, sl] = ocean
        self.kbot = self.apply_bc(kbot)   # halo-exchanged, as in wim_acc1.py
        return

    def set_initial(self):
        # allocate forcing fields and set them at t=0
        self.ustar = np.zeros((self.Ny, self.Nx), np.float64)
        self.udir  = np.zeros((self.Ny, self.Nx), np.float64)
        self.z0    = np.full((self.Ny, self.Nx), z0_val, np.float64)
        self.update_forcing(0.0)
        # small seed spectrum only where there is wind (unforced north stays 0)
        seed = np.where(self.ustar[None, None, :, :] > 0.0, 1e-6, 0.0)
        self.E[:] = seed
        self.A = self.E / self.omega_t[None, :, None, None]
        return

    def update_forcing(self, t):
        """Set ustar/udir from the eastward-propagating storm train at time t.
        Called every wave sub-step so the storms translate through the domain."""
        xc_glob = (0.5 + np.arange(self.nx)) * self.dx
        yc_glob = (0.5 + np.arange(self.ny)) * self.dy
        sx = slice(self.xs_pe - 1, self.xe_pe)
        sy = slice(self.ys_pe - 1, self.ye_pe)
        us_loc, ud_loc = storm_forcing(xc_glob[sx], yc_glob[sy], t)

        sl = slice(self.halo, -self.halo)
        self.ustar[sl, sl] = us_loc
        self.udir[sl, sl]  = ud_loc
        self.ustar = self._apply_bc_2d(self.ustar)
        self.udir  = self._apply_bc_2d(self.udir)
        return

    # --- diagnostic: wind -> wave energy input ------------------------------
    def wind_work(self):
        """
        Domain-integrated kinematic energy input rate of the wind to the wave
        field, P_in = g * int gamma_in E dtheta df dA  [m^5/s^3 per unit rho;
        multiply by rho_w for Watts]. Uses the gamma_in of the LAST time_step.
        Collective over the wave communicator (call on all wave PEs).
        """
        h = self.halo
        sl = slice(h, -h)
        p = float(self.grav * np.sum(
                (self.gamma_in * self.E)[sl, sl, sl, sl]
                * self.dd * self.dft[None, sl, None, None]) * self.dx * self.dy)
        # analytic beyond-grid tail input (tail_flux='input'): the wind pumps
        # energy directly into f > f_grid; Phi_oc carries it (equilibrium
        # booking), so the input book must too -- otherwise 'dissipation'
        # persistently exceeds 'wind work' by exactly this share and the
        # energy diag looks like creation. Same closed form as calc_phioc.
        if hasattr(self, '_tail_input_factors'):
            _, fac_phi, _, _ = self._tail_input_factors()
            Ecg = self.E[:, -self.halo, :, :]
            p += float(np.sum((Ecg * fac_phi * self.dd)[sl, sl, sl])
                       * self.dx * self.dy)
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            p = comm.allreduce(p, op=MPI.SUM)
        return p

    # --- diagnostic: reversible exchange int U.k S_nl dk (Eq. 11) -----------
    def snl_work(self):
        """
        Domain-integrated reversible exchange between wave energy and mean
        Lagrangian KE by the nonlinear transfer working against the sheared
        Doppler velocity, W_nl = int U.k S_nl dk dA. In the model's (theta,f)
        energy-density convention (action source S_nl/sigma, |k| = sigma^2/g,
        the g-scaling of E cancels):
            W_nl = sum (U_f sin(th) + V_f cos(th)) * omega * S_nl
                   * dtheta * df * dA        [m^5/s^3 per unit rho].
        Sign: positive = into the wave field. This is THE term the cycle
        Eq. (11) does not close over; it vanishes for depth-independent
        u_L^h over the wave penetration depth and is otherwise higher order
        -- diagnosing it quantifies the non-closure. Resolved (prognostic)
        range only: no tail credit, consistent with the sigma*S_nl integral
        of the stress Eq. (24) (the tail books through kS_in instead). Uses
        S_nl and ocean_U/V of the LAST time_step, like wind_work uses
        gamma_in. Collective over the wave communicator (call on all PEs).
        """
        if not (hasattr(self, 'S_nl') and hasattr(self, 'ocean_U')):
            return 0.0
        h = self.halo
        sl = slice(h, -h)
        Uk = (self.ocean_U[None, :, :, :] * self.sinth[:, None, None, None]
            + self.ocean_V[None, :, :, :] * self.costh[:, None, None, None]) \
            * self.omega_t[None, :, None, None]
        p = float(np.sum((Uk * self.S_nl)[sl, sl, sl, sl]
                         * self.dd * self.dft[None, sl, None, None])
                  * self.dx * self.dy)
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            p = comm.allreduce(p, op=MPI.SUM)
        return p

    def _apply_bc_2d(self, field2d):
        """Halo-exchange a horizontal (Ny,Nx) field by lifting it to
        (1,1,Ny,Nx), running the model's 4D apply_bc, and squeezing back."""
        a = field2d[None, None, :, :]
        a = self.apply_bc(a)
        return a[0, 0, :, :]


def get_wave_parameter():
    p = {'nx': nx, 'ny': ny, 'nf': 36, 'nd': 24,
         'dx': dx, 'dy': dy, 'dt': wave_dt,
         'f0': 0.035, 'df0': 0.1,
         'periodic_in_x': True, 'periodic_in_y': False,   # acc1: zonal channel
         'mask_energy': True, 'limit_growth': True,
         'snapint': snapint}
    return p


# ---------------------------------------------------------------------------
# OCEAN model : acc1 on pe_simple, wind now delivered by the wave model
# ---------------------------------------------------------------------------
class ocean(tke_mixin, ocean_base):

    # paper Sec. 5: the TKE closure must mix the EULERIAN velocity
    # u_bar = u_L - u_S, not the model velocity u_L (Eq. 22 then matches the
    # shear production of Eq. 21, incl. the Stokes part P_L). Consumed by
    # tke.implicit_vfric.
    mix_eulerian = True

    # --- ocean snapshot: append the TKE closure diagnostics -----------------
    # The core snap_nc4 writes a fixed field set; append the TKE state and
    # the shear-production split of paper Sec. 5 (K_diss_E = P_E >= 0
    # Eulerian, K_diss_L = P_L Stokes/Langmuir, sign-indefinite; all on W
    # levels) without touching the core: create the variables after the
    # parent prepare_for_run, and after the parent diagnose append to the
    # SAME record with the same PE-sequential pattern. hasattr-guarded so
    # the n=0 snapshot (before the first vfric call) writes fill values.
    _extra_snap = ('tke', 'K_diss_E', 'K_diss_L')

    def prepare_for_run(self, snap_file_name='snap.cdf',
                        show_each_time_step=True):
        super().prepare_for_run(snap_file_name, show_each_time_step)
        if self.my_pe == 0:
            from netCDF4 import Dataset as NF
            fid = NF(self.snap_file_name, mode='r+')
            for v in self._extra_snap:
                if v not in fid.variables:
                    fid.createVariable(v, OM.prec, ('time', 'zu', 'yt', 'xt'))
            fid.close()
        self.mpi_barrier()
        return

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        if not (n == 0 or np.mod(n, int(self.snapint / self.dt)) == 0.):
            return
        from netCDF4 import Dataset as NF
        self.mpi_barrier()
        m = 0
        if self.my_pe == 0:
            fid = NF(self.snap_file_name, mode='r')
            m = len(fid.dimensions['time']) - 1        # record the parent wrote
            fid.close()
        m = self.pe0_bcast(m)
        for pe in range(self.n_pes):
            self.mpi_barrier()
            if self.my_pe == pe:
                fid = NF(self.snap_file_name, mode='r+')
                sl = (m, slice(None), slice(self.ys_pe - 1, self.ye_pe),
                      slice(self.xs_pe - 1, self.xe_pe))
                for v in self._extra_snap:
                    if hasattr(self, v):
                        fid.variables[v][sl] = self.unpad(getattr(self, v))
                fid.close()
        return

    # --- acc1 beta-plane Coriolis ------------------------------------------
    def set_coriolis(self, f0=None):
        f_glob = 2 * omega * np.sin(phi0) + 2 * omega * np.cos(phi0) / radius * yt_glob()
        fyx = np.zeros((self.Ny, self.Nx), np.float64)
        j0 = self.ys_pe - 1
        for jj in range(self.halo, self.Ny - self.halo):
            jg = min(max(j0 + (jj - self.halo), 0), ny - 1)
            fyx[jj, :] = f_glob[jg]
        self.coriolis_t = self.apply_bc(np.asarray(fyx, np.float64))
        return

    # --- acc1 ridge topography ---------------------------------------------
    def set_topography(self):
        kbot = np.full((self.Ny, self.Nx), float(self.halo), np.float64)
        j0, i0 = self.ys_pe - 1, self.xs_pe - 1
        xg, yg = xt_glob(), yt_glob()
        for jj in range(self.halo, self.Ny - self.halo):
            jg = j0 + (jj - self.halo)
            if jg < 0 or jg >= ny:
                continue
            for ii in range(self.halo, self.Nx - self.halo):
                ig = i0 + (ii - self.halo)
                if ig < 0 or ig >= nx:
                    continue
                if (yg[jg] > L_y / 2.0) and (xg[ig] > L_x * 0.75 or xg[ig] < L_x * 0.25):
                    kbot[jj, ii] = 0.0
        self.kbot = np.asarray(kbot, np.float64)
        return

    # --- initial fields (uniform buoyancy, as acc1) ------------------------
    def make_initial_fields(self):
        temp = np.zeros((nz, ny, nx), np.float64)
        salt = np.zeros((nz, ny, nx), np.float64)
        u = np.zeros((nz, ny, nx), np.float64)
        v = np.zeros((nz, ny, nx), np.float64)
        p_s = np.zeros((ny, nx), np.float64)
        return u, v, temp, salt, p_s

    # --- store the surface-restoring target per local row ------------------
    def localise_forcing(self):
        j0 = self.ys_pe - 1
        yloc = np.zeros(self.Ny, np.float64)
        yg = yt_glob()
        for jj in range(self.halo, self.Ny - self.halo):
            jg = min(max(j0 + (jj - self.halo), 0), ny - 1)
            yloc[jj] = yg[jg]
        self._bstar = acc1_bstar(yloc)
        return

    # --- ocean forcing: acc1 buoyancy restoring + wave stress + vortex force
    # --- diagnostic: discrete u_L-work of the vortex force ------------------
    def vortex_work(self):
        """
        u_L-work of the vortex force -v_L zeta_S, +u_L zeta_S, EXACTLY as
        applied in set_forcing (force fields kept there). With the corner
        pairing the TOTAL vanishes identically by summation-by-parts,
        including walls and land masks (verified to machine precision on a
        masked channel with topography): a pure round-off monitor, so the
        wim.tex budget can state that the vortex force does no work. Local
        contributions are nonzero and cancel globally -- the orthogonality
        is a global property. Model velocity IS u_L: no +u_S composition.
        Collective over the ocean communicator.
        """
        if not hasattr(self, '_Fvort_u'):
            return 0.0
        h = self.halo
        sl = slice(h, -h)
        w = float(OM.np.sum((self.u * self._Fvort_u
                           + self.v * self._Fvort_v)[sl, sl, sl]) * dx * dy * dz)
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            w = comm.allreduce(w, op=MPI.SUM)
        return w

    # --- diagnostic: work of the wave stress on the ocean --------------------
    def stress_work_diag(self):
        """
        Domain-integrated kinematic work of the wave momentum forcing
        [m^5/s^3 per unit rho]. The model velocity is u_L, so this is the
        u_L^h . dz(tau) work of Eq. (6) of the paper, exactly as applied in
        set_forcing: surface pathway  W = sum u_surf*taux dA  (+ v*tauy), or
        depth-distributed pathway  W = sum (u*Fx + v*Fy) dV with the same
        T->U/V averaging. Collective over the ocean communicator.
        """
        h = self.halo
        sl = slice(h, -h)
        ks = -h - 1
        if self.use_stress_work:
            Fu = 0.5 * (self.tau_work_x + self.rollx(self.tau_work_x, -1)) * self.maskU
            Fv = 0.5 * (self.tau_work_y + self.rolly(self.tau_work_y, -1)) * self.maskV
            w = float(OM.np.sum((self.u * Fu + self.v * Fv)[sl, sl, sl])
                      * dx * dy * dz)
        else:
            w = float(OM.np.sum((self.u[ks] * self.taux * self.maskU[ks]
                               + self.v[ks] * self.tauy * self.maskV[ks])[sl, sl])
                      * dx * dy)
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            w = comm.allreduce(w, op=MPI.SUM)
        return w

    # --- diagnostic: stress work against the EULERIAN velocity --------------
    def stress_work_eul_diag(self):
        """
        Same stress forcing dotted with the EULERIAN velocity u = u_L - u_S
        (u_S at U/V points with the standard T->U/V averaging, using the
        received u_S timeline). Per the paper this is NOT the stress channel
        of the consistent u_L cycle (Eq. 6 wants u_L.dz(tau), see
        stress_work_diag) -- kept for comparison with the old Eulerian
        formulation and to expose the difference int u_S.dz(tau) dz, which
        the Eulerian |u|^2/2 budget routes through the Coriolis-Stokes and
        vortex channels instead. Collective over the ocean communicator.
        """
        h = self.halo
        sl = slice(h, -h)
        ks = -h - 1
        uSU = 0.5 * (self.u_S + self.rollx(self.u_S, -1)) * self.maskU
        vSV = 0.5 * (self.v_S + self.rolly(self.v_S, -1)) * self.maskV
        uE, vE = self.u - uSU, self.v - vSV
        if self.use_stress_work:
            Fu = 0.5 * (self.tau_work_x + self.rollx(self.tau_work_x, -1)) * self.maskU
            Fv = 0.5 * (self.tau_work_y + self.rolly(self.tau_work_y, -1)) * self.maskV
            w = float(OM.np.sum((uE * Fu + vE * Fv)[sl, sl, sl])
                      * dx * dy * dz)
        else:
            w = float(OM.np.sum((uE[ks] * self.taux * self.maskU[ks]
                               + vE[ks] * self.tauy * self.maskV[ks])[sl, sl])
                      * dx * dy)
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            w = comm.allreduce(w, op=MPI.SUM)
        return w

    # --- diagnostic: exchange work C = int u_L d(u_S)/dt dV ------------------
    def exchange_work_C(self):
        """
        Discrete exchange term of the wave-averaged energy cycle, now EXACT:
        set_forcing_direct books the work of the applied increment at its
        application point, sum (u_L + Delta u_S/2) . Delta u_S / dt, which is
        the identity (|u_L + d|^2 - |u_L|^2)/(2 dt) per point -- the energy
        the forcing actually put into the model, to machine precision. Model
        velocity IS u_L: no +u_S composition. Collective over the ocean
        communicator.
        """
        c = float(getattr(self, '_C_step', 0.0))
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            c = comm.allreduce(c, op=MPI.SUM)
        return c

    # --- d(u_S)/dt forcing of the u_L formulation (direct, non-AB) ----------
    def set_forcing_direct(self):
        """
        Plain forward increment u_L += Delta u_S over this ocean step (the
        d(u_S)/dt forcing of Eq. 5). Called by the core BETWEEN the AB update
        and the pressure solve: it must bypass the AB weights (they would
        break the exact telescoping u = u_L - u_S) and precede the projection
        (the increment has divergence the surface pressure must answer).

        u_S is piecewise constant between couplings and the layout is
        CONCURRENT (wave and ocean step the same interval in parallel), so
        u_S(n+1) cannot reach the ocean before it steps [n, n+1]: the
        increment is the difference of the RECEIVED timeline, u_S - u_S_old
        (kept by the coupler loop) -- one coupling interval behind the wave
        clock, like every other coupling field, and exact with respect to
        the ocean's own u_S bookkeeping (u = u_L - u_S with the received
        u_S). Applies the FULL difference once per step because coupling =
        one ocean step here; adapt if the ocean ever sub-steps. First step /
        old restarts without u_S_old: no forcing, C = 0. Books the exact
        local work sum for exchange_work_C.
        """
        if not hasattr(self, 'u_S_old'):
            self._C_step = 0.0
            return
        dus = self.u_S - self.u_S_old
        dvs = self.v_S - self.v_S_old
        incu = 0.5 * (dus + self.rollx(dus, -1)) * self.maskU
        incv = 0.5 * (dvs + self.rolly(dvs, -1)) * self.maskV
        h = self.halo
        sl = slice(h, -h)
        self._C_step = float(OM.np.sum(
            ((self.u + 0.5 * incu) * incu
           + (self.v + 0.5 * incv) * incv)[sl, sl, sl])
            * dx * dy * dz / self.dt)
        self.u = self.apply_bc(self.u + incu)
        self.v = self.apply_bc(self.v + incv)
        return

    def set_forcing(self, tau):
        h = self.halo
        ks = -h - 1                       # surface (top) level index

        # --- wave-stress momentum transfer (from wim) ----------------------
        if self.use_stress_work:
            # depth-dependent body force, distributed over the column already.
            # average tracer-point work onto velocity points, like u_S/v_S.
            self.du = OM.modify_array(
                self.du, (tau, slice(None), slice(None), slice(None)),
                self.du[tau] + 0.5 * (self.tau_work_x + self.rollx(self.tau_work_x, -1)),
                out_sharding=self.sharding_4D)
            self.dv = OM.modify_array(
                self.dv, (tau, slice(None), slice(None), slice(None)),
                self.dv[tau] + 0.5 * (self.tau_work_y + self.rolly(self.tau_work_y, -1)),
                out_sharding=self.sharding_4D)
        else:
            # surface wave stress on the top cell only (kinematic: /dz, no /rho_0)
            accu = OM.np.zeros_like(self.du[tau])
            accv = OM.np.zeros_like(self.dv[tau])
            accu = OM.modify_array(accu, (ks, slice(None), slice(None)),
                                   self.taux / dz * self.maskU[ks])
            accv = OM.modify_array(accv, (ks, slice(None), slice(None)),
                                   self.tauy / dz * self.maskV[ks])
            self.du = OM.modify_array(self.du, (tau, slice(None), slice(None), slice(None)),
                                      self.du[tau] + accu, out_sharding=self.sharding_4D)
            self.dv = OM.modify_array(self.dv, (tau, slice(None), slice(None), slice(None)),
                                      self.dv[tau] + accv, out_sharding=self.sharding_4D)

        # --- bottom drag (vertical friction is owned by the TKE closure's
        # implicit_vfric in the core -- the explicit A_v block was removed
        # here when enable_tke went in, to avoid double-counting friction) ---
        dvisc_u = OM.np.zeros_like(self.u)
        dvisc_v = OM.np.zeros_like(self.v)
        if getattr(self, 'r_bot', 0.0) > 0:
            bot_u = self.maskU * (1.0 - self.rollz(self.maskU, 1))
            bot_v = self.maskV * (1.0 - self.rollz(self.maskV, 1))
            dvisc_u = dvisc_u - self.r_bot  * bot_u * self.u
            dvisc_v = dvisc_v - self.r_bot  * bot_v * self.v
        self.du = OM.modify_array(self.du, (tau, slice(None), slice(None), slice(None)),
                                  self.du[tau] + dvisc_u, out_sharding=self.sharding_4D)
        self.dv = OM.modify_array(self.dv, (tau, slice(None), slice(None), slice(None)),
                                  self.dv[tau] + dvisc_v, out_sharding=self.sharding_4D)

        # --- vortex force, u_L formulation (Eq. 5 of the paper) -------------
        # The model velocity is the LAGRANGIAN MEAN u_L; its flux-form
        # advection provides u_L.grad(u_L), so the remaining piece of the
        # rotational term -u_L x nabla x (u_L - u_S) is  -u_L x (nabla x u_S)
        # (grad|u_L|^2/2 is absorbed in the solved pressure). HYDROSTATIC
        # truncation: only the vertical-vorticity part is kept,
        #     F_x = -v_L zeta_S ,  F_y = +u_L zeta_S ,
        #     zeta_S = dx(v_S) - dy(u_S),
        # which is exactly work-free on u_L -- the vortex force does no work,
        # as in Eq. (6). The w_L dz(u_S) 'Stokes shear' terms that complete
        # the curl term are DELIBERATELY OMITTED here: without the vertical
        # force component (hydrostatic core) their horizontal parts would do
        # uncancelled net work w_L u_L.dz(u_S) on u_L. They belong in
        # nh_simple, where the vertical component restores exact
        # orthogonality. The former Coriolis-Stokes block is GONE: the core
        # Coriolis acts on the model velocity = u_L, which IS f zhat x u_L of
        # Eq. (5), and its discrete u_L-work vanishes exactly (the old
        # block's argument, now for free). The former Stokes buoyancy
        # advection is GONE: tracers are advected by the model velocity u_L
        # and its continuity w (= w_L).
        if hasattr(self, 'u_S'):
            uSU = 0.5 * (self.u_S + self.rollx(self.u_S, -1)) * self.maskU
            vSV = 0.5 * (self.v_S + self.rolly(self.v_S, -1)) * self.maskV
            zeta_S = (self.rollx(vSV, -1) - vSV) / dx \
                   - (self.rolly(uSU, -1) - uSU) / dy
            zeta_S = self.apply_bc(zeta_S)
            # ENERGY-CONSERVING corner pairing (Sadourny type): form the
            # products at the zeta corners FIRST,
            #   P = zeta_S * (x-average of v to the corner)
            #   Q = zeta_S * (y-average of u to the corner)
            # then average P in y to U and Q in x to V. By summation-by-parts
            # sum u*F_x + sum v*F_y telescopes to zero IDENTICALLY -- verified
            # to machine precision INCLUDING walls and land masks (the masked
            # velocities kill the wrap/edge terms), so the vortex force does
            # no total work on u_L, discretely; vortex_work is a pure
            # round-off monitor of this.
            P = zeta_S * 0.5 * (self.v + self.rollx(self.v, -1))
            Q = zeta_S * 0.5 * (self.u + self.rolly(self.u, -1))
            Fu = -0.5 * (P + self.rolly(P, 1)) * self.maskU
            Fv =  0.5 * (Q + self.rollx(Q, 1)) * self.maskV
            self._Fvort_u, self._Fvort_v = Fu, Fv      # kept for vortex_work()
            self.du = OM.modify_array(
                self.du, (tau, slice(None), slice(None), slice(None)),
                self.du[tau] + Fu, out_sharding=self.sharding_4D)
            self.dv = OM.modify_array(
                self.dv, (tau, slice(None), slice(None), slice(None)),
                self.dv[tau] + Fv, out_sharding=self.sharding_4D)

        # --- acc1 surface buoyancy restoring -------------------------------
        bstar2 = OM.np.asarray(self._bstar[:, None] * OM.np.ones((1, self.Nx)), np.float64)
        restore = OM.np.zeros_like(self.dtemp[tau])
        forc = (1.0 / t_rest) * (bstar2 - self.temp[ks, :, :])
        restore = OM.modify_array(restore, (ks, slice(None), slice(None)),
                                  forc * self.maskT[ks])
        self.dtemp = OM.modify_array(self.dtemp, (tau, slice(None), slice(None), slice(None)),
                                     self.dtemp[tau] + restore, out_sharding=self.sharding_4D)
        return


def get_ocean_parameter():
    return dict(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=ocean_dt,
        eq_of_state=0,
        periodic_in_x=True, periodic_in_y=False,
        implicit_free_surface=True, explicit_free_surface=False,
        superbee_advection=True,
        Ahbi=Ahbi,
        # (favoring the validated standalone acc1.py: no Khbi -- superbee
        # provides tracer mixing; no kappa_T/A_v backgrounds -- the TKE
        # closure's floors are the backgrounds; no kappa_conv -- convection
        # lives inside the closure)
        enable_implicit_vmix=True, enable_tke=True,
        r_bot=r_bot,
        epsilon=1e-6, max_iterations=5000, snapint=snapint,
    )


# ===========================================================================
# PE distribution and communicators (COMM_WORLD split by colour)
