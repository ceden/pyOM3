"""
Coupled wave-ocean single-column experiment (wim_simple + nh_simple column),
script version of run.ipynb: runs BOTH stress formulations (surface stress
and depth-dependent stress force) and writes all recorded fields to a netCDF
file instead of plotting. The energy budgets of the notebook can be
reconstructed offline from the saved fields (U,V,Us,Vs,DUS,DVS,Fx,Fy).

Run:    python run_single_column.py [outfile.nc]
"""
import os, sys
home = os.path.expanduser('~')
sys.path.append(home + '/src/pyOM3/')
import numpy as np
import netCDF4
from pyOM3.wim_simple.model  import model as wave_model
from pyOM3.nh_simple.snapshot import model as ocean_model

outfile = sys.argv[1] if len(sys.argv) > 1 else 'single_column.nc'

dt  = 2 * 60.
Avu = 1e-5     # Eulerian-shear viscosity, single definition for run and budget
f0o = 1e-4     # ocean Coriolis parameter, single definition
runlen_days = 2.2

# wave setup
pw = {'nx': 1, 'ny': 1, 'nf': 62, 'nd': 80, 'dx': 1e3, 'dy': 1e3, 'dt': dt,
      'f0': 0.035, 'df0': 0.05, 'periodic_in_x': True, 'periodic_in_y': True,
      'mask_energy': True, 'limit_growth': True, 'tail_flux': 'input',
      'snonlin_backend': 'fortran', 'implicit_dissipation_only': True}

# ocean setup
nz, Lz = 50, 5.
dz, dx = Lz / nz, 1.0


class my_ocean_model(ocean_model):

    def set_forcing(self, tau):
        """wave stress: surface-only (tau/dz in top cell) or
        depth-dependent stress force from calc_stress_work"""
        if self.use_stress_work:
            self.du[tau, ...] += self.tau_work_x
            self.dv[tau, ...] += self.tau_work_y
        else:
            self.du[tau, :, :, -(self.halo + 1)] += self.taux / self.dz
            self.dv[tau, :, :, -(self.halo + 1)] += self.tauy / self.dz

    def set_forcing_direct(self):
        self.u += self.dt_us * self.dt
        self.v += self.dt_vs * self.dt
        # friction acting on the Eulerian velocity u = u_L - u_S
        # (plain forward increment, as the model's own friction; the
        #  boundary fluxes are zeroed inside harmonic_v_only, so the
        #  empty z-halos of u_S are irrelevant)
        self.u += self.dt * self.harmonic_v_only(self.u - self.u_S, self.Avu)
        self.v += self.dt * self.harmonic_v_only(self.v - self.v_S, self.Avu)


def run(use_stress_work):

    Mw = wave_model(pw)
    Mw.setup()
    Mw.E[1:-1, 1:-1, 1:-1, 1:-1] = 1e-2
    Mw.A = Mw.E / Mw.omega_t[None, :, None, None]
    Mw.ustar = np.ones((Mw.Ny, Mw.Nx), np.float64) * 0.35
    Mw.udir  = np.ones((Mw.Ny, Mw.Nx), np.float64) * (np.pi / 2)

    po = {'nx': 1, 'ny': 1, 'nz': nz, 'dx': dx, 'dy': dx, 'dz': dz, 'dt': dt,
          'f0': f0o, 'N0': 0e-4, 'periodic_in_z': False, 'Ro': 1., 'dsqr': 1.,
          'Avu': Avu, 'use_stress_work': use_stress_work}   # NOT 'Av': that would
          # additionally switch on the built-in friction acting on u_L
    Mo = my_ocean_model(po)
    Mo.setup()
    Mo.u_S   = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
    Mo.v_S   = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
    Mo.dt_us = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
    Mo.dt_vs = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
    if use_stress_work:
        Mo.tau_work_x = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
        Mo.tau_work_y = np.zeros((Mo.Nx, Mo.Ny, Mo.Nz), np.float64)
    else:
        Mo.taux = np.zeros((Mo.Nx, Mo.Ny), np.float64)
        Mo.tauy = np.zeros((Mo.Nx, Mo.Ny), np.float64)

    steps = int(86400 * runlen_days / Mw.dt)
    U   = np.zeros((steps, Mo.nz), np.float64)
    V   = np.zeros((steps, Mo.nz), np.float64)
    Us  = np.zeros((steps, Mo.nz), np.float64)
    Vs  = np.zeros((steps, Mo.nz), np.float64)
    DUS = np.zeros((steps, Mo.nz), np.float64)   # applied Stokes tendency
    DVS = np.zeros((steps, Mo.nz), np.float64)
    Fx  = np.zeros((steps, Mo.nz), np.float64)   # wave force per volume
    Fy  = np.zeros((steps, Mo.nz), np.float64)

    tA, tB, tC = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    print('running for ', steps, 'steps, use_stress_work =', use_stress_work)
    for n in range(steps):

        U[n, :] = Mo.u[1, 1, Mo.halo:-Mo.halo]
        V[n, :] = Mo.v[1, 1, Mo.halo:-Mo.halo]
        _u, _v = np.reshape(U[n, :], (Mo.nz, 1, 1)), np.reshape(V[n, :], (Mo.nz, 1, 1))
        Mw.ocean_U,  Mw.ocean_V   = Mw.calc_doppler_vel(   Mo.dz, Mo.nz, _u, _v)
        Mw.ocean_Udf, Mw.ocean_Vdf = Mw.calc_doppler_vel_df(Mo.dz, Mo.nz, _u, _v)

        Mw.time_step(tA, tB, tC, tau, taum1, taum2)

        # wave -> ocean products of the interval just stepped
        # (gamma_diss exists only after the first source-term step)
        dt_us, ust = Mw.calc_dt_us(Mo.dz, Mo.nz)
        Mo.u_S[1, 1, Mo.halo:-Mo.halo]   = ust[0, :, 0, 0]
        Mo.v_S[1, 1, Mo.halo:-Mo.halo]   = ust[1, :, 0, 0]
        Mo.dt_us[1, 1, Mo.halo:-Mo.halo] = dt_us[0, :, 0, 0]
        Mo.dt_vs[1, 1, Mo.halo:-Mo.halo] = dt_us[1, :, 0, 0]

        if use_stress_work:
            tw = Mw.calc_stress_work(Mo.dz, Mo.nz)
            Mo.tau_work_x[1, 1, Mo.halo:-Mo.halo] = tw[0, :, 0, 0]
            Mo.tau_work_y[1, 1, Mo.halo:-Mo.halo] = tw[1, :, 0, 0]
        else:
            ts = Mw.calc_stress()
            Mo.taux[1, 1] = ts[0, 0, 0]
            Mo.tauy[1, 1] = ts[1, 0, 0]

        Us[n, :], Vs[n, :]   = ust[0, :, 0, 0], ust[1, :, 0, 0]
        DUS[n, :], DVS[n, :] = dt_us[0, :, 0, 0], dt_us[1, :, 0, 0]
        if use_stress_work:
            Fx[n, :], Fy[n, :] = tw[0, :, 0, 0], tw[1, :, 0, 0]
        else:
            Fx[n, -1], Fy[n, -1] = ts[0, 0, 0] / Mo.dz, ts[1, 0, 0] / Mo.dz

        Mo.time_step(tA, tB, tC, tau, taum1, taum2)

        tau, taum1, taum2 = np.mod(tau + 1, 3), np.mod(taum1 + 1, 3), np.mod(taum2 + 1, 3)
        if tA == 1. and tB == 0. and tC == 0:      tA, tB, tC = 1.5, -0.5, 0.
        elif tA == 1.5 and tB == -0.5 and tC == 0: tA, tB, tC = 23./12., -16./12., 5./12.
        if n % 200 == 0: print('n=', n, '/', steps)
    print('done')
    return U, V, Us, Vs, DUS, DVS, Fx, Fy


# ---------------------------------------------------------------------------
# run both formulations and write everything to netCDF
# ---------------------------------------------------------------------------
R1 = run(False)   # surface stress  tau/dz in top cell
R2 = run(True)    # depth-dependent stress force
steps = R1[0].shape[0]

time = np.arange(steps) * dt
z = np.arange(nz) * dz
z = z - z[-1]                      # surface cell centre at z = 0 (as in the plots)

fid = netCDF4.Dataset(outfile, 'w', format='NETCDF4')
fid.title = ('coupled wave-ocean single column, wim_simple + nh_simple; '
             'exp 0: surface stress tau/dz in top cell, '
             'exp 1: depth-dependent stress force (calc_stress_work)')
fid.dt, fid.Avu, fid.f0, fid.dz = dt, Avu, f0o, dz
fid.ustar, fid.udir = 0.35, np.pi / 2

fid.createDimension('time', steps)
fid.createDimension('z', nz)
fid.createDimension('exp', 2)

v = fid.createVariable('time', 'f8', ('time',)); v.units = 's'; v[:] = time
v = fid.createVariable('z', 'f8', ('z',)); v.units = 'm'
v.long_name = 'cell centre depth, surface cell at 0'; v[:] = z
v = fid.createVariable('use_stress_work', 'i4', ('exp',))
v.long_name = '0: surface stress, 1: stress force'; v[:] = [0, 1]

names = [('u_L',   'Lagrangian velocity (x)',            'm/s'),
         ('v_L',   'Lagrangian velocity (y)',            'm/s'),
         ('u_S',   'Stokes velocity (x)',                'm/s'),
         ('v_S',   'Stokes velocity (y)',                'm/s'),
         ('dt_us', 'applied Stokes tendency (x)',        'm/s^2'),
         ('dt_vs', 'applied Stokes tendency (y)',        'm/s^2'),
         ('F_x',   'wave stress force per volume (x); '
                   'exp 0: tau_x/dz in the top cell only', 'm/s^2'),
         ('F_y',   'wave stress force per volume (y); '
                   'exp 0: tau_y/dz in the top cell only', 'm/s^2')]

for k, (name, long_name, units) in enumerate(names):
    v = fid.createVariable(name, 'f8', ('exp', 'time', 'z'))
    v.long_name, v.units = long_name, units
    v[0, ...] = R1[k]
    v[1, ...] = R2[k]

fid.close()
print('wrote', outfile)
