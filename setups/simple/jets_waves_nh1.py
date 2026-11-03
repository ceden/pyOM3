"""
jets_waves_nh1: two balanced jets with a small perturbation get unstable,
with a random internal gravity wave field (scaled GM spectrum) added
ps3D: test.f90 (Nov 3 2026)

Non-hydrostatic (nh_simple) in the scaled form of ps3D (Ro = 0.3,
dsqr = 0.2**2, f0 = N0 = 1, Lz = 1), rigid lids, periodic in x and y.
Two opposite Gaussian jets with cos(pi z/Lz) vertical structure in thermal
wind balance, perturbed with a sin(4 pi x/Lx) sin(2 pi y/Ly) mode in u
of amplitude perturbation (0.0001 in test.f90).
The initial state is balanced to order init_order with
pyOM3/nh_simple/diag_balance.py. Then the waves are added: every wavenumber
and both wave modes s = +1, -1 get the energy of the scaled GM spectrum
(function wave_energy) with a random phase, minus the geostrophic part of
this wave field, as add_initial_waves of test.f90. b is the analytic
thermal wind of u (test.f90 differentiates u in z and copies b to the
bottom and top level). Instead of the filter of test.f90
(enable_diag_balance_filter) the wave modes are slaved only below the
vertical wavenumber set by slave_modes, see diag_balance.py.

During the run, at every snapshot the model state is decomposed into the
balanced orders 0 to diag_order, written to jets_waves_nh1_balance.cdf in
the format of jets_balance_nh1 (plot with plot_jets_balance1 and
name = 'jets_waves_nh1'). The model output is in jets_waves_nh1.cdf.

fac = 1 is the low resolution version (100 x 100 x 20), fac = 5 the
resolution of test.f90 (500 x 500 x 100). The balance runs on one process only.

run:    python jets_waves_nh1.py [seed]          (one process)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.snapshot import snap_nc4
from pyOM3.nh_simple.diag_balance import balance
from netCDF4 import Dataset as NF

fac        = 1                      # 1: low resolution, 5: as test.f90
nx, ny, nz = 100*fac, 100*fac, 20*fac
Lx, Ly, Lz = 4.0, 4.0, 1.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
Ro         = 0.3
f0         = 1.
N0         = 1.
dsqr       = 0.2**2
dt         = 0.01/fac
Ahbi       = 5e-5*Ro*dx**3
Avbi       = 5e-5*Ro*0.2*dz**3
init_order = 2                      # the initial state is balanced to this order
diag_order = 4                      # decomposition during the run up to this order
add_waves  = True                   # add the random wave field after the balancing
perturbation = 0.05                 # amplitude of the sin(4 pi x/Lx) mode, 0.0001 in test.f90
slave_modes = 'auto'                # wave modes slaved only below a vertical wavenumber, see diag_balance.py
seed       = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # random phases of the waves
runlen     = 1.0
snapint    = 1.0
name       = 'jets_waves_nh1'

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt, Lx=Lx, Ly=Ly, Lz=Lz,
           f0=f0, N0=N0, Ro=Ro, dsqr=dsqr, periodic_in_z=False,
           Ahbi=Ahbi, Khbi=Ahbi, Avbi=Avbi, Kvbi=Avbi,
           snapint=snapint)


def wave_energy(kx, ky, kz):
    """the scaled GM spectrum, as function wave_energy of test.f90"""
    E0, na, nb, jstar = 1e-4, 0.6366197723675814, 2./np.pi, 3.
    kh2 = kx**2 + ky**2
    ok = (kh2 > 0) & (kz**2 > 0)
    kh2, kz2 = np.where(ok, kh2, 1.), np.where(ok, kz**2, 1.)
    om2 = (N0**2*kh2 + f0**2*kz2)/(dsqr*kh2 + kz2)
    om = np.sqrt(om2)
    cstar = N0/(np.pi*jstar)
    mstar = np.sqrt(N0**2 - dsqr*om2)/cstar
    Atl = na/(1 + kz2/mstar**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        B = nb*abs(f0)/om/np.sqrt(om2 - f0**2)
        E = E0/(4*np.pi)*Atl/mstar*B*(N0**2 - dsqr*om2)**2/kz2/om/(N0**2 - dsqr*f0**2)
    return np.where(ok, np.nan_to_num(E), 0.)


class jets_waves_nh1(balance, snap_nc4):

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        if n == 0 or np.mod(n, int(round(self.snapint/self.dt))) == 0:
            self.diagnose_balance(n*self.dt)

    def add_initial_waves(self, seed):
        """
        the random wave field of add_initial_waves in test.f90: energy
        wave_energy*Lx*Ly*Lz in each wavenumber and wave mode s = +1, -1
        with random phase, minus the geostrophic part of the wave field
        """
        rng = np.random.default_rng(seed)
        E = wave_energy(self.KX, self.KY, self.KZ)*Lx*Ly*Lz
        F = [0., 0., 0., 0.]
        for s in (1, -1):
            phase = np.exp(2j*np.pi*rng.random(E.shape))
            a = np.sqrt(2*E*self.coeff_p(s))*phase*(2*np.pi)**1.5   # factor (2 pi)^1.5 as in test.f90
            F = [FF + a*q for FF, q in zip(F, self.vec_q(s))]
        uw = self.backward(*F)
        g = self.project(*uw, 0)
        self.uw = [a - c for a, c in zip(uw, g)]                   # the waves, kept for diagnostics
        self.u, self.v, self.w, self.b = [a + c for a, c in zip((self.u, self.v, self.w, self.b), self.uw)]
        self.w = self.apply_vertical_flux_bc(self.w)
        return

    def prepare_balance_file(self):
        fid = NF(name + '_balance.cdf', mode='w')
        fid.createDimension('xt', nx); fid.createDimension('yt', ny); fid.createDimension('zt', nz)
        fid.createDimension('xu', nx); fid.createDimension('yu', ny); fid.createDimension('zu', nz)
        fid.createDimension('order', diag_order + 1); fid.createDimension('time', None)
        xt, yt, zt = (np.arange(nx) + 0.5)*dx, (np.arange(ny) + 0.5)*dy, (np.arange(nz) + 0.5)*dz - Lz
        for d, x in (('xt', xt), ('yt', yt), ('zt', zt), ('xu', xt + dx/2), ('yu', yt + dy/2), ('zu', zt + dz/2)):
            fid.createVariable(d, 'f8', (d,))[:] = x
        fid.createVariable('order', 'i4', ('order',))[:] = np.arange(diag_order + 1)
        fid.createVariable('time', 'f8', ('time',))
        fid.createVariable('norm', 'f8', ('time', 'order'))
        fid.variables['norm'].long_name = 'relative norm of (u,v,w,b) - (u,v,w,b)_bal, balanced up to this order'
        fid.createVariable('norm_b', 'f8', ('time', 'order'))
        fid.variables['norm_b'].long_name = 'relative norm of b - b_bal, balanced up to this order'
        for v, dims in (('u', ('zt', 'yt', 'xu')), ('v', ('zt', 'yu', 'xt')),
                        ('w', ('zu', 'yt', 'xt')), ('b', ('zt', 'yt', 'xt'))):
            fid.createVariable(v, 'f8', ('time',) + dims)
            for m in range(diag_order + 1):
                fid.createVariable('%s_%d' % (v, m), 'f8', ('time',) + dims)
        fid.close()

    def diagnose_balance(self, t):
        h = self.halo
        X = self.decompose(diag_order)
        state = (self.u, self.v, self.w, self.b)
        norms, norms_b = [], []
        for m in range(diag_order + 1):
            bal = [sum(x[i] for x in X[:m + 1]) for i in range(4)]
            norms.append(self.norm(state, bal))
            norms_b.append(self.norm(state[3:], bal[3:]))
        print('balance: t = %5.2f, norm of the residual per order:' % t,
              np.array2string(np.array(norms), precision=3), ' b only:', np.array2string(np.array(norms_b), precision=3))
        fid = NF(name + '_balance.cdf', mode='r+')
        i = len(fid.dimensions['time'])
        fid.variables['time'][i] = t
        fid.variables['norm'][i, :] = norms
        fid.variables['norm_b'][i, :] = norms_b
        for c, v in enumerate('uvwb'):
            fid.variables[v][i] = np.asarray(state[c])[h:-h, h:-h, h:-h].transpose()
            for m in range(diag_order + 1):
                fid.variables['%s_%d' % (v, m)][i] = np.asarray(X[m][c])[h:-h, h:-h, h:-h].transpose()
        fid.close()


if __name__ == '__main__':
    M = jets_waves_nh1(par)
    M.setup(snap_file_name=name + '.cdf', show_each_time_step=False)
    M.init_balance(slave_modes=slave_modes)

    # local coordinates with halo points, arrays are (x,y,z), z = 0 at the bottom
    h = M.halo
    x = (np.arange(M.Nx) - h + 0.5)*dx
    y = (np.arange(M.Ny) - h + 0.5)*dy
    z = (np.arange(M.Nz) - h + 0.5)*dz
    X, Y, Z = np.meshgrid(x + dx/2, y, z, indexing='ij')                  # U points
    jet = np.exp(-(Y - 3*Ly/4)**2/(Lx*0.04)**2) - np.exp(-(Y - Ly/4)**2/(Lx*0.04)**2)
    U = 0.7*(jet + perturbation*np.sin(X/Lx*4*np.pi)*np.sin(Y/Ly*2*np.pi))
    u = U*np.cos(Z/Lz*np.pi)
    # thermal wind f0 u_z = -b_y: b = f0 pi/Lz sin(pi z/Lz) times U integrated in y
    b = f0*np.pi/Lz*np.sin(Z/Lz*np.pi)*np.cumsum(U, axis=1)*dy
    # the horizontal mean of the integral depends only on where the integration
    # starts and would change the background stratification, so it is removed
    b = b - b[h:-h, h:-h, :].mean(axis=(0, 1))
    M.u = M.apply_bc(OM.np.asarray(u))
    M.b = M.apply_bc(OM.np.asarray(b))
    M.balance(init_order)
    if add_waves: M.add_initial_waves(seed)
    M.prepare_balance_file()
    M.loop(int(round(runlen/dt)) + 1)
