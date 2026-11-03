"""
jets_balance_nh1: two balanced jets with a small perturbation get unstable,
the flow is re-balanced order by order after Warn et al. (1995)
ps3D: test_obta_unscaled1.f90 (there with optimal balance)

Non-hydrostatic (nh_simple) with rigid lids, the counterpart of
jets_balance1.py (pe_simple), same jets, same parameters, dimensional
(Ro = 1, dsqr = 1). The initial state is balanced to order init_order with
pyOM3/nh_simple/diag_balance.py. During the run, at every snapshot the model state
is decomposed into the balanced orders 0 to diag_order, written to
jets_balance_nh1_balance.cdf (u_n, v_n, w_n, b_n of each order n) together
with the norm of the residual u - u_bal for the state balanced to order
0, 1, ..., diag_order (norm(time, order)) and the same norm of b alone
(norm_b). The model output is in jets_balance_nh1.cdf. plot_jets_balance1
shows the result with name = 'jets_balance_nh1' (arrays are written as
(z, y, x) as for pe_simple).

run:    python jets_balance_nh1.py          (one process)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.snapshot import snap_nc4
from pyOM3.nh_simple.diag_balance import balance
from netCDF4 import Dataset as NF

fac        = 1
nx, ny, nz = 100*fac, 100*fac, 10*fac
Lx, Ly, Lz = 500e3, 500e3, 500.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt         = 400./fac
f0         = 1e-4
N0         = 50*f0                  # L_r = N0 Lz/f0 = 25 km, jet width 10 km, Ro = u0/(f0 10 km) = 0.2
u0         = 0.2
init_order = 2                      # the initial state is balanced to this order
diag_order = 4                      # decomposition during the run up to this order
runlen     = 20*86400.
snapint    = 86400.
name       = 'jets_balance_nh1'

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt, Lx=Lx, Ly=Ly, Lz=Lz,
           f0=f0, N0=N0, Ro=1., dsqr=1., periodic_in_z=False,
           Ahbi=dx**4/(30*86400.), Khbi=dx**4/(30*86400.),
           snapint=snapint)


class jets_balance_nh1(balance, snap_nc4):

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        if n == 0 or np.mod(n, int(self.snapint/self.dt)) == 0:
            self.diagnose_balance(n*self.dt)

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
        print('balance: t = %5.1f d, norm of the residual per order:' % (t/86400.),
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
    M = jets_balance_nh1(par)
    M.setup(snap_file_name=name + '.cdf', show_each_time_step=False)
    M.init_balance()

    # local coordinates with halo points, arrays are (x,y,z), z = 0 at the surface
    h = M.halo
    x = (np.arange(M.Nx) - h + 0.5)*dx
    y = (np.arange(M.Ny) - h + 0.5)*dy
    z = (np.arange(M.Nz) - h + 0.5)*dz - Lz
    X, Y, Z = np.meshgrid(x + dx/2, y, z, indexing='ij')                  # U points
    jet = np.exp(-(Y - 3*Ly/4)**2/(0.02*Lx)**2) - np.exp(-(Y - Ly/4)**2/(0.02*Lx)**2)
    u = u0*(jet + 0.05*np.sin(X/Lx*10*np.pi)*np.sin(Y/Ly*2*np.pi))*np.cos((Z + Lz)/Lz*np.pi)
    # thermal wind b_y = -f0 u_z, integrated in y (buoyancy perturbation, the background N0 is separate),
    # from u BEFORE apply_bc, which changes the vertical halo points in place with the numpy backend
    b = -f0*np.cumsum(np.gradient(u, dz, axis=2), axis=1)*dy
    # the horizontal mean of the integral depends only on where the integration
    # starts and would change the background stratification, so it is removed
    b = b - b[h:-h, h:-h, :].mean(axis=(0, 1))
    M.u = M.apply_bc(OM.np.asarray(u))
    M.b = M.apply_bc(OM.np.asarray(b))
    M.prepare_balance_file()
    M.balance(init_order)
    M.loop(int(runlen/dt) + 1)
