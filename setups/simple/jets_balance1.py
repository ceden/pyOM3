"""
jets_balance1: two balanced jets with a small perturbation get unstable,
the flow is re-balanced order by order after Warn et al. (1995)
ps3D: test_obta_unscaled1.f90 (there with optimal balance)

Hydrostatic (pe_simple), rigid lid or implicit free surface, buoyancy as
tracer, periodic in x and y, 2nd order advection. Two opposite Gaussian jets
with cos(pi z/H) vertical structure in thermal wind balance, perturbed with a
sin(10 pi x/Lx) sin(2 pi y/Ly) mode in u. The initial state is balanced to
order init_order with pyOM3/pe_simple/diag_balance.py. During the run, at
every snapshot the model state is decomposed into the balanced orders 0 to
diag_order, written to jets_balance1_balance.cdf (u_n, v_n, w_n, b_n of each
order n and p_s_n) together with the norm of the residual u - u_bal for the
state balanced to order 0, 1, ..., diag_order (variable norm(time, order)),
as in Eden et al. (2019, JPO), and the same norm of b alone (norm_b). The
model output is in jets_balance1.cdf.
plot_jets_balance1 shows the result.

run:    python jets_balance1.py          (one process)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_simple.snapshot import snap_nc4
from pyOM3.pe_simple.diag_balance import balance
from netCDF4 import Dataset as NF

fac        = 1
nx, ny, nz = 100*fac, 100*fac, 10*fac
Lx, Ly, Lz = 500e3, 500e3, 500.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt         = 400./fac
f0         = 1e-4
N0         = 50*f0                  # L_r = N0 Lz/f0 = 25 km, jet width 10 km, Ro = u0/(f0 10 km) = 0.2
u0         = 0.2
free_surface = False                # rigid lid or implicit free surface
init_order = 2                      # the initial state is balanced to this order
diag_order = 4                      # decomposition during the run up to this order
runlen     = 20*86400.
snapint    = 86400.
name       = 'jets_balance1'

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt,
           periodic_in_x=True, periodic_in_y=True,
           eq_of_state=0, implicit_free_surface=free_surface, explicit_free_surface=False,
           epsilon=1e-14, max_iterations=5000,
           Ahbi=dx**4/(30*86400.), Khbi=dx**4/(30*86400.), vmix_scheme='explicit',
           snapint=snapint)

xt, yt = (np.arange(nx) + 0.5)*dx, (np.arange(ny) + 0.5)*dy
xu, yu = (np.arange(nx) + 1.0)*dx, (np.arange(ny) + 1.0)*dy
zt = (np.arange(nz) + 0.5)*dz - Lz               # z = 0 at the surface


class jets_balance1(balance, snap_nc4):

    def set_coriolis(self):
        self.coriolis_t = OM.np.asarray(f0 + np.zeros((self.Ny, self.Nx)))

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(zt, yt, xu, indexing='ij')
        jet = np.exp(-(Y - 3*Ly/4)**2/(0.02*Lx)**2) - np.exp(-(Y - Ly/4)**2/(0.02*Lx)**2)
        u = u0*(jet + 0.05*np.sin(X/Lx*10*np.pi)*np.sin(Y/Ly*2*np.pi))*np.cos((Z + Lz)/Lz*np.pi)
        # thermal wind b_y = -f0 u_z, integrated in y, plus the background N0^2 z.
        # The horizontal mean of the integral depends only on where the integration
        # starts and would change the background stratification, so it is removed.
        uz = np.gradient(u, zt, axis=0)
        bt = -f0*np.cumsum(uz, axis=1)*dy
        b = N0**2*Z + bt - bt.mean(axis=(1, 2), keepdims=True)
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(u, zeros, b, zeros, np.zeros((ny, nx)))

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        if n == 0 or np.mod(n, int(self.snapint/self.dt)) == 0:
            self.diagnose_balance(n*self.dt)

    def prepare_balance_file(self):
        fid = NF(name + '_balance.cdf', mode='w')
        fid.createDimension('xt', nx); fid.createDimension('yt', ny); fid.createDimension('zt', nz)
        fid.createDimension('xu', nx); fid.createDimension('yu', ny); fid.createDimension('zu', nz)
        fid.createDimension('order', diag_order + 1); fid.createDimension('time', None)
        for d, x in (('xt', xt), ('yt', yt), ('zt', zt), ('xu', xu), ('yu', yu), ('zu', zt + dz/2)):
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
        for m in range(diag_order + 1):
            fid.createVariable('p_s_%d' % m, 'f8', ('time', 'yt', 'xt'))
        fid.close()

    def diagnose_balance(self, t):
        h = self.halo
        X = self.decompose(diag_order)
        state = (self.u, self.v, self.w, self.temp - self.bal_b_back)
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
        for v, a in zip(('u', 'v', 'w', 'b'), state):
            fid.variables[v][i] = np.asarray(a)[h:-h, h:-h, h:-h]
            for m in range(diag_order + 1):
                fid.variables['%s_%d' % (v, m)][i] = np.asarray(X[m][('u', 'v', 'w', 'b').index(v)])[h:-h, h:-h, h:-h]
        for m in range(diag_order + 1):
            fid.variables['p_s_%d' % m][i] = np.asarray(X[m][4])[h:-h, h:-h]
        fid.close()


if __name__ == '__main__':
    M = jets_balance1(par)
    M.set_coriolis()
    M.set_topography()
    M.prepare_for_run(snap_file_name=name + '.cdf', show_each_time_step=False)
    M.set_initial_conditions()
    M.init_balance(N0)
    M.prepare_balance_file()
    M.balance(init_order)
    M.loop(int(runlen/dt) + 1)
