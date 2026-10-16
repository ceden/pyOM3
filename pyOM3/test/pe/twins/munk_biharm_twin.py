"""
Spherical Munk gyre twin, biharmonic friction, variable dxt(i), dyt(j), dzt(k):
pe_realistic (time_step_implicit_pressure) vs pyOM2 via f2py.

Closed basin 10-40 E, 20-52 N (nx=ny=30, dx 0.8..1.2 deg, dy 0.8..1.4 deg),
nz=4 (dzt 325,275,225,175 m, bottom at 1000 m), linear EOS (type 1 in both),
zonal wind stress -tau0 cos(pi (y-y0)/Ly), A_hbi 1e12 cos^1.5, constant
kappaH_0 = kappaM_0 = 1e-4 with implicit vertical friction/diffusion,
AB2 eps 0.1 in both, implicit free surface in both. Run NDAYS days at two
time steps and report the relative differences of u, v, temp, p_s; the
expected residual is the O(dt) splitting difference (pe applies the
friction and the implicit vertical friction to the AB-updated velocity,
pyOM2 adds du_mix to the tendency and applies vertical friction to u(tau)),
which must halve when dt halves.

  PYOM2=... PYOM3=... python3 munk_biharm_twin.py [ndays]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get('PYOM3', '.'))
sys.path.insert(0, os.path.join(os.environ.get('PYOM2', '../pyOM2'), 'py_src'))
import pyOM3 as OM
from pyOM3.pe_realistic.model import model
from pyOM3.density import get_rho
from pyOM import pyOM

nx, ny, nz = 30, 30, 4
x0, y0 = 10.0, 20.0
dxt_gl = np.linspace(0.8, 1.2, nx)
dyt_gl = np.linspace(0.8, 1.4, ny)
dzt_gl = np.array([325., 275., 225., 175.])   # gently varying: the u-centred dzw recurrence must stay positive
A_hbi, cosp = 1e12, 1.5
kap, avisc = 1e-4, 1e-4
tau0 = 1e-4                                    # kinematic, m^2/s^2
omega = np.pi/43082.0
ndays = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
Ly = dyt_gl.sum()


def taux_of_y(y):
    return -tau0*np.cos(np.pi*(y - y0)/Ly)


def temp_init(z):                              # z negative downwards, 20 C at surface, 4 C at 1000 m
    return 4.0 + 16.0*(1.0 + z/1000.0)


class twin(pyOM):
    def set_parameter(self):
        M = self.fortran.main_module
        M.nx, M.ny, M.nz = nx, ny, nz
        M.dt_mom = self.dt; M.dt_tracer = self.dt
        M.coord_degree = 1; M.enable_cyclic_x = 0
        M.enable_biharmonic_friction = 1; M.a_hbi = A_hbi; M.biha_friction_cospower = cosp
        M.enable_implicit_vert_friction = 1
        M.kappah_0 = kap; M.kappam_0 = avisc
        M.kappah_0_convect_adjust = kap        # no hidden convective adjustment
        M.enable_conserve_energy = 0
        M.enable_streamfunction = 0; M.enable_free_surface = 1
        M.enable_superbee_advection = 1
        M.ab_eps = 0.1
        M.eq_of_state_type = 1
        M.congr_epsilon = 1e-12; M.congr_max_iterations = 5000
    def set_grid(self):
        M = self.fortran.main_module
        M.x_origin, M.y_origin = x0, y0
        M.dxt[M.onx:-M.onx] = dxt_gl; M.dyt[M.onx:-M.onx] = dyt_gl; M.dzt[:] = dzt_gl
    def set_coriolis(self):
        M = self.fortran.main_module
        M.coriolis_t[:, :] = 2*omega*np.sin(M.yt[None, :]/180.*np.pi)
    def set_topography(self):
        M = self.fortran.main_module
        M.kbot[M.onx:-M.onx, M.onx:-M.onx] = 1        # flat bottom, wet everywhere in the interior
    def set_initial_conditions(self):
        M = self.fortran.main_module
        for k in range(nz):
            M.temp[:, :, k, :] = temp_init(M.zt[k])
        M.temp[...] *= M.maskt[:, :, :, None]
        M.salt[...] = 35.0*M.maskt[:, :, :, None]
    def set_forcing(self):
        M = self.fortran.main_module
        M.surface_taux[:, :] = taux_of_y(M.yt[None, :])*M.masku[:, :, -1]


def run_pyom2(dt, nsteps):
    P = twin.__new__(twin); P.dt = dt; pyOM.__init__(P)
    M = P.fortran.main_module
    for n in range(nsteps):
        P.time_step(); P.time_goes_by()
    onx, tau = M.onx, M.tau - 1
    T = lambda a: np.transpose(a, (2, 1, 0))[:, onx:-onx, onx:-onx]
    return dict(u=T(M.u[:, :, :, tau]), v=T(M.v[:, :, :, tau]), temp=T(M.temp[:, :, :, tau]),
                p_s=M.psi[onx:-onx, onx:-onx, tau].T, yt=M.yt[onx:-onx].copy())


class pe(model):
    def set_forcing(self, tau):
        h = self.halo; ks = -h - 1
        taux = self._taux*self.maskU[ks]
        self.du = OM.add_to_array(self.du, (tau, ks, slice(None), slice(None)), taux/self._dz_surf)


def run_pe(dt, nsteps):
    par = dict(nx=nx, ny=ny, nz=nz, dxt=dxt_gl, dyt=dyt_gl, dzt=dzt_gl, coord_degree=True,
               x_origin=x0, y_origin=y0, dt=dt, periodic_in_x=False, periodic_in_y=False,
               eq_of_state=1, explicit_free_surface=False, implicit_free_surface=True,
               superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
               kappa_T=kap, A_v=avisc, Ahbi=A_hbi, biha_friction_cosPower=cosp,
               epsilon=1e-12, max_iterations=5000)
    m = pe(par); h = m.halo
    m.coriolis_t = 2*omega*np.sin(np.asarray(m.yt)[:, None]/180.*np.pi)*np.ones((m.Ny, m.Nx))
    m.kbot = np.ones((m.Ny, m.Nx)); m.kbot[[0, -1], :] = 0; m.kbot[:, [0, -1]] = 0
    m.prepare_for_run()
    zt = np.asarray(m.zt) - dzt_gl.sum()          # pe zt is height above the bottom
    m.temp = m.apply_bc(temp_init(zt)[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx)))*m.maskT
    m.salt = 35.0*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    m._dz_surf = float(np.asarray(m.dzt)[-h-1])
    m._taux = taux_of_y(np.asarray(m.yt)[:, None])*np.ones((m.Ny, m.Nx))
    A, B, C = 1.6, -0.6, 0.0
    tau, taum1, taum2 = 2, 1, 0
    for n in range(nsteps):
        m.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
    sl = slice(h, -h)
    return dict(u=np.asarray(m.u)[sl, sl, sl], v=np.asarray(m.v)[sl, sl, sl],
                temp=np.asarray(m.temp)[sl, sl, sl], p_s=np.asarray(m.p_s)[taum1, sl, sl],
                yt=np.asarray(m.yt)[sl])


def compare(dt):
    nsteps = int(round(ndays*86400/dt))
    a = run_pyom2(dt, nsteps); b = run_pe(dt, nsteps)
    out = {}
    for k in ('u', 'v', 'temp', 'p_s'):
        ref = a[k] - (a[k].mean() if k == 'p_s' else 0.0)
        dif = (b[k] - a[k]) - ((b[k] - a[k]).mean() if k == 'p_s' else 0.0)
        out[k] = np.abs(dif).max()/max(np.abs(ref).max(), 1e-300)
    print('dt = %5.0f s, %d steps: rel diff u %.2e  v %.2e  temp %.2e  p_s %.2e   (max|u| pyOM2 %.3f pe %.3f m/s)'
          % (dt, nsteps, out['u'], out['v'], out['temp'], out['p_s'], np.abs(a['u']).max(), np.abs(b['u']).max()))
    return out


if __name__ == '__main__':
    # one pyOM2 instance per process (its module arrays cannot be re-allocated)
    if len(sys.argv) > 2:
        compare(float(sys.argv[2])); sys.exit(0)
    import subprocess
    for dt in (1200.0, 600.0):
        subprocess.run([sys.executable, __file__, sys.argv[1] if len(sys.argv) > 1 else '2', str(dt)], check=True)
    print('the u/v/p_s residual must halve from dt 1200 to dt 600 (first-order splitting difference)')
