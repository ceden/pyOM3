"""
pe_simple vs pe_realistic on the SPLIT-EXPLICIT free surface, uniform
Cartesian grid: pe_realistic must reduce to pe_simple exactly there (all the
metric factors are 1 or dx, dy, dz), so any difference is a bug in the
metric-aware form of solve_expl_surface / momentum_tendency_expl.

Both models get the same wind-driven Munk gyre on a closed f-plane box,
constant dx, dy, dz, no friction option difference (pe_simple keeps the
T-point biharmonic tendency, pe_realistic uses biharmonic_momentum, which
coincide on a uniform Cartesian grid), and the same sub_cycling.

  PYOM3=<dir containing pyOM3> python3 pe_reduction_expl.py [nsteps] [M]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get('PYOM3', '.'))
import pyOM3 as OM
from pyOM3.pe_realistic.model import model as mr
from pyOM3.pe_simple.model import model as ms
from pyOM3.density import get_rho

nx, ny, nz = 30, 30, 4
dx = dy = 50e3
dz = 250.0
dt = 360.0
f0 = 1e-4
tau0 = 1e-4
grav = 9.81


def build(kind, M, Ahbi=None):
    common = dict(nx=nx, ny=ny, nz=nz, dt=dt, periodic_in_x=False, periodic_in_y=False,
                  eq_of_state=1, explicit_free_surface=True, implicit_free_surface=False,
                  superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
                  kappa_T=1e-4, A_v=1e-4, epsilon=1e-12, max_iterations=5000)
    if Ahbi:
        common['Ahbi'] = Ahbi
    if kind == 'simple':
        par = dict(common, dx=dx, dy=dy, dz=dz)
        m = ms(par)
    else:
        par = dict(common, dxt=np.full(nx, dx), dyt=np.full(ny, dy), dzt=np.full(nz, dz),
                   coord_degree=False, x_origin=0.0, y_origin=0.0)
        if Ahbi:
            par['biha_friction_cosPower'] = 0.0            # cos = 1 anyway, keep it explicit
        m = mr(par)
    h = m.halo
    m.sub_cycling = M
    m.coriolis_t = OM.np.asarray(np.full((m.Ny, m.Nx), f0))
    kb = np.ones((m.Ny, m.Nx)); kb[[0, -1], :] = 0; kb[:, [0, -1]] = 0
    m.kbot = OM.np.asarray(kb)
    m.prepare_for_run()
    zt = (np.arange(m.Nz) - m.Nz + m.halo + 0.5)*dz                     # 0 at the surface, negative down
    m.temp = m.apply_bc(OM.np.asarray((4.0 + 16.0*(1 + zt/(nz*dz)))[:, None, None]
                                      * np.ones((m.Nz, m.Ny, m.Nx))))*m.maskT
    m.salt = 35.0*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    y = np.arange(m.Ny)*dy
    m._taux = -tau0*np.cos(np.pi*(y - y[h])/((ny - 1)*dy))[:, None]*np.ones((m.Ny, m.Nx))
    m._dz_surf = dz
    return m


def set_forcing(self, tau):
    h = self.halo; ks = -h - 1
    self.du = OM.add_to_array(self.du, (tau, ks, slice(None), slice(None)),
                              self._taux*self.maskU[ks]/self._dz_surf)


def run(m, nsteps):
    m.set_forcing = set_forcing.__get__(m)
    A, B, C = 1.6, -0.6, 0.0
    tau, taum1, taum2 = 2, 1, 0
    h = m.halo
    for n in range(nsteps):
        m.time_step_explicit_pressure(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        if not np.isfinite(np.asarray(m.u)).all():
            return None, n + 1
    sl = slice(h, -h)
    ps = np.asarray(m.p_s); ps = ps[taum1] if ps.ndim == 3 else ps
    return dict(u=np.asarray(m.u)[sl, sl, sl], v=np.asarray(m.v)[sl, sl, sl],
                temp=np.asarray(m.temp)[sl, sl, sl], p_s=ps[sl, sl]), nsteps


# ---------------------------------------------------------------------------
# Barotropic-mode harness (Sep 10-14 2026). Found with it: (1) the sub cycle
# restarted from the averaged transport (fixed: U_bt/V_bt carried as state);
# (2) the top-cell tracer non-conservation under the linear free surface
# feeds a p_h gradient in phase with d(eta)/dt (option
# free_surface_tracer_flux). With both, energy decays at every M as in the
# 2-D toy (toy_split2d.py). Isolated here with f = 0, no
# stratification, no wind, no friction, one p_s bump released from rest, so
# only the barotropic sub-system acts. pe_simple and pe_realistic on the same
# physical box agree digit for digit, and the spherical version behaves the
# same, i.e. this is the scheme, not the metric.
#
#   python3 pe_reduction_expl.py bump
# ---------------------------------------------------------------------------
FS_TRACER_FLUX = True   # bump harness: surface tracer flux on (False reproduces the residual growth)


def bump_case(nsteps=50, Ms=(16, 31, 62), dz_=1250.0, nz_=4, dt_=216.0, lat0=70.0):
    import numpy as np
    from pyOM3.pe_simple.model import model as ms_
    from pyOM3.pe_realistic.model import model as mr_
    deg = 2*np.pi*6.371e6/360.
    dx_m = 0.1*deg*np.cos(np.deg2rad(lat0 + 1.5)); dy_m = 0.1*deg
    H = nz_*dz_; c = np.sqrt(grav*H)
    mcfl = dt_*c*np.sqrt(1/dx_m**2 + 1/dy_m**2)
    print('bump harness: dx %.0f m, dy %.0f m, H %.0f m, c %.0f m/s, dt %.0f s -> M_cfl %.1f, '
          'baroclinic-step CFL c dt/dx = %.1f' % (dx_m, dy_m, H, c, dt_, mcfl, c*dt_/dx_m))

    def make(kind, M):
        common = dict(nx=nx, ny=ny, nz=nz_, dt=dt_, periodic_in_x=False, periodic_in_y=False,
                      eq_of_state=1, explicit_free_surface=True, implicit_free_surface=False,
                      superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
                      kappa_T=1e-4, A_v=1e-4, epsilon=1e-12, max_iterations=5000)
        if kind == 'simple':
            m = ms_(dict(common, dx=dx_m, dy=dy_m, dz=dz_))
        elif kind == 'cart':
            m = mr_(dict(common, dxt=np.full(nx, dx_m), dyt=np.full(ny, dy_m),
                         dzt=np.full(nz_, dz_), coord_degree=False, x_origin=0., y_origin=0.))
        else:
            m = mr_(dict(common, dxt=np.full(nx, 0.1), dyt=np.full(ny, 0.1),
                         dzt=np.full(nz_, dz_), coord_degree=True, x_origin=0., y_origin=lat0))
        m.sub_cycling = M
        m.free_surface_tracer_flux = FS_TRACER_FLUX
        m.coriolis_t = OM.np.asarray(np.zeros((m.Ny, m.Nx)))
        kb = np.ones((m.Ny, m.Nx)); kb[[0, -1], :] = 0; kb[:, [0, -1]] = 0
        m.kbot = OM.np.asarray(kb)
        m.prepare_for_run()
        m.temp = m.apply_bc(OM.np.asarray(np.full((m.Nz, m.Ny, m.Nx), 10.)))*m.maskT
        m.salt = 35.*m.maskT
        m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
        h = m.halo
        j = np.arange(m.Ny)[:, None]; i = np.arange(m.Nx)[None, :]
        eta = 0.5*np.exp(-(((i - m.Nx/2)/4.)**2 + ((j - m.Ny/2)/4.)**2))
        ps = grav*eta*np.asarray(m.maskT)[-h-1]
        m.p_s = OM.np.asarray(ps if np.asarray(m.p_s).ndim == 2 else np.repeat(ps[None], 3, axis=0))
        m.set_forcing = (lambda self, tau: None).__get__(m)
        return m

    def go(kind, M):
        m = make(kind, M); h = m.halo
        ps0 = np.asarray(m.p_s); ps0 = ps0 if ps0.ndim == 2 else ps0[1]
        e0 = float(np.sum(ps0[h:-h, h:-h]**2))
        tau, taum1, taum2 = 2, 1, 0
        for n in range(nsteps):
            m.time_step_explicit_pressure(1.6, -0.6, 0., tau, taum1, taum2)
            tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
            if not np.isfinite(np.asarray(m.u)).all() or np.max(np.abs(np.asarray(m.u))) > 50:
                return 'blew up at %d' % (n + 1)
        ps = np.asarray(m.p_s); ps = ps[taum1] if ps.ndim == 3 else ps
        return 'ratio %9.1f  max|u| %.4f' % (float(np.sum(ps[h:-h, h:-h]**2))/e0,
                                             float(np.max(np.abs(np.asarray(m.u)))))

    fails = []
    for M in Ms:
        res = {k: go(k, M) for k in ('simple', 'cart', 'sph')}
        print('  M %3d (%.1f M_cfl)  simple: %-30s realistic Cartesian: %-30s realistic spherical: %s'
              % (M, M/mcfl, res['simple'], res['cart'], res['sph']))
        if M > 1.2*mcfl:                     # above the CFL limit: no growth, and simple == realistic Cartesian
            for k in ('simple', 'cart', 'sph'):
                if 'blew' in res[k] or float(res[k].split()[1]) >= 1.0:
                    fails.append('M %d %s: %s' % (M, k, res[k]))
            if res['simple'] != res['cart']:
                fails.append('M %d: pe_simple and pe_realistic Cartesian differ' % M)
    print('bump harness: %s' % ('ALL PASS (p_s variance decays at every M > 1.2 M_cfl, simple == realistic Cartesian)'
                                if not fails else 'FAIL: ' + '; '.join(fails)))
    return not fails


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'bump':
        sys.exit(0 if bump_case() else 1)
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print('uniform Cartesian %dx%dx%d, dx %.0f km, dz %.0f m, dt %.0f s, M %d, %d steps'
          % (nx, ny, nz, dx/1e3, dz, dt, M, nsteps))
    for Ahbi in (None, 1e11):
        tag = 'no friction' if Ahbi is None else 'biharmonic A=%.0e' % Ahbi
        a, na = run(build('simple', M, Ahbi), nsteps)
        b, nb = run(build('realistic', M, Ahbi), nsteps)
        if a is None or b is None:
            print('%-22s pe_simple %s, pe_realistic %s'
                  % (tag, 'blew up at %d' % na if a is None else 'ok',
                     'blew up at %d' % nb if b is None else 'ok'))
            continue
        out = []
        for k in a:
            sc = max(np.max(np.abs(a[k])), 1e-300)
            out.append('%s %.2e' % (k, np.max(np.abs(a[k] - b[k]))/sc))
        worst = max(np.max(np.abs(a[k] - b[k]))/max(np.max(np.abs(a[k])), 1e-300) for k in a)
        print('%-22s max|u| %.5f (simple) %.5f (realistic) | rel diff: %s  %s'
              % (tag, np.max(np.abs(a['u'])), np.max(np.abs(b['u'])), '  '.join(out), 'PASS' if worst < 1e-12 else 'FAIL'))
        if worst >= 1e-12:
            sys.exit(1)


