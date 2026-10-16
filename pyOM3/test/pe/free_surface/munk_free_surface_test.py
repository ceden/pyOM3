"""
Implicit vs split-explicit free surface in pe_realistic, spherical Munk gyre.

History: the split-explicit path (free_surface.py, Demange et al 2019) blew up
in the POP 1/10 deg run within ~11 steps while the implicit path ran. Two
causes were found with these harnesses (Sep 2026): the sub cycle restarted
from the sub-cycle-averaged transport (now U_bt/V_bt are carried as state,
as in FESOM2), and the top-cell tracer non-conservation under the linear
free surface (option free_surface_tracer_flux, on here). With both, the
explicit path agrees with the implicit one to <1 % in u at every M.

Configuration: closed basin 10-40 E, 20-52 N, nx = ny = 30, variable dxt
(0.8..1.2 deg) and dyt (0.8..1.4 deg), nz = 4 (325/275/225/175 m), linear
EOS, zonal wind stress, biharmonic friction A_hbi cos^1.5, implicit vertical
mixing, AB2 with eps 0.1. Deep (H = 1000 m, c = 99 m/s) and shallow (H = 100 m
via dzt/10, c = 31 m/s) variants, because the POP failures sit in shallow
tropical cells.

Cases (each prints max|u|, max|eta|, and the difference to the implicit run):

  1 sub-cycling sweep: M from 1/4 to 4 times the forward-backward CFL limit
    M_cfl = dt c sqrt(1/dx_min^2 + 1/dy_min^2). If the explicit path is only
    CFL-limited, it is stable for M > M_cfl and unstable below, and the
    solution converges to the implicit one as M grows.
  2 long run at a safe M: 5 days, the two paths side by side; the difference
    is the barotropic gravity wave the implicit path damps.
  3 flat vs stepped topography at a safe M: the same run with a shelf (one
    wet level in a strip) to test whether the failure needs a topographic
    step, as the POP blow-up points suggest.
  4 stress test: no friction, no wind, a single p_s bump released from rest
    (pure barotropic adjustment). Energy after 200 steps must not grow.

  PYOM3=<dir containing pyOM3> python3 munk_free_surface_test.py [ndays]

Expectations (Sep 15 2026 numbers, checked and reported as PASS/FAIL, nonzero
exit on any FAIL): above 1.2 M_cfl the explicit path agrees with the implicit
one to U_TOL in u at every M and the agreement does not degrade with M (no
growth of the difference by more than GROWTH_TOL from the smallest to the
largest stable M); below M_cfl blow-ups are the CFL limit and are not failures.
eta differs by the gravity-wave content the implicit path damps (5-13 %) and
is reported, not tested.
"""
U_TOL = 0.03          # relative max|u - u_implicit| / max|u_implicit|
GROWTH_TOL = 1.5      # ratio of the u difference at the largest to the smallest stable M
FAILS = []


def check(name, value, tol):
    ok = value <= tol
    print('  CHECK %-60s %.2e (tol %.1e) %s' % (name, value, tol, 'PASS' if ok else 'FAIL'))
    if not ok:
        FAILS.append(name)

import os, sys
import numpy as np
sys.path.insert(0, os.environ.get('PYOM3', '.'))
import pyOM3 as OM
from pyOM3.pe_realistic.model import model
from pyOM3.density import get_rho

FS_TRACER_FLUX = True        # surface tracer flux w_surf*var (see thermodynamics.py); False = pyOM2 form
nx, ny, nz = 30, 30, 4
x0, y0 = 10.0, 20.0
dxt_gl = np.linspace(0.8, 1.2, nx)
dyt_gl = np.linspace(0.8, 1.4, ny)
dzt_deep = np.array([325., 275., 225., 175.])
A_hbi, cosp = 1e12, 1.5
kap, avisc = 1e-4, 1e-4
tau0 = 1e-4
omega = np.pi/43082.0
grav = 9.81
dt = 600.0
deg = 2*np.pi*6.371e6/360.


def taux_of_y(y, Ly):
    return -tau0*np.cos(np.pi*(y - y0)/Ly)


def temp_init(z, H):
    return 4.0 + 16.0*(1.0 + z/H)


class pe(model):
    def set_forcing(self, tau):
        if getattr(self, '_no_wind', False):
            return
        h = self.halo; ks = -h - 1
        self.du = OM.add_to_array(self.du, (tau, ks, slice(None), slice(None)),
                                  self._taux*self.maskU[ks]/self._dz_surf)


def build(explicit, M=None, dzt=None, shelf=False, friction=True, wind=True, bump=False):
    dzt = dzt_deep if dzt is None else dzt
    par = dict(nx=nx, ny=ny, nz=nz, dxt=dxt_gl, dyt=dyt_gl, dzt=dzt, coord_degree=True,
               x_origin=x0, y_origin=y0, dt=dt, periodic_in_x=False, periodic_in_y=False,
               eq_of_state=1, explicit_free_surface=explicit, implicit_free_surface=not explicit,
               superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
               kappa_T=kap, A_v=avisc, epsilon=1e-12, max_iterations=5000)
    if friction:
        par.update(Ahbi=A_hbi, biha_friction_cosPower=cosp)
    m = pe(par); h = m.halo
    m.free_surface_tracer_flux = FS_TRACER_FLUX
    if M:
        m.sub_cycling = M
    m._no_wind = not wind
    m.coriolis_t = 2*omega*np.sin(np.asarray(m.yt)[:, None]/180.*np.pi)*np.ones((m.Ny, m.Nx))
    kb = np.ones((m.Ny, m.Nx)); kb[[0, -1], :] = 0; kb[:, [0, -1]] = 0
    if shelf:
        kb[h+5:h+12, h+5:h+10] = nz                     # one wet level: a shelf
    m.kbot = kb
    m.prepare_for_run()
    H = dzt.sum()
    zt = np.asarray(m.zt) - H
    m.temp = m.apply_bc(temp_init(zt, H)[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx)))*m.maskT
    m.salt = 35.0*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    m._dz_surf = float(np.asarray(m.dzt)[-h-1])
    m._taux = taux_of_y(np.asarray(m.yt)[:, None], dyt_gl.sum())*np.ones((m.Ny, m.Nx))
    if bump:                                            # a p_s hill released from rest
        xx = np.asarray(m.xt)[None, :]; yy = np.asarray(m.yt)[:, None]
        xc, yc = 0.5*(xx.min() + xx.max()), 0.5*(yy.min() + yy.max())   # centre of THIS domain
        wx, wy = 0.15*(xx.max() - xx.min()), 0.15*(yy.max() - yy.min())
        eta = 0.5*np.exp(-(((xx - xc)/wx)**2 + ((yy - yc)/wy)**2))
        ps = grav*eta*np.asarray(m.maskT)[-h-1]
        m.p_s = OM.np.asarray(ps if np.asarray(m.p_s).ndim == 2
                              else np.repeat(ps[None], 3, axis=0))
    return m


def run(m, explicit, nsteps):
    h = m.halo
    A, B, C = 1.6, -0.6, 0.0
    tau, taum1, taum2 = 2, 1, 0
    for n in range(nsteps):
        if explicit:
            m.time_step_explicit_pressure(A, B, C, tau, taum1, taum2)
        else:
            m.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        umax = float(np.max(np.abs(np.asarray(m.u))))
        if not np.isfinite(umax) or umax > 50.0:
            return None, n + 1, umax
    sl = slice(h, -h)
    ps = np.asarray(m.p_s); ps = ps[taum1] if ps.ndim == 3 else ps
    state = dict(u=np.asarray(m.u)[sl, sl, sl], v=np.asarray(m.v)[sl, sl, sl],
                 eta=ps[sl, sl]/grav, temp=np.asarray(m.temp)[sl, sl, sl])
    return state, nsteps, float(np.max(np.abs(state['u'])))


def M_cfl(dzt):
    """forward-backward limit of the barotropic sub-step"""
    c = np.sqrt(grav*dzt.sum())
    dx = np.min(dxt_gl)*deg*np.cos(np.deg2rad(y0 + dyt_gl.sum()))
    dy = np.min(dyt_gl)*deg
    return dt*c*np.sqrt(1./dx**2 + 1./dy**2), c


def compare(a, b):
    return {k: np.max(np.abs(a[k] - b[k]))/max(np.max(np.abs(b[k])), 1e-300) for k in a}


def case_sweep(ndays, dzt, label):
    mcfl, c = M_cfl(dzt)
    nsteps = int(ndays*86400/dt)
    print('\n=== %s: H %.0f m, c %.0f m/s, CFL limit M_cfl = %.1f, %d steps (%.1f d)'
          % (label, dzt.sum(), c, mcfl, nsteps, ndays))
    ref, _, uref = run(build(False, dzt=dzt), False, nsteps)
    print('implicit: max|u| %.4f m/s, max|eta| %.3f m' % (uref, np.max(np.abs(ref['eta']))))
    du = []
    for fac in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 4.0):
        M = max(1, int(round(fac*mcfl)))
        st, nst, umax = run(build(True, M=M, dzt=dzt), True, nsteps)
        if st is None:
            print('  M = %4d (%.2f M_cfl): BLEW UP at step %d, |u| %.3g' % (M, fac, nst, umax))
            if fac >= 1.2:
                check('%s: explicit stable at %.1f M_cfl' % (label, fac), 1.0, 0.0)
        else:
            d = compare(st, ref)
            print('  M = %4d (%.2f M_cfl): max|u| %.4f, max|eta| %.3f, vs implicit: u %.2e eta %.2e temp %.2e'
                  % (M, fac, umax, np.max(np.abs(st['eta'])), d['u'], d['eta'], d['temp']))
            if fac >= 1.2:
                du.append(d['u'])
    if du:
        check('%s: u vs implicit at M >= 1.2 M_cfl' % label, max(du), U_TOL)
        check('%s: no growth of the u difference with M' % label, du[-1]/max(du[0], 1e-300), GROWTH_TOL)


def case_topography(ndays, dzt):
    mcfl, _ = M_cfl(dzt)
    M = max(1, int(round(2*mcfl)))
    nsteps = int(ndays*86400/dt)
    print('\n=== topography at M = %d (2 M_cfl), %d steps' % (M, nsteps))
    for shelf in (False, True):
        ref, _, _ = run(build(False, dzt=dzt, shelf=shelf), False, nsteps)
        st, nst, umax = run(build(True, M=M, dzt=dzt, shelf=shelf), True, nsteps)
        tag = 'with shelf   ' if shelf else 'flat bottom  '
        if st is None:
            print('  %s explicit BLEW UP at step %d, |u| %.3g' % (tag, nst, umax))
            check('topography %s: explicit stable' % tag.strip(), 1.0, 0.0)
        else:
            d = compare(st, ref)
            print('  %s explicit max|u| %.4f, vs implicit: u %.2e eta %.2e' % (tag, umax, d['u'], d['eta']))
            check('topography %s: u vs implicit' % tag.strip(), d['u'], 0.10 if shelf else U_TOL)   # shelf waves


def case_bump(dzt):
    mcfl, _ = M_cfl(dzt)
    M = max(1, int(round(2*mcfl)))
    print('\n=== barotropic adjustment from a 0.5 m p_s bump, no wind, no friction, M = %d' % M)
    for explicit, name in ((False, 'implicit'), (True, 'explicit')):
        m = build(explicit, M=M, dzt=dzt, friction=False, wind=False, bump=True)
        h = m.halo
        e0 = float(np.sum((np.asarray(m.p_s) if np.asarray(m.p_s).ndim == 2
                           else np.asarray(m.p_s)[1])[h:-h, h:-h]**2))
        st, nst, umax = run(m, explicit, 200)
        if st is None:
            print('  %s: BLEW UP at step %d, |u| %.3g' % (name, nst, umax))
        else:
            e1 = float(np.sum(st['eta']**2))*grav**2
            print('  %s: after 200 steps max|u| %.4f m/s, max|eta| %.3f m, p_s variance ratio %.3f'
                  % (name, umax, np.max(np.abs(st['eta'])), e1/max(e0, 1e-300)))
            check('bump %s: p_s variance ratio < 1 (no growth)' % name, e1/max(e0, 1e-300), 1.0)


def case_polar(ndays):
    """POP-like regime: 0.1 deg cells at 70-79 N over 5000 m, dt 216 s, so
    M_cfl ~ 25 as in the POP run and integer M resolves the margin. This is
    the case that matters for the 1/10 deg blow-up."""
    global dxt_gl, dyt_gl, dzt_deep, x0, y0, dt, nx, ny, nz
    keep = (dxt_gl, dyt_gl, dzt_deep, x0, y0, dt, nx, ny, nz)
    nx = ny = 30
    dxt_gl = np.full(nx, 0.1); dyt_gl = np.full(ny, 0.1)
    dzt_deep = np.array([1400., 1300., 1200., 1100.])          # H = 5000 m
    x0, y0, dt = 0.0, 70.0, 216.0
    mcfl, c = M_cfl(dzt_deep)
    nsteps = int(ndays*86400/dt)
    print('\n=== POP-like polar box: 0.1 deg at %.0f-%.0f N, H %.0f m, c %.0f m/s, '
          'dt %.0f s, M_cfl = %.1f, %d steps'
          % (y0, y0 + dyt_gl.sum(), dzt_deep.sum(), c, dt, mcfl, nsteps))
    ref, nst, uref = run(build(False, dzt=dzt_deep), False, nsteps)
    if ref is None:
        print('implicit BLEW UP at step %d (|u| %.3g)' % (nst, uref))
    else:
        print('implicit: max|u| %.4f m/s, max|eta| %.4f m' % (uref, np.max(np.abs(ref['eta']))))
    du = []
    for fac in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        M = max(1, int(round(fac*mcfl)))
        st, nst, umax = run(build(True, M=M, dzt=dzt_deep), True, nsteps)
        if st is None:
            print('  M = %4d (%.2f M_cfl): BLEW UP at step %d, |u| %.3g' % (M, fac, nst, umax))
            if fac >= 1.2:
                check('polar box: explicit stable at %.1f M_cfl' % fac, 1.0, 0.0)
        elif ref is None:
            print('  M = %4d (%.2f M_cfl): max|u| %.4f' % (M, fac, umax))
        else:
            d = compare(st, ref)
            print('  M = %4d (%.2f M_cfl): max|u| %.4f, vs implicit: u %.2e eta %.2e'
                  % (M, fac, umax, d['u'], d['eta']))
            if fac >= 1.2:
                du.append(d['u'])
    if du:
        check('polar box: u vs implicit at M >= 1.2 M_cfl', max(du), U_TOL)
        check('polar box: no growth of the u difference with M', du[-1]/max(du[0], 1e-300), GROWTH_TOL)
    dxt_gl, dyt_gl, dzt_deep, x0, y0, dt, nx, ny, nz = keep


if __name__ == '__main__':
    ndays = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    case_sweep(ndays, dzt_deep, 'deep basin')
    case_sweep(ndays, dzt_deep/10., 'shallow basin (100 m)')
    case_topography(ndays, dzt_deep)
    case_bump(dzt_deep)
    case_polar(min(ndays, 0.25))
    print('\nmunk_free_surface_test: %s' % ('ALL PASS' if not FAILS else 'FAIL: ' + '; '.join(FAILS)))
    sys.exit(1 if FAILS else 0)
