"""Shallow ragged coast at 73 S under the split-explicit free surface, with and
without bottom friction (implicit free surface as control). Reproduces the POP
1/10 deg failure of Sep 15 2026: a grid-scale checkerboard in p_s over 30 m
shelf cells that the implicit solver damps and the forward-backward sub cycle
does not; quadratic bottom friction is the remedy split-explicit models use.
  PYOM3=<dir containing pyOM3> python3 shelf_coast_test.py [nsteps]"""
import sys, os, numpy as np
sys.path.insert(0, os.environ.get('PYOM3', '.')); import pyOM3 as OM
from pyOM3.pe_realistic.model import model as mr
from pyOM3.density import get_rho
nx = ny = 40; nz = 10
dzt = np.array([160., 145., 130., 115., 100., 85., 70., 55., 40., 25.])      # smooth, top cell 25 m, shelf = 3 levels = 120 m -> too deep;
dzt = np.array([70., 65., 60., 55., 50., 45., 40., 35., 30., 25.])           # H 475 m, 3 top levels = 90 m
H = dzt.sum(); dt = 216.; grav = 9.81; lat0 = -73.0; deg = 2*np.pi*6.371e6/360.
NSHELF = 3                                                                   # wet levels on the shelf


def make(explicit, M, bf, seed=3):
    par = dict(nx=nx, ny=ny, nz=nz, dt=dt, periodic_in_x=True, periodic_in_y=False, eq_of_state=1,
               explicit_free_surface=explicit, implicit_free_surface=not explicit, superbee_advection=True,
               vmix_scheme='implicit', enable_conv_adjust=False, kappa_T=1e-4, A_v=1e-3, Ahbi=2.7e10,
               biha_friction_cosPower=1.5, epsilon=1e-12, max_iterations=5000,
               dxt=np.full(nx, 0.1), dyt=np.full(ny, 0.1), dzt=dzt, coord_degree=True, x_origin=0., y_origin=lat0)
    m = mr(par); h = m.halo; m.sub_cycling = M; m.free_surface_tracer_flux = True
    if bf:
        m.enable_quadratic_bottom_friction = True; m.r_quad_bot = 2e-3
    m.coriolis_t = OM.np.asarray(2*np.pi/43082.*np.sin(np.deg2rad(np.asarray(m.yt)))[:, None]*np.ones((m.Ny, m.Nx)))
    rng = np.random.default_rng(seed); kb = np.ones((m.Ny, m.Nx))
    j = np.arange(m.Ny)[:, None]*np.ones((1, m.Nx))
    kb[j <= h+12] = nz - NSHELF + 1                       # southern shelf, NSHELF wet levels
    kb[(j <= h+8) & (rng.random((m.Ny, m.Nx)) < 0.35)] = 0  # ragged land blobs along the coast
    kb[j <= h+3] = 0; kb[0, :] = 0; kb[-1, :] = 0
    m.kbot = OM.np.asarray(kb); m.prepare_for_run()
    zt = np.asarray(m.zt) - H; T = -1.5 + 3.0*(1 + zt/H)
    m.temp = m.apply_bc(OM.np.asarray(T[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx))))*m.maskT; m.salt = 34.*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    taux = 2e-4*np.ones((m.Ny, m.Nx)); tauy = -1e-4*np.ones((m.Ny, m.Nx))
    def sf(self, tau):
        self.du = OM.add_to_array(self.du, (tau, -h-1, slice(None), slice(None)), taux*self.maskU[-h-1]/dzt[-1])
        self.dv = OM.add_to_array(self.dv, (tau, -h-1, slice(None), slice(None)), tauy*self.maskV[-h-1]/dzt[-1])
    m.set_forcing = sf.__get__(m); return m


def go(explicit, bf, M, nst):
    m = make(explicit, M, bf); h = m.halo; tau, taum1, taum2 = 2, 1, 0; out = []
    for n in range(nst):
        (m.time_step_explicit_pressure if explicit else m.time_step_implicit_pressure)(1.6, -0.6, 0., tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        u = np.asarray(m.u)
        if not np.isfinite(u).all() or np.abs(u).max() > 20:
            return out + ['BLEW UP at step %d' % (n+1)]
        if (n+1) % (nst//4) == 0:
            ps = np.asarray(m.p_s); ps = ps[taum1] if ps.ndim == 3 else ps
            a = np.where(np.asarray(m.maskT)[-h-1][h:-h, h:-h] > 0, ps[h:-h, h:-h], 0)
            d2 = a[:, 2:] - 2*a[:, 1:-1] + a[:, :-2]
            out.append('n=%4d max|u| %.3f  max|eta| %.3f  2dx share of p_s %.2f'
                       % (n+1, np.abs(u).max(), np.abs(a).max()/grav, (d2**2).mean()/(16*(a**2).mean() + 1e-30)))
    return out


if __name__ == '__main__':
    nst = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    c = np.sqrt(grav*H); dx = 0.1*deg*np.cos(np.deg2rad(lat0)); mcfl = dt*c*np.sqrt(1/dx**2 + 1/(0.1*deg)**2); M = int(1.5*mcfl) + 1
    print('box: dx %.0f m, H %.0f m, shelf %d levels = %.0f m, M_cfl %.1f -> M %d, %d steps'
          % (dx, H, NSHELF, dzt[-NSHELF:].sum(), mcfl, M, nst))
    for explicit, bf, nm in ((False, False, 'implicit, no bottom friction'), (True, False, 'explicit, no bottom friction'),
                             (True, True, 'explicit, quadratic bottom friction C_d 2e-3'), (False, True, 'implicit, quadratic bottom friction')):
        print(nm); [print('   ' + l) for l in go(explicit, bf, M, nst)]
