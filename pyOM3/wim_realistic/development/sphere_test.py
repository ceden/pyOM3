"""
Sphere acceptance tests for wim_realistic (coord_degree=True).

1. CONSERVATION: adiabatic transport on the sphere conserves the
   area-weighted action  sum A area_t dft dd  to round-off (flux form:
   the metric divergences telescope against the area weights; x periodic,
   y closed by the land ring, frequency closed, direction periodic).

2. GREAT-CIRCLE RAY: a packet launched at (lam0, phi0) with direction
   theta0 must follow a great circle. Checked against the ray equations
     dlam/dt = c_g sin th/(R cos phi),  dphi/dt = c_g cos th/R,
     dth/dt  = c_g sin th tan phi/R
   (RK4 reference) and against the Clairaut invariant
     sin th cos phi = const.
   This exercises the metric fluxes, the cos-phi weighting and the
   great-circle theta flux against each other: if any one of the three
   is wrong, the ray misses or the action drifts.

3. UNIFORM-STATE CONSISTENCY: a state uniform in (lam, phi, theta) is an
   exact steady state of the continuous equations (the metric divergence
   of the cos-phi flux cancels the divergence of the great-circle theta
   flux). Discretely the cancellation is second order; measured at two
   resolutions.

4. ENERGY CONSISTENCY ON THE SPHERE: with a prescribed current and
   energy_consistent=True, adiabatic transport conserves the absolute
   energy  sum Omega A area_t dft dd  (Omega = omega + k.U) to round-off,
   with the coastal export booked by open_coast_loss when open_coasts.

Run from the pyOM3 parent directory:
    python3 -m pyOM3.wim_realistic.development.sphere_test
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', '..')))
import numpy as np
import pyOM3 as OM
from pyOM3.wim_realistic.model import model


R = 6370.0e3


def build(par, extra={}):
    p = dict(par); p.update(extra)
    M = model(p, halo_points=1)
    M.setup()
    return M


def action_total(M):
    h = M.halo; sl = slice(h, -h)
    w = M.dft[sl][None, :, None, None] * M.dd * M.area_t[h:-h, h:-h]
    return float(np.sum(M.A[sl, sl, sl, sl] * w))


def abs_energy_total(M):
    h = M.halo; sl = slice(h, -h)
    s = M.sinth[sl][:, None, None, None]; c = M.costh[sl][:, None, None, None]
    om = M.omega_t[sl][None, :, None, None]
    Om = om + om**2 / M.grav * (M.ocean_U[None, sl, h:-h, h:-h] * s
                                + M.ocean_V[None, sl, h:-h, h:-h] * c)
    w = M.dft[sl][None, :, None, None] * M.dd * M.area_t[h:-h, h:-h]
    return float(np.sum(Om * M.A[sl, sl, sl, sl] * w))


def euler_advect(M, nsteps, book_loss=False):
    """adiabatic propagation only (no sources, no floor), Euler."""
    for n in range(nsteps):
        M.xdot, M.ydot, M.fdot, M.ddot = \
            M.mk_advection(M.xdot, M.ydot, M.fdot, M.ddot)
        M.dA = M.add_propagation(M.A, M.dA, M.xdot, M.ydot, M.fdot, M.ddot, 0)
        M.A = M.A + M.dt * M.dA[0]
        if book_loss:
            M.loss_A += M.dt * M._lossA_lv[0]
            M.loss_Eabs += M.dt * M._lossO_lv[0]
        if hasattr(M, 'maskTp'):
            M.A = M.A * M.maskTp[None, None, :, :]
        M.A = M.apply_bc(M.A)
    return M


def centroid(M):
    """action-weighted packet position (degrees) and mean direction."""
    h = M.halo; sl = slice(h, -h)
    Ai = M.A[sl, sl, sl, sl]
    w = Ai * (M.dft[sl][None, :, None, None] * M.dd
              * M.area_t[h:-h, h:-h])
    W = w.sum()
    lam = float((w.sum(axis=(0, 1)) * M.xt_gl[None, :]).sum() / W)
    phi = float((w.sum(axis=(0, 1)) * M.yt_gl[:, None]).sum() / W)
    wd = w.sum(axis=(1, 2, 3))
    th = M.theta[sl]
    thm = float(np.arctan2((wd * np.sin(th)).sum(), (wd * np.cos(th)).sum()))
    return lam, phi, thm


def ray_reference(lam0, phi0, th0, cg, T, n=20000):
    """RK4 integration of the great-circle ray equations (radians in,
    degrees out)."""
    def rhs(s):
        lam, phi, th = s
        return np.array([cg * np.sin(th) / (R * np.cos(phi)),
                         cg * np.cos(th) / R,
                         cg * np.sin(th) * np.tan(phi) / R])
    s = np.array([lam0, phi0, th0], dtype=float)
    dt = T / n
    for i in range(n):
        k1 = rhs(s); k2 = rhs(s + dt / 2 * k1)
        k3 = rhs(s + dt / 2 * k2); k4 = rhs(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.degrees(s[0]), np.degrees(s[1]), s[2]


def test_conservation_and_ray(nd=24, verbose=True):
    par = dict(nx=72, ny=61, nf=2, nd=nd, f0=0.05, df0=0.1,
               dt=600.0, dxt=5.0, dyt=2.0, coord_degree=True,
               x_origin=0.0, y_origin=-61.0,
               periodic_in_x=True, periodic_in_y=False,
               snonlin_backend='numpy', floor_action=False)
    M = build(par)
    h = M.halo; sl = slice(h, -h)

    # monochromatic packet at the equator, one direction bin nearest 45 deg
    th0 = M.theta[sl][np.argmin(np.abs(M.theta[sl] - np.pi / 4))]
    kd = h + int(np.argmin(np.abs(M.theta[sl] - th0)))
    lam0, phi0 = 60.0, 0.0
    gl = np.exp(-(((M.xt_gl[None, :] - lam0) / 8.0) ** 2
                  + ((M.yt_gl[:, None] - phi0) / 5.0) ** 2))
    M.A = OM.modify_array(M.A, (kd, h, slice(h, -h), slice(h, -h)), OM.np.asarray(gl))
    M.A = M.apply_bc(M.A)

    A0 = action_total(M)
    T = 3.0 * 86400.0
    M = euler_advect(M, int(T / M.dt))
    dA = abs(action_total(M) - A0) / A0

    cg = M.grav / 2 / M.omega_t[h]
    lam_c, phi_c, th_c = centroid(M)
    lam_r, phi_r, th_r = ray_reference(np.radians(lam0), np.radians(phi0),
                                       float(th0), cg, T)
    clair0 = np.sin(float(th0)) * np.cos(np.radians(phi0))
    clair = np.sin(th_c) * np.cos(np.radians(phi_c))
    if verbose:
        print('conservation (3 d, sphere): |dA|/A = %.3e  %s'
              % (dA, 'PASS' if dA < 1e-13 else 'FAIL'))
        print('ray, nd=%d: model (lam,phi,theta) = (%7.2f, %6.2f, %6.2f deg)'
              % (nd, lam_c, phi_c, np.degrees(th_c)))
        print('          ref            = (%7.2f, %6.2f, %6.2f deg)'
              % (lam_r, phi_r, np.degrees(th_r)))
        print('          Clairaut sin(th)cos(phi): %.4f -> %.4f (drift %.3f%%)'
              % (clair0, clair, 100 * (clair / clair0 - 1)))
    err = np.hypot(lam_c - lam_r, phi_c - phi_r)
    return dA < 1e-13, err, abs(np.degrees(th_c - th_r))


def test_uniform_state(dy, nd):
    par = dict(nx=24, ny=int(80 / dy) + 1, nf=2, nd=nd, f0=0.05, df0=0.1,
               dt=600.0, dxt=5.0, dyt=dy, coord_degree=True,
               y_origin=-40.0 - dy / 2,
               periodic_in_x=True, periodic_in_y=False,
               snonlin_backend='numpy', floor_action=False)
    M = build(par)
    h = M.halo
    M.A = OM.modify_array(M.A, (slice(h, -h),)*4, 1.0)
    M.A = M.apply_bc(M.A)
    M.xdot, M.ydot, M.fdot, M.ddot = \
        M.mk_advection(M.xdot, M.ydot, M.fdot, M.ddot)
    M.dA = M.add_propagation(M.A, M.dA, M.xdot, M.ydot, M.fdot, M.ddot, 0)
    # interior rows away from the wall rows (the closed boundary is a wall,
    # not part of the uniform-state statement)
    cg = float(M.grav / 2 / M.omega_t[h])
    r = M.dA[0][h:-h, h, 5:-5, h:-h]
    scale = cg / R          # magnitude of each of the two cancelling terms
    return float(np.max(np.abs(r))) / scale


def test_energy_fix_sphere(open_coasts):
    par = dict(nx=48, ny=41, nf=12, nd=24, f0=0.05, df0=0.1,
               dt=300.0, dxt=5.0, dyt=1.5, coord_degree=True,
               y_origin=-30.75,
               periodic_in_x=True, periodic_in_y=False,
               snonlin_backend='numpy', floor_action=False,
               energy_consistent=True, open_coasts=open_coasts)
    M = build(par)
    h = M.halo; sl = slice(h, -h)
    ny, nx = M.ny_loc, M.nx_loc

    # prescribed sheared current through the Doppler machinery
    nz, dz = 30, 5.0
    z = (np.arange(nz) + 0.5) * dz
    prof = np.exp((z - nz * dz) / 40.0)
    yy = (M.yt_gl - M.yt_gl.mean()) / 10.0
    UJ = 1.0 * np.exp(-yy ** 2)[:, None] * np.ones((1, nx))
    UL = prof[:, None, None] * UJ[None, :, :]
    VL = 0.3 * np.roll(UL, 5, axis=-1)
    M.ocean_U,   M.ocean_V   = M.calc_doppler_vel(dz, nz, UL, VL)
    M.ocean_Udf, M.ocean_Vdf = M.calc_doppler_vel_df(dz, nz, UL, VL)

    fseed = np.exp(-((M.ft[sl] - 0.08) / 0.03) ** 2)
    gl = np.exp(-(((M.xt_gl[None, :] - M.xt_gl.mean()) / 25.0) ** 2
                  + (yy ** 2)[:, None]))
    M.A = OM.modify_array(M.A, (sl, sl, sl, sl), OM.np.asarray((0.1 + gl[None, None, :, :])
                          * fseed[None, :, None, None]))
    M.A = M.apply_bc(M.A)

    E0 = abs_energy_total(M)
    M = euler_advect(M, 60, book_loss=open_coasts)
    E1 = abs_energy_total(M) + (M.loss_Eabs if open_coasts else 0.0)
    return abs(E1 - E0) / abs(E0)


if __name__ == '__main__':
    ok = True

    c_ok, err24, dth24 = test_conservation_and_ray(nd=24)
    ok &= c_ok
    _, err48, dth48 = test_conservation_and_ray(nd=48, verbose=False)
    # gates: absolute accuracy after 36 deg of arc. The position error is
    # dominated by the SPATIAL discretisation (5 x 2 deg cells), not by nd,
    # so it is gated in absolute terms (a tenth of the zonal cell size);
    # the direction centroid must sit within half a direction bin of the
    # ray reference at each nd.
    p_ok = err24 < 0.5 and err48 < 0.5
    d_ok = dth24 < np.degrees(np.pi / 24) and dth48 < np.degrees(np.pi / 48)
    print('ray position error: %.2f deg (nd=24), %.2f deg (nd=48) '
          '(gate < 0.5)  %s' % (err24, err48, 'PASS' if p_ok else 'FAIL'))
    print('ray direction error: %.2f deg (gate < 7.5), %.2f deg '
          '(gate < 3.75)  %s' % (dth24, dth48, 'PASS' if d_ok else 'FAIL'))
    ok &= p_ok and d_ok

    r1 = test_uniform_state(dy=2.0, nd=24)
    r2 = test_uniform_state(dy=1.0, nd=48)
    print('uniform-state residual / (c_g/R): %.3e (2 deg, nd=24) -> '
          '%.3e (1 deg, nd=48), ratio %.2f (2nd order ~ 4)'
          % (r1, r2, r1 / r2))
    ok &= r2 < r1

    e_closed = test_energy_fix_sphere(open_coasts=False)
    e_open = test_energy_fix_sphere(open_coasts=True)
    print('absolute-energy closure on the sphere: closed %.3e, '
          'open+booked %.3e  %s'
          % (e_closed, e_open,
             'PASS' if max(e_closed, e_open) < 1e-12 else 'FAIL'))
    ok &= max(e_closed, e_open) < 1e-12

    print('sphere test:', 'ALL PASS' if ok else 'FAILURES')
    sys.exit(0 if ok else 1)
