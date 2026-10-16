"""Test 2: discrete identities in a short run on the small box (pe family).

Generic ones, computed from the model STATE with the package's own operators
(no configuration bookkeeping), ported from global4_setup.ke_budget:

  conversion   KE_ph = CONV_int + CONV_top          pressure-gradient work of u
               equals the buoyancy flux through the faces plus the surface
               face work, by summation by parts with the divergence that
               defines w                                          round-off
  advection    KE_adv = KE_div                      the centred flux form
               telescopes to -1/2 sum u^2 div(T) with the face transports
               the fluxes use (top face closed)                   round-off
  friction     KE(u_new) - KE(u) = sum u (u_new - u) + 1/2 sum (u_new - u)^2
               for one biharmonic step (stencil + forward-Euler part)  round-off
  continuity   the carried (u, v, w) satisfies the discrete continuity
               equation at the end of the step (w recomputed from the final
               u, v equals the model's w)                          exact
  salt         content conserved (closed box, no surface salt flux)  round-off
  heat         content change = applied surface flux (AB2 end effect)  bound
  tracer flux  uniform tracer stays uniform with free_surface_tracer_flux
The TKE budget identity stays with the global4 configuration: it needs the
closure's lhs/rhs booked inside integrate_tke, which the package does not
expose.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, line, GRAV
import pyOM3 as OM


def ke_identities(m):
    """conversion and advection identities from the current state"""
    h = m.halo; ks = -h-1
    I = (slice(h, -h),)*3; I2 = (slice(h, -h),)*2
    rx = lambda a, s: np.roll(a, s, axis=-1); ry = lambda a, s: np.roll(a, s, axis=-2); rz = lambda a, s: np.roll(a, s, axis=0)
    # the model's own w: both paths must leave (u, v, w) consistent at the end
    # of a step (the explicit path once applied vertical friction after w and
    # gave 1.6e-8 here; fixed Sep 15 2026 by moving the friction before w)
    w_ = m.w; w = np.asarray(w_)
    u, v = np.asarray(m.u), np.asarray(m.v)
    mU, mV, mT, mW = (np.asarray(x) for x in (m.maskU, m.maskV, m.maskT, m.maskW))
    dz3 = np.asarray(m.dzt_z); cost, cosu = np.asarray(m.cost_y), np.asarray(m.cosu_y)
    dxt, dxu, dyt, dyu = (np.asarray(x) for x in (m.dxt_x, m.dxu_x, m.dyt_y, m.dyu_y))
    Au3, Av3, At3 = (np.asarray(x) for x in (m.area_u, m.area_v, m.area_t))
    mu = (mU*Au3*dz3)[I]; mv = (mV*Av3*dz3)[I]
    KEu = lambda a, b: float(np.sum((u*a)[I]*mu) + np.sum((v*b)[I]*mv))
    # pressure-gradient work and its conversion form
    p_h = np.asarray(m.hydrostatic_pressure(m.rho, OM.np.zeros_like(m.rho), m.maskT))
    du_ph = -(rx(p_h, -1) - p_h)/(cost*dxu)*mU; dv_ph = -(ry(p_h, -1) - p_h)/dyu*mV
    KE_ph = KEu(du_ph, dv_ph)
    wk = w*mW
    CONV_int = float(np.sum((wk*(rz(p_h, -1) - p_h)*At3)[h:ks, h:-h, h:-h]))
    CONV_top = -float(np.sum((wk[ks]*p_h[ks]*At3[0])[I2]))
    conv = abs(KE_ph - (CONV_int + CONV_top))/max(abs(KE_ph), 1e-300)
    # advection: flux form vs telescoped divergence form
    fe, fn, ft = (np.asarray(x) for x in m.mom_flux_2nd_u(m.u, m.u, m.v, w_, m.maskU, m.maskV, m.maskW))
    du_adv = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mU/(Au3*dz3)
    fe, fn, ft = (np.asarray(x) for x in m.mom_flux_2nd_v(m.v, m.u, m.v, w_, m.maskU, m.maskV, m.maskW))
    dv_adv = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mV/(Av3*dz3)
    KE_adv = KEu(du_adv, dv_adv)
    utr = u*mU*dz3*dyt; vtr = v*mV*dz3*cosu*dxt; wtr = w*mW*At3
    divT = lambda a, b, c: (a - rx(a, 1)) + (b - ry(b, 1)) + (c - rz(c, 1))
    Tx, Ty, Tz = 0.5*(utr + rx(utr, -1)), 0.5*(vtr + rx(vtr, -1)), 0.5*(wtr + rx(wtr, -1)); Tz[ks] = 0.0
    KE_div = -0.5*float(np.sum((u*u*divT(Tx, Ty, Tz)*mU)[I]))
    Tx, Ty, Tz = 0.5*(utr + ry(utr, -1)), 0.5*(vtr + ry(vtr, -1)), 0.5*(wtr + ry(wtr, -1)); Tz[ks] = 0.0
    KE_div += -0.5*float(np.sum((v*v*divT(Tx, Ty, Tz)*mV)[I]))
    adv = abs(KE_adv - KE_div)/max(abs(KE_adv), 1e-300)
    return conv, adv


def continuity_residual(m):
    """the carried triple (u, v, w) at the end of a step must satisfy the
    discrete continuity equation the model diagnoses w from: recompute w
    from the final u, v and compare with the model's w (relative to max|w|).
    Exact zero on both paths since the friction-order fix of Sep 15 2026."""
    h = m.halo; I = (slice(h, -h),)*3
    w_new = np.asarray(m.vertical_velocity(m.u, m.v, m.w, m.maskW))
    w = np.asarray(m.w)
    return float(np.max(np.abs(w_new - w)[I]))/max(float(np.max(np.abs(w)[I])), 1e-300)


def friction_identity(m):
    """one biharmonic step: KE difference = stencil work + forward-Euler part"""
    h = m.halo; I = (slice(h, -h),)*3
    u, v = np.asarray(m.u), np.asarray(m.v)
    un, vn = (np.asarray(x) for x in m.biharmonic_momentum(m.u, m.v, m.Ahbi, m.maskU, m.maskV))
    mu = (np.asarray(m.maskU)*np.asarray(m.area_u)*np.asarray(m.dzt_z))[I]
    mv = (np.asarray(m.maskV)*np.asarray(m.area_v)*np.asarray(m.dzt_z))[I]
    dKE = 0.5*float(np.sum(((un*un - u*u)[I])*mu) + np.sum(((vn*vn - v*v)[I])*mv))
    stencil = float(np.sum((u*(un - u))[I]*mu) + np.sum((v*(vn - v))[I]*mv))
    fwd = 0.5*float(np.sum(((un - u)**2)[I]*mu) + np.sum(((vn - v)**2)[I]*mv))
    KE = 0.5*float(np.sum((u*u)[I]*mu) + np.sum((v*v)[I]*mv))
    return abs(dKE - (stencil + fwd))/max(KE, 1e-300)        # relative to KE: dKE is a small difference


def run(nsteps=8):
    ok = True
    for explicit in (False, True):
        m = small_model(explicit=explicit); tag = 'explicit' if explicit else 'implicit'
        h = m.halo; sl = slice(h, -h)
        dz = np.asarray(m.dzt)[:, None, None]
        area = np.asarray(m.dxt)[None, :]*np.asarray(m.cost)[:, None]*np.asarray(m.dyt)[:, None]
        vol = dz*area[None]*np.asarray(m.maskT)
        heat0 = float(np.sum((np.asarray(m.temp)*vol)[sl, sl, sl])); salt0 = float(np.sum((np.asarray(m.salt)*vol)[sl, sl, sl]))
        flux = 0.0; conv = adv = 0.0
        for n in range(nsteps):
            step(m, 1)
            flux += float(np.sum((np.asarray(m._qsurf)*np.asarray(m.maskT)[-h-1]*area)[sl, sl]))*m.dt
            c, a = ke_identities(m); conv = max(conv, c); adv = max(adv, a)
        heat1 = float(np.sum((np.asarray(m.temp)*vol)[sl, sl, sl])); salt1 = float(np.sum((np.asarray(m.salt)*vol)[sl, sl, sl]))
        ok &= line('2 KE_ph = CONV_int + CONV_top, %s (max over steps)' % tag, conv, 1e-12)
        ok &= line('2 KE_adv = KE_div (flux form telescopes), %s' % tag, adv, 1e-12)
        ok &= line('2 biharmonic step: dKE = stencil + fwd-Euler, %s' % tag, friction_identity(m), 1e-12)
        ok &= line('2 continuity of the carried (u, v, w), %s' % tag, continuity_residual(m), 0.0)
        ok &= line('2 salt content conserved, %s' % tag, abs(salt1 - salt0)/salt0, 1e-13)
        ok &= line('2 heat content change = surface flux, %s' % tag, abs((heat1 - heat0) - flux)/max(abs(flux), 1e-300), 0.3)
    for flag in (True, False):
        m = small_model(explicit=True, tracer_flux=flag); h = m.halo
        m.salt = 35.0*m.maskT; step(m, nsteps)
        s = np.asarray(m.salt)[h:-h, h:-h, h:-h]; wet = np.asarray(m.maskT)[h:-h, h:-h, h:-h] > 0
        spread = float(np.ptp(s[wet]))
        if flag:
            ok &= line('2 uniform tracer stays uniform (free_surface_tracer_flux)', spread, 1e-12)
        else:
            print('%-56s %10.2e  (pyOM2 form: top-cell -T dt div_h(u) anomaly, diagnostic)' % ('2 uniform tracer spread without the surface flux', spread))
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
