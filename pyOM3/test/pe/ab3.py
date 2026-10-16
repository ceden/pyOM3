"""Test 11: the true third-order Adams-Bashforth weights (23/12, -16/12, 5/12),
step(m, n, 'ab3') of config.py with the cold-start ramp of pe's loop(),
against the AB2 with eps 0.1 (1.6, -0.6, 0) of pyOM2 and of the coupled runs.

  11a restart     bit for bit with AB3 (the ramp counter m._nstep belongs to
                  the restart like _tau3), both free surface paths, with and
                  without vadv_sub_cycling
  11b content     per step: heat content change = the surface forcing with the
                  THREE weights actually used (+ the surface-face flux with
                  vadv_sub_cycling)                                 round-off
  11c uniform     a uniform tracer stays uniform with AB3
  11d identities  KE_ph = CONV and the exact continuity of (u, v, w) with AB3
  11e LEAK        the energy the time stepping loses. Frictionless, unforced
                  box: a warm anomaly adjusts, internal waves and inertial
                  oscillations remain. E = KE + PE_surface, and
                     LEAK = [E(t1) - E(t0)] - int (KE_adv + KE_cor + KE_ph
                                                   + KE_ps + PEs_dot) dt
                  with the rates from the state (trapezoid in time), relative
                  to int |KE_ph| dt, the energy that went through. Measured
                  AFTER the first 100 x 900 s: the implicit free surface
                  damps the barotropic waves of the adjustment completely at
                  any dt (1.5 percent of the throughput) and would hide the
                  AB part. Expected: AB2 eps is first order, LEAK halves with
                  dt; AB3 has no such term, at least 20 times smaller.
"""
import os, sys, io, contextlib; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, state, line, FIELDS, GRAV, OMEGA
import pyOM3 as OM
from pyOM3.density import get_rho
AB2 = (1.6, -0.6, 0.0)


def restart(N=6):
    from restart import RESTART_FIELDS
    ok = True
    for explicit in (False, True):
        for vadv in (0, 2):
            kw = dict(explicit=explicit, vadv=vadv, tracer_flux=True)
            a = state(step(small_model(**kw), 2*N, 'ab3'))
            m = step(small_model(**kw), N, 'ab3')
            saved = {f: np.array(getattr(m, f)) for f in RESTART_FIELDS if hasattr(m, f)}
            m2 = small_model(**kw); m2._tau3 = m._tau3; m2._nstep = m._nstep
            for f, v in saved.items():
                setattr(m2, f, OM.np.asarray(v))   # back to the backend's arrays (JAX)
            b = state(step(m2, N, 'ab3'))
            d = max(float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS)
            ok &= line('11a restart bit-identical, AB3, %s, M = %d' % ('explicit' if explicit else 'implicit', vadv), d, 0.0)
    return ok


def content(nsteps=12):
    ok = True
    for vadv in (0, 2):
        m = small_model(vadv=vadv, tracer_flux=bool(vadv)); h = m.halo; sl = slice(h, -h); ks = -h-1
        area = (np.asarray(m.dxt)[None, :]*np.asarray(m.cost)[:, None]*np.asarray(m.dyt)[:, None])
        vol = np.asarray(m.dzt)[:, None, None]*area[None]*np.asarray(m.maskT)
        C = lambda a: float(np.sum((np.asarray(a)*vol)[sl, sl, sl]))
        F = float(np.sum((np.asarray(m._qsurf)*np.asarray(m.maskT)[ks]*area)[sl, sl]))
        res = 0.0; hist = [0.0, 0.0]                       # the forcing of the two steps before
        for n in range(nsteps):
            T0 = C(m.temp); step(m, 1, 'ab3'); A_, B_, C_ = m._ab
            fT = float(np.sum((np.asarray(m.vadv_fs_flux['temp'])*area)[sl, sl])) if vadv else 0.0
            res = max(res, abs((C(m.temp) - T0)/m.dt - (A_*F + B_*hist[0] + C_*hist[1]) - fT)/abs(T0/m.dt))
            hist = [F, hist[0]]
        ok &= line('11b heat content = AB3-weighted forcing (+ face flux), M = %d' % vadv, res, 1e-14, ' of content')
    return ok


def uniform_and_identities(nsteps=30):
    from identities import ke_identities, continuity_residual
    ok = True
    for vadv in (0, 2):
        m = small_model(tracer_flux=True, vadv=vadv); h = m.halo; conv = 0.0
        for n in range(nsteps):
            step(m, 1, 'ab3'); conv = max(conv, ke_identities(m)[0])
        s = np.asarray(m.salt)[h:-h, h:-h, h:-h][np.asarray(m.maskT)[h:-h, h:-h, h:-h] > 0]
        ok &= line('11c uniform tracer stays uniform, AB3, M = %d' % vadv, float(np.ptp(s)), 1e-11)
        ok &= line('11d KE_ph = CONV, AB3, M = %d' % vadv, conv, 1e-12)
        ok &= line('11d continuity of (u, v, w), AB3, M = %d' % vadv, continuity_residual(m), 0.0)
    return ok


def _box(dt, vadv, nx=16, ny=12, nz=10):
    """frictionless, unforced: no TKE, no lateral or vertical friction"""
    from pyOM3.pe_realistic.model import model
    par = dict(nx=nx, ny=ny, nz=nz, dt=dt, periodic_in_x=False, periodic_in_y=False, eq_of_state=1,
               explicit_free_surface=False, implicit_free_surface=True, superbee_advection=True,
               dxt=np.linspace(0.9, 1.1, nx), dyt=np.linspace(0.8, 1.2, ny), dzt=np.full(nz, 100.),
               coord_degree=True, x_origin=10., y_origin=25., epsilon=1e-14, max_iterations=5000)
    with contextlib.redirect_stdout(io.StringIO()):
        m = model(par)
    h = m.halo
    m.coriolis_t = OM.np.asarray(2*OMEGA*np.sin(np.deg2rad(np.asarray(m.yt)))[:, None]*np.ones((m.Ny, m.Nx)))
    jg = (np.arange(m.Ny)-h)[:, None]*np.ones((1, m.Nx)); ig = (np.arange(m.Nx)-h)[None, :]*np.ones((m.Ny, 1))
    kb = np.ones((m.Ny, m.Nx)); kb[(jg >= 8) & (ig <= 3)] = nz-2; kb[(jg < 0) | (jg >= ny) | (ig < 0) | (ig >= nx)] = 0
    m.kbot = m.apply_bc(OM.np.asarray(kb)); m.vadv_sub_cycling = vadv; m.prepare_for_run()
    zt = np.asarray(m.zt); H = nz*100.
    T = 4. + 16.*(zt/H)[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx))
    T += 2.0*np.exp(-((ig-nx/2)/3.)**2 - ((jg-ny/2)/2.)**2)[None]*(zt/H)[:, None, None]
    m.temp = m.apply_bc(OM.np.asarray(T))*m.maskT; m.salt = 35.*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state); m._tau3 = (2, 1, 0)
    return m


def _rates(m):
    """E = KE + PE_surface, the sum of the rates of the KE budget formed from the
    state with the package's operators (as exp_ocean.ke_budget), and |KE_ph|"""
    h = m.halo; ks = -h-1; I = (slice(h, -h),)*3; I2 = (slice(h, -h),)*2
    rx = lambda a, s: np.roll(a, s, axis=-1); ry = lambda a, s: np.roll(a, s, axis=-2); rz = lambda a, s: np.roll(a, s, axis=0)
    u, v = np.asarray(m.u), np.asarray(m.v); mU, mV, mT = (np.asarray(x) for x in (m.maskU, m.maskV, m.maskT))
    dz3 = np.asarray(m.dzt_z); cost, cosu = np.asarray(m.cost_y), np.asarray(m.cosu_y)
    dxt, dxu, dyt, dyu = (np.asarray(x) for x in (m.dxt_x, m.dxu_x, m.dyt_y, m.dyu_y))
    Au, Av, At = (np.asarray(x) for x in (m.area_u, m.area_v, m.area_t))
    mu = (mU*Au*dz3)[I]; mv = (mV*Av*dz3)[I]
    KEu = lambda a, b: float(np.sum((u*a)[I]*mu) + np.sum((v*b)[I]*mv))
    p_s = np.asarray(m.p_s[m._tau3[1]])
    fe, fn, ft = (np.asarray(x) for x in m.mom_flux_2nd_u(m.u, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW))
    dua = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mU/(Au*dz3)
    fe, fn, ft = (np.asarray(x) for x in m.mom_flux_2nd_v(m.v, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW))
    dva = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mV/(Av*dz3)
    duc, dvc = (np.asarray(x) for x in m.coriolis_and_metric(m.u, m.v, m.coriolis_t, m.maskU, m.maskV))
    p_h = np.asarray(m.hydrostatic_pressure(m.rho, OM.np.zeros_like(m.rho), m.maskT))
    duph = -(rx(p_h, -1) - p_h)/(cost*dxu)*mU; dvph = -(ry(p_h, -1) - p_h)/dyu*mV
    dups = -(rx(p_s, -1) - p_s)[None]/(cost*dxu)*mU; dvps = -(ry(p_s, -1) - p_s)[None]/dyu*mV
    divU = np.sum(((u*mU - rx(u*mU, 1))/(cost*dxt) + ((v*mV*cosu) - ry(v*mV*cosu, 1))/(cost*dyt))*dz3*mT, axis=0)
    A2 = (mT[ks]*At[0])[I2]
    E = 0.5*float(np.sum(u[I]**2*mu) + np.sum(v[I]**2*mv)) + 0.5*float(np.sum((p_s*p_s/GRAV)[I2]*A2))
    ph = KEu(duph, dvph)
    return E, KEu(dua, dva) + KEu(duc, dvc) + ph + KEu(dups, dvps) - float(np.sum((p_s*divU)[I2]*A2)), abs(ph)


def _leak(ab, dt, vadv, T0=100*900., T=100*900.):
    m = _box(dt, vadv); step(m, int(round(T0/dt)), ab)
    E0, r0, _ = _rates(m); acc = thru = 0.0
    for n in range(int(round(T/dt))):
        step(m, 1, ab); E, r, t = _rates(m); acc += 0.5*(r0 + r)*dt; thru += t*dt; r0 = r
    return (E - E0 - acc)/thru


def leak():
    a1, a2 = _leak(AB2, 900., 2), _leak(AB2, 450., 2)
    b1 = _leak('ab3', 900., 2); b0 = _leak('ab3', 900., 0)
    print('%-56s %+10.2e  (of the energy throughput)' % ('11e LEAK, AB2 eps 0.1, dt 900', a1))
    print('%-56s %+10.2e' % ('11e LEAK, AB2 eps 0.1, dt 450', a2))
    print('%-56s %+10.2e' % ('11e LEAK, AB3, dt 900, M = 2', b1))
    print('%-56s %+10.2e' % ('11e LEAK, AB3, dt 900, M = 0', b0))
    g1 = 1.6 < a1/a2 < 2.4
    print('%-56s %10.2f  (first order: 2, tol 1.6-2.4)  %s' % ('11e AB2 eps: LEAK(dt)/LEAK(dt/2)', a1/a2, 'PASS' if g1 else 'FAIL'))
    ok = line('11e |LEAK AB3| / |LEAK AB2 eps| at dt 900, M = 2', abs(b1/a1), 0.05)
    ok &= line('11e |LEAK AB3| / |LEAK AB2 eps| at dt 900, M = 0', abs(b0/a1), 0.05)
    return ok & g1


def run():
    ok = True
    for t in (restart, content, uniform_and_identities, leak):
        ok &= t()
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
