"""Test 10: tracer advection outside the AB step with the vertical direction
sub-cycled (pe_realistic/vadv_split.py, option vadv_sub_cycling = M).

  10a uniform     a uniform tracer stays uniform in the full model, both free
                 surface paths, M = 1, 2, 4 (the pseudo density and the final
                 velocities do that)                               round-off
     control     the same sweeps WITHOUT the pseudo density must fail this
                 test, otherwise 10a proves nothing
  10b content     per step: content change = AB-weighted surface forcing
                 + the surface-face flux the sub-steps report
                 (vadv_fs_flux), with and without free_surface_tracer_flux
                                                                   round-off
  10c cell        kinematic overturning cell with a narrow downwelling (the
                 staircase-slope situation: vertical CFL ten times the
                 horizontal one), a halocline advected through it.
                 unsplit at CFL 0.64 must DIE (that is the icon01 crash),
                 split M = 1 must hold 0.64 and 0.9 without any new extremum
                 and conserve the content; at CFL 1.8 M = 1 must die and
                 M = 2 must hold (the limit is w dt/dz <= M)
  10d order       split against unsplit in the forced box: the difference is
                 the first-order splitting error, it must halve with dt
  10e identities  KE_ph = CONV and the exact continuity of (u, v, w) with the
                 option on (the tracers now use the FINAL u, v, w)
  10f restart     bit for bit with the option on, both paths (no new state)
  10g M           M = 1 against M = 4 at small CFL: a diagnostic, no tolerance
The decomposition test (decomposition.py) has its own 'split' case.
"""
import os, sys, io, contextlib; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from config import small_model, step, state, line, FIELDS
import pyOM3 as OM


def _wet(m):
    h = m.halo; return np.asarray(m.maskT)[h:-h, h:-h, h:-h] > 0


def uniform(nsteps=40):
    ok = True
    for explicit in (False, True):
        for M in (1, 2, 4):
            m = small_model(explicit=explicit, tracer_flux=True, vadv=M); h = m.halo
            step(m, nsteps)
            s = np.asarray(m.salt)[h:-h, h:-h, h:-h][_wet(m)]
            ok &= line('10a uniform tracer stays uniform, %s, M = %d' % ('explicit' if explicit else 'implicit', M),
                       float(np.ptp(s)), 1e-11)
    # control: the same sweeps without the pseudo density
    m = small_model(tracer_flux=True, vadv=2); h = m.halo
    sb = m._vadv_superbee
    m._vadv_superbee = lambda T, vel, rho, roll, dist, mask, dts, ex: sb(T*rho, vel, 1.0, roll, dist, mask, dts, ex)
    step(m, nsteps)                                   # T*rho = q: fluxes formed with the content
    s = np.asarray(m.salt)[h:-h, h:-h, h:-h][_wet(m)]
    sp = float(np.ptp(s)); good = sp > 1e-8
    print('%-56s %10.2e  (must be > 1e-08)          %s' % ('10a control: without the pseudo density', sp, 'PASS' if good else 'FAIL'))
    return ok & good


def content(nsteps=12):
    ok = True
    for flag in (False, True):
        m = small_model(tracer_flux=flag, vadv=2); h = m.halo; sl = slice(h, -h); ks = -h-1
        m.salt = m.salt + 0.5*m.temp*m.maskT/20.                 # a salinity with structure
        area = (np.asarray(m.dxt)[None, :]*np.asarray(m.cost)[:, None]*np.asarray(m.dyt)[:, None])
        vol = np.asarray(m.dzt)[:, None, None]*area[None]*np.asarray(m.maskT)
        C = lambda a: float(np.sum((np.asarray(a)*vol)[sl, sl, sl]))
        F = float(np.sum((np.asarray(m._qsurf)*np.asarray(m.maskT)[ks]*area)[sl, sl]))
        A_, B_ = 1.6, -0.6; resT = resS = big = 0.0; Fold = 0.0
        for n in range(nsteps):
            T0, S0 = C(m.temp), C(m.salt)
            step(m, 1)
            fT = float(np.sum((np.asarray(m.vadv_fs_flux['temp'])*area)[sl, sl]))
            fS = float(np.sum((np.asarray(m.vadv_fs_flux['salt'])*area)[sl, sl]))
            # residual relative to the content change of the step (the content
            # itself is 1e9 times larger, so this is near its round-off)
            dT, dS = C(m.temp) - T0, C(m.salt) - S0
            resT = max(resT, abs(dT/m.dt - (A_*F + B_*Fold) - fT)/abs(T0/m.dt))
            resS = max(resS, abs(dS/m.dt - fS)/abs(S0/m.dt))
            big = max(big, abs(fT)/abs(T0/m.dt))
            Fold = F
        tag = 'surface face -w T' if flag else 'surface face closed'
        ok &= line('10b heat content = forcing + face flux, %s' % tag, resT, 1e-14, ' of content')
        ok &= line('10b salt content = face flux, %s' % tag, resS, 1e-14, ' of content')
        if flag:
            print('%-56s %10.2e  (same units: what the test would see without the booking)' % ('10b    size of the face flux term', big))
    return ok


def _cell_model(M, dt, nx=24, nz=32, dx=2000., dz=5.):
    from pyOM3.pe_realistic.model import model
    par = dict(nx=nx, ny=3, nz=nz, dt=dt, periodic_in_x=False, periodic_in_y=True, eq_of_state=1,
               explicit_free_surface=False, implicit_free_surface=True, superbee_advection=True,
               dxt=np.full(nx, dx), dyt=np.full(3, dx), dzt=np.full(nz, dz), coord_degree=False,
               x_origin=0., y_origin=0., epsilon=1e-12, max_iterations=500)
    with contextlib.redirect_stdout(io.StringIO()):
        m = model(par)
    h = m.halo
    kb = np.ones((m.Ny, m.Nx)); kb[:, :h] = 0; kb[:, -h:] = 0
    m.kbot = m.apply_bc(OM.np.asarray(kb)); m.vadv_sub_cycling = M; m.prepare_for_run()
    return m


def _cell(M, cfl, nsteps=300, dt=100., dx=2000., dz=5.):
    """steady streamfunction psi = sin(pi z/H) exp(-((x-L/2)/1.5dx)^2), w from
    the model's own continuity; returns (steps survived, overshoot, content error)"""
    m = _cell_model(M, dt); h = m.halo; nz = m.Nz-2*h; nx = m.Nx-2*h
    xu = (np.arange(m.Nx)-h+1.0)*dx; zw = (np.arange(m.Nz)-h+1.0)*dz; H = nz*dz; L = nx*dx
    psi = np.sin(np.pi*zw/H)[:, None]*np.exp(-((xu-0.5*L)/(1.5*dx))**2)[None, :]
    u = -(psi - np.roll(psi, 1, axis=0))/dz
    m.u = OM.np.asarray(np.repeat(u[:, None, :], m.Ny, axis=1)*np.asarray(m.maskU)); m.v = 0*m.u
    w = np.asarray(m.vertical_velocity(m.u, m.v, m.w, m.maskW))
    m.u = m.u*(cfl/(np.abs(w).max()*dt/dz)); m.w = m.vertical_velocity(m.u, m.v, m.w, m.maskW)
    zt = (np.arange(m.Nz)-h+0.5)*dz
    S = 34. + 0.5*np.tanh((zt-0.6*H)/(2*dz))[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx))
    m.salt = m.apply_bc(OM.np.asarray(S))*m.maskT; m.temp = m.salt*1
    I = (slice(h, -h),)*3; wet = np.asarray(m.maskT)[I] > 0
    s0 = np.asarray(m.salt)[I][wet]; c0 = s0.sum(); lo, hi = s0.min(), s0.max()
    A, B, C = 1.6, -0.6, 0.; tau, t1, t2 = 2, 1, 0
    for n in range(nsteps):
        if M > 0:
            m.dsalt = m.vadv_zero_tendency(m.dsalt, tau)
            m.salt, _ = m.vadv_tracer(m.salt, m.dsalt, m.u, m.v, m.w, A, B, C, tau, t1, t2,
                                      m.maskU, m.maskV, m.maskW, m.maskT)
        else:
            m.dsalt = m.tracer_advection(m.salt, m.dsalt, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
            m.temp, m.salt = m.update_thermodynamics_AB3(A, B, C, tau, t1, t2, m.temp, m.salt,
                                                         m.dtemp, m.dsalt, m.p0, m.maskT)
        tau, t1, t2 = (tau+1) % 3, (t1+1) % 3, (t2+1) % 3
        s1 = np.asarray(m.salt)[I][wet]
        if not np.isfinite(s1).all() or np.abs(s1).max() > 1e3:
            return n, np.inf, np.inf
    return nsteps, float(max(s1.max()-hi, lo-s1.min())), float(abs(s1.sum()-c0)/c0)


def cell():
    ok = True
    def dies(name, r):
        good = not np.isfinite(r[1])
        print('%-56s %10s  (must blow up)             %s' % (name, 'step %d' % r[0] if good else 'survived', 'PASS' if good else 'FAIL'))
        return good
    with np.errstate(all='ignore'):
        ok &= dies('10c cell, CFL 0.64, unsplit (the icon01 crash)', _cell(0, 0.64))
        ok &= dies('10c cell, CFL 1.80, split M = 1', _cell(1, 1.8))
    for M, cfl in ((1, 0.64), (1, 0.9), (2, 1.8), (4, 3.6)):
        n, over, cons = _cell(M, cfl)
        ok &= line('10c cell, CFL %.2f, split M = %d: new extrema' % (cfl, M), over, 1e-10)
        ok &= line('10c cell, CFL %.2f, split M = %d: content' % (cfl, M), cons, 1e-13)
    return ok


def order(tend=16*900.0):
    d = []
    dt0 = config.DT
    try:
        for dt in (900.0, 450.0, 225.0):
            config.DT = dt; n = int(round(tend/dt))
            a = state(step(small_model(), n)); b = state(step(small_model(vadv=2), n))
            d.append(float(np.mean(np.abs(a['temp'] - b['temp']))))
    finally:
        config.DT = dt0
    r1, r2 = d[0]/d[1], d[1]/d[2]
    print('%-56s %10s  (mean |dT| %.1e %.1e %.1e K)' % ('10d split - unsplit at dt, dt/2, dt/4', '', d[0], d[1], d[2]))
    good = (1.3 < r1 < 2.6) and (1.3 < r2 < 2.6)
    print('%-56s %5.2f %4.2f  (first order: 2, tol 1.3-2.6)  %s' % ('10d ratios', r1, r2, 'PASS' if good else 'FAIL'))
    return good


def identities_(nsteps=8):
    from identities import ke_identities, continuity_residual
    ok = True
    for explicit in (False, True):
        m = small_model(explicit=explicit, vadv=2); conv = 0.0; tag = 'explicit' if explicit else 'implicit'
        for n in range(nsteps):
            step(m, 1); conv = max(conv, ke_identities(m)[0])
        ok &= line('10e KE_ph = CONV with the option on, %s' % tag, conv, 1e-12)
        ok &= line('10e continuity of (u, v, w) with the option on, %s' % tag, continuity_residual(m), 0.0)
    return ok


def restart(N=6):
    from restart import RESTART_FIELDS
    ok = True
    for explicit in (False, True):
        kw = dict(explicit=explicit, vadv=2, tracer_flux=True)
        a = state(step(small_model(**kw), 2*N))
        m = step(small_model(**kw), N)
        saved = {f: np.array(getattr(m, f)) for f in RESTART_FIELDS if hasattr(m, f)}
        m2 = small_model(**kw); m2._tau3 = m._tau3
        for f, v in saved.items():
            setattr(m2, f, OM.np.asarray(v))   # back to the backend's arrays (JAX)
        b = state(step(m2, N))
        d = max(float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS)
        ok &= line('10f restart bit-identical with the option on, %s' % ('explicit' if explicit else 'implicit'), d, 0.0)
    return ok


def m_dependence(nsteps=40):
    a = state(step(small_model(vadv=1), nsteps)); b = state(step(small_model(vadv=4), nsteps))
    print('%-56s %10.2e  (K, diagnostic: CFL is small in this box)' % ('10g M = 1 against M = 4, max |dT|',
          float(np.max(np.abs(a['temp'] - b['temp'])))))
    return True


def run():
    ok = True
    for t in (uniform, content, cell, order, identities_, restart, m_dependence):
        ok &= t()
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
