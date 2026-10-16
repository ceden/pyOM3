"""Test 2: discrete identities in a short run on the small box (nh family).

Computed from the model STATE with the package's own operators, for the
rigid-lid and the periodic-in-z model and for the three buoyancy advection
schemes. Each is an identity of the discrete equations, so it holds to
round-off:

  continuity   the carried (u, v, w) is divergence free at the end of the
               step (the projection)                               round-off
  pressure     sum u . grad p = 0 for the projected velocity (summation by
               parts with w = 0 at the lids)                       round-off
  advection    sum (u adv_u + v adv_v + dsqr w adv_w) = 0: the energy
               conserving flux form telescopes to -1/2 sum u^2 div(T)
               with divergence-free transports; u+v part and w part
               separately                                          round-off
  conversion   sum w . 1/2 (b + b_above) = - sum b . bgterm / N0^2, the
               buoyancy work on w equals the exchange with the background
               potential energy (the KE_ph = CONV of the pe family)  round-off
  Coriolis     sum (u du_cor + v dv_cor) = 0 (the four-point average is
               self-adjoint)                                       round-off
  b content    sum of the advection tendency = 0 (flux form, all three
               schemes); sum of the background term = 0 (horizontal
               integral of w vanishes at every level)              round-off
  b variance   2nd order: sum b adv(b) = 0; superbee, upwind3: the sign
               of sum b adv is printed (the CFL-dependent antidiffusive
               part of the limited flux is dissipative over a step, not
               instantaneously, so it is no identity)
  b budget     content change over the run = AB-weighted surface flux
               (unforced: conserved)                               round-off
  friction     sum of the harmonic tendency = 0; one harmonic or
               biharmonic step: dKE = stencil + forward-Euler part;
               biharmonic = harmonic(harmonic)                     round-off
  LES (tke)    sum shear_prod = - sum u du / dt of les_mix_momentum, i.e.
               the resolved energy the closure removes (the documented
               identity of les_shear_production); dz sum buoy_prod =
               d/dt sum z b of les_mix_buoyancy (TKE buoyancy sink = resolved
               PE gain); booked les_ke_loss = actual KE change; TKE budget
               with tke_min = 0: d sum tke = dt sum (shear + buoy + stokes
               + source - diss), the diffusion telescopes          round-off
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, line, interior, ab_weights, FIELDS
import pyOM3 as OM

rel = lambda a, b: abs(a)/max(abs(b), 1e-300)


def ke_identities(m):
    h = m.halo; I = (slice(h, -h),)*3
    rx, ry, rz = m.rollx, m.rolly, m.rollz
    u, v, w, b, p = (np.asarray(getattr(m, f)) for f in FIELDS)
    div = ((u - rx(u, 1))/m.dx + (v - ry(v, 1))/m.dy + (w - rz(w, 1))/m.dz)[I]
    cont = np.abs(div).max()/(np.abs(u[I]).max()/m.dx)
    gpx = (rx(p, -1) - p)/m.dx; gpy = (ry(p, -1) - p)/m.dy
    gpz = np.asarray(m.apply_vertical_flux_bc((rz(p, -1) - p)/m.dz))
    pwork = rel((u*gpx + v*gpy + w*gpz)[I].sum(), np.abs(u*gpx)[I].sum())
    au, av, aw = (np.asarray(x) for x in (m.momentum_advection_u(), m.momentum_advection_v(), m.momentum_advection_w()))
    adv_uv = rel((u*au + v*av)[I].sum(), np.abs(u*au)[I].sum())
    adv_w = rel((m.dsqr*w*aw)[I].sum(), np.abs(w*aw)[I].sum())
    wb = (w*0.5*(b + rz(b, -1)))[I].sum()
    bg = (b*(-0.5*(w + rz(w, 1))*m.N0**2))[I].sum()/m.N0**2
    conv = rel(wb + bg, wb)
    duc = (v + ry(v, 1) + rx(v + ry(v, 1), -1))*0.25*m.f0
    dvc = -(u + rx(u, 1) + ry(u + rx(u, 1), -1))*0.25*m.f0
    cor = rel((u*duc + v*dvc)[I].sum(), np.abs(u*duc)[I].sum())
    return cont, pwork, adv_uv, adv_w, conv, cor


def b_identities(m, b_adv):
    h = m.halo; I = (slice(h, -h),)*3
    b, w = np.asarray(m.b), np.asarray(m.w)
    adv = np.asarray({'2nd': m.buoyancy_advection_2nd, 'superbee': m.buoyancy_advection_superbee,
                      'upwind3': m.buoyancy_advection_upwind3}[b_adv]())
    content = rel(adv[I].sum(), np.abs(adv[I]).sum())
    var = (b*adv)[I].sum()/np.abs(b*adv)[I].sum()
    bgt = (-0.5*(w + m.rollz(w, 1))*m.N0**2)[I]
    bgsum = rel(bgt.sum(), np.abs(bgt).sum())
    return content, var, bgsum


def friction_identities(m):
    h = m.halo; I = (slice(h, -h),)*3
    u, b = np.asarray(m.u), np.asarray(m.b)
    Ah, Av = 2e-3, 1e-3
    tu = np.asarray(m.harmonic(m.u, Ah, Av)); tb = np.asarray(m.harmonic(m.b, Ah, Av))
    csum = max(rel(tu[I].sum(), np.abs(tu[I]).sum()), rel(tb[I].sum(), np.abs(tb[I]).sum()))
    KE = 0.5*(u*u)[I].sum(); out = []
    for un in (u + m.dt*tu, u + m.dt*np.asarray(m.biharmonic(m.u, 2e-2, 5e-3))):
        dKE = 0.5*((un*un - u*u)[I]).sum(); st = (u*(un - u))[I].sum(); fw = 0.5*((un - u)**2)[I].sum()
        out.append(rel(dKE - st - fw, KE))
    bi = np.asarray(m.biharmonic(m.u, 2e-2, 5e-3))
    hh = np.asarray(m.harmonic(m.apply_bc(np.asarray(m.harmonic(m.u, -2e-2, -5e-3))), 1.0, 1.0))
    bihh = np.abs(bi - hh)[I].max()/np.abs(bi[I]).max()
    return csum, out[0], out[1], bihh


def les_identities(m):
    """on the current state: one call of the closure's pieces"""
    h = m.halo; I = (slice(h, -h),)*3; dt = m.dt
    # TKE budget, tke_min = 0 so that the clipping does not enter
    m.tke_min = 0.0
    S = np.asarray(m.shear_prod + m.buoy_prod + (m.stokes_prod if m.les_stokes_prod is not None else 0.0)
                   + (m.tke_source if hasattr(m, 'tke_source') else 0.0))
    tke0 = np.asarray(m.tke)[I].sum()
    m.les_tke_closure()
    dtke = np.asarray(m.tke)[I].sum() - tke0
    rhs = dt*(S[I].sum() - np.asarray(m.tke_diss)[I].sum())
    budget = rel(dtke - rhs, np.abs(S[I]).sum()*dt)
    for f in ('visc_00', 'visc_12', 'visc_13', 'visc_23', 'kappah', 'kappav'):
        setattr(m, f, m.apply_bc(getattr(m, f)))
    # shear production = energy removed from the resolved flow
    u0, v0, w0 = (np.asarray(x).copy() for x in (m.u, m.v, m.w))
    m.les_mix_momentum(m.u, m.v, m.w)
    du, dv, dw = np.asarray(m.u) - u0, np.asarray(m.v) - v0, np.asarray(m.w) - w0
    removed = -((u0*du + v0*dv + m.dsqr*w0*dw)[I].sum())/dt
    sp = np.asarray(m.shear_prod)[I].sum()
    shear = rel(sp - removed, np.abs(np.asarray(m.shear_prod)[I]).sum())
    dKE = 0.5*((np.asarray(m.u)**2 - u0**2 + np.asarray(m.v)**2 - v0**2 + m.dsqr*(np.asarray(m.w)**2 - w0**2))[I].sum())
    booked = -dt*(np.asarray(m.les_ke_loss) + np.asarray(m.les_ke_loss_w))[I].sum()
    book = rel(dKE - booked, abs(dKE))
    # buoyancy production = resolved potential energy gain of the mixing
    z = (np.arange(m.Nz) - h + 0.5)*m.dz
    b0 = np.asarray(m.b).copy()
    m.les_mix_buoyancy()
    dpe = ((np.asarray(m.b) - b0)*z[None, None, :])[I].sum()*m.dz/dt     # from the increment: no cancellation
    bp = m.dz*np.asarray(m.buoy_prod)[I].sum()
    buoy = rel(bp - dpe, m.dz*np.abs(np.asarray(m.buoy_prod)[I]).sum())
    return budget, shear, book, buoy


def run(nsteps=6):
    ok = True
    for kind in ('lid', 'periodic'):
        for b_adv in ('2nd', 'superbee', 'upwind3'):
            tag = '%s, %s' % (kind, b_adv)
            m = small_model(kind, b_adv=b_adv, forcing=False)
            b0 = interior(m, m.b).sum()
            res = np.zeros(6)
            for n in range(nsteps):
                step(m, 1)
                res = np.maximum(res, ke_identities(m))
            cont, pwork, adv_uv, adv_w, conv, cor = res
            content, var, bgsum = b_identities(m, b_adv)
            ok &= line('2 continuity of the carried (u, v, w), %s' % tag, cont, 1e-13)
            ok &= line('2 pressure work of the projected velocity = 0, %s' % tag, pwork, 1e-13)
            ok &= line('2 KE advection u+v part = 0, %s' % tag, adv_uv, 1e-13)
            ok &= line('2 KE advection w part = 0, %s' % tag, adv_w, 1e-13)
            ok &= line('2 buoyancy work = -background PE exchange, %s' % tag, conv, 1e-13)
            ok &= line('2 Coriolis work = 0, %s' % tag, cor, 1e-13)
            ok &= line('2 b advection: content (sum of tendency) = 0, %s' % tag, content, 1e-13)
            if b_adv == '2nd':
                ok &= line('2 b advection: variance conserved, %s' % tag, abs(var), 1e-13)
            else:   # limited and upwind-biased fluxes: the instantaneous sign is not an identity
                print('%-56s %10.2e  (sum b adv / sum |b adv|, diagnostic)' % ('2 b advection: variance tendency, %s' % tag, var))
            ok &= line('2 background term: sum = 0, %s' % tag, bgsum, 1e-13)
            b1 = interior(m, m.b).sum()
            ok &= line('2 b content conserved, unforced, %s' % tag, rel(b1 - b0, np.abs(interior(m, m.b)).sum()), 1e-13)
            if b_adv == '2nd':
                csum, harm, biharm, bihh = friction_identities(m)
                ok &= line('2 harmonic: sum of tendency = 0, %s' % kind, csum, 1e-13)
                ok &= line('2 harmonic step: dKE = stencil + fwd-Euler, %s' % kind, harm, 1e-13)
                ok &= line('2 biharmonic step: dKE = stencil + fwd-Euler, %s' % kind, biharm, 1e-13)
                ok &= line('2 biharmonic = harmonic(harmonic), %s' % kind, bihh, 1e-13)
    # forced: content change = AB-weighted applied surface flux
    m = small_model('lid'); h = m.halo
    b0 = interior(m, m.b).sum(); flux = 0.0
    inc = np.asarray(m._bflux)[h:-h, h:-h].sum()/m.dz*m.dt        # one application (into db[tau] of every step)
    for n in range(nsteps):
        A, B, C = ab_weights(n)                                    # the levels taum1, taum2 hold the same forcing
        flux += inc*(A + (B if n > 0 else 0.0) + (C if n > 1 else 0.0))
        step(m, 1)
    b1 = interior(m, m.b).sum()
    ok &= line('2 b content change = AB-weighted surface flux, lid, forced', rel((b1 - b0) - flux, flux), 1e-12)
    # LES TKE closure
    for b_adv in ('2nd', 'superbee'):
        m = small_model('lid', b_adv=b_adv, les='tke'); step(m, nsteps)
        budget, shear, book, buoy = les_identities(m)
        tag = 'tke closure, %s' % b_adv
        ok &= line('2 TKE budget closes (tke_min 0), %s' % tag, budget, 1e-12)
        # du is 1e-3 of u, so u du from the increment carries 1e-13 relative round-off
        ok &= line('2 sum shear_prod = resolved KE removed, %s' % tag, shear, 1e-11)
        ok &= line('2 booked les_ke_loss = KE change, %s' % tag, book, 1e-11)
        if m.les_mix_b:
            ok &= line('2 dz sum buoy_prod = resolved PE gain, %s' % tag, buoy, 1e-12)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
