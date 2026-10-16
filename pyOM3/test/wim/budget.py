"""Test 14: the energy budget of a full time step from the model's own
booking, and the closure of the source step (wim family).

The source update is semi-implicit, dE = S dt/(1 - dt dS/dE), the linearised
implicit step: the sources act at E^{n+1} to first order, and the fluxes the
coupler builds after the step from gamma_in E, gamma_ds E, S_nl are the rates
at E^{n+1}. time_step books the applied increments (dE_src, its split dE_in,
dE_ds, dE_nl in proportion to the explicit rates, dE_tail for the f^-5
overwrite, A_clip for the action floor) so that the budget can be checked:

  split        dE_in + dE_ds + dE_nl = dE_src bin by bin            round-off
  homogeneous  sum (E_new - E_old) = sum (dE_src + dE_tail)         round-off
  full step    wim_simple with island, current, open coasts, and
               wim_realistic (sphere, janssen): sum omega (A_new - A_old) =
               sum omega dA_AB (propagation and outflow, the weights of the
               step) - sum omega A_clip + sum (dE_src + dE_tail)    round-off
  source step  dt S(E^{n+1}) - dE, the difference between the booked rates
               at the new spectrum and the applied increment, DECOMPOSED
               EXACTLY (round-off) into: the DIA's cubic nonlinearity
               S_nl(E^{n+1}) - S_nl(E^n) - lambda dE; the dependence of
               gamma_ds on E through alpha and om; the limiter, which cuts
               S but not dS/dE; the positive part of dS/dE dropped by
               implicit_dissipation_only; and the share in the diagnostic
               (overwritten) bins. Spun-up homogeneous sea (480 steps of
               300 s), u* 0.3 and 0.6, in units of the wind work. Measured
               Oct 12 2026: 1e-4 at u* 0.3; +0.16 at u* 0.6, of which the
               limiter +0.156 and +0.159 in the diagnostic bins (with the
               limiter off: +0.11, the DIA nonlinearity, half of it in the
               prognostic range). The sizes are printed, the decomposition
               is the PASS/FAIL.
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, homogeneous_model, step, line, weights
import pyOM3 as OM

rel = lambda a, b: float(np.max(np.abs(a)))/max(float(np.max(np.abs(b))), 1e-300)


def one_step(m):
    """one time_step with the weights of config.step, returning the budget pieces"""
    if not hasattr(m, '_tau3'):
        m._tau3, m._nstep = (2, 1, 0), 0
    tau, taum1, taum2 = m._tau3
    tA, tB, tC = (1., 0., 0.) if m._nstep == 0 else (1.55, -0.55, 0.)
    h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
    w = weights(m); om = np.asarray(m.omega_t)[sl][None, :, None, None]
    A0 = np.asarray(m.A)[I].copy()
    m.time_step(tA, tB, tC, tau, taum1, taum2)
    dA = np.asarray(m.dA)
    dA_prop = m.dt*(tA*dA[tau] + tB*dA[taum1] + tC*dA[taum2])[I]
    m._tau3 = ((tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3); m._nstep += 1
    A1 = np.asarray(m.A)[I]
    total = float(np.sum(w*om*(A1 - A0)))
    prop = float(np.sum(w*om*dA_prop))
    clip = -float(np.sum(w*om*np.asarray(m.A_clip)[I])) if hasattr(m, 'A_clip') else 0.0
    src = float(np.sum(w*np.asarray(m.dE_src)[I])); tail = float(np.sum(w*np.asarray(m.dE_tail)[I]))
    gross = float(np.sum(w*om*np.abs(A1 - A0))) + abs(prop) + abs(src) + abs(tail)
    split = rel((np.asarray(m.dE_in) + np.asarray(m.dE_ds) + np.asarray(m.dE_nl) - np.asarray(m.dE_src))[I],
                np.asarray(m.dE_src)[I])
    return dict(residual=abs(total - (prop + clip + src + tail))/gross, split=split)


def run(nsteps=6):
    ok = True
    cases = (('homogeneous', lambda: homogeneous_model(ustar=0.5, udir=0.3)),
             ('wim_simple, island + current + open coasts', lambda: small_model('simple')),
             ('wim_realistic, sphere', lambda: small_model('realistic')),
             ('wim_realistic, janssen', lambda: small_model('realistic', janssen=True)))
    from pyOM3.wim_simple.model import model as _wim
    if 'dE_src' not in _wim.time_step.__code__.co_names:
        print('14 energy budget: skipped (this tree does not book the applied source increments, dE_src, in wim_simple/model.py time_step)')
        return True
    for name, build in cases:
        m = build()
        res = split = 0.0
        for n in range(nsteps):
            r = one_step(m)
            res = max(res, r['residual']); split = max(split, r['split'])
        ok &= line('14 source split dE_in + dE_ds + dE_nl = dE_src, %s' % name, split, 1e-13)
        ok &= line('14 energy budget of the step closes, %s' % name, res, 1e-13)
    # SOURCE-STEP CLOSURE: the explicit rates at the new spectrum E^{n+1} (what
    # calc_stress, calc_phioc and wind_work are built from) against the
    # increment the step applied, dt S(E^{n+1}) - dE, decomposed EXACTLY into
    #   nl    the DIA is cubic: S_nl(E^{n+1}) - S_nl(E^n) - lambda_nl dE
    #   ds    gamma_ds depends on E through alpha and om: (gamma_ds(E^{n+1})
    #         - gamma_ds(E^n)) E^{n+1}
    #   lim   the limiter cuts S but not dSdE: S - min(S, limit)
    #   ido   implicit_dissipation_only drops the positive part of dSdE
    # and the share of the residual in the diagnostic (overwritten) bins.
    for ust in (0.3, 0.6):
        for ido in (True, False):
            m = homogeneous_model(ustar=ust, udir=0.0, amp=0.3, implicit_dissipation_only=ido)
            for n in range(480):
                one_step(m)
            h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
            w = np.asarray(m.dft)[sl][None, :, None, None]*m.dd
            E0 = np.asarray(m.E_pre_src); dE = np.asarray(m.dE_src); E1 = E0 + dE
            gin = np.asarray(m.gamma_in); gds0 = np.asarray(m.gamma_diss)[None]
            Snl0, fl = (np.asarray(x) for x in m.Snonlin(OM.np.asarray(E0)))     # the step's S_nl and its derivative
            Snl1 = np.asarray(m.Snonlin(OM.np.asarray(E1))[0])
            gds1 = np.asarray(m.Sdiss(OM.np.asarray(E1))[2])[None]
            S_unl = Snl0 + (gin + gds0)*E0
            ust_eff = np.maximum(np.asarray(m.ustar), 0.01)
            limit = 1e-7*m.grav*ust_eff[None]/np.asarray(m.ft)[:, None, None]**4*np.asarray(m.om_mean)[None]/(2*np.pi)
            S_lim = np.minimum(S_unl, limit[None]) if m.limit_growth else S_unl
            dSdE = fl + gin + gds0; dSdE_imp = np.minimum(dSdE, 0.0) if ido else dSdE
            S1 = gin*E1 + gds1*E1 + Snl1
            dt = m.dt
            R = dt*S1 - dE
            P_nl = dt*(Snl1 - Snl0 - fl*dE); P_ds = dt*(gds1 - gds0)*E1
            P_lim = dt*(S_unl - S_lim); P_ido = dt*(dSdE - dSdE_imp)*dE
            win = float(np.sum(w*gin[I]*E1[I]))*dt
            tot = lambda a: float(np.sum(w*a[I]))/win
            diag = np.broadcast_to(~np.asarray(m.freq_mask)[None, sl, sl, sl].astype(bool), R[I].shape)
            tag = 'u* %.1f, implicit_dissipation_only %s' % (ust, ido)
            ok &= line('14 source step: residual = nl + ds + lim + ido (exact), %s' % tag,
                       abs(tot(R) - tot(P_nl + P_ds + P_lim + P_ido)), 1e-12)
            print('%-56s %+8.4f  = nl %+.4f  ds %+.4f  lim %+.4f  ido %+.4f ; in the diagnostic bins %+.4f  (of the wind work; diagnostic)'
                  % ('14 source step: dt S(E_new) - dE, %s' % tag, tot(R), tot(P_nl), tot(P_ds), tot(P_lim), tot(P_ido),
                     float(np.sum((w*R[I])[diag]))/win))
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
