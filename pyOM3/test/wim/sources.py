"""Test 13: the source terms of wim_simple (WAMDI input and dissipation,
the DIA, the tail machinery and the source time stepping). The source
terms are physics, not discrete identities, so the strategy differs from
test 2: each routine is checked against its published formula with the
spectral means recomputed by direct quadrature, the DIA against the
conservation laws it is built to satisfy, and the source integration
against its analytic solution.

  Sin          Sinput_wamdi = max(0, 0.25 rho_a/rho_w (28 u* cos(theta -
               phi)/c - 1)) omega, c = g/omega  (WAMDI 1988)      exact
  Sds          Sdiss_wamdi = -2.33e-5 om (omega/om)^2 (alpha/alpha_PM)^2,
               om = E/int E/omega, alpha = E om^4/g^2, alpha_PM =
               0.66 x 4.57e-3, with E, om from a direct quadrature   exact
  cut-off      frcutindex: f <= max(2.5 f_mean, 3 f_PM), f_PM = g/(28 2pi
               u*), the lowest bin always prognostic               exact
  Phi_oc       calc_phioc = -g int S_diss df dtheta (tail_flux wamdi) exact
  DIA          on a spectrum confined to the interior of the frequency
               grid, Snonlin conserves action and energy          round-off
               and momentum to 1e-2 (the quadruplet is interpolated onto
               the directional grid, which shifts its k vectors: the known
               property of the WAM DIA)                           measured
               On the JONSWAP sea the interactions reaching the cut-off are
               dropped; the net loss of the resolved range is printed, and
               the cascade credit the interface books equals it     exact
  integration  wind input alone, limiter and tail off: the explicit growth
               E_n = E_0 (1 + gamma dt)^n and the implicit one
               E_0/(1 - gamma dt)^n bracket exp(gamma t) and both are
               first order: the error halves with dt (ratio 2 +- 10 %)
  limiter      fraction of prognostic bins where limit_growth cuts S and
               the lowest frequency cut, for a PM sea under moderate and
               strong wind and for a near-empty sea (diagnostic: the cap
               is active in the upper prognostic range of any developing
               sea, as in WAM; on an empty sea the absolute growth is far
               below it)
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import homogeneous_model, step, line, NF, ND
import pyOM3 as OM

rel = lambda a, b: float(np.max(np.abs(a)))/max(float(np.max(np.abs(b))), 1e-300)


def quadrature(m, E):
    """E, mean omega (WAM definition) and alpha from the interior bins"""
    h = m.halo; sl = slice(h, -h)
    dft = np.asarray(m.dft)[sl]; om = np.asarray(m.omega_t)[sl]
    Ed = np.sum(E[sl, sl], axis=0)*m.dd                      # (nf, Ny, Nx)
    Et = np.sum(Ed*dft[:, None, None], axis=0)
    Eo = np.sum(Ed/om[:, None, None]*dft[:, None, None], axis=0)
    omm = Et/np.maximum(Eo, 1e-32)
    return Et, omm


def formula_tests(m):
    h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
    E = np.asarray(m.E); ust = np.asarray(m.ustar); ud = np.asarray(m.udir)
    th = np.asarray(m.theta); om = np.asarray(m.omega_t); ft = np.asarray(m.ft); g = m.grav
    # Sin
    gin = np.asarray(m.Sinput(m.ustar, m.udir))
    c = g/om
    ref = np.maximum(0.0, 0.25*(1.225/1000.)*(28*ust[None, None]*np.cos(th[:, None, None, None] - ud[None, None])
                                               / c[None, :, None, None] - 1.0))*om[None, :, None, None]
    d_in = rel((gin - ref)[I], ref[I])
    # Sds
    Et, omm = quadrature(m, E)
    omm = np.clip(omm, om[h], om[-h-1])
    alpha = Et*omm**4/g**2
    ref = -2.33e-5*omm[None]*(om[sl][:, None, None]/omm[None])**2*(alpha[None]/(0.66*4.57e-3))**2
    e_mean, om_mean, gds = (np.asarray(x) for x in m.Sdiss(m.E))
    d_ds = max(rel(gds[sl] - ref, ref), rel(e_mean - Et, Et), rel(om_mean - omm, omm))
    # cut-off
    f_pm = g/(28*2*np.pi*ust)
    fmax = np.maximum(2.5*om_mean/(2*np.pi), 3*f_pm)
    ref = ft[:, None, None] <= fmax[None]
    ref[h] = True
    mask = np.asarray(m.frcutindex(om_mean/(2*np.pi), m.ustar))
    d_cut = float(np.sum(mask[sl] != ref[sl]))
    # Phi_oc
    m.gamma_diss = gds; m.gamma_in = gin
    phi = np.asarray(m.calc_phioc())
    dft = np.asarray(m.dft)[sl]
    ref = -g*np.sum(np.sum((gds[None]*E)[I]*m.dd, axis=0)*dft[:, None, None], axis=0)
    d_phi = rel(phi - ref, ref)
    return d_in, d_ds, d_cut, d_phi


def dia_conservation(m):
    h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
    S, _ = m.Snonlin(m.E); S = np.asarray(S)[I]
    om = np.asarray(m.omega_t)[sl][None, :, None, None]; dft = np.asarray(m.dft)[sl][None, :, None, None]
    th = np.asarray(m.theta)[sl][:, None, None, None]
    w = m.dd*dft; k = om**2/m.grav
    gross = np.sum(np.abs(S)*w, axis=(0, 1))
    act = np.abs(np.sum(S/om*w, axis=(0, 1)))/np.sum(np.abs(S)/om*w, axis=(0, 1))
    ene = np.abs(np.sum(S*w, axis=(0, 1)))/gross
    mom = np.hypot(np.sum(S/om*k*np.sin(th)*w, axis=(0, 1)), np.sum(S/om*k*np.cos(th)*w, axis=(0, 1))) \
        / np.sum(np.abs(S)/om*k*w, axis=(0, 1))
    return float(act.max()), float(ene.max()), float(mom.max())


def confined_sea(fp=0.14, wid=0.015):
    """a narrow swell in the middle of the frequency grid, 1e-13 of the peak
    at the lowest bin and nothing at the highest"""
    m = homogeneous_model(ustar=0.0, udir=0.0, amp=0.0)
    h = m.halo; sl = slice(h, -h)
    ft, th = np.asarray(m.ft)[sl], np.asarray(m.theta)[sl]
    A = np.zeros((m.Nd, m.Nf, m.Ny, m.Nx))
    A[sl, sl] = (0.5*np.exp(-((th - 1.0)/0.4)**2))[:, None, None, None] \
                * (np.exp(-((ft - fp)/wid)**2)/(2*np.pi*ft))[None, :, None, None]
    m.A = m.apply_bc(OM.np.asarray(A)); m.E = m.A*m.omega_t[None, :, None, None]
    return m


def growth_order(implicit):
    """wind input alone: E_n against exp(gamma t), two step sizes"""
    errs = []
    for fac in (1, 2):
        m = homogeneous_model(ustar=0.4, udir=0.0, limit_growth=False, mask_energy=False,
                              implicit_dissipation_only=not implicit)
        m.dt = m.dt/fac
        gin = np.asarray(m.Sinput(m.ustar, m.udir))
        m.Sdiss = lambda E, _m=m: _m.calc_emean_wamdi(E) + (0.0*np.asarray(E)[:, :, 0],)  # no dissipation
        m.Sdiss = lambda E, _m=m: (_m.calc_emean_wamdi(E)[0], _m.calc_emean_wamdi(E)[1], OM.np.zeros_like(E[0]))
        m.Snonlin = lambda E: (OM.np.zeros_like(E), OM.np.zeros_like(E))
        E0 = np.asarray(m.E).copy(); n = 6*fac
        step(m, n)
        h = m.halo; I = (slice(h, -h),)*4
        exact = E0*np.exp(gin*n*m.dt)
        sel = (E0[I] > 1e-3*E0.max()) & (gin[I] > 0)
        err = (np.asarray(m.E)[I] - exact[I])[sel]/exact[I][sel]
        errs.append(err)
    e1, e2 = errs
    sign = np.sign(np.median(e1))
    return abs(float(np.median(e1))/float(np.median(e2))), sign


def limiter_fraction(m):
    """fraction of prognostic interior bins where limit_growth cuts S in one
    step, and the lowest frequency where it does"""
    h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
    E0 = np.asarray(m.E).copy()
    step(m, 1)
    S = np.asarray(m.S_nl) + (np.asarray(m.gamma_in) + np.asarray(m.gamma_diss)[None])*E0
    ust = np.maximum(np.asarray(m.ustar), 0.01)
    limit = 1e-7*m.grav*ust[None]/np.asarray(m.ft)[:, None, None]**4*np.asarray(m.om_mean)[None]/(2*np.pi)
    cut = (S > limit[None])[I] & np.asarray(m.freq_mask)[None, sl, sl, sl].astype(bool)
    fmin = np.asarray(m.ft)[sl][np.where(cut)[1]].min() if cut.any() else np.inf
    return float(np.mean(cut)), float(fmin)


def run():
    ok = True
    m = homogeneous_model(ustar=0.5, udir=0.3); step(m, 3)
    d_in, d_ds, d_cut, d_phi = formula_tests(m)
    ok &= line('13 Sinput_wamdi = WAMDI formula', d_in, 1e-14)
    ok &= line('13 Sdiss_wamdi = WAMDI formula, means by quadrature', d_ds, 1e-13)
    ok &= line('13 frcutindex = max(2.5 f_mean, 3 f_PM), bins differing', d_cut, 0.0)
    ok &= line('13 calc_phioc = -g int S_diss (tail_flux wamdi)', d_phi, 1e-14)
    # DIA: a spectrum confined to the interior of the frequency grid (no
    # interaction reaches the cut-off): action and energy exact, momentum
    # measured; the full JONSWAP sea: what leaves the resolved range is the
    # cascade credit the interface books (definition), its size reported
    m = confined_sea()
    act, ene, mom = dia_conservation(m)
    ok &= line('13 DIA conserves action, confined spectrum (|net| / gross)', act, 1e-12)
    ok &= line('13 DIA conserves energy, confined spectrum', ene, 1e-12)
    ok &= line('13 DIA momentum, confined spectrum (quadruplet interpolation)', mom, 3e-2)
    m = homogeneous_model(ustar=0.5, udir=0.0); step(m, 2)
    act, ene, mom = dia_conservation(m)
    print('%-56s %10.2e %9.2e %9.2e  (action, energy, momentum: the flux to the tail, diagnostic)'
          % ('13 DIA net / gross on the JONSWAP sea', act, ene, mom))
    h = m.halo; sl = slice(h, -h); I = (sl, sl, sl, sl)
    S = np.asarray(m.S_nl); dft = np.asarray(m.dft)[sl]
    if hasattr(m, 'freq_mask'):
        S = np.where(np.asarray(m.freq_mask)[None], S, 0.0)
    ce_ref = -m.grav*np.sum(np.sum(S[I]*m.dd, axis=0)*dft[:, None, None], axis=0)
    cx, cy, ce = (np.asarray(m.unpad(x)) if OM.jax else np.asarray(x) for x in m._cascade_credit())
    ok &= line('13 cascade credit = -g int S_nl over the prognostic bins', rel(ce - ce_ref, ce_ref), 1e-14)
    for implicit in (False, True):
        ratio, sign = growth_order(implicit)
        tag = 'implicit' if implicit else 'explicit (implicit_dissipation_only)'
        ok &= line('13 wind-input step first order, %s: error ratio dt/(dt/2)' % tag, abs(ratio - 2.0), 0.2)
        ok &= line('13 wind-input step %s exp(gamma t), %s' % ('above' if implicit else 'below', tag),
                   0.0 if sign == (1 if implicit else -1) else 1.0, 0.0)
    for tag, kw in (('u* 0.3, PM sea', dict(ustar=0.3, amp=1.0)), ('u* 1.5, PM sea', dict(ustar=1.5, amp=1.0)),
                    ('u* 1.5, near-empty sea', dict(ustar=1.5, amp=1e-6))):
        frac, fmin = limiter_fraction(homogeneous_model(udir=0.0, **kw))
        print('%-56s %10.3f  (fraction of prognostic bins cut, lowest cut f %.2f Hz; diagnostic)'
              % ('13 limit_growth, %s' % tag, frac, fmin))
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
