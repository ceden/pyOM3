"""Canonical four-wave interaction kernel T(k1,k2,k3,k4) for deep-water
surface gravity waves (Krasitskii 1994 convention).

Strategy
--------
Instead of transcribing Krasitskii's printed expressions (error-prone), the
kernel is *derived* here from first principles:

1.  Deep-water Hamiltonian through quartic order (Craig--Sulem expansion of
    the Dirichlet--Neumann operator, G0 = |D|):

        H2 = 1/2 int  g z^2 + psi G0 psi                    dx
        H3 = 1/2 int  z [ (grad psi)^2 - (G0 psi)^2 ]       dx
        H4 = 1/2 int  z (G0 psi) [ G0(z G0 psi) + z lap psi ] dx

2.  Normal variables (2D convention, H2 = int omega |a|^2 dk):

        zeta_hat(k) = lam f(k) (a_k + a*_{-k}),   f = sqrt(q/(2 omega))
        psi_hat(k)  = -i lam h(k) (a_k - a*_{-k}), h = sqrt(omega/(2 q))
        lam = 1/(2 pi),  omega = sqrt(g q),  q = |k|

3.  Equations of motion i da/dt = dH/da* are extracted by a mechanical
    monomial collector (no hand algebra).

4.  A canonical near-identity transformation a = b + Q[b] (Q quadratic)
    removes all triad terms (non-resonant for gravity waves).  The induced
    cubic terms in the b-equation are assembled by explicit composition of
    the quadratic EOM kernels with the generator kernels Q = K/Delta.
    The coefficient of b*_1 b_2 b_3 on the wavenumber shell
    k0 + k1 = k2 + k3 is the Zakharov/Krasitskii kernel T(k0,k1,k2,k3):

        i db_0/dt = int T_{0123} b*_1 b_2 b_3 delta(k0+k1-k2-k3) dk123

Validation (see test_kernel.py) against independently published closed
forms in the same convention:

  (a) T(k,k,k,k) = k^3 / (4 pi^2)                       [Stuhlmeier 2024, A.1]
  (b) T(ka,kb,ka,kb) = ka kb min(ka,kb) / (4 pi^2)      [collinear, A.2]
  (c) directional two-wave kernel T(ki,kj,ki,kj)        [A.5; Leblanc 2009 /
      with explicit sum/difference channel denominators  Stiassnie & Gramstad]
  (d) degenerate quartet & generic unidirectional forms [A.3, A.4; DZ gauge --
      compared on the wavenumber shell, informational]
plus symmetry, homogeneity (degree 3), rotation invariance and
g-independence on the resonance manifold.

Notes
-----
* All EOM kernels are real in this convention (each of the two psi legs in
  H3 and H4 carries a factor -i; (-i)^2 = -1 is folded into the vertex).
* Momentum bookkeeping: a monomial factor a^{sigma}(p) at a Hamiltonian leg
  carrying Fourier argument k means  sigma=+1: a_p with k = +p,
  sigma=-1: a*_p with k = -p.  EOM class constraint: k_out = sum sigma_j p_j.
* Triad frequency denominators never vanish for deep-water gravity waves
  (omega concave), but legs with |k| -> 0 make individual factors large
  while the assembled kernel stays finite; evaluate limits by small offsets.

Author: generated for the WIM exact-S_nl reference implementation.
"""

import numpy as np

TWO_PI = 2.0 * np.pi
LAM = 1.0 / TWO_PI  # normal-variable normalisation (2D)


# ----------------------------------------------------------------- basics
def _q(k):
    return float(np.hypot(k[0], k[1]))


def _omega(k, g):
    return np.sqrt(g * _q(k))


def _f_zeta(k, g):
    q = _q(k)
    return np.sqrt(q / (2.0 * _omega(k, g)))


def _h_psi(k, g):
    q = _q(k)
    return np.sqrt(_omega(k, g) / (2.0 * q))


# ------------------------------------------------- Hamiltonian definition
# Each Hamiltonian term: (prefactor, leg types, kernel over leg momenta).
# Leg types: 'z' (zeta) or 'p' (psi).  Kernel args follow leg order.
# The factor (-1) from (-i)^2 of the two psi legs is included in `pref`.

def _h3_kernel(ks):
    # 1/2 z [ (grad psi)^2 - (G0 psi)^2 ] ; legs: (z, p, p)
    k1, k2 = ks[1], ks[2]
    return -(k1[0] * k2[0] + k1[1] * k2[1] + _q(k1) * _q(k2))


def _h4_kernel_a(ks):
    # 1/2 z (G0 psi) G0( z (G0 psi) ) ; legs: (z_outer, z_inner, p_outer, p_inner)
    kb, k1, k2 = ks[1], ks[2], ks[3]
    kk = (kb[0] + k2[0], kb[1] + k2[1])
    return _q(k1) * _q(k2) * _q(kk)


def _h4_kernel_b(ks):
    # 1/2 z^2 (G0 psi) (lap psi) ; legs: (z, z, p_G0, p_lap)
    k1, k2 = ks[2], ks[3]
    return -_q(k1) * _q(k2) ** 2


_H3_TERMS = [(-0.5 * TWO_PI ** 2, ('z', 'p', 'p'), _h3_kernel)]
_H4_TERMS = [(-0.5 * TWO_PI ** 2, ('z', 'z', 'p', 'p'), _h4_kernel_a),
             (-0.5 * TWO_PI ** 2, ('z', 'z', 'p', 'p'), _h4_kernel_b)]
# pref = (1/2) * (2 pi)^2 [Fourier delta] * (-1) [two psi legs]


def _leg_factor(leg_type, sigma, k, g):
    """Normal-variable factor of one Hamiltonian leg (real convention)."""
    if leg_type == 'z':
        return LAM * _f_zeta(k, g)
    return LAM * _h_psi(k, g) * (1.0 if sigma > 0 else -1.0)


# ------------------------------------------------ EOM monomial collectors
def _perms(items):
    if len(items) == 1:
        yield [items[0]]
        return
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in _perms(rest):
            yield [items[i]] + p


def _eom_kernel(terms, slots, g):
    """Collected EOM kernel  i da_p/dt = ... for monomial with `slots`.

    slots: list of (p_vec, sigma) for the monomial factors a^{sigma}(p),
    in a fixed order; the caller supplies momenta satisfying the class
    constraint  p_out = sum sigma_j p_j  (p_out is implicit).
    Returns sum over all Hamiltonian legs / star patterns / bijections.
    """
    n = len(slots)
    # symmetry factor: identical-sigma slots are indistinguishable in the
    # ordered-integral convention; divide so the result is the per-ordered
    # (symmetrised) kernel.
    n_plus = sum(1 for _, sg in slots if sg > 0)
    n_minus = n - n_plus
    from math import factorial
    S = factorial(n_plus) * factorial(n_minus)
    total = 0.0
    for pref, legtypes, kern in terms:
        if len(legtypes) != n + 1:
            continue
        for ell in range(n + 1):          # leg hit by d/da*_p (starred)
            others = [j for j in range(n + 1) if j != ell]
            p_out = np.zeros(2)
            for (pv, sg) in slots:
                p_out += sg * np.asarray(pv, dtype=float)
            for assign in _perms(list(range(n))):   # slot -> role bijection
                ks = [None] * (n + 1)
                ks[ell] = (-p_out[0], -p_out[1])    # starred leg: k = -p_out
                fac = pref * _leg_factor(legtypes[ell], -1, ks[ell], g)
                for role, slot_idx in zip(others, assign):
                    pv, sg = slots[slot_idx]
                    kk = (sg * pv[0], sg * pv[1])
                    ks[role] = kk
                    fac *= _leg_factor(legtypes[role], sg, kk, g)
                total += fac * kern(ks)
    return total / S


def _delta(slots, g):
    """Frequency mismatch  sum sigma_j omega_j - omega(p_out)."""
    p_out = np.zeros(2)
    d = 0.0
    for (pv, sg) in slots:
        p_out += sg * np.asarray(pv, dtype=float)
        d += sg * _omega(pv, g)
    return d - _omega(p_out, g)


# The three quadratic EOM classes, as (sigma1, sigma2) patterns:
_Q_CLASSES = [(+1, +1), (-1, +1), (-1, -1)]


def _cubic_from_H4(p1, p2, p3, g):
    """Direct quartic contribution: coefficient of b*_1 b_2 b_3."""
    return _eom_kernel(_H4_TERMS, [(p1, -1), (p2, +1), (p3, +1)], g)


def _compose(host_slots, ins_at, ins_slots, g, host_from_Q, ins_sign):
    """Kernel value of one composed cubic term.

    host_slots : slots of the host quadratic kernel, one of which
                 (index ins_at) is the composite leg.
    ins_slots  : slots of the inserted quadratic kernel (its own class).
    host_from_Q: True for type-(iii) terms (host is the generator Q,
                 divided by its own Delta), False for type-(ii)
                 (host is N2, insertion is a generator Q leg).
    ins_sign   : +1/-1 bookkeeping sign determined by the caller.
    """
    # momentum of the composite leg = output momentum of inserted vertex
    P = np.zeros(2)
    for (pv, sg) in ins_slots:
        P += sg * np.asarray(pv, dtype=float)
    sigma_host = host_slots[ins_at][1]
    slots_h = list(host_slots)
    slots_h[ins_at] = ((P[0], P[1]), sigma_host)

    K_host = _eom_kernel(_H3_TERMS, slots_h, g)
    K_ins = _eom_kernel(_H3_TERMS, ins_slots, g)
    if host_from_Q:
        return ins_sign * (K_host / _delta(slots_h, g)) * K_ins
    return ins_sign * K_host * (K_ins / _delta(ins_slots, g))


def T(k1, k2, k3, k4, g=9.81):
    """Canonical Zakharov/Krasitskii kernel T(k1,k2,k3,k4), deep water.

    Arguments are 2-vectors (tuples/arrays) with k1 + k2 = k3 + k4 assumed
    (wavenumber closure).  On the resonance manifold
    omega1 + omega2 = omega3 + omega4 this is the unique canonical kernel
    entering the kinetic (Hasselmann) equation; off the frequency shell it
    is the minimal-generator normal form (Krasitskii's gauge).

    Homogeneous of degree 3, rotation invariant, g-independent on shell.
    Do not evaluate exactly at trivial quadruplets ({k1,k2}={k3,k4});
    use a small offset and take the (finite, regular) limit.
    """
    k1 = np.asarray(k1, float); k2 = np.asarray(k2, float)
    k3 = np.asarray(k3, float); k4 = np.asarray(k4, float)

    # target monomial in  i db_{k1}/dt :  b*_{k2} b_{k3} b_{k4}
    p_star, pa, pb = k2, k3, k4
    total = _cubic_from_H4(p_star, pa, pb, g)

    # ---- composed (virtual-channel) terms -------------------------------
    # Normal-form derivation: with a = b + Q[b] and Q chosen to remove all
    # triad terms, the transformed cubic coefficient is  N3 + DN2[b].Q[b]
    # (type ii only; the naive "-i dQ/dt with N2 inserted" (type iii) terms
    # vanish because the transformed quadratic dynamics is zero).  On the
    # resonance manifold each type-iii term equals its type-ii partner
    # (Delta_host = -Delta_ins on shell), so the manifestly symmetric
    # combination  (1/4)(ii + iii)  equals the correct  (1/2) ii  on shell
    # and extends it off shell in the (12)<->(34)-symmetric gauge.
    # host N2 class (s1,s2) at ordered slots; composite occupies one slot.
    for host_cls in _Q_CLASSES:
        for ins_at in (0, 1):
            s_comp = host_cls[ins_at]
            s_other = host_cls[1 - ins_at]
            for ins_cls in _Q_CLASSES:
                # monomial sigmas produced by the insertion:
                eff = [s * (1 if s_comp > 0 else -1) for s in ins_cls]
                sigmas = eff + [s_other]
                if sorted(sigmas) != [-1, 1, 1]:
                    continue
                # assign (p_star,-1),(pa,+1),(pb,+1) to the three produced
                # slots, respecting sigma pattern; enumerate bijections.
                targets = [(tuple(p_star), -1), (tuple(pa), +1),
                           (tuple(pb), +1)]
                for assign in _perms([0, 1, 2]):
                    ok = all(targets[assign[j]][1] == sigmas[j]
                             for j in range(3))
                    if not ok:
                        continue
                    # inserted slots in their *own* (unflipped) class:
                    ins_slots = []
                    for j in range(2):
                        pv, sg = targets[assign[j]]
                        ins_slots.append((pv, sg * (1 if s_comp > 0 else -1)))
                    host_other = targets[assign[2]]
                    host_slots = [None, None]
                    host_slots[1 - ins_at] = host_other
                    host_slots[ins_at] = ((0.0, 0.0), s_comp)  # placeholder
                    total += 0.25 * _compose(host_slots, ins_at, ins_slots, g,
                                      host_from_Q=False, ins_sign=1.0)

    # ---- type (iii): -i dQ/dt with nonlinear b-dot inserted -------------
    for q_cls in _Q_CLASSES:
        for ins_at in (0, 1):
            s_comp = q_cls[ins_at]
            s_other = q_cls[1 - ins_at]
            sign = -1.0 if s_comp > 0 else +1.0
            for ins_cls in _Q_CLASSES:
                eff = [s * (1 if s_comp > 0 else -1) for s in ins_cls]
                sigmas = eff + [s_other]
                if sorted(sigmas) != [-1, 1, 1]:
                    continue
                targets = [(tuple(p_star), -1), (tuple(pa), +1),
                           (tuple(pb), +1)]
                for assign in _perms([0, 1, 2]):
                    ok = all(targets[assign[j]][1] == sigmas[j]
                             for j in range(3))
                    if not ok:
                        continue
                    ins_slots = []
                    for j in range(2):
                        pv, sg = targets[assign[j]]
                        ins_slots.append((pv, sg * (1 if s_comp > 0 else -1)))
                    host_other = targets[assign[2]]
                    host_slots = [None, None]
                    host_slots[1 - ins_at] = host_other
                    host_slots[ins_at] = ((0.0, 0.0), s_comp)
                    total += 0.25 * _compose(host_slots, ins_at, ins_slots, g,
                                      host_from_Q=True, ins_sign=sign)

    return total


# --------------------------------------------- published reference values
def T_selfself(k, g=9.81):
    """T(k,k,k,k) = k^3 / (4 pi^2)  [Stuhlmeier 2024 Eq. (A.1)]."""
    return _q(np.asarray(k, float)) ** 3 / (4.0 * np.pi ** 2)


def T_twowave_directional(ki, kj, g=9.81):
    """Directional two-wave kernel T(ki,kj,ki,kj), deep water.

    Stuhlmeier 2024 Eq. (A.5) (= Leblanc 2009; Stiassnie & Gramstad 2009):

    T = -1/(16 pi^2 sqrt(qi qj)) * [ 3 (qi qj)^2
        + (ki.kj) ( ki.kj - 4 (qi+qj) sqrt(qi qj) )
        + 2 (wi-wj)^2 (ki.kj + qi qj)^2 / ( g|ki-kj| - (wi-wj)^2 )
        + 2 (wi+wj)^2 (ki.kj - qi qj)^2 / ( g|ki+kj| - (wi+wj)^2 ) ]
    """
    ki = np.asarray(ki, float); kj = np.asarray(kj, float)
    qi, qj = _q(ki), _q(kj)
    wi, wj = _omega(ki, g), _omega(kj, g)
    dot = float(ki @ kj)
    dm = g * _q(ki - kj) - (wi - wj) ** 2
    dp = g * _q(ki + kj) - (wi + wj) ** 2
    term_m = 0.0 if abs(dm) < 1e-300 else \
        2.0 * (wi - wj) ** 2 * (dot + qi * qj) ** 2 / dm
    term_p = 2.0 * (wi + wj) ** 2 * (dot - qi * qj) ** 2 / dp
    br = 3.0 * (qi * qj) ** 2 + dot * (dot - 4.0 * (qi + qj)
                                       * np.sqrt(qi * qj)) + term_m + term_p
    return -br / (16.0 * np.pi ** 2 * np.sqrt(qi * qj))


def T_unidirectional(ka, kb, kc, kd):
    """Generic unidirectional formula, scalar k>0 [Stuhlmeier A.3; DZ gauge]."""
    s = (ka * kb * kc * kd) ** 0.25
    br = np.sqrt(ka * kb) + np.sqrt(kc * kd)
    diffs = abs(ka - kc) + abs(ka - kd) + abs(kb - kc) + abs(kb - kd)
    return s * br * (ka + kb + kc + kd - diffs) / (32.0 * np.pi ** 2)


def xc_webb(K1, K2, K3, K4, grav=9.81):
    """WW3/xnl deep-water coupling coefficient (Webb 1978 / Dungey & Hui 1979).

    Direct port of xc_webb from van Vledder's mod_xnl4v5.f90 (release 5.0,
    WAVEWATCH III model/src).  Returns  g^2 (pi/4) D^2 / (w1 w2 w3 w4)  with
    w_i = sqrt(k_i), i.e. the full WRT kinetic-integrand coupling factor.

    On the resonance manifold (k1+k2 = k3+k4, w1+w2 = w3+w4):

        xc_webb = 64 pi^5 g^2 T^2  =  4 pi g^2 [ (2 pi)^2 T ]^2

    where T is the canonical kernel of this module ((2 pi)^2 T is the kernel
    in Krasitskii's 2pi-free normalisation and 4 pi is the Zakharov
    collision-integral prefactor).  Verified to machine precision in
    test_kernel.py; this fixes the convention constant for the WRT assembly.
    """
    import numpy as _np
    K1, K2, K3, K4 = (_np.asarray(v, float) for v in (K1, K2, K3, K4))
    k1, k2, k3, k4 = (_np.hypot(*v) for v in (K1, K2, K3, K4))
    w1, w2, w3, w4 = _np.sqrt(k1), _np.sqrt(k2), _np.sqrt(k3), _np.sqrt(k4)
    d12 = K1 @ K2; d13 = K1 @ K3; d14 = K1 @ K4
    d23 = K2 @ K3; d24 = K2 @ K4; d34 = K3 @ K4
    z12 = _np.hypot(*(K1 + K2)) - (w1 + w2) ** 2
    z13 = _np.hypot(*(K1 - K3)) - (w1 - w3) ** 2
    z14 = _np.hypot(*(K1 - K4)) - (w1 - w4) ** 2
    p1 = 2 * (w1 + w2) ** 2 * (k1 * k2 - d12) * (k3 * k4 - d34) / z12
    p2 = (2 * (w1 - w3) ** 2 * (k1 * k3 + d13) * (k2 * k4 + d24) / z13
          if abs(z13) > 1e-300 else 0.0)
    p3 = (2 * (w1 - w4) ** 2 * (k1 * k4 + d14) * (k2 * k3 + d23) / z14
          if abs(z14) > 1e-300 else 0.0)
    p4 = 0.5 * (d12 * d34 + d13 * d24 + d14 * d23)
    p5 = 0.25 * (d13 + d24) * (w1 - w3) ** 4
    p6 = -0.25 * (d12 + d34) * (w1 + w2) ** 4
    p7 = 0.25 * (d14 + d23) * (w1 - w4) ** 4
    p8 = 2.5 * k1 * k2 * k3 * k4
    p9 = (w1 + w2) ** 2 * (w1 - w3) ** 2 * (w1 - w4) ** 2 * (k1 + k2 + k3 + k4)
    D = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    return grav ** 2 * (_np.pi / 4) * D * D / (w1 * w2 * w3 * w4)
