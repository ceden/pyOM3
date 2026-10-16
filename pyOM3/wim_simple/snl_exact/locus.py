"""Resonance-locus tracer for the deep-water WRT computation.

For fixed k1, k3 the four-wave resonance
    k1 + k2 = k3 + k4,   w1 + w2 = w3 + w4,   w = sqrt(g|k|)
defines a closed curve (locus) in the k2-plane:

    z(k2) = q + w(k2) - w(k2 + P) = 0,   P = k1 - k3,  q = w1 - w3,

with k4 = k2 + P.  The trivial quadruplet k2 = k3 (k4 = k1) lies on every
locus (z(k3) = 0 identically).

Deep-water scale invariance: with lengths scaled by |P| and frequencies by
sqrt(g |P|), the locus depends on the single parameter

    qhat = q / sqrt(g |P|)  in (0, 1)

(qhat >= 1: no solutions; qhat -> 0: locus degenerates to the
perpendicular-bisector line, radius ~ 1/(4 qhat^2)).  WLOG qhat > 0 by the
(1,3) relabelling; the dimensional wrapper handles q < 0 by solving the
swapped problem, whose k4-points are the original k2-points.

Axis crossings (normalized frame, P = ex) are analytic:
    x_plus  = ((1 - qhat^2) / (2 qhat))^2          (k2 parallel to P)
    x_minus = -((sqrt(2 - qhat^2) - qhat)/2)^2     (k2 anti-parallel)

Tracing: radial continuation about the interior centre
c = ((x_plus + x_minus)/2, 0) on a dense theta grid (bracketed bisection
with continuation from the previous ray), mirror symmetry in y.  The
arc-length element is computed *analytically* from the gradient
(r'(theta) = -dz/dtheta / dz/dr on the locus), then the curve is resampled
to equal arc length; periodic-trapezoid weights are spectrally accurate.

Returned per point: normalized k2, k4, Jacobian 1/|grad z| and quadrature
weight w = ds/|grad z| — the line measure of the frequency delta function,

    int f(k2) delta(z(k2)) d^2 k2  =  sum_i w_i f(k2_i).

Dimensional scaling: k2 = R(phi) |P| k2hat with phi the direction of P;
z = sqrt(g|P|) zhat; |grad z| = sqrt(g/|P|) |grad zhat|; ds = |P| dshat;
=> w = ds/|grad z| = |P|^{3/2} g^{-1/2} what.

Validation in test_locus.py: residuals, analytic crossings, resonance
closure of dimensional quadruplets (both signs of q), delta-function
quadrature against a brute-force 2D integral, the |P|^{3/2} scaling law,
spectral convergence in point count, and presence of the trivial point.
"""

import numpy as np


# --------------------------------------------------------- normalized frame
def zhat(k2, qhat):
    """Normalized locus function (g = 1, P = ex)."""
    k2 = np.asarray(k2, float)
    w2 = np.hypot(k2[..., 0], k2[..., 1]) ** 0.5
    w4 = np.hypot(k2[..., 0] + 1.0, k2[..., 1]) ** 0.5
    return qhat + w2 - w4


def grad_zhat(k2):
    """Analytic gradient of zhat (independent of qhat)."""
    k2 = np.asarray(k2, float)
    q2 = np.hypot(k2[..., 0], k2[..., 1])
    k4 = k2.copy()
    k4[..., 0] += 1.0
    q4 = np.hypot(k4[..., 0], k4[..., 1])
    g2 = 0.5 * q2[..., None] ** -1.5 * k2   # grad w = khat / (2 sqrt(k))
    g4 = 0.5 * q4[..., None] ** -1.5 * k4
    return g2 - g4


def axis_crossings(qhat):
    """Analytic intersections of the locus with the P-axis."""
    if not 0.0 < qhat < 1.0:
        raise ValueError("locus exists for 0 < qhat < 1")
    x_plus = ((1.0 - qhat ** 2) / (2.0 * qhat)) ** 2
    x_minus = -((np.sqrt(2.0 - qhat ** 2) - qhat) / 2.0) ** 2
    return x_plus, x_minus


def _radial_root(c, e, qhat, r_lo, r_hi, tol=1e-14, maxit=200):
    """Bracketed bisection for zhat(c + r e) = 0 on [r_lo, r_hi]."""
    f = lambda r: zhat(c + r * e, qhat)
    flo, fhi = f(r_lo), f(r_hi)
    if flo * fhi > 0:
        return None
    for _ in range(maxit):
        mid = 0.5 * (r_lo + r_hi)
        fm = f(mid)
        if flo * fm <= 0:
            r_hi, fhi = mid, fm
        else:
            r_lo, flo = mid, fm
        if r_hi - r_lo < tol * max(1.0, r_hi):
            break
    return 0.5 * (r_lo + r_hi)


def trace_locus(qhat, npoints=64, ntheta_internal=None):
    """Trace the normalized locus and return equal-arclength quadrature.

    Parameters
    ----------
    qhat : float in (0,1)
    npoints : number of returned quadrature points (equal arc length)
    ntheta_internal : dense tracing resolution (default max(256, 4*npoints))

    Returns dict with:
      k2 (n,2), k4 (n,2)      normalized wavenumbers on the locus
      jac (n,)                1/|grad zhat|
      w (n,)                  quadrature weight  dshat/|grad zhat|
      ds (n,)                 arc-length weight
      s (n,), length          arc-length positions and total length
      x_plus, x_minus, center, qhat, residual (max |zhat| at points)
    """
    if ntheta_internal is None:
        ntheta_internal = max(256, 4 * npoints)
    x_plus, x_minus = axis_crossings(qhat)
    cx = 0.5 * (x_plus + x_minus)
    c = np.array([cx, 0.0])
    a = x_plus - cx                      # semi-axis along P

    # dense trace, upper half (theta in [0, pi]), continuation in r
    nth = ntheta_internal // 2 + 1
    thetas = np.linspace(0.0, np.pi, nth)
    rs = np.empty(nth)
    r_prev = a
    for i, th in enumerate(thetas):
        e = np.array([np.cos(th), np.sin(th)])
        lo, hi = 0.25 * r_prev, 4.0 * r_prev
        root = _radial_root(c, e, qhat, lo, hi)
        while root is None and hi < 1e12:
            lo *= 0.25
            hi *= 4.0
            root = _radial_root(c, e, qhat, lo, hi)
        if root is None:
            raise RuntimeError(f"locus bracket failure at qhat={qhat}, "
                               f"theta={th}")
        rs[i] = root
        r_prev = root

    # analytic ds/dtheta from the gradient:  on the locus,
    # dz/dr * r' + dz/dtheta = 0,  dz/dr = grad.e,  dz/dtheta = grad.(r e_perp)
    def _geom(thetas, rs):
        e = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
        ep = np.stack([-np.sin(thetas), np.cos(thetas)], axis=-1)
        pts = c + rs[:, None] * e
        gr = grad_zhat(pts)
        dzdr = np.einsum('ij,ij->i', gr, e)
        dzdth = rs * np.einsum('ij,ij->i', gr, ep)
        rprime = -dzdth / dzdr
        dsdth = np.sqrt(rs ** 2 + rprime ** 2)
        gnorm = np.hypot(gr[:, 0], gr[:, 1])
        return pts, dsdth, gnorm

    pts_u, dsdth_u, _ = _geom(thetas, rs)

    # cumulative arc length on the upper half (fine trapezoid)
    s_u = np.concatenate([[0.0], np.cumsum(0.5 * (dsdth_u[1:] + dsdth_u[:-1])
                                           * np.diff(thetas))])
    half_len = s_u[-1]
    length = 2.0 * half_len

    # resample the FULL closed curve at equal arc length.
    # full curve: theta(s) on [0, half_len] upper, mirrored lower.
    s_targets = (np.arange(npoints) + 0.5) / npoints * length
    th_of_s = lambda st: np.interp(st, s_u, thetas)
    th_pts = np.empty(npoints)
    sign = np.empty(npoints)
    for j, st in enumerate(s_targets):
        if st <= half_len:
            th_pts[j] = th_of_s(st)
            sign[j] = 1.0
        else:
            th_pts[j] = th_of_s(length - st)
            sign[j] = -1.0

    # refine r at resampled thetas by continuation from interpolated r
    r_pts = np.interp(th_pts, thetas, rs)
    for j in range(npoints):
        e = np.array([np.cos(th_pts[j]), np.sin(th_pts[j])])
        root = _radial_root(c, e, qhat, 0.5 * r_pts[j], 2.0 * r_pts[j])
        if root is not None:
            r_pts[j] = root
    pts, _, gnorm = _geom(th_pts, r_pts)
    pts[:, 1] *= sign                     # mirror lower half

    k4 = pts.copy()
    k4[:, 0] += 1.0
    ds = np.full(npoints, length / npoints)   # periodic equal-s trapezoid
    jac = 1.0 / gnorm
    res = float(np.max(np.abs(zhat(pts, qhat))))
    return dict(k2=pts, k4=k4, jac=jac, w=ds * jac, ds=ds,
                s=s_targets, length=length, x_plus=x_plus, x_minus=x_minus,
                center=c, qhat=qhat, residual=res)


# ------------------------------------------------------- dimensional wrapper
def locus_dimensional(k1, k3, g=9.81, npoints=64, **kw):
    """Dimensional locus for given k1, k3 (2-vectors) and gravity g.

    Returns dict with k2 (n,2), k4 (n,2), w (n,) = ds/|grad z| in
    dimensional units, jac, ds, qhat, swapped, and the normalized trace.
    Handles q = w1 - w3 of either sign by the (1,3) swap (the swapped
    problem's k4-points are the original k2-points).
    """
    k1 = np.asarray(k1, float)
    k3 = np.asarray(k3, float)
    w = lambda v: np.sqrt(g * np.hypot(v[0], v[1]))
    q = w(k1) - w(k3)
    if q == 0.0:
        raise ValueError("q = 0: degenerate (bisector-line) locus")
    swapped = q < 0.0
    a1, a3 = (k3, k1) if swapped else (k1, k3)
    P = a1 - a3
    Pm = np.hypot(P[0], P[1])
    qhat = (w(a1) - w(a3)) / np.sqrt(g * Pm)
    tr = trace_locus(qhat, npoints=npoints, **kw)

    phi = np.arctan2(P[1], P[0])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    k2n = (R @ (Pm * tr['k2']).T).T
    k4n = (R @ (Pm * tr['k4']).T).T
    if swapped:
        k2n, k4n = k4n, k2n              # swapped k4 = original k2
    scale_w = Pm ** 1.5 / np.sqrt(g)
    return dict(k2=k2n, k4=k4n, w=scale_w * tr['w'],
                jac=np.sqrt(Pm / g) * tr['jac'],
                ds=Pm * tr['ds'], qhat=qhat, swapped=swapped,
                normalized=tr, P=P, q=q)


# ------------------------------------------------- degenerate q = 0 locus
def trace_locus_degenerate(t_max, npoints=64):
    """Straight-line locus for q = w1 - w3 = 0 (|k1| = |k3|, k1 != k3).

    In the normalized frame (P = ex, g = 1) the locus is the perpendicular
    bisector  {k2 : |k2| = |k2 + P|} = {(-1/2, t)}.  The line is unbounded;
    t is sampled uniformly on [-t_max, t_max] (choose t_max from the grid
    support; out-of-grid points are masked at assembly).  Weights are
    trapezoidal:  w = dt / |grad zhat|.
    """
    t = (np.arange(npoints) + 0.5) / npoints * 2.0 * t_max - t_max
    pts = np.stack([np.full(npoints, -0.5), t], axis=-1)
    k4 = pts.copy()
    k4[:, 0] += 1.0
    gr = grad_zhat(pts)
    gnorm = np.hypot(gr[:, 0], gr[:, 1])
    ds = np.full(npoints, 2.0 * t_max / npoints)
    jac = 1.0 / gnorm
    return dict(k2=pts, k4=k4, jac=jac, w=ds * jac, ds=ds, qhat=0.0,
                t=t, residual=float(np.max(np.abs(zhat(pts, 0.0)))))
