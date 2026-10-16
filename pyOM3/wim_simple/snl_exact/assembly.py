"""WRT assembly of the exact nonlinear transfer S_nl on an (f,theta) grid.

Kinetic equation (action density n(k), 2D wavenumber space):

    dn1/dt = 4 pi int |T_nat|^2 delta^2(k1+k2-k3-k4) delta(Dw)
             [ n3 n4 (n1+n2) - n1 n2 (n3+n4) ] dk2 dk3 dk4

with T_nat = (2 pi)^2 T the kernel in Krasitskii's 2pi-free normalisation.
Eliminating the deltas the WRT way (k4 = k2 + P, P = k1 - k3; locus in the
k2-plane):

    dn1/dt = int dk3  sum_s  w_s C_s f_s ,
    w_s = ds/|grad z|  (locus quadrature, locus.py),
    C_s = 4 pi T_nat^2 = xc_webb / g^2   (verified identity, kernel.py),
    f_s = n3 n4 (n1+n2) - n1 n2 (n3+n4).

Discretisation
--------------
Grid: geometric frequencies (uniform in ln k), uniform periodic directions.
The table is built for theta1 = 0 over (i1f, i3f, i3d) and applied to every
theta1 by integer direction shifts (exact on the uniform grid).  Roles 1,3
are on-node; roles 2,4 are gathered/scattered with bilinear stencils in
(ln k, theta).

Conservative evaluation: each table entry deposits the quadruplet action
rate R = (1/4) A1 A3 w C f with signs (+,+,-,-) onto the four roles using
the same stencil weights for gather and scatter.  The 1/4: summed over the
full (k1,k3) enumeration, each role's aggregate reproduces the same
continuum integral, so the all-role scatter is the average of four
consistent estimators — and conserves total action per quadruplet exactly.
Deposits and gathers falling outside the grid are truncated; truncated
deposits are accumulated in `leak` so that
    sum_bins A dn/dt + leak = 0   to machine precision.
A direct (role-1 only, non-conservative) mode provides an independent
discretisation of the same integral for cross-validation.

Scaling identity for validation: n(k) = lam phi(k/rho) implies
S_nl -> lam^3 rho^{19/2} (deep water); on the geometric grid a shift by an
integer number of k-cells commutes exactly with the discretisation in the
interior, so this is testable to near machine precision.
"""

import numpy as np
from pyOM3.wim_simple.snl_exact.locus import axis_crossings, zhat, grad_zhat


# ----------------------------------------------------------------- grid
class PolarGrid:
    """Geometric frequency grid (uniform ln k), uniform directions."""

    def __init__(self, f0=0.08, ratio=1.10, nf=36, nd=24, g=9.81):
        self.g = g
        self.f = f0 * ratio ** np.arange(nf)
        self.k = (2 * np.pi * self.f) ** 2 / g
        self.nf, self.nd = nf, nd
        self.th = 2 * np.pi * np.arange(nd) / nd
        self.dth = 2 * np.pi / nd
        self.lnk = np.log(self.k)
        self.dlnk = self.lnk[1] - self.lnk[0]
        # bin widths in k (log-centred)
        r = ratio ** 2
        self.dk = self.k * (np.sqrt(r) - 1.0 / np.sqrt(r))
        self.area = self.k * self.dk * self.dth      # (nf,) per direction

    def stencil(self, kx, ky):
        """Bilinear stencil in (ln k, theta) for points (kx,ky).

        Returns (ik, fk, id_, fd, inside): lower ln-k index and fraction,
        lower theta index and fraction, and validity mask (k in range).
        """
        km = np.hypot(kx, ky)
        th = np.mod(np.arctan2(ky, kx), 2 * np.pi)
        x = (np.log(np.maximum(km, 1e-300)) - self.lnk[0]) / self.dlnk
        ik = np.floor(x).astype(np.int64)
        fk = x - ik
        inside = (x >= 0.0) & (x <= self.nf - 1.0)
        ik = np.clip(ik, 0, self.nf - 2)
        fk = np.clip(x - ik, 0.0, 1.0)
        y = th / self.dth
        id_ = np.floor(y).astype(np.int64) % self.nd
        fd = y - np.floor(y)
        return ik, fk, id_, fd, inside


# --------------------------------------------------- fast batched tracer
def trace_batch(qhat, npts):
    """Vectorised locus trace for an array of qhat values.

    Returns normalized k2 points (m, npts, 2), quadrature weights
    what = ds/|grad zhat| (m, npts) using theta-quadrature with analytic
    ds/dtheta (periodic trapezoid; spectrally accurate for smooth loci).
    """
    qhat = np.asarray(qhat, float)
    m = qhat.size
    xp = ((1.0 - qhat ** 2) / (2.0 * qhat)) ** 2
    xm = -((np.sqrt(2.0 - qhat ** 2) - qhat) / 2.0) ** 2
    cx = 0.5 * (xp + xm)
    L = xp - xm
    th = (np.arange(npts) + 0.5) / npts * 2 * np.pi        # (npts,)
    ct, st = np.cos(th), np.sin(th)
    r_lo = np.full((m, npts), 1e-4) * L[:, None]
    r_hi = 2.0 * L[:, None] * np.ones((m, npts))

    def zval(r):
        x = cx[:, None] + r * ct[None, :]
        y = r * st[None, :]
        w2 = (x * x + y * y) ** 0.25
        w4 = ((x + 1.0) ** 2 + y * y) ** 0.25
        return qhat[:, None] + w2 - w4

    flo = zval(r_lo)
    # expand upper bracket where needed (zhat -> qhat > 0 far away)
    for _ in range(6):
        fhi = zval(r_hi)
        bad = flo * fhi > 0
        if not bad.any():
            break
        r_hi = np.where(bad, r_hi * 3.0, r_hi)
    for _ in range(64):
        mid = 0.5 * (r_lo + r_hi)
        fm = zval(mid)
        take_hi = flo * fm <= 0
        r_hi = np.where(take_hi, mid, r_hi)
        r_lo = np.where(take_hi, r_lo, mid)
        flo = np.where(take_hi, flo, fm)
    r = 0.5 * (r_lo + r_hi)

    x = cx[:, None] + r * ct[None, :]
    y = r * st[None, :]
    pts = np.stack([x, y], axis=-1)                        # (m,npts,2)
    gr = grad_zhat(pts.reshape(-1, 2)).reshape(m, npts, 2)
    e = np.stack([np.broadcast_to(ct, (m, npts)),
                  np.broadcast_to(st, (m, npts))], axis=-1)
    ep = np.stack([-e[..., 1], e[..., 0]], axis=-1)
    dzdr = np.einsum('mpj,mpj->mp', gr, e)
    dzdth = r * np.einsum('mpj,mpj->mp', gr, ep)
    rp = -dzdth / dzdr
    dsdth = np.sqrt(r * r + rp * rp)
    gnorm = np.hypot(gr[..., 0], gr[..., 1])
    what = dsdth * (2 * np.pi / npts) / gnorm
    return pts, what


def xc_webb_vec(K1, K2, K3, K4, grav):
    """Vectorised Webb-Dungey-Hui coefficient (arrays (...,2))."""
    k1 = np.hypot(K1[..., 0], K1[..., 1]); w1 = np.sqrt(k1)
    k2 = np.hypot(K2[..., 0], K2[..., 1]); w2 = np.sqrt(k2)
    k3 = np.hypot(K3[..., 0], K3[..., 1]); w3 = np.sqrt(k3)
    k4 = np.hypot(K4[..., 0], K4[..., 1]); w4 = np.sqrt(k4)
    dot = lambda a, b: a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    d12, d13, d14 = dot(K1, K2), dot(K1, K3), dot(K1, K4)
    d23, d24, d34 = dot(K2, K3), dot(K2, K4), dot(K3, K4)
    z12 = np.hypot(K1[..., 0] + K2[..., 0], K1[..., 1] + K2[..., 1]) \
        - (w1 + w2) ** 2
    z13 = np.hypot(K1[..., 0] - K3[..., 0], K1[..., 1] - K3[..., 1]) \
        - (w1 - w3) ** 2
    z14 = np.hypot(K1[..., 0] - K4[..., 0], K1[..., 1] - K4[..., 1]) \
        - (w1 - w4) ** 2
    eps = 1e-300
    p1 = 2 * (w1 + w2) ** 2 * (k1 * k2 - d12) * (k3 * k4 - d34) \
        / np.where(np.abs(z12) < eps, np.inf, z12)
    p2 = 2 * (w1 - w3) ** 2 * (k1 * k3 + d13) * (k2 * k4 + d24) \
        / np.where(np.abs(z13) < eps, np.inf, z13)
    p3 = 2 * (w1 - w4) ** 2 * (k1 * k4 + d14) * (k2 * k3 + d23) \
        / np.where(np.abs(z14) < eps, np.inf, z14)
    p4 = 0.5 * (d12 * d34 + d13 * d24 + d14 * d23)
    p5 = 0.25 * (d13 + d24) * (w1 - w3) ** 4
    p6 = -0.25 * (d12 + d34) * (w1 + w2) ** 4
    p7 = 0.25 * (d14 + d23) * (w1 - w4) ** 4
    p8 = 2.5 * k1 * k2 * k3 * k4
    p9 = (w1 + w2) ** 2 * (w1 - w3) ** 2 * (w1 - w4) ** 2 \
        * (k1 + k2 + k3 + k4)
    D = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    return grav ** 2 * (np.pi / 4) * D * D / (w1 * w2 * w3 * w4)


# ------------------------------------------------------------ table build
def build_table(grid, npts=24, npts_line=None, qhat_min=2e-3,
                qhat_max=1.0 - 1e-9, triv_filter=0.025, edge='strict',
                verbose=False):
    """Precompute the interaction table for theta1 = 0.

    For each (i1f, i3f, i3d) with q != 0: trace the (swap-normalised)
    locus, form the dimensional quadruplet, coupling, and bilinear
    stencils for roles 2 and 4.  Locus points within triv_filter * |P| of
    the trivial point k2 = k3 are dropped (their exact contribution is
    O(filter^2); convergence in the filter is testable).

    Returns dict of flat arrays: i1f, i3d3, i3f (int), base (= A3 w C),
    role-2 and role-4 stencils (ik, fk, id, fd, inside).
    """
    g = grid.g
    K = grid.k
    nf, nd = grid.nf, grid.nd
    out = {n: [] for n in ('i1f', 'i3f', 'i3d', 'base',
                           'ik2', 'fk2', 'id2', 'fd2', 'in2',
                           'ik4', 'fk4', 'id4', 'fd4', 'in4')}
    for i1f in range(nf):
        k1 = np.array([K[i1f], 0.0])
        # all (i3f, i3d) partners in one batch
        pairs = []
        qhats = []
        geo = []
        for i3f in range(nf):
            for i3d in range(nd):
                if i3f == i1f and i3d == 0:
                    continue
                k3 = K[i3f] * np.array([np.cos(grid.th[i3d]),
                                        np.sin(grid.th[i3d])])
                P = k1 - k3
                Pm = np.hypot(*P)
                if Pm < 1e-12:
                    continue
                q = np.sqrt(g) * (np.sqrt(K[i1f]) - np.sqrt(K[i3f]))
                if q == 0.0:
                    # degenerate same-ring pair: straight-line locus
                    from pyOM3.wim_simple.snl_exact.locus import trace_locus_degenerate
                    tmax = np.sqrt(max((K[-1] * 1.02 / Pm) ** 2 - 0.25,
                                       1.0))
                    trd = trace_locus_degenerate(tmax,
                                                 npts_line or 4 * npts)
                    phi = np.arctan2(P[1], P[0])
                    cr, sr = np.cos(phi), np.sin(phi)
                    k2h = Pm * trd['k2']
                    k2v = np.stack([cr * k2h[:, 0] - sr * k2h[:, 1],
                                    sr * k2h[:, 0] + cr * k2h[:, 1]], -1)
                    k4v = np.stack([cr * (k2h[:, 0] + Pm) - sr * k2h[:, 1],
                                    sr * (k2h[:, 0] + Pm) + cr * k2h[:, 1]],
                                   -1)
                    w = Pm ** 1.5 / np.sqrt(g) * trd['w']
                    dtr = np.hypot(k2v[:, 0] - k3[0],
                                   k2v[:, 1] - k3[1]) / Pm
                    keep = dtr > triv_filter
                    if not keep.any():
                        continue
                    k2v, k4v, w = k2v[keep], k4v[keep], w[keep]
                    k1a = np.broadcast_to(k1, k2v.shape)
                    k3a = np.broadcast_to(k3, k2v.shape)
                    Cc = xc_webb_vec(k1a, k2v, k3a, k4v, g) / g ** 2
                    ik2, fk2, id2, fd2, in2 = grid.stencil(k2v[:, 0],
                                                           k2v[:, 1])
                    ik4, fk4, id4, fd4, in4 = grid.stencil(k4v[:, 0],
                                                           k4v[:, 1])
                    good = (in2 & in4) if edge == 'strict' else (in2 | in4)
                    if not good.any():
                        continue
                    npt = int(good.sum())
                    out['i1f'].append(np.full(npt, i1f, np.int32))
                    out['i3f'].append(np.full(npt, i3f, np.int32))
                    out['i3d'].append(np.full(npt, i3d, np.int32))
                    out['base'].append((grid.area[i3f] * w * Cc)[good])
                    for nm, arr in (('ik2', ik2), ('fk2', fk2),
                                    ('id2', id2), ('fd2', fd2),
                                    ('in2', in2), ('ik4', ik4),
                                    ('fk4', fk4), ('id4', id4),
                                    ('fd4', fd4), ('in4', in4)):
                        out[nm].append(arr[good])
                    continue
                swapped = q < 0.0
                a1, a3 = (k3, k1) if swapped else (k1, k3)
                Ps = a1 - a3
                qh = (np.sqrt(np.hypot(*a1)) - np.sqrt(np.hypot(*a3))) \
                    / np.sqrt(Pm)
                if not (qhat_min < qh < qhat_max):
                    continue
                # prefilter: locus radius scale vs grid range
                xpl, xmi = axis_crossings(qh)
                if xpl * Pm < 0.5 * K[0]:      # entire locus below grid
                    continue
                pairs.append((i3f, i3d, swapped))
                qhats.append(qh)
                geo.append((Ps, Pm, k3))
        if not pairs:
            continue
        qhats = np.array(qhats)
        pts, what = trace_batch(qhats, npts)     # normalized
        m = len(pairs)
        for jm in range(m):
            i3f, i3d, swapped = pairs[jm]
            Ps, Pm, k3 = geo[jm]
            phi = np.arctan2(Ps[1], Ps[0])
            c, s = np.cos(phi), np.sin(phi)
            k2h = Pm * pts[jm]                       # (npts,2) unrotated
            k2v = np.stack([c * k2h[:, 0] - s * k2h[:, 1],
                            s * k2h[:, 0] + c * k2h[:, 1]], axis=-1)
            k4v = np.stack([c * (k2h[:, 0] + Pm) - s * k2h[:, 1],
                            s * (k2h[:, 0] + Pm) + c * k2h[:, 1]], axis=-1)
            if swapped:
                k2v, k4v = k4v, k2v
            w = Pm ** 1.5 / np.sqrt(g) * what[jm]
            # trivial-point filter (k2 -> k3, k4 -> k1)
            dtr = np.hypot(k2v[:, 0] - k3[0], k2v[:, 1] - k3[1]) / Pm
            keep = dtr > triv_filter
            if not keep.any():
                continue
            k2v, k4v, w = k2v[keep], k4v[keep], w[keep]
            k1a = np.broadcast_to(k1, k2v.shape)
            k3a = np.broadcast_to(k3, k2v.shape)
            C = xc_webb_vec(k1a, k2v, k3a, k4v, g) / g ** 2
            ik2, fk2, id2, fd2, in2 = grid.stencil(k2v[:, 0], k2v[:, 1])
            ik4, fk4, id4, fd4, in4 = grid.stencil(k4v[:, 0], k4v[:, 1])
            good = (in2 & in4) if edge == 'strict' else (in2 | in4)
            if not good.any():
                continue
            npt = int(good.sum())
            A3 = grid.area[i3f]
            out['i1f'].append(np.full(npt, i1f, np.int32))
            out['i3f'].append(np.full(npt, i3f, np.int32))
            out['i3d'].append(np.full(npt, i3d, np.int32))
            out['base'].append((A3 * w * C)[good])
            for nm, arr in (('ik2', ik2), ('fk2', fk2), ('id2', id2),
                            ('fd2', fd2), ('in2', in2), ('ik4', ik4),
                            ('fk4', fk4), ('id4', id4), ('fd4', fd4),
                            ('in4', in4)):
                out[nm].append(arr[good])
        if verbose and i1f % 6 == 0:
            print(f"  build: i1f={i1f}/{nf}, entries so far "
                  f"{sum(len(a) for a in out['base'])}")
    table = {nm: np.concatenate(v) for nm, v in out.items()}
    table['npts'] = npts
    return table


# --------------------------------------------------------------- evaluate
def _gather(n, ik, fk, idx, fd, inside):
    """Bilinear gather of density n (nf, nd) at stencil points."""
    nd = n.shape[1]
    i0, i1 = ik, ik + 1
    d0, d1 = idx % nd, (idx + 1) % nd
    v = ((1 - fk) * (1 - fd) * n[i0, d0] + (1 - fk) * fd * n[i0, d1]
         + fk * (1 - fd) * n[i1, d0] + fk * fd * n[i1, d1])
    return np.where(inside, v, 0.0)


def _scatter(acc, R, Aeff, ik, fk, idx, fd, inside, leak):
    """Adjoint of _gather: deposit action R (divided by effective area
    Aeff) with the gather weights; excluded action goes to leak."""
    nf, nd = acc.shape
    v = np.where(inside, R / Aeff, 0.0)
    leak[0] += float(np.sum(np.where(inside, 0.0, R)))
    i0, i1 = ik, ik + 1
    d0, d1 = idx % nd, (idx + 1) % nd
    flat = acc.ravel()
    for ii, dd, wgt in ((i0, d0, (1 - fk) * (1 - fd)),
                        (i0, d1, (1 - fk) * fd),
                        (i1, d0, fk * (1 - fd)),
                        (i1, d1, fk * fd)):
        np.add.at(flat, ii * nd + dd, wgt * v)


def snl(n, grid, table, mode='conservative', gross=False,
        deriv=False):
    """Evaluate dn/dt (nf, nd) for action density n (nf, nd) on the grid.

    mode = 'conservative': (+,+,-,-) scatter with weight 1/4, machine-
           precision action budget (returned leak closes it).
    mode = 'direct': role-1 only, independent non-conservative
           discretisation of the same integral (for cross-checks).
    gross=True: one-sided rate (|gain| + |loss|), the natural
           normalisation for equilibrium-null residuals.
    deriv=True: also return diag['dSdn'], the diagonal functional
           derivative dS(k1)/dn(k1) (role-1 sense), for semi-implicit
           stepping; invariant under the E <-> n conversion.
    Returns (dndt, diagnostics dict).
    """
    nf, nd = grid.nf, grid.nd
    dndt = np.zeros((nf, nd))
    dSdn = np.zeros((nf, nd)) if deriv else None
    leak = [0.0]
    A = grid.area                      # (nf,)
    i1f = table['i1f']; i3f = table['i3f']; i3d = table['i3d']
    for j in range(nd):                # theta1 shift
        d1 = j
        d3 = (i3d + j) % nd
        n1 = n[i1f, d1]
        n3 = n[i3f, d3]
        n2 = _gather(n, table['ik2'], table['fk2'], table['id2'] + j,
                     table['fd2'], table['in2'])
        n4 = _gather(n, table['ik4'], table['fk4'], table['id4'] + j,
                     table['fd4'], table['in4'])
        if gross:
            f = np.abs(n3 * n4 * (n1 + n2)) + np.abs(n1 * n2 * (n3 + n4))
        else:
            f = n3 * n4 * (n1 + n2) - n1 * n2 * (n3 + n4)
        rate = table['base'] * f       # = A3 w C f  (density rate for n1)
        if deriv:
            fd = n3 * n4 - n2 * (n3 + n4)      # d(bracket)/d n1
            np.add.at(dSdn[:, d1], i1f, table['base'] * fd)
        if mode == 'direct':
            np.add.at(dndt[:, d1], i1f, rate)
            continue
        R = 0.25 * rate * A[i1f]       # quadruplet action rate / 4
        np.add.at(dndt[:, d1], i1f, R / A[i1f])
        np.add.at(dndt.ravel(), i3f * nd + d3, -R / A[i3f])
        A2 = A[table['ik2']] * (1 - table['fk2']) + A[table['ik2'] + 1] \
            * table['fk2']             # effective area at gathered point
        A4 = A[table['ik4']] * (1 - table['fk4']) + A[table['ik4'] + 1] \
            * table['fk4']
        _scatter(dndt, R, A2, table['ik2'], table['fk2'],
                 table['id2'] + j, table['fd2'], table['in2'], leak)
        _scatter(dndt, -R, A4, table['ik4'], table['fk4'],
                 table['id4'] + j, table['fd4'], table['in4'], leak)
    diag = {}
    if deriv:
        diag['dSdn'] = dSdn                # diagonal dS(k1)/dn(k1)
    if mode == 'conservative':
        # action budget: sum A dndt + leak_action = 0 (machine precision)
        diag['action_rate'] = float(np.sum(A[:, None] * dndt))
        diag['leak'] = leak[0]
    return dndt, diag
