"""Validation battery for the locus tracer.

Run (from any directory, package import): python3 -m pyOM3.wim_simple.snl_exact.test_locus
"""
import numpy as np
from pyOM3.wim_simple.snl_exact.locus import trace_locus, locus_dimensional, zhat, grad_zhat, \
    axis_crossings

rng = np.random.default_rng(11)


def report(name, err, tol):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} err={err:9.2e}")
    return ok


def main():
    allok = True

    # ---- (1) residuals and analytic axis crossings ----------------------
    worst_res, worst_cross = 0.0, 0.0
    for qh in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95):
        tr = trace_locus(qh, npoints=96)
        worst_res = max(worst_res, tr['residual'])
        xp, xm = axis_crossings(qh)
        worst_cross = max(worst_cross,
                          abs(zhat(np.array([xp, 0.0]), qh)),
                          abs(zhat(np.array([xm, 0.0]), qh)))
    allok &= report("locus residual |zhat| at returned points", worst_res,
                    1e-11)
    allok &= report("analytic axis crossings satisfy zhat = 0", worst_cross,
                    1e-13)

    # ---- (2) dimensional resonance closure, both signs of q -------------
    worst = 0.0
    for _ in range(8):
        k1 = rng.normal(size=2) * rng.uniform(0.3, 3.0)
        k3 = rng.normal(size=2) * rng.uniform(0.3, 3.0)
        g = rng.choice([1.0, 9.81])
        try:
            L = locus_dimensional(k1, k3, g=g, npoints=48)
        except ValueError:
            continue
        w = lambda v: np.sqrt(g * np.hypot(v[..., 0], v[..., 1]))
        defect = w(k1[None]) + w(L['k2']) - w(k3[None]) - w(L['k4'])
        kdef = np.max(np.abs(k1[None] + L['k2'] - k3[None] - L['k4']))
        worst = max(worst, np.max(np.abs(defect)) / w(k1[None])[0], kdef)
    allok &= report("dimensional quadruplets close (omega and k)", worst,
                    1e-10)

    # ---- (3) delta-function line quadrature vs brute-force 2D -----------
    # I = int F(k2) delta(zhat(k2)) d^2k2, F a Gaussian bump near the locus
    qh = 0.45
    tr = trace_locus(qh, npoints=128)
    c0 = tr['k2'][10] + 0.0
    F = lambda p: np.exp(-((p[..., 0] - c0[0]) ** 2
                           + (p[..., 1] - c0[1]) ** 2) / 0.4)
    I_locus = float(np.sum(tr['w'] * F(tr['k2'])))
    xp, xm = tr['x_plus'], tr['x_minus']
    span = xp - xm
    xs = np.linspace(xm - 0.6 * span, xp + 0.6 * span, 1401)
    ys = np.linspace(-0.9 * span, 0.9 * span, 1401)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X, Y], axis=-1)
    Z = zhat(pts, qh)
    vals = []
    for eps in (0.004, 0.002, 0.001):
        delta = np.exp(-Z ** 2 / (2 * eps ** 2)) / (np.sqrt(2 * np.pi) * eps)
        vals.append(np.sum(F(pts) * delta) * (xs[1] - xs[0]) * (ys[1] - ys[0]))
    I_grid = vals[-1] + (vals[-1] - vals[-2]) / 3.0   # Richardson-ish
    err = abs(I_locus - I_grid) / abs(I_grid)
    allok &= report("delta-line quadrature vs 2D brute force", err, 2e-3)

    # ---- (4) |P|^{3/2} g^{-1/2} measure scaling -------------------------
    k1 = np.array([1.2, 0.3]); k3 = np.array([0.5, -0.4])
    lam = 2.7
    L1 = locus_dimensional(k1, k3, g=9.81, npoints=64)
    L2 = locus_dimensional(lam * k1, lam * k3, g=9.81, npoints=64)
    r = np.sum(L2['w']) / np.sum(L1['w'])
    err = abs(r / lam ** 1.5 - 1.0)
    allok &= report("total measure scales as lambda^{3/2}", err, 1e-12)

    # ---- (5) spectral convergence in point count ------------------------
    qh = 0.3
    G = lambda p: np.cos(1.3 * p[..., 0]) * np.exp(-0.2 * p[..., 1] ** 2)
    I_ref = None
    errs = []
    for n in (32, 64, 128, 512):
        t = trace_locus(qh, npoints=n, ntheta_internal=4096)
        I = float(np.sum(t['w'] * G(t['k2'])))
        if I_ref is None or n < 512:
            errs.append(I)
        I_ref = I
    conv = [abs(e - I_ref) / abs(I_ref) for e in errs]
    print(f"      convergence (n=32,64,128 vs 512): "
          f"{conv[0]:.2e} {conv[1]:.2e} {conv[2]:.2e}")
    allok &= report("quadrature converged by n=128", conv[2], 1e-8)

    # ---- (6) trivial point k2 = k3 lies on the locus --------------------
    k1 = np.array([1.0, 0.0]); k3 = np.array([1.5, 0.9])
    g = 9.81
    L = locus_dimensional(k1, k3, g=g, npoints=512)
    d = np.min(np.hypot(*(L['k2'] - k3[None]).T))
    scale = np.hypot(*(k1 - k3))
    allok &= report("trivial point k2=k3 on locus (min dist/|P|)", d / scale,
                    5e-2)

    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return allok


if __name__ == "__main__":
    main()
