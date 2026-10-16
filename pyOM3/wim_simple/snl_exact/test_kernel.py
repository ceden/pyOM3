"""Validation battery for the derived canonical four-wave kernel.

Run (from any directory, package import): python3 -m pyOM3.wim_simple.snl_exact.test_kernel
"""
import numpy as np
from pyOM3.wim_simple.snl_exact.kernel import T, T_selfself, T_twowave_directional, T_unidirectional

rng = np.random.default_rng(42)
G = 9.81


def offset_pair(ki, kj, eps):
    """Wavenumber-closed near-diagonal quadruplet (ki,kj,ki+d,kj-d)."""
    d = eps * np.array([0.6180339887, -0.3819660113])
    return ki, kj, ki + d, kj - d


def rel(a, b):
    den = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / den


def report(name, err, tol):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} err={err:9.2e}")
    return ok


def main():
    allok = True

    # ---- (1) self-interaction limit: T(k,k,k,k) = k^3/4pi^2 -------------
    k = np.array([1.3, 0.7])
    vals = []
    for eps in (1e-3, 1e-4, 1e-5):
        a, b, c, d = offset_pair(k, k, eps)
        vals.append(T(a, b, c, d, G))
    err = rel(vals[-1], T_selfself(k))
    allok &= report("T(k,k,k,k) -> k^3/4pi^2", err, 1e-3)

    # ---- (2) directional two-wave kernel vs published A.5 ---------------
    worst = 0.0
    for _ in range(12):
        ki = rng.normal(size=2) * rng.uniform(0.3, 3.0)
        kj = rng.normal(size=2) * rng.uniform(0.3, 3.0)
        a, b, c, d = offset_pair(ki, kj, 1e-6)
        worst = max(worst, rel(T(a, b, c, d, G),
                               T_twowave_directional(ki, kj, G)))
    allok &= report("T(ki,kj,ki,kj) vs directional closed form", worst, 1e-4)

    # ---- (3) collinear special case ------------------------------------
    worst = 0.0
    for _ in range(6):
        qa, qb = rng.uniform(0.2, 4.0, size=2)
        ki, kj = np.array([qa, 0.0]), np.array([qb, 0.0])
        a, b, c, d = offset_pair(ki, kj, 1e-6)
        ref = qa * qb * min(qa, qb) / (4 * np.pi ** 2)
        worst = max(worst, rel(T(a, b, c, d, G), ref))
    allok &= report("collinear T = ka kb min/4pi^2", worst, 1e-4)

    # ---- (4) symmetries on generic resonant quadruplets -----------------
    # build a nontrivial resonant quadruplet: fix k1,k3; solve for k4 on a
    # ray so that omega closure holds with k2 = k3 + k4 - k1.
    def resonant_quadruplet(k1, k3, theta):
        e = np.array([np.cos(theta), np.sin(theta)])

        def defect(s):
            k4 = s * e
            k2 = k3 + k4 - k1
            w = lambda v: np.sqrt(G * np.hypot(v[0], v[1]))
            return w(k1) + w(k2) - w(k3) - w(k4)

        lo, hi = 1e-6, 60.0
        flo = defect(lo)
        s = None
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            fm = defect(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
            if hi - lo < 1e-13 * hi:
                s = 0.5 * (lo + hi)
                break
        if s is None or abs(defect(s)) > 1e-9:
            return None
        k4 = s * e
        return k1, k3 + k4 - k1, k3, k4

    worst_s, worst_i = 0.0, 0.0
    found = 0
    for th in np.linspace(0.3, 2.6, 14):
        quad = resonant_quadruplet(np.array([1.0, 0.0]),
                                   np.array([1.6, 0.8]), th)
        if quad is None:
            continue
        k1, k2, k3, k4 = quad
        if min(np.hypot(*(k1 - k3)), np.hypot(*(k1 - k4))) < 1e-3:
            continue  # skip trivial neighborhood
        found += 1
        t0 = T(k1, k2, k3, k4, G)
        worst_s = max(worst_s,
                      rel(t0, T(k2, k1, k3, k4, G)),
                      rel(t0, T(k1, k2, k4, k3, G)),
                      rel(t0, T(k3, k4, k1, k2, G)))
        # rotation, scaling, g-independence
        R = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
        worst_i = max(worst_i,
                      rel(t0, T(R @ k1, R @ k2, R @ k3, R @ k4, G)),
                      rel(2.0 ** 3 * t0, T(2 * k1, 2 * k2, 2 * k3, 2 * k4, G)),
                      rel(t0, T(k1, k2, k3, k4, g=1.0)))
    print(f"      ({found} nontrivial resonant quadruplets found)")
    allok &= report("on-shell symmetries 1<->2, 3<->4, (12)<->(34)",
                    worst_s, 1e-9)
    allok &= report("rotation / scaling(deg 3) / g-independence",
                    worst_i, 1e-9)

    # ---- (5) unidirectional formula (DZ gauge; informational) -----------
    ka, kb, kc = 1.0, 2.2, 0.7
    kd = ka + kb - kc
    e = np.array([1.0, 0.0])
    t = T(ka * e, kb * e, kc * e + 1e-7, kd * e - 1e-7, G)
    err = rel(t, T_unidirectional(ka, kb, kc, kd))
    print(f"INFO  unidirectional off-shell vs DZ-gauge A.3       "
          f"err={err:9.2e}  (gauge-dependent off shell)")


    # ---- (6) cross-check vs WW3/xnl Webb-Dungey-Hui coefficient ---------
    from pyOM3.wim_simple.snl_exact.kernel import xc_webb
    C = 64 * np.pi ** 5 * G ** 2
    worst = 0.0
    nq = 0
    for k1b, k3b in [(np.array([1.0, 0.0]), np.array([1.6, 0.8])),
                     (np.array([2.3, 0.0]), np.array([0.9, 1.1])),
                     (np.array([0.6, 0.2]), np.array([0.5, -0.7]))]:
        for th in np.linspace(0.3, 2.7, 7):
            quad = resonant_quadruplet(k1b, k3b, th)
            if quad is None:
                continue
            k1, k2, k3, k4 = quad
            if min(np.hypot(*(k1 - k3)), np.hypot(*(k1 - k4))) < 5e-2:
                continue
            t = T(k1, k2, k3, k4, G)
            if abs(t) < 1e-6:
                continue
            nq += 1
            worst = max(worst, abs(xc_webb(k1, k2, k3, k4, G) / t ** 2 / C - 1))
    print(f"      ({nq} quadruplets vs WW3 xc_webb)")
    allok &= report("xc_webb = 64 pi^5 g^2 T^2 (WW3 cross-check)", worst, 1e-9)

    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return allok


if __name__ == "__main__":
    main()
