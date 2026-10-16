"""Validation battery for the WRT assembly.

Run: python3 test_assembly.py    (takes a couple of minutes)
"""
import numpy as np
from assembly import PolarGrid, build_table, snl

rng = np.random.default_rng(5)


def report(name, err, tol):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} err={err:9.2e}")
    return ok


def jonswap_like(grid, fp=0.2, alpha=0.01, gamma=3.3, spread=2):
    """Peaked action density n(k,theta) = g F / omega, JONSWAP-ish."""
    g = grid.g
    f = grid.f[:, None]
    sig = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * sig ** 2 * fp ** 2))
    Ef = alpha * g ** 2 * (2 * np.pi) ** -4 * f ** -5 \
        * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    D = np.cos(0.5 * (grid.th[None, :] - np.pi)) ** (2 * spread)
    D = D / (np.sum(D[0]) * grid.dth)
    Eft = Ef * D                            # E(f, theta)
    om = 2 * np.pi * f
    # n(k) d2k = E(f,th)/om df dth  ->  n = E/(om * k dk/df)
    dkdf = 8 * np.pi ** 2 * f / g
    n = Eft / (om * grid.k[:, None] * dkdf)
    return n


def main():
    allok = True
    grid = PolarGrid(f0=0.09, ratio=1.10, nf=30, nd=24, g=9.81)
    print(f"grid: nf={grid.nf} nd={grid.nd} "
          f"f=[{grid.f[0]:.3f},{grid.f[-1]:.3f}] Hz")
    table = build_table(grid, npts=20, triv_filter=0.025)
    print(f"table entries: {len(table['base'])}")

    n = jonswap_like(grid)
    S, diag = snl(n, grid, table, mode='conservative')

    # ---- (1) machine-precision action budget ----------------------------
    scale = float(np.sum(np.abs(grid.area[:, None] * S)))
    err = abs(diag['action_rate'] + diag['leak']) / scale
    allok &= report("action budget sum(A dn/dt) + leak = 0", err, 1e-12)

    # ---- (2) exact rotation equivariance --------------------------------
    m = 5
    S_rot, _ = snl(np.roll(n, m, axis=1), grid, table, mode='conservative')
    err = np.max(np.abs(S_rot - np.roll(S, m, axis=1))) / np.max(np.abs(S))
    allok &= report("rotation equivariance (theta shift)", err, 1e-13)

    # ---- (3) isotropic spectrum -> isotropic transfer -------------------
    n_iso = np.mean(n, axis=1, keepdims=True) * np.ones((1, grid.nd))
    S_iso, _ = snl(n_iso, grid, table, mode='conservative')
    err = np.max(np.std(S_iso, axis=1)) / np.max(np.abs(S_iso))
    allok &= report("isotropic input -> isotropic transfer", err, 1e-13)

    # ---- (4) cubic homogeneity in n -------------------------------------
    S3, _ = snl(2.0 * n, grid, table, mode='conservative')
    err = np.max(np.abs(S3 - 8.0 * S)) / np.max(np.abs(8.0 * S))
    allok &= report("S(2n) = 8 S(n)", err, 1e-13)

    # ---- (5) k-shift self-similarity: S ~ k^{19/2} n^3 ------------------
    # closed-locus part of the table is exactly shift-covariant (relative
    # stencils); same-ring line loci cap at the absolute grid edge, so they
    # are excluded here.  Residual = compact-bump tail leakage at the
    # fixed grid boundary.
    mask = table['i3f'] != table['i1f']
    t_nd = {k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in table.items()}
    nb = np.zeros_like(n)
    prof = np.exp(-0.5 * ((np.arange(grid.nf) - 14.0) / 2.0) ** 2)
    D = np.cos(0.5 * (grid.th - np.pi)) ** 4
    nb[:] = prof[:, None] * D[None, :] * 1e-4
    nb_sh = np.zeros_like(nb)
    nb_sh[1:] = nb[:-1]
    Sb, _ = snl(nb, grid, t_nd, mode='conservative')
    Sb_sh, _ = snl(nb_sh, grid, t_nd, mode='conservative')
    rho = grid.k[1] / grid.k[0]
    m = np.zeros_like(Sb, bool)
    m[6:grid.nf - 6] = np.abs(Sb[6:grid.nf - 6]) > 1e-3 * np.abs(Sb).max()
    err = np.max(np.abs(Sb_sh[1:][m[:-1]] - rho ** 9.5 * Sb[:-1][m[:-1]])) \
        / np.max(np.abs(Sb))
    allok &= report("geometric shift: S scales as rho^{19/2}", err, 1e-6)

    # ---- (6) direct vs conservative discretisations ---------------------
    S_dir, _ = snl(n, grid, table, mode='direct')
    num = np.sum(grid.area[:, None] * np.abs(S_dir - S))
    den = np.sum(grid.area[:, None] * np.abs(S))
    print(f"      direct-vs-conservative L1 difference: {num/den:.3f}")
    allok &= report("direct/conservative agree (independent disc.)",
                    num / den, 0.35)

    # ---- (7) three-lobe structure of the 1D energy transfer -------------
    om = 2 * np.pi * grid.f
    Se = np.sum(S * grid.area[:, None], axis=1) * om
    thr = 0.02 * np.max(np.abs(Se))
    sgn = [int(np.sign(s)) for s in Se[:grid.nf - 2] if abs(s) > thr]
    pattern = [sgn[0]]
    for s in sgn[1:]:
        if s != pattern[-1]:
            pattern.append(s)
    print(f"      lobe sign pattern (energetic rings): {pattern}")
    ok = pattern == [1, -1, 1]
    print(f"{'PASS' if ok else 'FAIL'}  three-lobe (+,-,+) structure")
    allok &= ok

    # ---- (8) trivial-filter convergence ---------------------------------
    t2 = build_table(grid, npts=20, triv_filter=0.05)
    S2, _ = snl(n, grid, t2, mode='conservative')
    d = np.sum(grid.area[:, None] * np.abs(S2 - S)) \
        / np.sum(grid.area[:, None] * np.abs(S))
    allok &= report("filter 0.025 vs 0.05 consistency", d, 0.05)

    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return allok


if __name__ == "__main__":
    main()
