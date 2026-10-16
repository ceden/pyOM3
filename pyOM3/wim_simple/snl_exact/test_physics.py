"""Physics battery for the exact S_nl: equilibrium nulls, KZ cancellation,
and the WAM-DIA absolute-magnitude anchor.

Run: python3 test_physics.py    (a few minutes)
"""
import numpy as np
from assembly import PolarGrid, build_table, snl
from wim_interface import jonswap_Eft, Eft_to_nk, Snk_to_SEft, dia_snl, \
    export_for_xnl


def cancellation(S, Sg, A, sl):
    """|net| / gross on rings sl — how completely gain and loss cancel."""
    num = abs(np.sum(A[sl, None] * S[sl]))
    den = np.sum(A[sl, None] * Sg[sl])
    return num / den


def gross(n, grid, table):
    """Gross one-sided rate: evaluate with bracket replaced by |bracket|
    via the positive/negative split S = S(n) on n and the same n with the
    sign trick: gross ~ (|S_gain| + |S_loss|)/2 estimated from |S|."""
    S, _ = snl(n, grid, table, mode='direct')
    return np.abs(S)


def main():
    allok = True
    grid = PolarGrid(f0=0.10, ratio=1.10, nf=28, nd=24, g=9.81)
    print(f"grid: nf={grid.nf} nd={grid.nd} "
          f"f=[{grid.f[0]:.3f},{grid.f[-1]:.3f}] Hz")
    table = build_table(grid, npts=24, triv_filter=0.025)
    A = grid.area
    om = 2 * np.pi * grid.f
    inner = slice(6, grid.nf - 6)

    # ---- (1) equilibria: bracket vanishes on the resonance manifold ----
    # n = const  and  n = c/omega (Rayleigh-Jeans).  Residual measures
    # bilinear interpolation error of n at roles 2,4 (O(h^2)), since the
    # on-locus resonance is machine-exact.
    scale = 1e-4
    for name, prof, tol in (("n = const", np.ones(grid.nf), 1e-12),
                            ("n = c/omega (RJ)", 1.0 / om, 3e-4)):
        n_eq = scale * prof[:, None] * np.ones((1, grid.nd))
        S_eq, _ = snl(n_eq, grid, table, mode='direct')
        G_eq, _ = snl(n_eq, grid, table, mode='direct', gross=True)
        r = np.max(np.abs(S_eq[inner]) / np.maximum(G_eq[inner], 1e-300))
        ok = r < tol
        print(f"{'PASS' if ok else 'FAIL'}  equilibrium null {name:18s} "
              f"residual/gross = {r:.2e}")
        allok &= ok

    # ---- (2) KZ direct-cascade spectrum: E(f) ~ f^-4  =>  n ~ k^-4 ----
    n_kz = 1e-5 * (grid.k[:, None] / grid.k[14]) ** -4.0 \
        * np.ones((1, grid.nd))
    S_kz, _ = snl(n_kz, grid, table, mode='direct')
    n_j5 = 1e-5 * (grid.k[:, None] / grid.k[14]) ** -5.0 \
        * np.ones((1, grid.nd))          # steeper: strongly non-null
    S_j5, _ = snl(n_j5, grid, table, mode='direct')
    r_kz = np.max(np.abs(S_kz[inner] / n_kz[inner] ** 3)) \
        / np.max(np.abs(S_j5[inner] / n_j5[inner] ** 3))
    print(f"      KZ f^-4 normalized residual vs k^-5 reference: "
          f"{r_kz:.3f}")
    ok = r_kz < 0.35
    print(f"{'PASS' if ok else 'FAIL'}  KZ f^-4 near-null vs non-KZ slope")
    allok &= ok

    # ---- (3) JONSWAP vs WAM DIA: absolute-magnitude anchor -------------
    E = jonswap_Eft(grid, fp=0.3, alpha=0.0081, gamma=3.3)
    n = Eft_to_nk(E, grid)
    S_n, diag = snl(n, grid, table, mode='conservative')
    S_E = Snk_to_SEft(S_n, grid)
    S_D = dia_snl(E, grid)
    S1 = np.sum(S_E, axis=1) * grid.dth       # m^2/Hz/s
    D1 = np.sum(S_D, axis=1) * grid.dth
    ip = int(np.argmax(np.sum(E, axis=1)))
    print("      f[Hz]    exact S1        DIA S1")
    for i in range(2, grid.nf - 2, 2):
        print(f"      {grid.f[i]:.3f}  {S1[i]:+.4e}  {D1[i]:+.4e}")
    # anchor: positive low-f lobe maxima within a factor 3 (DIA quality)
    lo = slice(0, ip)
    ratio = np.max(S1[lo]) / np.max(D1[lo])
    print(f"      low-f positive lobe: exact/DIA peak ratio = {ratio:.2f}")
    ok = 1.0 / 3.0 < ratio < 3.0
    print(f"{'PASS' if ok else 'FAIL'}  absolute magnitude within DIA "
          f"tolerance (rules out 2pi-power errors)")
    allok &= ok

    # ---- (4) export for the definitive WW3 mod_xnl4v5 diff -------------
    export_for_xnl("/home/claude/snl_exact/xnl_reference_case.txt",
                   grid, E, S_E)
    print("      export written: xnl_reference_case.txt "
          "(JONSWAP fp=0.3, alpha=0.0081, gamma=3.3, cos^2)")

    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
    return allok


if __name__ == "__main__":
    main()
