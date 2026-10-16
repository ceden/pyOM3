"""WIM-facing interface for the exact S_nl reference implementation.

Conventions
-----------
E(f, theta): variance density [m^2 / Hz / rad],  total <zeta^2> = int E df dth.
n(k, theta): action density per d^2k used by assembly.py,  n = g F(k)/omega
with F(k) the variance density per d^2k.  Deep water:  k = (2 pi f)^2 / g,
dk/df = 8 pi^2 f / g, and

    n(k, theta) = E(f, theta) / (omega * k * dk/df),
    S_E(f, theta) = S_n(k, theta) * omega * k * dk/df      [m^2/Hz/rad/s].

The WAM cycle-3 DIA (Hasselmann et al. 1985; WW3 manual) is provided as an
order-of-magnitude anchor: DIA was tuned against exact computations, so
agreement in shape and O(1) magnitude validates the whole constant chain
(a stray 2pi-power would appear as x6.3 or x39).
"""

import numpy as np
from assembly import PolarGrid, build_table, snl


# ------------------------------------------------------------ conversions
def jonswap_Eft(grid, fp=0.3, alpha=0.0081, gamma=3.3, spread_m=2,
                thm=np.pi):
    """JONSWAP E(f,theta) [m^2/Hz/rad] with cos^m spreading (m=2: cos^2)."""
    g = grid.g
    f = grid.f[:, None]
    sig = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * sig ** 2 * fp ** 2))
    Ef = alpha * g ** 2 * (2 * np.pi) ** -4 * f ** -5 \
        * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    dth = np.mod(grid.th[None, :] - thm + np.pi, 2 * np.pi) - np.pi
    D = np.where(np.abs(dth) < np.pi / 2, np.cos(dth) ** spread_m, 0.0) \
        if spread_m == 2 else np.cos(0.5 * dth) ** (2 * spread_m)
    D = D / np.sum(D[0] * grid.dth)
    return Ef * D


def Eft_to_nk(E, grid):
    """n(k,theta) = g F / omega with F = E / (k dk/df):  n = g E / (om k dk/df)."""
    om = 2 * np.pi * grid.f[:, None]
    dkdf = 8 * np.pi ** 2 * grid.f[:, None] / grid.g
    return grid.g * E / (om * grid.k[:, None] * dkdf)


def Snk_to_SEft(S, grid):
    """Inverse mapping for the tendency: S_E = S_n * om k dk/df / g."""
    om = 2 * np.pi * grid.f[:, None]
    dkdf = 8 * np.pi ** 2 * grid.f[:, None] / grid.g
    return S * om * grid.k[:, None] * dkdf / grid.g


# ------------------------------------------------------------------- DIA
def dia_snl(E, grid, Cnl4=3.00e7, lam=0.25):
    """WAM cycle-3 DIA on E(f,theta) [m^2/Hz/rad] -> S [m^2/Hz/rad/s].

    Representative quadruplet f3 = (1+lam) f, f4 = (1-lam) f at angles
    +11.48 deg / -33.56 deg (and the mirror), increments (-2, +1, +1).
    Bilinear gather/scatter in (ln f, theta); off-grid legs use a f^-5
    variance tail below/above the grid (zero outside a factor safety).
    """
    g = grid.g
    nf, nd = grid.nf, grid.nd
    lnf = np.log(grid.f)
    dlnf = lnf[1] - lnf[0]
    dth1, dth2 = np.deg2rad(11.48), np.deg2rad(-33.56)

    def stencil_f(ftar):
        x = (np.log(ftar) - lnf[0]) / dlnf
        i0 = np.clip(np.floor(x).astype(int), 0, nf - 2)
        fr = x - i0
        inside = (x >= 0) & (x <= nf - 1)
        return i0, np.clip(fr, 0, 1), inside

    def gather(A, i0, fr, joff, inside):
        j = (np.arange(nd)[None, :] + joff) % nd
        v = (1 - fr)[:, None] * A[i0][:, j[0]] + fr[:, None] * A[i0 + 1][:, j[0]]
        return np.where(inside[:, None], v, 0.0)

    S = np.zeros_like(E)
    fq = grid.f
    for sgn in (+1, -1):
        o1 = int(round(sgn * dth1 / grid.dth))
        o2 = int(round(sgn * dth2 / grid.dth))
        i3, r3, in3 = stencil_f((1 + lam) * fq)
        i4, r4, in4 = stencil_f((1 - lam) * fq)
        F1 = E
        F3 = gather(E, i3, r3, o1, in3)
        F4 = gather(E, i4, r4, o2, in4)
        dS = Cnl4 * g ** -4 * fq[:, None] ** 11 * (
            F1 ** 2 * (F3 / (1 + lam) ** 4 + F4 / (1 - lam) ** 4)
            - 2 * F1 * F3 * F4 / (1 - lam ** 2) ** 4)
        S += -2 * dS
        # scatter +1 increments back to the (f3, f4) legs
        # satellite deposits carry the density Jacobian df1/df_sat
        for (ii, rr, oo, ins, w) in ((i3, r3, o1, in3, 1.0 / (1 + lam)),
                                     (i4, r4, o2, in4, 1.0 / (1 - lam))):
            j = (np.arange(nd)[None, :] + oo) % nd
            add = np.where(ins[:, None], w * dS, 0.0)
            np.add.at(S, (ii[:, None], j), (1 - rr)[:, None] * add)
            np.add.at(S, (ii[:, None] + 1, j), rr[:, None] * add)
    return S                # both mirror configurations are summed (WAM)


# ---------------------------------------------------------------- export
def export_for_xnl(path, grid, E, S_E):
    """Write spectrum + exact S_nl in plain text for a WW3 xnl diff."""
    with open(path, 'w') as h:
        h.write("# exact WRT S_nl reference export\n")
        h.write("# grid: geometric f, uniform theta; deep water, "
                f"g = {grid.g}\n")
        h.write("# E(f,theta) [m^2/Hz/rad]; S(f,theta) [m^2/Hz/rad/s]\n")
        h.write(f"nf {grid.nf}\nnd {grid.nd}\n")
        h.write("f " + " ".join(f"{x:.8e}" for x in grid.f) + "\n")
        h.write("th " + " ".join(f"{x:.8e}" for x in grid.th) + "\n")
        for name, A in (("E", E), ("S", S_E)):
            for i in range(grid.nf):
                h.write(name + " " +
                        " ".join(f"{x:.8e}" for x in A[i]) + "\n")
