"""Test 16: the result must not depend on the number of halo points.
The small model with halo_points = 2 and 3 against the default 1, after N
steps, every field of the state, bit for bit on one PE. In the wave model
the halo width also sets the number of ghost bins of the frequency and the
direction axis (Nf = nf + 2 halo, Nd = nd + 2 halo), so A is compared on
its interior bins and the two OUTERMOST ghost bins separately (the f^-5
tail of imphftail and the ghost the tail stress reads live there).
wim_realistic (sphere, variable grid), wim_simple, reflection, janssen,
and the adiabatic propagation alone.

Writing it found (Oct 17 2026): set_grid filled only the outermost ghost
bin of ft, fu, dft, dfu and theta, so with halo_points > 1 the inner ghost
bins had omega = 0 (division by zero in every source term and in the
propagation); and the horizontal u-centered grid recursion of
wim_realistic ran from the outermost ghost point, which on the periodic x
axis with variable dxt moved xt by O(dx) with the halo width (see
grid.u_centered_grid_halo)."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, reldiff
import pyOM3 as OM

CASES = (('realistic', dict(kind='realistic')),
         ('simple', dict(kind='simple')),
         ('realistic reflection', dict(kind='realistic', reflection=True)),
         ('realistic janssen', dict(kind='realistic', janssen=True)),
         ('realistic adiabatic', dict(kind='realistic', sources=False)))


def _split(s, h):
    """A -> interior frequency bins, A_fghost -> the two outermost ghost bins"""
    s = dict(s)
    s['A_fghost'] = s['A'][:, [h - 1, -h]]
    s['A'] = s['A'][:, h:-h]
    return s


def run(N=6, halos=(2, 3)):
    ok = True
    for name, kw in CASES:
        ref = _split(state(step(small_model(**kw), N)), 1)
        for h in halos:
            d = max(reldiff(_split(state(step(small_model(halo=h, **kw), N)), h), ref).values())
            # exact on numpy; on the jax backend XLA compiles each array shape
            # on its own and the fused kernels round differently (GPU: 3e-16)
            ok &= line('16 halo %d = halo 1, %s' % (h, name), d, 1e-15 if OM.jax else 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
