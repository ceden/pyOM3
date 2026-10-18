"""Test 16: the result must not depend on the number of halo points.
The small model with halo_points = 2 and 3 against the default 1, after N
steps, every field of the state, bit for bit on one PE (nothing in the
discrete problem changes; a difference is an index that counts from the
array edge instead of from the halo, or a sum that runs over halo
points). pe_simple and pe_realistic, implicit and explicit free surface,
both mixing lengths, the vadv split and AB3 on pe_realistic.

Writing it found (Oct 17 2026): make_topo_masks compared kbot with the
array index k instead of the level number k - halo + 1 (both packages; the
whole topography was shifted by halo - 1 levels for halo_points > 1), and
the horizontal u-centered grid recursion of pe_realistic ran from the
outermost ghost point, so the coordinates moved at round-off with the halo
width (and by O(dx) on a periodic axis with variable spacing, see
grid.u_centered_grid_halo)."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, reldiff

CASES = (('realistic implicit', dict(kind='realistic')),
         ('realistic explicit', dict(kind='realistic', explicit=True)),
         ('realistic implicit mxl 2', dict(kind='realistic', mxl_choice=2)),
         ('realistic implicit vadv 2', dict(kind='realistic', vadv=2)),
         ('realistic explicit tracer_flux', dict(kind='realistic', explicit=True, tracer_flux=True)),
         ('simple implicit', dict(kind='simple')),
         ('simple explicit', dict(kind='simple', explicit=True)),
         ('simple implicit mxl 2', dict(kind='simple', mxl_choice=2)))


def run(N=6, halos=(2, 3, 4)):
    ok = True
    for name, kw in CASES:
        ab = 'ab3' if 'vadv' in kw else (1.6, -0.6, 0.0)
        ref = state(step(small_model(**kw), N, ab))
        for h in halos:
            d = max(reldiff(state(step(small_model(halo=h, **kw), N, ab)), ref).values())
            ok &= line('16 halo %d = halo 1, %s' % (h, name), d, 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
