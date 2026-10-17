"""Test 16: the result must not depend on the number of halo points.
nh_realistic on the pe configuration (stretched Cartesian grid, shelf and
seamount, TKE closure) with halo_points = 2 and 3 against the default 1,
after N steps, every field of the state including p_nh, bit for bit on one
PE; implicit and split-explicit free surface. The 3-D solve runs with its
default criterion: on one PE its sums do not depend on the halo width.

Oct 17 2026: passes once the two pe_realistic fixes of pe/halo.py are in
(make_topo_masks level number, u_centered_grid_halo); nh_realistic itself
needed no change."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line
from common import reldiff                                   # test/common.py, on the path once config is imported

CASES = (('implicit', dict()),
         ('explicit', dict(explicit=True)))


def run(N=6, halos=(2, 3)):
    ok = True
    for name, kw in CASES:
        ref = state(step(small_model(**kw), N))
        for h in halos:
            d = max(reldiff(state(step(small_model(halo=h, **kw), N)), ref).values())
            ok &= line('16 halo %d = halo 1, nh_realistic %s' % (h, name), d, 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
