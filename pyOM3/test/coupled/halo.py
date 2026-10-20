"""Test 16 for the coupling code: the result must not depend on the number
of halo points. The coupled and the forced ocean with halo_points = 2 and 3
against the default 1, after N steps, every field of the state bit for bit
(the pe family's test on the package; here the shared code's own index
arithmetic: the block cut of the forcing, the pad of the forcing fields,
kbot_global's halo-1 convention, the surface level ks = -h-1, the
shortwave profile, the coupler's interior fill of the wave fields, the
diagnostics' interior sums)."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_ocean, step, state, line, reldiff

CASES = (('forced', dict(coupled=False)),
         ('forced relwind ice sw_bottom', dict(coupled=False, relwind=True, ice=0.3, sw_bottom=True, average='6h')),
         ('coupled', dict(coupled=True)),
         ('coupled direct shear mxl 2', dict(coupled=True, direct=True, shear=True, mxl_choice=2)),
         ('coupled distributed stress', dict(coupled=True, stress_work=True, ke_budget=0)))


def run(N=6, halos=(2, 3)):
    ok = True
    for name, kw in CASES:
        ref = state(step(small_ocean(tag='h1', **kw), N))
        for h in halos:
            d = max(reldiff(state(step(small_ocean(halo=h, tag='h%d' % h, **kw), N)), ref).values())
            ok &= line('16 halo %d = halo 1, %s' % (h, name), d, 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
