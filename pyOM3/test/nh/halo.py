"""Test 16: the result must not depend on the number of halo points.
The small model with halo_points = 2 and 3 against the default 1, after N
steps, every field of the state, bit for bit on one PE (the FFT solver
works on the interior, nothing in the discrete problem changes; a
difference is an index that counts from the array edge instead of from the
halo). Lid default, superbee + TKE closure, Smagorinsky, biharmonic,
periodic in z. p3dfft holds one plan per process, so every model is
finished before the next one is built (del before the next small_model).

Oct 17 2026: nh_simple passes as it is, nothing was changed."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, reldiff

CASES = (('lid', dict(kind='lid')),
         ('lid superbee + tke', dict(kind='lid', b_adv='superbee', les='tke')),
         ('lid smagorinsky', dict(kind='lid', les='smag')),
         ('lid biharmonic', dict(kind='lid', biharmonic=True)),
         ('periodic', dict(kind='periodic')))


def run(N=6, halos=(2, 3)):
    ok = True
    for name, kw in CASES:
        m = step(small_model(**kw), N); ref = state(m); del m
        for h in halos:
            m = step(small_model(halo=h, **kw), N); s = state(m); del m
            ok &= line('16 halo %d = halo 1, %s' % (h, name), max(reldiff(s, ref).values()), 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
