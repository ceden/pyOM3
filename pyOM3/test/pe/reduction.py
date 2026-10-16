"""Test 1: pe_simple against pe_realistic on a uniform Cartesian grid, for the
implicit and the explicit free surface and both mixing lengths."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, reldiff, line

def run(nsteps=8):
    ok = True
    for explicit in (False, True):
        for mxl in (1, 2):
            a = state(step(small_model('simple', explicit=explicit, mxl_choice=mxl), nsteps))
            b = state(step(small_model('cartesian', explicit=explicit, mxl_choice=mxl), nsteps))
            d = reldiff(a, b)
            ok &= line('1 reduction simple=realistic, %s, mxl %d (max over fields)'
                       % ('explicit' if explicit else 'implicit', mxl), max(d.values()), 1e-12)
    return ok

if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
