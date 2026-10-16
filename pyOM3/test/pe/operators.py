"""Test 5: operator unit tests on a model object with a prescribed state.

  surface TKE flux vs its definition |tau|^1.5                       1e-8
  bottom friction acts on the lowest wet cell only, -dt r u          1e-15
  (mixing length vs the independent NEMO-order implementation: the
   script of the mxl_choice work, run it from there)
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, line
import pyOM3 as OM

def run():
    ok = True
    m = small_model(); step(m, 1); h = m.halo
    forc = np.asarray(m._surface_forcing())[h:-h, h:-h]
    ref = np.hypot(np.asarray(m.taux), np.asarray(m.tauy))[h:-h, h:-h]**1.5
    ok &= line('5 surface TKE flux = |tau|^1.5', float(np.max(np.abs(forc - ref)))/max(ref.max(), 1e-300), 1e-8)
    if hasattr(m, 'bottom_friction'):
        r = 1e-4; m.enable_bottom_friction = True; m.r_bot = r
        u0 = np.asarray(m.u).copy(); mu = np.asarray(m.maskU)
        un, vn = m.bottom_friction(m.u, m.v, m.maskU, m.maskV); un = np.asarray(un)
        below = np.concatenate([np.zeros_like(mu[:1]), mu[:-1]], axis=0)
        bot = mu*(1 - below)                                  # lowest wet cell of each column
        expect = u0*(1 - bot*(1 - 1/(1 + m.dt*r)))
        ok &= line('5 bottom friction: lowest wet cell only, u/(1+dt r)', float(np.max(np.abs(un - expect))), 1e-15)
        m.enable_bottom_friction = False
    else:
        print('%-52s skipped (no bottom_friction in this tree)' % '5 bottom friction')
    return ok

if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
