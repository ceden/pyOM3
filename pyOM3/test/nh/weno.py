"""Test 17: WENO5 buoyancy advection (nh_simple/advection.py
buoyancy_advection_weno, pyOM3/advection_weno.py) on the small
configuration, halo_points = 3 (the scheme refuses less), rigid lids and
periodic in z.

  17a  a uniform buoyancy has no advective tendency (the projected
       velocity is divergence free, the face value of a constant is the
       constant whatever the order chosen at a face)
  17b  the scheme is conservative: the advective tendency sums to zero
       over the interior (periodic sides, zero flux through the lids)
  17c  the result does not depend on the halo width: halo 4 against
       halo 3 after N steps, bit for bit, lid and periodic
  17d  N steps with the TKE closure taking the implicit buoyancy flux of
       the scheme (les_mix_b False, as with superbee) stay finite

The reference scale of the buoyancy is set (weno_ref_b = 1e-4, the size
of the anomaly of the configuration); with the default 1 the weights
would be the linear fifth-order ones. Cannot run without the p3dfft
module; the code path was exercised with a stand-in for the boundary
class when it was written (Oct 18 2026), the suite run is the real test.
Decomposition under MPI: run_decomposition.py does not cover it yet."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, reldiff, OM

WENO = dict(b_adv='weno', halo=3, weno_ref_b=1e-4)


def run(N=6):
    ok = True
    for kind in ('lid', 'periodic'):
        m = small_model(kind=kind, **WENO)
        h = m.halo; sl = slice(h, -h)
        step(m, 3)                                          # a nontrivial projected velocity
        b_save = m.b
        m.b = m.apply_bc(OM.np.full_like(m.b, 3e-4))
        db = np.asarray(m.buoyancy_advection_weno())[sl, sl, sl]
        ok &= line('17a uniform b, |db| per step, %s' % kind, float(np.max(np.abs(db)))*m.dt, 1e-16)
        m.b = b_save
        db = np.asarray(m.buoyancy_advection_weno())[sl, sl, sl]
        ok &= line('17b sum of the advective tendency / sum |tendency|, %s' % kind,
                   abs(float(np.sum(db)))/float(np.sum(np.abs(db))), 1e-13)
        del m
        ref = state(step(small_model(kind=kind, **WENO), N));
        kw = dict(WENO); kw['halo'] = 4
        d = max(reldiff(state(step(small_model(kind=kind, **kw), N)), ref).values())
        ok &= line('17c halo 4 = halo 3, WENO5, %s, %d steps' % (kind, N), d, 0.0)
    m = step(small_model(kind='lid', les='tke', **WENO), 20)
    st = state(m); st['tke'] = np.asarray(m.tke)
    d = 0.0 if all(np.all(np.isfinite(v)) for v in st.values()) else np.inf
    ok &= line('17d 20 steps with the TKE closure on the implicit flux finite (0 = yes)', d, 0.0)
    del m
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
