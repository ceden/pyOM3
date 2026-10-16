"""Test 3: 2N steps in one go against N steps, a restart from the saved
fields, and N more steps. Bit for bit. Rigid lid (default, superbee +
TKE closure, Smagorinsky, biharmonic) and periodic in z.

RESTART_FIELDS is the reference list: the prognostic fields, the three
Adams-Bashforth tendency levels, the level pointers and the cold-start
ramp counter (_tau3, _nstep), and with the TKE closure its carried state:
tke, diff (last step's diffusivity, used for the TKE diffusion before it is
recomputed), shear_prod, buoy_prod and stokes_prod (production of the last
step, used in this one). p is diagnostic (recomputed from scratch). The
viscosities and the Smagorinsky/Ducros fields are recomputed before use.

Control: a TKE-closure restart WITHOUT shear_prod and diff must FAIL,
otherwise the test proves nothing about the carried state."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, LES_FIELDS

RESTART_FIELDS = ('u', 'v', 'w', 'b', 'du', 'dv', 'dw', 'db') + LES_FIELDS
CASES = (('lid default', dict()),
         ('lid superbee + tke', dict(b_adv='superbee', les='tke')),
         ('lid smagorinsky', dict(les='smag')),
         ('lid biharmonic', dict(biharmonic=True)),
         ('periodic', dict(kind='periodic')))


def restart_diff(kw, N, drop=()):
    a = state(step(small_model(**kw), 2*N))
    m = step(small_model(**kw), N)
    saved = {f: np.array(getattr(m, f)) for f in RESTART_FIELDS if hasattr(m, f) and f not in drop}
    m2 = small_model(**kw)
    for f, v in saved.items():
        setattr(m2, f, v)
    m2._tau3, m2._nstep = m._tau3, m._nstep
    b = state(step(m2, N))
    return max(float(np.max(np.abs(a[f] - b[f]))) for f in a)


def run(N=5):
    ok = True
    for name, kw in CASES:
        ok &= line('3 restart bit-identical, %s' % name, restart_diff(kw, N), 0.0)
    d = restart_diff(dict(b_adv='superbee', les='tke'), N, drop=('shear_prod', 'diff'))
    ok &= line('3 control: tke restart without shear_prod, diff must FAIL (> 1e-9)', 1.0 if d <= 1e-9 else 0.0, 0.0)
    return ok


if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
