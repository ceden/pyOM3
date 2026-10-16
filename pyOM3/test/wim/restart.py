"""Test 3: 2N steps in one go against N steps, a restart from the saved
fields, and N more steps. Bit for bit, the exchange fields (stress, Stokes
drift) included.

RESTART_FIELDS is the reference list for every wave restart. A is saved WITH
its frequency ghost bins: imphftail writes the f^-5 continuation there and
the next step's frequency advection and the tail stress read it. The control
saves A without them, as a file holding the interior bins would, and must
FAIL, otherwise the test says nothing about that slab."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, state, line, FIELDS

RESTART_FIELDS = ('A', 'dA', 'loss_A', 'loss_E', 'loss_Eabs', '_lossA_lv', '_lossE_lv', '_lossO_lv',
                  'tau_w',                          # input_scheme janssen: last step's wave stress (z0 follows from it)
                  '_tau3', '_nstep')                # time-level pointers and the Euler-start counter

CASES = (('realistic', {}), ('realistic, janssen', dict(janssen=True)),
         ('realistic, tail_flux input', dict(tail_flux='input')),
         ('realistic, reflection', dict(open_coasts=False, reflection=True)),
         ('simple', dict(kind='simple')))


def _restart_run(kw, N, fields=RESTART_FIELDS, drop_fghost=False):
    m = step(small_model(**kw), N)
    saved = {f: np.array(getattr(m, f)) if isinstance(getattr(m, f), np.ndarray) else getattr(m, f)
             for f in fields if hasattr(m, f)}
    if drop_fghost:
        h = m.halo; a = np.array(saved['A']); a[:, :h] = 0.0; a[:, -h:] = 0.0; saved['A'] = a
    m2 = small_model(**kw)
    for f, v in saved.items():
        setattr(m2, f, v.copy() if isinstance(v, np.ndarray) else v)
    return state(step(m2, N))


def run(N=6):
    ok = True
    for name, kw in CASES:
        a = state(step(small_model(**kw), 2*N))
        b = _restart_run(kw, N)
        d = max(float(np.max(np.abs(a[f] - b[f]))) for f in a)
        ok &= line('3 restart bit-identical, %s' % name, d, 0.0)
    a = state(step(small_model(), 2*N)); b = _restart_run({}, N, drop_fghost=True)
    d = max(float(np.max(np.abs(a[f] - b[f]))/max(np.max(np.abs(a[f])), 1e-300)) for f in a)
    good = d > 1e-12
    print('%-56s %10.2e  (must differ)        %s' % ('3 control: A without its frequency ghosts', d, 'PASS' if good else 'FAIL'))
    return ok & good


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
