"""Test 1: wim_simple against wim_realistic on a uniform Cartesian grid
(coord_degree=False: cost = cosu = 1, tantr = 0, every spacing the constant).
wim_simple has no metrics to get wrong, so this is the test that catches
metric errors in wim_realistic. One forced run per option set in use: the
energy fixer with open coasts and a current, wall reflection, the plain
scheme without a current, the input-based tail flux; and the default again
with the fortran propagation kernels when they are built."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, reldiff, line

CASES = (('open coasts, energy fixer, current', {}),
         ('wall reflection, closed coasts',     dict(open_coasts=False, reflection=True)),
         ('no current, no energy fixer',        dict(current=False, energy_consistent=False)),
         ('tail_flux input',                    dict(tail_flux='input')))


def fortran_built():
    try:
        from pyOM3.wim_simple.Propagation import Propagation      # noqa
        return True
    except ImportError:
        return False


def run(nsteps=12):
    ok = True
    cases = list(CASES)
    if fortran_built():
        cases.append(('fortran propagation kernels', dict(backend='fortran')))
    else:
        print('%-56s skipped (wim_simple/Propagation not built)' % '1 simple=realistic, fortran propagation kernels')
    for name, kw in cases:
        a = state(step(small_model('simple', **kw), nsteps))
        b = state(step(small_model('cartesian', **kw), nsteps))
        ok &= line('1 simple=realistic, %s' % name, max(reldiff(a, b).values()), 1e-12)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
