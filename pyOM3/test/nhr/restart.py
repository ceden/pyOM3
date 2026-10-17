"""Test 3 (nhr): 2N steps in one go against N steps, a restart from the saved
fields, and N more steps. Bit for bit. The restart list is the pe one plus
the w tendency levels dw and the non-hydrostatic pressure p_nh (the first
guess of the 3-D solve: without it the iteration takes another path and
the result differs at the solver tolerance)."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, FIELDS
import pyOM3 as OM

RESTART_FIELDS = ('u', 'v', 'w', 'temp', 'salt', 'rho', 'p_h', 'tke', 'K_diss_v', 'du', 'dv',
                  'dtemp', 'dsalt', 'p_s', 'du_cor', 'dv_cor', 'U_bt', 'V_bt', 'tau_modulus', 'taux', 'tauy',
                  'dw', 'p_nh')

def run(N=6):
    ok = True
    for explicit in (False, True):
        kw = dict(explicit=explicit)
        a = state(step(small_model(**kw), 2*N))
        m = step(small_model(**kw), N)
        saved = {f: np.array(getattr(m, f)) for f in RESTART_FIELDS if hasattr(m, f)}
        saved['_tau3'] = m._tau3
        m2 = small_model(**kw)
        for f, v in saved.items():
            setattr(m2, f, v if f == '_tau3' else OM.np.asarray(v))   # back to the backend's arrays (JAX)
        b = state(step(m2, N))
        d = max(float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS + ('p_nh',))
        ok &= line('3 restart bit-identical, %s' % ('explicit' if explicit else 'implicit'), d, 0.0)
        # control: without p_nh in the restart the 3-D solve starts from zero
        # and the run is no longer bit-identical
        m3 = small_model(**kw)
        for f, v in saved.items():
            if f != 'p_nh':
                setattr(m3, f, v if f == '_tau3' else OM.np.asarray(v))
        c = state(step(m3, N))
        d = max(float(np.max(np.abs(a[f] - c[f]))) for f in FIELDS)
        ok &= line('3 control: restart without p_nh must differ (> 0)', 0.0 if d > 0.0 else 1.0, 0.0)
    return ok

if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
