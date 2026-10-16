"""Test 3: 2N steps in one go against N steps, a restart from the saved
fields, and N more steps. Bit for bit."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, FIELDS
import pyOM3 as OM

RESTART_FIELDS = ('u', 'v', 'w', 'temp', 'salt', 'rho', 'p_h', 'tke', 'K_diss_v', 'du', 'dv',
                  'dtemp', 'dsalt', 'p_s', 'du_cor', 'dv_cor', 'U_bt', 'V_bt', 'tau_modulus', 'taux', 'tauy')

def run(N=6):
    ok = True
    for explicit in (False, True):
        for mxl in (1, 2):
            kw = dict(explicit=explicit, mxl_choice=mxl)
            a = state(step(small_model(**kw), 2*N))
            m = step(small_model(**kw), N)
            saved = {f: np.array(getattr(m, f)) for f in RESTART_FIELDS if hasattr(m, f)}
            saved['_tau3'] = m._tau3
            m2 = small_model(**kw)
            for f, v in saved.items():
                setattr(m2, f, v if f == '_tau3' else OM.np.asarray(v))   # back to the backend's arrays (JAX)
            b = state(step(m2, N))
            d = max(float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS)
            ok &= line('3 restart bit-identical, %s, mxl %d' % ('explicit' if explicit else 'implicit', mxl), d, 0.0)
    return ok

if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
