"""Helpers shared by every test family (pe/, nh/, wim/, coupled/):
one-line reporting, relative differences, order-independent checksums."""
import numpy as np


def checksum(st):
    return {k: float(np.sum(np.abs(v))) + float(np.sum(v**2)) for k, v in st.items()}


def reldiff(a, b):
    return {k: float(np.max(np.abs(a[k] - b[k]))/max(np.max(np.abs(b[k])), 1e-300)) for k in a}


def line(name, value, tol, unit=''):
    ok = value <= tol
    print('%-56s %10.2e  (tol %8.1e%s)  %s' % (name, value, tol, unit, 'PASS' if ok else 'FAIL'))
    return ok
