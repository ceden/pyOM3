"""Test 4: one PE against a decomposition. p3dfft decomposes y and z
(nh_simple: n_pes_x splits y, n_pes_y splits z, x is always local), so
the layouts are 1x2 (z split, the lids and the TKE surface point on
different ranks), 2x1 (y split) and 2x2. Paths: rigid lid default,
periodic in z, the LES path (superbee + TKE closure: halo-sensitive
limiter slopes, viscosities and production terms), Smagorinsky with the
harmonic friction (the closure reads the halos after the friction
increment). Run under MPI, from this directory:

    mpirun -np 1 python3 decomposition.py serial       -> decomp_ref_<path>.npz
    mpirun -np 2 python3 decomposition.py 1 2          -> z-split
    mpirun -np 2 python3 decomposition.py 2 1          -> y-split
    mpirun -np 4 python3 decomposition.py 2 2

Exact in every layout: p3dfft transposes the data so that each 1D
transform runs over a complete line, the same line in every layout, and
the tridiagonal solve is local in z in spectral space; there is no global
reduction. Any difference is therefore a halo that is read before it is
exchanged (what the LES closure after the friction increment looked like).
Without MPI (one rank without arguments) it says so."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, gathered_state, line, HERE
PATHS = (('lid', dict()), ('periodic', dict(kind='periodic')),
         ('les', dict(b_adv='superbee', les='tke')), ('smag', dict(les='smag')))
REF = lambda name: os.path.join(HERE, 'decomp_ref_%s.npz' % name)


def run(argv):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if len(argv) > 1 and argv[1] == 'serial':
        for name, kw in PATHS:
            st = gathered_state(step(small_model(comm=comm, **kw), 8))
            np.savez(REF(name), **st)
        print('4 decomposition: serial references written (%s)' % ', '.join(n for n, _ in PATHS))
        return True
    if len(argv) < 3:
        print('4 decomposition: skipped (run under mpirun with NPX NPY, see docstring)'); return True
    npx, npy = int(argv[1]), int(argv[2]); ok = True
    for name, kw in PATHS:
        st = gathered_state(step(small_model(comm=comm, n_pes_x=npx, n_pes_y=npy, **kw), 8))
        if comm.Get_rank() != 0:
            continue
        if not os.path.exists(REF(name)):
            print('4 decomposition %s: no serial reference (run "serial" first)' % name); ok = False; continue
        ref = np.load(REF(name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in st)
        ok &= line('4 decomposition %dx%d (y x z split) vs 1 PE, %s' % (npx, npy, name), d, 0.0)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv) else 1)
