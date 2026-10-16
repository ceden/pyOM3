"""Test 4: one PE against a decomposition, implicit AND explicit free surface,
the implicit path with vadv_sub_cycling = 2 (vadv_split.py), and that with AB3.
Run under MPI, from this directory:

    mpirun -np 1 python3 decomposition.py serial       -> decomp_ref_implicit.npz, decomp_ref_explicit.npz
    mpirun -np 2 python3 decomposition.py 1 2          -> y-split, exact
    mpirun -np 2 python3 decomposition.py 2 1          -> x-split, 1e-12 (the CG's global sums reduce in rank order)
    mpirun -np 4 python3 decomposition.py 2 2

Each decomposed run checks both paths against their serial reference. The
explicit path exchanges U, V, p_s every sub step and carries U_bt, so it has
more layout-sensitive pieces than the implicit one. Without MPI (one rank
without arguments) it says so."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, gathered_state, line, HERE
AB2 = (1.6, -0.6, 0.0)
PATHS = ((False, 'implicit', 0, AB2), (True, 'explicit', 0, AB2), (False, 'split', 2, AB2),   # split: vadv_sub_cycling = 2
         (False, 'ab3', 2, 'ab3'))                                                          # split + ramped AB3
REF = lambda name: os.path.join(HERE, 'decomp_ref_%s.npz' % name)


def run(argv):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if len(argv) > 1 and argv[1] == 'serial':
        for explicit, name, vadv, ab in PATHS:
            st = gathered_state(step(small_model(comm=comm, explicit=explicit, vadv=vadv), 8, ab))
            np.savez(REF(name), **st)
        print('4 decomposition: serial references written (implicit, explicit, split, ab3)')
        return True
    if len(argv) < 3:
        print('4 decomposition: skipped (run under mpirun with NPX NPY, see docstring)'); return True
    npx, npy = int(argv[1]), int(argv[2]); ok = True
    for explicit, name, vadv, ab in PATHS:
        st = gathered_state(step(small_model(comm=comm, n_pes_x=npx, n_pes_y=npy, explicit=explicit, vadv=vadv), 8, ab))
        if comm.Get_rank() != 0:
            continue
        if not os.path.exists(REF(name)):
            print('4 decomposition %s: no serial reference (run "serial" first)' % name); ok = False; continue
        ref = np.load(REF(name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in st)
        # explicit path: halo exchanges only, no global reduction -> exact in every
        # layout; implicit: the CG's dot products reduce in rank order -> x-splits 1e-12
        ok &= line('4 decomposition %dx%d vs 1 PE, %s' % (npx, npy, name), d,
                   0.0 if (npx == 1 or explicit) else 1e-12)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv) else 1)
