"""Test 4: one PE against a decomposition. Run under MPI, from this directory:

    mpirun -np 1 python3 decomposition.py serial       -> decomp_ref_<case>.npz
    mpirun -np 2 python3 decomposition.py 1 2          -> y-split
    mpirun -np 2 python3 decomposition.py 2 1          -> x-split
    mpirun -np 4 python3 decomposition.py 2 2

The wave model has no global solver: halo exchanges only, so EVERY layout must
be exact in the state and the exchange fields. The outflow totals are one
Allreduce over the PEs, whose summation order follows the layout: 1e-14.
Cases: the default (open coasts, energy fixer, current), wall reflection, and
input_scheme janssen. The island of the small basin sits across both seams of
the 2 x 2 layout, so the coastal-face ownership of open_tables is exercised."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, gathered_state, line, HERE
CASES = (('default', {}), ('reflection', dict(open_coasts=False, reflection=True)), ('janssen', dict(janssen=True)))
REF = lambda name: os.path.join(HERE, 'decomp_ref_%s.npz' % name)
LOSS = ('loss_A', 'loss_E', 'loss_Eabs')


def run(argv):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if len(argv) > 1 and argv[1] == 'serial':
        for name, kw in CASES:
            np.savez(REF(name), **gathered_state(step(small_model(comm=comm, **kw), 8)))
        print('4 decomposition: serial references written (%s)' % ', '.join(n for n, _ in CASES))
        return True
    if len(argv) < 3:
        print('4 decomposition: skipped (run under mpirun with NPX NPY, see docstring)'); return True
    npx, npy = int(argv[1]), int(argv[2]); ok = True
    # the three cases with the default halo, then the default case with
    # halo_points = 2 (test 16 under MPI) against the same reference; A carries
    # its frequency ghosts, so it is compared on the interior bins and the
    # outermost ghost bins (test 16)
    for name, kw, halo in [c + (1,) for c in CASES] + [CASES[0] + (2,)]:
        st = gathered_state(step(small_model(comm=comm, n_pes_x=npx, n_pes_y=npy, halo=halo, **kw), 8))
        if comm.Get_rank() != 0:
            continue
        if not os.path.exists(REF(name)):
            print('4 decomposition %s: no serial reference (run "serial" first)' % name); ok = False; continue
        ref = dict(np.load(REF(name)))
        if halo > 1:
            st = dict(st); st['A_fghost'] = st['A'][:, [halo - 1, -halo]]; st['A'] = st['A'][:, halo:-halo]
            ref['A_fghost'] = ref['A'][:, [0, -1]]; ref['A'] = ref['A'][:, 1:-1]
        rel = lambda k: float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300))
        ok &= line('4 decomposition %dx%d vs 1 PE, %s%s' % (npx, npy, name, ', halo 2' if halo > 1 else ''),
                   max(rel(k) for k in st if k not in LOSS), 0.0)
        if kw.get('open_coasts', True):
            ok &= line('4 ... outflow totals (one Allreduce)', max(rel(k) for k in LOSS), 1e-14)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv) else 1)
