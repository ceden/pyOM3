"""Test 4 (nhr): nh_realistic, one PE against a decomposition, implicit AND explicit free surface,
the implicit path with vadv_sub_cycling = 2 (vadv_split.py), and that with AB3.
Run under MPI, from this directory:

    mpirun -np 1 python3 decomposition.py serial       -> decomp_ref_implicit.npz, decomp_ref_explicit.npz
    mpirun -np 2 python3 decomposition.py 1 2          -> y-split, 1e-9 (the global sums of the 3-D CG reduce in rank order)
    mpirun -np 2 python3 decomposition.py 2 1          -> x-split, 1e-9
    mpirun -np 4 python3 decomposition.py 2 2

Each decomposed run checks the paths against their serial reference. The
non-hydrostatic solve adds global sums (dot products, gauge means) on every
path, so the x-split tolerance applies to the explicit path too. Without MPI
(one rank without arguments) it says so."""
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
            st = gathered_state(step(small_model(eps_nh=1e-16, comm=comm, explicit=explicit, vadv=vadv), 8, ab))
            np.savez(REF(name), **st)
        print('4 decomposition: serial references written (implicit, explicit, split, ab3)')
        return True
    if len(argv) < 3:
        print('4 decomposition: skipped (run under mpirun with NPX NPY, see docstring)'); return True
    npx, npy = int(argv[1]), int(argv[2]); ok = True
    # the four paths with the default halo, then the implicit and the explicit
    # path with halo_points = 2 (test 16 under MPI) against the same references
    for explicit, name, vadv, ab, halo in [p + (1,) for p in PATHS] + [p + (2,) for p in PATHS[:2]]:
        st = gathered_state(step(small_model(eps_nh=1e-16, comm=comm, n_pes_x=npx, n_pes_y=npy, explicit=explicit,
                                             vadv=vadv, halo=halo), 8, ab))
        if comm.Get_rank() != 0:
            continue
        if not os.path.exists(REF(name)):
            print('4 decomposition %s: no serial reference (run "serial" first)' % name); ok = False; continue
        ref = np.load(REF(name))
        rel = lambda k: float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300))
        d = max(rel(k) for k in st if k != 'p_nh')
        # The global sums of the 3-D CG (dot products, gauge means) reduce in
        # rank order, so no layout is exact, on the explicit path too. The
        # stop criterion is the remaining divergence, which the vertical
        # part of p_nh fixes; the horizontal structure of p_nh is weakly
        # determined (vertical coefficients 3e4 to 2e7 times the horizontal ones in
        # this hydrostatic-aspect box) and still moves when the criterion is
        # met, which is why the runs here solve to eps_nh = 1e-16 (300
        # iterations): the fields then agree to 1e-10 (tke, K_diss_v), p_nh to 1e-6.
        # With the default 1e-12 the fields differ by 1e-8 between layouts.
        ok &= line('4 decomposition %dx%d vs 1 PE, %s%s' % (npx, npy, name, ', halo 2' if halo > 1 else ''), d, 1e-9)
        ok &= line('4 ... p_nh (weakly determined, see the comment)', rel('p_nh'), 1e-5)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv) else 1)
