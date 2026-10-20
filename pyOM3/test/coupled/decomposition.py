"""Test 4 for the coupling code: one PE against a decomposition, the forced
and the coupled ocean. The shared code's block logic is what this covers:
the forcing object built on each rank's block (local_block, geom), the
pad of the forcing fields, the coupler-side fill of the wave fields, the
collective reductions of the diagnostics and the gathered restart/means
paths are not run here (coupler_io, test 20 covers one rank).
Run under MPI, from this directory:

    mpirun -np 1 python3 decomposition.py serial   -> decomp_ref_<case>.npz
    mpirun -np 2 python3 decomposition.py 1 2      -> y-split, exact
    mpirun -np 2 python3 decomposition.py 2 1      -> x-split, 1e-10 (the CG's global sums, amplified by the closure)
    mpirun -np 4 python3 decomposition.py 2 2

Each decomposed run checks every case against its serial reference, and
the forced and the coupled default once more with halo_points = 2 (test 16
under MPI, same references). Without MPI (one rank without arguments) it
says so."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_ocean, step, gathered_state, line, HERE
CASES = (('forced', dict(coupled=False)),
         ('forced_relwind', dict(coupled=False, relwind=True, ice=0.3, sw_bottom=True, average='6h')),
         ('coupled', dict(coupled=True)),
         ('coupled_full', dict(coupled=True, direct=True, shear=True, mxl_choice=2)),
         ('coupled_stress', dict(coupled=True, stress_work=True, ke_budget=0)))
REF = lambda name: os.path.join(HERE, 'decomp_ref_%s.npz' % name)


def run(argv):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if len(argv) > 1 and argv[1] == 'serial':
        for name, kw in CASES:
            st = gathered_state(step(small_ocean(comm=comm, tag='d_' + name, **kw), 8))
            np.savez(REF(name), **st)
        print('4 decomposition: serial references written (%s)' % ', '.join(n for n, _ in CASES))
        return True
    if len(argv) < 3:
        print('4 decomposition: skipped (run under mpirun with NPX NPY, see docstring)'); return True
    npx, npy = int(argv[1]), int(argv[2]); ok = True
    for (name, kw), halo in [(c, 1) for c in CASES] + [(CASES[0], 2), (CASES[2], 2)]:
        st = gathered_state(step(small_ocean(comm=comm, n_pes_x=npx, n_pes_y=npy, halo=halo,
                                             tag='d_' + name, **kw), 8))
        if comm.Get_rank() != 0:
            continue
        if not os.path.exists(REF(name)):
            print('4 decomposition %s: no serial reference (run "serial" first)' % name); ok = False; continue
        ref = np.load(REF(name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in st)
        ok &= line('4 decomposition %dx%d vs 1 PE, %s%s' % (npx, npy, name, ', halo 2' if halo > 1 else ''), d,
                   0.0 if npx == 1 else 1e-10)   # x-split: the CG's global sums reduce in rank order (pe: 5e-13 in u; here the
                                                    # TKE closure amplifies it to 1.5e-11 in tke, K_diss_v after 8 steps)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv) else 1)
