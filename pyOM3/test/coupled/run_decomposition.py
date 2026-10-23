"""Test 4 in one go: the serial references and the 1x2, 2x1, 2x2 decompositions,
forced and coupled ocean, each as its own mpirun. Prints the result
lines of the four runs and a summary.

    python3 run_decomposition.py            # uses mpirun
    MPIRUN="srun -n" python3 run_decomposition.py
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
MPIRUN = os.environ.get('MPIRUN', 'mpirun -np').split()
RUNS = (('serial',), ('1', '2'), ('2', '1'), ('2', '2'))
NP = (1, 2, 2, 4)
ok = True
for n, args in zip(NP, RUNS):
    cmd = MPIRUN + [str(n), sys.executable, os.path.join(HERE, 'decomposition.py')] + list(args)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith('4 ')]
    for l in lines:
        print(l)
    if r.returncode != 0:
        ok = False
        print('   mpirun %d %s failed (exit %d):' % (n, ' '.join(args), r.returncode))
        print('\n'.join('      ' + l for l in (r.stderr or r.stdout).splitlines()[-12:]))
    ok &= not any(l.endswith('FAIL') for l in lines)
import glob
for f in glob.glob(os.path.join(HERE, 'decomp_ref_*.npz')):    # the references are run products
    os.remove(f)
print('decomposition: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
