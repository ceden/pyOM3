"""Fast part of the routine suite, all families (README_test_suite.md).
    python3 run_fast.py            all families
    python3 run_fast.py pe         one family
A family whose fast part is not built yet says so and passes."""
import sys, os, time, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = {'pe':  ['reduction', 'identities', 'restart', 'operators', 'golden', 'vadv', 'ab3', 'halo', 'weno'],   # 4 needs mpirun
            'nh':  ['reduction', 'identities', 'restart', 'operators', 'golden', 'halo', 'weno'],  # nh_simple (4 needs mpirun)
            'nhr': ['realistic', 'twin_nh_simple', 'restart', 'halo', 'weno'],                     # nh_realistic (4 needs mpirun; jax_sharding.py in run_slow.py)
            'wim': ['reduction', 'identities', 'restart', 'operators', 'golden',
                    'symmetry', 'sources', 'budget', 'halo'],                           # wim_simple, wim_realistic (wim_idemix: not built)
            'coupled': ['golden', 'forcing', 'identities', 'restart', 'halo', 'wave', 'endtoend']}   # setups/coupled_common + forcing_common (4 needs mpirun; jax_backend.py in run_slow.py)
want = sys.argv[1:] or list(FAMILIES)
t0 = time.time(); ok = True
for fam in want:
    mods = FAMILIES[fam]
    if not mods:
        print('%s: fast part not built (see README_test_suite.md)' % fam); continue
    sys.path.insert(0, os.path.join(HERE, fam))
    for name in mods:
        t = time.time()
        try:
            mod = importlib.import_module(name)
            r = mod.run()
        except ModuleNotFoundError as e:
            # a family whose package needs something this environment lacks
            # (nh_simple: mpi4py and the p3dfft build on a JAX/GPU node): say
            # so and go on with the other families instead of aborting the suite
            print('%s/%s: skipped (%s)' % (fam, name, e)); continue
        ok &= r
        print('   [%s/%s %.0f s]' % (fam, name, time.time() - t))
    sys.path.remove(os.path.join(HERE, fam))           # exactly the family dir (a config may insert more entries in front)
    for name in mods:
        sys.modules.pop(name, None); sys.modules.pop('config', None)
    # test 4 (decomposition) needs a launcher: mpirun on the path or MPIRUN set
    drv = os.path.join(HERE, fam, 'run_decomposition.py')
    if os.path.exists(drv):
        import shutil, subprocess
        launcher = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
        if launcher:
            t = time.time()
            r = subprocess.run([sys.executable, drv], env=dict(os.environ, MPIRUN=' '.join(launcher)))
            ok &= (r.returncode == 0)
            print('   [%s/decomposition %.0f s]' % (fam, time.time() - t))
        else:
            print('4 decomposition: skipped (no mpirun on the path; set MPIRUN)')
print('fast suite: %s in %.0f s' % ('ALL PASS' if ok else 'FAILURES', time.time() - t0))
sys.exit(0 if ok else 1)
