"""Slow part, pe: 7 pyOM2 twins (need PYOM2 and the f2py build), 8 free surface
implicit vs explicit (the harnesses in free_surface/), 9 coupled smoke test
(not written: needs the coupler and mpirun, see README_test_suite.md).
wim: 8w the sphere acceptance tests (great-circle ray, conservation over 3 d,
closure on the sphere) and 7w the ecwam reference of the Janssen input, both
run where they live, in wim_realistic/development/ (7w needs the f2py build
of development/ecwam_ref, see the README there). 15 wim_simple against the
classical growth and kinematics results and the exact S_nl (wim/benchmarks.py)."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.path.join(HERE, 'pe', 'free_surface'); TW = os.path.join(HERE, 'pe', 'twins')
env = dict(os.environ, PYOM3=os.path.abspath(os.path.join(HERE, '..', '..')))
jobs = [('8 free surface: reduction + bump harness', FS, ['pe_reduction_expl.py', 'bump']),
        ('8 free surface: Munk implicit vs explicit', FS, ['munk_free_surface_test.py', '0.25'])]
if os.environ.get('PYOM2'):
    jobs += [('7 twin: biharmonic operator', TW, ['biharm_twin.py']),
             ('7 twin: Munk gyre, biharmonic, variable grid', TW, ['munk_biharm_twin.py', '1'])]
else:
    print('7 pyOM2 twins skipped: set PYOM2=<pyOM2 clone with py_src/pyOM_code.so>')
WD = os.path.join(HERE, '..', 'wim_realistic', 'development')
jobs += [('8w wim sphere: conservation, great-circle ray, uniform state, closure', WD, ['sphere_test.py'])]
import glob
if glob.glob(os.path.join(WD, 'ecwam_ref', '*.so')) + glob.glob(os.path.join(WD, '*.so')):
    jobs += [('7w wim twin: Janssen input against the ecwam routines', WD, ['janssen_ref_test.py'])]
else:
    print('7w ecwam twin skipped: build wim_realistic/development/ecwam_ref (make ECWAM=<ecwam clone>)')
jobs += [('15 wim_simple benchmarks: PM growth, JONSWAP fetch law, swell c_g, Snell, DIA vs exact S_nl',
          os.path.join(HERE, 'wim'), ['benchmarks.py'])]
ok = True
for name, cwd, cmd in jobs:
    print('=== ' + name, flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env=env)
    ok &= (r.returncode == 0)
print('slow suite: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
