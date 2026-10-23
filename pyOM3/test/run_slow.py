"""Slow part, pe: 7 pyOM2 twins (need PYOM2 and the f2py build), 8 free surface
implicit vs explicit (the harnesses in free_surface/). (9 is now the golden
test of the coupled family in the fast part, coupled/golden.py.)
wim: 8w the sphere acceptance tests (great-circle ray, conservation over 3 d,
closure on the sphere) and 7w the ecwam reference of the Janssen input, both
run where they live, in wim_realistic/development/ (7w needs the f2py build
of development/ecwam_ref, see the README there). 15 wim_simple against the
classical growth and kinematics results and the exact S_nl (wim/benchmarks.py).
J the jax layouts of pe, nhr and wim (jax_sharding.py: numpy reference, 1x1,
1x2, 2x1, 2x2 shards and the 1x1, 2x2 layouts with halo_points = 2, on four
fake CPU devices; about 30 min in all), when jax is installed, and J of the
coupled family (coupled/jax_backend.py: setups/coupled_common and
forcing_common on the jax backend, 1x1 and 2x2 shards, about 3 min)."""
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
if subprocess.run([sys.executable, '-c', 'import jax'], capture_output=True).returncode == 0:
    jobs += [('J jax layouts, %s (jax_sharding.py, halo 1 and 2)' % fam, os.path.join(HERE, fam), ['jax_sharding.py'])
             for fam in ('pe', 'nhr', 'wim')]
    jobs += [('J jax backend, coupled (jax_backend.py: the shared coupling/forcing code, 1x1 and 2x2 shards)',
              os.path.join(HERE, 'coupled'), ['jax_backend.py']),
             ('22d run.py on the jax backend, ocean alone, two segments on 1x1 and 2x2 shards (endtoend.py --jax)',
              os.path.join(HERE, 'coupled'), ['endtoend.py', '--jax'])]
else:
    print('J jax layouts skipped: jax not installed')
ok = True
for name, cwd, cmd in jobs:
    print('=== ' + name, flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env=env)
    ok &= (r.returncode == 0)
print('slow suite: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
