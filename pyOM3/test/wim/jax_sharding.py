"""JAX sharding check for wim_simple and wim_realistic (no MPI involved).

    python3 jax_sharding.py                 # everything, on 4 fake CPU devices
    python3 jax_sharding.py realistic 2 2   # one run, needs the references

Everything: for each kind ('simple', 'realistic', 'cartesian') first the
numpy reference (PYOM3_BACKEND=numpy), then jax 1x1, 1x2, 2x1 and 2x2
shards against it, each in its own process because the backend is chosen
when pyOM3 is imported. The devices come from
XLA_FLAGS=--xla_force_host_platform_device_count=4, so this runs on a
laptop without MPI. The configurations are those of test 4
(decomposition.py): default (island, current, open coasts, energy fixer,
sources), reflection (closed coasts, specular reflection: the vectorised
reflect_flux of the JAX path against the loop), janssen (wim_realistic
only: the Janssen input with its state tau_w through the pure source
hooks), plus tail_flux='input' and the adiabatic path (sources=False).
8 steps, the fields of config.FIELDS on the global interior (config.state
unpads the inline shard halos).

Tolerance 1e-12: jax and numpy differ by round-off (XLA fusion), about
1e-15 here; decomposed shards differ from 1x1 by the per-shard compilation.
A layout, halo or seam error shows as O(1)."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

CASES = {'simple':    (('default', {}), ('reflection', dict(open_coasts=False, reflection=True)),
                       ('tail_input', dict(tail_flux='input')), ('adiabatic', dict(sources=False))),
         'realistic': (('default', {}), ('reflection', dict(open_coasts=False, reflection=True)),
                       ('janssen', dict(janssen=True)), ('tail_input', dict(tail_flux='input')),
                       ('adiabatic', dict(sources=False))),
         'cartesian': (('default', {}), ('janssen', dict(janssen=True)))}
REF = lambda kind, name: os.path.join(HERE, 'jax_ref_%s_%s.npz' % (kind, name))
TOL = 1e-12


def run(kind, npx, npy, write_ref=False):
    import io, contextlib
    from config import small_model, step, state, line, OM   # config puts the tree on sys.path
    if write_ref and OM.jax or not write_ref and not OM.jax:
        print('   wrong backend (jax %s), set PYOM3_BACKEND' % bool(OM.jax)); return False
    ok = True
    for name, kw in CASES[kind]:
        with contextlib.redirect_stdout(io.StringIO()):
            st = state(step(small_model(kind, n_pes_x=npx, n_pes_y=npy, **kw), 8))
        if write_ref:
            np.savez(REF(kind, name), **st); continue
        if not os.path.exists(REF(kind, name)):
            print('J wim_%s jax: no numpy reference' % kind); return False
        ref = np.load(REF(kind, name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in st)
        ok &= line('J wim %s jax %dx%d vs numpy, %s' % (kind, npx, npy, name), d, TOL)
    if write_ref:
        print('J wim %s: numpy references written' % kind)
    return ok


def run_all():
    import tempfile
    # the persistent compilation cache: every model instance traces its
    # jitted step anew (self is a static argument), the cache makes the 60
    # runs share the compiled executables of equal configurations
    cache = os.environ.get('JAX_COMPILATION_CACHE_DIR') or tempfile.mkdtemp(prefix='wim_jax_cache_')
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4',
               JAX_COMPILATION_CACHE_DIR=cache, JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0.5')
    ok = True
    for kind in ('simple', 'realistic', 'cartesian'):
        for backend, args in (('numpy', ('ref',)), ('jax', ('1', '1')), ('jax', ('1', '2')),
                              ('jax', ('2', '1')), ('jax', ('2', '2'))):
            env['PYOM3_BACKEND'] = backend
            cmd = [sys.executable, os.path.abspath(__file__), kind] + list(args)
            r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
            lines = [l for l in r.stdout.splitlines() if l.startswith('J ')]
            print('\n'.join(lines))
            if r.returncode != 0:
                ok = False
                print('   %s %s %s failed (exit %d):' % (backend, kind, ' '.join(args), r.returncode))
                print('\n'.join('      ' + l for l in (r.stderr or r.stdout).splitlines()[-12:]))
            ok &= not any(l.endswith('FAIL') for l in lines)
    for kind in CASES:                                  # the references are run products
        for name, _ in CASES[kind]:
            if os.path.exists(REF(kind, name)): os.remove(REF(kind, name))
    print('jax sharding: %s' % ('ALL PASS' if ok else 'FAILURES'))
    return ok


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(0 if run_all() else 1)
    kind = sys.argv[1]
    if sys.argv[2] == 'ref':
        sys.exit(0 if run(kind, 1, 1, write_ref=True) else 1)
    sys.exit(0 if run(kind, int(sys.argv[2]), int(sys.argv[3])) else 1)
