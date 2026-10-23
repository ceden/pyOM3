"""JAX backend check of the shared coupling and forcing code (no MPI).

    python3 jax_backend.py                 # everything, on 4 fake CPU devices
    python3 jax_backend.py ref             # the numpy references (PYOM3_BACKEND=numpy)
    python3 jax_backend.py 1 1             # one jax run against them
    python3 jax_backend.py 2 2             # the same on 2x2 shards

Everything: first the numpy references, then jax 1x1 and 2x2 shards against
them, each in its own process because the backend is chosen when pyOM3 is
imported (XLA_FLAGS=--xla_force_host_platform_device_count=4 gives the four
devices). The cases of test 9 (golden.py), 8 steps, the fields of
config.FIELDS and the coupling fields on the global interior.

THIS IS THE REFERENCE FOR THE JAXIFICATION of setups/coupled_common and
setups/forcing_common: the numpy path defines the answer, the jax path must
reproduce it to round-off. A case that does not run on the jax backend
prints its last traceback lines and counts as a failure. Tolerance 1e-10:
XLA fusion changes the round-off, and here it shows in the TKE closure
(tke, K_diss_v up to 7e-11 after 8 steps) and in the small restoring
salt flux; the pe family alone stays below 1e-12."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
REF = lambda name: os.path.join(HERE, 'jax_ref_%s.npz' % name)
TOL = 1e-10


def run(npx, npy, write_ref=False):
    import io, contextlib, traceback
    from config import small_ocean, step, state, diagnostics, line, OM
    from golden import CASES
    if write_ref and OM.jax or not write_ref and not OM.jax:
        print('   wrong backend (jax %s), set PYOM3_BACKEND' % bool(OM.jax)); return False
    ok = True
    for name, kw in CASES:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = step(small_ocean(n_pes_x=npx, n_pes_y=npy, tag='j_' + name, **kw), 8)
                st = state(m)
                st.update({'diag_' + k: v for k, v in diagnostics(m).items()})
        except Exception:
            tb = traceback.format_exc().splitlines()
            print('J coupled jax %dx%d, %s: does not run on the jax backend' % (npx, npy, name))
            print('\n'.join('      ' + l for l in tb[-6:]))
            ok = False; continue
        if write_ref:
            np.savez(REF(name), **st); continue
        if not os.path.exists(REF(name)):
            print('J coupled jax: no numpy reference for %s' % name); ok = False; continue
        ref = np.load(REF(name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300))
                for k in ref if not k.startswith('diag_'))
        ok &= line('J coupled jax %dx%d vs numpy, %s' % (npx, npy, name), d, TOL)
        # the diagnostics; the vortex work is a round-off monitor (1e-14 of
        # the stress work), so every scalar is compared on the stress-work scale
        scale = float(np.abs(ref['diag_stress_work'][0]))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), scale, 1e-300))
                for k in ref if k.startswith('diag_'))
        ok &= line('J coupled jax %dx%d vs numpy, %s, diagnostics' % (npx, npy, name), d, TOL)
    # the wave side: exp_wave on wim_realistic, both grids, one step
    from config import small_wave, wave_state
    for c in (1, 2):
        name = 'wave_c%d' % c
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                st = wave_state(small_wave(coarsen=c, n_pes_x=npx, n_pes_y=npy))
        except Exception:
            tb = traceback.format_exc().splitlines()
            print('J coupled jax %dx%d, %s: does not run on the jax backend' % (npx, npy, name))
            print('\n'.join('      ' + l for l in tb[-6:]))
            ok = False; continue
        if write_ref:
            np.savez(REF(name), **st); continue
        ref = np.load(REF(name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in ref)
        ok &= line('J coupled jax %dx%d vs numpy, %s' % (npx, npy, name), d, TOL)
    if write_ref:
        print('J coupled: numpy references written')
    return ok


def run_all():
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4')
    ok = True
    for backend, args in (('numpy', ('ref',)), ('jax', ('1', '1')), ('jax', ('2', '2'))):
        env['PYOM3_BACKEND'] = backend
        cmd = [sys.executable, os.path.abspath(__file__)] + list(args)
        r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith('J ') or l.startswith('      ')]
        print('\n'.join(lines))
        if r.returncode != 0 and not lines:
            ok = False
            print('   %s %s failed (exit %d):' % (backend, ' '.join(args), r.returncode))
            print('\n'.join('      ' + l for l in (r.stderr or r.stdout).splitlines()[-12:]))
        ok &= r.returncode == 0 and not any(l.endswith('FAIL') for l in lines)
    import glob
    for f in glob.glob(os.path.join(HERE, 'jax_ref_*.npz')):     # the references are run products
        os.remove(f)
    print('jax backend (coupled): %s' % ('ALL PASS' if ok else 'FAILURES'))
    return ok


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(0 if run_all() else 1)
    if sys.argv[1] == 'ref':
        sys.exit(0 if run(1, 1, write_ref=True) else 1)
    sys.exit(0 if run(int(sys.argv[1]), int(sys.argv[2])) else 1)
