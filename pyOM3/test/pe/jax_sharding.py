"""JAX sharding check for pe_simple and pe_realistic (no MPI involved).

    python3 jax_sharding.py                 # everything, on 4 fake CPU devices
    python3 jax_sharding.py realistic 2 2   # one run, needs the references
    python3 jax_sharding.py realistic 2 2 2 # the same with halo_points = 2

Everything: for each package first the numpy reference (PYOM3_BACKEND=numpy),
then jax 1x1, 1x2, 2x1 and 2x2 shards against it, each in its own process
because the backend is chosen when pyOM3 is imported. The devices come from
XLA_FLAGS=--xla_force_host_platform_device_count=4, so this runs on a laptop
without MPI. Paths as test 4 (implicit, explicit, split, ab3; pe_simple:
implicit, explicit), 8 steps, the fields of config.FIELDS on the global
interior (config.state unpads the inline shard halos). Then 1x1 and 2x2
once more with halo_points = 2 against the same references (test 16 on the
jax layout: inline halos of width 2 per shard, ppermute strips, unpad).

Tolerance 1e-11: jax and numpy differ by round-off (XLA fusion), 1e-14 for
pe_simple and up to 1e-12 for pe_realistic in tke and K_diss_v; decomposed
shards differ from 1x1 by 1e-13 because XLA compiles each shard shape anew.
A layout or halo error shows as O(1)."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
AB2 = (1.6, -0.6, 0.0)
PATHS = {'simple': ((False, 'implicit', 0, AB2), (True, 'explicit', 0, AB2)),
         'realistic': ((False, 'implicit', 0, AB2), (True, 'explicit', 0, AB2),
                       (False, 'split', 2, AB2), (False, 'ab3', 2, 'ab3'))}
REF = lambda kind, name: os.path.join(HERE, 'jax_ref_%s_%s.npz' % (kind, name))
TOL = 1e-11


def run(kind, npx, npy, write_ref=False, halo=1):
    import io, contextlib
    from config import small_model, step, state, line, OM   # config puts the tree on sys.path
    if write_ref and OM.jax or not write_ref and not OM.jax:
        print('   wrong backend (jax %s), set PYOM3_BACKEND' % bool(OM.jax)); return False
    ok = True
    for explicit, name, vadv, ab in PATHS[kind]:
        with contextlib.redirect_stdout(io.StringIO()):
            st = state(step(small_model(kind, n_pes_x=npx, n_pes_y=npy, explicit=explicit, vadv=vadv, halo=halo), 8, ab))
        if write_ref:
            np.savez(REF(kind, name), **st); continue
        if not os.path.exists(REF(kind, name)):
            print('J pe_%s jax: no numpy reference' % kind); return False
        ref = np.load(REF(kind, name))
        d = max(float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in st)
        ok &= line('J pe_%s jax %dx%d%s vs numpy, %s' % (kind, npx, npy, ' halo 2' if halo > 1 else '', name), d, TOL)
    if write_ref:
        print('J pe_%s: numpy references written' % kind)
    return ok


def run_all():
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4')
    ok = True
    for kind in ('simple', 'realistic'):
        for backend, args in (('numpy', ('ref',)), ('jax', ('1', '1')), ('jax', ('1', '2')),
                              ('jax', ('2', '1')), ('jax', ('2', '2')),
                              ('jax', ('1', '1', '2')), ('jax', ('2', '2', '2'))):     # halo_points = 2 (test 16)
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
    print('jax sharding: %s' % ('ALL PASS' if ok else 'FAILURES'))
    return ok


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(0 if run_all() else 1)
    kind = sys.argv[1]
    if sys.argv[2] == 'ref':
        sys.exit(0 if run(kind, 1, 1, write_ref=True) else 1)
    halo = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    sys.exit(0 if run(kind, int(sys.argv[2]), int(sys.argv[3]), halo=halo) else 1)
