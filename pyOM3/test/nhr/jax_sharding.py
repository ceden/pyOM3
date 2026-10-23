"""JAX sharding check for nh_realistic (no MPI involved), the pe pattern.

    python3 jax_sharding.py                 # everything, on 4 fake CPU devices
    python3 jax_sharding.py nh 2 2          # one run, needs the references
    python3 jax_sharding.py nh 2 2 2        # the same with halo_points = 2

Everything: first the numpy reference (PYOM3_BACKEND=numpy),
then jax 1x1, 1x2, 2x1 and 2x2 shards against it, each in its own process
because the backend is chosen when pyOM3 is imported. The devices come from
XLA_FLAGS=--xla_force_host_platform_device_count=4, so this runs on a laptop
without MPI. Paths as test 4 (implicit, explicit, split, ab3; pe_simple:
implicit, explicit), 8 steps, the fields of config.FIELDS on the global
interior (config.state unpads the inline shard halos). Then 1x1 and 2x2
once more with halo_points = 2 against the same references (test 16 on the
jax layout).

Tolerance 1e-9 (p_nh 1e-5), the 3-D solve run to eps_nh = 1e-16 as in
decomposition.py (see the comment there): jax and numpy differ by round-off
(XLA fusion), amplified by the weakly determined horizontal part of p_nh.
A layout or halo error shows as O(1)."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
AB2 = (1.6, -0.6, 0.0)
PATHS = {'nh': ((False, 'implicit', 0, AB2), (True, 'explicit', 0, AB2),
                (False, 'split', 2, AB2), (False, 'ab3', 2, 'ab3'))}
REF = lambda kind, name: os.path.join(HERE, 'jax_ref_%s_%s.npz' % (kind, name))
TOL = 1e-9


def run(kind, npx, npy, write_ref=False, halo=1):
    import io, contextlib
    from config import small_model, step, state, line, OM   # config puts the tree on sys.path
    if write_ref and OM.jax or not write_ref and not OM.jax:
        print('   wrong backend (jax %s), set PYOM3_BACKEND' % bool(OM.jax)); return False
    ok = True
    for explicit, name, vadv, ab in PATHS[kind]:
        with contextlib.redirect_stdout(io.StringIO()):
            st = state(step(small_model(eps_nh=1e-16, n_pes_x=npx, n_pes_y=npy, explicit=explicit, vadv=vadv,
                                        halo=halo), 8, ab))
        if write_ref:
            np.savez(REF(kind, name), **st); continue
        if not os.path.exists(REF(kind, name)):
            print('J %s_realistic jax: no numpy reference' % kind); return False
        ref = np.load(REF(kind, name))
        rel = lambda k: float(np.max(np.abs(st[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300))
        d = max(rel(k) for k in st if k != 'p_nh')
        ok &= line('J %s_realistic jax %dx%d%s vs numpy, %s' % (kind, npx, npy, ' halo 2' if halo > 1 else '', name), d, TOL)
        ok &= line('J ... p_nh (weakly determined, see decomposition.py)', rel('p_nh'), 1e-5)
    if write_ref:
        print('J %s_realistic: numpy references written' % kind)
    return ok


def run_all():
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4')
    ok = True
    for kind in ('nh',):
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
    import glob
    for f in glob.glob(os.path.join(HERE, 'jax_ref_*.npz')):     # the references are run products
        os.remove(f)
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
