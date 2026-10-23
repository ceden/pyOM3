"""Test 22: setups/coupled_common/run.py end to end, the ocean alone, on
the test configuration run_setup.py (the synthetic source, no input files).
Every run is its own process in its own directory under _work/, as a job
is: EXP_SETUP=run_setup, EXP_WAVES=0, half a model day.

  22a segments     one run of 0.5 d against two segments of 0.25 d chained
                   through the gathered restart (run.py's own restart path:
                   the segment tag, the restart read, the means' partial
                   window). Every field of the final restart file bit for
                   bit, the means file of the last window too      exact
  22b raw restart  the same with EXP_RESTART_RAW=1 (the per-rank form)  exact
  22c mpi          2 ranks (1x2), one segment, against the serial run: the
                   gathered restart of a decomposed run           1e-10
  22d jax          PYOM3_BACKEND=jax, one process, 1x1 and 2x2 shards (four
                   fake CPU devices), two segments: the final restart
                   against the serial numpy one                    1e-10
                   (the closure's round-off; a run that fails prints the
                   tail of its log)
  22e explicit     the SPLIT-EXPLICIT free surface (EXP_FREE_SURFACE=explicit,
                   run_setup.py: sub_cycling 12): two segments through the
                   gathered restart = one segment, bit for bit -- the
                   restart carries the method's own state (2-D p_s, U_bt,
                   V_bt, du_cor, dv_cor); the same through the raw restart;
                   and the restart of the other method is refused     exact
  22f explicit jax the explicit run on jax 1x1, two segments, against the
                   serial explicit numpy run                       1e-10
  22h jax raw      jax 2x2 shards with EXP_RESTART_RAW=1: the per-SHARD raw
                   restart (one block per device), two segments, against
                   the serial numpy run                             1e-10
  22g jax, 2 processes  the same on TWO jax processes (mpirun, one CPU
                   device each, 2x1 shards, PYOM3_JAX_COORDINATOR): the
                   multi-process layout (block_io: pad_block per shard,
                   interior by allgather, SoloComm over the processes),
                   two segments through the gathered restart, against the
                   jax single-process 2x1 run                       1e-10
22c needs mpirun (MPIRUN as for test 4), 22d/22f jax; both say so otherwise.

(The file is not called runpy.py: that is a stdlib module.)

    python3 endtoend.py       # 22a, 22b, 22c (run_fast.py)
    python3 endtoend.py --jax # the serial reference and 22d (run_slow.py)
"""
import os, sys, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import line, HERE, ROOT, WORK
RUNPY = os.path.join(ROOT, 'setups', 'coupled_common', 'run.py')
BASE = dict(EXP_SETUP='run_setup', EXP_WAVES='0', EXP_RUNLEN_DAYS='0.5', EXP_RESTART_DAYS='0',
            EXP_AVERAGE='3h', EXP_PROFILE='0', EXP_SNAPINT_OCEAN_DAYS='30', EXP_KE_BUDGET='1')
FIELDS = ('u', 'v', 'temp', 'salt', 'p_s', 'tke', 'K_diss_v', 'du', 'dtemp', 'avg_temp', 'avg_u',
          'U_bt', 'V_bt', 'du_cor')                       # the last three: explicit runs only


def _env(**kw):
    e = dict(os.environ)
    e.update(BASE); e.update(kw)
    e['PYOM3_ROOT'] = ROOT
    e['PYTHONPATH'] = HERE + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    return e


def _run(name, segments, launcher=(), **kw):
    """run.py `segments` times in _work/run_<name>; returns (ok, log tail)"""
    d = os.path.join(WORK, 'run_' + name)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = _env(EXP_SEGMENT_DAYS='%g' % (0.5/segments), **kw)
    log = ''
    for s in range(segments):
        r = subprocess.run(list(launcher) + [sys.executable, '-u', RUNPY], cwd=d, env=env,
                           capture_output=True, text=True)
        log += r.stdout + r.stderr
        if r.returncode != 0:
            return False, d, log
    return True, d, log


def _restart(d):
    from netCDF4 import Dataset
    f = Dataset(os.path.join(d, 'runtest_restart_ocean.nc'))
    out = {k: np.array(f[k][:]) for k in FIELDS if k in f.variables}
    out['n_cpl'] = np.array([int(f.getncattr('n_cpl'))])
    f.close()
    return out


def _restart_raw_jax(d, npx, npy):
    """the fields of a per-shard raw restart (jax), assembled from the shard
    files: halos stripped (h = 1, z included for 3-D), shards row major"""
    from netCDF4 import Dataset
    f = Dataset(os.path.join(d, 'runtest_restart_ocean.nc')); n = int(f.getncattr('n_cpl')); f.close()
    R = [np.load(os.path.join(d, 'runtest_restart_ocean_pe', '%05d.npz' % k)) for k in range(npx*npy)]
    def strip(a):
        if a.ndim == 2: return a[1:-1, 1:-1]
        if a.ndim == 3 and a.shape[0] == 3: return a[:, 1:-1, 1:-1]      # p_s, three levels
        if a.ndim == 3: return a[1:-1, 1:-1, 1:-1]
        return a[:, 1:-1, 1:-1, 1:-1]
    out = {}
    for k in FIELDS:
        if k not in R[0].files: continue
        rows = [np.concatenate([strip(R[sy*npx + sx][k]) for sx in range(npx)], axis=-1) for sy in range(npy)]
        out[k] = np.concatenate(rows, axis=-2)
    out['n_cpl'] = np.array([n])
    return out


def _diff(a, b):
    return max(float(np.max(np.abs(a[k] - b[k]))/max(float(np.max(np.abs(b[k]))), 1e-300)) for k in b)


def _tail(log, n=8):
    return '\n'.join('      ' + l for l in log.strip().splitlines()[-n:])


def run(jax_only=False):
    ok = True
    good, d1, log = _run('serial', 1)
    if not good:
        print('22 run.py serial run failed:\n' + _tail(log)); return False
    ref = _restart(d1)
    if jax_only:
        return _run_jax(ref)
    ok &= line('22a serial run: 48 steps in one segment (n_cpl)', abs(ref['n_cpl'][0] - 48), 0.0)
    good, d2, log = _run('two_segments', 2)
    if not good:
        print('22a two-segment run failed:\n' + _tail(log)); ok = False
    else:
        ok &= line('22a two segments through the gathered restart = one segment', _diff(_restart(d2), ref), 0.0)
    good, d3, log = _run('raw', 2, EXP_RESTART_RAW='1')
    if not good:
        print('22b raw two-segment run failed:\n' + _tail(log)); ok = False
    else:
        # the raw restart's header has no fields: compare the means file's
        # fields through the gathered restart of a THIRD segment is not
        # possible, so read the raw block of rank 0 (one rank) instead
        R = np.load(os.path.join(d3, 'runtest_restart_ocean_pe', '00000.npz'))
        h = 1; sl = slice(h, -h)
        got = {'u': R['u'][sl, sl, sl], 'temp': R['temp'][sl, sl, sl], 'p_s': R['p_s'][:, sl, sl],
               'tke': R['tke'][sl, sl, sl], 'n_cpl': ref['n_cpl']}
        ok &= line('22b two segments through the RAW restart = one segment', _diff(got, {k: ref[k] for k in got}), 0.0)
    launcher = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
    if launcher:
        good, d4, log = _run('mpi12', 1, launcher=launcher + ['2'], EXP_NPX_OCEAN='1', EXP_NPY_OCEAN='2')
        if not good:
            print('22c mpi 1x2 run failed:\n' + _tail(log)); ok = False
        else:
            ok &= line('22c 2 ranks (1x2), gathered restart = serial', _diff(_restart(d4), ref), 1e-10)
    else:
        print('22c mpi: skipped (no mpirun on the path; set MPIRUN)')
    ok &= _run_explicit()
    return ok


def _run_explicit():
    ok = True
    EX = dict(EXP_FREE_SURFACE='explicit', EXP_KE_BUDGET='0')
    good, d1, log = _run('expl_serial', 1, **EX)
    if not good:
        print('22e explicit serial run failed:\n' + _tail(log)); return False
    ref = _restart(d1)
    ok &= line('22e explicit: the restart carries U_bt, V_bt, du_cor, 2-D p_s',
               0.0 if all(k in ref for k in ('U_bt', 'V_bt', 'du_cor')) and ref['p_s'].ndim == 2 else 1.0, 0.0)
    good, d2, log = _run('expl_two', 2, **EX)
    if not good:
        print('22e explicit two-segment run failed:\n' + _tail(log)); ok = False
    else:
        ok &= line('22e explicit: two segments through the gathered restart = one', _diff(_restart(d2), ref), 0.0)
    good, d3, log = _run('expl_raw', 2, EXP_RESTART_RAW='1', **EX)
    if not good:
        print('22e explicit raw two-segment run failed:\n' + _tail(log)); ok = False
    else:
        R = np.load(os.path.join(d3, 'runtest_restart_ocean_pe', '00000.npz'))
        h = 1; sl = slice(h, -h)
        got = {'u': R['u'][sl, sl, sl], 'temp': R['temp'][sl, sl, sl], 'p_s': R['p_s'][sl, sl],
               'U_bt': R['U_bt'][sl, sl], 'du_cor': R['du_cor'][:, sl, sl, sl], 'n_cpl': ref['n_cpl']}
        ok &= line('22e explicit: two segments through the RAW restart = one', _diff(got, {k: ref[k] for k in got}), 0.0)
    # the implicit run's restart must not start an explicit run (and vice versa)
    d4 = os.path.join(WORK, 'run_expl_refuse'); shutil.rmtree(d4, ignore_errors=True); os.makedirs(d4)
    shutil.copy(os.path.join(WORK, 'run_serial', 'runtest_restart_ocean.nc'), d4)
    env = _env(EXP_SEGMENT_DAYS='0.25', **EX)
    r = subprocess.run([sys.executable, '-u', RUNPY], cwd=d4, env=env, capture_output=True, text=True)
    refused = r.returncode != 0 and 'written with the implicit free surface' in (r.stdout + r.stderr)
    ok &= line('22e explicit run refuses the implicit restart', 0.0 if refused else 1.0, 0.0)
    return ok


def _run_jax(ref):
    ok = True
    if subprocess.run([sys.executable, '-c', 'import jax'], capture_output=True).returncode == 0:
        for npx, npy in ((1, 1), (2, 2)):
            good, d5, log = _run('jax%d%d' % (npx, npy), 2, PYOM3_BACKEND='jax',
                                 XLA_FLAGS='--xla_force_host_platform_device_count=%d' % (npx*npy),
                                 EXP_NPX_OCEAN=str(npx), EXP_NPY_OCEAN=str(npy))
            if not good:
                print('22d jax %dx%d run failed:\n' % (npx, npy) + _tail(log)); ok = False
            else:
                ok &= line('22d jax %dx%d shards, two segments = serial numpy' % (npx, npy), _diff(_restart(d5), ref), 1e-10)
        good, dh, log = _run('jax22raw', 2, PYOM3_BACKEND='jax',
                             XLA_FLAGS='--xla_force_host_platform_device_count=4',
                             EXP_NPX_OCEAN='2', EXP_NPY_OCEAN='2', EXP_RESTART_RAW='1')
        if not good:
            print('22h jax raw run failed:\n' + _tail(log)); ok = False
        else:
            got = _restart_raw_jax(dh, 2, 2)
            ok &= line('22h jax 2x2 shards, RAW per-shard restart, two segments = serial numpy',
                       _diff(got, {k: ref[k] for k in got}), 1e-10)
        good, d6, log = _run('jax_expl', 2, PYOM3_BACKEND='jax',
                             XLA_FLAGS='--xla_force_host_platform_device_count=1',
                             EXP_NPX_OCEAN='1', EXP_NPY_OCEAN='1',
                             EXP_FREE_SURFACE='explicit', EXP_KE_BUDGET='0')
        good2, d7, log2 = _run('np_expl', 1, EXP_FREE_SURFACE='explicit', EXP_KE_BUDGET='0')
        if not (good and good2):
            print('22f jax explicit run failed:\n' + _tail(log if not good else log2)); ok = False
        else:
            ok &= line('22f explicit on jax 1x1, two segments = serial numpy', _diff(_restart(d6), _restart(d7)), 1e-10)
        launcher = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
        if launcher:
            EX = dict(EXP_FREE_SURFACE='explicit', EXP_KE_BUDGET='0', EXP_NPX_OCEAN='2', EXP_NPY_OCEAN='1',
                      EXP_CANARY_EVERY='5')
            good, d8, log = _run('jax_one21', 1, PYOM3_BACKEND='jax',
                                 XLA_FLAGS='--xla_force_host_platform_device_count=2', **EX)
            good2, d9, log2 = _run('jax_two_procs', 2, launcher=launcher + ['2'], PYOM3_BACKEND='jax',
                                   XLA_FLAGS='--xla_force_host_platform_device_count=1',
                                   PYOM3_JAX_COORDINATOR='127.0.0.1:%d' % (12400 + os.getpid() % 1000), **EX)
            if not (good and good2):
                print('22g jax two-process run failed:\n' + _tail(log if not good else log2)); ok = False
            else:
                ok &= line('22g explicit on 2 jax processes (2x1), two segments = 1 process', _diff(_restart(d9), _restart(d8)), 1e-10)
        else:
            print('22g jax two processes: skipped (no mpirun on the path; set MPIRUN)')
    else:
        print('22d jax: skipped (jax not installed)')
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(jax_only='--jax' in sys.argv) else 1)
