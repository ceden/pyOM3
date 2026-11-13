"""Test 23, coupled part: the particles of setups/coupled_common/particle_diag.py
through run.py end to end (the configuration run_setup.py of test 22, the
ocean alone, half a model day of 48 steps), each run its own process in its
own directory under _work/, as a job is.

  23p snapshot    the record at a snapshot step (EXP_SNAPINT_OCEAN_DAYS
                  0.25: n = 24) against scipy's trilinear interpolation of
                  that raw snapshot's u, v, w at the recorded positions: the
                  record samples the snapshot's state      4 float32 ulp
  23i records     a box of moving particles (stride 2, three depths) and a
                  box of fixed points, a record every 4 steps: 12 records,
                  the times n dt (n = 0, 4, ..: snap_raw's label of the
                  same state), the first record at the seeds, the fixed
                  points never move, the seeds in the file are the box's
                  T points                                          exact
  23j segments    two segments of 0.25 d through the gathered restart
                  (<restart>_particles.npz next to it) against one: the two
                  record files together = the one file, and the particle
                  restarts at the end                              exact
  23k raw         the same with EXP_RESTART_RAW=1                  exact
  23l branch      EXP_INIT_OCEAN from the serial run's restart: the particles
                  start from their seeds (the restart of another run is not
                  read), the first record = a fresh serial run's first
                  record from the same state is not checked, only that the
                  log says so and the file has 12 records          exact
  23m mpi         2 ranks (1x2) and 4 (2x2), one segment, against serial
  23n jax         jax 2x2 shards (fake CPU devices), two segments, the raw
                  per-shard restart, against serial numpy
  23o jax procs   two jax processes (mpirun, one CPU device each, 2x1
                  shards), the explicit free surface as 22g, two segments,
                  against one jax process with 2 devices
                  23m-o: lon, lat 1e-10 of the displacement; the float32
                  records within one float32 ulp of their largest value (a
                  1e-12 difference of the model may flip a stored bit)
23m and 23o need mpirun, 23n and 23o jax.

    python3 particles_run.py         23i-m (run_fast.py)
    python3 particles_run.py --jax   23n, 23o (run_slow.py)
"""
import os, sys, shutil, subprocess, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import line, WORK
import endtoend as E

PART = dict(EXP_PARTICLES='1', EXP_PARTICLE_BOX='12,22,27,34', EXP_PARTICLE_STRIDE='2',
            EXP_PARTICLE_DEPTHS='5,50,300', EXP_MOORING_BOX='14,18,28,31', EXP_MOORING_STRIDE='3',
            EXP_MOORING_DEPTHS='20,100', EXP_PARTICLE_OUT_STEPS='4', EXP_RESTART_DAYS='0.25',
            EXP_AVERAGE='0', EXP_SNAPINT_OCEAN_DAYS='0.25')                  # snapshots at n = 0, 24: record steps
VARS = ('lon', 'lat', 'depth', 'u', 'v', 'w', 'temp', 'salt')


def _records(d):
    """the record files of a run directory, in segment order, concatenated"""
    from netCDF4 import Dataset
    files = sorted(glob.glob(os.path.join(d, 'runtest_particles_seg*.nc')))
    out = {k: [] for k in VARS + ('time',)}
    static = None
    for fn in files:
        with Dataset(fn) as f:
            for k in VARS + ('time',):
                out[k].append(np.array(f[k][:]))
            if static is None:
                static = {k: np.array(f[k][:]) for k in ('id', 'moving', 'lon0', 'lat0', 'depth0')}
    rec = {k: np.concatenate(v) for k, v in out.items()} if files else {}
    return rec, static, len(files)


def _prestart(d):
    return dict(np.load(os.path.join(d, 'runtest_restart_ocean_particles.npz')))


def _cmp(a, b, keys):
    return max(float(np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float))))
               / max(float(np.max(np.abs(np.asarray(b[k], float)))), 1e-300) for k in keys)


def _rel_disp(a, b):
    """lon, lat (float64) relative to the displacement of the reference; the
    float32 records (depth, u, v, w, temp, salt) relative to their largest
    value, in units of one float32 ulp of it: a 1e-12 difference of the model
    may flip the last bit of a stored value (w in 23m 2x2: 4.5e-13 at 6e-6)"""
    d0 = max(float(np.max(np.abs(b[k] - b[k][0:1]))) for k in ('lon', 'lat'))
    dp = max(float(np.max(np.abs(a[k] - b[k]))) for k in ('lon', 'lat'))/max(d0, 1e-300)
    dv = _cmp(a, b, ('depth', 'u', 'v', 'w', 'temp', 'salt'))/np.finfo(np.float32).eps
    return dp, dv


def _lines(tag, a, b):
    dp, dv = _rel_disp(a, b)
    ok = line(tag + ', lon/lat', dp, 1e-10)
    ok &= line(tag + ', float32 records [ulp of the largest]', dv, 1.0)
    return ok


def _snapshot_check(d, R):
    """23p: the record at the snapshot step n = 24 against an independent
    trilinear interpolation (scipy) of that raw snapshot's u, v, w at the
    recorded positions: the record samples exactly the snapshot's state"""
    from scipy.interpolate import RegularGridInterpolator as RGI
    import config as C
    from pyOM3.wim_realistic.grid import u_centered_grid
    sd = os.path.join(d, 'raw_ocean_runtest_seg0000000', 'pe=0', 'n=24')
    if not os.path.isdir(sd):
        return line('23p snapshot n = 24 written (EXP_SNAPINT_OCEAN_DAYS 0.25)', 1.0, 0.0)
    t = float(np.load(os.path.join(sd, 't.npy')))
    k = int(np.argmin(np.abs(R['time'] - t)))
    ok = line('23p a record at the snapshot time t = %g s' % t, float(abs(R['time'][k] - t)), 0.0)
    xu, xt, _ = u_centered_grid(C.DXT); yu, yt, _ = u_centered_grid(C.DYT)
    xt = xt + C.X0 - xu[0]; xu = xu + C.X0 - xu[0]; yt = yt + C.Y0 - yu[0]; yu = yu + C.Y0 - yu[0]
    # the vertical as pe_realistic builds it (grid.calc_grid): the pyOM2
    # recurrence, heights above the bottom face; on the stretched column zt
    # is NOT the cell midpoint
    zw_b, zt_b, _ = u_centered_grid(C.DZT); off = zw_b[0] - C.DZT[0]; zw_b = zw_b - off; zt_b = zt_b - off
    H = zw_b[-1]
    pts = np.stack([H - R['depth'][k].astype(float), R['lat'][k], R['lon'][k]], axis=-1)
    u = np.load(os.path.join(sd, 'u.npy')); v = np.load(os.path.join(sd, 'v.npy')); w = np.load(os.path.join(sd, 'w.npy'))
    wz = np.concatenate([[0.0], zw_b]); w0 = np.concatenate([np.zeros((1,) + w.shape[1:]), w], axis=0)
    cl = lambda p, a, b, c: np.stack([np.clip(p[:, 0], a[0], a[-1]), np.clip(p[:, 1], b[0], b[-1]),
                                     np.clip(p[:, 2], c[0], c[-1])], axis=-1)
    ref = {'u': RGI((zt_b, yt, xu), u)(cl(pts, zt_b, yt, xu)),
           'v': RGI((zt_b, yu, xt), v)(cl(pts, zt_b, yu, xt)),
           'w': RGI((wz, yt, xt), w0)(cl(pts, wz, yt, xt))}
    for nme in ('u', 'v', 'w'):
        dd = float(np.max(np.abs(R[nme][k] - ref[nme])))/float(np.max(np.abs(ref[nme])))
        ok &= line('23p record = scipy trilinear of the raw snapshot, %s [float32 ulp]' % nme,
                   dd/np.finfo(np.float32).eps, 4.0)
    return ok


def run(jax=False):
    """23i-m (run_fast.py); jax=True: the serial reference and 23n, 23o (run_slow.py)"""
    ok = True
    good, d1, log = E._run('part_serial', 1, **PART)
    if not good:
        print('23 particles serial run failed:\n' + E._tail(log, 14)); return False
    R1, S1, nf = _records(d1)
    if jax:
        return _run_jax(R1)
    nrec = len(R1['time'])
    ok &= line('23i records: 12 records of every 4th step in one file', abs(nrec - 12) + abs(nf - 1), 0.0)
    dt = 900.0
    ok &= line('23i records: times n dt, n = 0, 4, .., 44 (the snapshots\' label)',
               float(np.max(np.abs(R1['time'] - dt*4*np.arange(0, 12)))), 0.0)
    ok &= line('23i records: the first record at the seeds',
               float(np.max(np.abs(R1['lon'][0] - S1['lon0']) + np.abs(R1['lat'][0] - S1['lat0']))), 0.0)
    fixed = S1['moving'] == 0
    ok &= line('23i records: fixed points (%d) never move' % int(fixed.sum()),
               float(np.max(np.abs(R1['lon'][:, fixed] - S1['lon0'][fixed]))
                     + np.max(np.abs(R1['depth'][:, fixed] - S1['depth0'][fixed]))), 0.0)
    mv = ~fixed
    ok &= line('23i records: moving particles move (control)', 0.0 if np.max(np.abs(R1['lon'][-1, mv] - S1['lon0'][mv])) > 0 else 1.0, 0.0)
    # the seeds are T points of the box with the stride
    import config as C
    xt, yt = C._global_coords()
    ix = np.nonzero((xt >= 12) & (xt < 22))[0]; ix = ix[ix % 2 == 0]
    jy = np.nonzero((yt >= 27) & (yt < 34))[0]; jy = jy[jy % 2 == 0]
    ok &= line('23i records: the moving seeds are the T points of the box',
               float(np.max(np.abs(np.sort(np.unique(S1['lon0'][mv])) - np.sort(np.unique(xt[ix]))))
                     + np.max(np.abs(np.sort(np.unique(S1['lat0'][mv])) - np.sort(np.unique(yt[jy]))))), 1e-12)
    ok &= _snapshot_check(d1, R1)
    P1 = _prestart(d1)
    # 23j / 23k: two segments
    for tag, raw in (('23j segments, gathered restart', '0'), ('23k segments, raw restart', '1')):
        good, d2, log = E._run('part_two_%s' % raw, 2, EXP_RESTART_RAW=raw, **PART)
        if not good:
            print('%s: run failed:\n' % tag + E._tail(log, 14)); ok = False; continue
        R2, S2, nf2 = _records(d2)
        ok &= line('%s: two record files = the one file' % tag,
                   (0.0 if nf2 == 2 and len(R2['time']) == nrec else 1.0) + _cmp(R2, R1, VARS + ('time',)), 0.0)
        P2 = _prestart(d2)
        ok &= line('%s: particle restart at the end' % tag,
                   _cmp(P2, P1, ('p_pos', 'p_hist', 'p_ids', 'p_tau', 'p_nab', 'p_frozen', 'n_cpl')), 0.0)
        ok &= line('%s: the second segment read the particle restart' % tag,
                   0.0 if 'particles: read' in log else 1.0, 0.0)
    # 23l: a branch seeds anew
    good, db, log = E._run('part_branch', 1, EXP_INIT_OCEAN=os.path.join(d1, 'runtest_restart_ocean.nc'),
                           EXP_INIT_CHECK='0', **PART)
    if not good:
        print('23l branch run failed:\n' + E._tail(log, 14)); ok = False
    else:
        Rb, Sb, _ = _records(db)
        ok &= line('23l branch: seeds anew (no particle restart read), 12 records',
                   (0.0 if 'particles: read' not in log and 'branch' in log else 1.0)
                   + abs(len(Rb['time']) - 12) + _cmp(Sb, S1, ('id', 'lon0', 'lat0', 'depth0')), 0.0)
    launcher = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
    if launcher:
        for npx, npy in ((1, 2), (2, 2)):
            good, dm, log = E._run('part_mpi%d%d' % (npx, npy), 1, launcher=launcher + [str(npx*npy)],
                                   EXP_NPX_OCEAN=str(npx), EXP_NPY_OCEAN=str(npy), **PART)
            if not good:
                print('23m mpi %dx%d run failed:\n' % (npx, npy) + E._tail(log, 14)); ok = False; continue
            Rm, _, _ = _records(dm)
            ok &= _lines('23m %d ranks (%dx%d) = serial' % (npx*npy, npx, npy), Rm, R1)
    else:
        print('23m mpi: skipped (no mpirun on the path; set MPIRUN)')
    return ok


def _run_jax(R1):
    ok = True
    launcher = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
    if subprocess.run([sys.executable, '-c', 'import jax'], capture_output=True).returncode == 0:
        good, dj, log = E._run('part_jax22', 2, PYOM3_BACKEND='jax', XLA_FLAGS='--xla_force_host_platform_device_count=4',
                               EXP_NPX_OCEAN='2', EXP_NPY_OCEAN='2', EXP_RESTART_RAW='1', **PART)
        if not good:
            print('23n jax 2x2 run failed:\n' + E._tail(log, 14)); ok = False
        else:
            Rj, _, _ = _records(dj)
            ok &= _lines('23n jax 2x2 shards, two segments, raw restart = serial numpy', Rj, R1)
        if launcher:
            EX = dict(EXP_FREE_SURFACE='explicit', EXP_KE_BUDGET='0', EXP_NPX_OCEAN='2', EXP_NPY_OCEAN='1', **PART)
            good, d8, log = E._run('part_jax_one21', 1, PYOM3_BACKEND='jax',
                                   XLA_FLAGS='--xla_force_host_platform_device_count=2', **EX)
            good2, d9, log2 = E._run('part_jax_two_procs', 2, launcher=launcher + ['2'], PYOM3_BACKEND='jax',
                                     XLA_FLAGS='--xla_force_host_platform_device_count=1',
                                     PYOM3_JAX_COORDINATOR='127.0.0.1:%d' % (13400 + os.getpid() % 1000), **EX)
            if not (good and good2):
                print('23o jax two-process run failed:\n' + E._tail(log if not good else log2, 14)); ok = False
            else:
                Ra, _, _ = _records(d8); Rb2, _, _ = _records(d9)
                ok &= _lines('23o 2 jax processes (2x1), two segments = 1 process', Rb2, Ra)
        else:
            print('23o jax processes: skipped (no mpirun on the path; set MPIRUN)')
    else:
        print('23n/23o jax: skipped (jax not importable)')
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(jax='--jax' in sys.argv) else 1)
