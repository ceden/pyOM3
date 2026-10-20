"""
ni_report.py -- the numbers and maps of the near-inertial diagnostic
(ni_diag.py) from the online means, per window and as a SEASONAL CYCLE, in
the units of Rimac et al. (2016):

    python3 ni_report.py [-o NAME] [-j NPROC] ocean_mean_icon033_seg*  (raw means directories)
    python3 ni_report.py [-o NAME] ocean_mean_hs_sector_seg0000000.cdf (netCDF means)

The blocks of a raw directory are read by NPROC processes (default the
number of cores, at most 16). All windows of all arguments are collected (a run of several segments is one
call), every window is printed, then the windows are grouped by CALENDAR
MONTH of their start and averaged over the years, which is the seasonal
cycle. Per window and per month:

    wind      area integral of the wind work tau . u_L on the surface current [TW]
    ni_wind   the same on the near-inertial surface current                    [TW]
    ni_pw     near-inertial p w at the mixed layer base, positive UP           [TW]
    ni_ph     near-inertial p u dh/dx + p v dh/dy there, the part through the
              SLOPE of the base, positive up                                   [TW]
    pw, ph    the same two for the total pressure work                         [TW]
    ratio     -(ni_pw + ni_ph) / ni_wind, the fraction of the near-inertial
              wind work that leaves the mixed layer (Rimac 0.11, Raja 0.30)
    mld       area mean of the mixed layer depth used                          [m]
Windows written before the horizontal part existed have no ni_ph, ph: they
are marked with * and the ratio there is from ni_pw alone.

Written: NAME_windows.txt (the per-window table), NAME_climatology.nc with
the monthly climatology of the 2-D fields (month, yt, xt), the number of
windows per month, the scalars above per month, and the area integral of
the near-inertial and the total pressure work at EVERY W face as a profile
(month, zu) [TW], the depth dependence of the flux. The cell area is
R^2 cos(phi) dlambda dphi from the saved axes; wet = mld_ni > 0.
"""
import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset

FIELDS2 = ('ni_pw_mld', 'pw_mld', 'ni_ph_mld', 'ph_mld', 'ni_pu_mld', 'ni_pv_mld',
           'ni_wind', 'wind', 'mld_ni')
OPTIONAL = ('ni_ph_mld', 'ph_mld', 'ni_pu_mld', 'ni_pv_mld')   # absent in older windows
FIELDS3 = ('ni_pw', 'pw')                 # profiles of their area integrals
RADIUS = 6370e3


def _spacing(x):
    d = np.empty_like(x)
    d[1:-1] = 0.5*(x[2:] - x[:-2])
    d[0], d[-1] = x[1] - x[0], x[-1] - x[-2]
    return d


def cell_area(xt, yt):
    dx, dy = _spacing(xt), _spacing(yt)
    return (RADIUS*np.radians(dy))[:, None]*(RADIUS*np.radians(dx))[None, :] \
        * np.cos(np.radians(yt))[:, None]


class Window:
    def __init__(self, label, ym, F2, prof, partial=False):
        self.label, self.ym, self.F2, self.prof, self.partial = label, ym, F2, prof, partial


def _chunks(seq, k):
    k = max(1, min(k, len(seq)))
    return [seq[i::k] for i in range(k)]


def _shapes(job):
    """shape of mld_ni of each block of the chunk (None if the window is
    missing or has no ni fields)."""
    n, pes = job
    out = {}
    for pe in pes:
        fn = os.path.join(pe, 'n=%d' % n, 'mld_ni.npy')
        out[int(pe.split('=')[-1])] = (np.load(fn, mmap_mode='r').shape
                                        if os.path.exists(fn) else None)
    return out


def _blocks(job):
    """the 2-D fields of the blocks of the chunk and their contribution to
    the profiles, integrated block by block so no 3-D field is assembled."""
    d, n, items, A = job
    F2 = {f: [] for f in FIELDS2}
    prof = {f: None for f in FIELDS3}
    partial = False
    for bd, sl in items:
        blk = {}
        for f in FIELDS2:
            fn = os.path.join(bd, f + '.npy')
            if os.path.exists(fn):
                blk[f] = np.load(fn)
            elif f in OPTIONAL:
                partial = True
                blk[f] = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
            else:
                blk[f] = np.load(fn)
        for f in FIELDS2:
            F2[f].append((sl, blk[f]))
        wA = A[sl]*(blk['mld_ni'] > 0.0)
        for f in FIELDS3:
            fn = os.path.join(bd, f + '.npy')
            if os.path.exists(fn):
                c = np.einsum('kji,ji->k', np.load(fn), wA)
                prof[f] = c if prof[f] is None else prof[f] + c
    return F2, prof, partial


def read_raw(d, A_of, pool=None, nproc=1):
    """Windows of one raw means directory. The blocks of a window are read
    by the processes of pool in chunks: the time is the many small files on
    the file system, not the arithmetic."""
    xt, yt = np.load(os.path.join(d, 'xt.npy')), np.load(os.path.join(d, 'yt.npy'))
    zu = np.load(os.path.join(d, 'zu.npy'))
    A = A_of(xt, yt)
    npx, npy = (int(v) for v in np.load(os.path.join(d, 'decomposition.npy')))
    pes = sorted(glob.glob(os.path.join(d, 'pe=*')), key=lambda s: int(s.split('=')[-1]))
    steps = sorted(int(os.path.basename(s).split('=')[-1])
                   for s in glob.glob(os.path.join(pes[0], 'n=*')))
    run = pool.map if pool is not None else (lambda f, jobs: [f(j) for j in jobs])
    out = []
    for n in steps:
        shp = {}
        for part in run(_shapes, [(n, c) for c in _chunks(pes, nproc)]):
            shp.update(part)
        if any(v is None for v in shp.values()):
            print('%s n=%d incomplete or without the ni fields, skipped' % (d, n))
            continue
        xoff = np.cumsum([0] + [shp[i][1] for i in range(npx - 1)])
        yoff = np.cumsum([0] + [shp[j*npx][0] for j in range(npy - 1)])
        items = []
        for pe in pes:
            r = int(pe.split('=')[-1])
            bx, by = r % npx, r // npx
            ny, nx = shp[r]
            items.append((os.path.join(pe, 'n=%d' % n),
                          (slice(yoff[by], yoff[by]+ny), slice(xoff[bx], xoff[bx]+nx))))
        F2 = {f: np.zeros((len(yt), len(xt))) for f in FIELDS2}
        prof = {f: np.zeros(len(zu)) for f in FIELDS3}
        partial = False
        for F2p, pp, pt in run(_blocks, [(d, n, c, A) for c in _chunks(items, nproc)]):
            partial = partial or pt
            for f in FIELDS2:
                for sl, blk in F2p[f]:
                    F2[f][sl] = blk
            for f in FIELDS3:
                if pp[f] is not None:
                    prof[f] += pp[f]
        label, ym = 'n=%d' % n, None
        w = os.path.join(d, 'window_n=%d.npy' % n)
        if os.path.exists(w):
            tm, t0, t1, nsteps, d0, d1 = np.load(w)
            label = '%d .. %d (%d steps)' % (int(d0), int(d1), int(nsteps))
            ym = int(d0) // 10000                # YYYYMMDDHH -> YYYYMM of the window start
        out.append(Window(label + (' *' if partial else ''), ym, F2, prof, partial))
    return xt, yt, zu, A, out


def read_nc(fn, A_of):
    f = Dataset(fn)
    xt, yt, zu = (np.asarray(f[v][:]) for v in ('xt', 'yt', 'zu'))
    A = A_of(xt, yt)
    out = []
    for m in range(len(f.dimensions['time'])):
        if 'mld_ni' not in f.variables:
            break
        F2 = {v: (np.asarray(f[v][m]) if v in f.variables else np.zeros((len(yt), len(xt))))
              for v in FIELDS2}
        partial = any(v not in f.variables for v in OPTIONAL)
        wA = A*(F2['mld_ni'] > 0.0)
        prof = {v: np.einsum('kji,ji->k', np.asarray(f[v][m]), wA) for v in FIELDS3
                if v in f.variables}
        d0 = int(f['date_start'][m])
        label = '%d .. %d (%d steps)' % (d0, int(f['date_end'][m]), int(f['nsteps'][m]))
        out.append(Window(label + (' *' if partial else ''), d0 // 10000, F2, prof, partial))
    f.close()
    return xt, yt, zu, A, out


def numbers(F2, A):
    wet = F2['mld_ni'] > 0.0
    tw = lambda f: float(np.sum(F2[f]*A*wet))*1e-12
    W, Wn = tw('wind'), tw('ni_wind')
    Pn, Hn, P, H = tw('ni_pw_mld'), tw('ni_ph_mld'), tw('pw_mld'), tw('ph_mld')
    mld = float(np.sum(F2['mld_ni']*A*wet)/max(np.sum(A*wet), 1.0))
    return W, Wn, Pn, Hn, P, H, (-(Pn + Hn)/Wn if Wn != 0.0 else np.nan), mld


KEYS = ('wind', 'ni_wind', 'ni_pw', 'ni_ph', 'pw', 'ph', 'ratio', 'mld')
HEAD = '%-36s %8s %8s %8s %8s %8s %8s %6s %6s' % (('window', 'wind TW') + KEYS[1:-1] + ('mld m',))
ROW = '%-36s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.3f %6.1f'


def main(args):
    name, nproc = 'ni_report', min(os.cpu_count() or 1, 16)
    while args and args[0] in ('-o', '-j'):
        if args[0] == '-o':
            name = args[1]
        else:
            nproc = int(args[1])
        args = args[2:]
    if not args:
        print(__doc__); sys.exit(1)
    A_of = cell_area
    xt = yt = zu = A = None
    windows = []
    pool = None
    if nproc > 1 and any(os.path.isdir(a) for a in args):
        from multiprocessing import Pool
        pool = Pool(nproc)
    for arg in args:
        xt, yt, zu, A, w = (read_raw(arg, A_of, pool, nproc) if os.path.isdir(arg)
                            else read_nc(arg, A_of))
        windows += w
        print('%s: %d windows' % (arg, len(w)), flush=True)
    if pool is not None:
        pool.close()
    if not windows:
        print('no complete windows with the ni fields found'); sys.exit(1)
    windows.sort(key=lambda w: (w.ym or 0, w.label))

    lines = [HEAD] + [ROW % ((w.label,) + numbers(w.F2, A)) for w in windows]
    print('\n'.join(lines))
    open(name + '_windows.txt', 'w').write('\n'.join(lines) + '\n')

    # the seasonal cycle: calendar month of the window start, averaged over the years
    months = sorted({w.ym % 100 for w in windows if w.ym})
    g = Dataset(name + '_climatology.nc', 'w')
    g.createDimension('month', 12); g.createDimension('xt', len(xt))
    g.createDimension('yt', len(yt)); g.createDimension('zu', len(zu))
    g.createVariable('month', 'i4', ('month',))[:] = np.arange(1, 13)
    g.createVariable('xt', 'f8', ('xt',))[:] = xt
    g.createVariable('yt', 'f8', ('yt',))[:] = yt
    g.createVariable('zu', 'f8', ('zu',))[:] = zu
    g.createVariable('area', 'f8', ('yt', 'xt'))[:] = A
    vc = g.createVariable('n_windows', 'i4', ('month',)); vc[:] = 0
    v2 = {f: g.createVariable(f, 'f8', ('month', 'yt', 'xt'), fill_value=np.nan) for f in FIELDS2}
    v3 = {f: g.createVariable('prof_' + f, 'f8', ('month', 'zu'), fill_value=np.nan) for f in FIELDS3}
    vs = {k: g.createVariable('tw_' + k, 'f8', ('month',), fill_value=np.nan)
          for k in KEYS[:6]}
    vs['ratio'] = g.createVariable('ratio', 'f8', ('month',), fill_value=np.nan)
    vs['mld'] = g.createVariable('mld', 'f8', ('month',), fill_value=np.nan)
    for f in FIELDS3:
        v3[f].units = 'TW'; v3[f].long_name = 'area integral of %s at the W faces' % f
    print('\nseasonal cycle, mean over %d windows by calendar month' % len(windows))
    print(HEAD.replace('window', 'month (n)'))
    for mo in months:
        sel = [w for w in windows if w.ym and w.ym % 100 == mo]
        F2 = {f: np.mean([w.F2[f] for w in sel], axis=0) for f in FIELDS2}
        nums = numbers(F2, A)
        print(ROW % (('%02d (%d)' % (mo, len(sel)),) + nums))
        k = mo - 1
        vc[k] = len(sel)
        for f in FIELDS2:
            v2[f][k] = F2[f]
        for f in FIELDS3:
            if all(f in w.prof for w in sel):
                v3[f][k] = np.mean([w.prof[f] for w in sel], axis=0)*1e-12
        for key, val in zip(KEYS, nums):
            vs[key][k] = val
    yrs = sorted({w.ym // 100 for w in windows if w.ym})
    g.windows = '%d windows, years %s' % (len(windows), ', '.join(str(y) for y in yrs))
    g.close()
    print('\nwritten: %s_windows.txt, %s_climatology.nc' % (name, name))


if __name__ == '__main__':
    main(sys.argv[1:])
