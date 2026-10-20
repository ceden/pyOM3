"""
Build the ERA5 surface forcing of a pyOM3 experiment ON THE ERA5 GRID.

    python3 prep_era5_native.py 1990-05 2000-05 --out DIR --tag NAME \
                                [--fields LIST] [--workers N]

writes DIR/era5_<tag>_YYYY-MM.nc, one file per month, ON THE ERA5 GRID (the
regular Gaussian N320 grid, 1280 x 640, about 0.28 degrees), NOT on the model
grid. The model interpolates to its own grid while it runs
(forcing_era5_native.py).

WHY NOT ON THE MODEL GRID, as prep_era5.py does for global4:
    3600 x 2163 x 4 bytes             = 31 MB per record
    x 720 hours x 7 fields            = 157 GB per month
    x 120 months                      = 19 TB for ten years
On the ERA5 grid the same is 3.3 MB per record, 17 GB per month, 2.0 TB for
ten years -- and nothing is lost: ERA5 is 0.28 degrees, so a 0.1 degree copy
of it holds no extra information, only eight times the same numbers.

THE 10 m WIND, u10 and v10 [m/s] (ERA5 165/166, hourly analyses at their
stamp like sst), is what the relative-wind stress of the ocean needs
(coupled_common/exp_ocean.py, STRESS_MODE 'relwind'). It is not in the
default set; build it as its own file set beside the existing one,
    python3 prep_era5_native.py 1990-05 1991-04 --out DIR --tag icon033w --fields u10,v10
and give the run EXP_FORCING_WIND_TAG icon033w. A set without ssr is written
without the solar checks, as the e,tp set is.

FIELDS AND CONVENTIONS are those of prep_era5.py / forcing_era5.py:
    taux, tauy [N m-2] and sshf, slhf, ssr, str [W m-2, positive DOWNWARD]
        are the MEAN OVER THE HOUR that starts at the record time,
    sst [degC] is the analysis at the START of that hour.

PIPELINE per field and month (one cdo call, read right to left):
    concatenate the daily GRIB files -> select the hours by validity time
    -> land to missing ON THE NATIVE GRID (ERA5 lsm >= 0.5)
    -> reduced Gaussian to regular Gaussian
    -> accumulations / 3600 s, sst - 273.15
and then, in this script while the records are copied, every land point gets
the value of its nearest ocean point. That fill is NOT done by cdo: it
depends only on the land-sea mask, so the index map is built once per job by
a breadth-first search from all ocean points (cyclic in longitude) and
applied with one gather per record. cdo's setmisstonn aborts on this grid in
some versions with "number of valid values differ".
The nearest-neighbour fill is what makes the model-side interpolation safe:
every ERA5 point carries an ocean value, so a bilinear stencil that reaches
over a coast never picks up a land value. There is no remapping here -- the
model does that, which is also why this script is ~10 times faster than the
global4 one.

CHECKS per month, identical to prep_era5.py (the script stops on failure):
time stamps are exactly the hours of the month; the global mean of ssr is
between 0 and 600 W m-2 with an hour-to-hour ratio below 3; and the subsolar
longitude of ssr, from the first zonal harmonic between 30S and 30N, agrees
with the record time to within half an hour.

MEMORY. One field-month on the ERA5 grid is 744 x 640 x 1280 = 2.4 GB, so the
month is NOT held in memory: the output file is created first and each field
is streamed into it record by record while the checks accumulate. A worker
needs a few hundred MB and, in scratch, one temporary field (2.4 GB) plus the
concatenated GRIB.
"""
import os
import sys
import glob
import argparse
import calendar
import datetime
import shutil
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from netCDF4 import Dataset, num2date

FC = {'taux': 180, 'tauy': 181, 'sshf': 146, 'slhf': 147, 'ssr': 176, 'str': 177,
      'e': 182, 'tp': 228}
AN = {'sst': 34, 'u10': 165, 'v10': 166}       # hourly analyses, at their stamp
# fields written when --fields is not given: the set the ocean and the waves
# are forced with. Add e,tp for the freshwater flux; they then sit in the
# same files.
DEFAULT_FIELDS = ('taux', 'tauy', 'sshf', 'slhf', 'ssr', 'str', 'sst')
UNITS = {'taux': 'N m-2', 'tauy': 'N m-2', 'sshf': 'W m-2', 'slhf': 'W m-2',
         'ssr': 'W m-2', 'str': 'W m-2', 'sst': 'degC',
         'e': 'm s-1', 'tp': 'm s-1', 'u10': 'm s-1', 'v10': 'm s-1'}
LONG = {'taux': 'eastward turbulent surface stress, mean over the hour',
        'tauy': 'northward turbulent surface stress, mean over the hour',
        'sshf': 'surface sensible heat flux, positive down, mean over the hour',
        'slhf': 'surface latent heat flux, positive down, mean over the hour',
        'ssr':  'surface net solar radiation, positive down, mean over the hour',
        'str':  'surface net thermal radiation, positive down, mean over the hour',
        'sst':  'sea surface temperature at the start of the hour',
        'e':    'evaporation, DOWNWARD positive (negative where water leaves '
                'the ocean), mean over the hour',
        'tp':   'total precipitation, positive down, mean over the hour',
        'u10':  '10 m eastward wind, analysis at the start of the hour',
        'v10':  '10 m northward wind, analysis at the start of the hour'}


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('cdo failed:\n%s\n%s' % (cmd, r.stderr[-2000:]))
    return r.stdout


def ocean_mask(pool, tmp):
    """1 on ocean (lsm < 0.5), missing on land, on the native grid."""
    src = sorted(glob.glob(os.path.join(pool, 'sf/an/IV/172/*172.grb')))
    if not src:
        raise FileNotFoundError('no land-sea mask under %s/sf/an/IV/172/' % pool)
    fn = os.path.join(tmp, 'ocean_mask_native.grb')
    if not os.path.exists(fn):
        run('cdo -s -O setctomiss,0 -ltc,0.5 %s %s' % (src[0], fn))
    return fn


def fill_map(mask_native, tmp):
    """For every point of the REGULAR Gaussian grid the flat index of the
    nearest ocean point, measured in grid steps and cyclic in longitude.
    Ocean points map to themselves. Breadth-first search from all ocean
    points, so the map is built once and every record is then filled with a
    single gather."""
    fn = os.path.join(tmp, 'ocean_mask_regular.nc')
    if not os.path.exists(fn):
        run('cdo -s -O -f nc4 setgridtype,regular %s %s' % (mask_native, fn))
    with Dataset(fn) as f:
        v = list(f.variables)[-1]
        a = np.ma.filled(np.squeeze(f.variables[v][:]), np.nan)
    ocean = np.isfinite(a)
    ny, nx = ocean.shape
    if not ocean.any():
        raise ValueError('%s has no ocean points' % fn)
    idx = np.full(ocean.size, -1, np.int64)
    flat = ocean.ravel()
    idx[flat] = np.nonzero(flat)[0]
    front = np.nonzero(flat)[0]
    while front.size:
        j, i = np.divmod(front, nx)
        tgt, src = [], []
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jj, ii = j + dj, np.mod(i + di, nx)
            ok = (jj >= 0) & (jj < ny)
            tgt.append(jj[ok]*nx + ii[ok]); src.append(front[ok])
        tgt = np.concatenate(tgt); src = np.concatenate(src)
        new = idx[tgt] < 0
        tgt, src = tgt[new], src[new]
        if tgt.size == 0:
            break
        uniq, first = np.unique(tgt, return_index=True)
        idx[uniq] = idx[src[first]]
        front = uniq
    if (idx < 0).any():
        raise ValueError('%s: %d points have no ocean neighbour'
                         % (fn, int((idx < 0).sum())))
    return ocean, idx


def daily_files(pool, name, first, last):
    if name in FC:
        p = FC[name]; sub, tid = 'fc', '12'
    else:
        p = AN[name]; sub, tid = 'an', '00'
    out = []
    d = first
    while d <= last:
        f = os.path.join(pool, 'sf/%s/1H/%03d/E5sf%s_1H_%s_%03d.grb'
                         % (sub, p, tid, d.strftime('%Y-%m-%d'), p))
        if os.path.exists(f):
            out.append(f)
        d += datetime.timedelta(days=1)
    return out


def build_month(args):
    y, m, pool, outdir, tmp, ocean, fidx, fields, tag = args
    land = ~ocean.ravel()
    nd = calendar.monthrange(y, m)[1]
    nt = nd*24
    t0 = datetime.datetime(y, m, 1)
    t1 = t0 + datetime.timedelta(days=nd)
    final = os.path.join(outdir, 'era5_%s_%04d-%02d.nc' % (tag, y, m))
    if os.path.exists(final):
        return '%04d-%02d exists, skipped' % (y, m)
    mask = os.path.join(tmp, 'ocean_mask_native.grb')
    hours = [t0 + datetime.timedelta(hours=j) for j in range(nt)]
    tmpf = final + '.tmp'
    if os.path.exists(tmpf):
        os.remove(tmpf)

    gmean = {}                       # field -> (nt,) global cos-lat mean
    subsolar = []                    # per record, from ssr
    sst_min = sst_max = None
    out = None
    try:
        for name in [n for n in list(FC) + list(AN) if n in fields]:
            files = daily_files(pool, name, t0 - datetime.timedelta(days=1), t1)
            need = daily_files(pool, name, t0, t1 - datetime.timedelta(days=1))
            if len(need) < nd:
                raise FileNotFoundError('%s %04d-%02d: only %d of %d daily files found'
                                        % (name, y, m, len(need), nd))
            if name in FC:      # the hour [t, t+1h] is the record valid at t+1h
                valid = [h + datetime.timedelta(hours=1) for h in hours]
                conv = '-divc,3600'
            else:
                valid = hours
                conv = '-subc,273.15' if name == 'sst' else ''
            fo = os.path.join(tmp, '%s_%04d-%02d.nc' % (name, y, m))
            fi = os.path.join(tmp, '%s_%04d-%02d_in.grb' % (name, y, m))
            try:
                with open(fi, 'wb') as dst:
                    for src in files:
                        with open(src, 'rb') as g:
                            shutil.copyfileobj(g, dst, 16*1024*1024)
                run('cdo -s -O -f nc4 -b F32 setname,%s %s '
                    '-setgridtype,regular -mul -seldate,%s,%s %s %s %s'
                    % (name, conv, valid[0].strftime('%Y-%m-%dT%H:%M:%S'),
                       valid[-1].strftime('%Y-%m-%dT%H:%M:%S'), fi, mask, fo))
            finally:
                if os.path.exists(fi):
                    os.remove(fi)

            with Dataset(fo) as f:
                tv = f.variables['time']
                st = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute)
                      for d in num2date(tv[:], tv.units,
                                        getattr(tv, 'calendar', 'standard'))]
                if st != valid:
                    raise ValueError('%s %04d-%02d: cdo returned %d stamps %s .. %s, '
                                     'expected %d stamps %s .. %s'
                                     % (name, y, m, len(st), st[0] if st else '-',
                                        st[-1] if st else '-', len(valid),
                                        valid[0], valid[-1]))
                lat = np.asarray(f.variables['lat'][:], np.float64)
                lon = np.asarray(f.variables['lon'][:], np.float64)
                v_in = f.variables[name]

                if ocean.shape != (len(lat), len(lon)):
                    raise ValueError('%s %04d-%02d: %d x %d, the land-sea mask '
                                     'is %d x %d' % (name, y, m, len(lon), len(lat),
                                                     ocean.shape[1], ocean.shape[0]))
                if out is None:                      # create the output file once
                    cosy = np.cos(np.deg2rad(lat))
                    wsum = cosy.sum()*len(lon)
                    band = np.abs(lat) <= 30.0
                    elam = np.exp(1j*np.deg2rad(lon))
                    out = Dataset(tmpf, 'w', format='NETCDF4')
                    out.createDimension('time', nt)
                    out.createDimension('y', len(lat)); out.createDimension('x', len(lon))
                    tvo = out.createVariable('time', 'f8', ('time',))
                    tvo[:] = np.arange(nt, dtype=np.float64)
                    tvo.units = 'hours since %s' % t0.strftime('%Y-%m-%d 00:00:00')
                    tvo.long_name = 'start of the hour the record belongs to'
                    out.createVariable('lat', 'f8', ('y',))[:] = lat
                    out.createVariable('lon', 'f8', ('x',))[:] = lon
                    grid_lat, grid_lon = lat, lon
                elif (len(lat) != len(grid_lat) or len(lon) != len(grid_lon)
                      or np.abs(lat - grid_lat).max() > 1e-9):
                    raise ValueError('%s %04d-%02d: fields are on different grids'
                                     % (name, y, m))

                # one record per chunk: the model reads single records
                v_out = out.createVariable(name, 'f4', ('time', 'y', 'x'),
                                           zlib=False,
                                           chunksizes=(1, len(lat), len(lon)))
                v_out.units = UNITS[name]; v_out.long_name = LONG[name]
                gm = np.empty(nt)
                for j in range(nt):
                    a = np.ma.filled(v_in[j], np.nan).astype(np.float64)
                    # land to its nearest ocean value (see fill_map)
                    af = a.ravel()
                    bad = ~np.isfinite(af)
                    if (bad & ~land).any():
                        raise ValueError('%s %04d-%02d record %d: %d missing '
                                         'values at points the mask calls ocean'
                                         % (name, y, m, j, int((bad & ~land).sum())))
                    af[land] = af[fidx[land]]
                    a = af.reshape(a.shape)
                    if not np.isfinite(a).all():
                        raise ValueError('%s %04d-%02d record %d: missing values '
                                         'survived the fill' % (name, y, m, j))
                    v_out[j] = a.astype(np.float32)
                    gm[j] = (a*cosy[:, None]).sum()/wsum
                    if name == 'ssr':
                        z = (a[band].sum(axis=0)*elam).sum()
                        h = hours[j]
                        sub = 15.0*(11.5 - (h.hour + h.minute/60.0))
                        subsolar.append(np.angle(z*np.exp(-1j*np.deg2rad(sub)), deg=True))
                    elif name == 'sst':
                        lo, hi = float(a.min()), float(a.max())
                        sst_min = lo if sst_min is None else min(sst_min, lo)
                        sst_max = hi if sst_max is None else max(sst_max, hi)
                gmean[name] = gm
            os.remove(fo)

        # ---- checks ----------------------------------------------------------
        # Both checks read the solar radiation. A field set without it
        # (--fields e,tp) is written unchecked; the set it accompanies was
        # checked when it was built, and the two come from the same files.
        off = float('nan')
        if 'ssr' not in gmean:
            out.source = ('ERA5 (DKRZ /pool/data/ERA5/E5) on the native regular '
                          'Gaussian grid, land filled by nearest neighbour, '
                          'prep_era5_native.py')
            pme = (gmean['e'] + gmean['tp']).mean()*86400.0*1000.0 if (
                  'e' in gmean and 'tp' in gmean) else float('nan')
            out.close(); out = None
            os.replace(tmpf, final)
            return ('%04d-%02d: %d x %d, P-E %+7.3f mm/day (cos-lat mean of %s)'
                    % (y, m, len(grid_lon), len(grid_lat), pme,
                       ','.join(sorted(gmean))))
        ssr_g = gmean['ssr']
        if ssr_g.min() < 0.0 or ssr_g.max() > 600.0 or ssr_g.max() > 3.0*max(ssr_g.min(), 1e-3):
            raise ValueError('%04d-%02d: global mean ssr %.1f .. %.1f W/m2 per hour. '
                             'The records are not hourly accumulations.'
                             % (y, m, ssr_g.min(), ssr_g.max()))
        off = np.angle(np.mean(np.exp(1j*np.deg2rad(subsolar))), deg=True)
        if abs(off) > 7.5:
            raise ValueError('%04d-%02d: subsolar longitude of ssr is off by %.1f deg '
                             '(%.2f h) from the record time. The time labels are wrong.'
                             % (y, m, off, off/15.0))
        out.source = ('ERA5 (DKRZ /pool/data/ERA5/E5) on the native regular '
                      'Gaussian grid, land filled by nearest neighbour, '
                      'prep_era5_native.py')
        out.subsolar_offset_deg = float(off)
    finally:
        if out is not None:
            out.close()
    os.replace(tmpf, final)
    qnet = sum(gmean[n] for n in ('sshf', 'slhf', 'ssr', 'str') if n in gmean).mean()
    gm = lambda n: gmean[n].mean() if n in gmean else float('nan')
    return ('%04d-%02d: %d x %d, qnet %6.1f  ssr %6.1f  taux %+.3f tauy %+.3f '
            '(cos-lat means) sst %s..%s  subsolar offset %+.1f deg'
            % (y, m, len(grid_lon), len(grid_lat), qnet, ssr_g.mean(),
               gm('taux'), gm('tauy'), sst_min, sst_max, off))


def months(a, b):
    y, m = (int(v) for v in a.split('-'))
    y1, m1 = (int(v) for v in b.split('-'))
    out = []
    while (y, m) <= (y1, m1):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('first'); ap.add_argument('last')
    ap.add_argument('--out', default='.')
    ap.add_argument('--pool', default='/pool/data/ERA5/E5')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--fields', default=','.join(DEFAULT_FIELDS),
                    help='comma list; e,tp builds the freshwater set')
    ap.add_argument('--tag', default='icon033',
                    help="name part of the files, era5_<tag>_YYYY-MM.nc")
    a = ap.parse_args()
    fields = tuple(f.strip() for f in a.fields.split(',') if f.strip())
    for f in fields:
        if f not in FC and f not in AN:
            raise SystemExit('unknown field %s' % f)
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, 'tmp_prep')
    os.makedirs(tmp, exist_ok=True)
    print(run('cdo --version 2>&1 | head -1').strip(), flush=True)
    _mask = ocean_mask(a.pool, tmp)
    _ocean, _fidx = fill_map(_mask, tmp)
    print('land-sea mask %d x %d, %d ocean points, fill map built'
          % (_ocean.shape[1], _ocean.shape[0], int(_ocean.sum())), flush=True)
    jobs = [(y, m, a.pool, a.out, tmp, _ocean, _fidx, fields, a.tag)
            for (y, m) in months(a.first, a.last)]
    print('%d months, %d workers' % (len(jobs), a.workers), flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for (y, m, *_), fut in zip(jobs, [ex.submit(build_month, j) for j in jobs]):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print('%04d-%02d FAILED: %s' % (y, m, e), flush=True)
    sys.exit(1 if failed else 0)
