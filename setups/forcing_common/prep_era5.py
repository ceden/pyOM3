"""
Build the ERA5 surface forcing of a pyOM3 experiment on ITS OWN model grid.

    python3 prep_era5.py 1990-05 2000-05 --grid grid_topo.nc --tag icon15 \
                         --out DIR [--workers N]

writes DIR/era5_<tag>_YYYY-MM.nc for every month from the first to the last
argument, inclusive. The record convention is described in forcing_era5.py.
Needs cdo (module load cdo) and netCDF4.

THE MODEL GRID comes from --grid, a netCDF file holding dxt(nx), dyt(ny) and
the first U point xu(1), yu(1), i.e. the grid file the fortran setup reads.
The T coordinates are rebuilt here with the pyOM2 u-centred recurrence
(numerics.f90 u_centered_grid) rather than read from the file, because that
is what the model itself builds and forcing_era5.py compares against; on a
stretched grid the file's own yt can differ from the recurrence by the
rounding of its float32 spacings. Two consequences of a stretched grid:
  * yt is NOT the centre of its cell. The cell faces are the yu points, so
    the cdo grid description carries EXPLICIT xbounds/ybounds built from the
    faces. cdo's default bounds are the midpoints between neighbouring
    values, which on this grid are the wrong cell edges.
  * the global means of the checks below are weighted with the true cell
    area dxt cos(yt) dyt, not with cos(yt) alone.
A uniform grid (the 4 degree global4 case) is the special case of the same
code and gives the same files as the earlier fixed-grid version.

SOURCE. The DKRZ ERA5 pool on Levante (docs.dkrz.de, "ECMWF Reanalysis
Products"), GRIB on the native reduced Gaussian grid N320:
    forecast, accumulated  POOL/sf/fc/1H/PPP/E5sf12_1H_YYYY-MM-DD_PPP.grb
        180 ewss, 181 nsss  [N m-2 s]
        146 sshf, 147 slhf, 176 ssr, 177 str  [J m-2]
    analysis               POOL/sf/an/1H/034/E5sf00_1H_YYYY-MM-DD_034.grb
        034 sst [K]
    land-sea mask          POOL/sf/an/IV/172/E5sf00_IV_INVARIANT_172.grb
ERA5 short-forecast accumulations are over the ONE HOUR ENDING at the
validity time (ECMWF ERA5 data documentation), so a record divided by 3600 s
is the mean flux over that hour. The daily pool files merge the forecast
steps of two runs. Which validity times one file holds is therefore not
assumed here: every month reads one extra day on both sides and selects the
hours by validity time, and the time stamps cdo returns are checked.

STEPS per field and month (one cdo pipeline, read right to left):
    the daily GRIB files are concatenated into one temporary file (GRIB
    records are self-contained, so a byte copy is a valid file; cdo's
    bracket syntax for many inputs is not understood by every cdo version)
    -> select the hours by validity time
    -> land to missing ON THE NATIVE GRID (ERA5 lsm >= 0.5)
    -> reduced to regular Gaussian -> first-order conservative remapping to
    the model T cells (mean over the ocean part of each cell)
    -> cells without any ocean source get their nearest neighbour
    -> accumulations / 3600 s, sst - 273.15
Masking before the grid conversion matters. The conversion interpolates
linearly, and done first it smears land values into the coastal ocean
points (measured with a synthetic land block: 4.4 % error in coastal cells
against 0.9 % when masked first).

COVERAGE CHECK, once before the months (new with the free grid): the ERA5
ocean mask alone is remapped to the model grid and compared with the model's
own wet cells (kbot > 0 in the grid file, when it is there). A grid
description that does not sit where the model sits -- a longitude convention
off by 180 degrees, an origin taken from xt instead of xu -- still produces
a full set of plausible-looking files, and none of the per-month checks
below would see it. The run stops if fewer than 70 % of the model's wet
cells have any ERA5 ocean water in them; the remaining ones take a nearest
neighbour and are only counted.

CHECKS per month, the script stops on failure:
  * time stamps are exactly the hours of the month (fluxes: validity hour
    ends 01 UTC of the 1st to 00 UTC of the next month)
  * the global ocean mean of ssr lies between 0 and 600 W m-2, and its
    hour-to-hour max/min ratio is below 3 (a record accumulated since the
    forecast start instead of over one hour would fail both)
  * TIME LABEL TEST from the data themselves. The first zonal harmonic of
    ssr between 30S and 30N points at the subsolar longitude. For the hour
    that starts at j UTC that longitude is 15 deg x (11.5 - j). The mean
    offset over the month must lie within 7.5 deg (half an hour). One hour
    of labelling error gives 15 deg. The equation of time stays below 4 deg.
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
AN = {'sst': 34}
UNITS = {'taux': 'N m-2', 'tauy': 'N m-2', 'sshf': 'W m-2', 'slhf': 'W m-2',
         'ssr': 'W m-2', 'str': 'W m-2', 'sst': 'degC',
         'e': 'm s-1', 'tp': 'm s-1'}
LONG = {'taux': 'eastward turbulent surface stress, mean over the hour',
        'tauy': 'northward turbulent surface stress, mean over the hour',
        'sshf': 'surface sensible heat flux, positive down, mean over the hour',
        'slhf': 'surface latent heat flux, positive down, mean over the hour',
        'ssr':  'surface net solar radiation, positive down, mean over the hour',
        'str':  'surface net thermal radiation, positive down, mean over the hour',
        'sst':  'sea surface temperature at the start of the hour',
        'e':    'evaporation, DOWNWARD positive (negative where water leaves '
                'the surface), mean over the hour',
        'tp':   'total precipitation, positive down, mean over the hour'}


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('cdo failed:\n%s\n%s' % (cmd, r.stderr[-2000:]))
    return r.stdout


# ---------------------------------------------------------------------------
# the model grid
# ---------------------------------------------------------------------------
def u_centered_grid(dxt):
    """pyOM2 numerics.f90 u_centered_grid, verbatim (0-based):
    xu_0 = 0, xu_i = xu_{i-1} + dxt_i, xt_0 = xu_0 - dxt_0/2,
    xt_{i+1} = 2 xu_i - xt_i. Returns (xu, xt), relative."""
    n = len(dxt)
    xu = np.zeros(n); xt = np.zeros(n)
    for i in range(1, n):
        xu[i] = xu[i-1] + dxt[i]
    xt[0] = xu[0] - dxt[0]*0.5
    for i in range(1, n):
        xt[i] = 2*xu[i-1] - xt[i-1]
    return xu, xt


_GRID = {}
def model_grid(gridfile):
    """(XT, YT, XU, YU, DXT, DYT, kbot) of the model T grid, cached per
    process. XU/YU are the cell faces: cell i spans xu[i-1] .. xu[i], and
    xu[-1] of the first cell is xu[0] - dxt[0]."""
    if gridfile in _GRID:
        return _GRID[gridfile]
    with Dataset(gridfile) as f:
        dxt = np.asarray(f.variables['dxt'][:], np.float64)
        dyt = np.asarray(f.variables['dyt'][:], np.float64)
        x0  = float(np.asarray(f.variables['xu'][0]))
        y0  = float(np.asarray(f.variables['yu'][0]))
        kbot = (np.asarray(f.variables['kbot'][:]) if 'kbot' in f.variables
                else None)
    xu, xt = u_centered_grid(dxt); xt += x0 - xu[0]; xu += x0 - xu[0]
    yu, yt = u_centered_grid(dyt); yt += y0 - yu[0]; yu += y0 - yu[0]
    _GRID[gridfile] = (xt, yt, xu, yu, dxt, dyt, kbot)
    return _GRID[gridfile]


def gridfile_cdo(tmp, gridfile):
    """cdo grid description of the model T cells, with explicit bounds."""
    XT, YT, XU, YU, DXT, DYT, _ = model_grid(gridfile)
    xb = np.stack([XU - DXT, XU], axis=1)          # (nx,2) west, east face
    yb = np.stack([YU - DYT, YU], axis=1)          # (ny,2) south, north face
    fmt = lambda a: ' '.join('%.10g' % v for v in np.asarray(a).ravel())
    fn = os.path.join(tmp, 'grid_model.txt')
    with open(fn, 'w') as f:
        f.write('gridtype = lonlat\nxsize = %d\nysize = %d\n'
                'xname = lon\nyname = lat\n'
                'xunits = "degrees_east"\nyunits = "degrees_north"\n'
                'xvals = %s\nxbounds = %s\nyvals = %s\nybounds = %s\n'
                % (len(XT), len(YT), fmt(XT), fmt(xb), fmt(YT), fmt(yb)))
    return fn


def ocean_mask(pool, tmp):
    """1 on ocean (lsm < 0.5), missing on land, on the native grid."""
    src = sorted(glob.glob(os.path.join(pool, 'sf/an/IV/172/*172.grb')))
    if not src:
        raise FileNotFoundError('no land-sea mask under %s/sf/an/IV/172/' % pool)
    fn = os.path.join(tmp, 'ocean_mask_native.grb')
    if not os.path.exists(fn):
        run('cdo -s -O setctomiss,0 -ltc,0.5 %s %s' % (src[0], fn))
    return fn


def check_coverage(tmp, gridfile):
    """Remap the ERA5 ocean mask alone to the model grid and compare with the
    model's wet cells. Catches a grid description that is not where the model
    is -- nothing else here would."""
    XT, YT, XU, YU, DXT, DYT, kbot = model_grid(gridfile)
    grid = os.path.join(tmp, 'grid_model.txt')
    mask = os.path.join(tmp, 'ocean_mask_native.grb')
    fo = os.path.join(tmp, 'ocean_cover.nc')
    run('cdo -s -O -f nc4 -b F64 setname,cover -remapcon,%s -setgridtype,regular %s %s'
        % (grid, mask, fo))
    with Dataset(fo) as f:
        lat = np.asarray(f.variables['lat'][:]); lon = np.asarray(f.variables['lon'][:])
        if np.abs(lat - YT).max() > 1e-6 or np.abs(lon - XT).max() > 1e-6:
            raise ValueError('the remapped grid is not the model grid: cdo returned '
                             'lon %.4f..%.4f lat %.4f..%.4f'
                             % (lon[0], lon[-1], lat[0], lat[-1]))
        cover = np.isfinite(np.ma.filled(f.variables['cover'][:], np.nan)).reshape(len(YT), len(XT))
    if kbot is None:
        print('grid file has no kbot: ERA5 ocean water in %.1f %% of all cells'
              % (100*cover.mean()), flush=True)
        return
    wet = np.asarray(kbot) > 0
    frac = float(cover[wet].mean())
    print('coverage: %d model wet cells, %.1f %% of them have ERA5 ocean water; '
          '%d wet cells take a nearest neighbour; %d ERA5 ocean cells are model land'
          % (int(wet.sum()), 100*frac, int((wet & ~cover).sum()),
             int((~wet & cover).sum())), flush=True)
    if frac < 0.7:
        raise ValueError('only %.1f %% of the model wet cells contain ERA5 ocean '
                         'water. The grid description does not sit where the model '
                         'sits (longitude convention, or the origin taken from xt '
                         'instead of xu).' % (100*frac))


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
    y, m, pool, outdir, tmp, gridfile, tag, fields = args
    XT, YT, XU, YU, DXT, DYT, _ = model_grid(gridfile)
    NX, NY = len(XT), len(YT)
    nd = calendar.monthrange(y, m)[1]
    t0 = datetime.datetime(y, m, 1)
    t1 = t0 + datetime.timedelta(days=nd)
    final = os.path.join(outdir, 'era5_%s_%04d-%02d.nc' % (tag, y, m))
    if os.path.exists(final):
        return '%04d-%02d exists, skipped' % (y, m)
    grid, mask = os.path.join(tmp, 'grid_model.txt'), os.path.join(tmp, 'ocean_mask_native.grb')
    hours = [t0 + datetime.timedelta(hours=j) for j in range(nd*24)]
    data = {}
    for name in fields:
        files = daily_files(pool, name, t0 - datetime.timedelta(days=1), t1)
        need = daily_files(pool, name, t0, t1 - datetime.timedelta(days=1))
        if len(need) < nd:
            raise FileNotFoundError('%s %04d-%02d: only %d of %d daily files found'
                                    % (name, y, m, len(need), nd))
        if name in FC:      # hour [t, t+1h] is the record valid at t+1h
            valid = [h + datetime.timedelta(hours=1) for h in hours]
            conv = '-divc,3600'
        else:
            valid = hours
            conv = '-subc,273.15'
        fo = os.path.join(tmp, '%s_%04d-%02d.nc' % (name, y, m))
        fi = os.path.join(tmp, '%s_%04d-%02d_in.grb' % (name, y, m))
        try:
            with open(fi, 'wb') as dst:
                for src in files:
                    with open(src, 'rb') as g:
                        shutil.copyfileobj(g, dst, 16*1024*1024)
            run('cdo -s -O -f nc4 -b F64 setname,%s %s -setmisstonn -remapcon,%s '
                '-setgridtype,regular -mul -seldate,%s,%s %s %s %s'
                % (name, conv, grid, valid[0].strftime('%Y-%m-%dT%H:%M:%S'),
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
                                    st[-1] if st else '-', len(valid), valid[0], valid[-1]))
            lat = np.asarray(f.variables['lat'][:]); lon = np.asarray(f.variables['lon'][:])
            if np.abs(lat - YT).max() > 1e-6 or np.abs(lon - XT).max() > 1e-6:
                raise ValueError('%s: remapped grid is not the model grid' % fo)
            a = np.ma.filled(f.variables[name][:], np.nan).reshape(nd*24, NY, NX)
        if not np.isfinite(a).all():
            raise ValueError('%s %04d-%02d: missing values survived the fill' % (name, y, m))
        data[name] = a
        os.remove(fo)

    # ---- checks --------------------------------------------------------------
    # true cell area of the stretched grid, dxt cos(yt) dyt
    w = (DXT[None, :]*(np.cos(np.deg2rad(YT))*DYT)[:, None])[None, :, :]
    gm = lambda a: (a*w).sum(axis=(1, 2))/w.sum()
    off = np.nan
    if 'ssr' not in data:
        # the two checks below both read the solar radiation; a field set
        # without it (--fields e,tp) is written unchecked
        pass
    else:
     ssr_g = gm(data['ssr'])
     if ssr_g.min() < 0.0 or ssr_g.max() > 600.0 or ssr_g.max() > 3.0*max(ssr_g.min(), 1e-3):
        raise ValueError('%04d-%02d: global mean ssr %.1f .. %.1f W/m2 per hour. The '
                         'records are not hourly accumulations.' % (y, m, ssr_g.min(), ssr_g.max()))
     band = np.abs(YT) <= 30.0
     lam = np.deg2rad(XT)
     off = []
     for j, h in enumerate(hours):
        z = (data['ssr'][j][band].sum(axis=0)*np.exp(1j*lam)).sum()
        sub = 15.0*(11.5 - (h.hour + h.minute/60.0))
        off.append(np.angle(z*np.exp(-1j*np.deg2rad(sub)), deg=True))
     off = np.angle(np.mean(np.exp(1j*np.deg2rad(off))), deg=True)
     if abs(off) > 7.5:
        raise ValueError('%04d-%02d: subsolar longitude of ssr is off by %.1f deg '
                         '(%.2f h) from the record time. The time labels are wrong.'
                         % (y, m, off, off/15.0))

    # ---- write ----------------------------------------------------------------
    tmpf = final + '.tmp'
    with Dataset(tmpf, 'w', format='NETCDF4') as f:
        f.createDimension('time', nd*24); f.createDimension('y', NY); f.createDimension('x', NX)
        v = f.createVariable('time', 'f8', ('time',))
        v[:] = np.arange(nd*24, dtype=np.float64)
        v.units = 'hours since %s' % t0.strftime('%Y-%m-%d 00:00:00')
        v.long_name = 'start of the hour the record belongs to'
        f.createVariable('xt', 'f8', ('x',))[:] = XT
        f.createVariable('yt', 'f8', ('y',))[:] = YT
        for name in fields:
            v = f.createVariable(name, 'f4', ('time', 'y', 'x'), zlib=False)
            v[:] = data[name].astype(np.float32)
            v.units = UNITS[name]; v.long_name = LONG[name]
        f.source = 'ERA5 (DKRZ /pool/data/ERA5/E5), remapped by prep_era5.py'
        f.grid = os.path.abspath(gridfile)
        f.subsolar_offset_deg = float(off)
    os.replace(tmpf, final)
    if 'ssr' in data:
        tau = np.hypot(data['taux'], data['tauy'])
        qnet = data['sshf'] + data['slhf'] + data['ssr'] + data['str']
        return ('%04d-%02d: qnet %6.1f  ssr %6.1f  |tau| %.3f  W/m2,Pa (area means) '
                'sst %.1f..%.1f  subsolar offset %+.1f deg'
                % (y, m, gm(qnet).mean(), ssr_g.mean(), gm(tau).mean(),
                   data['sst'].min(), data['sst'].max(), off))
    pme = gm(data['e'] + data['tp']).mean()*86400.0*1000.0
    return ('%04d-%02d: P-E %+7.3f mm/day, evaporation %+7.3f, precipitation %+7.3f '
            '(area means)' % (y, m, pme, gm(data['e']).mean()*86400e3,
                              gm(data['tp']).mean()*86400e3))


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
    ap.add_argument('--grid', default='grid_topo.nc',
                    help='model grid file holding dxt, dyt, xu, yu (and kbot)')
    ap.add_argument('--tag', default='model',
                    help='name part of the output files era5_<tag>_YYYY-MM.nc')
    ap.add_argument('--fields', default=None,
                    help='comma list of fields to build (default all of '
                         'taux,tauy,sshf,slhf,ssr,str,sst). Use e,tp to add the '
                         'freshwater flux as its OWN small file set, with its own '
                         '--tag, leaving the existing files untouched')
    ap.add_argument('--out', default='.')
    ap.add_argument('--pool', default='/pool/data/ERA5/E5')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, 'tmp_prep')
    os.makedirs(tmp, exist_ok=True)
    print(run('cdo --version 2>&1 | head -1').strip(), flush=True)
    XT, YT, XU, YU, DXT, DYT, kbot = model_grid(a.grid)
    print('model grid %s: %d x %d, xt %.4f .. %.4f (dxt %.4f .. %.4f), '
          'yt %.4f .. %.4f (dyt %.4f .. %.4f)'
          % (a.grid, len(XT), len(YT), XT[0], XT[-1], DXT.min(), DXT.max(),
             YT[0], YT[-1], DYT.min(), DYT.max()), flush=True)
    gridfile_cdo(tmp, a.grid); ocean_mask(a.pool, tmp)
    check_coverage(tmp, a.grid)
    default_fields = ('taux', 'tauy', 'sshf', 'slhf', 'ssr', 'str', 'sst')
    fields = tuple(default_fields if a.fields is None else
                   [f.strip() for f in a.fields.split(',') if f.strip()])
    for f_ in fields:
        if f_ not in FC and f_ not in AN:
            sys.exit('unknown field %s' % f_)
    print('fields: %s' % ', '.join(fields), flush=True)
    jobs = [(y, m, a.pool, a.out, tmp, a.grid, a.tag, fields)
            for (y, m) in months(a.first, a.last)]
    print('%d months, %d workers, output %s/era5_%s_YYYY-MM.nc'
          % (len(jobs), a.workers, a.out, a.tag), flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for (y, m, *_), fut in zip(jobs, [ex.submit(build_month, j) for j in jobs]):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print('%04d-%02d FAILED: %s' % (y, m, e), flush=True)
    sys.exit(1 if failed else 0)
