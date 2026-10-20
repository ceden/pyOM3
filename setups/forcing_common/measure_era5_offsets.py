"""
Measure EXP_QNET_OFFSET and EXP_EMP_OFFSET of an ERA5-forced experiment.

    python3 measure_era5_offsets.py 1990-05 1991-04 --grid grid_topo.nc \
                                    --dir ../era5_icon033 --tag icon033 \
                                    [--fw-tag NAME] [--stride 7] [--workers 8]

ONE SCRIPT FOR BOTH FILE LAYOUTS. The forcing files of an experiment are
either on the ERA5 grid, with lat and lon, and interpolated by the model
while it runs (icon033, forcing_era5_native.py), or already on the MODEL grid,
with xt and yt (icon15 and global4, prep_era5.py and forcing_era5.py). Which
one a set is, is read off the first file. On the model grid there is nothing
to interpolate and the weight is the cell area itself; on the ERA5 grid the
weight map below does the work. Everything after that is identical.

    icon033:  --dir ../era5_icon033 --tag icon033
    icon15:   --dir ../era5_icon15  --tag icon15  --fw-tag icon15fw
--fw-tag names a SECOND file set holding e and tp, which icon15 keeps apart
from the flux fields; without it e and tp are expected in the same files, as
icon033 has them. Leave it out if there is no freshwater set at all: the
heat offset is then measured on its own.

WHAT IS MEASURED. ERA5 runs on a PRESCRIBED sea surface temperature, so its
surface energy budget is free and does not close: over a model domain the
annual mean of

    Q = sshf + slhf + ssr + str          (all positive DOWNWARD)

is several W/m2 into the ocean where the real ocean absorbs about 0.5. The
model cannot export that heat, so the SST restoring has to carry it back out,
which it can only do by holding a global warm bias of offset/(dQ/dT). The
same holds for the freshwater flux: over the ocean evaporation exceeds
precipitation and the rivers return the difference, which this configuration
does not have, so the mean of

    P - E = e + tp                       (e is NEGATIVE where water leaves)

is not zero and the salinity drifts. Both numbers are properties of the
FORCING AND THE DOMAIN, not of the model, so they can be measured before the
run and given to it as EXP_QNET_OFFSET [W/m2] and EXP_EMP_OFFSET [mm/day].

THE AVERAGE IS THE ONE THE MODEL SEES: over the WET CELLS of grid_topo.nc
(kbot > 0), weighted with the true cell area dxt cos(yt) dyt of the
pseudo-spherical grid, and over the fields AFTER the bilinear interpolation
from the ERA5 grid that forcing_era5_native.py does at run time. The model's
xt and yt are rebuilt here with the pyOM2 u-centred recurrence, as the model
builds them, not read from the grid file.

HOW IT AVOIDS INTERPOLATING 100 000 RECORDS. Bilinear interpolation is
LINEAR, so the area-weighted mean over the model's wet cells of an
interpolated field equals a weighted sum of the ERA5 field itself:

    sum_cells area * interp(a)  =  sum_ERA5points W * a ,

where W is assembled ONCE by pushing each model cell's area onto the four
ERA5 points of its stencil. Every record is then a dot product on the ERA5
grid and nothing is interpolated. The result is identical to the long way,
to round-off.

ICE. The model zeroes the heat and salt fluxes where the surface is at or
below -1.8 C and the flux is a cooling, so those cells do not receive what
ERA5 offers them. A second mean is therefore reported over the cells where
the ERA5 SST is above -1.5 C. That criterion is applied on the ERA5 grid, not
on the model grid, so it is close to but not exactly the model's ice mask;
the two numbers bracket what the restoring has to remove.

COST. The files are read, and they are large. --stride 7 (the default) uses
every seventh hour and reads a seventh of the bytes. SEVEN, NOT SIX: the
solar radiation has a diurnal cycle of 200 W/m2, so a stride that divides 24
samples the same few phases of it every day and biases the mean by several
W/m2 -- more than the offset being measured. A stride coprime with 24 walks
through all 24 phases within a day, so the sampling error of a monthly mean
is a fraction of a W/m2. --stride 1 reads everything. Months are done in
parallel with --workers.
"""
import os
import sys
import argparse
import calendar
import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from netCDF4 import Dataset

HEAT = ('sshf', 'slhf', 'ssr', 'str')
WATER = ('e', 'tp')
FILE_PATTERN = 'era5_%s_%04d-%02d.nc'


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


def model_grid(gridfile):
    """(xt, yt, area, wet) of the model's interior T cells."""
    with Dataset(gridfile) as f:
        dxt = np.asarray(f.variables['dxt'][:], np.float64)
        dyt = np.asarray(f.variables['dyt'][:], np.float64)
        x0  = float(np.asarray(f.variables['xu'][0]))
        y0  = float(np.asarray(f.variables['yu'][0]))
        kbot = np.asarray(f.variables['kbot'][:], np.int64)
    # anchor the relative recurrence on the first U point, as the model does
    xu, xt = u_centered_grid(dxt); xt += x0 - xu[0]
    yu, yt = u_centered_grid(dyt); yt += y0 - yu[0]
    area = (dxt[None, :]*(np.cos(np.deg2rad(yt))*dyt)[:, None])
    return xt, yt, area, kbot > 0


def file_axes(fname):
    """('native', lat, lon, flip) for files on the ERA5 grid, ('model',
    xt, yt, False) for files already on the model grid."""
    with Dataset(fname) as f:
        v = f.variables
        if 'lat' in v and 'lon' in v:
            lat = np.asarray(v['lat'][:], np.float64)
            lon = np.asarray(v['lon'][:], np.float64)
            flip = lat[0] > lat[-1]
            return 'native', (lat[::-1] if flip else lat), lon, flip
        if 'xt' in v and 'yt' in v:
            return ('model', np.asarray(v['yt'][:], np.float64),
                    np.asarray(v['xt'][:], np.float64), False)
    raise ValueError('%s has neither lat/lon nor xt/yt: I cannot tell which '
                     'grid it is on' % fname)


def weight_map(xt, yt, area, wet, lat, lon):
    """W on the ERA5 grid (south to north) such that
    sum(W*a) = sum over model wet cells of area * bilinear(a).
    The stencil is the one forcing_era5_native.py builds: longitude uniform
    and cyclic, latitude Gaussian and clipped at the poles."""
    ne_y, ne_x = len(lat), len(lon)
    dlon = lon[1] - lon[0]
    p  = np.mod(xt - lon[0], 360.0)/dlon
    i0 = np.floor(p).astype(int)
    wx = (p - i0)[None, :]
    i0, i1 = np.mod(i0, ne_x), np.mod(i0 + 1, ne_x)
    yq = np.clip(yt, lat[0], lat[-1])
    j0 = np.clip(np.searchsorted(lat, yq) - 1, 0, ne_y - 2)
    j1 = j0 + 1
    wy = ((yq - lat[j0])/(lat[j0+1] - lat[j0]))[:, None]

    a = np.where(wet, area, 0.0)
    W = np.zeros((ne_y, ne_x))
    for jj, wj in ((j0, 1.0 - wy), (j1, wy)):
        for ii, wi in ((i0, 1.0 - wx), (i1, wx)):
            np.add.at(W, (jj[:, None].repeat(len(i0), 1), ii[None, :].repeat(len(j0), 0)),
                      a*wj*wi)
    return W


def _open_month(directory, tag, y, m, nt_exp):
    fname = os.path.join(directory, FILE_PATTERN % (tag, y, m))
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    f = Dataset(fname)
    f.set_auto_mask(False)
    nt = len(f.dimensions['time'])
    if nt != nt_exp:
        f.close()
        raise ValueError('%s has %d records, %04d-%02d has %d hours'
                         % (fname, nt, y, m, nt_exp))
    return f, nt


def one_month(args):
    y, m, directory, tag, fw_tag, stride, W, flip, fields = args
    nt_exp = calendar.monthrange(y, m)[1]*24
    acc = {n: 0.0 for n in fields}
    acc_ice = {n: 0.0 for n in fields}
    wsum = wsum_ice = 0.0
    nrec = 0
    f, nt = _open_month(directory, tag, y, m, nt_exp)
    g = _open_month(directory, fw_tag, y, m, nt_exp)[0] if fw_tag else None
    try:
        src = {n: (g if (g is not None and n in WATER) else f) for n in fields}
        has_sst = 'sst' in f.variables
        for k in range(0, nt, stride):
            sst = None
            if has_sst:
                sst = np.asarray(f.variables['sst'][k], np.float64)
                if flip:
                    sst = sst[::-1]
                Wi = np.where(sst > -1.5, W, 0.0)
                wsum_ice += Wi.sum()
            wsum += W.sum()
            for n in fields:
                a = np.asarray(src[n].variables[n][k], np.float64)
                if flip:
                    a = a[::-1]
                acc[n] += float((W*a).sum())
                if sst is not None:
                    acc_ice[n] += float((Wi*a).sum())
            nrec += 1
    finally:
        f.close()
        if g is not None:
            g.close()
    return (y, m, nrec, wsum, wsum_ice, acc, acc_ice)


def months(a, b):
    y, m = (int(v) for v in a.split('-'))
    y1, m1 = (int(v) for v in b.split('-'))
    out = []
    while (y, m) <= (y1, m1):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('first'); p.add_argument('last')
    p.add_argument('--grid', default='grid_topo.nc')
    p.add_argument('--dir', default='.', help='directory of era5_<tag>_YYYY-MM.nc')
    p.add_argument('--tag', default='icon033')
    p.add_argument('--fw-tag', dest='fw_tag', default=None,
                   help='second file set holding e and tp (icon15: icon15fw)')
    p.add_argument('--stride', type=int, default=7, help='use every n-th hour; keep it coprime with 24')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()

    xt, yt, area, wet = model_grid(a.grid)
    y0, m0 = (int(v) for v in a.first.split('-'))
    f0 = os.path.join(a.dir, FILE_PATTERN % (a.tag, y0, m0))
    kind, ax1, ax2, flip = file_axes(f0)
    if kind == 'native':
        W = weight_map(xt, yt, area, wet, ax1, ax2)
        where = 'ERA5 grid %d x %d, bilinear to the model cells' % (len(ax2), len(ax1))
    else:
        # already on the model grid: the weight IS the wet cell area, and the
        # file's own coordinates must be the model's
        if (ax1.shape != yt.shape or ax2.shape != xt.shape
                or np.abs(ax1 - yt).max() > 1e-4 or np.abs(ax2 - xt).max() > 1e-4):
            raise ValueError('%s carries xt/yt that are not the grid of %s'
                             % (f0, a.grid))
        W = np.where(wet, area, 0.0)
        where = 'files already on the model grid, nothing interpolated'
    fields = []
    with Dataset(f0) as f:
        fields += [n for n in HEAT + WATER if n in f.variables]
    if a.fw_tag:
        with Dataset(os.path.join(a.dir, FILE_PATTERN % (a.fw_tag, y0, m0))) as f:
            fields += [n for n in WATER if n in f.variables and n not in fields]
        fields = [n for n in HEAT + WATER if n in fields]
    print('model grid %d x %d, %d wet cells; %s; fields %s; every %d-th hour'
          % (len(xt), len(yt), int(wet.sum()), where,
             ','.join(fields), a.stride), flush=True)
    if a.stride > 1 and 24 % a.stride == 0:
        print('WARNING: stride %d divides 24, so the same phases of the '
              'diurnal cycle are sampled every day and the solar radiation '
              'is biased. Use 5, 7, 11, ...' % a.stride, flush=True)
    if not set(HEAT) <= set(fields):
        print('WARNING: the heat fields are incomplete, the qnet offset is '
              'meaningless', flush=True)

    jobs = [(y, m, a.dir, a.tag, a.fw_tag, a.stride, W, flip, fields)
            for (y, m) in months(a.first, a.last)]
    tot = {n: 0.0 for n in fields}; tot_ice = dict(tot)
    wtot = wtot_ice = 0.0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for y, m, nrec, wsum, wsum_ice, acc, acc_ice in ex.map(one_month, jobs):
            q = sum(acc[n] for n in HEAT if n in acc)/max(wsum, 1e-30)
            pme = (sum(acc[n] for n in WATER if n in acc)/max(wsum, 1e-30)
                   *86400.0*1000.0) if set(WATER) <= set(fields) else float('nan')
            print('%04d-%02d (%d records): qnet %+8.3f W/m2   P-E %+8.4f mm/day'
                  % (y, m, nrec, q, pme), flush=True)
            for n in fields:
                tot[n] += acc[n]; tot_ice[n] += acc_ice[n]
            wtot += wsum; wtot_ice += wsum_ice

    q  = sum(tot[n] for n in HEAT if n in tot)/max(wtot, 1e-30)
    qi = sum(tot_ice[n] for n in HEAT if n in tot_ice)/max(wtot_ice, 1e-30)
    print('\n=== %s to %s, area-weighted over the model wet cells ==='
          % (a.first, a.last))
    for n in HEAT:
        if n in tot:
            print('  %-5s %+9.3f W/m2' % (n, tot[n]/wtot))
    if wtot_ice > 0.0:
        print('  qnet  %+9.3f W/m2   (%+9.3f excluding cells with ERA5 SST <= -1.5 C)'
              % (q, qi))
    else:
        print('  qnet  %+9.3f W/m2   (no sst in the files, no ice estimate)' % q)
    if set(WATER) <= set(tot):
        pme = sum(tot[n] for n in WATER)/wtot*86400.0*1000.0
        print('  e     %+9.4f mm/day' % (tot['e']/wtot*86400e3))
        print('  tp    %+9.4f mm/day' % (tot['tp']/wtot*86400e3))
        print('  P-E   %+9.4f mm/day' % pme)
    print('\nset in the run script:')
    print('setenv EXP_QNET_OFFSET          %.2f' % q)
    if set(WATER) <= set(tot):
        print('setenv EXP_EMP_OFFSET           %.4f' % pme)
    print('\nqnet is subtracted as it stands, which leaves the model with a '
          'closed surface heat budget.\nLeave about 0.5 W/m2 in it if the '
          'observed ocean heat uptake is wanted instead.')


if __name__ == '__main__':
    main()
