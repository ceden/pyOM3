"""
Initial conditions for a pyOM3 configuration from a hydrographic climatology,
on the model grid of grid_topo.nc. Written for the WOCE/Argo Global
Hydrographic Climatology (WAGHC, Gouretski 2018; 0.25 deg, 65 standard
levels, monthly; WAGHC_PYC_* interpolated on isopycnals, WAGHC_BAR_* on
isobars -- PYC is the default and the right one here). The older WGHC
(wghc_params.nc, 0.5 deg, 44 levels, no time axis) is read by the same code.

    python3 make_initial_conditions.py --src <file, directory or glob>
                            [--month 1] [--grid grid_topo.nc]
                            [--out initial_conditions.nc]
                            [--old initial_conditions_old.nc]
                            [--tvar NAME --svar NAME --zvar NAME]
                            [--in-situ | --potential]
                            [--check-column LON LAT]

Nothing here is configuration specific: the grid, its bathymetry and the
vertical axis all come from --grid, so the same call serves the 1.5 degree
Icon grid, the 4 degree grid and the 1/10 degree POP grid. The output has the
variable names, dimensions and float32 layout the setup modules read (temp,
salt on zt, yt, xt, bottom first, dry cells 0).

WHY IT EXISTS. The initial conditions shipped with the fortran setups were
built with a HORIZONTAL nearest-valid fill at each depth. Where the model is
deeper than the climatology -- a basin below its sill, the flank of a ridge --
that fill takes water from whatever is closest at that level, which can be on
the far side of a sill. In the 1.5 degree file it puts 9.89 C under 1.24 C at
4414 m in the deep South China Sea (12 inversions above 1 K below 1000 m, 246
statically unstable interfaces, worst 1.21 kg/m3). The same defect drove a
5 m/s deep jet in the 1/10 degree POP run within hours of the start. The fill
below is by CONNECTIVITY instead, which cannot cross a sill.

--src   one file--src   one file, a directory (e.g. /pool/data/ICDC/ocean/waghc/DATA, which
        holds BAR and PYC together: --version picks, default PYC), a glob, or
        the single WGHC file. Monthly files are keyed by the month in their
        name, not by sort order.
--month 1..12 picks the month (default 1, the fortran setup starts in
        January); --month 0 averages all months.
Variable names are detected (temperature: TPOTEN, TEMP, temperature, t_an,
theta; salinity: SALINITY, salinity, s_an; depth: ZAX, depth, DEPTH, lev)
and can be overridden. Whether the source temperature is in situ or
potential is MEASURED from the deep vertical gradient (detect_in_situ: in
situ rises ~0.14 K/km below 4 km, potential keeps falling), not guessed from
the variable name; in-situ data are converted to potential temperature with
gsw. --in-situ / --potential override the test. WAGHC's 'temperature'
measures as POTENTIAL (checked Sep 2026 on the deep South Pacific).

Steps
  1. temperature and salinity of the source climatology (WAGHC: 0.25 deg,
     65 levels, one month or the annual mean; WGHC: 0.5 deg, 44 levels);
     in-situ temperature (measured by detect_in_situ, not guessed from the
     variable name) is converted to potential temperature with gsw.
  2. VERTICAL EXTENSION in every source column: below the deepest valid level
     the deepest valid values are used (a column with data to 2000 m keeps
     its 2000 m water down to the bottom). Enclosed basins below their sill
     (South China Sea, Sulu, Japan Sea, Mediterranean, ...) thereby get the
     water at sill depth, which is what fills them in reality, and every
     column is statically stable by construction where it was stable in the
     data. This replaces the previous file's horizontal fill, which put
     9.9 C / 34.47 under 2.2 C / 34.64 in the South China Sea at 4000 m.
     Missing values ABOVE the first valid level (rare) are filled from below.
  3. HORIZONTAL FILL at every level: land/no-data points take the value
     of the nearest valid point (Euclidean on the index grid, periodic in
     longitude), so model wet points at coasts and in narrow straits always
     find a neighbour.
  4. HORIZONTAL TRANSFER to the model T grid (xt, yt, and the cell faces
     xu, yu, built from grid_topo.nc with the pyOM2 u-centred recurrence),
     periodic in longitude, by one of two operators (--hinterp, default
     auto):
       block    every source cell whose centre falls inside a model cell
                enters that cell's mean, weighted by cos(lat) and by its own
                validity. This is what a coarse model grid needs: with a
                0.25 degree source and 1.5 degree cells a model cell holds
                about 36 source cells, and sampling four of them instead of
                averaging all of them both aliases the field and declares a
                mostly-wet cell dry whenever its centre lands on a source
                land point. A model cell is valid where the valid share of
                its weight exceeds 1/2, exactly the rule the bilinear form
                uses. Model cells that contain no source centre at all fall
                back to the bilinear value.
       bilinear masked bilinear interpolation of the four surrounding source
                points, the right operator when the source is COARSER than
                the model (the 1/10 degree case).
       auto     block when the typical model cell holds at least four source
                centres, bilinear otherwise.
     Then linear interpolation in depth to the model zt (model levels below
     the deepest source level take the deepest source level).
  5. CONVECTIVE ADJUSTMENT of every model column (--no-stabilize to skip):
     inverted pairs are mixed with their thicknesses as weights until the
     column is stable, conserving the column's heat and salt content. The
     per-level horizontal fill can otherwise leave a heavy layer over a
     light one where two neighbouring source columns are very different
     (Kattegat, Baltic, river mouths).
  6. Cells below the model bottom (kbot) are set to 0; the file has no
     missing_value, and the grid coordinates are written alongside.
  7. Report: columns that needed the vertical extension and how many levels,
     horizontal fills per level, and a static-stability check of every wet
     model column with sigma_0 (gsw if available, else a linear proxy).
"""
import sys, os, argparse, time
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt, label, binary_dilation
from scipy.interpolate import RegularGridInterpolator


def u_centered_grid(dyt):
    """pyOM2 u_centered_grid: yu(0) = 0, yu(i) = yu(i-1) + dyt(i);
    yt(0) = -dyt(0)/2, yt(i) = 2 yu(i-1) - yt(i-1)  (all relative)"""
    n = len(dyt)
    yu = np.cumsum(dyt)
    yt = np.empty(n); yt[0] = yu[0] - dyt[0]/2.0
    for i in range(1, n):
        yt[i] = 2*yu[i-1] - yt[i-1]
    return yt, yu


def model_grid(gridfile):
    with Dataset(gridfile) as f:
        dxt = np.asarray(f['dxt'][:], np.float64); dyt = np.asarray(f['dyt'][:], np.float64)
        dzt = np.asarray(f['dzt'][:], np.float64)
        xu0 = float(np.asarray(f['xu'][0])); yu0 = float(np.asarray(f['yu'][0]))
        kbot = np.asarray(f['kbot'][:], np.float64)
    # pyOM: xu(0) = x_origin, xt(0) = xu(0) - dxt(0)/2, recurrence
    xt_rel, xu_rel = u_centered_grid(dxt); xt = xu0 + (xt_rel - xu_rel[0]); xu = xu0 + (xu_rel - xu_rel[0])
    yt_rel, yu_rel = u_centered_grid(dyt); yt = yu0 + (yt_rel - yu_rel[0]); yu = yu0 + (yu_rel - yu_rel[0])
    # vertical: pe_realistic calc_grid = pyOM2 recurrence on the nz interior,
    # height above the bottom face (zt(0) = dzt(0)/2, top face = sum dzt);
    # written to the file as negative depth (surface face 0), as before
    zt_rel, zw_rel = u_centered_grid(dzt)                      # zt_rel(0) = dzt(0)/2
    H = dzt.sum()
    zt = zt_rel - H; zw = zw_rel - H                            # negative, k up
    return xt, yt, xu, yu, zt, zw, dxt, dyt, dzt, kbot


def fill_upward(a, valid):
    """missing values ABOVE the first valid level (rare) take the value below.
    No downward extension here: a WAGHC column ends where ITS 0.25 deg
    bathymetry ends, which over a seamount or ridge is far above the model's
    wet cells on the flanks -- extending the column downward there put
    thermocline water at 4000 m next to 2 C water (POP blow-up, Sep 2026).
    a: (nz, ny, nx), k = 0 the SURFACE (source axis, positive down)."""
    out = a.copy(); v = valid.copy(); n = 0
    for k in range(a.shape[0]-2, -1, -1):
        need = ~v[k] & v[k+1]
        out[k][need] = out[k+1][need]; v[k] |= need; n += int(need.sum())
    return out, v, n


def fill_connected(temp, salt, valid, wet):
    """Model-grid fill, per level from the top down (k = 0 bottom, k up).
    A wet cell without data that is CONNECTED through wet cells at that level
    to valid data takes the nearest valid cell of its connected region
    (seamount flanks, ridges, straits, coasts: same-depth water). A wet cell
    in a region with no valid data at that level -- a basin below its sill --
    takes the level above (sill water), which is already final. Returns the
    counts (horizontal, sill) per level."""
    nz, ny, nx = temp.shape
    nh = np.zeros(nz, int); nsill = np.zeros(nz, int)
    for k in range(nz-1, -1, -1):
        w = wet[k]; v = valid[k] & w; need = w & ~v
        if not need.any():
            continue
        lab, nlab = label(w)                                       # 4-connected regions of wet cells
        has_valid = np.zeros(nlab+1, bool); has_valid[np.unique(lab[v])] = True
        reach = need & has_valid[lab]; pocket = need & ~has_valid[lab]
        if reach.any():
            idx = distance_transform_edt(~v, return_distances=False, return_indices=True)
            jn, inn = idx[0], idx[1]
            same = lab[jn, inn] == lab
            ok = reach & same
            temp[k][ok] = temp[k][jn[ok], inn[ok]]; salt[k][ok] = salt[k][jn[ok], inn[ok]]
            rest = reach & ~same                                   # nearest valid lies across land: grow inside the region
            filled = v | ok
            it = 0
            while rest.any() and it < 2000:
                grown = binary_dilation(filled, structure=np.array([[0,1,0],[1,1,1],[0,1,0]])) & rest
                if not grown.any():
                    break
                # take the mean of already-filled 4-neighbours
                acc = np.zeros_like(temp[k]); accs = np.zeros_like(salt[k]); cnt = np.zeros_like(temp[k])
                for dj, di in ((0,1),(0,-1),(1,0),(-1,0)):
                    nb = np.roll(filled, (dj, di), axis=(0,1)); tn = np.roll(temp[k], (dj, di), axis=(0,1)); sn = np.roll(salt[k], (dj, di), axis=(0,1))
                    acc += np.where(nb, tn, 0.0); accs += np.where(nb, sn, 0.0); cnt += nb
                temp[k][grown] = acc[grown]/cnt[grown]; salt[k][grown] = accs[grown]/cnt[grown]
                filled |= grown; rest &= ~grown; it += 1
            if rest.any():                                         # unreachable after all: treat as pocket
                pocket |= rest
            nh[k] = int(ok.sum() + (reach & ~rest & ~ok).sum())
        if pocket.any():
            if k == nz-1:
                # isolated water bodies (Caspian, Black Sea, lakes, small
                # enclosed seas): no data at any level, nothing above to take.
                # Nearest data on the surface level, wherever it is.
                idx = distance_transform_edt(~v, return_distances=False, return_indices=True)
                jn, inn = idx[0], idx[1]
                temp[k][pocket] = temp[k][jn[pocket], inn[pocket]]; salt[k][pocket] = salt[k][jn[pocket], inn[pocket]]
                nreg = len(np.unique(lab[pocket]))
                print('surface level: %d wet cells in %d isolated water bodies without any data, '
                      'filled from the nearest data on the surface level' % (int(pocket.sum()), nreg))
            else:
                temp[k][pocket] = temp[k+1][pocket]; salt[k][pocket] = salt[k+1][pocket]
            nsill[k] = int(pocket.sum())
        valid[k] = w                                               # this level is final
    return nh, nsill


_TCAND = ('TPOTEN', 'temperature', 'TEMP', 'temp', 't_an', 'theta', 'PTEMP')
_SCAND = ('SALINITY', 'salinity', 'SALT', 'salt', 's_an', 'PSAL')
_ZCAND = ('ZAX', 'depth', 'DEPTH', 'lev', 'LEV', 'z', 'Z')
_XCAND = ('LON', 'lon', 'longitude', 'LONGITUDE', 'x')
_YCAND = ('LAT', 'lat', 'latitude', 'LATITUDE', 'y')
_POTENTIAL_NAMES = ('TPOTEN', 'theta', 'PTEMP', 'ptemp')


def _pick(f, cand, given=None):
    if given:
        if given not in f.variables:
            sys.exit('variable %s not in the file (have: %s)' % (given, ', '.join(f.variables)))
        return given
    for c in cand:
        if c in f.variables:
            return c
    sys.exit('none of %s in the file (have: %s)' % (list(cand), ', '.join(f.variables)))


def _read_one(fname, a):
    """(lon, lat, zax, T, S, valid, tvar) from one file; a month axis is
    reduced by --month (1..12) or averaged (--month 0)."""
    with Dataset(fname) as f:
        tv = _pick(f, _TCAND, a.tvar); sv = _pick(f, _SCAND, a.svar)
        zv = _pick(f, _ZCAND, a.zvar); xv = _pick(f, _XCAND); yv = _pick(f, _YCAND)
        lon = np.asarray(f[xv][:], np.float64); lat = np.asarray(f[yv][:], np.float64)
        zax = np.asarray(f[zv][:], np.float64)
        T = np.ma.filled(f[tv][:], np.nan).astype(np.float64)
        S = np.ma.filled(f[sv][:], np.nan).astype(np.float64)
        mv = [float(getattr(f[v], 'missing_value', -9.0)) for v in (tv, sv)]
    if T.ndim == 4:                                   # (time, z, y, x)
        nt = T.shape[0]
        if a.month == 0:
            with np.errstate(invalid='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)   # all-missing columns
                    T = np.nanmean(np.where(np.isclose(T, mv[0]), np.nan, T), axis=0)
                    S = np.nanmean(np.where(np.isclose(S, mv[1]), np.nan, S), axis=0)
        else:
            k = (a.month - 1) % nt
            T = T[k]; S = S[k]
    valid = (np.isfinite(T) & np.isfinite(S) & ~np.isclose(T, mv[0]) & ~np.isclose(S, mv[1])
             & (T > -3.0) & (T < 40.0) & (S > 1.0) & (S < 45.0))
    return lon, lat, zax, T, S, valid, tv


def month_of(fname):
    """month number from a WAGHC file name (WAGHC_PYC_07_UHAM-ICDC_v1_0.nc),
    None for files that carry no month (Mixedlayer, Oceanareacode, WGHC)"""
    import re
    m = re.search(r'_(0[1-9]|1[0-2])_', os.path.basename(fname))
    return int(m.group(1)) if m else None


def read_climatology(a):
    """--src: one file, a directory, or a glob. The ICDC directory holds BOTH
    versions (WAGHC_BAR_* isobaric, WAGHC_PYC_* isopycnal) plus files without
    a month, so a bare directory selects PYC (isopycnal, the one that does not
    create artificial water masses across fronts) and a glob is taken as given.
    Files are keyed by the month in their name, not by sort order."""
    import glob as _glob
    if os.path.isdir(a.src):
        files = sorted(_glob.glob(os.path.join(a.src, '*%s*.nc' % a.version)))
        if not files:
            files = sorted(_glob.glob(os.path.join(a.src, '*.nc')))
        print('directory %s: %d files matching %s' % (a.src, len(files), a.version))
    else:
        files = sorted(_glob.glob(a.src)) or [a.src]
    months = {month_of(f): f for f in files if month_of(f)}
    if len(files) == 1:
        lon, lat, zax, T, S, valid, tv = _read_one(files[0], a)
        print('source: %s (month %s)' % (os.path.basename(files[0]),
                                         'mean' if a.month == 0 else a.month))
    elif a.month != 0:
        if a.month not in months:
            sys.exit('no file for month %d among: %s'
                     % (a.month, ', '.join(os.path.basename(f) for f in files)))
        fn = months[a.month]
        lon, lat, zax, T, S, valid, tv = _read_one(fn, a)
        print('source: %s (month %d, %d monthly files found)' % (os.path.basename(fn), a.month, len(months)))
    else:
        files = [months[k] for k in sorted(months)]
        acc_t = acc_s = cnt = None
        for fn in files:
            lon, lat, zax, T, S, valid, tv = _read_one(fn, a)
            if acc_t is None:
                acc_t = np.zeros_like(T); acc_s = np.zeros_like(S); cnt = np.zeros(T.shape)
            acc_t += np.where(valid, T, 0.0); acc_s += np.where(valid, S, 0.0); cnt += valid
        with np.errstate(invalid='ignore'):
            T = acc_t/np.where(cnt > 0, cnt, 1); S = acc_s/np.where(cnt > 0, cnt, 1)
        valid = cnt > 0
        print('source: annual mean of %d monthly files (%s ...)' % (len(files), os.path.basename(files[0])))
    if a.in_situ is None:
        a.in_situ = detect_in_situ(T, zax, valid)
        print('temperature variable %s treated as %s' % (tv, 'IN SITU' if a.in_situ else 'potential'))
    return lon, lat, zax, T, S, valid, tv


def detect_in_situ(T, zax, valid, z1=3500.0, z2=5000.0):
    """in situ or potential temperature, measured rather than guessed from
    the variable name: below ~4 km adiabatic compression makes IN SITU
    temperature INCREASE downward by about 0.14 K/km, while potential
    temperature keeps decreasing slowly. The mean vertical gradient over the
    columns that reach z2 decides (WAGHC's 'temperature' comes out potential;
    WGHC ships TEMP in situ and TPOTEN potential)."""
    k = [i for i, z in enumerate(zax) if z1 <= z <= z2]
    if len(k) < 2:
        print('no levels between %.0f and %.0f m: assuming potential temperature' % (z1, z2))
        return False
    k1, k2 = k[0], k[-1]
    both = valid[k1] & valid[k2]
    if both.sum() < 100:
        print('too few deep columns to test: assuming potential temperature')
        return False
    dT = np.mean(T[k2][both] - T[k1][both])/((zax[k2] - zax[k1])/1000.0)   # K per km downward
    print('deep vertical gradient %.0f->%.0f m over %d columns: %+.3f K/km downward '
          '(in situ ~ +0.14, potential < 0)' % (zax[k1], zax[k2], int(both.sum()), dT))
    return dT > 0.05


def to_potential(T, S, zax, lat, valid):
    """in situ -> potential temperature (reference 0 dbar). gsw if available:
    p from depth and latitude, SA from practical salinity, CT then pt."""
    try:
        import gsw
    except ImportError:
        print('gsw not available: temperature left as in situ (up to ~0.5 K too warm at 5000 m)')
        return T
    out = T.copy()
    for k, z in enumerate(zax):
        p = gsw.p_from_z(-abs(z), lat)[:, None]*np.ones((1, T.shape[2]))
        v = valid[k]
        if not v.any():
            continue
        SA = gsw.SA_from_SP(S[k], p, 0.0, lat[:, None])
        pt = gsw.pt0_from_t(SA, T[k], p)
        out[k] = np.where(v, pt, T[k])
    print('in situ -> potential temperature (gsw): mean change %.3f K, max %.3f K'
          % (np.nanmean(np.where(valid, out - T, np.nan)), np.nanmax(np.abs(np.where(valid, out - T, 0)))))
    return out


def stabilize(temp, salt, wet, dzt, tol=1e-3, max_sweeps=60):
    """Convective adjustment of the model columns, as WAGHC does to its own
    isobaric profiles: wherever the cell above is denser than the cell below
    (sigma_0, more than tol), the two are mixed with their thicknesses as
    weights and the sweep continues downward, so a heavy surface layer is
    carried down as far as it needs to go. Repeated until no inversion is
    left or max_sweeps is reached. temp/salt are modified in place; k = 0 is
    the BOTTOM level, so dzt[k] is the thickness of level k.

    Heat and salt content of each column are conserved to round-off (mixing
    is a thickness-weighted average), so this moves properties vertically
    but adds none.

    tol must not be 0: density is not linear in T and S, so the mixture of
    two cells can be slightly denser than either (cabbeling), which creates
    a new inversion of order 1e-3 kg/m3 against the cell below and the sweep
    never converges. The default 1e-3 kg/m3 leaves inversions with
    N^2 ~ -3e-7 s^-2 over a 250 m cell, four orders below the real
    stratification and removed by the model's closure in one step.
    """
    nz = temp.shape[0]
    t0_ = temp.copy(); s0_ = salt.copy()
    sig = sigma0(temp.astype(np.float64), salt.astype(np.float64))
    for sweep in range(max_sweeps):
        nfix = 0
        for k in range(nz - 2, -1, -1):                 # pair (k below, k+1 above)
            bad = wet[k] & wet[k+1] & (sig[k+1] > sig[k] + tol)
            n = int(bad.sum())
            if not n:
                continue
            nfix += n
            w1, w2 = dzt[k], dzt[k+1]
            tm = (w1*temp[k] + w2*temp[k+1])/(w1 + w2)
            sm = (w1*salt[k] + w2*salt[k+1])/(w1 + w2)
            for a, m in ((temp, tm), (salt, sm)):
                a[k] = np.where(bad, m, a[k]); a[k+1] = np.where(bad, m, a[k+1])
            sig[k] = sigma0(temp[k].astype(np.float64), salt[k].astype(np.float64))
            sig[k+1] = sig[k]*bad + sig[k+1]*(~bad)
        if nfix == 0:
            break
    dT = np.abs(temp - t0_)[wet]; dS = np.abs(salt - s0_)[wet]
    print('convective adjustment (tol %.1e kg/m3): %d sweeps, %d cells changed, '
          'max |dT| %.3f K, max |dS| %.3f, mean |dT| %.2e K'
          % (tol, sweep + 1, int((dT > 0).sum()), dT.max(), dS.max(), dT.mean()))
    if sweep + 1 == max_sweeps and nfix:
        both = wet[1:] & wet[:-1]
        worst = np.max(np.where(both, sig[1:] - sig[:-1], -1.0))
        print('  not fully converged: %d pairs still inverted, largest %.4f kg/m3 '
              '(cabbeling; harmless below ~0.01)' % (nfix, worst))
    return temp, salt


def sigma0(t, s):
    try:
        import gsw
        return gsw.sigma0(s, t)                     # potential temperature ~ conservative for the check
    except Exception:
        return 0.8*s - 0.2*t                        # linear proxy, kg/m3


def source_counts(lon, lat, xt, yt, xu, dxt, yu, dyt):
    """(ix, iy of every source point in the model grid, count per model cell).
    The model cell (j, i) spans xu[i-1]..xu[i] and yu[j-1]..yu[j], with the
    first face one cell width below the first U point. x is periodic over
    360 degrees, y is not: source rows outside the model's latitude range
    have iy < 0 or iy >= ny and are dropped."""
    nx, ny = len(xt), len(yt)
    xe = np.concatenate([[xu[0] - dxt[0]], xu])              # nx+1, spans 360
    ye = np.concatenate([[yu[0] - dyt[0]], yu])              # ny+1
    lonm = xe[0] + np.mod(np.asarray(lon, np.float64) - xe[0], 360.0)
    ix = np.clip(np.searchsorted(xe, lonm, side='right') - 1, 0, nx - 1)
    iy = np.searchsorted(ye, np.asarray(lat, np.float64), side='right') - 1
    inside = (iy >= 0) & (iy < ny)
    cnt = np.zeros(ny*nx)
    if inside.any():
        flat = (iy[inside, None]*nx + ix[None, :]).ravel()
        cnt = np.bincount(flat, minlength=ny*nx).astype(np.float64)
    return ix, iy, inside, cnt.reshape(ny, nx)


def block_to_model(lon, lat, zax, T, S, valid, xt, yt, xu, dxt, yu, dyt, t0):
    """Block average of the source onto the model cells: every source centre
    inside a cell contributes with weight cos(lat) times its validity. NaN
    where the valid share is at or below 1/2 and where the cell holds no
    source centre at all (the caller fills those from the bilinear form)."""
    nx, ny = len(xt), len(yt)
    ix, iy, inside, cnt = source_counts(lon, lat, xt, yt, xu, dxt, yu, dyt)
    flat = (iy[inside, None]*nx + ix[None, :]).ravel()
    w_lat = np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat, np.float64)))[inside, None],
                            (int(inside.sum()), len(lon))).ravel()
    w_all = np.bincount(flat, weights=w_lat, minlength=ny*nx)     # level independent
    Th = np.full((len(zax), ny, nx), np.nan, np.float32); Sh = np.full_like(Th, np.nan)
    for k in range(len(zax)):
        w = w_lat*valid[k][inside].ravel()
        ws = np.bincount(flat, weights=w,           minlength=ny*nx)
        ts = np.bincount(flat, weights=w*T[k][inside].ravel(), minlength=ny*nx)
        ss = np.bincount(flat, weights=w*S[k][inside].ravel(), minlength=ny*nx)
        ok = (w_all > 0) & (ws > 0.5*w_all)
        ot = np.full(ny*nx, np.nan); os_ = np.full(ny*nx, np.nan)
        ot[ok] = ts[ok]/ws[ok]; os_[ok] = ss[ok]/ws[ok]
        Th[k] = ot.reshape(ny, nx); Sh[k] = os_.reshape(ny, nx)
        if k % 10 == 0:
            print('  level %2d (%5.0f m) averaged, %.0f s' % (k, zax[k], time.time() - t0), flush=True)
    return Th, Sh, cnt


def bilinear_to_model(lon, lat, zax, T, S, valid, xt, yt, t0, cells=None):
    """Masked bilinear interpolation to the model T points: the four
    surrounding source points, each weighted by its validity, and the model
    point valid where that weight exceeds 1/2. `cells` restricts the result
    to a boolean (ny, nx) mask (the block form's empty cells); the
    interpolation itself is done everywhere, it is cheap against the read."""
    ny, nx = len(yt), len(xt)
    lonp = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    xm = np.mod(xt, 360.0)
    xm = np.where(xm < lonp[0], xm + 360.0, xm)
    xm = np.where(xm > lonp[-1], xm - 360.0, xm)
    ym = np.clip(yt, lat[0], lat[-1])
    n_clip = int((yt < lat[0]).sum() + (yt > lat[-1]).sum())
    if n_clip:
        print('latitude: source covers %.2f..%.2f, model %.2f..%.2f -> %d model rows take the '
              'nearest source row (Antarctic/Arctic edge)' % (lat[0], lat[-1], yt[0], yt[-1], n_clip))
    pts = np.stack(np.meshgrid(ym, xm, indexing='ij'), axis=-1).reshape(-1, 2)
    Th = np.full((len(zax), ny, nx), np.nan, np.float32); Sh = np.full_like(Th, np.nan)
    for k in range(len(zax)):
        wv = valid[k].astype(np.float64)
        flds = []
        for s in (T[k]*wv, S[k]*wv, wv):
            fld = np.concatenate([s[:, -1:], s, s[:, :1]], axis=1)
            itp = RegularGridInterpolator((lat, lonp), fld, method='linear',
                                          bounds_error=False, fill_value=None)
            flds.append(itp(pts).reshape(ny, nx))
        wsum = flds[2]; okk = wsum > 0.5
        if cells is not None:
            okk &= cells
        Th[k][okk] = (flds[0][okk]/wsum[okk]); Sh[k][okk] = (flds[1][okk]/wsum[okk])
        if k % 10 == 0:
            print('  level %2d (%5.0f m) interpolated, %.0f s' % (k, zax[k], time.time() - t0), flush=True)
    return Th, Sh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='wghc_params.nc',
                   help='WAGHC file, directory or glob of monthly files, or the WGHC file')
    p.add_argument('--month', type=int, default=1, help='1..12, or 0 for the annual mean')
    p.add_argument('--version', default='PYC', choices=['PYC', 'BAR'],
                   help='WAGHC version selected when --src is a directory: PYC = interpolated '
                        'on isopycnals (default), BAR = on isobars')
    p.add_argument('--grid', default='grid_topo.nc')
    p.add_argument('--out', default='initial_conditions.nc')
    p.add_argument('--old', default=None, help='previous IC file, column comparison')
    p.add_argument('--tvar', default=None); p.add_argument('--svar', default=None)
    p.add_argument('--zvar', default=None)
    p.add_argument('--in-situ', dest='in_situ', action='store_true', default=None,
                   help='source temperature is in situ (default for WAGHC): convert to potential')
    p.add_argument('--potential', dest='in_situ', action='store_false',
                   help='source temperature is already potential (WGHC TPOTEN)')
    p.add_argument('--no-stabilize', dest='stabilize', action='store_false', default=True,
                   help='skip the convective adjustment of the interpolated columns')
    p.add_argument('--stabilize-tol', type=float, default=1e-3,
                   help='density inversion tolerated without mixing, kg/m3 (default 1e-3; '
                        '0 never converges, see stabilize())')
    p.add_argument('--hinterp', default='auto', choices=['auto', 'block', 'bilinear'],
                   help='horizontal transfer to the model cells: block average of the '
                        'source cells inside each model cell (a coarse model grid), '
                        'masked bilinear of the four surrounding source points (a model '
                        'grid finer than the source), or auto (default): block from four '
                        'source centres per model cell upward')
    p.add_argument('--check-column', nargs=2, type=float, default=None,
                   help='longitude and latitude of a column to print at the end '
                        '(default: the column with the largest density inversion '
                        'left, or the deepest column if there is none)')
    a = p.parse_args()
    t0 = time.time()

    # ---------------------------------------------------------------- source
    lon, lat, zax, T, S, valid, tname = read_climatology(a)
    print('%s: %d levels x %d lat x %d lon, lon %.2f..%.2f, lat %.1f..%.1f, '
          'depth %.0f..%.0f m, valid %.1f %%'
          % (tname, len(zax), len(lat), len(lon), lon[0], lon[-1], lat[0], lat[-1],
             zax[0], zax[-1], 100*valid.mean()))
    if a.in_situ:
        T = to_potential(T, S, zax, lat, valid)
    T, valid, nT_up = fill_upward(T, valid)                 # surface gaps only
    S, valid, nS_up = fill_upward(S, valid)
    print('upward fill of missing surface levels: %d cells' % nT_up)
    T = np.where(valid, T, 0.0); S = np.where(valid, S, 0.0)

    # ---------------------------------------------------------------- model grid
    xt, yt, xu, yu, zt, zw, dxt_deg, dyt_deg, dzt, kbot = model_grid(a.grid)
    nx, ny, nz = len(xt), len(yt), len(zt)
    depth_t = -zt                                                # m, k=0 bottom .. k=nz-1 surface
    print('model: %d x %d x %d, xt %.2f..%.2f, yt %.2f..%.2f, T-level depths %.0f..%.0f m'
          % (nx, ny, nz, xt[0], xt[-1], yt[0], yt[-1], depth_t[-1], depth_t[0]))

    # ---------------------------------------------------------------- horizontal transfer
    method = a.hinterp
    ix, iy, inside, cnt = source_counts(lon, lat, xt, yt, xu, dxt_deg, yu, dyt_deg)
    per_cell = float(np.median(cnt)) if cnt.size else 0.0
    if method == 'auto':
        method = 'block' if per_cell >= 4.0 else 'bilinear'
    print('horizontal transfer: %s (median %.1f source centres per model cell, '
          '%d cells hold none)' % (method, per_cell, int((cnt == 0).sum())))
    if method == 'block':
        Th, Sh, cnt = block_to_model(lon, lat, zax, T, S, valid, xt, yt,
                                     xu, dxt_deg, yu, dyt_deg, t0)
        empty = cnt == 0
        if empty.any():
            print('  %d model cells hold no source centre: bilinear there'
                  % int(empty.sum()))
            Tb, Sb = bilinear_to_model(lon, lat, zax, T, S, valid, xt, yt, t0,
                                       cells=empty)
            take = np.isfinite(Tb)
            Th = np.where(take, Tb, Th); Sh = np.where(take, Sb, Sh)
    else:
        Th, Sh = bilinear_to_model(lon, lat, zax, T, S, valid, xt, yt, t0)

    # ---------------------------------------------------------------- vertical interpolation to zt
    temp = np.full((nz, ny, nx), np.nan, np.float32); salt = np.full_like(temp, np.nan)
    for k in range(nz):
        d = depth_t[k]
        if d <= zax[0]:
            l, w = 0, 0.0
        elif d >= zax[-1]:
            l, w = len(zax) - 2, 1.0
        else:
            l = int(np.searchsorted(zax, d) - 1); w = (d - zax[l])/(zax[l+1] - zax[l])
        a0, a1 = Th[l], Th[l+1]; b0, b1 = Sh[l], Sh[l+1]
        both = np.isfinite(a0) & np.isfinite(a1)
        temp[k] = np.where(both, (1 - w)*a0 + w*a1, np.where(np.isfinite(a0), a0, a1))   # one side only: that side
        salt[k] = np.where(both, (1 - w)*b0 + w*b1, np.where(np.isfinite(b0), b0, b1))
    del Th, Sh
    valid3 = np.isfinite(temp) & np.isfinite(salt)

    # ---------------------------------------------------------------- mask with kbot
    # pyOM kbot: 1-based index of the deepest wet level counted from the bottom
    # (0 = land); wet where k0 >= kbot - 1 with k0 0-based from the bottom
    kk = np.arange(nz)[:, None, None]
    wet = (kbot[None] > 0) & (kk >= (kbot[None] - 1))
    print('wet cells: %d of %d (%.1f %%); wet columns %d; wet cells without data before the fill: %d'
          % (wet.sum(), wet.size, 100*wet.mean(), (kbot > 0).sum(), int((wet & ~valid3).sum())))
    nh, nsill = fill_connected(temp, salt, valid3, wet)
    print('connectivity fill: %d cells from same-depth neighbours, %d cells from the level above (pockets below their sill)'
          % (nh.sum(), nsill.sum()))
    ks = np.where(nsill > 0)[0]
    if len(ks):
        print('  pocket fill per level (depth m: cells): %s' % ', '.join('%.0f: %d' % (depth_t[k], nsill[k]) for k in ks[::-1][:12]))
    temp[~wet] = 0.0; salt[~wet] = 0.0
    assert np.isfinite(temp).all() and np.isfinite(salt).all()

    # ---------------------------------------------------------------- stabilization
    if a.stabilize:
        temp, salt = stabilize(temp, salt, wet, dzt, tol=a.stabilize_tol)
        temp[~wet] = 0.0; salt[~wet] = 0.0

    # ---------------------------------------------------------------- stability check
    sig = sigma0(temp.astype(np.float64), salt.astype(np.float64))
    dsig = sig[1:] - sig[:-1]                                    # upper minus lower, must be <= 0
    both = wet[1:] & wet[:-1]
    unst = both & (dsig > 0.02)
    print('static stability (sigma_0, upper minus lower > 0.02 kg/m3): %d of %d wet interfaces, max %.3f kg/m3'
          % (unst.sum(), both.sum(), np.max(np.where(both, dsig, -1))))
    worst_ji = None
    if unst.any():
        kk_, jj_, ii_ = np.unravel_index(np.argmax(np.where(both, dsig, -1)), dsig.shape)
        worst_ji = (jj_, ii_)
        print('  worst at lon %.2f lat %.2f between %.0f and %.0f m' % (xt[ii_], yt[jj_], depth_t[kk_], depth_t[kk_+1]))

    # ---------------------------------------------------------------- write
    tmp = a.out + '.tmp'
    with Dataset(tmp, 'w', format='NETCDF4') as f:
        for nm, v in (('xt', xt), ('yt', yt), ('xu', xu), ('yu', yu)):
            f.createDimension(nm, len(v)); vv = f.createVariable(nm, 'f4', (nm,)); vv[:] = v
            vv.long_name = 'Longitude' if nm[0] == 'x' else 'Latitude'; vv.units = 'degrees'
        for nm, v in (('zt', zt), ('zw', zw)):
            f.createDimension(nm, len(v)); vv = f.createVariable(nm, 'f4', (nm,)); vv[:] = v
            vv.long_name = 'depth'; vv.units = 'm'; vv.positive = 'up'
        vt = f.createVariable('temp', 'f4', ('zt', 'yt', 'xt')); vt[:] = temp
        vt.long_name = 'Potential temperature'; vt.units = 'deg C'
        vs = f.createVariable('salt', 'f4', ('zt', 'yt', 'xt')); vs[:] = salt
        vs.long_name = 'Salinity'; vs.units = 'g/kg'
        f.setncatts(dict(source=os.path.abspath(a.src), month=('mean' if a.month == 0 else a.month), grid=os.path.abspath(a.grid),
                         horizontal=method,
                         method='masked bilinear, connectivity fill (nearest same-depth water in the connected region, sill water in pockets), '
                                'bilinear to the model T grid, linear in depth, convective '
                                'adjustment; dry cells 0' if a.stabilize else
                                'bilinear to the model T grid, linear in depth; dry cells 0',
                         created=time.ctime()))
    os.replace(tmp, a.out)
    print('wrote %s  (%.0f s)' % (a.out, time.time() - t0))

    # ---------------------------------------------------------------- column check
    # no column asked for: show the one that ended up least stable, which is
    # the one worth looking at, and the deepest column if none is inverted
    if a.check_column is not None:
        lon0, lat0 = a.check_column
        i0 = int(np.argmin(np.abs(xt - lon0))); j0 = int(np.argmin(np.abs(yt - lat0)))
    elif worst_ji is not None:
        j0, i0 = (int(v) for v in worst_ji)
    else:
        j0, i0 = (int(v) for v in np.unravel_index(np.argmin(np.where(kbot > 0, kbot, nz + 1)), kbot.shape))
    print('column at lon %.2f lat %.2f (kbot %d): depth, temp, salt (new%s)'
          % (xt[i0], yt[j0], kbot[j0, i0], ', old' if a.old else ''))
    told = sold = None
    if a.old:
        with Dataset(a.old) as f:
            told = np.asarray(f['temp'][:, j0, i0]); sold = np.asarray(f['salt'][:, j0, i0])
    for k in range(nz):
        if wet[k, j0, i0]:
            line = '  %6.0f m  %6.2f  %6.3f' % (depth_t[k], temp[k, j0, i0], salt[k, j0, i0])
            if told is not None:
                line += '   old %6.2f  %6.3f' % (told[k], sold[k])
            print(line)


if __name__ == '__main__':
    main()
