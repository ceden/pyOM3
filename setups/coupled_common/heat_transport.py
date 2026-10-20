"""
heat_transport.py -- meridional heat and freshwater transport and the
overturning streamfunction per basin from the online means
(exp_ocean.py, AVERAGE_TRACER_FLUX), per window and as a seasonal cycle:

    python3 heat_transport.py [-o NAME] [-j NPROC] [--grid grid_topo.nc] ocean_mean_icon033_seg*

The blocks of a raw directory are read by NPROC processes (default the
number of cores, at most 16).

Per window (a month in production) and per basin, as a function of latitude:

    heat        rho_0 c_p sum_x sum_z <vT> dx dz                      [PW]  northward positive
    heat_mean   the same with <v> (<T>_j + <T>_j+1)/2, the part the time-mean flow carries
    heat_eddy   heat - heat_mean, the transient (eddy and seasonal) part
    heat_stokes rho_0 c_p sum_x sum_z <v_S T> dx dz, the part the Stokes drift carries
                (coupled runs, from <vstemp>; v is u_L there, so heat contains it;
                zero in a run without waves)
    fw          -(1/S_0) sum_x sum_z (<vS> - S_0 <v>) dx dz, S_0 = 35  [Sv]  freshwater, northward
    moc         -sum_x sum_{z' < z} <v> dx dz'                        [Sv]  at the W faces

<vT> and <vS> are the means of the model's centred advective flux v (T_j +
T_j+1)/2 (AVERAGE_TRACER_FLUX), so the transport is the advective one; the
superbee correction to the centred face value and the vertical mixing do
not move heat meridionally, and icon033 has no GM or Redi.

BASINS from the land mask (kbot of --grid, default grid_topo.nc, else the
cells where <temp> is nonzero): the wet surface cells are cut by four walls,
68 W south of Cape Horn, 20 E south of Africa, 147 E south of Tasmania and
the Bering Strait, and the
connected pieces are labelled. The piece holding (30 W, 0 N) is the
Atlantic with the Arctic and its Southern Ocean sector, the piece holding
(180 E, 0 N) the Indo-Pacific with its sector. Everything is 'global'. A
domain without those coasts (a sector experiment) has the global basin alone.

The windows are grouped by calendar month of their start and averaged over
the years, as ni_report.py does. Written: NAME_transport.nc with
(month, basin, yu) for heat, heat_mean, heat_eddy, fw and (month, basin,
zu, yu) for moc, plus the table NAME_transport.txt with the values at
selected latitudes.
"""
import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy import ndimage

RADIUS, RHO0, CP, S0 = 6370e3, 1024.0, 3991.86795711963, 35.0
BASINS = ('global', 'atlantic', 'indopacific')
LATS = (-30.0, 0.0, 26.0, 45.0)


def _spacing(x):
    d = np.empty_like(x)
    d[1:-1] = 0.5*(x[2:] - x[:-2])
    d[0], d[-1] = x[1] - x[0], x[-1] - x[-2]
    return d


def basin_masks(wet, xt, yt):
    """(nbasin, ny, nx) bool, BASINS order, from the wet surface mask."""
    ny, nx = wet.shape
    lon = np.mod(xt, 360.0)
    cut = wet.copy()
    def wall(lon0, lat_lo, lat_hi, half_width=0.0):
        i = np.argmin(np.abs(np.mod(lon - lon0 + 180.0, 360.0) - 180.0))
        if abs(np.mod(lon[i] - lon0 + 180.0, 360.0) - 180.0) > 2.0:
            return
        j = (yt >= lat_lo) & (yt <= lat_hi)
        cut[j, i] = False
    def wall_lat(lat0, lon_lo, lon_hi):
        j = np.argmin(np.abs(yt - lat0))
        if abs(yt[j] - lat0) > 2.0:
            return
        i = (lon >= lon_lo) & (lon <= lon_hi)
        cut[j, i] = False
    wall(292.0, -90.0, -54.5)          # Drake Passage, Cape Horn southward
    wall(20.0, -90.0, -34.5)           # Cape Agulhas southward
    wall(147.0, -90.0, -43.5)          # Tasmania southward
    wall_lat(65.8, 185.0, 196.0)       # Bering Strait
    lab, n = ndimage.label(cut)
    # cyclic x: merge the labels that touch across the seam
    parent = {k: k for k in range(1, n + 1)}
    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for j in range(ny):
        a, b = lab[j, 0], lab[j, -1]
        if a and b and root(a) != root(b):
            parent[root(a)] = root(b)
    for k in range(1, n + 1):
        lab[lab == k] = root(k)
    # the wall cells themselves are wet and belong to the basin next to them
    for shift, axis in ((-1, 1), (1, 1), (-1, 0), (1, 0)):
        hole = wet & (lab == 0)
        lab[hole] = np.roll(lab, shift, axis=axis)[hole]
    def seed(lon0, lat0):
        dl = np.abs(np.mod(lon - lon0 + 180.0, 360.0) - 180.0)
        i, j = np.argmin(dl), np.argmin(np.abs(yt - lat0))
        if dl[i] > 3.0 or abs(yt[j] - lat0) > 3.0:    # the seed is outside the domain
            return 0
        return lab[j, i]
    masks = [wet]
    for lon0, lat0 in ((330.0, 0.0), (180.0, 0.0)):
        k = seed(lon0, lat0)
        masks.append((lab == k) & wet if k else np.zeros_like(wet))
    return np.array(masks)


class Grid:
    def __init__(self, xt, yt, zt, zu, kbot):
        self.xt, self.yt, self.zt, self.zu = xt, yt, zt, zu
        self.yu = np.empty_like(yt)                    # V points: north faces
        self.yu[:-1] = 0.5*(yt[:-1] + yt[1:]); self.yu[-1] = yt[-1] + 0.5*(yt[-1] - yt[-2])
        dxt = _spacing(xt)
        self.dxv = (RADIUS*np.radians(dxt))[None, :]*np.cos(np.radians(self.yu))[:, None]
        zlo = np.empty_like(zu); zlo[1:] = zu[:-1]; zlo[0] = zu[0] - 2.0*(zu[0] - zt[0])
        self.dz = zu - zlo                             # bottom first, as the fields
        nz = len(zu)
        k = np.arange(1, nz + 1)[:, None, None]        # 1-based from the bottom
        self.wet3 = (kbot[None] > 0) & (k >= kbot[None])
        wet = kbot > 0
        self.wetv3 = self.wet3 & np.roll(self.wet3, -1, axis=1)   # V face j: cells j and j+1
        self.basins = basin_masks(wet, xt, yt)


def read_kbot(grid_file, nx, ny):
    if grid_file and os.path.exists(grid_file):
        with Dataset(grid_file) as f:
            kb = np.asarray(f['kbot'][:], np.int64)
        if kb.shape == (ny, nx):
            return kb
        print('%s: kbot is %s, the means are (%d, %d), using <temp> for the mask'
              % (grid_file, kb.shape, ny, nx))
    return None


class Window:
    def __init__(self, label, ym):
        self.label, self.ym = label, ym
        self.heat = self.heat_mean = self.fw = self.vdxdz = self.heat_stokes = None


def accumulate(W, G, v, temp, vtemp, vsalt, tnorth, j0, i0, vstemp=None):
    """Add one block (nz, ny_b, nx_b), bottom first, to the window sums.
    tnorth is the first row of <temp> of the block to the north (or zeros).
    vstemp is the Stokes flux <v_S T> of the block or None."""
    nz, nyb, nxb = v.shape
    sj, si = slice(j0, j0 + nyb), slice(i0, i0 + nxb)
    m = G.wetv3[:, sj, si]
    dxdz = G.dz[:, None, None]*G.dxv[None, sj, si]*m
    tn = np.concatenate([temp[:, 1:], tnorth[:, None]], axis=1)
    vT_mean = v*0.5*(temp + tn)
    for b in range(len(BASINS)):
        bm = G.basins[b][sj, si][None]
        w = dxdz*bm
        W.heat[b, sj]      += RHO0*CP*np.sum(vtemp*w, axis=(0, 2))
        W.heat_mean[b, sj] += RHO0*CP*np.sum(vT_mean*w, axis=(0, 2))
        W.fw[b, sj]        += -np.sum((vsalt - S0*v)*w, axis=(0, 2))/S0
        W.vdxdz[b, :, sj]  += np.sum(v*w, axis=2)
        if vstemp is not None:
            W.heat_stokes[b, sj] += RHO0*CP*np.sum(vstemp*w, axis=(0, 2))
    return


def _shapes(job):
    n, pes = job
    out = {}
    for pe in pes:
        fn = os.path.join(pe, 'n=%d' % n, 'vtemp.npy')
        out[int(pe.split('=')[-1])] = (np.load(fn, mmap_mode='r').shape
                                        if os.path.exists(fn) else None)
    return out


def _blocks(job):
    """the window sums over the blocks of the chunk, in a worker process"""
    G, items, nbasin, ny, nz = job
    W = Window('', None)
    W.heat = np.zeros((nbasin, ny)); W.heat_mean = np.zeros((nbasin, ny))
    W.fw = np.zeros((nbasin, ny)); W.vdxdz = np.zeros((nbasin, nz, ny))
    W.heat_stokes = np.zeros((nbasin, ny))
    for bd, bdn, j0, i0 in items:
        ld = lambda f: np.load(os.path.join(bd, f + '.npy'))
        v, temp, vtemp, vsalt = ld('v'), ld('temp'), ld('vtemp'), ld('vsalt')
        fs = os.path.join(bd, 'vstemp.npy')
        vstemp = np.load(fs) if os.path.exists(fs) else None
        if bdn is not None:
            tnorth = np.load(os.path.join(bdn, 'temp.npy'), mmap_mode='r')[:, 0, :]
        else:
            tnorth = np.zeros((nz, v.shape[2]))
        accumulate(W, G, v, temp, vtemp, vsalt, np.asarray(tnorth), j0, i0, vstemp)
    return W.heat, W.heat_mean, W.fw, W.vdxdz, W.heat_stokes


def _chunks(seq, k):
    k = max(1, min(k, len(seq)))
    return [seq[i::k] for i in range(k)]


def read_raw(d, grid_file, pool=None, nproc=1):
    """Windows of one raw means directory, the blocks read in chunks by the
    processes of pool (the time is the many small files on the file system)."""
    xt, yt = np.load(os.path.join(d, 'xt.npy')), np.load(os.path.join(d, 'yt.npy'))
    zt, zu = np.load(os.path.join(d, 'zt.npy')), np.load(os.path.join(d, 'zu.npy'))
    npx, npy = (int(v) for v in np.load(os.path.join(d, 'decomposition.npy')))
    pes = sorted(glob.glob(os.path.join(d, 'pe=*')), key=lambda s: int(s.split('=')[-1]))
    steps = sorted(int(os.path.basename(s).split('=')[-1])
                   for s in glob.glob(os.path.join(pes[0], 'n=*')))
    ny, nx, nz = len(yt), len(xt), len(zt)
    run = pool.map if pool is not None else (lambda f, jobs: [f(j) for j in jobs])
    G, out = None, []
    for n in steps:
        shp = {}
        for part in run(_shapes, [(n, c) for c in _chunks(pes, nproc)]):
            shp.update(part)
        if any(v is None for v in shp.values()):
            print('%s n=%d incomplete or without vtemp, skipped' % (d, n)); continue
        blocks = {r: os.path.join(d, 'pe=%d' % r, 'n=%d' % n) for r in shp}
        xoff = np.cumsum([0] + [shp[i][2] for i in range(npx - 1)])
        yoff = np.cumsum([0] + [shp[j*npx][1] for j in range(npy - 1)])
        if G is None:
            kb = read_kbot(grid_file, nx, ny)
            if kb is None:                      # mask from the first window's <temp>
                kb = np.zeros((ny, nx), np.int64)
                for r, bd in blocks.items():
                    bx, by = r % npx, r // npx
                    t = np.load(os.path.join(bd, 'temp.npy'))
                    wet = np.abs(t) > 0.0
                    kbb = np.where(wet.any(axis=0), nz - np.sum(wet, axis=0) + 1, 0)
                    kb[yoff[by]:yoff[by]+t.shape[1], xoff[bx]:xoff[bx]+t.shape[2]] = kbb
            G = Grid(xt, yt, zt, zu, kb)
        items = []
        for r, bd in blocks.items():
            bx, by = r % npx, r // npx
            items.append((bd, blocks[r + npx] if by + 1 < npy else None, yoff[by], xoff[bx]))
        W = Window('n=%d' % n, None)
        W.heat = np.zeros((len(BASINS), ny)); W.heat_mean = np.zeros((len(BASINS), ny))
        W.fw = np.zeros((len(BASINS), ny)); W.vdxdz = np.zeros((len(BASINS), nz, ny))
        W.heat_stokes = np.zeros((len(BASINS), ny))
        for he, hm, fw, vd, hs in run(_blocks, [(G, c, len(BASINS), ny, nz) for c in _chunks(items, nproc)]):
            W.heat += he; W.heat_mean += hm; W.fw += fw; W.vdxdz += vd; W.heat_stokes += hs
        wf = os.path.join(d, 'window_n=%d.npy' % n)
        if os.path.exists(wf):
            tm, t0, t1, nsteps, d0, d1 = np.load(wf)
            W.label = '%d .. %d (%d steps)' % (int(d0), int(d1), int(nsteps))
            W.ym = int(d0) // 10000
        out.append(W)
    return G, out


def read_nc(fn, grid_file):
    f = Dataset(fn)
    xt, yt, zt, zu = (np.asarray(f[v][:]) for v in ('xt', 'yt', 'zt', 'zu'))
    ny, nx, nz = len(yt), len(xt), len(zt)
    out, G = [], None
    for m in range(len(f.dimensions['time'])):
        if 'vtemp' not in f.variables:
            break
        v, temp, vtemp, vsalt = (np.asarray(f[k][m]) for k in ('v', 'temp', 'vtemp', 'vsalt'))
        if G is None:
            kb = read_kbot(grid_file, nx, ny)
            if kb is None:
                wet = np.abs(temp) > 0.0
                kb = np.where(wet.any(axis=0), nz - np.sum(wet, axis=0) + 1, 0)
            G = Grid(xt, yt, zt, zu, kb)
        d0 = int(f['date_start'][m])
        W = Window('%d .. %d (%d steps)' % (d0, int(f['date_end'][m]), int(f['nsteps'][m])),
                   d0 // 10000)
        W.heat = np.zeros((len(BASINS), ny)); W.heat_mean = np.zeros((len(BASINS), ny))
        W.fw = np.zeros((len(BASINS), ny)); W.vdxdz = np.zeros((len(BASINS), nz, ny))
        W.heat_stokes = np.zeros((len(BASINS), ny))
        vstemp = np.asarray(f['vstemp'][m]) if 'vstemp' in f.variables else None
        accumulate(W, G, v, temp, vtemp, vsalt, np.zeros((nz, nx)), 0, 0, vstemp)
        out.append(W)
    f.close()
    return G, out


def moc_of(vdxdz):
    """(nbasin, nz, ny) sums of v dx dz per level -> streamfunction at the W
    faces, -sum over the levels below, Sv."""
    return -np.cumsum(vdxdz, axis=1)*1e-6


def main(args):
    name, grid_file, nproc = 'heat', 'grid_topo.nc', min(os.cpu_count() or 1, 16)
    while args and args[0].startswith('-'):
        if args[0] == '-o':
            name = args[1]
        elif args[0] == '--grid':
            grid_file = args[1]
        elif args[0] == '-j':
            nproc = int(args[1])
        args = args[2:]
    if not args:
        print(__doc__); sys.exit(1)
    windows, G = [], None
    pool = None
    if nproc > 1 and any(os.path.isdir(a) for a in args):
        from multiprocessing import Pool
        pool = Pool(nproc)
    for arg in args:
        Gi, w = (read_raw(arg, grid_file, pool, nproc) if os.path.isdir(arg)
                 else read_nc(arg, grid_file))
        if Gi is not None:          # a segment without complete windows has no grid
            G = Gi
        windows += w
        print('%s: %d windows' % (arg, len(w)), flush=True)
    if pool is not None:
        pool.close()
    if not windows:
        print('no complete windows with vtemp found (AVERAGE_TRACER_FLUX)'); sys.exit(1)
    windows.sort(key=lambda w: (w.ym or 0, w.label))
    have = [b for b in range(len(BASINS)) if G.basins[b].any()]
    print('basins: %s' % ', '.join('%s (%d wet cells)' % (BASINS[b], G.basins[b].sum()) for b in have))

    jl = [int(np.argmin(np.abs(G.yu - la))) for la in LATS]
    head = '%-34s ' % 'window' + ' '.join('%s@%+.0f' % (BASINS[b][:3], la) for b in have for la in LATS)
    lines = [head + '   [heat PW]']
    for w in windows:
        lines.append('%-34s ' % w.label + ' '.join('%7.3f' % (w.heat[b, j]*1e-15) for b in have for j in jl))
    print('\n'.join(lines))

    months = sorted({w.ym % 100 for w in windows if w.ym})
    g = Dataset(name + '_transport.nc', 'w')
    for dn, nn in (('month', 12), ('basin', len(BASINS)), ('yu', len(G.yu)), ('zu', len(G.zu))):
        g.createDimension(dn, nn)
    g.createVariable('month', 'i4', ('month',))[:] = np.arange(1, 13)
    g.createVariable('yu', 'f8', ('yu',))[:] = G.yu
    g.createVariable('zu', 'f8', ('zu',))[:] = G.zu
    g.basins = ', '.join(BASINS)
    vc = g.createVariable('n_windows', 'i4', ('month',)); vc[:] = 0
    V = {}
    for nm, un in (('heat', 'PW'), ('heat_mean', 'PW'), ('heat_eddy', 'PW'), ('heat_stokes', 'PW'), ('fw', 'Sv')):
        V[nm] = g.createVariable(nm, 'f8', ('month', 'basin', 'yu'), fill_value=np.nan); V[nm].units = un
    V['moc'] = g.createVariable('moc', 'f8', ('month', 'basin', 'zu', 'yu'), fill_value=np.nan)
    V['moc'].units = 'Sv'
    g.createDimension('xt', len(G.xt)); g.createVariable('xt', 'f8', ('xt',))[:] = G.xt
    g.createDimension('yt', len(G.yt)); g.createVariable('yt', 'f8', ('yt',))[:] = G.yt
    g.createVariable('basin_mask', 'i1', ('basin', 'yt', 'xt'))[:] = G.basins.astype(np.int8)
    lines.append('\nseasonal cycle (mean over the years), heat [PW]:')
    lines.append(head.replace('window', 'month (n)'))
    for mo in months:
        sel = [w for w in windows if w.ym and w.ym % 100 == mo]
        k = mo - 1
        vc[k] = len(sel)
        heat = np.mean([w.heat for w in sel], axis=0)
        hmean = np.mean([w.heat_mean for w in sel], axis=0)
        V['heat'][k] = heat*1e-15; V['heat_mean'][k] = hmean*1e-15
        V['heat_eddy'][k] = (heat - hmean)*1e-15
        V['heat_stokes'][k] = np.mean([w.heat_stokes for w in sel], axis=0)*1e-15
        V['fw'][k] = np.mean([w.fw for w in sel], axis=0)*1e-6
        V['moc'][k] = moc_of(np.mean([w.vdxdz for w in sel], axis=0))
        lines.append('%-34s ' % ('%02d (%d)' % (mo, len(sel)))
                     + ' '.join('%7.3f' % (heat[b, j]*1e-15) for b in have for j in jl))
    if any(np.any(w.heat_stokes != 0.0) for w in windows):
        lines.append('\nStokes part of the heat transport, rho_0 c_p sum <v_S T> dx dz [PW]:')
        lines.append(head.replace('window', 'month (n)'))
        for mo in months:
            sel = [w for w in windows if w.ym and w.ym % 100 == mo]
            hs = np.mean([w.heat_stokes for w in sel], axis=0)
            lines.append('%-34s ' % ('%02d (%d)' % (mo, len(sel)))
                         + ' '.join('%7.3f' % (hs[b, j]*1e-15) for b in have for j in jl))
    g.close()
    print('\n'.join(lines[len(windows) + 1:]))
    open(name + '_transport.txt', 'w').write('\n'.join(lines) + '\n')
    print('\nwritten: %s_transport.txt, %s_transport.nc' % (name, name))


if __name__ == '__main__':
    main(sys.argv[1:])
