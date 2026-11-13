"""
The Lagrangian frequency filter of draft1 (after Shakespeare and Hogg 2017)
on the particle records of a coupled run (setups/coupled_common/particle_diag.py,
e.g. icon01_particles; Nov 2026, moved here from icon01_particles/filter_particles.py):
the velocities and tracers along each trajectory are low-pass filtered in
time with the cut-off 0.9 f, and the low-pass is evaluated in the middle of
the record. The fixed points (moving = 0) of the same files get the same
filter at fixed positions, the Eulerian filter of the draft.

    python3 -m pyOM3.diagnostics.particles.lagrangian_filter <run directory or record files> [options]

(pyOM3 on the PYTHONPATH, or the file run as a script.)

      --out=NAME         output npz (default lagr_filter_<filter>.npz)
      --filter=fft       the draft's filter: the FFT of the whole record, all
                         frequencies |omega| >= cut |f| removed (no window, no
                         detrend), as in the 2-D and 3-D parts of draft1
               hann      linear detrend + Hann window + the same FFT cut-off,
                         the window divided out at the middle (the leakage check
                         of Nov 3 2026)
               butter    4th-order Butterworth, forward and backward (zero phase,
                         sosfiltfilt)
      --cut=0.9          the cut-off in units of |f|
      --tmid=S           the evaluation time [s, model time as in the files];
                         default the middle record (index N//2). For the band
                         comparison it must be the snapshot's time: for the icon01
                         particle run (run_particles_gpu.sh, 62 days) the middle
                         of the records of both segments is day 31 = 2678400 s,
                         the snapshot of step 18600
      --spectra=2000     particles per seed depth (moving and fixed separately)
                         whose u, v series are kept for the mean frequency
                         spectrum; 0 = none
      --selftest         checks the weights against the direct filters and stops

THE CUT-OFF is local: f = 2 Omega sin(lat) at the particle's latitude at the
evaluation time (Omega = 7.2921e-5 1/s as pyOM3/diagnostics/balance/window.py). The record of N
values at the interval dt resolves omega_k = 2 pi k/(N dt); the FFT filter
keeps k = 0, +-1, .., +-K with K the largest k with omega_k < cut |f|.

HOW IT IS COMPUTED. Every filter here is linear in the series, so its value
at the evaluation index m is a weighted sum over the records,
    x_lp(m) = sum_t W(t) x(t) ,
with W the row m of the filter. For the FFT cut-off W is the Dirichlet kernel
    W(t) = [1 + 2 sum_{k=1..K} cos(2 pi k (m - t)/N)]/N ,
exactly the FFT low-pass at m. For hann and butter W is the row m of the
filter applied to the unit vectors. W depends on the particle only through
K, so the particles are grouped by K (exact for fft; for hann and butter the
group's cut-off is cut |f| at the group's mean |f|, within one frequency bin
of the particle's). The files are then read once, record by record, and the
sums accumulated: the memory is a few arrays of the number of particles, not
the series (6200 records x 350 000 particles x 5 variables are 43 GB).

THE OUTPUT (npz): per particle id, moving, lon0, lat0, depth0 (the seed),
lon, lat, depth (at the evaluation time), f, K; u, v, w, temp, salt (the
values at the evaluation time) and u_lp, v_lp, w_lp, temp_lp, salt_lp (the
low-pass there); t_mid, m, N, dt_rec, cut, filter; and, with --spectra, the
mean kinetic energy spectrum (|U|^2 + |V|^2)/2 of the Hann-windowed series
on the frequency axis omega/|f| (spec_om, bins of 0.05 up to 20) for every
seed depth and moving/fixed (spec, spec_depth0, spec_moving, spec_count).
"""
import os, sys, glob, time
import numpy as np

OMEGA = 7.2921e-5
VARS = ('u', 'v', 'w', 'temp', 'salt')


# ---------------------------------------------------------------------------
# the filters on full series (columns), for the weights and the self-test
# ---------------------------------------------------------------------------
def fft_lowpass(X, dt, cut_omega):
    """X (N, n): FFT, |omega| >= cut_omega removed, back"""
    N = X.shape[0]
    om = 2*np.pi*np.fft.fftfreq(N, d=dt)
    keep = np.abs(om) < cut_omega
    return np.fft.ifft(np.where(keep[:, None], np.fft.fft(X, axis=0), 0.0), axis=0).real


def hann_lowpass(X, dt, cut_omega):
    from scipy.signal import detrend
    N = X.shape[0]
    Xd = detrend(X, axis=0)
    trend = X - Xd
    w = np.hanning(N + 2)[1:-1]                    # no zeros at the ends (the window is divided out)
    return fft_lowpass(Xd*w[:, None], dt, cut_omega)/w[:, None] + trend


def butter_lowpass(X, dt, cut_omega):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cut_omega/(np.pi/dt), btype='low', output='sos')   # sections: the (b, a) form loses digits at low cut-offs
    return sosfiltfilt(sos, X, axis=0)


FILTERS = dict(fft=fft_lowpass, hann=hann_lowpass, butter=butter_lowpass)


def dirichlet_row(N, m, K):
    """the row m of the FFT low-pass keeping k = 0, +-1, .., +-K"""
    t = np.arange(N)
    k = np.arange(1, K + 1)
    if K == 0:
        return np.full(N, 1.0/N)
    return (1.0 + 2.0*np.cos(2*np.pi*np.outer(m - t, k)/N).sum(axis=1))/N


def filter_row(name, N, m, dt, K, cut_omega, chunk=512):
    """the row m of a filter: the filter applied to the unit vectors"""
    if name == 'fft':
        return dirichlet_row(N, m, K)
    row = np.zeros(N)
    f = FILTERS[name]
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        E = np.zeros((N, c1 - c0)); E[np.arange(c0, c1), np.arange(c1 - c0)] = 1.0
        row[c0:c1] = f(E, dt, cut_omega)[m]
    return row


def K_of(f, N, dt, cut):
    """the largest k with 2 pi k/(N dt) < cut |f|"""
    x = cut*np.abs(f)*N*dt/(2*np.pi)
    return np.maximum(np.ceil(x).astype(np.int64) - 1, 0)


# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    N, dt, m = 600, 1000.0, 300
    ok = True
    for name in ('fft', 'hann', 'butter'):
        for K, cutom in ((0, 0.5*2*np.pi/(N*dt)), (7, 7.5*2*np.pi/(N*dt)), (31, 31.4*2*np.pi/(N*dt))):
            X = rng.normal(size=(N, 5)) + np.linspace(0, 3, N)[:, None]
            direct = FILTERS[name](X, dt, cutom)[m]
            w = filter_row(name, N, m, dt, K, cutom)
            d = float(np.max(np.abs(w @ X - direct))/np.max(np.abs(direct)))
            good = d < 1e-10
            ok &= good
            print('selftest %-6s K %2d: weights against the direct filter %.2e %s' % (name, K, d, 'PASS' if good else 'FAIL'))
    # K_of against the mask of fft_lowpass
    for f in (7e-5, 9.3e-5, 1.1e-4):
        om = 2*np.pi*np.fft.fftfreq(N, d=dt)
        nkeep = int(np.sum(np.abs(om) < 0.9*f))
        K = int(K_of(np.array([f]), N, dt, 0.9)[0])
        good = nkeep == 2*K + 1
        ok &= good
        print('selftest K_of f=%.1e: K %d, the FFT keeps %d = 2K+1 %s' % (f, K, nkeep, 'PASS' if good else 'FAIL'))
    return ok


# ---------------------------------------------------------------------------
def main(argv):
    opt = lambda key, default: next((a[len(key) + 3:] for a in argv if a.startswith('--%s=' % key)), default)
    if '--selftest' in argv:
        return 0 if selftest() else 1
    args = [a for a in argv if not a.startswith('--')]
    if not args:
        print(__doc__); return 1
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, '*_particles_seg*.nc'))) if os.path.isdir(a) else [a]
    files = sorted(files)
    if not files:
        print('no record files found in', args); return 1
    name = opt('filter', 'fft'); cut = float(opt('cut', 0.9))
    if name not in FILTERS:
        print('--filter must be one of', list(FILTERS)); return 1
    nspec = int(opt('spectra', 2000))
    out = opt('out', 'lagr_filter_%s.npz' % name)
    from netCDF4 import Dataset
    ds = [Dataset(f) for f in files]
    times = [np.array(d['time'][:]) for d in ds]
    T = np.concatenate(times)
    seg = np.concatenate([np.full(len(t), i) for i, t in enumerate(times)])
    rec = np.concatenate([np.arange(len(t)) for t in times])
    N = len(T)
    dts = np.diff(T)
    dt = float(dts[0])
    if N < 2 or np.max(np.abs(dts - dt)) > 1e-6*dt:
        print('the records are not equally spaced (%d records, intervals %g to %g s): a segment missing?'
              % (N, dts.min(), dts.max())); return 1
    ids = [np.array(d['id'][:]) for d in ds]
    if any(not np.array_equal(ids[0], i) for i in ids[1:]):
        print('the files carry different particles'); return 1
    tmid = opt('tmid', None)
    if tmid is None:
        m = N//2
    else:
        m = int(np.argmin(np.abs(T - float(tmid))))
        if abs(T[m] - float(tmid)) > 1e-6*dt:
            print('no record at t = %s s (nearest %.1f s)' % (tmid, T[m])); return 1
    print('%d files, %d records every %g s (%.2f days), evaluation at record %d, t = %.1f s = %.3f days; '
          'filter %s, cut-off %g |f|' % (len(files), N, dt, N*dt/86400., m, T[m], T[m]/86400., name, cut), flush=True)
    d0 = ds[seg[m]]; r0 = rec[m]
    o = {k: np.array(d0[k][:]) for k in ('id', 'moving', 'lon0', 'lat0', 'depth0')}
    for k in ('lon', 'lat', 'depth') + VARS:
        o[k] = np.array(d0[k][r0, :], dtype=np.float64)
    f = 2*OMEGA*np.sin(np.deg2rad(o['lat']))
    K = K_of(f, N, dt, cut)
    o['f'], o['K'] = f, K
    Ku, gidx = np.unique(K, return_inverse=True)
    print('%d particles (%d moving, %d fixed); f %.3e to %.3e 1/s, K %d to %d (%d groups), frequency resolution %.3e 1/s'
          % (len(K), int(o['moving'].sum()), int((o['moving'] == 0).sum()), np.abs(f).min(), np.abs(f).max(),
             Ku.min(), Ku.max(), len(Ku), 2*np.pi/(N*dt)), flush=True)
    t0 = time.time()
    W = np.zeros((len(Ku), N))
    for g, k in enumerate(Ku):
        fg = np.abs(f[gidx == g]).mean()
        W[g] = filter_row(name, N, m, dt, int(k), cut*fg)
    print('weights of the %d groups: %.0f s; sum of the weights %.6f to %.6f (1 for a low-pass)'
          % (len(Ku), time.time() - t0, W.sum(axis=1).min(), W.sum(axis=1).max()), flush=True)
    # the spectra subset: per seed depth and moving flag
    sel = np.zeros(0, np.int64)
    groups = []
    if nspec > 0:
        rng = np.random.default_rng(0)
        for mv in (1, 0):
            for dep in np.unique(o['depth0']):
                ii = np.nonzero((o['moving'] == mv) & (o['depth0'] == dep))[0]
                if len(ii) == 0:
                    continue
                pick = np.sort(rng.choice(ii, size=min(nspec, len(ii)), replace=False))
                groups.append((dep, mv, len(sel), len(sel) + len(pick)))
                sel = np.concatenate([sel, pick])
    Su = np.zeros((N, len(sel)), np.float32); Sv = np.zeros((N, len(sel)), np.float32)
    acc = {k: np.zeros(len(K)) for k in VARS}
    t0 = time.time()
    for t in range(N):
        d = ds[seg[t]]; r = rec[t]
        wt = W[gidx, t]
        for k in VARS:
            x = np.asarray(d[k][r, :], dtype=np.float64)
            acc[k] += wt*x
            if k == 'u' and len(sel):
                Su[t] = x[sel]
            elif k == 'v' and len(sel):
                Sv[t] = x[sel]
        if t % 500 == 0 or t == N - 1:
            print('   record %d of %d, %.0f s' % (t + 1, N, time.time() - t0), flush=True)
    for k in VARS:
        o[k + '_lp'] = acc[k]
    o.update(t_mid=T[m], m=m, N=N, dt_rec=dt, cut=cut, filter=name, files=np.array(files))
    if len(sel):
        bins = np.arange(0.0, 20.0 + 1e-9, 0.05)
        om = 2*np.pi*np.fft.rfftfreq(N, d=dt)
        w = np.hanning(N)[:, None]
        spec = np.zeros((len(groups), len(bins) - 1)); cnt = np.zeros((len(groups), len(bins) - 1))
        for g, (dep, mv, a, b) in enumerate(groups):
            E = 0.5*(np.abs(np.fft.rfft(Su[:, a:b]*w, axis=0))**2 + np.abs(np.fft.rfft(Sv[:, a:b]*w, axis=0))**2)
            E = E*dt/(np.pi*N*np.mean(w**2))                    # one-sided density per rad/s
            x = om[:, None]/np.abs(f[sel[a:b]])[None, :]
            idx = np.digitize(x, bins) - 1
            ok = (idx >= 0) & (idx < len(bins) - 1)
            # density per unit omega/|f|: times |f| of the particle
            Ef = E*np.abs(f[sel[a:b]])[None, :]
            np.add.at(spec[g], idx[ok], Ef[ok]); np.add.at(cnt[g], idx[ok], 1.0)
        o.update(spec_om=0.5*(bins[1:] + bins[:-1]), spec=spec/np.maximum(cnt, 1),            # mean over particles and frequencies in the bin
                 spec_depth0=np.array([g[0] for g in groups]), spec_moving=np.array([g[1] for g in groups]),
                 spec_count=np.array([g[3] - g[2] for g in groups]))
    np.savez(out, **o)
    print('written %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
