"""
the band method on a window of a raw snapshot of a pe_realistic run
(Nov 2026; the general form of balance_cmp1/band_icon033.py, same method,
same output)

  python3 -m pyOM3.diagnostics.balance.run_window <raw_snap dir> <step n> <lon0> <lon1> <lat0> <lat1> [order]
          [--forcing] [--lowpass=2] [--emit [--periods=4] [--sponge=8]] [--lanczos=150] [--rim=N] [--edge=N] [--out=FILE]
          [--dt=S] [--Ahbi=A] [--Khbi=K] [--eq_of_state=5] [--rho_0=1024] [--free_surface=implicit|explicit [--sub_cycling=12]]
          [--era5=DIR] [--tag=T] [--start=YYYY-MM-DD] [--cycle=D]

(pyOM3 on the PYTHONPATH, or the file run as a script.)

THE PARAMETERS OF THE RUN come from run_params.json in the snapshot
directory, which setups/coupled_common/run.py writes into every raw
snapshot directory since Nov 2026 (window.write_run_params): eq_of_state,
rho_0, dt, Ahbi, Khbi, the free surface (implicit or split-explicit with
sub_cycling and fb_theta), and the forcing settings of the run (ERA5: directory, tag, start date, cycle).
Every option above overrides the file, and stands in for it for older runs
without one; then --dt and --Ahbi must be given (eq_of_state 5 and rho_0
1024 are the defaults, as in band_icon033.py). The driver prints every
parameter with where it came from. dt only enters the rigid-lid term,
which cancels, and the emission runs; Ahbi (and Khbi) the friction term of
the second order. The cosPower factor of the run's friction
(biha_friction_cosPower) is NOT applied, as in band_icon033.py; the driver
prints the run's value. The window model's own physics is that of band_icon033.py
whatever the run had: no vertical mixing, superbee tracer advection, the
tracer flux through the free surface (these enter the emission runs only).

THE FORCING TERM: --forcing adds the ERA5 stress of the run (from
run_params.json), --era5=DIR does so with that directory (and --tag,
--start, --cycle where given, else the file's, else icon033, 1990-05-01,
365), low-passed with a Hann window of half width --lowpass days (default
2) around the snapshot's time, interpolated to the window and spread over
the mixed layer (Delta rho = 0.03 kg/m^3 from 10 m) as the body force F,
with dF/dt from the same window (window.era5_stress_lowpass,
mixed_layer_depth, wind_body_force). The output then ends in _F.npz and
carries the stress, the mixed layer depth and F.

The window is the T cells with lon0 <= xt < lon1, lat0 <= yt < lat1 of the
global grid, closed by walls, with the topography and the state of the
snapshot (window.window_model). The free surface of the window model is
the run's (or --free_surface): the balance does not depend on it (the band operator is the
rigid-lid one in both cases, the results are identical), only the emission
runs do (the split-explicit window needs half the run's time step there,
--dt=108 for icon033). --emit runs the emission test (window.emission: the
plain pe_realistic model, superbee tracer advection, no mixing, a sponge of
--sponge cells from the walls, sampled beyond sponge + 2 cells, --periods
inertial periods; with the forcing term the low-pass stress as a steady
body force), a rough measure. --lanczos=N: the Lanczos steps (default 150,
against 600 steps P X is accurate to 3e-7 and q(B) N to 3e-5 on the icon033
window). --edge=N leaves N cells along the walls out of all printed
statistics (the fields in the npz are complete). --rim=N tapers the state
to the background over N cells from the walls; the balanced state is
meaningful beyond about 10 cells from the walls in any case.

Writes band_<n>_<lon0>_<lat0>[_F].npz (or --out) with the state and the
balanced state of every order (u, v, temp, salt, p_s of the window
interior), the fast tendency field |r|/f of the state and of every order
(fast_state, fast<o>: band_quality), and prints the residual per order,
the quality (the rms of |r|/f relative to the state's) and the
convergence of the orders.
"""
import os, sys, time
if __package__ in (None, ''):                    # run as a file: pyOM3 is three levels up
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import numpy as np


DEFAULTS = dict(eq_of_state=5, rho_0=1024., free_surface='implicit', sub_cycling=12)
ERA5_DEFAULTS = dict(tag='icon033', start='1990-05-01', cycle_days=365.)
OPTIONS = ('lowpass', 'periods', 'sponge', 'lanczos', 'rim', 'edge', 'out', 'dt', 'Ahbi', 'Khbi', 'eq_of_state',
           'rho_0', 'free_surface', 'sub_cycling', 'fb_theta', 'era5', 'tag', 'start', 'cycle')
FLAGS = ('--forcing', '--emit')


def parameters(path, opt):
    """
    the window's model parameters (window_model's par), the forcing settings
    (None without the forcing term) and the report lines, from the options
    opt (dict of the --key=value strings), run_params.json in path, and the
    defaults, in this order
    """
    from pyOM3.diagnostics.balance.window import read_run_params
    rp = read_run_params(path)
    run = (rp or {}).get('model', {})
    src = {}

    def get(key, conv, default=None, run_key=None, run_val=None):
        rk = run_key or key
        if key in opt:
            src[key] = '--%s' % key
            return conv(opt[key])
        if run_val is not None or rk in run:
            src[key] = 'run_params.json'
            return conv(run_val if run_val is not None else run[rk])
        if default is not None:
            src[key] = 'default'
            return conv(default)
        raise SystemExit('run_window: no %s: %s has no run_params.json with it, give --%s'
                         % (key, path, key))

    fs_run = None
    if 'explicit_free_surface' in run or 'implicit_free_surface' in run:
        fs_run = 'explicit' if run.get('explicit_free_surface') else 'implicit'
    fs = get('free_surface', str, DEFAULTS['free_surface'], run_val=fs_run)
    if fs not in ('implicit', 'explicit'):
        raise SystemExit('run_window: --free_surface=implicit or explicit')
    par = dict(eq_of_state=get('eq_of_state', int, DEFAULTS['eq_of_state']),
               rho_0=get('rho_0', float, DEFAULTS['rho_0']),
               implicit_free_surface=(fs == 'implicit'), explicit_free_surface=(fs == 'explicit'),
               sub_cycling=get('sub_cycling', int, DEFAULTS['sub_cycling']),
               # the window's own physics, whatever the run had (as band_icon033.py):
               # no vertical mixing; superbee tracer advection and the tracer flux
               # through the free surface, without which the emission runs blow up
               # at the walls (second-order advection) and in the top cells (split-
               # explicit window); the band operator itself uses neither
               vmix_scheme='explicit', superbee_advection=True, free_surface_tracer_flux=True,
               dt=get('dt', float), Ahbi=get('Ahbi', float))
    if fs == 'explicit' and ('fb_theta' in opt or 'fb_theta' in run):
        par['fb_theta'] = get('fb_theta', float)
    if 'Khbi' in opt or float(run.get('Khbi', 0.) or 0.) != 0.:
        par['Khbi'] = get('Khbi', float)
    lines = ['parameters of %s (%s):' % (path, 'run_params.json of %s%s, written %s'
             % (rp.get('exp_name', '?'), rp.get('segment', ''), rp.get('written', '?')) if rp else 'no run_params.json')]
    for k in ('eq_of_state', 'rho_0', 'dt', 'Ahbi', 'Khbi', 'free_surface', 'sub_cycling', 'fb_theta'):
        if k in src and (k not in ('sub_cycling', 'fb_theta') or fs == 'explicit'):
            v = fs if k == 'free_surface' else par[k]
            lines.append('   %-24s %-12s (%s)' % (k, ('%g' % v) if isinstance(v, float) else v, src[k]))
    if 'biha_friction_cosPower' in run:
        lines.append('   %-24s %-12g (the run\'s, not applied in the window)' % ('biha_friction_cosPower',
                                                                                 run['biha_friction_cosPower']))
    lines.append('   window physics: no vertical mixing, superbee advection, free-surface tracer flux (fixed)')
    # the forcing term
    forcing = None
    if '--forcing' in opt.get('_flags', ()) or 'era5' in opt:
        f = (rp or {}).get('forcing') or {}
        if f and f.get('source', 'era5') != 'era5' and 'era5' not in opt:
            raise SystemExit('run_window: the run was forced by %s, the forcing term needs ERA5 (--era5=DIR)'
                             % f.get('source'))
        fsrc = {}
        def fget(key, okey, conv):
            if okey in opt:
                fsrc[key] = '--%s' % okey; return conv(opt[okey])
            if key in f and f[key] is not None:
                fsrc[key] = 'run_params.json'; return conv(f[key])
            if key in ERA5_DEFAULTS:
                fsrc[key] = 'default'; return conv(ERA5_DEFAULTS[key])
            raise SystemExit('run_window: the forcing term needs the ERA5 directory: --era5=DIR '
                             '(%s has no run_params.json with it)' % path)
        forcing = dict(dir=fget('dir', 'era5', str), tag=fget('tag', 'tag', str),
                       start=fget('start', 'start', str), cycle_days=fget('cycle_days', 'cycle', float),
                       half_width=float(opt.get('lowpass', 2.0)))
        fsrc['half_width'] = '--lowpass' if 'lowpass' in opt else 'default'
        lines.append('forcing term: ERA5 stress, low-pass')
        for k in ('dir', 'tag', 'start', 'cycle_days', 'half_width'):
            lines.append('   %-24s %-12s (%s)' % (k, forcing[k], fsrc[k]))
    return par, forcing, lines


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    if len(args) < 6:
        print(__doc__)
        raise SystemExit(1)
    opt = dict(a[2:].split('=', 1) for a in argv if a.startswith('--') and '=' in a)
    opt['_flags'] = [a for a in argv if a.startswith('--') and '=' not in a]
    unknown = [k for k in opt if k != '_flags' and k not in OPTIONS] + [a for a in opt['_flags'] if a not in FLAGS]
    if unknown:
        raise SystemExit('run_window: unknown options %s (see the docstring)' % ' '.join(unknown))
    path, n = args[0], int(args[1])
    lon0, lon1, lat0, lat1 = [float(a) for a in args[2:6]]
    order = int(args[6]) if len(args) > 6 else 2
    emit = '--emit' in opt['_flags']
    rim = int(opt.get('rim', 0))
    edge = int(opt.get('edge', 0))               # cells along the walls left out of the printed statistics

    from pyOM3.diagnostics.balance.window import (read_raw, read_coords, window_model, era5_stress_lowpass,
                                                  mixed_layer_depth, wind_body_force)
    PAR, ERA5, lines = parameters(path, opt)
    print('\n'.join(lines), flush=True)

    t0 = time.time()
    co = read_coords(path)
    xt, yt = co['xt'], co['yt']
    i0, i1 = int(np.searchsorted(xt, lon0)), int(np.searchsorted(xt, lon1))
    j0, j1 = int(np.searchsorted(yt, lat0)), int(np.searchsorted(yt, lat1))
    if i1 - i0 < 4 or j1 - j0 < 4:
        raise SystemExit('run_window: the window i %d:%d j %d:%d is empty or too small' % (i0, i1, j0, j1))
    snap = read_raw(path, n, window=(i0, i1, j0, j1))          # only the blocks of the window
    print('snapshot step %d, t = %.1f days, window i %d:%d j %d:%d (%d x %d x %d), read in %.0f s'
          % (n, snap['t']/86400., i0, i1, j0, j1, i1 - i0, j1 - j0, len(snap['zt']), time.time() - t0), flush=True)
    M = window_model(snap, i0, i1, j0, j1, PAR, rim=rim)
    I = M._interior
    h = M.halo
    wet = I(M.maskT) > 0
    if edge > 0:
        wet[:, :edge, :] = wet[:, -edge:, :] = wet[:, :, :edge] = wet[:, :, -edge:] = False
    print('wet cells %d of %d, f %.2e to %.2e' % (wet.sum(), wet.size, np.abs(np.asarray(M.coriolis_t)[h:-h, h:-h]).min(),
                                                np.abs(np.asarray(M.coriolis_t)[h:-h, h:-h]).max()), flush=True)
    M.init_balance()
    M.init_band(om_cut=0.5, om_width=0.3, m_lanczos=int(opt.get('lanczos', 150)), verbose=True, check=True)
    u0, v0, b0, _ = M.model_state()
    F = Fd = None
    if ERA5 is not None:
        t1 = time.time()
        tx, ty, dtx, dty = era5_stress_lowpass(ERA5['dir'], ERA5['tag'], ERA5['start'], ERA5['cycle_days'], snap['t'],
                                               np.asarray(M.xt)[h:-h], np.asarray(M.yt)[h:-h], ERA5['half_width'])
        hml = mixed_layer_depth(M)
        F = wind_body_force(M, tx, ty, hml); Fd = wind_body_force(M, dtx, dty, hml)
        wet2 = I(M.maskT)[-1] > 0
        print('forcing term: low-pass stress (half width %.1f d) rms %.3f N/m^2, |dtau/dt| rms %.2e N/m^2/s, mixed layer %.0f m mean (%.0f to %.0f), '
              'Ekman transport tau/(rho f) %.2f m^2/s rms, %.0f s'
              % (ERA5['half_width'], np.sqrt(np.mean((tx**2 + ty**2)[wet2])), np.sqrt(np.mean((dtx**2 + dty**2)[wet2])),
                 hml[wet2].mean(), hml[wet2].min(), hml[wet2].max(),
                 np.sqrt(np.mean((np.hypot(tx, ty)/(M.rho_0*np.abs(np.asarray(M.coriolis_t)[h:-h, h:-h])))[wet2]**2)), time.time() - t1), flush=True)
    rms = lambda a: np.sqrt(np.mean(np.asarray(a)[wet]**2)) if np.asarray(a).ndim == 3 else np.sqrt(np.mean(np.asarray(a)**2))
    # copies, not views of the model arrays: the emission runs change the model state
    out = dict(u=np.array(I(M.u)), v=np.array(I(M.v)), temp=np.array(I(M.temp)), salt=np.array(I(M.salt)), p_s=np.array(M.band_ps())[h:-h, h:-h],
               xt=np.asarray(M.xt)[h:-h], yt=np.asarray(M.yt)[h:-h], zt=np.asarray(M.zt)[h:-h], maskT=np.array(I(M.maskT)), t=snap['t'])
    prev = []
    for o in range(order + 1):
        t = time.time()
        B = M.band_balanced(o, forcing=F, forcing_dot=Fd)
        u, v, temp, salt, p_s = M.band_apply_TS(B)
        res = [rms(I(B[k]) - I(x))/max(rms(I(x)), 1e-30) for k, x in ((0, u0), (1, v0), (3, b0))]
        print('order %d: %.0f s, Lanczos steps %d, |X - Bal(X)|/|X| (u v b) %.3e %.3e %.3e, temp change rms %.3e K, blocked throughflow %.2e of the state'
              % (o, time.time() - t, M.band_its, *res, rms(I(temp) - I(M.temp)), M.band_phi_rms), flush=True)
        # the quality: the fast tendency of the balanced state against the state's, and the convergence of the orders
        t = time.time()
        q = M.band_quality(B, forcing=F)
        step = ''
        if o >= 1:
            d1 = np.sqrt(np.mean((I(B[0]) - prev[-1][0])[wet]**2 + (I(B[1]) - prev[-1][1])[wet]**2))
            step = ', change against order %d %.3e of the state' % (o - 1, d1/rms(np.hypot(I(u0), I(v0))))
            if o >= 2:
                d2 = np.sqrt(np.mean((prev[-1][0] - prev[-2][0])[wet]**2 + (prev[-1][1] - prev[-2][1])[wet]**2))
                step += ', ratio to the previous change %.2e' % (d1/max(d2, 1e-30))
        print('   quality: fast tendency %.3f of the state\'s (|r|/f %.2e against %.2e m/s per inertial period/2pi)%s, %.0f s'
              % (rms(q['field'])/rms(q['field_state']), rms(q['field']), rms(q['field_state']), step, time.time() - t), flush=True)
        prev.append((I(B[0]), I(B[1])))
        out.update({'u_bal%d' % o: np.array(I(u)), 'v_bal%d' % o: np.array(I(v)), 'temp_bal%d' % o: np.array(I(temp)),
                    'p_s_bal%d' % o: np.array(p_s)[h:-h, h:-h], 'fast%d' % o: q['field']})
        if o == 0:
            out['fast_state'] = q['field_state']
    if F is not None:
        out.update(taux=tx, tauy=ty, hml=hml, Fu=F[0], Fv=F[1])
    fn = opt.get('out', 'band_%d_%g_%g%s.npz' % (n, lon0, lat0, '_F' if F is not None else ''))
    np.savez(fn, **out)
    print('written %s' % fn, flush=True)
    if emit:
        from pyOM3.diagnostics.balance.window import emission
        periods = float(opt.get('periods', 4))
        sponge = int(opt.get('sponge', 8)); sedge = max(edge, sponge + 2)
        kw = dict(periods=periods, forcing=F, sponge=sponge, edge=sedge)
        t = time.time()
        # the states from `out` (copies): band_apply_TS returns the model's own
        # salt array, which the previous emission run has changed
        e = emission(M, out['u'], out['v'], out['temp'], out['salt'], out['p_s'], **kw)
        print('emission of the state %.3e (%.1f inertial periods%s, sponge %d cells, sampled beyond %d cells from the walls, %.0f s)'
              % (e, periods, ', steady low-pass wind' if F is not None else '', sponge, sedge, time.time() - t), flush=True)
        for o in range(order + 1):
            e = emission(M, out['u_bal%d' % o], out['v_bal%d' % o], out['temp_bal%d' % o], out['salt'], out['p_s_bal%d' % o], **kw)
            print('emission of the order %d %.3e' % (o, e), flush=True)
    return fn


if __name__ == '__main__':
    main(sys.argv[1:])
