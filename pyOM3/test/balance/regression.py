"""
Test 24: regression of the balance diagnostics. Every method of the
current code runs on the small cases of config.py, and the fingerprints of
its output fields (config.fingerprint: the 2-norm and three products with
fixed random vectors) are compared with those stored in golden.json. Written
before the balance code moved into pyOM3/diagnostics/balance (Nov 2026): the
pure move reproduces them exactly, a refactoring must do so to round-off.

    python3 regression.py               all cases against golden.json
    python3 regression.py --update      rewrite golden.json (after an intended change)
    python3 regression.py case ...      only these cases
    python3 regression.py --sensitivity the deviation of every case when its
                                        initial velocity is changed by 1e-13
                                        (no golden.json needed): how far
                                        round-off propagates, the floor for
                                        the tolerance after a refactoring

The deviation of an entry is max(|dl|, |dr_i|)/L, dl and dr_i the changes of
the norm l and of the three products r_i, and L the largest reference norm
in the case of the entry's group of variables (velocity u, v, w and the
velocity residual of band_quality; momentum tendencies du, dv; momentum
forcing Fu, Fv; buoyancy b, temp, salt; pressure p, p_s; any other name its
own group): the higher orders and w are measured against the size of the
flow, not against their own size. One line per case: the largest deviation
of its entries relative to their tolerance (1e-10; 1e-7 for the residual of
band_quality, see TOL_QUALITY; 1e-8 for the fourth order of nh_simple, see
TOL_NH4). --sensitivity with 1e-13 moved the cases by
at most 4e-9 of the flow (the fourth order of nh_simple) and the
band_quality residual by 2e-6 (Nov 12 2026). The same code on a Mac and on
Linux differs about as much (nh order 4 2.4e-9, band_quality 7e-7, the skew
of the window 2e-10), so golden.json is written with --update on the machine
that runs the test.
"""
import json, os, sys, time, shutil, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ps_model, pr_model, nh_model, pr_snapshot, fingerprint, quiet, line, HERE,
                    N0, NX, NY, NZ)
GOLD = os.path.join(HERE, 'golden.json')
TOL = 1e-10
# the residual of band_quality, the fast tendency of a balanced state, is the
# Lanczos projection of a tendency dominated by its fast part: a 1e-13 change
# of the input moves it by 2e-6 of the flow (--sensitivity, Nov 12 2026),
# Mac against Linux by 7e-7, jax against numpy by 1.4e-6 (the slow
# projection of that tendency differs by 8e-3 between 60 and 300 Lanczos
# steps, Nov 13 2026). Its entries are compared with TOL_QUALITY
TOL_QUALITY = 1e-7
# the fourth order of nh_simple (decompose4/order4) amplifies round-off by
# about 4e4 (--sensitivity), and p3dfft plans its FFTs with FFTW_MEASURE: the
# algorithm, and so the round-off, can change from run to run (1.4e-9 seen in
# the full fast suite, Nov 13 2026). Compared with TOL_NH4
TOL_NH4 = 1e-8
I = lambda m, a: np.asarray(a)[m.halo:-m.halo, m.halo:-m.halo, m.halo:-m.halo]
I2 = lambda m, a: np.asarray(a)[m.halo:-m.halo, m.halo:-m.halo]
NAMES6 = ('u', 'v', 'w', 'b', 'p', 'p_s')


def _fields(m, out, tag, X, names):
    for a, nm in zip(X, names):
        a = np.asarray(a)
        out['%s/%s' % (tag, nm)] = I(m, a) if a.ndim == 3 else (I2(m, a) if a.ndim == 2 else a)


def _orders(m, out, tag, X):
    for n, x in enumerate(X):
        _fields(m, out, '%s/order%d' % (tag, n), x, ('u', 'v', 'w', 'b', 'p_s'))


def _state(m, out, tag, names=('u', 'v', 'w', 'temp')):
    for nm in names:
        out['%s/%s' % (tag, nm)] = I(m, getattr(m, nm))
    out['%s/p_s' % tag] = I2(m, np.asarray(m.p_s)[0] if np.asarray(m.p_s).ndim == 3 else m.p_s)


# ----------------------------------------------------------------------
# pe_simple, Warn et al. (warn_pe_simple)
def case_ps_warn():
    from pyOM3.diagnostics.balance.warn import warn_pe_simple
    out = {}
    m = ps_model((warn_pe_simple,))
    with quiet():
        m.init_balance(N0)
        _orders(m, out, 'decompose4', m.decompose(4))
        _fields(m, out, 'project+1', m.project(1), NAMES6)
        _fields(m, out, 'project0', m.project(0), NAMES6)
        st = m.model_state()
        out['norm'] = m.norm((st[0], st[1], m.w, st[2]), m.decompose(0)[0][:4])
        m.balance(2)
    _state(m, out, 'balance2')
    return out


def case_ps_warn_options():
    from pyOM3.diagnostics.balance.warn import warn_pe_simple
    out = {}
    m = ps_model((warn_pe_simple,))
    with quiet():
        m.init_balance(N0, nsteps=4, pTf1_numeric=True)
        _orders(m, out, 'pTf1_numeric', m.decompose(2))
        m.init_balance(N0, slave_modes='auto')
        _orders(m, out, 'auto', m.decompose(3))
        m.init_balance(N0, slave_modes=2, damping=False)
        _orders(m, out, 'cut2_nodamp', m.decompose(2))
    return out


def case_ps_warn_fs():
    from pyOM3.diagnostics.balance.warn import warn_pe_simple
    out = {}
    m = ps_model((warn_pe_simple,), free_surface=True)
    with quiet():
        m.init_balance(N0)
        _orders(m, out, 'decompose3', m.decompose(3))
    return out


def case_ps_warn_n2_beta():
    from pyOM3.diagnostics.balance.warn import warn_pe_simple
    out = {}
    m = ps_model((warn_pe_simple,), n2=True)
    with quiet():
        m.init_balance(m.case_N0)
        _orders(m, out, 'n2', m.decompose(2))
    m = ps_model((warn_pe_simple,), beta=2e-11)
    with quiet():
        m.init_balance(N0)
        _orders(m, out, 'beta', m.decompose(2))
        out['beta/pv_its'] = float(m.bal_pv_its)
    return out


# ----------------------------------------------------------------------
# pe_simple, kernel (kernel_pe_simple)
def case_ps_kernel():
    from pyOM3.diagnostics.balance.kernel import kernel_pe_simple
    out = {}
    m = ps_model((kernel_pe_simple,))
    with quiet():
        m.init_balance(N0)
        m.init_kernel()
        # the iterations of the conjugate gradients are kept: a change of the
        # preconditioner changes them, the converged fields only by pcg_tol
        _fields(m, out, 'kernel2', m.kernel_balanced(2), NAMES6)
        out['kernel2/keep2'] = np.asarray(m.ker_keep2, dtype='f8')
        out['kernel2/its'] = float(m.ker_its)
        _fields(m, out, 'kernel1_all', m.kernel_balanced(1, per_mode=False), NAMES6)
        m.init_kernel(R_over_L=3., floor_cells=4)
        _fields(m, out, 'kernel2_R3', m.kernel_balanced(2), NAMES6)
        out['kernel2_R3/its'] = float(m.ker_its)
        m.init_kernel(trunc=False)
        _fields(m, out, 'untrunc2', m.kernel_balanced(2, per_mode=False), NAMES6)
    m = ps_model((kernel_pe_simple,), beta=2e-11)
    with quiet():
        m.init_balance(N0)
        m.init_kernel()
        _fields(m, out, 'beta2', m.kernel_balanced(2), NAMES6)
        out['beta2/its'], out['beta2/pv_its'] = float(m.ker_its), float(m.ker_pv_its)
    return out


# ----------------------------------------------------------------------
# pe_simple, band (band_pe_simple). 60 Lanczos steps instead of 300: the
# regression needs the code path, not the converged filter (300 steps: 3 min)
ML = 60
def _forcing(m):
    """a wind on the top cell (interior tendencies) and its time derivative"""
    yt = (np.arange(NY) + 0.5)*m.dy
    Fu = np.zeros((NZ, NY, NX)); Fu[-1] = 1e-7*np.cos(2*np.pi*yt/(NY*m.dy))[:, None]
    return (Fu, 0.*Fu, 0.*Fu), (1e-5*Fu, 0.*Fu, 0.*Fu)


def case_ps_band():
    from pyOM3.diagnostics.balance.band import band_pe_simple
    out = {}
    m = ps_model((band_pe_simple,))
    with quiet():
        m.init_balance(N0)
        m.init_band(m_lanczos=ML)
        out['skew'] = m.band_skew
        _fields(m, out, 'band2', m.band_balanced(2), NAMES6)
        out['band2/its'] = float(m.band_its)
        F, Fd = _forcing(m)
        _fields(m, out, 'band2_forced', m.band_balanced(2, forcing=F, forcing_dot=Fd), NAMES6)
        m.init_band(dt_mode='model', om_cut=0.6, om_width=0.2, m_lanczos=ML)
        _fields(m, out, 'band2_model', m.band_balanced(2), NAMES6)
        m.init_band(om_width=0., m_lanczos=ML, check=False)
        _fields(m, out, 'band1_sharp', m.band_balanced(1), NAMES6)
    return out


def case_ps_band_topo_n2():
    from pyOM3.diagnostics.balance.band import band_pe_simple
    out = {}
    m = ps_model((band_pe_simple,), seamount=True)
    with quiet():
        m.init_balance(N0)
        m.init_band(m_lanczos=ML)
        out['seamount/skew'] = m.band_skew
        _fields(m, out, 'seamount2', m.band_balanced(2), NAMES6)
    m = ps_model((band_pe_simple,), n2=True)
    with quiet():
        m.init_balance(m.case_N0)
        for norm in ('modal', 'diag', 'tlevel'):
            m.init_band(n2_norm=norm, m_lanczos=ML)
            _fields(m, out, 'n2_%s' % norm, m.band_balanced(2), NAMES6)
    return out


# ----------------------------------------------------------------------
# pe_simple, optimal balance (optimal_pe_simple)
OB = dict(OB_period=0.25*86400., OB_average=2*np.pi/1e-4, OB_average_times=1, OB_max_itts=2)


def case_ps_ob():
    from pyOM3.diagnostics.balance.optimal import optimal_pe_simple
    from pyOM3.diagnostics.balance.kernel import kernel_pe_simple
    out = {}
    m = ps_model((optimal_pe_simple,), wave=True)
    with quiet():
        m.balance_it(**OB)
    _state(m, out, 'average')
    m = ps_model((optimal_pe_simple,), wave=True)
    with quiet():
        m.balance_it(**OB, OB_window='hann', OB_friction=False, exchange_base_point=False)
    _state(m, out, 'hann_nofric_nobase')
    for proj in ('fft', 'kernel'):
        m = ps_model((kernel_pe_simple, optimal_pe_simple), wave=True)
        with quiet():
            m.balance_it(**OB, OB_projection=proj)
        _state(m, out, proj)
    return out


# ----------------------------------------------------------------------
# pe_realistic: band and optimal balance on the Cartesian jets
def case_pr_band():
    from pyOM3.diagnostics.balance.band import band_pe_realistic
    out = {}
    m = pr_model((band_pe_realistic,))
    with quiet():
        m.init_balance()
        m.init_band(m_lanczos=ML)
        out['skew'] = m.band_skew
        B = m.band_balanced(2)
        _fields(m, out, 'band2', B, NAMES6)
        q = m.band_quality(B)
        # the residual is kept in m/s, measured against the flow (the ratio
        # 'fast' of band_quality is a small residual over its own size:
        # 5e-5 under --sensitivity)
        out['quality/fast_abs'] = q['fast_abs']
        out['quality/field'], out['quality/field_state'] = q['field'], q['field_state']
        u, v, b = m.model_state()[:3]
        _fields(m, out, 'fast_tendency', m.band_fast_tendency(u, v, b), ('du', 'dv', 'db'))
    return out


def case_pr_ob():
    # the method of pe_simple since step 3 (Nov 2026): golden.json rewritten
    # for this case, the old pe_realistic backward integration did not ramp
    # down the non-linear terms and its time averaging used the friction
    from pyOM3.diagnostics.balance.optimal import optimal_pe_realistic
    out = {}
    m = pr_model((optimal_pe_realistic,))
    with quiet():
        m.balance_it(**OB)
    _state(m, out, 'average')
    m = pr_model((optimal_pe_realistic,))
    with quiet():
        m.balance_it(**OB, OB_window='hann', OB_friction=False)
    _state(m, out, 'hann_nofric')
    return out


# ----------------------------------------------------------------------
# pe_realistic: a window of a raw snapshot (window.py), equation of state,
# topography, spherical grid
def _window_case(explicit):
    from pyOM3.diagnostics.balance import window as BW
    out = {}
    d = tempfile.mkdtemp(prefix='bal_raw_')
    try:
        with quiet():
            par = pr_snapshot(d)
            if explicit:
                par.update(implicit_free_surface=False, explicit_free_surface=True, sub_cycling=12)
            snap = BW.read_raw(d, 20, window=(2, 14, 1, 11))
            co = BW.read_coords(d)
            out['read/u'], out['read/temp'], out['read/p_s'] = snap['u'], snap['temp'], snap['p_s']
            out['read/wet'] = snap['wet'].astype('f8')
            out['coords/xt'], out['coords/zu'] = co['xt'], co['zu']
            M = BW.window_model(snap, 3, 13, 2, 10, par, rim=2)
            out['kbot'] = np.asarray(M.win_kbot, dtype='f8')
            M.init_balance()
            out['N2w'] = M.bal_N2w
            M.init_band(m_lanczos=ML)
            out['skew'] = M.band_skew
            hml = BW.mixed_layer_depth(M)
            out['hml'] = hml
            h = M.halo
            xt, yt = np.asarray(M.xt)[h:-h], np.asarray(M.yt)[h:-h]
            taux = 0.1*np.cos(np.deg2rad(yt))[:, None]*np.ones(len(xt))[None]
            tauy = 0.02*np.sin(np.deg2rad(2*xt))[None]*np.ones(len(yt))[:, None]
            F = BW.wind_body_force(M, taux, tauy, hml)
            _fields(M, out, 'forcing', F, ('Fu', 'Fv', 'Fb'))
            B = M.band_balanced(2, forcing=F)
            _fields(M, out, 'band2', B, NAMES6)
            for k, a in zip(('u', 'v', 'temp', 'salt', 'p_s'), M.band_apply_TS(B)):
                a = np.asarray(a)
                out['TS/%s' % k] = (I(M, a) if a.shape[-1] == M.Nx else a) if a.ndim == 3 else \
                                   (I2(M, a) if a.shape[-1] == M.Nx else a)
            q = M.band_quality(B, forcing=F)
            out['quality/fast_abs'], out['quality/field'] = q['fast_abs'], q['field']
            u, v, t, s, ps = [I(M, a) for a in (M.u, M.v, M.temp, M.salt)] + [I2(M, M.band_ps())]
            r = BW.rim_taper(I(M, M.maskT) > 0, u, v, t, s, ps, 3)
            for k, a in zip(('u', 'v', 'temp', 'salt', 'p_s'), r):
                out['rim3/%s' % k] = a
            out['emission'] = BW.emission(M, u, v, t, s, ps, periods=0.25, forcing=F, edge=1, sponge=2)
            _state(M, out, 'after_emission', ('u', 'v', 'temp', 'salt'))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def case_pr_window():
    return _window_case(False)


def case_pr_window_explicit():
    return _window_case(True)


# ----------------------------------------------------------------------
# nh_simple, Warn et al. (warn_nh_simple)
def case_nh_warn():
    out = {}
    m = nh_model()
    with quiet():
        X = m.decompose(4)
        for n, x in enumerate(X):
            for a, nm in zip(x, ('u', 'v', 'w', 'b')):
                out['decompose4/order%d/%s' % (n, nm)] = I(m, a)
        out['norm'] = m.norm((m.u, m.v, m.w, m.b), X[0])
        m.balance(3)
    for nm in ('u', 'v', 'w', 'b'):
        out['balance3/%s' % nm] = I(m, getattr(m, nm))
    m = nh_model(periodic_z=True, slave_modes='auto')
    with quiet():
        X = m.decompose(3)
        for n, x in enumerate(X):
            for a, nm in zip(x, ('u', 'v', 'w', 'b')):
                out['periodic_auto/order%d/%s' % (n, nm)] = I(m, a)
    return out


CASES = [k[5:] for k in list(globals()) if k.startswith('case_')]


# ----------------------------------------------------------------------
def compute(names):
    res, times = {}, {}
    for c in names:
        t = time.time()
        try:
            out = globals()['case_' + c]()
        except ModuleNotFoundError as e:                # nh_simple needs the p3dfft build
            print('24 %s: skipped (%s)' % (c, e)); continue
        res[c] = {k: fingerprint(v) for k, v in out.items()}
        times[c] = time.time() - t
    return res, times


GROUPS = dict(u='vel', v='vel', w='vel', field='vel', field_state='vel', fast_abs='vel', du='tend', dv='tend',
              Fu='force', Fv='force', b='buo', temp='buo', salt='buo', p='pre', p_s='pre')


def _group(key):
    var = key.split('/')[-1]
    return GROUPS.get(var, var)


def _tol(key, case=None):
    if key.startswith('quality/'):
        return TOL_QUALITY
    if case == 'nh_warn' and '/order4/' in key:
        return TOL_NH4
    return TOL


def deviation(new, ref, case=None):
    """the largest deviation of the entries of a case relative to their
    tolerance: (deviation, entry, tolerance of the entry)"""
    scale = {}
    for k, f in ref.items():
        scale[_group(k)] = max(scale.get(_group(k), 0.), f[0])
    # band_skew, the check of init_band that L is skew-adjoint, is a
    # dimensionless number at round-off level (8e-17 on the periodic jets):
    # its own size is no scale, it is compared absolutely
    scale['skew'] = 1.
    worst, where, wt = 0., '', TOL
    for k, f in ref.items():
        if k not in new:
            return np.inf, k + ' missing', TOL
        g = new[k]
        if not np.all(np.isfinite(g)):
            return np.inf, k + ' not finite', TOL
        d = max(abs(a - b) for a, b in zip(f, g))/max(scale[_group(k)], 1e-300)
        if d/_tol(k, case) > worst/wt:
            worst, where, wt = d, k, _tol(k, case)
    extra = set(new) - set(ref)
    if extra:
        return np.inf, 'new entries ' + ', '.join(sorted(extra)[:3]), TOL
    return worst, where, wt


def sensitivity(names):
    """the deviation of every case under a 1e-13 change of its initial velocity"""
    import config
    ref, _ = compute(names)
    config.PERTURB = 1e-13
    try:
        new, _ = compute(names)
    finally:
        config.PERTURB = 0.
    for c in ref:
        d, where, t = deviation(new[c], ref[c], c)
        print('24 sensitivity %-22s %9.2e  at %s (tolerance %.0e)' % (c, d, where, t))
    return True


def run(update=False, names=None):
    names = names or CASES
    res, times = compute(names)
    if update or not os.path.exists(GOLD):
        old = json.load(open(GOLD)) if os.path.exists(GOLD) else {}
        old.update(res)
        json.dump(old, open(GOLD, 'w'), indent=0)
        print('24 balance regression: %s %s in %s' % (', '.join(res), 'rewritten' if update else 'created', GOLD))
        return True
    ref = json.load(open(GOLD))
    ok = True
    for c in res:
        if c not in ref:
            print('24 %s: no reference in golden.json (run --update)' % c); ok = False; continue
        d, where, t = deviation(res[c], ref[c], c)
        ok &= line('24 balance %-22s (%3d entries, %4.0f s)' % (c, len(ref[c]), times[c]), d, t)
        if d > t:
            print('      largest at %s' % where)
    return ok


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--sensitivity' in sys.argv:
        sys.exit(0 if sensitivity(args or CASES) else 1)
    sys.exit(0 if run(update='--update' in sys.argv, names=args or None) else 1)
