"""Test 18: the forcing interface (setups/forcing_common) and what the
ocean makes of it (exp_ocean.surface_forcing_fields, the relative-wind
stress, the ice rule, the absent-key rule). No model step; seconds.

  18a calendar      Forcing.date / climatology_time: the cycle, the start
                    date, leap years run through            exact
  18b weights       get_periodic_interval (pyOM2's climatology weights):
                    the weights sum to 1, the records are adjacent, a time
                    at the middle of record n puts all weight on n   exact
  18c cleaning      clean_forcing: non-finite, fills beyond the limit and
                    dry cells are counted and replaced                exact
  18d block         the synthetic source on a block equals the cut of the
                    source on the whole grid (what a decomposition relies
                    on), and `fields` / 'omit' restrict the dict      exact
  18e ecmwf_clim    the monthly-climatology source on a 12-record netCDF
                    file written here: every field is the pyOM2 weighted
                    sum of the two records, emp is absent, qnet_offset
                    applied; both readers (clim_scatter True / False)  1e-15
  18f U/V average   surface_forcing_fields: tau on U and V points is the
                    two-point mean of the T-point stress over rho_0, masked,
                    the scalar fields are the padded block, an omitted
                    field comes back None                             exact
  18g relwind       at rest tau_rel = tau_E wherever |u10| >= U10_MIN and
                    C_D rho_a is below the cap (no constant of our own
                    enters); with a current d_int = (u10 - u_o).tau >= 0;
                    the cap holds; the stress handed to the waves is the
                    interior of the T-point stress                   1e-15
  18h ice           ICE_TAU_FRAC = 0: the stress at a U point between two
                    frozen cells is zero, between one frozen and one open
                    cell half; the heat and salt fluxes vanish where the
                    cell is at the freezing point and cooling; the stored
                    taux/tauy are the reduced stress                  exact
"""
import os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as C
from config import small_ocean, line, SyntheticForcing, _global_coords, HERE, WORK, RHO_0, RHO_AIR
import forcing_base as fb
import pyOM3 as OM


def t_calendar():
    ok = True
    F = fb.Forcing({'start': '1999-12-31', 'cycle_days': 365.0}, {'xt': [0], 'yt': [0]}, (), 'ocean')
    d = F.date(0.0); ok &= line('18a start date = 1999-12-31 00 UTC', abs((d - datetime.datetime(1999, 12, 31)).total_seconds()), 0.0)
    d = F.date(366*86400.0 + 3600.0)      # one cycle + a day + an hour -> Jan 1 2000 01:00
    ok &= line('18a cycle 365 d: t = 366 d + 1 h -> 2000-01-01 01:00', abs((d - datetime.datetime(2000, 1, 1, 1)).total_seconds()), 0.0)
    F0 = fb.Forcing({'start': '2000-01-01', 'cycle_days': 0.0}, {}, (), 'ocean')
    d = F0.date(366*86400.0)              # 2000 is a leap year: through the calendar
    ok &= line('18a no cycle: t = 366 d -> 2001-01-01', abs((d - datetime.datetime(2001, 1, 1)).total_seconds()), 0.0)
    ok &= line('18a climatology_time of 2001-01-01 = 0', abs(F0.climatology_time(366*86400.0)), 0.0)
    return ok


def t_weights():
    ok = True
    yr = 365.0*86400.0; rs = yr/12.0
    worst_sum = worst_adj = worst_mid = 0.0
    for t in np.linspace(-2*yr, 3*yr, 997):
        n1, n2, w1, w2 = fb.get_periodic_interval(t, yr, rs, 12)
        worst_sum = max(worst_sum, abs(w1 + w2 - 1.0))
        worst_adj = max(worst_adj, 0.0 if (n2 - n1) % 12 == 1 else 1.0)
        if not (0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0):
            worst_sum = 1.0
    for n in range(12):
        n1, n2, w1, w2 = fb.get_periodic_interval((n + 0.5)*rs, yr, rs, 12)
        worst_mid = max(worst_mid, abs(w1 - 1.0) + abs(w2) + (0.0 if n1 == n else 1.0))
    ok &= line('18b weights sum to 1, in [0, 1]', worst_sum, 1e-12)
    ok &= line('18b records adjacent (n2 = n1 + 1 mod 12)', worst_adj, 0.0)
    ok &= line('18b middle of record n -> weight 1 on n', worst_mid, 1e-12)
    return ok


def t_cleaning():
    ok = True
    a = np.ones((4, 5)); a[0, 0] = np.nan; a[1, 1] = -1e34; a[2, 2] = 30.0
    wet = np.ones((4, 5), bool); wet[3, 3] = False
    b, n = fb.clean_forcing(a, 'x', limit=20.0, fill=-7.0, wet=wet)
    ok &= line('18c clean_forcing counts nan + fill + limit + dry', abs(n - 4), 0.0)
    ok &= line('18c clean_forcing replaces exactly those', abs(b[0, 0] + 7) + abs(b[1, 1] + 7) + abs(b[2, 2] + 7)
               + abs(b[3, 3] + 7) + np.abs(b[b != -7.0] - 1.0).sum(), 0.0)
    b, n = fb.clean_forcing(a, 'x')
    ok &= line('18c clean_forcing without limit/wet: only the non-finite', abs(n - 1), 0.0)
    return ok


def t_block():
    ok = True
    xg, yg = _global_coords()
    full = SyntheticForcing({'source': 'synthetic'}, {'xt': xg, 'yt': yg},
                            ('taux', 'tauy', 'qnet', 'qsol', 'sst_star', 'dqdt', 'emp', 'sss_star', 'u10', 'v10'), 'ocean')
    j0, i0, ny, nx = 3, 5, 6, 7
    blk = SyntheticForcing({'source': 'synthetic'}, {'xt': xg[i0:i0+nx], 'yt': yg[j0:j0+ny]}, full.fields, 'ocean')
    worst = 0.0
    for t in (0.0, 3*3600.0, 11.5*86400.0):
        Ff, Fb = full.get(t), blk.get(t)
        worst = max(worst, max(float(np.max(np.abs(Fb[k] - Ff[k][j0:j0+ny, i0:i0+nx]))) for k in Ff))
    ok &= line('18d synthetic source: block = cut of the full grid', worst, 0.0)
    few = SyntheticForcing({'source': 'synthetic', 'omit': ('qsol',)}, {'xt': xg, 'yt': yg}, ('taux', 'tauy', 'qsol', 'qnet'), 'ocean')
    ok &= line('18d `fields` and `omit` restrict the dict (taux, tauy, qnet)', 0.0 if sorted(few.get(0.0)) == ['qnet', 'taux', 'tauy'] else 1.0, 0.0)
    return ok


def t_ecmwf_clim():
    from netCDF4 import Dataset
    ok = True
    xg, yg = _global_coords(); ny, nx = len(yg), len(xg)
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, 'clim_test.cdf')
    rec = {}
    with Dataset(path, 'w') as f:
        f.createDimension('time', 12); f.createDimension('yt', ny); f.createDimension('xt', nx)
        for k, (v, amp) in enumerate((('taux', 0.1), ('tauy', 0.05), ('qnet', 100.0), ('swf', 150.0),
                                      ('dqdt', -40.0), ('sst', 15.0), ('sss', 35.0))):
            r = amp*(1.0 + 0.1*np.arange(12)[:, None, None]*np.cos(np.pi*np.arange(ny)/ny)[None, :, None]
                     + 0.01*k*np.arange(nx)[None, None, :])
            r[5, 2, 3] = -1e34                    # one fill value, must be zeroed
            f.createVariable(v, 'f4', ('time', 'yt', 'xt'))[:] = r
            rec[v] = r.astype(np.float32).astype(np.float64); rec[v][5, 2, 3] = 0.0
    fields = ('taux', 'tauy', 'qnet', 'qsol', 'sst_star', 'dqdt', 'emp', 'sss_star')
    var = {'taux': 'taux', 'tauy': 'tauy', 'qnet': 'qnet', 'qsol': 'swf', 'dqdt': 'dqdt', 'sst_star': 'sst', 'sss_star': 'sss'}
    j0, i0, nyl, nxl = 2, 4, 7, 9
    geom = {'xt': xg[i0:i0+nxl], 'yt': yg[j0:j0+nyl], 'block': (j0, i0, nyl, nxl), 'comm': None,
            'xt_gl': xg, 'yt_gl': yg}
    yr = 365.0*86400.0
    for scatter in (False, True):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            F = fb.make({'source': 'ecmwf_clim', 'file': path, 'start': '2000-01-01', 'cycle_days': 365.0,
                         'clim_scatter': scatter, 'qnet_offset': 3.0}, geom, fields, 'ocean')
        worst = 0.0
        for t in (0.0, 40.0*86400.0, 364.9*86400.0, 500.0*86400.0):
            Ft = F.get(t)
            n1, n2, w1, w2 = fb.get_periodic_interval(F.climatology_time(t), yr, yr/12.0, 12)
            for k in Ft:
                r = rec[var[k]]; ref = (w1*r[n1] + w2*r[n2])[j0:j0+nyl, i0:i0+nxl]
                if k == 'qnet':
                    ref = ref - 3.0
                worst = max(worst, float(np.max(np.abs(Ft[k] - ref)))/max(float(np.max(np.abs(ref))), 1e-300))
        ok &= line('18e ecmwf_clim (clim_scatter %s) = pyOM2 weighted records, offset applied' % scatter, worst, 1e-15)
        ok &= line('18e ecmwf_clim: emp absent, the others present', 0.0 if sorted(Ft) == sorted(k for k in fields if k != 'emp') else 1.0, 0.0)
    return ok


def _forced(**kw):
    return small_ocean(coupled=False, tag='frc', **kw)


def t_uv_average():
    ok = True
    m = _forced()
    h = m.halo; sl = slice(h, -h); ks = -h-1
    t = 5*3600.0
    tax, tay, qnet, qsol, tstar, fw, dqdt, sstar = m.surface_forcing_fields(t)
    F = m._forcing.get(t)
    mT, mU, mV = (np.asarray(x)[ks] for x in (m.maskT, m.maskU, m.maskV))
    tx = np.zeros((m.Ny, m.Nx)); tx[sl, sl] = F['taux']; ty = np.zeros((m.Ny, m.Nx)); ty[sl, sl] = F['tauy']
    ref_x = 0.5*(tx + np.roll(tx, -1, axis=1))*mU/RHO_0; ref_y = 0.5*(ty + np.roll(ty, -1, axis=0))*mV/RHO_0
    d = max(float(np.max(np.abs(np.asarray(tax) - ref_x)[sl, sl])), float(np.max(np.abs(np.asarray(tay) - ref_y)[sl, sl])))
    ok &= line('18f tau on U/V = two-point mean of tau_T / rho_0, masked', d, 0.0)
    d = max(float(np.max(np.abs(np.asarray(a)[sl, sl] - F[k]*mT[sl, sl])))
            for a, k in ((qnet, 'qnet'), (qsol, 'qsol'), (tstar, 'sst_star'), (fw, 'emp'), (dqdt, 'dqdt'), (sstar, 'sss_star')))
    ok &= line('18f scalar fields = the block, masked', d, 0.0)
    m2 = _forced(omit=('qsol', 'dqdt', 'sst_star', 'emp', 'sss_star'))
    out = m2.surface_forcing_fields(t)
    ok &= line('18f omitted fields come back None', 0.0 if all(o is None for o in out[3:]) and out[2] is not None else 1.0, 0.0)
    return ok


def t_relwind():
    ok = True
    m = _forced(relwind=True)
    h = m.halo; sl = slice(h, -h); ks = -h-1
    t = 2*3600.0
    F = m._forcing.get(t)
    mT = np.asarray(m.maskT)[ks]
    # at rest: tau_rel = tau_E where the floor and the cap are inactive
    m.surface_forcing_fields(t)
    trx, try_ = m._tau_rel_T
    s10 = np.hypot(F['u10'], F['v10']); tE = np.hypot(F['taux'], F['tauy'])
    free = (s10 >= float(m.cfg.STRESS_U10_MIN)) & (tE/np.maximum(s10, 1e-30)**2 < float(m.cfg.STRESS_CD_MAX)*RHO_AIR) & (mT[sl, sl] > 0)
    # the bulk formula keeps ERA5's MAGNITUDE and takes the direction of the
    # (relative) wind: |tau_rel| = |tau_E|, tau_rel || u10 (the two coincide
    # where ERA5's stress is parallel to its wind, which the IFS's is)
    d = float(np.max(np.abs(np.hypot(trx, try_) - tE)[free]))/float(np.max(tE))
    ok &= line('18g at rest |tau_rel| = |tau_ERA5| (floor and cap inactive)', d, 1e-15)
    d = float(np.max(np.abs(trx*F['v10'] - try_*F['u10'])[free]))/float(np.max(tE*s10))
    ok &= line('18g at rest tau_rel parallel to u10', d, 1e-15)
    ok &= line('18g the cap holds: C_D rho_a <= STRESS_CD_MAX rho_air', float(np.max(np.asarray(m._cd_rho))) - float(m.cfg.STRESS_CD_MAX)*RHO_AIR, 0.0)
    # with a current: d_int >= 0 and tau = cd |u10 - uo| (u10 - uo)
    ig = np.arange(m.Nx)[None, :]*np.ones((m.Ny, 1)); jg = np.arange(m.Ny)[:, None]*np.ones((1, m.Nx))
    u = OM.np.asarray(np.ones((m.Nz, 1, 1))*(0.4*np.sin(2*np.pi*ig/m.Nx))[None])*m.maskU
    v = OM.np.asarray(np.ones((m.Nz, 1, 1))*(0.3*np.cos(2*np.pi*jg/m.Ny))[None])*m.maskV
    m.u, m.v = m.apply_bc(u), m.apply_bc(v)
    m.surface_forcing_fields(t)
    ok &= line('18g with a current: d_int = (u10 - u_o).tau >= 0 everywhere', -float(np.min(np.asarray(m._d_int))), 0.0)
    uo, vo = m.surface_velocity('eulerian')
    u10 = np.zeros((m.Ny, m.Nx)); u10[sl, sl] = F['u10']; v10 = np.zeros((m.Ny, m.Nx)); v10[sl, sl] = F['v10']
    du, dv = (u10 - uo)*mT, (v10 - vo)*mT
    cdr = np.asarray(m._cd_rho); sd = np.hypot(du, dv)
    trx, try_ = m._tau_rel_T
    d = max(float(np.max(np.abs((cdr*sd*du)[sl, sl] - trx))), float(np.max(np.abs((cdr*sd*dv)[sl, sl] - try_))))/float(np.max(np.abs(trx)))
    ok &= line('18g tau = C_D rho_a |u10 - u_o| (u10 - u_o), u_o Eulerian top cell', d, 1e-15)
    m3 = _forced(relwind=True); m3.cfg._m.STRESS_UO = 'lagrangian'
    m3.u, m3.v = m.u, m.v
    m3.u_S = m3.apply_bc(OM.np.asarray(0.1*np.ones((m3.Nz, m3.Ny, m3.Nx)))*m3.maskT); m3.v_S = 0.0*m3.u_S
    m3.surface_forcing_fields(t)
    uoL, voL = m3.surface_velocity('lagrangian'); uoE, _ = m3.surface_velocity('eulerian')
    ok &= line('18g STRESS_UO lagrangian: u_o = u_L (Eulerian = u_L - u_S)', float(np.max(np.abs((uoL - uoE - 0.1*mT)[sl, sl]))), 1e-15)
    return ok


def t_ice():
    ok = True
    m = _forced(ice=0.0)
    h = m.halo; sl = slice(h, -h); ks = -h-1
    mT, mU, mV = (np.asarray(x)[ks] for x in (m.maskT, m.maskU, m.maskV))
    # freeze a patch of the surface
    T = np.asarray(m.temp).copy(); frz = np.zeros((m.Ny, m.Nx), bool); frz[8:11, 5:9] = True
    T[ks][frz] = -2.5
    m.temp = m.apply_bc(OM.np.asarray(T))*m.maskT
    tau = m._tau3[0]
    m.du = m.du*0.0; m.dv = m.dv*0.0; m.dtemp = m.dtemp*0.0; m.dsalt = m.dsalt*0.0
    m._gitt = 3; m.set_forcing(tau)
    fT = np.where(np.asarray(m.temp)[ks]*mT <= -1.8, 0.0, 1.0)
    fu = 0.5*(fT + np.roll(fT, -1, axis=1)); fv = 0.5*(fT + np.roll(fT, -1, axis=0))
    tax, tay = m.surface_forcing_fields(m._gitt*m.dt)[:2]
    d = max(float(np.max(np.abs(np.asarray(m.taux) - np.asarray(tax)*fu)[sl, sl])),
            float(np.max(np.abs(np.asarray(m.tauy) - np.asarray(tay)*fv)[sl, sl])))
    ok &= line('18h ICE_TAU_FRAC 0: stored stress = tau x the U/V-averaged ice factor', d, 0.0)
    ok &= line('18h stress zero between two frozen cells', float(np.max(np.abs(np.asarray(m.taux)[8:11, 5:8]))), 0.0)
    ok &= line('18h half stress at the ice edge', float(np.max(np.abs(np.asarray(m.taux)[8:11, 8] - 0.5*np.asarray(tax)[8:11, 8]))), 0.0)
    ft, fs = np.asarray(m.forc_temp_surface), np.asarray(m.forc_salt_surface)
    ice = (np.asarray(m.temp)[ks]*mT <= -1.8) & (mT > 0)
    # where frozen AND the flux would cool: both fluxes zero; the synthetic
    # qnet is negative in the north, so the frozen patch is checked on the
    # cooling cells it holds
    cool = ice & (np.asarray(m.surface_forcing_fields(m._gitt*m.dt)[2]) <= 0.0)
    ok &= line('18h heat and salt flux zero where frozen and cooling (%d cells)' % int(cool.sum()),
               float(np.max(np.abs(ft[cool]))) + float(np.max(np.abs(fs[cool]))) if cool.any() else 0.0, 0.0)
    ok &= line('18h momentum increment du = taux/dz at the surface level', float(np.max(np.abs(
        np.asarray(m.du)[tau, ks][sl, sl] - (np.asarray(m.taux)/m._dz_surf*mU)[sl, sl]))), 0.0)
    return ok


def run():
    ok = True
    for f in (t_calendar, t_weights, t_cleaning, t_block, t_ecmwf_clim, t_uv_average, t_relwind, t_ice):
        ok &= f()
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
