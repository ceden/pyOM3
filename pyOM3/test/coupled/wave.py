"""Test 21: the wave side of the shared code (exp_wave.wave_common on
wim_realistic): the land mask from the ocean's topography, the wind from
the forcing object, the coarser wave grid, the 'coupled' source of the
relative-wind stress, and the two energy diagnostics. No ocean involved.

  21a mask        kbot = 1 where the ocean's kbot_global > 0, 0 on land,
                  on the ocean grid; coarsened by 2 a wave cell is sea if
                  ANY ocean cell of its group is wet                 exact
  21b wind        u* = sqrt(|tau_a|/rho_air), udir = atan2(tau_e, tau_n)
                  from the source at the model time, masked          1e-14
  21c group mean  coarsened by 2: the wave cell's stress is the area mean
                  of tau over its ocean cells (weights cos(lat) dxt dyt),
                  formed independently here from the full-grid source,
                  u* from the MEAN VECTOR (not the mean of u*)       1e-15
  21d coupled     STRESS_MODE relwind: the forcing is the 'coupled' source
                  on the wave's own grid; F.set() of a block changes u*
                  and udir accordingly, before the first set() u* = 0,
                  every wet cell seeded                               exact
  21e wind input  sum of wind_input_field x A_t / rho_0 = wind_work, after
                  one step, both grids                             round-off
  21f snl work    snl_work = 0 without a current, the Doppler projection
                  of a uniform current through calc_doppler_vel makes
                  it the resolved int U.k S_nl dk evaluated here     1e-12
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_wave, SyntheticForcing, _global_coords, kbot_global, line, RHO_AIR, RHO_0, DXT, DYT
import pyOM3 as OM


def _step(m, n=1):
    tau, taum1, taum2 = 2, 1, 0
    for k in range(n):
        tA, tB, tC = (1., 0., 0.) if k == 0 else (1.55, -0.55, 0.)
        m.time_step(tA, tB, tC, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
    return m


def _full_source(t):
    xg, yg = _global_coords()
    F = SyntheticForcing({'source': 'synthetic'}, {'xt': xg, 'yt': yg}, ('taux', 'tauy'), 'wave')
    return F.get(t)


def run():
    ok = True
    import io, contextlib
    # ---- the ocean grid ----
    m = small_wave(coarsen=1)
    h = m.halo; sl = slice(h, -h)
    kb = np.asarray(m.kbot)[sl, sl]
    ok &= line('21a mask on the ocean grid = (kbot_global > 0)', float(np.max(np.abs(kb - (kbot_global() > 0)))), 0.0)
    t = 7*3600.0
    m.update_forcing(t)
    F = _full_source(t)
    us = np.sqrt(np.hypot(F['taux'], F['tauy'])/RHO_AIR)*(kbot_global() > 0)
    ud = np.arctan2(F['taux'], F['tauy'])
    # (1e-14: the model's coordinates come through the halo recursion, 4e-15 off the plain one)
    ok &= line('21b u* = sqrt(|tau|/rho_air), masked', float(np.max(np.abs(np.asarray(m.ustar)[sl, sl] - us))), 1e-14)
    ok &= line('21b udir = atan2(tau_east, tau_north)', float(np.max(np.abs(np.asarray(m.udir)[sl, sl] - ud))), 1e-14)
    # ---- the coarse grid ----
    c = small_wave(coarsen=2)
    hc = c.halo; sc = slice(hc, -hc)
    je, ie = c._setup._je_w, c._setup._ie_w
    kbg = kbot_global() > 0
    kbc = np.maximum.reduceat(np.maximum.reduceat(kbg.astype(float), je[:-1], axis=0), ie[:-1], axis=1)
    ok &= line('21a mask coarsened by 2 = any wet ocean cell of the group', float(np.max(np.abs(np.asarray(c.kbot)[sc, sc] - kbc))), 0.0)
    c.update_forcing(t)
    xg, yg = _global_coords()
    w = (np.cos(np.deg2rad(yg))*DYT)[:, None]*DXT[None, :]
    gs = lambda a: np.add.reduceat(np.add.reduceat(a, je[:-1], axis=0), ie[:-1], axis=1)
    tx = gs(w*F['taux'])/gs(w); ty = gs(w*F['tauy'])/gs(w)
    usc = np.sqrt(np.hypot(tx, ty)/RHO_AIR)*kbc
    d = float(np.max(np.abs(np.asarray(c.ustar)[sc, sc] - usc)))/float(np.max(usc))
    ok &= line('21c coarse u* from the area-mean stress vector', d, 1e-15)
    ok &= line('21c coarse udir from the mean vector', float(np.max(np.abs(np.asarray(c.udir)[sc, sc] - np.arctan2(tx, ty)))), 1e-15)
    # the mean of u* is NOT what is used (control: it must differ)
    usm = gs(w*np.sqrt(np.hypot(F['taux'], F['tauy'])/RHO_AIR))/gs(w)*kbc
    ok &= line('21c control: mean of u* differs from u* of the mean vector (must be > 1e-3)', 1e-3 - float(np.max(np.abs(usm - usc)))/float(np.max(usc)), 0.0)
    # ---- the 'coupled' source (relwind) ----
    r = small_wave(coarsen=1, relwind=True)
    ok &= line('21d relwind: the source is the coupled one on the wave grid', 0.0 if (r._forcing.__class__.__name__ == 'CoupledForcing' and r._forcing.own_grid) else 1.0, 0.0)
    ok &= line('21d relwind: u* = 0 before the first exchange', float(np.max(np.abs(np.asarray(r.ustar)))), 0.0)
    E0 = np.asarray(r.E)[..., sl, sl]
    ok &= line('21d relwind: every wet cell seeded (E > 0 on all wet, 0 on land)',
               float(np.max(np.abs((E0[0, 0] > 0).astype(float) - kbg))), 0.0)
    r._forcing.set({'taux': F['taux'], 'tauy': F['tauy']})
    r.update_forcing(123.0)
    ok &= line('21d relwind: u* after F.set() = sqrt(|tau|/rho_air)', float(np.max(np.abs(np.asarray(r.ustar)[sl, sl] - us))), 0.0)
    # ---- the energy diagnostics after one step ----
    for name, mm in (('ocean grid', m), ('coarse grid', c)):
        with contextlib.redirect_stdout(io.StringIO()):
            _step(mm, 1)
        hh = mm.halo; ss = slice(hh, -hh)
        p_fld = float(np.sum(mm.wind_input_field()*np.asarray(mm.area_t)[ss, ss]))/RHO_0
        p = mm.wind_work()
        ok &= line('21e %s: sum p_in A_t / rho_0 = wind_work' % name, abs(p_fld - p)/abs(p), 1e-13)
    ok &= line('21f snl_work = 0 without a current', abs(m.snl_work()), 0.0)
    # a uniform current: U.k S_nl summed over the resolved spectrum
    nz = 10; dz = 20.0*np.ones(nz)
    UL = 0.3*np.ones((nz, m.Ny - 2*h, m.Nx - 2*h)); VL = -0.1*np.ones_like(UL)
    m.ocean_U, m.ocean_V = m.calc_doppler_vel(dz, nz, UL, VL)
    Uk = (np.asarray(m.ocean_U)[None]*np.asarray(m.sinth)[:, None, None, None]
          + np.asarray(m.ocean_V)[None]*np.asarray(m.costh)[:, None, None, None])*np.asarray(m.omega_t)[None, :, None, None]
    fld = np.sum(np.sum((Uk*np.asarray(m.S_nl))[sl, sl, sl, sl]*np.asarray(m.dd), axis=0)*np.asarray(m.dft)[sl, None, None], axis=0)
    ref = float(np.sum(fld*np.asarray(m.area_t)[sl, sl]))
    ok &= line('21f snl_work with a uniform current = int U.k S_nl dk dA', abs(m.snl_work() - ref)/max(abs(ref), 1e-300), 1e-12)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
