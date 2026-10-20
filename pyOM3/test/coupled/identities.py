"""Test 19: discrete identities of the coupling terms in a short coupled
run, the ones exp_ocean books at the moment of application. Each is an
identity of the discrete operators, so it holds to round-off.

  19a vortex force     the area-weighted corner pairing does no work on
                       u_L: |W_vortex| / |W_tau|                    round-off
  19b Stokes shear     the horizontal Stokes shear terms and the pressure
                       p_SS cancel, W_h + W_p = 0 (stokes_shear_force
                       evaluated on the current state)              round-off
  19c exchange C       the d(u_S)/dt increment: KE(u + du) - KE(u) over dt
                       equals the booked C exactly (its definition is the
                       local identity (|u + d|^2 - |u|^2)/(2 dt))   round-off
  19d stress work      sum of the local work field stress_work_field x A_t
                       = rho_0 x stress_work_diag, with the direct branch
                       and with the depth-distributed stress        round-off
  19e direct branch    tau_o + branch = tau_a where tau_in < tau_a, the
                       branch is zero where tau_in >= tau_a            exact
  19f TKE budget       the closure's lhs = rhs booking (integrate_tke)  1e-12
  19g KE_ph = CONV     the conversion identity of the KE budget        1e-13
  19h D_h identity     horizontal friction = stencil + forward-Euler  1e-13
  19i heat and salt    content change per step = AB-weighted surface flux
                       (the SW leak of the shelf columns booked)      1e-10
  19j shortwave        the column integral of the penetrating source is
                       zero over every full-depth column; with SW_BOTTOM
                       over every column                               1e-18
  19k surface TKE      coupled: the flux is phi_oc; alone: the Craig-Banner
                       flux of the T-point stress                      exact
  19l mixing length    coupled, mxl_choice 2: z0_surf = HS_FACTOR hs    exact
All on the coupled model after 6 steps (19g-19i also on the forced one).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_ocean, step, line, RHO_0, DZT
import pyOM3 as OM


def _ke(m, u, v):
    return m._ke_of(u, v)


def run(N=6):
    ok = True
    # ---- the coupled default, the full force and the distributed stress ----
    for name, kw in (('coupled', dict()), ('coupled_full', dict(direct=True, shear=True, mxl_choice=2)),
                     ('coupled_stress', dict(stress_work=True, ke_budget=0, direct=True))):
        m = step(small_ocean(coupled=True, tag='id_' + name, **kw), N)
        h = m.halo; sl = slice(h, -h); ks = -h-1
        W_tau = abs(m.stress_work_diag())
        ok &= line('19a %s: |W_vortex| / |W_tau|' % name, abs(m._vortex_step)/W_tau, 1e-13)
        if kw.get('shear'):
            Fu, Fv, W_h, W_p = m.stokes_shear_force()
            ok &= line('19b %s: (W_h + W_p) / |W_h|' % name, abs(W_h + W_p)/max(abs(W_h), 1e-300), 1e-12)
        # the d(u_S)/dt increment on the current state
        u0, v0 = m.u*1.0, m.v*1.0
        k0 = _ke(m, u0, v0)
        m.set_forcing_direct()
        dk = (_ke(m, m.u, m.v) - k0)/m.dt
        ok &= line('19c %s: (KE(u+du) - KE(u))/dt = C' % name, abs(dk - m._C_step)/max(abs(m._C_step), 1e-300), 1e-12)
        m.u, m.v = u0, v0
        # stress work: field vs diagnostic
        At = np.asarray(m.area_t)[0]
        w_fld = float(np.sum((m.stress_work_field()*At)[sl, sl]))
        w_dia = RHO_0*m.stress_work_diag()
        ok &= line('19d %s: sum w_o A_t = rho_0 stress_work_diag' % name, abs(w_fld - w_dia)/abs(w_dia), 1e-13)
        if kw.get('direct'):
            # the branch of the CURRENT fields: re-run set_forcing on a copy of
            # the tendencies (the step's _tau_dir_x belongs to the previous tau_in)
            keep = [getattr(m, a)*1.0 for a in ('du', 'dv', 'dtemp', 'dsalt')]
            m.set_forcing(m._tau3[0])
            for a, v in zip(('du', 'dv', 'dtemp', 'dsalt'), keep):
                setattr(m, a, v)
            tax, tay = m.surface_forcing_fields(m._gitt*m.dt)[:2]
            tin = np.hypot(0.5*(np.asarray(m.tau_in_x) + np.roll(np.asarray(m.tau_in_x), -1, axis=1)),
                           0.5*(np.asarray(m.tau_in_y) + np.roll(np.asarray(m.tau_in_y), -1, axis=0)))
            ta = np.hypot(np.asarray(tax), np.asarray(tay))
            young = (tin >= ta) & (np.asarray(m.maskU)[ks] > 0)
            old = (tin < ta) & (np.asarray(m.maskU)[ks] > 0)
            dx = np.asarray(m._tau_dir_x)
            d_young = float(np.max(np.abs(dx[young]))) if young.any() else 0.0
            # where the sea is old, branch = tau_a (1 - tau_in/tau_a)
            ref = np.asarray(tax)*(1.0 - tin/np.maximum(ta, 1e-20))
            d_old = float(np.max(np.abs((dx - ref)[old])))/float(np.max(ta))
            ok &= line('19e %s: branch = tau_a - tau_in where tau_in < tau_a (%d U cells), 0 elsewhere (%d)'
                       % (name, int(old.sum()), int(young.sum())), d_young + d_old, 1e-15)
        if int(m.cfg.KE_BUDGET) == 1:
            ok &= line('19f %s: TKE budget |lhs - rhs| / |lhs|' % name, abs(m._tke_lhs - m._tke_rhs)/abs(m._tke_lhs), 1e-12)
        if kw.get('mxl_choice') == 2:
            z0 = np.asarray(m.z0_surf); ref = np.maximum(0.0, float(m.cfg.HS_FACTOR)*np.asarray(m.hs))*np.asarray(m.maskT)[ks]
            ok &= line('19l %s: z0_surf = HS_FACTOR hs' % name, float(np.max(np.abs(z0 - ref))), 0.0)
        # surface TKE flux = phi_oc (the flux of the CURRENT fields: the one
        # the last step applied belongs to the phi_oc delivered before it)
        ok &= line('19k %s: surface TKE flux = max(phi_oc, 0)' % name,
                   float(np.max(np.abs(np.asarray(m._surface_forcing()) - np.maximum(0.0, np.asarray(m.phi_oc)))[sl, sl])), 0.0)

    # ---- the KE budget's own identity accumulators, coupled and forced ----
    for name, kw in (('coupled', dict(coupled=True)), ('forced', dict(coupled=False)),
                     ('forced_sw_bottom', dict(coupled=False, sw_bottom=True))):
        m = step(small_ocean(tag='id2_' + name, **kw), N)
        h = m.halo; sl = slice(h, -h); ks = -h-1
        T = m.eb_n*m.dt; A = m.eb_area; r = m.rho_0
        chk = m.eb_chk/T/A*r                                   # mean |residual| [W/m2]
        acc = np.abs(m.eb_acc)/T/A*r                           # the terms themselves
        ok &= line('19g %s: |KE_ph - CONV| / |KE_ph|' % name, chk[0]/max(acc[2], 1e-300), 1e-13)
        ok &= line('19h %s: |D_h - (stencil + fwd-Euler)| / |D_h|' % name, chk[1]/max(acc[8], 1e-300), 1e-13)
        trc = m.eb_trc/max(m.eb_n - 1, 1)/A                    # [K m/s], [psu m/s]
        At = np.asarray(m.area_t)[0]; mT = np.asarray(m.maskT)[ks]
        FT = float(np.sum((np.abs(np.asarray(m.forc_temp_surface))*At*mT)[sl, sl]))/A
        FS = float(np.sum((np.abs(np.asarray(m.forc_salt_surface))*At*mT)[sl, sl]))/A
        dz3 = np.asarray(m.dzt_z); mT3 = np.asarray(m.maskT)
        Tc = float(np.sum((np.asarray(m.temp)*np.asarray(m.area_t)*dz3*mT3)[sl, sl, sl]))/A/m.dt
        Sc = float(np.sum((np.asarray(m.salt)*np.asarray(m.area_t)*dz3*mT3)[sl, sl, sl]))/A/m.dt
        # the residual against the flux (what the budget line prints) and
        # against the content change it is a difference of (its round-off level)
        ok &= line('19i %s: heat content residual / mean |F_T|' % name, trc[0]/FT, 1e-8)
        ok &= line('19i %s: heat content residual / (content/dt)' % name, trc[0]/Tc, 1e-13)
        ok &= line('19i %s: salt content residual / mean |F_S|' % name, trc[1]/FS, 1e-8)
        ok &= line('19i %s: salt content residual / (content/dt)' % name, trc[1]/Sc, 1e-13)
        # the shortwave source integrates to zero over full-depth columns
        sw = np.asarray(m._sw_col)
        full = (np.asarray(m.kbot) == h) & (mT > 0)
        qs = float(np.max(np.abs(np.asarray(m.surface_forcing_fields(m._gitt*m.dt)[3]))))/m.cfg.cp_0/m.cfg.rho_0
        ok &= line('19j %s: SW column integral over full columns / (qsol/cp rho)' % name, float(np.max(np.abs(sw[full])))/qs, 1e-14)
        if kw.get('sw_bottom'):
            ok &= line('19j %s: SW column integral over ALL wet columns (SW_BOTTOM)' % name,
                       float(np.max(np.abs(sw[mT > 0])))/qs, 1e-14)
        if not kw['coupled']:
            tx = 0.5*(np.asarray(m.taux) + np.roll(np.asarray(m.taux), 1, axis=1))
            ty = 0.5*(np.asarray(m.tauy) + np.roll(np.asarray(m.tauy), 1, axis=0))
            ref = m.alpha_CB*np.hypot(tx, ty)**1.5*mT
            ok &= line('19k %s: surface TKE flux = alpha_CB |tau_T|^1.5' % name,
                       float(np.max(np.abs(np.asarray(m.forc_tke_surface) - ref)[sl, sl])), 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
