"""
The coupled time loops (wave and ocean side) and the Adams-Bashforth
coefficient ramps. Everything experiment-specific arrives via the model
instance M and the cfg dict; this module holds no configuration state.

cfg keys: steps, sub_cycle, rsteps, do_restart, restart_wave_file,
restart_ocean_file, spinup_time, use_stress_work, ocean_dz_r, ocean_nz_r,
ny, nx, AB_eps, cpl_writer (wave side only).
"""
import time
import numpy as np
import pyOM3 as OM
from coupler_mpi import gather_unpadded_mpi, scatter_array_mpi
from coupler_io  import (wave_restart_write, wave_restart_read,
                         ocean_restart_write, ocean_restart_read)


def ab_coeffs(tA, tB, tC):
    if tA == 1. and tB == 0. and tC == 0.:      return 1.5, -0.5, 0.
    if tA == 1.5 and tB == -0.5 and tC == 0.:   return 23./12., -16./12., 5./12.
    return tA, tB, tC


def make_ab2_coeffs(eps):
    def ab2_coeffs(tA, tB, tC):
        if tA == 1. and tB == 0. and tC == 0.:  return 1.5+eps, -0.5-eps, 0.
        return tA, tB, tC
    return ab2_coeffs


def run_wave(M, cfg, comm, intercomm, active):
    steps = cfg['steps']; sub_cycle = cfg['sub_cycle']; rsteps = cfg['rsteps']
    ny = cfg['ny']; nx = cfg['nx']
    ocean_dz_r = cfg['ocean_dz_r']; ocean_nz_r = cfg['ocean_nz_r']
    use_stress_work = cfg['use_stress_work']
    cpl_writer = cfg.get('cpl_writer')
    restart_wave_file = cfg['restart_wave_file']
    ab2_coeffs = make_ab2_coeffs(cfg['AB_eps'])

    if cfg['do_restart']:
        att = wave_restart_read(M, restart_wave_file, ny, nx, comm)
        n0  = int(att['n_cpl']);  pre = int(att['pre'])
        tau, taum1, taum2 = int(att['tau']), int(att['taum1']), int(att['taum2'])
        tA, tB, tC = float(att['tA']), float(att['tB']), float(att['tC'])
    else:
        # --- wave-only spinup (fast model, its own dt) ---
        n0 = 0
        pre = int(cfg['spinup_time'] / M.dt)
        tA, tB, tC = 1., 0., 0.
        tau, taum1, taum2 = 2, 1, 0
        for n in range(pre):
            M.update_forcing(n * M.dt)
            M.time_step(tA, tB, tC, tau, taum1, taum2)
            if n % 100 == 0: M.check_monotonicity(n)
            M.diagnose(n, tau)
            tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
            #tA, tB, tC = ab2_coeffs(tA, tB, tC)
        if comm.Get_rank() == 0:
            print('wave spinup done')

    # --- coupled run: sub_cycle wave steps per ocean (coupling) step ---
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    tA, tB, tC = 1., 0., 0. # always Euler forward
    lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
    for n in range(n0, n_stop):
        for m in range(sub_cycle):
            step = pre + n*sub_cycle + m           # absolute wave step index
            M.update_forcing(step * M.dt)
            M.time_step(tA, tB, tC, tau, taum1, taum2)
            if step % 10 == 0: M.check_monotonicity(step)   # 12-step event; 100 missed it
            M.diagnose(step, tau)
            tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
            #tA, tB, tC = ab2_coeffs(tA, tB, tC)


        # --- wave-side ALARM (extended): spectral AND spatial advection
        # velocities, NON-FINITE detection, and -- on cfl_f/cfl_d alarms --
        # a one-shot INGREDIENT dump at the argmax point, so the corrupted
        # input names itself (ocean U/V exchange, Udf/Vdf kernel, or the
        # om_mean/fc spectral ratios). All prints per-rank, no collectives;
        # ingredient dump capped per PE and try/except-wrapped: diagnostics
        # must never take down the model. ---
        h = M.halo
        cfl_d = np.abs(M.ddot[:, :, h:-h, h:-h]) * M.dt / M.dd
        cfl_f = np.abs(M.fdot[:, h:-h, h:-h, h:-h]) * M.dt \
                / M.dft[None, h:-h, None, None]
        cfl_x = np.abs(M.xdot[:, :, h:-h, h:-h]) * M.dt / M.dx
        cfl_y = np.abs(M.ydot[:, :, h:-h, h:-h]) * M.dt / M.dy
        Aloc  = M.A[:, :, h:-h, h:-h].sum(axis=(0, 1))
        if not hasattr(M, '_alarm_dumps'):
            M._alarm_dumps = 0
        for name, fld, thr in (('cfl_d', cfl_d, 0.5), ('cfl_f', cfl_f, 0.5),
                               ('cfl_x', cfl_x, 0.35), ('cfl_y', cfl_y, 0.35),
                               ('A2d', Aloc, 50.0*float(np.median(Aloc)) + 1e-30)):
            m = float(np.max(fld))
            if not np.isfinite(m):
                print('WAVE ALARM pe %d step %d: %s NON-FINITE'
                      % (M.my_pe, n, name), flush=True)
                continue
            if m > thr:
                idx = np.unravel_index(int(np.argmax(fld)), fld.shape)
                print('WAVE ALARM pe %d step %d: %s = %.3e at local idx %s'
                      % (M.my_pe, n, name, m, idx), flush=True)
                if name.startswith('cfl') and M._alarm_dumps < 20:
                    M._alarm_dumps += 1
                    try:
                        if name == 'cfl_f':
                            _d, _f, _j, _i = idx
                            _f = _f + h                  # back to haloed f index
                        else:
                            _d, _f, _j, _i = idx
                        _J, _I = _j + h, _i + h          # back to haloed y/x
                        def _g(a, *ix):
                            try:    return float(np.asarray(getattr(M, a))[ix])
                            except Exception: return float('nan')
                        print('  ingredients pe %d (J%d I%d f%d d%d): '
                              'U=%.3e V=%.3e Udf=%.3e Vdf=%.3e | '
                              'om_mean=%.3e fc_idx=%s | fdot=%.3e ddot=%.3e '
                              '| A(d,f,J,I)=%.3e'
                              % (M.my_pe, _J, _I, _f, _d,
                                 _g('ocean_U', _J, _I), _g('ocean_V', _J, _I),
                                 _g('ocean_Udf', _f, _J, _I),
                                 _g('ocean_Vdf', _f, _J, _I),
                                 _g('om_mean', _J, _I),
                                 (str(int(np.asarray(M.frcutindex)[_J, _I]))
                                  if hasattr(M, 'frcutindex') else '?'),
                                 _g('fdot', _d, _f, _J, _I),
                                 _g('ddot', _d, _f, _J, _I),
                                 _g('A', _d, _f, _J, _I)), flush=True)
                    except Exception as _e:
                        print('  ingredient dump failed pe %d: %r'
                              % (M.my_pe, _e), flush=True)

        
        # --- couple once per ocean step (after the full wave sub-cycle) ---
        # each wave PE computes its LOCAL block; gather to wave-rank-0 -> full domain
        # --- couple once per ocean step (after the full wave sub-cycle) ---
        pw = M.wind_work()                 # collective: call on ALL wave PEs
        wnl = M.snl_work()                 # collective: int U.k S_nl dk (Eq. 11)
        # wave->ocean dissipation energy flux (PHIOC): local field (for the
        # future TKE surface BC) + its domain integral for the energy budget
        phi_loc = M.calc_phioc()                                   # (ny_loc,nx_loc)
        pdiss = float(np.sum(phi_loc) * M.dx * M.dy)
        if comm.Get_size() > 1:
            from mpi4py import MPI
            pdiss = comm.allreduce(pdiss, op=MPI.SUM)
        if n % 20 == 0 and comm.Get_rank() == 0:
            print('energy diag [m^5/s^3 /rho]: wind-work into waves %.3e | '
                  'dissipation flux to ocean %.3e | U.k S_nl exchange %.3e'
                  % (pw, pdiss, wnl), flush=True)
        # ---- momentum budget [kinematic; stresses m^2/s^2 * area] ----
        # input tau_in = int gamma_in E omega (sin,cos) dOmega  (resolved
        # + analytic tail if the tree provides it); wave momentum content
        # M_w = int E omega (sin,cos) dOmega  (so tau = dM_w/dt bin-wise).
        isl = slice(M.halo, -M.halo)
        kxw = (M.omega_t[None, :]*M.sinth[:, None])[isl, isl]
        kyw = (M.omega_t[None, :]*M.costh[:, None])[isl, isl]
        dft = M.dft[isl]
        def _mfield(F):                       # local (2,ny_loc,nx_loc) field
            fx = np.sum(np.sum(F[isl,isl,isl,isl]*kxw[:,:,None,None]*M.dd,
                        axis=0)*dft[:,None,None], axis=0)
            fy = np.sum(np.sum(F[isl,isl,isl,isl]*kyw[:,:,None,None]*M.dd,
                        axis=0)*dft[:,None,None], axis=0)
            return np.stack([fx, fy])
        tin_loc = _mfield(M.gamma_in*M.E)     # wind -> wave input stress field
        if hasattr(M, '_tail_input_factors'):        # analytic tail input
            ft_t, _, _, _ = M._tail_input_factors()
            Ecg = M.E[:, -M.halo, :, :]
            tin_loc[0] += np.sum(Ecg[isl,isl,isl]*ft_t[isl,isl,isl]
                        *M.sinth[isl,None,None]*M.dd, axis=0)
            tin_loc[1] += np.sum(Ecg[isl,isl,isl]*ft_t[isl,isl,isl]
                        *M.costh[isl,None,None]*M.dd, axis=0)
        min_x = float(np.sum(tin_loc[0])*M.dx*M.dy)
        min_y = float(np.sum(tin_loc[1])*M.dx*M.dy)
        mwf = _mfield(M.E)
        mw_x = float(np.sum(mwf[0])*M.dx*M.dy)
        mw_y = float(np.sum(mwf[1])*M.dx*M.dy)
        u_S_loc = M.calc_stokes(ocean_dz_r, ocean_nz_r)             # (2,nz,ny_loc,nx_loc)
        if use_stress_work:
            stress_loc = M.calc_stress_work(ocean_dz_r, ocean_nz_r)  # (2,nz,ny_loc,nx_loc)
        else:
            stress_loc = M.calc_stress()                            # (2,ny_loc,nx_loc)

        # delivered stress, surface-equivalent domain integral
        if use_stress_work:
            dsl = stress_loc.sum(axis=1)*ocean_dz_r                 # column sum
        else:
            dsl = stress_loc
        mdel_x = float(np.sum(dsl[0])*M.dx*M.dy)
        mdel_y = float(np.sum(dsl[1])*M.dx*M.dy)
        if comm.Get_size() > 1:
            from mpi4py import MPI as _MPI
            (min_x, min_y, mw_x, mw_y, mdel_x, mdel_y) = comm.allreduce(
                np.array([min_x, min_y, mw_x, mw_y, mdel_x, mdel_y]), op=_MPI.SUM)
        # cumulative budget since segment start (dt_cpl per coupling step)
        if n == n0:
            cum_in = np.zeros(2); cum_del = np.zeros(2); mw0 = np.array([mw_x, mw_y])
        dt_cpl = sub_cycle*M.dt
        cum_in  += np.array([min_x, min_y])*dt_cpl
        cum_del += np.array([mdel_x, mdel_y])*dt_cpl
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            dmw = np.array([mw_x, mw_y]) - mw0
            res = cum_in - cum_del - dmw
            print('momentum budget x (cum. since seg start) [m^4/s]: in %.3e | '
                  'delivered %.3e (%.0f%%) | d storage %.3e (%.0f%%) | residual %.3e (%.0f%%)'
                  % (cum_in[0], cum_del[0], 100*cum_del[0]/max(cum_in[0],1e-30),
                     dmw[0], 100*dmw[0]/max(cum_in[0],1e-30),
                     res[0], 100*res[0]/max(cum_in[0],1e-30)), flush=True)
            # windowed budget: the state NOW (last 100 steps), free of the
            # spin-up era that dominates the cumulative for a long time
            if 'cum_in_p' in dir():
                win_in  = cum_in  - cum_in_p
                win_del = cum_del - cum_del_p
                win_mw  = np.array([mw_x, mw_y]) - mw_p
                print('momentum budget x (last 100 steps): delivered %.1f%% | '
                      'd storage %.1f%% | residual %.1f%%'
                      % (100*win_del[0]/max(win_in[0],1e-30),
                         100*win_mw[0]/max(win_in[0],1e-30),
                         100*(win_in[0]-win_del[0]-win_mw[0])/max(win_in[0],1e-30)),
                      flush=True)
            cum_in_p, cum_del_p = cum_in.copy(), cum_del.copy()
            mw_p = np.array([mw_x, mw_y])
        # gather each component's full field on wave-rank-0 (unpadded output)
        u_S    = gather_unpadded_mpi(M, u_S_loc,    ny, nx)        # rank0: (2,nz,ny,nx) else None
        stress = gather_unpadded_mpi(M, stress_loc, ny, nx)        # rank0: (2,[nz,]ny,nx) else None
        phioc  = gather_unpadded_mpi(M, phi_loc,    ny, nx)        # rank0: (ny,nx)
        tau_in = gather_unpadded_mpi(M, tin_loc,    ny, nx)        # rank0: (2,ny,nx)

        # friction-velocity vector in wim's direction convention
        # (theta=0 -> +y/north, pi/2 -> +x/east): east=sin(udir), north=cos(udir).
        sl = slice(M.halo, -M.halo)
        ustar_x_loc = (M.ustar * np.sin(M.udir))[sl, sl]   # eastward (+x)
        ustar_y_loc = (M.ustar * np.cos(M.udir))[sl, sl]   # northward (+y)
        ustar_x = gather_unpadded_mpi(M, ustar_x_loc, ny, nx)     # rank0: (ny,nx)
        ustar_y = gather_unpadded_mpi(M, ustar_y_loc, ny, nx)

        if comm.Get_rank() == 0:
            # receive full-domain ocean current, send wave fields (full domain)
            ocean_U = intercomm.recv(source=0, tag=1)               # (nz,ny,nx)
            ocean_V = intercomm.recv(source=0, tag=2)  
            intercomm.send(u_S, dest=0, tag=3)
            intercomm.send(stress, dest=0, tag=4)
            intercomm.send(phioc, dest=0, tag=5)   # wave->ocean TKE surface flux
            # ocean-side energy diagnostics for THIS coupling step (computed
            # after the ocean has received u_S/stress), then write the record
            wst, wste, cxc, wvf = intercomm.recv(source=0, tag=7)
            if cpl_writer is not None:
                cpl_writer.write((pre + (n + 1) * sub_cycle) * M.dt,
                                 ustar_x, ustar_y, u_S, stress, phioc=phioc,
                                 stress_in=tau_in,
                                 ediag=(pw, wst, wste, cxc, wvf, pdiss, wnl),
                                 mdiag=(min_x, min_y, mdel_x, mdel_y, mw_x, mw_y))
        else:
            ocean_U, ocean_V = None, None

        # scatter the full ocean current to each wave PE's local block, then
        # every wave PE Doppler-projects its own block (calc_doppler_vel does
        # the local pad + halo exchange across wave PEs).
        UL = scatter_array_mpi(M, ocean_U, (ocean_nz_r,))          # (nz,ny_loc,nx_loc)
        VL = scatter_array_mpi(M, ocean_V, (ocean_nz_r,))
        u, v = M.calc_doppler_vel(ocean_dz_r, ocean_nz_r, UL, VL)
        if hasattr(M, 'ocean_U'):
            M.ocean_Udf, M.ocean_Vdf = M.calc_doppler_vel_df(ocean_dz_r, ocean_nz_r, UL, VL)
        M.ocean_U, M.ocean_V = u, v

        # periodic restart write (collective gathers: all wave PEs call)
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            wave_restart_write(M, restart_wave_file, ny, nx, n + 1, pre,
                               tau, taum1, taum2, tA, tB, tC)

        _t1 = time.perf_counter()
        active.Barrier()
        _t2 = time.perf_counter()
        lb_comp += _t1 - _t0; lb_wait += _t2 - _t1; _t0 = _t2
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            print('load wave : compute %.3f s/step | waiting %.3f s/step '
                  '(%.0f%% idle)' % (lb_comp/100., lb_wait/100.,
                  100.*lb_wait/max(lb_comp+lb_wait, 1e-30)), flush=True)
            lb_comp = lb_wait = 0.0
    if comm.Get_rank() == 0:
        if cpl_writer is not None:
            cpl_writer.close()
        print('wave done')

    return n_stop


def run_ocean(M, cfg, comm, intercomm, active):
    steps = cfg['steps']; rsteps = cfg['rsteps']
    ny = cfg['ny']; nx = cfg['nx']
    use_stress_work = cfg['use_stress_work']
    restart_ocean_file = cfg['restart_ocean_file']

    if cfg['do_restart']:
        att = ocean_restart_read(M, restart_ocean_file, ny, nx, use_stress_work, comm)
        n0 = int(att['n_cpl'])
        tau, taum1, taum2 = int(att['tau']), int(att['taum1']), int(att['taum2'])
        tA, tB, tC = float(att['tA']), float(att['tB']), float(att['tC'])
    else:
        n0 = 0
        tA, tB, tC = 1., 0., 0.
        tau, taum1, taum2 = 2, 1, 0
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
    for n in range(n0, n_stop):
        # one ocean step per coupling interval
        M.time_step_implicit_pressure(tA, tB, tC, tau, taum1, taum2)
        if (n + 1) % 100 == 0:
            umax = float(np.abs(M.u[:, hb, hb]).max())
            if comm.Get_size() > 1:
                from mpi4py import MPI as _MPI
                umax = comm.allreduce(umax, op=_MPI.MAX)
            if comm.Get_rank() == 0:
                print('ocean canary: max|u| = %.3f m/s%s'
                      % (umax, '   <-- WARNING: accelerating past tuning range'
                         if umax > 2.5 else ''), flush=True)
            if not np.isfinite(umax):
                raise RuntimeError('ocean state is NaN at coupling step %d -- '
                                   'aborting BEFORE the restart write' % n)
        M.diagnose(n, tau)
        tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
        tA, tB, tC = ab_coeffs(tA, tB, tC)

        # ocean -> wave: gather full-domain current to ocean PE0
        gU = M.gather_array_mpi(M.u)          # (nz,ny,nx) on PE0, else None
        gV = M.gather_array_mpi(M.v)
        if M.my_pe == 0:
            intercomm.send(np.ascontiguousarray(gU), dest=0, tag=1)
            intercomm.send(np.ascontiguousarray(gV), dest=0, tag=2)
            u_S    = intercomm.recv(source=0, tag=3)
            stress = intercomm.recv(source=0, tag=4)
            phioc  = intercomm.recv(source=0, tag=5)   # (ny,nx)

        else:
            u_S, stress, phioc = None, None, None

        # broadcast wave fields to all ocean PEs; each slices its own block
        u_S    = comm.bcast(u_S,    root=0)
        stress = comm.bcast(stress, root=0)
        phioc  = comm.bcast(phioc,  root=0)

        ysl = slice(M.ys_pe - 1, M.ye_pe)
        xsl = slice(M.xs_pe - 1, M.xe_pe)
        hb  = slice(M.halo, -M.halo)

        # Stokes drift -> interior, then halo fill
        # keep the previous Stokes drift for the exchange-work diagnostic
        # C = int u_L d(u_S)/dt dV  (explicit copies: modify_array mutates
        # in place with the numpy backend)
        M.u_S_old, M.v_S_old = M.u_S.copy(), M.v_S.copy()
        M.u_S = OM.modify_array(M.u_S, (hb, hb, hb), u_S[0][:, ysl, xsl])
        M.v_S = OM.modify_array(M.v_S, (hb, hb, hb), u_S[1][:, ysl, xsl])
        M.u_S, M.v_S = M.apply_bc(M.u_S), M.apply_bc(M.v_S)

        if use_stress_work:
            M.tau_work_x = OM.modify_array(M.tau_work_x, (hb, hb, hb), stress[0][:, ysl, xsl])
            M.tau_work_y = OM.modify_array(M.tau_work_y, (hb, hb, hb), stress[1][:, ysl, xsl])
            M.tau_work_x = M.apply_bc(M.tau_work_x)
            M.tau_work_y = M.apply_bc(M.tau_work_y)
        else:
            M.taux = OM.modify_array(M.taux, (hb, hb), stress[0][ysl, xsl])
            M.tauy = OM.modify_array(M.tauy, (hb, hb), stress[1][ysl, xsl])
            M.taux, M.tauy = M.apply_bc(M.taux), M.apply_bc(M.tauy)
        # wave dissipation energy flux (WAM: PHIOC) -> surface TKE flux BC of
        # the ocean's turbulence closure (WP1: tke.py). Kinematic [m^3/s^3];
        # genuinely 2D (Ny,Nx), same fill pattern as taux.
        M.phi_oc = OM.modify_array(M.phi_oc, (hb, hb), phioc[ysl, xsl])
        M.phi_oc = M.apply_bc(M.phi_oc)

        # energy diagnostics every coupling step (collective over the ocean
        # communicator: call on ALL ocean PEs); rank 0 sends the scalars to
        # wave-rank-0 (tag 7) for storage in the coupling netCDF record.
        # MUST run BEFORE active.Barrier(): wave-rank-0 blocks on this recv
        # before reaching its own barrier, so sending after ours deadlocks.
        wvf = M.vortex_work()
        wst = M.stress_work_diag()
        wste = M.stress_work_eul_diag()      # against u = u_L - u_S (comparison)
        cxc = M.exchange_work_C()
        if comm.Get_rank() == 0:
            intercomm.send((wst, wste, cxc, wvf), dest=0, tag=7)
        if M.my_pe == 0 and n % 20 == 0:
            print('energy diag [m^5/s^3 /rho]: stress-work on ocean u_L %.3e '
                  '(vs Eulerian u %.3e) | exchange C (exact) %.3e | '
                  'vortex-force u_L-work residual %.3e' % (wst, wste, cxc, wvf),
                  flush=True)
        if (n + 1) % 100 == 0 and getattr(M, 'enable_tke', False):
            hh = M.halo
            ftke = float(np.sum(M.forc_tke_surface[hh:-hh, hh:-hh]) * M.dx * M.dy)
            if comm.Get_size() > 1:
                from mpi4py import MPI as _MPI
                ftke = comm.allreduce(ftke, op=_MPI.SUM)
            if comm.Get_rank() == 0:
                print('tke closure: domain-int surface TKE forcing %.4e m^5/s^3 '
                      '(compare diss_work on the wave side)' % ftke, flush=True)
        _t1 = time.perf_counter()
        active.Barrier()
        _t2 = time.perf_counter()
        lb_comp += _t1 - _t0; lb_wait += _t2 - _t1; _t0 = _t2
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            print('load ocean: compute %.3f s/step | waiting %.3f s/step '
                  '(%.0f%% idle)' % (lb_comp/100., lb_wait/100.,
                  100.*lb_wait/max(lb_comp+lb_wait, 1e-30)), flush=True)
            lb_comp = lb_wait = 0.0
        # periodic restart write (collective gathers: all ocean PEs call);
        # same coupling steps as the wave side -> consistent file pair
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            ocean_restart_write(M, restart_ocean_file, ny, nx, use_stress_work, n + 1,
                                tau, taum1, taum2, tA, tB, tC)
        if M.my_pe == 0 and n % 20 == 0:
            print('coupled ocean step %d / %d' % (n, steps), flush=True)
    if M.my_pe == 0:
        print('ocean done')
    return n_stop
