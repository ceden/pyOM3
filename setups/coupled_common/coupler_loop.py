"""
The coupled time loops (wave and ocean side). Everything experiment-specific
arrives via the model instance M and the cfg dict; this module holds no
configuration state.

cfg keys: steps, sub_cycle, rsteps, seg_steps, do_restart, restart_wave_file,
restart_ocean_file, init_ocean, spinup_time, use_stress_work, ocean_dz_r,
ocean_nz_r, ny, nx, ocean_ab, cpl_write_every, profile, profile_every,
cpl_writer (wave side only), cmap.

A COARSER WAVE GRID (cfg['cmap'], a coupler_mpi.CoarseMap, None = same grid).
ny and nx are then the component's OWN grid size: the wave loop gets the wave
grid's, the ocean loops the ocean's. The exchange is still block to block
(coupler_mpi.BlockExchange): the wave fields arrive on the ocean grid as they
always did, copied into the ocean cells of each wave cell, and the current
arrives on the wave grid as the area mean over the WET ocean cells of each
wave cell, level by level (weights area_u*maskU and area_v*maskV, summed on
the ocean side, normalised on the wave side with a sum of the weights that is
exchanged ONCE before the loop, tags 11 and 12). With cmap = None nothing
below differs from the same-grid loop.

run_ocean_only is the ocean without the wave model (GLOBAL4_WAVES=0): the
same time step, diagnostics, restart and profiling, no exchange.
init_ocean (both ocean loops): start from another run's ocean restart with
the step counter at 0 (a branch). The coupling fields of that file are
zeroed, so a coupled branch from an ocean-alone state starts as a cold
coupled start does: no wave stress, no Stokes drift, no Phi_oc in the first
step, the direct branch carries tau_a.

PROFILING (see coupler_prof.py). cfg['profile'] = 0 off, 1 section timers,
2 additionally wraps the model's own methods on the INSTANCE -- no timing
code goes into the package. Both components print a table every
cfg['profile_every'] coupling steps: the barrier and exchange rows are the
time each one spends waiting for the other.

Differences to the acc1 version, all forced by the pseudo-spherical grid and
the realistic forcing:
  * every domain integral is weighted with the true cell area area_t (the
    scalar dx*dy of the Cartesian channel does not exist -- grid.py sets
    self.dx = None on purpose), and the CFL alarm uses the same metric
    factors as the transport itself (cost*dxt in x, cosu/cost*dyt in y);
  * the ocean sets its step counter _gitt before each step, because the
    monthly climatology in set_forcing needs the model time;
  * the wave input stress tau_in is handed to the ocean as well: it is the
    quantity the optional direct branch tau_a - tau_in needs, and it is a
    useful ocean-side diagnostic in any case;
  * the coupling-file write cadence is gated by cpl_write_every. Only the
    write and the WRITER-ONLY gathers are gated; the ocean-bound gathers and
    the tag-7 receive run every step (gating them would deadlock).
"""
import os
import time
import numpy as np
import pyOM3 as OM
from coupler_mpi import gather_unpadded_mpi, scatter_array_mpi, block_exchange

# Smallest number of live (wet, non-empty) cells in a PE's block for which the
# A2d wave alarm's median is a meaningful reference. See the alarm block in
# run_wave.
A2D_MIN_LIVE = 50
from coupler_io  import (atmos_restart_write, atmos_restart_read, AtmosCouplingWriter,
                         wave_restart_write, wave_restart_read,
                         ocean_restart_write, ocean_restart_read)
from coupler_prof import Profiler, WAVE_METHODS, OCEAN_METHODS

# =============================================================================
# THE ATMOSPHERE (energy_cycle.tex Sec. 6): a third component, the ocean is
# the hub. Every coupling step the ocean receives from the atmosphere its
# stress and heat flux as block copies onto the ocean cells of each atmosphere
# cell, and sends back area sums over the wet ocean cells of each atmosphere
# cell: SST, u_o (means when divided by the static wet area), the stress work
# w_o, the wave input p_in (forwarded from the waves) and the applied heat
# (sums per atmosphere cell area). The waves get the atmosphere's stress from
# the ocean's copy as the area mean over their own cells (one more step of
# lag) and hand p_in to the ocean. Tags 21-23 atmosphere -> ocean, 24-31
# ocean -> atmosphere (30 the static weights), 9-10 and 13-14 ocean -> waves
# (stress, static weights), 12 waves -> ocean (p_in).
# =============================================================================
ATM_FIELDS2D = ('sst', 'tau_ax', 'tau_ay', 'q_s', 'u_o', 'v_o', 'w_o', 'p_in', 'q_ocn', 'f_ice')


def _ocean_hub_start(M, cfg, comm):
    """Ocean side, once: the block exchange with the atmosphere and the static
    weights (wet area of the ocean cells) the atmosphere normalises with.
    The weights are also what the ocean forwards to the waves for the mean
    of the stress, so they are built whenever the WIND HUB is on (cfg
    'wind_hub': an atmosphere, or the relative-wind stress of the forced
    ocean, which the waves must take from the ocean too)."""
    h = M.halo; hb = slice(h, -h); ks = -h-1
    if cfg.get('wind_hub'):
        M._w_wet = np.asarray(M.area_t*M.maskT[ks])[0][hb, hb] if np.asarray(M.area_t).ndim == 3 \
                   else np.asarray(M.area_t*M.maskT[ks])[hb, hb]
    if not cfg.get('atmos'):
        return None
    bxa = block_exchange(M, comm, cfg['intercomm_ao'], cfg['cmap_a'], 'ocean', key='atmos')
    bxa.send_sums(None, M._w_wet, 30)
    bxa.wait_sends()
    M._p_in_w = np.zeros(M._w_wet.shape)          # p_in from the waves, ocean cells, W/m2
    return bxa


def _ocean_hub_exchange(M, bxa, prof):
    """Ocean side, every coupling step, AFTER the ocean step and its wave
    exchange: post the receives of the atmosphere's blocks, send the sums,
    complete, and store the received stress and heat flux in the forcing
    object for the next step."""
    if bxa is None:
        return
    h = M.halo; hb = slice(h, -h); ks = -h-1
    with prof.section('atmos exchange'):
        hTx = bxa.post_recv((), 21); hTy = bxa.post_recv((), 22); hQ = bxa.post_recv((), 23)
        w = M._w_wet
        uo, vo = M.surface_velocity()
        sst = (np.asarray(M.temp)[ks] + 273.15)[hb, hb]
        wo  = M.stress_work_field()[hb, hb]
        # heat applied after the freezing floor (K m/s -> W/m2) and the ice mask
        ft  = np.asarray(M.forc_temp_surface)[hb, hb]*M.cfg.cp_0*M.cfg.rho_0
        ice = np.where(np.asarray(M.temp)[ks][hb, hb]*np.asarray(M.maskT)[ks][hb, hb] <= -1.8, 1.0, 0.0)
        bxa.send_sums(sst, w, 24); bxa.send_sums(uo[hb, hb], w, 25); bxa.send_sums(vo[hb, hb], w, 26)
        bxa.send_sums(wo, w, 27); bxa.send_sums(M._p_in_w, w, 28); bxa.send_sums(ft, w, 29)
        bxa.send_sums(ice, w, 31)
        tx = bxa.complete_recv(hTx); ty = bxa.complete_recv(hTy); qs = bxa.complete_recv(hQ)
        bxa.wait_sends()
        M._forcing.set({'taux': tx, 'tauy': ty, 'qnet': -qs})
    return


def _wave_hub_start(M, cfg, comm, bx):
    """Wave side, once: the static wet-area weights for the mean of the stress
    the ocean forwards (2-D, T points)."""
    if not cfg.get('wind_hub'):
        return None
    if cfg.get('cmap') is None:
        return 1.0
    hW = bx.post_recv((), 13)
    wsum = bx.complete_recv_sums(hW)
    return np.where(wsum > 0.0, 1.0/np.where(wsum > 0.0, wsum, 1.0), 0.0)



# ---------------------------------------------------------------------------
# consistency diagnostics of the coupled acc runs, on the INSTANCE (nothing in
# the package):
#   wave side   W_dtA = int U.k d(A)/dt dk dA over the coupling interval, the
#               wave-side form of the exchange C = int u_L d(u_S)/dt dV (equal
#               up to the one-interval lag of ocean_U and the diagnostic tail)
#   ocean side  KE of u_L and its change per step, with the KE change done by
#               vertical friction (= -int K_diss_v dV - numerical) and by
#               horizontal friction, booked by the ocean class in
#               global4_setup.py (vertical_friction / harmonic_momentum
#               overrides); the full per-term KE budget is the class's
#               ke_budget (global4_energy.txt)
# ---------------------------------------------------------------------------
def wave_dtA_work(M, A_old, dt_cpl):
    if not hasattr(M, 'ocean_U'):
        return 0.0
    h  = M.halo
    sl = slice(h, -h)
    Uk = (M.ocean_U[None, :, :, :]*M.sinth[:, None, None, None]
        + M.ocean_V[None, :, :, :]*M.costh[:, None, None, None]) \
        * M.omega_t[None, :, None, None]
    fld = np.sum(np.sum((Uk*(M.A - A_old))[sl, sl, sl, sl]*M.dd, axis=0)
                 *M.dft[sl, None, None], axis=0)/dt_cpl
    return float(np.sum(fld*M.area_t[sl, sl]))


def ocean_ke(M, u=None, v=None):
    """Local (this PE) kinetic energy of u_L, int |u_L|^2/2 dV."""
    h  = M.halo
    sl = slice(h, -h)
    u = M.u if u is None else u
    v = M.v if v is None else v
    return 0.5*float(OM.np.sum((u*u*M.area_u + v*v*M.area_v)[sl, sl, sl]*M.dzt_z[sl]))




def _ab_weights(ab, n, cfg):
    """The Adams-Bashforth weights of ocean step n. The two-level set
    (1.6,-0.6,0) is used from the first step on, as always. A three-level set
    is ramped at a COLD start, where the old tendency levels are empty:
    forward Euler, AB2, then the full weights. A restart and a branch
    (init_ocean) carry all three tendency levels, so they need no ramp; the
    branch is the case that starts at n = 0 with a valid history."""
    if ab[2] == 0.0 or n > 1 or cfg.get('init_ocean'):
        return ab
    return (1.0, 0.0, 0.0) if n == 0 else (1.5, -0.5, 0.0)


def _ocean_start(M, cfg, comm):
    """Initial state of the ocean loops: own restart, another run's restart
    (branch, step counter 0), or the initial conditions.
    Returns (n0, tau, taum1, taum2)."""
    ny, nx = cfg['ny'], cfg['nx']
    if cfg['do_restart']:
        att = ocean_restart_read(M, cfg['restart_ocean_file'], ny, nx,
                                 cfg['use_stress_work'], comm)
        if hasattr(M, 'set_wave_roughness'):     # z0_surf from the restart hs
            M.set_wave_roughness()
        if M.my_pe == 0 and getattr(M, '_avg', None) is not None:
            print('ocean means: partial window of %d steps taken from the restart'
                  % M._avg_n, flush=True)
        return (int(att['n_cpl']), int(att['tau']), int(att['taum1']),
                int(att['taum2']))
    if cfg.get('init_ocean'):
        att = ocean_restart_read(M, cfg['init_ocean'], ny, nx,
                                 cfg['use_stress_work'], comm,
                                 restore_average=False)
        for name in ('u_S', 'v_S', 'taux', 'tauy', 'tau_work_x', 'tau_work_y',
                     'phi_oc', 'tau_in_x', 'tau_in_y'):
            if hasattr(M, name):
                setattr(M, name, getattr(M, name)*0.0)
        for name in ('u_S_old', 'v_S_old'):
            if hasattr(M, name):
                delattr(M, name)
        if hasattr(M, 'set_wave_roughness'):
            M.set_wave_roughness()
        if M.my_pe == 0:
            print('branch: ocean state of %s (its step %d), coupling fields '
                  'zeroed, this run starts at step 0'
                  % (cfg['init_ocean'], int(att['n_cpl'])), flush=True)
        return 0, int(att['tau']), int(att['taum1']), int(att['taum2'])
    return 0, 2, 1, 0


# Bounds the canary calls an alarm on. They are not physics: they are the
# values beyond which the TEOS-10 polynomial of density.py starts to overflow
# and the state turns into NaN a few steps later. A run that reaches any of
# them is already lost, so the point is to SAY WHERE and to leave a snapshot
# of the step, not to continue.
CANARY_BOUNDS = dict(u=10.0, temp=(-5.0, 60.0), salt=(1.0, 45.0))
# how many alarming cells to print per rank
CANARY_LIST = 5
# HOW OFTEN THE CHECK RUNS, in ocean steps. 100 is enough to stop a run
# before it wastes a job, and far too coarse to find WHERE it went wrong: the
# implicit free surface is a GLOBAL solve, so one NaN in the density enters
# p_h and the next solve spreads it to every cell of the domain within one
# step. A dump taken up to 100 steps later is uniformly NaN and says nothing.
# Set EXP_CANARY_EVERY=1 for a debugging run: the check is four isfinite
# passes over this rank's block and one reduction, a millisecond or two
# against a step of order a second, and the dump is then the step the first
# NaN appeared in, while it is still local.
CANARY_EVERY = max(1, int(os.environ.get('EXP_CANARY_EVERY', '100')))

# THE VERTICAL CFL MONITOR, EXP_CANARY_CFL, off by default.
# w is not an independent field: it is the vertically integrated horizontal
# divergence, and on a continental slope that is w = u.grad(H), so one cell
# carries the whole depth change. On a fine horizontal grid with thin surface
# cells, w dt/dz is therefore the number that decides the time step, and it
# scales like dt/dx even though nothing in the vertical changed. A violation
# shows as a two-cell vertical sawtooth in whichever tracer has a gradient
# there, and the run is lost a few steps later.
# The monitor reports max |w| dt/dz over the domain, with its location, on
# the same line as max|u|, and raises the ALARM above CANARY_CFL_ALARM, so
# the margin can be watched eroding through a season instead of being
# discovered by a crash. Cost: one pass over w on this rank's block and one
# reduction.
#   EXP_CANARY_CFL=0    off (default)
#   EXP_CANARY_CFL=1    report it
#   EXP_CANARY_CFL=0.8  report it and stop above 0.8
# The scheme's own limit is not a sharp number; icon01 ran stably at a domain
# maximum of 0.20 and blew up at 0.64. Treat 0.5 as the edge.
_ccfl = os.environ.get('EXP_CANARY_CFL', '0')
CANARY_CFL = _ccfl not in ('0', '', 'no')
CANARY_CFL_ALARM = float(_ccfl) if (CANARY_CFL and float(_ccfl) != 1.0) else 0.0


def _canary_fields(M):
    """(name, array, lo, hi) of what is checked, interior only. p_s is in it
    for non-finite only: it is where the global solve deposits the damage, so
    a NaN p_s beside clean tracers says the CG is the amplifier and the origin
    is in the momentum or the density of that same step."""
    h  = M.halo
    hb = slice(h, -h)
    out = [('u', np.asarray(M.u)[hb, hb, hb], -CANARY_BOUNDS['u'], CANARY_BOUNDS['u']),
           ('v', np.asarray(M.v)[hb, hb, hb], -CANARY_BOUNDS['u'], CANARY_BOUNDS['u'])]
    for nm in ('temp', 'salt'):
        lo, hi = CANARY_BOUNDS[nm]
        out.append((nm, np.asarray(getattr(M, nm))[hb, hb, hb], lo, hi))
    ps = np.asarray(M.p_s)
    out.append(('p_s', ps[..., hb, hb], -np.inf, np.inf))
    return out


def _canary_dump(M, n, why):
    """Write THIS step as a raw snapshot beside the ordinary ones, so the
    state that produced the alarm survives the abort. Every rank writes its
    own block, as snap_raw does, into <snapshots>_alarm/pe=<r>/n=<step>/.
    Never raises: a diagnostic must not be the thing that kills the run."""
    import os
    try:
        base = getattr(M, 'snap_file_name', 'raw_snap')
        base = base[:-4] if base.endswith('.cdf') else base
        d = os.path.join(base + '_alarm', 'pe=%d' % M.my_pe, 'n=%d' % n)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 't.npy'), np.asarray(n*M.dt))
        with open(os.path.join(d, 'why.txt'), 'w') as fh:
            fh.write(why + '\n')
        h = M.halo
        for nm in ('u', 'v', 'w', 'temp', 'salt', 'p_h', 'p_s',
                   'tke', 'K_diss_E', 'K_diss_L', 'kappaH_tke', 'kappaM_tke',
                   'forc_temp_surface', 'forc_salt_surface', 'taux', 'tauy'):
            a = getattr(M, nm, None)
            if a is None:
                continue
            a = np.asarray(a)
            if nm == 'p_s' and a.ndim == 3:            # (3, Ny, Nx), one level
                a = a[0]
            np.save(os.path.join(d, '%s.npy' % nm), M.unpad(a))
        if M.my_pe == 0:
            print('canary: state of step %d written to %s_alarm/'
                  % (n, base), flush=True)
    except Exception as e:
        if M.my_pe == 0:
            print('canary: the alarm dump failed (%r), continuing to the abort'
                  % e, flush=True)
    return


def _ocean_canary(M, comm, n, quiet=False):
    """max|u| (printed every 100 steps, `quiet` suppresses the line on the
    steps in between), and an ALARM on anything non-finite or beyond
    CANARY_BOUNDS. On an alarm every rank that holds such a cell prints where
    it is (level, longitude, latitude, depth, the column's wet levels and the
    four neighbouring values), the whole step is written as a raw snapshot
    (_canary_dump) and the run stops BEFORE the restart write, so nothing on
    disk is poisoned and the state that produced it survives.

    The check is on temp and salt as well as u, because a tracer is what goes
    first: the TEOS-10 polynomial overflows on an out-of-range T or S and the
    velocity still looks calm one step before the NaN."""
    from mpi4py import MPI as _MPI
    h  = M.halo
    hb = slice(h, -h)
    flds = _canary_fields(M)
    umax = float(np.abs(np.asarray(M.u)[hb, hb, hb]).max())
    # vertical CFL, and the flat index of the cell it sits in (EXP_CANARY_CFL)
    cmax, cwhere = 0.0, -1
    if CANARY_CFL:
        wc = (np.abs(np.asarray(M.w)[hb, hb, hb])*M.dt
              / np.asarray(M.dzt)[hb][:, None, None]
              * np.asarray(M.maskT)[hb, hb, hb])
        cwhere = int(wc.argmax())
        cmax = float(wc.flat[cwhere])
    # local alarm: how many cells are out of bounds, in each field
    bad = {}
    for nm, a, lo, hi in flds:
        m = ~np.isfinite(a) | (a < lo) | (a > hi)
        if m.any():
            bad[nm] = m
    nloc = sum(int(m.sum()) for m in bad.values())
    if comm.Get_size() > 1:
        umax = comm.allreduce(umax, op=_MPI.MAX)
        ntot = comm.allreduce(nloc, op=_MPI.SUM)
    else:
        ntot = nloc
    cfl_txt = ''
    if CANARY_CFL:
        # MAXLOC over the ranks, so the line can say WHERE the worst cell is
        if comm.Get_size() > 1:
            cmax, rank_of = comm.allreduce((cmax, comm.Get_rank()), op=_MPI.MAXLOC)
        else:
            rank_of = 0
        if comm.Get_rank() == rank_of and cwhere >= 0:
            shp = np.asarray(M.maskT)[hb, hb, hb].shape
            k, j, i = np.unravel_index(cwhere, shp)
            dzt_ = np.asarray(M.dzt)[hb]
            dep = float((np.cumsum(dzt_[::-1])[::-1] - 0.5*dzt_)[k])
            cfl_txt = (' | max w dt/dz = %.3f at lon %.3f lat %.3f, %.1f m'
                       % (cmax, float(np.asarray(M.xt)[hb][i]),
                          float(np.asarray(M.yt)[hb][j]), dep))
        if comm.Get_size() > 1:
            cfl_txt = comm.bcast(cfl_txt, root=rank_of)
        if CANARY_CFL_ALARM and cmax > CANARY_CFL_ALARM:
            if comm.Get_rank() == 0:
                print('OCEAN ALARM at coupling step %d (t = %.3f d): the '
                      'vertical CFL reached %.3f, above EXP_CANARY_CFL = %.3f'
                      '%s' % (n, n*M.dt/86400.0, cmax, CANARY_CFL_ALARM,
                              cfl_txt), flush=True)
            _canary_dump(M, n, 'coupling step %d, vertical CFL %.3f above %.3f'
                         % (n, cmax, CANARY_CFL_ALARM))
            comm.Barrier()
            raise RuntimeError('vertical CFL %.3f above EXP_CANARY_CFL = %.3f '
                               'at coupling step %d -- aborting BEFORE the '
                               'restart write' % (cmax, CANARY_CFL_ALARM, n))
    if ntot == 0:
        if comm.Get_rank() == 0 and not quiet:
            print('ocean canary: max|u| = %.3f m/s%s%s'
                  % (umax, '   <-- WARNING: unusually fast'
                     if umax > 2.5 else '', cfl_txt), flush=True)
        return
    # ---- alarm ----------------------------------------------------------
    if comm.Get_rank() == 0:
        print('OCEAN ALARM at coupling step %d (t = %.3f d): %d cells outside '
              'u %g, temp %g..%g, salt %g..%g, or non-finite'
              % (n, n*M.dt/86400.0, ntot, CANARY_BOUNDS['u'],
                 CANARY_BOUNDS['temp'][0], CANARY_BOUNDS['temp'][1],
                 CANARY_BOUNDS['salt'][0], CANARY_BOUNDS['salt'][1]), flush=True)
        if CANARY_EVERY > 1:
            print('  the check runs every %d steps, so this may have started '
                  'up to %d steps (%.2f h) ago -- the free-surface solve is '
                  'global and spreads a NaN over the whole domain in one '
                  'step. Rerun with EXP_CANARY_EVERY=1 to catch it locally.'
                  % (CANARY_EVERY, CANARY_EVERY - 1,
                     (CANARY_EVERY - 1)*M.dt/3600.0), flush=True)
    # where this rank's block starts, and its coordinates
    j0, i0 = M.ys_pe - 1, M.xs_pe - 1
    xt = np.asarray(M.xt)[hb]; yt = np.asarray(M.yt)[hb]
    dzt = np.asarray(M.dzt)[hb]
    depth = np.cumsum(dzt[::-1])[::-1] - 0.5*dzt          # below the surface
    mT = np.asarray(M.maskT)[hb, hb, hb]
    nwet = mT.sum(axis=0)
    for nm, m in bad.items():
        a = dict((f[0], f[1]) for f in flds)[nm]
        idx = np.argwhere(m)[:CANARY_LIST]
        for k, j, i in idx:
            print('  ALARM pe %d %-4s = %12.4e at lon %8.3f lat %8.3f  level '
                  'k=%d (%.1f m, %d wet in the column) | W %10.3e E %10.3e '
                  'S %10.3e N %10.3e | below %10.3e above %10.3e'
                  % (M.my_pe, nm, a[k, j, i], xt[i], yt[j], k, depth[k],
                     nwet[j, i],
                     a[k, j, max(i-1, 0)], a[k, j, min(i+1, a.shape[2]-1)],
                     a[k, max(j-1, 0), i], a[k, min(j+1, a.shape[1]-1), i],
                     a[max(k-1, 0), j, i], a[min(k+1, a.shape[0]-1), j, i]),
                  flush=True)
        if len(np.argwhere(m)) > CANARY_LIST:
            print('  ALARM pe %d %-4s: %d more cells not listed'
                  % (M.my_pe, nm, len(np.argwhere(m)) - CANARY_LIST), flush=True)
    _canary_dump(M, n, 'coupling step %d, %d cells outside the canary bounds'
                 % (n, ntot))
    comm.Barrier()                       # let every rank finish its dump
    raise RuntimeError('ocean state outside the canary bounds at coupling step '
                       '%d (%d cells) -- aborting BEFORE the restart write'
                       % (n, ntot))


def run_wave(M, cfg, comm, intercomm, active):
    steps = cfg['steps']; sub_cycle = cfg['sub_cycle']; rsteps = cfg['rsteps']
    ny = cfg['ny']; nx = cfg['nx']
    ocean_dz_r = cfg['ocean_dz_r']; ocean_nz_r = cfg['ocean_nz_r']
    use_stress_work = cfg['use_stress_work']
    cpl_writer = cfg.get('cpl_writer')
    every = max(1, int(cfg.get('cpl_write_every', 1)))
    restart_wave_file = cfg['restart_wave_file']
    cmap = cfg.get('cmap')
    bx = block_exchange(M, comm, intercomm, cmap, 'wave')   # block to block
    if cmap is not None:
        # coarser wave grid: the static sums of the ocean's weights, per wave
        # cell and level. A wave cell with no wet ocean cell at a level gets
        # a zero current there.
        hDu = bx.post_recv((ocean_nz_r,), 11)
        hDv = bx.post_recv((ocean_nz_r,), 12)
        _du, _dv = bx.complete_recv_sums(hDu), bx.complete_recv_sums(hDv)
        inv_wU = np.where(_du > 0.0, 1.0/np.where(_du > 0.0, _du, 1.0), 0.0)
        inv_wV = np.where(_dv > 0.0, 1.0/np.where(_dv > 0.0, _dv, 1.0), 0.0)
        del _du, _dv

    prof = Profiler(enabled=cfg.get('profile', 0) > 0)
    if cfg.get('profile', 0) > 1:
        prof.wrap(M, WAVE_METHODS, prefix='wim.')
    prof_every = max(1, int(cfg.get('profile_every', 20)))
    inv_w2 = _wave_hub_start(M, cfg, comm, bx)       # None unless the stress comes from the ocean

    h   = M.halo
    isl = slice(h, -h)
    area = M.area_t[isl, isl]                      # (ny_loc,nx_loc), spherical
    # physical cell lengths for the CFL alarm. cost_y/dyt_y are (Ny,1) and
    # dxt_x is (1,Nx), so they must be broadcast to (Ny,Nx) BEFORE the
    # interior slice -- slicing a (Ny,1) array in x leaves nothing.
    dx_phys = np.broadcast_to(M.cost_y*M.dxt_x, (M.Ny, M.Nx))[isl, isl]
    dy_phys = np.broadcast_to(M.cost_y*M.dyt_y, (M.Ny, M.Nx))[isl, isl]
    cosu_i  = np.broadcast_to(M.cosu_y,         (M.Ny, M.Nx))[isl, isl]

    if cfg['do_restart']:
        att = wave_restart_read(M, restart_wave_file, ny, nx, comm)
        n0  = int(att['n_cpl']);  pre = int(att['pre'])
        tau, taum1, taum2 = int(att['tau']), int(att['taum1']), int(att['taum2'])
    else:
        # --- wave-only spinup (fast model, its own dt), at the t = 0 wind ---
        n0  = 0
        pre = int(cfg['spinup_time']/M.dt)
        tau, taum1, taum2 = 2, 1, 0
        _ts = time.perf_counter()
        for n in range(pre):
            M.update_forcing(0.0)
            M.time_step(1., 0., 0., tau, taum1, taum2)
            if n % 100 == 0: M.check_monotonicity(n)
            M.diagnose(n, tau)
            tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
            # the spinup is wave-only: the ocean is blocked and NO coupler
            # diagnostic fires, so say where we are or the run looks hung
            if comm.Get_rank() == 0 and pre > 20 and (n + 1) % max(1, pre//10) == 0:
                el = time.perf_counter() - _ts
                print('wave spinup %d/%d sub-steps (%.0f%%), %.2f s/step, '
                      '~%.0f s to the first coupling step'
                      % (n+1, pre, 100.*(n+1)/pre, el/(n+1),
                         el/(n+1)*(pre-n-1)), flush=True)
        if comm.Get_rank() == 0:
            print('wave spinup done (%d steps)' % pre, flush=True)

    # --- coupled run: sub_cycle wave steps per ocean (coupling) step ---
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
    _tw0 = time.perf_counter()
    cum_in = np.zeros(2); cum_del = np.zeros(2); mw0 = None
    for n in range(n0, n_stop):
        prof.steps += 1
        t_cpl = n*sub_cycle*M.dt                       # = n*ocean_dt, absolute
        A_old = M.A.copy()                           # for W_dtA (below)
        pin_loc = 0.0
        with prof.section('wave sub-cycle'):
            for m in range(sub_cycle):
                step = pre + n*sub_cycle + m           # absolute wave step index
                M.update_forcing(t_cpl + m*M.dt)
                M.time_step(1., 0., 0., tau, taum1, taum2)  # always Euler forward
                if inv_w2 is not None:                  # wind input for the atmosphere's q_int
                    pin_loc = pin_loc + M.wind_input_field()/sub_cycle
                if step % 10 == 0: M.check_monotonicity(step)
                M.diagnose(step, tau)
                tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)

        # --- wave-side ALARM: spectral AND spatial advection velocities,
        # NON-FINITE detection, and -- on a cfl alarm -- a one-shot INGREDIENT
        # dump at the argmax point, so the corrupted input names itself. All
        # prints per-rank, no collectives; the dump is capped per PE and
        # try/except-wrapped: diagnostics must never take down the model.
        # The horizontal CFLs carry exactly the metric factors the transport
        # uses (propagation.tracer_flux_superbee).
        prof._enter() if prof.enabled else None
        _tal = time.perf_counter()
        cfl_d = np.abs(M.ddot[:, :, isl, isl])*M.dt/M.dd
        cfl_f = np.abs(M.fdot[:, isl, isl, isl])*M.dt/M.dft[None, isl, None, None]
        cfl_x = np.abs(M.xdot[:, :, isl, isl])*M.dt/dx_phys
        cfl_y = np.abs(M.ydot[:, :, isl, isl]*cosu_i)*M.dt/dy_phys
        # A2d alarm: column-summed action against the LOCAL median. On a
        # global block that is part land and part calm the plain median
        # collapses toward zero, so it is taken over WET, NON-EMPTY cells
        # only, and the alarm is disarmed until the field is meaningful. The
        # factor is 1000, not 50: a storm against a calm region is routinely
        # 100x in action, and 50x fires on every ordinary sea state.
        Aloc  = M.A[:, :, isl, isl].sum(axis=(0, 1))
        _wet  = np.asarray(M.maskTp)[isl, isl] > 0
        _live = Aloc[_wet & (Aloc > 0.0)] if _wet.any() else Aloc[Aloc > 0.0]
        _med  = float(np.median(_live)) if _live.size else 0.0
        # A MINIMUM SAMPLE, or the median is not a reference. This alarm
        # compares a cell with the median of its OWN PE's block, so it gets
        # weaker as the decomposition gets finer: at 64 x 64 wave PEs the
        # blocks are 17 x 11 = 187 cells, and one that is mostly land or
        # mostly calm has a median over a handful of values, which any
        # ordinary energetic cell then exceeds a thousandfold. That fired
        # steadily on one PE of the 1/3 degree run at a flat 2.6 % drift,
        # i.e. on nothing. Below A2D_MIN_LIVE live cells the check is
        # disarmed and the physical Hs bound below carries the block.
        _a2thr = (1e3*_med if (_med > 1e-12 and _live.size >= A2D_MIN_LIVE)
                  else np.inf)
        # physical bound, independent of any median: significant wave height.
        # The largest ever measured is ~20 m, so 30 m is unambiguously wrong
        # rather than merely unusual.
        _Hs = 4.0*np.sqrt(np.maximum(np.asarray(M.e_mean)[isl, isl], 0.0))
        if not hasattr(M, '_alarm_dumps'):
            M._alarm_dumps = 0
        for name, fld, thr in (('cfl_d', cfl_d, 0.5), ('cfl_f', cfl_f, 0.5),
                               ('cfl_x', cfl_x, 0.35), ('cfl_y', cfl_y, 0.35),
                               ('A2d', Aloc, _a2thr),
                               ('Hs [m]', _Hs, 30.0)):
            mx = float(np.max(fld))
            if not np.isfinite(mx):
                print('WAVE ALARM pe %d step %d: %s NON-FINITE'
                      % (M.my_pe, n, name), flush=True)
                continue
            if mx > thr:
                idx = np.unravel_index(int(np.argmax(fld)), fld.shape)
                print('WAVE ALARM pe %d step %d: %s = %.3e at local idx %s'
                      % (M.my_pe, n, name, mx, idx), flush=True)
                if name.startswith('cfl') and M._alarm_dumps < 20:
                    M._alarm_dumps += 1
                    try:
                        _d, _f, _j, _i = idx
                        if name == 'cfl_f': _f = _f + h    # back to haloed f
                        _J, _I = _j + h, _i + h            # back to haloed y/x
                        def _g(a, *ix):
                            try:    return float(np.asarray(getattr(M, a))[ix])
                            except Exception: return float('nan')
                        print('  ingredients pe %d (J%d I%d f%d d%d): '
                              'U=%.3e V=%.3e Udf=%.3e Vdf=%.3e | om_mean=%.3e '
                              '| ustar=%.3e udir=%.3e | fdot=%.3e ddot=%.3e '
                              '| A=%.3e'
                              % (M.my_pe, _J, _I, _f, _d,
                                 _g('ocean_U', _J, _I), _g('ocean_V', _J, _I),
                                 _g('ocean_Udf', _f, _J, _I),
                                 _g('ocean_Vdf', _f, _J, _I),
                                 _g('om_mean', _J, _I),
                                 _g('ustar', _J, _I), _g('udir', _J, _I),
                                 _g('fdot', _d, _f, _J, _I),
                                 _g('ddot', _d, _f, _J, _I),
                                 _g('A', _d, _f, _J, _I)), flush=True)
                    except Exception as _e:
                        print('  ingredient dump failed pe %d: %r'
                              % (M.my_pe, _e), flush=True)
        if prof.enabled: prof._exit('cfl alarm', _tal)

        # --- couple once per ocean step (after the full wave sub-cycle) ---
        # each wave PE computes its LOCAL block; gather to wave-rank-0
        # DETERMINISTIC IN n ONLY. This gate gates COLLECTIVE gathers below,
        # so it must evaluate identically on every wave PE. `cpl_writer` is
        # non-None on rank 0 alone, so folding it in here made rank 0 enter a
        # gather the other ranks skipped -- a hang at the first write step.
        do_write = ((n + 1) % every == 0)
        with prof.section('energy diagnostics'):
            pw  = M.wind_work()            # collective: call on ALL wave PEs
            wnl = M.snl_work()             # collective: int U.k S_nl dk (Eq. 11)
            wdtA = wave_dtA_work(M, A_old, sub_cycle*M.dt)   # local, reduced below
            if comm.Get_size() > 1:
                from mpi4py import MPI
                wdtA = comm.allreduce(wdtA, op=MPI.SUM)
            del A_old
            phi_loc = M.calc_phioc()                          # (ny_loc,nx_loc)
            # significant wave height Hs = 4 sqrt(m0), m0 = int E df dtheta,
            # for the ocean's surface mixing length (NEMO ln_mxhsw: z0 =
            # 1.6 Hs). Prognostic range only, no tail.
            _slw = slice(M.halo, -M.halo)
            _m0 = np.sum(np.sum(M.E[_slw, _slw, _slw, _slw]*M.dd, axis=0)
                         * M.dft[_slw, None, None], axis=0)
            hs_loc = 4.0*np.sqrt(np.maximum(_m0, 0.0))         # (ny_loc,nx_loc)
            pdiss = float(np.sum(phi_loc*area))
            if comm.Get_size() > 1:
                from mpi4py import MPI
                pdiss = comm.allreduce(pdiss, op=MPI.SUM)
        if n % 20 == 0 and comm.Get_rank() == 0:
            print('energy diag [m^5/s^3 /rho]: wind-work into waves %.3e | '
                  'dissipation flux to ocean %.3e | U.k S_nl exchange %.3e'
                  % (pw, pdiss, wnl), flush=True)
        if getattr(M, 'open_coasts', False) and (n + 1) % 100 == 0:
            lA, lE, lO = M.coastal_loss()
            if comm.Get_rank() == 0:
                print('open coasts: cumulative export  A %.3e  E %.3e  '
                      'Eabs %.3e' % (lA, lE, lO), flush=True)

        # ---- momentum budget [kinematic; stresses m^2/s^2 * area] ----
        # input tau_in = int gamma_in E omega (sin,cos) dOmega (resolved +
        # analytic tail); wave momentum content M_w = int E omega (sin,cos)
        # dOmega, so tau = dM_w/dt bin-wise.
        prof._enter() if prof.enabled else None
        _tmb = time.perf_counter()
        kxw = (M.omega_t[None, :]*M.sinth[:, None])[isl, isl]
        kyw = (M.omega_t[None, :]*M.costh[:, None])[isl, isl]
        dft = M.dft[isl]

        def _mfield(F):                       # local (2,ny_loc,nx_loc) field
            fx = np.sum(np.sum(F[isl, isl, isl, isl]*kxw[:, :, None, None]*M.dd,
                               axis=0)*dft[:, None, None], axis=0)
            fy = np.sum(np.sum(F[isl, isl, isl, isl]*kyw[:, :, None, None]*M.dd,
                               axis=0)*dft[:, None, None], axis=0)
            return np.stack([fx, fy])

        tin_loc = _mfield(M.gamma_in*M.E)      # wind -> wave input stress field
        if hasattr(M, '_tail_input_factors'):          # analytic tail input
            ft_t, _, _, _ = M._tail_input_factors()
            Ecg = M.E[:, -h, :, :]
            tin_loc[0] += np.sum(Ecg[isl, isl, isl]*ft_t[isl, isl, isl]
                                 *M.sinth[isl, None, None]*M.dd, axis=0)
            tin_loc[1] += np.sum(Ecg[isl, isl, isl]*ft_t[isl, isl, isl]
                                 *M.costh[isl, None, None]*M.dd, axis=0)
        min_x = float(np.sum(tin_loc[0]*area))
        min_y = float(np.sum(tin_loc[1]*area))
        mwf   = _mfield(M.E)
        mw_x  = float(np.sum(mwf[0]*area))
        mw_y  = float(np.sum(mwf[1]*area))

        if prof.enabled: prof._exit('momentum budget', _tmb)

        with prof.section('coupling fields'):
            u_S_loc = M.calc_stokes(ocean_dz_r, ocean_nz_r)  # (2,nz,ny_loc,nx_loc)
            if use_stress_work:
                stress_loc = M.calc_stress_work(ocean_dz_r, ocean_nz_r)
                dsl = np.sum(stress_loc*np.asarray(ocean_dz_r)[None, :, None, None],
                             axis=1)                      # column sum
            else:
                stress_loc = M.calc_stress()              # (2,ny_loc,nx_loc)
                dsl = stress_loc
        mdel_x = float(np.sum(dsl[0]*area))
        mdel_y = float(np.sum(dsl[1]*area))
        if comm.Get_size() > 1:
            from mpi4py import MPI as _MPI
            (min_x, min_y, mw_x, mw_y, mdel_x, mdel_y) = comm.allreduce(
                np.array([min_x, min_y, mw_x, mw_y, mdel_x, mdel_y]), op=_MPI.SUM)
        # cumulative budget since segment start (dt_cpl per coupling step)
        if mw0 is None:
            mw0 = np.array([mw_x, mw_y])
        dt_cpl   = sub_cycle*M.dt
        cum_in  += np.array([min_x, min_y])*dt_cpl
        cum_del += np.array([mdel_x, mdel_y])*dt_cpl
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            dmw = np.array([mw_x, mw_y]) - mw0
            res = cum_in - cum_del - dmw
            print('momentum budget x (cum. since seg start) [m^4/s]: in %.3e | '
                  'delivered %.3e (%.0f%%) | d storage %.3e (%.0f%%) | '
                  'residual %.3e (%.0f%%)'
                  % (cum_in[0], cum_del[0], 100*cum_del[0]/max(abs(cum_in[0]), 1e-30),
                     dmw[0], 100*dmw[0]/max(abs(cum_in[0]), 1e-30),
                     res[0], 100*res[0]/max(abs(cum_in[0]), 1e-30)), flush=True)

        # BLOCK-TO-BLOCK: every wave PE posts its own block of each field
        # straight to the ocean PEs that own those cells. Nothing is
        # assembled on rank 0 any more (see coupler_mpi.BlockExchange).
        with prof.section('send blocks to ocean'):
            # EVERY RECEIVE POSTED BEFORE ANY SEND: see BlockExchange.post_recv
            hU = bx.post_recv((ocean_nz_r,), 1)
            hV = bx.post_recv((ocean_nz_r,), 2)
            if inv_w2 is not None:
                hTx = bx.post_recv((), 9); hTy = bx.post_recv((), 10)
                bx.send_blocks(pin_loc, 12)
            bx.send_blocks(u_S_loc,    3)
            bx.send_blocks(stress_loc, 4)
            bx.send_blocks(phi_loc,    5)
            bx.send_blocks(tin_loc,    6)
            bx.send_blocks(hs_loc,     8)
        if do_write:
            # WRITER-ONLY gathers: deterministic in n, so the collective is safe
            sl = slice(h, -h)
            ustar_x = gather_unpadded_mpi(
                M, (M.ustar*np.sin(M.udir))[sl, sl], ny, nx)     # eastward
            ustar_y = gather_unpadded_mpi(
                M, (M.ustar*np.cos(M.udir))[sl, sl], ny, nx)     # northward
            # the writer wants the full domain on rank 0; only it does, and
            # only on its own cadence
            u_S    = gather_unpadded_mpi(M, u_S_loc,    ny, nx)  # (2,nz,ny,nx)
            stress = gather_unpadded_mpi(M, stress_loc, ny, nx)  # (2,[nz,]ny,nx)
            phioc  = gather_unpadded_mpi(M, phi_loc,    ny, nx)  # (ny,nx)
            tau_in = gather_unpadded_mpi(M, tin_loc,    ny, nx)  # (2,ny,nx)

        # receive this wave PE's own block of the ocean current directly from
        # the ocean PEs that own it, then Doppler-project it (calc_doppler_vel
        # does the local pad + halo exchange across wave PEs).
        #
        # THIS COMES BEFORE THE RANK-0 SCALAR EXCHANGE BELOW, and it must.
        # With the scalars first the two sides deadlock: the ocean cannot
        # finish its Isend of u and v until this Irecv is posted (a large
        # message goes through rendezvous), it therefore never reaches its
        # tag-7 send, and wave rank 0 waits for that tag-7 forever. The two
        # block transfers have to match each other directly.
        with prof.section('recv blocks + Doppler'):
            if cmap is None:
                UL = bx.complete_recv(hU)           # (nz,ny_loc,nx_loc)
                VL = bx.complete_recv(hV)
            else:                   # mean over the wet ocean cells of the cell
                UL = bx.complete_recv_sums(hU)*inv_wU
                VL = bx.complete_recv_sums(hV)*inv_wV
            if inv_w2 is not None:              # the atmosphere's stress, area mean per wave cell
                if cmap is None:
                    tx_w = bx.complete_recv(hTx); ty_w = bx.complete_recv(hTy)
                else:
                    tx_w = bx.complete_recv_sums(hTx)*inv_w2; ty_w = bx.complete_recv_sums(hTy)*inv_w2
                M._forcing.set({'taux': tx_w, 'tauy': ty_w})
            bx.wait_sends()
            u, v = M.calc_doppler_vel(ocean_dz_r, ocean_nz_r, UL, VL)
            if hasattr(M, 'ocean_U'):
                M.ocean_Udf, M.ocean_Vdf = M.calc_doppler_vel_df(
                    ocean_dz_r, ocean_nz_r, UL, VL)
            M.ocean_U, M.ocean_V = u, v

        prof._enter() if prof.enabled else None
        _tex = time.perf_counter()
        if comm.Get_rank() == 0:
            # only the scalars still go rank 0 to rank 0: the ocean-side
            # energy diagnostics of THIS coupling step, computed after the
            # ocean received u_S and the stress.
            # THIS RECEIVE IS WHERE THE WAVE MODEL WAITS FOR THE OCEAN.
            (wst, wste, cxc, wvf, oke, dke, wvfr, kdv, whf, wvnum) = intercomm.recv(source=0, tag=7)
            if do_write and cpl_writer is not None:
                cpl_writer.write(t_cpl + dt_cpl,
                                 ustar_x, ustar_y, u_S, stress, phioc=phioc,
                                 stress_in=tau_in,
                                 ediag=(pw, wst, wste, cxc, wvf, pdiss, wnl),
                                 kediag=(oke, dke, wvfr, kdv, whf, wdtA, wvnum),
                                 mdiag=(min_x, min_y, mdel_x, mdel_y, mw_x, mw_y))
        if prof.enabled: prof._exit('exchange with ocean', _tex)

        # periodic restart write (collective gathers: all wave PEs call)
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            with prof.section('restart write'):
                wave_restart_write(M, restart_wave_file, ny, nx, n + 1, pre,
                                   tau, taum1, taum2, 1., 0., 0.)

        _t1 = time.perf_counter()
        with prof.section('barrier (waiting for ocean)'):
            active.Barrier()
        _t2 = time.perf_counter()
        lb_comp += _t1 - _t0; lb_wait += _t2 - _t1; _t0 = _t2
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            print('load wave : compute %.3f s/step | waiting %.3f s/step '
                  '(%.0f%% idle)' % (lb_comp/100., lb_wait/100.,
                  100.*lb_wait/max(lb_comp+lb_wait, 1e-30)), flush=True)
            lb_comp = lb_wait = 0.0
        # the cadence is in COUPLING steps, not wave sub-steps; one early
        # report so a short run is not silent
        # THE FIRST COUPLING STEP IS NOT MEASURABLE and is dropped from the
        # profile: on the wave side it follows the wave-only spinup, and on
        # the ocean side it CONTAINS the wait for that spinup (the ocean
        # blocks in its first collective after the intercomm receive, which
        # showed up as a 99 s 'broadcast + fill' row). Everything before the
        # second step is startup, so the window restarts here.
        if n == n0:
            prof.reset(); _tw0 = time.perf_counter()
            lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
        _due = ((n + 1) % prof_every == 0) or \
               (n + 1 - n0 == 6 and prof_every > 6)
        if prof.enabled and _due:
            _tab = prof.report(comm, 'WAVE', time.perf_counter() - _tw0)
            if _tab: print(_tab, flush=True)
            prof.reset(); _tw0 = time.perf_counter()
    if comm.Get_rank() == 0:
        if cpl_writer is not None:
            cpl_writer.close()
        print('wave done', flush=True)
    return n_stop


def run_ocean(M, cfg, comm, intercomm, active):
    steps = cfg['steps']; rsteps = cfg['rsteps']
    ny = cfg['ny']; nx = cfg['nx']
    use_stress_work = cfg['use_stress_work']
    restart_ocean_file = cfg['restart_ocean_file']
    tA, tB, tC = cfg.get('ocean_ab', (1.6, -0.6, 0.0))

    prof = Profiler(enabled=cfg.get('profile', 0) > 0)
    if cfg.get('profile', 0) > 1:
        prof.wrap(M, OCEAN_METHODS, prefix='pe.')
    prof_every = max(1, int(cfg.get('profile_every', 20)))
    # KE change by vertical/horizontal friction is booked by the ocean class
    # itself (global4_setup.py: vertical_friction / harmonic_momentum overrides)

    h  = M.halo
    hb = slice(h, -h)
    # (the wave fields arrive as blocks now, see the scatter below, so this
    # PE's global slices are no longer needed here)

    cmap = cfg.get('cmap')
    bx = block_exchange(M, comm, intercomm, cmap, 'ocean')  # block to block
    bxa = _ocean_hub_start(M, cfg, comm)                    # the atmosphere, or None
    hub = bool(cfg.get('wind_hub'))                          # the waves' wind comes from here
    if cmap is not None:
        # coarser wave grid: the current goes out as sum(w u) per wave cell,
        # w = cell area times the level's mask at the U and V points. The sums
        # of w are static and sent once, here.
        w_u = np.asarray(M.area_u*M.maskU)[hb, hb, hb]
        w_v = np.asarray(M.area_v*M.maskV)[hb, hb, hb]
        bx.send_sums(None, w_u, 11)
        bx.send_sums(None, w_v, 12)
        if hub:
            bx.send_sums(None, M._w_wet, 13)                # for the mean of the stress
        bx.wait_sends()

    n0, tau, taum1, taum2 = _ocean_start(M, cfg, comm)
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
    _tw0 = time.perf_counter()
    for n in range(n0, n_stop):
        prof.steps += 1
        # one ocean step per coupling interval. _gitt is the model clock the
        # monthly climatology in set_forcing interpolates on.
        M._gitt = n
        ke_prev = ocean_ke(M)
        with prof.section('ocean time step'):
            M._ab = _ab_weights((tA, tB, tC), n, cfg)   # exp_ocean's identity reads it
            M.time_step_implicit_pressure(*M._ab, tau, taum1, taum2)
        if (n + 1) % CANARY_EVERY == 0:
            # the check runs on its own cadence, the line stays every 100
            _ocean_canary(M, comm, n, quiet=(n + 1) % 100 != 0)
        with prof.section('snapshot'):
            M.diagnose(n, tau)
        tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)

        # ocean -> wave: every ocean PE posts its OWN block of the current
        # straight to the wave PEs that own those cells. No gather, no
        # full-domain array, nothing through rank 0.
        with prof.section('send blocks to waves'):
            # EVERY RECEIVE POSTED BEFORE ANY SEND: see BlockExchange.post_recv
            nzi = M.Nz - 2*h
            hS  = bx.post_recv((2, nzi), 3)
            hT  = bx.post_recv((2, nzi) if use_stress_work else (2,), 4)
            hP  = bx.post_recv((), 5)
            hI  = bx.post_recv((2,), 6)
            hH  = bx.post_recv((), 8)
            if hub:
                hPin = bx.post_recv((), 12)                 # p_in from the waves
            # the INTERIOR, z halo included: gather_array_mpi returned
            # (nz, ny, nx) and the wave model sizes its buffer on ocean_nz_r
            if cmap is None:
                bx.send_blocks(np.asarray(M.u)[hb, hb, hb], 1)
                bx.send_blocks(np.asarray(M.v)[hb, hb, hb], 2)
            else:
                bx.send_sums(np.asarray(M.u)[hb, hb, hb], w_u, 1)
                bx.send_sums(np.asarray(M.v)[hb, hb, hb], w_v, 2)
            if hub:                                         # the stress on to the waves
                if bxa is not None:                         # the atmosphere's, as received
                    F = M._forcing.get(0.0)
                    txw, tyw = F['taux'], F['tauy']
                else:                                       # the relative-wind stress of the last step
                    txw, tyw = getattr(M, '_tau_rel_T', (np.zeros(M._w_wet.shape),)*2)
                if cmap is None:
                    bx.send_blocks(txw, 9); bx.send_blocks(tyw, 10)
                else:
                    bx.send_sums(txw, M._w_wet, 9); bx.send_sums(tyw, M._w_wet, 10)

        # ... and receives its own block of the wave fields the same way.
        # THIS RECEIVE IS WHERE THE OCEAN WAITS FOR THE WAVE MODEL.
        #
        # Two earlier forms, both recorded because they are why this one
        # exists. A comm.bcast of the full domain gave every PE the whole
        # 1.52 GB u_S and OOM-killed the ocean. A Scatterv from rank 0 fixed
        # the memory but not the time: the root's strided pack of the
        # full-domain array is the floor, 4.6 s of a 7.5 s coupling step.
        # Now nothing is ever assembled -- see coupler_mpi.BlockExchange.
        prof._enter() if prof.enabled else None
        _tfl = time.perf_counter()
        u_S    = bx.complete_recv(hS)
        stress = bx.complete_recv(hT)
        phioc  = bx.complete_recv(hP)
        tau_in = bx.complete_recv(hI)
        hs     = bx.complete_recv(hH)               # significant wave height
        if hub:
            M._p_in_w = bx.complete_recv(hPin)
        bx.wait_sends()

        # Stokes drift -> interior, then halo fill. Keep the previous one for
        # the exchange-work diagnostic C = int u_L d(u_S)/dt dV (explicit
        # copies: modify_array mutates in place with the numpy backend).
        M.u_S_old, M.v_S_old = M.u_S.copy(), M.v_S.copy()
        M.u_S = OM.modify_array(M.u_S, (hb, hb, hb), u_S[0])
        M.v_S = OM.modify_array(M.v_S, (hb, hb, hb), u_S[1])
        M.u_S, M.v_S = M.apply_bc(M.u_S), M.apply_bc(M.v_S)

        if use_stress_work:
            M.tau_work_x = M.apply_bc(OM.modify_array(
                M.tau_work_x, (hb, hb, hb), stress[0]))
            M.tau_work_y = M.apply_bc(OM.modify_array(
                M.tau_work_y, (hb, hb, hb), stress[1]))
        else:
            M.taux = M.apply_bc(OM.modify_array(M.taux, (hb, hb),
                                                stress[0]))
            M.tauy = M.apply_bc(OM.modify_array(M.tauy, (hb, hb),
                                                stress[1]))
        # wave dissipation energy flux (WAM: PHIOC) -> surface TKE flux BC of
        # the ocean's turbulence closure. Kinematic [m^3/s^3], 2-D.
        M.phi_oc = M.apply_bc(OM.modify_array(M.phi_oc, (hb, hb),
                                              phioc))
        M.tau_in_x = M.apply_bc(OM.modify_array(M.tau_in_x, (hb, hb),
                                                tau_in[0]))
        M.tau_in_y = M.apply_bc(OM.modify_array(M.tau_in_y, (hb, hb),
                                                tau_in[1]))
        if hasattr(M, 'hs'):
            # significant wave height, and the surface mixing length the
            # configuration derives from it (NEMO ln_mxhsw). apply_bc here is
            # the ONE exchange this needs: the closure then reads z0_surf with
            # a valid halo, and hs is restart state, so the first step after a
            # restart uses the right value.
            M.hs = M.apply_bc(OM.modify_array(M.hs, (hb, hb), hs))
            if hasattr(M, 'set_wave_roughness'):
                M.set_wave_roughness()
        if prof.enabled: prof._exit('recv blocks + fill', _tfl)

        # energy diagnostics every coupling step (collective over the ocean
        # communicator: call on ALL ocean PEs); rank 0 sends the scalars to
        # wave-rank-0 (tag 7) for the coupling record. MUST run BEFORE
        # active.Barrier(): wave-rank-0 blocks on this receive before reaching
        # its own barrier, so sending after ours deadlocks.
        with prof.section('energy diagnostics'):
            wvf  = M.vortex_work()
            wst  = M.stress_work_diag()
            wste = M.stress_work_eul_diag()  # against u = u_L - u_S (comparison)
            cxc  = M.exchange_work_C()
            # KE budget of u_L for this step (all as rates, m^5/s^3 per rho):
            # ocean KE, dKE/dt, KE change by vertical friction, -int K_diss_v
            # dV it must equal, KE change by horizontal friction
            oke  = ocean_ke(M)
            loc  = np.array([oke, (oke - ke_prev)/M.dt,
                             M._ke_vfric/M.dt, M._ke_hfric/M.dt,
                             float(OM.np.sum(M.K_diss_v[hb, hb, hb]*M.area_t[0][hb, hb]
                                             *M.dzw_z[hb]))
                             if hasattr(M, 'K_diss_v') else 0.0,
                             M._ke_vnum/M.dt])
            if comm.Get_size() > 1:
                from mpi4py import MPI as _MPI
                loc = comm.allreduce(loc, op=_MPI.SUM)
            oke, dke, wvfr, whf, kdv, wvnum = (float(x) for x in loc)
            if comm.Get_rank() == 0:
                intercomm.send((wst, wste, cxc, wvf, oke, dke, wvfr, kdv, whf, wvnum),
                               dest=0, tag=7)
        if M.my_pe == 0 and n % 20 == 0:
            print('energy diag [m^5/s^3 /rho]: stress-work on ocean u_L %.3e '
                  '(vs Eulerian u %.3e) | exchange C (exact) %.3e | '
                  'vortex-force u_L-work residual %.3e' % (wst, wste, cxc, wvf),
                  flush=True)
            print('KE budget  [m^5/s^3 /rho]: dKE/dt %.3e | vfric %.3e = -K_diss %.3e '
                  '- numerical %.3e (residual %.1e rel) | hfric %.3e'
                  % (dke, wvfr, -kdv, wvnum,
                     abs(wvfr + kdv + wvnum)/max(abs(kdv), 1e-30), whf), flush=True)
        if (n + 1) % 100 == 0 and getattr(M, 'enable_tke', False):
            ftke = float(np.sum(M.forc_tke_surface[hb, hb]*M.area_t[0][hb, hb]))
            if comm.Get_size() > 1:
                from mpi4py import MPI as _MPI
                ftke = comm.allreduce(ftke, op=_MPI.SUM)
            if comm.Get_rank() == 0:
                print('tke closure: domain-int surface TKE forcing %.4e m^5/s^3 '
                      '(compare diss_work on the wave side)' % ftke, flush=True)
        _ocean_hub_exchange(M, bxa, prof)
        _t1 = time.perf_counter()
        with prof.section('barrier (waiting for waves)'):
            active.Barrier()
        _t2 = time.perf_counter()
        lb_comp += _t1 - _t0; lb_wait += _t2 - _t1; _t0 = _t2
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            print('load ocean: compute %.3f s/step | waiting %.3f s/step '
                  '(%.0f%% idle)' % (lb_comp/100., lb_wait/100.,
                  100.*lb_wait/max(lb_comp+lb_wait, 1e-30)), flush=True)
            lb_comp = lb_wait = 0.0
        # periodic restart write (collective gathers: all ocean PEs call);
        # the same coupling steps as the wave side -> a consistent file pair
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            with prof.section('restart write'):
                ocean_restart_write(M, restart_ocean_file, ny, nx,
                                    use_stress_work, n + 1, tau, taum1, taum2,
                                    tA, tB, tC)
        # the cadence is in COUPLING steps, not wave sub-steps; one early
        # report so a short run is not silent
        # THE FIRST COUPLING STEP IS NOT MEASURABLE and is dropped from the
        # profile: on the wave side it follows the wave-only spinup, and on
        # the ocean side it CONTAINS the wait for that spinup (the ocean
        # blocks in its first collective after the intercomm receive, which
        # showed up as a 99 s 'broadcast + fill' row). Everything before the
        # second step is startup, so the window restarts here.
        if n == n0:
            prof.reset(); _tw0 = time.perf_counter()
            lb_comp = lb_wait = 0.0; _t0 = time.perf_counter()
        _due = ((n + 1) % prof_every == 0) or \
               (n + 1 - n0 == 6 and prof_every > 6)
        if prof.enabled and _due:
            _tab = prof.report(comm, 'OCEAN', time.perf_counter() - _tw0)
            if _tab: print(_tab, flush=True)
            prof.reset(); _tw0 = time.perf_counter()
        if M.my_pe == 0 and n % 20 == 0:
            print('coupled ocean step %d / %d' % (n, steps), flush=True)
    if M.my_pe == 0:
        print('ocean done', flush=True)
    return n_stop


def run_ocean_only(M, cfg, comm, active):
    """The ocean without the wave model: time step, diagnostics, canary,
    restart and profiling exactly as run_ocean, no exchange. The surface
    forcing comes entirely from set_forcing (coupled = False)."""
    steps = cfg['steps']; rsteps = cfg['rsteps']
    ny = cfg['ny']; nx = cfg['nx']
    use_stress_work = cfg['use_stress_work']
    restart_ocean_file = cfg['restart_ocean_file']
    tA, tB, tC = cfg.get('ocean_ab', (1.6, -0.6, 0.0))

    prof = Profiler(enabled=cfg.get('profile', 0) > 0)
    if cfg.get('profile', 0) > 1:
        prof.wrap(M, OCEAN_METHODS, prefix='pe.')
    prof_every = max(1, int(cfg.get('profile_every', 20)))

    h  = M.halo
    hb = slice(h, -h)
    bxa = _ocean_hub_start(M, cfg, comm)                    # the atmosphere, or None
    n0, tau, taum1, taum2 = _ocean_start(M, cfg, comm)
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    lb_comp = 0.0; _t0 = time.perf_counter()
    _tw0 = time.perf_counter()
    for n in range(n0, n_stop):
        prof.steps += 1
        M._gitt = n                    # model clock of the forcing calendar
        with prof.section('ocean time step'):
            M._ab = _ab_weights((tA, tB, tC), n, cfg)   # exp_ocean's identity reads it
            M.time_step_implicit_pressure(*M._ab, tau, taum1, taum2)
        if (n + 1) % CANARY_EVERY == 0:
            # the check runs on its own cadence, the line stays every 100
            _ocean_canary(M, comm, n, quiet=(n + 1) % 100 != 0)
        with prof.section('snapshot'):
            M.diagnose(n, tau)
        tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
        if (n + 1) % 100 == 0 and getattr(M, 'enable_tke', False):
            ftke = float(np.sum(M.forc_tke_surface[hb, hb]*M.area_t[0][hb, hb]))
            if comm.Get_size() > 1:
                from mpi4py import MPI as _MPI
                ftke = comm.allreduce(ftke, op=_MPI.SUM)
            if comm.Get_rank() == 0:
                print('tke closure: domain-int surface TKE forcing %.4e m^5/s^3'
                      % ftke, flush=True)
        if bxa is not None:
            # without waves the Craig-Banner surface TKE flux takes the place of p_in
            # (energy_cycle.tex Sec. 6.5)
            if hasattr(M, 'forc_tke_surface'):
                M._p_in_w = M.cfg.rho_0*np.asarray(M.forc_tke_surface)[hb, hb]
            _ocean_hub_exchange(M, bxa, prof)
            with prof.section('barrier (waiting for the atmosphere)'):
                active.Barrier()
        _t1 = time.perf_counter()
        lb_comp += _t1 - _t0; _t0 = _t1
        if (n + 1) % 100 == 0 and comm.Get_rank() == 0:
            print('load ocean: %.3f s/step' % (lb_comp/100.), flush=True)
            lb_comp = 0.0
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            with prof.section('restart write'):
                ocean_restart_write(M, restart_ocean_file, ny, nx,
                                    use_stress_work, n + 1, tau, taum1, taum2,
                                    tA, tB, tC)
        _due = ((n + 1) % prof_every == 0) or \
               (n + 1 - n0 == 6 and prof_every > 6)
        if prof.enabled and _due:
            _tab = prof.report(comm, 'OCEAN', time.perf_counter() - _tw0)
            if _tab: print(_tab, flush=True)
            prof.reset(); _tw0 = time.perf_counter()
        if M.my_pe == 0 and n % 20 == 0:
            print('ocean step %d / %d' % (n, steps), flush=True)
    if M.my_pe == 0:
        print('ocean done', flush=True)
    return n_stop


# ===========================================================================
def run_atmos(M, cfg, comm, intercomm_ao, active):
    """The atmosphere's coupled loop. Per coupling step (one ocean step): n_sub
    atmosphere steps with the bulk fluxes evaluated every step from the
    current ground-cell state and the SST, u_o of the previous step, the
    interval means of tau and Q_s accumulated; then the exchange with the
    OCEAN (the only partner): the means go down as block copies, the ocean's
    area sums come up. Energy booking as energy_cycle.tex Sec. 6.8."""
    from mpi4py import MPI
    steps = cfg['steps']; rsteps = cfg['rsteps']; ny = cfg['ny']; nx = cfg['nx']
    n_sub = int(cfg['atmos_sub'])
    writer = cfg.get('cpl_writer'); every = max(1, int(cfg.get('cpl_write_every', 1)))
    restart_file = cfg['restart_atmos_file']
    prof = Profiler(enabled=cfg.get('profile', 0) > 0)
    if cfg.get('profile', 0) > 1: prof.wrap(M, OCEAN_METHODS, prefix='atm.')
    prof_every = max(1, int(cfg.get('profile_every', 20)))
    h = M.halo; hb = slice(h, -h); kg = -h-1
    area = np.asarray(M.area_t)[0][hb, hb] if np.asarray(M.area_t).ndim == 3 else np.asarray(M.area_t)[hb, hb]
    area_tot = comm.allreduce(float(area.sum()), op=MPI.SUM)
    mT = np.asarray(M.maskT)[kg]

    # the block exchange with the ocean: the atmosphere is the COARSE side
    bxa = block_exchange(M, comm, intercomm_ao, cfg['cmap_a'], 'wave', key='ocean')
    hW = bxa.post_recv((), 30)
    wsum = bxa.complete_recv_sums(hW)                 # wet ocean area under each cell
    inv_w = np.where(wsum > 0.0, 1.0/np.where(wsum > 0.0, wsum, 1.0), 0.0)
    M.set_ocean_fraction(wsum/area)
    fw = M.f_wet[hb, hb]

    if cfg['do_restart']:
        att = atmos_restart_read(M, restart_file, ny, nx, comm, ATM_FIELDS2D, 'atmosphere')
        n0 = int(att['n_cpl']); tau, taum1, taum2 = int(att['tau']), int(att['taum1']), int(att['taum2'])
        ab = (float(att['tA']), float(att['tB']), float(att['tC']))
    else:
        n0 = 0; tau, taum1, taum2 = 2, 1, 0; ab = (1., 0., 0.)
    seg = cfg.get('seg_steps', 0)
    n_stop = steps if seg <= 0 else min(steps, n0 + seg)
    _tw0 = time.perf_counter()

    def _fill(a, b):
        return np.asarray(M.apply_bc(OM.modify_array(OM.np.asarray(a), (hb, hb), b)))

    for n in range(n0, n_stop):
        prof.steps += 1
        acc = np.zeros((4, M.Ny, M.Nx)); w_tau = q_s = q_int = q_neg = d_bl = d_bih = 0.0
        with prof.section('atmosphere sub-cycle'):
            for m in range(n_sub):
                M.bulk_fluxes()                                   # tau, q_s, w_a, q_int from the current state
                acc[0] += M.tau_ax; acc[1] += M.tau_ay; acc[2] += M.q_s; acc[3] += M.q_int
                M.time_step_implicit_pressure(ab[0], ab[1], ab[2], tau, taum1, taum2)
                w_tau += M._wa_step; q_int += M._qint_diag; q_neg += M._qint_neg
                d_bl  += M._dbl_step; d_bih += M._bih_step
                q_s   += float(np.sum((M.q_s*mT)[hb, hb]*area))
                M.diagnose(n*n_sub + m + 1, tau)
                tau, taum1, taum2 = np.mod(tau+1, 3), np.mod(taum1+1, 3), np.mod(taum2+1, 3)
                ab = _ab_next(ab)
        acc /= n_sub
        scal = comm.allreduce(np.array([w_tau, q_s, q_int, q_neg, d_bl, d_bih])/n_sub, op=MPI.SUM)/area_tot
        w_tau, q_s, q_int, q_neg, d_bl, d_bih = (float(x) for x in scal)
        umax = comm.allreduce(float(np.abs(np.asarray(M.u)[:, hb, hb]).max()), op=MPI.MAX)
        if not np.isfinite(umax):
            raise RuntimeError('atmosphere state is NaN at coupling step %d' % n)
        # the w_o and p_in that entered q_int during this step, before they are replaced:
        # W_a + Q_int + W_o_used + P_in_used is the identity without the lag (roundoff)
        used = comm.allreduce(np.array([float(np.sum((M.w_o*mT)[hb, hb]*area)),
                                        float(np.sum((M.p_in*mT)[hb, hb]*area))]), op=MPI.SUM)/area_tot
        with prof.section('exchange with ocean'):
            # EVERY RECEIVE POSTED BEFORE ANY SEND (BlockExchange.post_recv)
            hS = bxa.post_recv((), 24); hU = bxa.post_recv((), 25); hV = bxa.post_recv((), 26)
            hWo = bxa.post_recv((), 27); hP = bxa.post_recv((), 28); hQ = bxa.post_recv((), 29)
            hI = bxa.post_recv((), 31)
            # the heat flux goes down per unit WET area: q_s is per unit cell area
            qs_wet = np.where(fw > 0.0, acc[2][hb, hb]/np.maximum(fw, 1e-12), 0.0)
            bxa.send_blocks(acc[0][hb, hb], 21); bxa.send_blocks(acc[1][hb, hb], 22)
            bxa.send_blocks(qs_wet, 23)
            sst = bxa.complete_recv_sums(hS)*inv_w
            uo  = bxa.complete_recv_sums(hU)*inv_w; vo = bxa.complete_recv_sums(hV)*inv_w
            wo  = bxa.complete_recv_sums(hWo)/area; pin = bxa.complete_recv_sums(hP)/area
            qoc = bxa.complete_recv_sums(hQ)/area;  fice = bxa.complete_recv_sums(hI)/area
            bxa.wait_sends()
        with prof.section('fill'):
            # cells without ocean keep the last SST (unused: no heat flux there)
            M.sst  = _fill(M.sst, np.where(wsum > 0.0, sst, M.sst[hb, hb]))
            M.u_o  = _fill(M.u_o, uo); M.v_o = _fill(M.v_o, vo)
            M.w_o  = _fill(M.w_o, wo); M.p_in = _fill(M.p_in, pin)
            M.q_ocn = _fill(M.q_ocn, qoc); M.f_ice = _fill(M.f_ice, fice)
            tot = comm.allreduce(np.array([float(np.sum((M.w_o*mT)[hb, hb]*area)),
                                           float(np.sum((M.p_in*mT)[hb, hb]*area)),
                                           float(np.sum((M.q_ocn*mT)[hb, hb]*area)),
                                           float(np.sum((M.f_ice*mT)[hb, hb]*area))]), op=MPI.SUM)/area_tot
        do_write = (n % every == 0)
        if do_write:
            names = ('tau_ax', 'tau_ay', 'q_s', 'q_int', 'u10', 'v10', 't10', 'sst', 'u_o', 'v_o',
                     'w_o', 'p_in', 'q_ocn', 'f_ice', 'bl_h')
            stack = np.stack([acc[0], acc[1], acc[2], acc[3]] + [np.asarray(getattr(M, k)) for k in names[4:]])
            full = gather_unpadded_mpi(M, stack[:, hb, hb], ny, nx)
        if comm.Get_rank() == 0:
            if n % 20 == 0:
                print('coupling %6d  atm: W_a=%8.4f Q_s=%8.4f Q_int=%8.4f (neg %.2f) D_bl=%8.4f D_bih=%8.4f | '
                      'ocn: W_o=%8.4f Q=%8.4f ice %.3f | P_in=%8.4f | closure %9.5f (lagged) %9.2e (exact) '
                      'W/m2 | max|u_a| %.1f m/s'
                      % (n, w_tau, q_s, q_int, q_neg, d_bl, d_bih, tot[0], tot[2], tot[3], tot[1],
                         w_tau + tot[0] + tot[1] + q_int, w_tau + q_int + used[0] + used[1], umax), flush=True)
            if writer is not None and do_write:
                writer.write((n+1)*cfg['cpl_dt'],
                             dict(tau_x=full[0], tau_y=full[1], q_s=full[2], q_int=full[3], u10=full[4],
                                  v10=full[5], t10=full[6], sst=full[7], u_o=full[8], v_o=full[9],
                                  w_o=full[10], p_in=full[11], q_ocn=full[12], f_ice=full[13], bl_h=full[14]),
                             dict(W_tau_atm=w_tau, Q_s_atm=q_s, Q_int_atm=q_int, neg_area=q_neg,
                                  D_bl_atm=d_bl, D_bih_atm=d_bih, P_in=tot[1],
                                  W_tau_ocn=tot[0], Q_ocn=tot[2], ice_area=tot[3]))
        with prof.section('barrier (waiting for the ocean)'):
            active.Barrier()
        if (rsteps > 0 and (n + 1) % rsteps == 0) or n == n_stop - 1:
            with prof.section('restart write'):
                atmos_restart_write(M, restart_file, ny, nx, n + 1, tau, taum1, taum2, ab,
                                    ATM_FIELDS2D, 'atmosphere')
        if n == n0:
            prof.reset(); _tw0 = time.perf_counter()
        _due = ((n + 1) % prof_every == 0) or (n + 1 - n0 == 6 and prof_every > 6)
        if prof.enabled and _due:
            tab = prof.report(comm, 'ATMOSPHERE', time.perf_counter() - _tw0)
            if tab: print(tab, flush=True)
            prof.reset(); _tw0 = time.perf_counter()
    if writer is not None: writer.close()
    if comm.Get_rank() == 0:
        print('atmosphere done', flush=True)
    return n_stop


def _ab_next(ab):
    """Adams-Bashforth ramp of pe_realistic.loop: Euler -> AB2 -> AB3."""
    A, B, C = ab
    if A == 1. and B == 0. and C == 0.: return (1.5, -0.5, 0.)
    if A == 1.5 and B == -0.5:          return (23./12., -16./12., 5./12.)
    return ab
