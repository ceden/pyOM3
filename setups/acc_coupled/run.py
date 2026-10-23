"""
Coupled WIM (surface waves) <-> pe_simple (ocean) driver on the acc1 channel.

This is the EXPERIMENT file: run control, PE layout, restart settings, model
assembly. The physics lives in acc1_setup.py, the coupler machinery in
coupler_mpi.py / coupler_io.py / coupler_loop.py.

Run:  mpirun -np <n_pes_wave + n_pes_ocean> python run.py
"""
# add here pyOM3 code location, here it's $HOME/src/pyOM3/
import os,sys; sys.path.append(os.path.expanduser('~')+'/src/pyOM3/')
import numpy as np
import pyOM3 as OM

from coupler_mpi  import split_world
from coupler_io   import CouplingWriter
from coupler_loop import run_wave, run_ocean
from acc1_setup   import (wave, ocean, get_wave_parameter, get_ocean_parameter,
                          nx, ny, dx, dy, wave_dt, ocean_dt, use_stress_work,
                          wave_snap_file, ocean_snap_file)

# ---------------------------------------------------------------------------
# run control
# ---------------------------------------------------------------------------
n_pes_x_wave,  n_pes_y_wave  = 2, 2
n_pes_x_ocean, n_pes_y_ocean = 1, 2
halo = 1

spinup_time = 0 * 86400.0      # wave-only spinup [s]
# total experiment length and per-job segment, both in model days; settable
# from the batch script for chain jobs (segment 0 = run to completion)
runlen       = float(os.environ.get('ACC1_RUNLEN_DAYS',  '365')) * 86400.0
segment_days = float(os.environ.get('ACC1_SEGMENT_DAYS', '60'))

AB_eps = 0.1                   # AB2 stabilisation for the WAVE propagation

# restart (coupler-level, netCDF, global fields)
restart_base     = 'acc1_restart'
restart_read     = True
restart_interval = 30 * 86400.0

# ---------------------------------------------------------------------------
# communicators
# ---------------------------------------------------------------------------
n_pes_wave  = n_pes_x_wave  * n_pes_y_wave
n_pes_ocean = n_pes_x_ocean * n_pes_y_ocean
world, my_pe, color, comm, intercomm, active = split_world(n_pes_wave,
                                                           n_pes_ocean)

# ---------------------------------------------------------------------------
# fail loudly, everywhere: any uncaught exception on ANY rank -- in the model
# build, the restart read, or the run loop -- prints its traceback and aborts
# the WHOLE job. Without this, the surviving ranks wait forever at the next
# barrier/recv and the allocation burns to the walltime limit (observed three
# times: restart-read MPI_ERR_TRUNCATE, ocean-NaN canary, startup NameError).
# ---------------------------------------------------------------------------
def _abort_on_exception(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stdout.flush(); sys.stderr.flush()
    try:
        world.Abort(1)
    except Exception:
        os._exit(1)
sys.excepthook = _abort_on_exception
if world.Get_size() < n_pes_wave + n_pes_ocean:
    if my_pe == 0:
        print('ERROR: need at least %d ranks (%d wave + %d ocean), got %d'
              % (n_pes_wave + n_pes_ocean, n_pes_wave, n_pes_ocean,
                 world.Get_size()))
    sys.exit(1)
n_idle = world.Get_size() - (n_pes_wave + n_pes_ocean)
if my_pe == 0 and n_idle > 0:
    print('%d wave + %d ocean PEs, %d idle ranks'
          % (n_pes_wave, n_pes_ocean, n_idle), flush=True)

# restart bookkeeping (both components decide identically: shared filesystem)
restart_wave_file  = restart_base + '_wave.nc'
restart_ocean_file = restart_base + '_ocean.nc'
do_restart = bool(restart_read and os.path.exists(restart_wave_file)
                  and os.path.exists(restart_ocean_file))
do_restart = world.bcast(do_restart, root=0)
rsteps  = int(restart_interval / ocean_dt) if restart_interval > 0 else 0
# segment tag = coupling step of the restart being resumed: monotonic,
# chronologically sortable, collision-free (the old mtime%100000 tag wrapped
# daily and could collide/reorder).
def _restart_step():
    from netCDF4 import Dataset
    f = Dataset(restart_wave_file); n = int(f.getncattr('n_cpl')); f.close()
    return n
if do_restart:
    seg_n = world.bcast(_restart_step() if my_pe == 0 else None, root=0)
else:
    seg_n = 0                     # from initial conditions: segment 0
segment = '_seg%07d' % seg_n

# ---------------------------------------------------------------------------
# build models
# ---------------------------------------------------------------------------
if color == 0:                                   # ---- wave ----
    M = wave(get_wave_parameter(), n_pes_x=n_pes_x_wave, n_pes_y=n_pes_y_wave,
             comm_in=comm, halo_points=halo)
    M.setup(snap_file_name=wave_snap_file.replace('.cdf', segment + '.cdf'))
    M.enable_reflection         = True   # specular reflection at the walls
    M.energy_consistent         = True   # conservative frequency-flux fixer
    M.implicit_dissipation_only = True   # dissipative-only implicit denominator
    M.tail_flux = 'input'
    #M.f_limiter = 'upwind'
    M.limit_fdot = True
    M.refraction_cfl = 0.1
    M.set_initial()

    if M.my_pe == 0:
        print('CONFIG wave : tail_flux=%s | f_limiter=%s | energy_consistent=%s | '
              'reflection=%s | implicit_diss_only=%s | dt=%g s | nf=%s nd=%s'
              % (getattr(M, 'tail_flux', 'wamdi'),
                 getattr(M, 'f_limiter', 'superbee'),
                 getattr(M, 'energy_consistent', False),
                 getattr(M, 'enable_reflection', False),
                 getattr(M, 'implicit_dissipation_only', False),
                 M.dt, getattr(M, 'nf', '?'), getattr(M, 'nd', '?')), flush=True)
    
    
elif color == 1:                                 # ---- ocean ----
    M = ocean(get_ocean_parameter(),
              n_pes_x=n_pes_x_ocean, n_pes_y=n_pes_y_ocean, comm_in=comm,
              halo_points=halo)
    M.use_stress_work = use_stress_work
    M.set_coriolis()
    M.set_topography()
    M.make_topo_masks()
    u0, v0, t0, s0, ps0 = M.make_initial_fields()
    M.set_initial_conditions(u0, v0, t0, s0, ps0)
    # WP3: TKE closure, acc-family calibration; surface TKE flux = Phi_oc
    # from the wave model (delivered by the coupler each coupling step)
    M.alpha_tke = 1.0
    M.mxl_min   = 1e-8
    M.tke_surf_from_waves = True
    M.init_tke()
    M.localise_forcing()
    M.prepare_for_run(snap_file_name=ocean_snap_file.replace('.cdf', segment + '.cdf'))
    # coupling fields on the local (haloed) grid
    M.u_S = OM.np.zeros((M.Nz, M.Ny, M.Nx), np.float64)
    M.v_S = OM.np.zeros((M.Nz, M.Ny, M.Nx), np.float64)
    if use_stress_work:
        M.tau_work_x = OM.np.zeros((M.Nz, M.Ny, M.Nx), np.float64)
        M.tau_work_y = OM.np.zeros((M.Nz, M.Ny, M.Nx), np.float64)
    else:
        M.taux = OM.np.zeros((M.Ny, M.Nx), np.float64)
        M.tauy = OM.np.zeros((M.Ny, M.Nx), np.float64)
    # wave->ocean dissipation energy flux (surface TKE flux for the closure)
    M.phi_oc = OM.np.zeros((M.Ny, M.Nx), np.float64)

world.Barrier()

# ---------------------------------------------------------------------------
# dt handshake: coupling once per ocean step, wave sub-cycles
# ---------------------------------------------------------------------------
if color == 0:
    if comm.Get_rank() == 0:
        intercomm.send(M.dt, dest=0, tag=1)
        ocean_dt_r = intercomm.recv(source=0, tag=2)
        ocean_dz_r = intercomm.recv(source=0, tag=3)
        ocean_nz_r = intercomm.recv(source=0, tag=4)
        if ocean_dt_r < M.dt or abs(round(ocean_dt_r / M.dt) * M.dt - ocean_dt_r) > 1e-9:
            raise RuntimeError('ocean dt (%g s) must be a positive integer '
                               'multiple of wave dt (%g s)' % (ocean_dt_r, M.dt))
        sub_cycle = int(round(ocean_dt_r / M.dt))
    else:
        ocean_dt_r = ocean_dz_r = ocean_nz_r = None
        sub_cycle = 0
    sub_cycle  = comm.bcast(sub_cycle,  root=0)
    ocean_dz_r = comm.bcast(ocean_dz_r, root=0)
    ocean_nz_r = comm.bcast(ocean_nz_r, root=0)
    if comm.Get_rank() == 0:
        print('wave model sub-cycles', sub_cycle, 'times per ocean step')
    cpl_writer = None
    if comm.Get_rank() == 0:
        cpl_writer = CouplingWriter('acc_coupling%s.cdf' % segment, nx, ny,
                                    ocean_nz_r, dx, dy,
                                    stress_is_3d=use_stress_work,
                                    dt_cpl=sub_cycle*M.dt)
elif color == 1:
    if M.my_pe == 0:
        wave_dt_r = intercomm.recv(source=0, tag=1)
        intercomm.send(M.dt, dest=0, tag=2)
        intercomm.send(M.dz, dest=0, tag=3)
        intercomm.send(M.nz, dest=0, tag=4)
        if M.dt < wave_dt_r or abs(round(M.dt / wave_dt_r) * wave_dt_r - M.dt) > 1e-9:
            raise RuntimeError('ocean dt (%g s) must be a positive integer '
                               'multiple of wave dt (%g s)' % (M.dt, wave_dt_r))
        sub_cycle = int(round(M.dt / wave_dt_r))
    else:
        sub_cycle = 0
    sub_cycle = comm.bcast(sub_cycle, root=0)
    ocean_dz_r = ocean_nz_r = None
    cpl_writer = None

world.Barrier()

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
if color < 2:      # idle ranks lack the branch-defined names (sub_cycle etc.)
    cfg = dict(steps=int(runlen / ocean_dt), sub_cycle=sub_cycle, rsteps=rsteps,
               seg_steps=int(segment_days * 86400.0 / ocean_dt),
               do_restart=do_restart, restart_wave_file=restart_wave_file,
               restart_ocean_file=restart_ocean_file, spinup_time=spinup_time,
               use_stress_work=use_stress_work, ocean_dz_r=ocean_dz_r,
               ocean_nz_r=ocean_nz_r, ny=ny, nx=nx, AB_eps=AB_eps,
               cpl_writer=cpl_writer)




try:
    if color == 0:
        n_end = run_wave(M, cfg, comm, intercomm, active)
    elif color == 1:
        n_end = run_ocean(M, cfg, comm, intercomm, active)
    else:
        n_end = None
except Exception:
    import traceback; traceback.print_exc()
    world.Abort(1)          # kill ALL ranks: no barrier deadlock, no wasted walltime

world.Barrier()
if my_pe == 0:
    if n_end >= cfg['steps']:
        open('RUN_DONE', 'w').write('completed %d coupling steps\n' % n_end)
        print('all done')
    else:
        print('segment done at coupling step %d / %d' % (n_end, cfg['steps']))
