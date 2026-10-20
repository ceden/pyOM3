"""
Coupled WIM (surface waves, wim_realistic) <-> pe_realistic ocean, forced by
hourly ERA5. With EXP_WAVES=0 the same ocean runs alone. With
EXP_NPX_ATMOS x EXP_NPY_ATMOS > 0 a THIRD component, the Held-Suarez
atmosphere of exp_atmos.py, replaces the forcing files (energy_cycle.tex
Sec. 6): the experiment module then defines `atmosphere`, get_atmos_parameter,
atmos_coarsen, nx_atm, ny_atm, atmos_dt, atmos_snap_file, atmos_initial and a
'coupled' forcing source.

This file is CONFIGURATION INDEPENDENT: run control, PE layout, restart
settings, model assembly. Which experiment it runs is decided by EXP_SETUP,
the python module holding the grid, the input files, the forcing and the two
model classes (icon15_setup.py, global4_setup.py, ...). Every output file is
named after that module's exp_name. The coupler machinery is in
coupler_mpi.py / coupler_io.py / coupler_loop.py.

Run:  EXP_SETUP=icon15_setup mpirun -np <n_pes_wave + n_pes_ocean> python run.py

Environment:
  PYOM3            path to the pyOM3 tree      (default $HOME/src/pyOM3)
  EXP_SETUP        experiment module, e.g. icon15_setup   (NO default)
  EXP_WAVES        1 coupled to the wave model (default), 0 ocean alone
  EXP_INPUT        directory of the input files of the experiment module
  EXP_GRID         grid file in it              (default grid_topo.nc)
  EXP_IC           initial conditions in it     (default initial_conditions.nc)
  EXP_FORCING_DIR         directory of era5_<tag>_YYYY-MM.nc (prep_era5.py)
  EXP_FORCING_TAG         tag of those files    (default icon15)
  EXP_FORCING_START       date of model time 0, YYYY-MM-DD (00 UTC)
  EXP_FORCING_CYCLE_DAYS  365 repeat year, 0 through the calendar
  EXP_INIT_CHECK   0 = only warn if EXP_INIT_OCEAN is not on a whole repeat
                   year (default 1 = stop)
  EXP_INIT_OCEAN   ocean restart file of ANOTHER run to start from (a
                   branch). Used only while this run has no restart of its
                   own. The step counter starts at 0, model time 0 is
                   EXP_FORCING_START. The waves start cold.
  EXP_SNAPINT_OCEAN_DAYS / EXP_SNAPINT_WAVE_DAYS  snapshot intervals
  EXP_RESTART_DAYS  restart write interval in model days (default 10; the end of a segment always writes)
  EXP_AVERAGE      online ocean means: 0 off (default), month, day, <N>d, <N>h
  EXP_AVERAGE_FIELDS   comma list restricting the online means (default all)
  EXP_OCEAN_AB     2 = AB2 with eps 0.1 (1.6,-0.6,0), default; 3 = AB3
  EXP_MXL_CHOICE   mixing length: 1 pyOM2, 2 NEMO nn_mxl 3
  EXP_MXL_BETA     Charnock coefficient of choice 2
  EXP_HS_FACTOR    coupled + choice 2: z0 = HS_FACTOR * Hs sets the surface
                   mixing length (1.6 = NEMO ln_mxhsw; 0 = Charnock)
  EXP_ALPHA_CB     surface TKE flux alpha_CB |tau/rho_0|^{3/2} of the ocean
                   alone (1 = pyOM2); unused with waves
  EXP_ALPHA_TKE    TKE diffusivity factor alpha_tke
  EXP_MXL_SURF_FACTOR  mixing length at the surface W point in units of the
                   top cell thickness (choice 1 only; 1 = pyOM2 behaviour)
  EXP_DIRECT_BRANCH  1 hands tau_a - tau_in to the ocean as well (default 0)
  EXP_CPL_WRITE_EVERY  coupling-file record every n coupling steps (6)
  EXP_RUNLEN_DAYS   total experiment length in model days (default 365)
  EXP_SEGMENT_DAYS  length of THIS job's segment; 0 = run to completion
  EXP_SPINUP_DAYS   wave-only spinup before coupling
  EXP_OCEAN_DT / EXP_WAVE_DT  time steps in seconds
  EXP_WAVE_COARSEN  c = 1 (default) waves on the ocean grid; c = 2, 3, 4 the
                   wave grid is the ocean grid coarsened by c in x and y
                   (experiment modules that support it; see coupler_mpi.py).
                   The wave layout EXP_NPX_WAVE x EXP_NPY_WAVE then divides
                   the COARSE grid.
  EXP_NPX_ATMOS / EXP_NPY_ATMOS  atmosphere layout (default 0 0 = no atmosphere)
  EXP_SNONLIN      Snonlin backend: fortran (default) | numpy | jax
  EXP_PROFILE       0 off, 1 section timers (default), 2 + model internals
  EXP_PROFILE_EVERY report cadence in COUPLING steps, not wave
                    sub-steps (default 20; one early report at step 5)
  EXP_NF / EXP_ND   spectral resolution
  EXP_DF0           geometric frequency ratio df/f
  EXP_FMAX          top of the frequency grid in Hz; f0 is solved from it.
                    Set 0 to give EXP_F0 explicitly instead.
The defaults of everything after EXP_WAVES are the experiment module's, not
this file's. See the spectral-grid section of README.md before changing nf
or df0.
"""
import os, sys
# LAYOUT. This file lives in <root>/coupled_common, next to <root>/pyOM3 and
# <root>/forcing_common; an experiment directory holds only its setup module,
# its batch scripts and its input files and runs
#     python3 ../coupled_common/run.py
# The experiment module is found in the CURRENT directory, which comes LAST:
# a stale copy of a coupler or forcing file left in an experiment directory
# must not shadow the shared one.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path[:0] = [_here, os.path.join(_root, 'forcing_common'),
                os.environ.get('PYOM3_ROOT', _root), os.getcwd()]
import numpy as np
import pyOM3 as OM

from coupler_mpi  import split_world, split_world3, CoarseMap
from coupler_io   import CouplingWriter, AtmosCouplingWriter
from coupler_loop import run_wave, run_ocean, run_ocean_only, run_atmos

# ---------------------------------------------------------------------------
# THE EXPERIMENT MODULE. Everything configuration specific lives there and
# nothing below names a configuration: the grid, the input files, the forcing,
# the two model classes and every physical parameter come from it, and the
# output files are named after its exp_name.
# ---------------------------------------------------------------------------
import importlib
setup_name = os.environ.get('EXP_SETUP', '').strip()
if not setup_name:
    raise SystemExit('EXP_SETUP is not set. Give the name of the experiment '
                     'module in this directory, for example '
                     'EXP_SETUP=icon15_setup')
setup = importlib.import_module(setup_name[:-3] if setup_name.endswith('.py')
                                else setup_name)
wave,  ocean = setup.wave, setup.ocean
get_wave_parameter, get_ocean_parameter = (setup.get_wave_parameter,
                                           setup.get_ocean_parameter)
nx, ny, nz       = setup.nx, setup.ny, setup.nz
wave_dt, ocean_dt = setup.wave_dt, setup.ocean_dt
use_stress_work   = setup.use_stress_work
wave_snap_file, ocean_snap_file = setup.wave_snap_file, setup.ocean_snap_file
spectral_grid_report, average_mode = setup.spectral_grid_report, setup.average_mode
exp_name = getattr(setup, 'exp_name', setup_name)
# a coarser wave grid (EXP_WAVE_COARSEN): the experiment module owns the
# switch, builds the coarse wave grid and names its size. Modules without it
# run the waves on the ocean grid, as before.
wave_coarsen     = int(getattr(setup, 'wave_coarsen', 1))
nx_wave, ny_wave = getattr(setup, 'nx_wave', nx), getattr(setup, 'ny_wave', ny)
cmap = CoarseMap(ny, nx, wave_coarsen) if wave_coarsen > 1 else None
if cmap is not None and (cmap.ny_c, cmap.nx_c) != (ny_wave, nx_wave):
    raise SystemExit('%s: wave grid %d x %d is not the ocean grid %d x %d '
                     'coarsened by %d (%d x %d)' % (setup_name, nx_wave, ny_wave,
                     nx, ny, wave_coarsen, cmap.nx_c, cmap.ny_c))
# the ATMOSPHERE (optional third component): its grid is the ocean grid
# coarsened by atmos_coarsen, a multiple of wave_coarsen so that the wave
# grid nests in it as well (the ocean forwards the stress to the waves)
n_pes_x_atmos, n_pes_y_atmos = int(os.environ.get('EXP_NPX_ATMOS', '0')), \
                               int(os.environ.get('EXP_NPY_ATMOS', '0'))
n_pes_atmos = n_pes_x_atmos*n_pes_y_atmos
atmos = n_pes_atmos > 0
cmap_a = atmos_dt = atmos_sub = None
if atmos:
    atmos_coarsen = int(getattr(setup, 'atmos_coarsen', 1))
    nx_atm, ny_atm = getattr(setup, 'nx_atm', nx), getattr(setup, 'ny_atm', ny)
    cmap_a = CoarseMap(ny, nx, atmos_coarsen) if atmos_coarsen > 1 else None
    if cmap_a is not None and (cmap_a.ny_c, cmap_a.nx_c) != (ny_atm, nx_atm):
        raise SystemExit('%s: atmosphere grid %d x %d is not the ocean grid coarsened by %d'
                         % (setup_name, nx_atm, ny_atm, atmos_coarsen))
    if atmos_coarsen % max(wave_coarsen, 1) != 0:
        raise SystemExit('atmos_coarsen %d is not a multiple of wave_coarsen %d'
                         % (atmos_coarsen, wave_coarsen))
    atmos_dt = float(setup.atmos_dt)
    if abs(round(ocean_dt/atmos_dt)*atmos_dt - ocean_dt) > 1e-9:
        raise SystemExit('atmosphere dt %g does not divide the ocean dt %g' % (atmos_dt, ocean_dt))
    atmos_sub = int(round(ocean_dt/atmos_dt))

# ---------------------------------------------------------------------------
# run control
# ---------------------------------------------------------------------------
waves = bool(int(os.environ.get('EXP_WAVES', '1')))
n_pes_x_wave,  n_pes_y_wave  = int(os.environ.get('EXP_NPX_WAVE',  '2')), \
                               int(os.environ.get('EXP_NPY_WAVE',  '2'))
n_pes_x_ocean, n_pes_y_ocean = int(os.environ.get('EXP_NPX_OCEAN', '1')), \
                               int(os.environ.get('EXP_NPY_OCEAN', '1'))
halo = 1                      # wim_simple/wim_realistic require halo_points=1

spinup_time = float(os.environ.get('EXP_SPINUP_DAYS', '0')) * 86400.0
runlen       = float(os.environ.get('EXP_RUNLEN_DAYS',  '365')) * 86400.0
segment_days = float(os.environ.get('EXP_SEGMENT_DAYS', '30'))

# ocean Adams-Bashforth weights. EXP_OCEAN_AB = 2 (default): the fixed
# (1.6,-0.6,0), i.e. pyOM2's AB2 with AB_eps = 0.1, from the first step on, as
# always. EXP_OCEAN_AB = 3: the third-order (23/12,-16/12,5/12); a cold start
# is ramped (forward Euler, AB2, then AB3) in coupler_loop.py. With superbee
# INSIDE the AB step (VADV_SUB_CYCLING = 0) AB3 lowers the tracer CFL limit
# from 0.46 to 0.28: use it together with the split tracer advection.
_ab = os.environ.get('EXP_OCEAN_AB', '2').strip()
if _ab not in ('2', '3'):
    raise ValueError('EXP_OCEAN_AB = %s: use 2 or 3' % _ab)
ocean_ab = (1.6, -0.6, 0.0) if _ab == '2' else (23./12., -16./12., 5./12.)

# profiling (coupler_prof.py): 0 off, 1 section timers (free), 2 additionally
# wraps the models' own methods on the instance -- nothing is added to the
# pyOM3 package. Both components print a table every profile_every steps.
profile       = int(os.environ.get('EXP_PROFILE', '2'))
profile_every = int(os.environ.get('EXP_PROFILE_EVERY', '6'))

# coupling-file cadence: every n-th coupling step gets a record (coupling is
# hourly, so 6 = 6-hourly). Gates the write and the writer-only gathers only.
cpl_write_every = int(os.environ.get('EXP_CPL_WRITE_EVERY', '6'))

# restart (coupler-level, netCDF, global fields)
restart_base     = exp_name + '_restart'
restart_read     = True
restart_interval = float(os.environ.get('EXP_RESTART_DAYS', '10')) * 86400.0   # 0 = end of segment only
# branch start: the ocean state of another run (its final restart)
init_ocean_file  = os.environ.get('EXP_INIT_OCEAN', '').strip()

# ---------------------------------------------------------------------------
# communicators
# ---------------------------------------------------------------------------
n_pes_wave  = n_pes_x_wave  * n_pes_y_wave if waves else 0
n_pes_ocean = n_pes_x_ocean * n_pes_y_ocean
if atmos:
    # colours 0 wave, 1 ocean, 2 ATMOSPHERE, 3 idle
    world, my_pe, color, comm, intercomm, active, intercomm_ao = split_world3(
        n_pes_wave, n_pes_ocean, n_pes_atmos)
else:
    world, my_pe, color, comm, intercomm, active = split_world(n_pes_wave, n_pes_ocean)
    intercomm_ao = None
IDLE = 3 if atmos else 2

# ---------------------------------------------------------------------------
# fail loudly, everywhere: any uncaught exception on ANY rank -- in the model
# build, the restart read, or the run loop -- prints its traceback and aborts
# the WHOLE job. Without this the surviving ranks wait forever at the next
# barrier/recv and the allocation burns to the walltime limit.
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

if world.Get_size() < n_pes_wave + n_pes_ocean + n_pes_atmos:
    if my_pe == 0:
        print('ERROR: need at least %d ranks (%d wave + %d ocean + %d atmosphere), got %d'
              % (n_pes_wave + n_pes_ocean + n_pes_atmos, n_pes_wave, n_pes_ocean, n_pes_atmos,
                 world.Get_size()))
    sys.exit(1)
n_idle = world.Get_size() - (n_pes_wave + n_pes_ocean + n_pes_atmos)
if my_pe == 0:
    print('CONFIG ocean time stepping: Adams-Bashforth weights (%.4f, %.4f, %.4f)%s'
          % (ocean_ab + ((' = AB3, a cold start is ramped' if ocean_ab[2] else
                          ' = AB2 with eps 0.1') + ' (EXP_OCEAN_AB)',)), flush=True)
if my_pe == 0 and n_idle > 0:
    print('%d wave + %d ocean PEs, %d idle ranks'
          % (n_pes_wave, n_pes_ocean, n_idle), flush=True)

# restart bookkeeping (both components decide identically: shared filesystem)
restart_wave_file  = restart_base + '_wave.nc'
restart_ocean_file = restart_base + '_ocean.nc'
restart_atmos_file = restart_base + '_atmos.nc'
do_restart = bool(restart_read and os.path.exists(restart_ocean_file)
                  and (os.path.exists(restart_wave_file) or not waves)
                  and (os.path.exists(restart_atmos_file) or not atmos))
do_restart = world.bcast(do_restart, root=0)
# a branch reads the other run's ocean state only while it has none of its own
do_init = world.bcast(bool(init_ocean_file) and not do_restart, root=0)
rsteps = int(restart_interval / ocean_dt) if restart_interval > 0 else 0
# segment tag = the coupling step of the restart being resumed: monotonic,
# chronologically sortable, collision-free.
def _restart_step(fname):
    from netCDF4 import Dataset
    f = Dataset(fname); n = int(f.getncattr('n_cpl')); f.close()
    return n
if do_restart:
    seg_n = world.bcast(_restart_step(restart_ocean_file) if my_pe == 0 else None,
                        root=0)
else:
    seg_n = 0                     # from initial conditions: segment 0
segment = '_seg%07d' % seg_n
if do_init:
    # the branch continues the forcing of the run it starts from only if that
    # run ended on a whole number of repeat years (365 d)
    n_init = world.bcast(_restart_step(init_ocean_file) if my_pe == 0 else None,
                         root=0)
    if my_pe == 0:
        print('branch start: ocean state from %s (its step %d = %.2f d = %.4f '
              'years of 365 d); this run starts at step 0 = %s'
              % (init_ocean_file, n_init, n_init*ocean_dt/86400.0,
                 n_init*ocean_dt/(365*86400.0),
                 setup.FORCING_START), flush=True)
    # EXP_INIT_CHECK=0 turns the stop below into a warning: for a short TEST
    # branch from a spinup that is still running, where the jump in the
    # forcing at the branch point does not matter.
    if (n_init*ocean_dt) % (365*86400.0) != 0.0 and \
            os.environ.get('EXP_INIT_CHECK', '1') in ('0', 'no'):
        if my_pe == 0:
            print('WARNING: EXP_INIT_OCEAN ends at %.2f d, not on a whole repeat '
                  'year; the forcing jumps at the branch point (EXP_INIT_CHECK=0)'
                  % (n_init*ocean_dt/86400.0), flush=True)
    elif (n_init*ocean_dt) % (365*86400.0) != 0.0:
        raise RuntimeError('EXP_INIT_OCEAN %s ends at %.2f d, not on a whole '
                           'repeat year: the forcing would jump at the branch '
                           'point' % (init_ocean_file, n_init*ocean_dt/86400.0))

# ---------------------------------------------------------------------------
# build models
# ---------------------------------------------------------------------------
if color == 0:                                   # ---- wave ----
    M = wave(get_wave_parameter(), n_pes_x=n_pes_x_wave, n_pes_y=n_pes_y_wave,
             comm_in=comm, halo_points=halo)
    M.setup(snap_file_name=wave_snap_file.replace('.cdf', segment + '.cdf'))
    # RADIATING COASTS, not reflection: with a real coastline the flux-masked
    # wall piles incident energy up in the last wet cell (a 2dx sawtooth), and
    # specular reflection sends swell back out into the basin. open_coasts
    # lets the action leave through coastal faces by first-order upwind flux
    # and books the export in coastal_loss().
    M.enable_reflection         = False
    M.open_coasts               = True
    M.energy_consistent         = True   # conservative frequency-flux fixer
    M.implicit_dissipation_only = True   # dissipative-only implicit denominator
    M.tail_flux = 'input'
    M.limit_fdot = True
    M.refraction_cfl = 0.1
    M.set_initial()

    if M.my_pe == 0:
        print(spectral_grid_report(), flush=True)
        print('CONFIG wave : tail_flux=%s | energy_consistent=%s | '
              'reflection=%s | open_coasts=%s | implicit_diss_only=%s | '
              'dt=%g s | nf=%s nd=%s | grid %d x %d (ocean grid coarsened by %d)'
              % (getattr(M, 'tail_flux', 'wamdi'),
                 getattr(M, 'energy_consistent', False),
                 getattr(M, 'enable_reflection', False),
                 getattr(M, 'open_coasts', False),
                 getattr(M, 'implicit_dissipation_only', False),
                 M.dt, M.nf, M.nd, nx_wave, ny_wave, wave_coarsen), flush=True)
        if hasattr(setup, 'wave_grid_report'):
            print(setup.wave_grid_report(), flush=True)

elif color == 1:                                 # ---- ocean ----
    M = ocean(get_ocean_parameter(),
              n_pes_x=n_pes_x_ocean, n_pes_y=n_pes_y_ocean, comm_in=comm,
              halo_points=halo)
    M.use_stress_work = use_stress_work
    M.coupled = waves
    M.hs_factor = getattr(setup, 'HS_FACTOR', 1.6)
    M.energy_file = '%s_energy%s.txt' % (exp_name, segment)
    M.set_coriolis()
    M.set_topography()
    M.make_topo_masks()
    u0, v0, t0, s0, ps0 = M.make_initial_fields()
    M.set_initial_conditions(u0, v0, t0, s0, ps0)
    # surface TKE flux = Phi_oc from the wave model (delivered by the coupler
    # each coupling step); the closure mixes the Eulerian velocity u_L - u_S.
    # Ocean alone: |tau_a|^{3/2}, and u_S is zero so there is nothing to
    # subtract.
    M.tke_surf_from_waves = waves
    if not waves:
        M.mix_eulerian = False
    M.init_tke()
    M.localise_forcing()
    M.prepare_for_run(snap_file_name=ocean_snap_file.replace('.cdf',
                                                             segment + '.cdf'),
                      show_each_time_step=False)
    # online time means (EXP_AVERAGE), one file per segment
    M.average_setup('ocean_mean_%s%s.cdf' % (exp_name, segment), average_mode)
    if M.my_pe == 0:
        print('CONFIG ocean: waves=%s | surface TKE flux %s | alpha_tke=%g | '
              'mxl_choice=%d | surface mixing length %.3g m (%s) | r_bot=%g | '
              'kappaH_min=%g | forcing %s cycle %g d | means %s'
              % (waves, 'phi_oc from the wave model' if waves else
                 'alpha_CB=%g * |tau/rho_0|^1.5' % M.alpha_CB,
                 M.alpha_tke, M.mxl_choice, M.surface_mixing_length(),
                 ('z0=%g*Hs from the waves' % M.hs_factor) if
                 (M.mxl_choice == 2 and waves and M.hs_factor > 0) else
                 'at |tau|=0.1 Pa, NEMO Charnock beta=%g' % M.mxl_beta if M.mxl_choice == 2
                 else '%g x top cell' % M.mxl_surf_factor,
                 getattr(M, 'r_bot', 0.0) if getattr(M, 'enable_bottom_friction', False) else 0.0,
                 getattr(M, 'kappaH_min', 0.0),
                 setup.FORCING_START, setup.FORCING_CYCLE_DAYS,
                 # the MODE the means actually run in: the experiment module
                 # parsed it, from whatever environment name it reads
                 average_mode if average_mode else 'off'), flush=True)
    # coupling fields on the local (haloed) grid. Ocean alone they stay zero
    # (taux/tauy then hold tau_a), so the restart has the same content
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
    # wind->wave input stress (diagnostic; consumed by the optional direct
    # branch in set_forcing)
    M.tau_in_x = OM.np.zeros((M.Ny, M.Nx), np.float64)
    M.tau_in_y = OM.np.zeros((M.Ny, M.Nx), np.float64)
    # significant wave height from the wave model; with EXP_HS_FACTOR > 0
    # and mxl_choice = 2 it sets the surface mixing length (NEMO ln_mxhsw)
    if waves:
        M.hs = OM.np.zeros((M.Ny, M.Nx), np.float64)

elif color == 2 and atmos:                       # ---- ATMOSPHERE ---- (colour 2 is IDLE without one)
    M = setup.atmosphere(setup.get_atmos_parameter(), n_pes_x=n_pes_x_atmos,
                         n_pes_y=n_pes_y_atmos, comm_in=comm, halo_points=halo)
    # free ground: grav/rho_0 = 9.81/1024 kept, the free surface equation runs
    # with alpha_s in the role of g (energy_cycle.tex Sec. 2)
    M.grav, M.rho_0 = setup.alpha_s, 1024.0*setup.alpha_s/9.81
    M.set_coriolis()
    M.set_topography()
    M.energy_file = '%s_atmos_energy%s.txt' % (exp_name, segment)
    M.prepare_for_run(snap_file_name=setup.atmos_snap_file.replace('.cdf', segment + '.cdf'),
                      show_each_time_step=False)
    M.prepare_hs()
    M.set_initial_conditions(*setup.atmos_initial())
    cpl_writer_a = None
    if comm.Get_rank() == 0:
        cpl_writer_a = AtmosCouplingWriter('%s_atmos_coupling%s.cdf' % (exp_name, segment),
                                           nx_atm, ny_atm, M.xt_gl, M.yt_gl, dt_cpl=ocean_dt)
        print('CONFIG atmosphere: grid %d x %d (ocean grid coarsened by %d), %d steps per '
              'ocean step, energy budget in %s' % (nx_atm, ny_atm, atmos_coarsen, atmos_sub,
                                                   M.energy_file), flush=True)

world.Barrier()

sub_cycle = 1                                     # ocean alone: no handshake
ocean_dz_r = ocean_nz_r = None
cpl_writer = None
# ---------------------------------------------------------------------------
# dt handshake: coupling once per ocean step, the wave sub-cycles. The ocean
# also announces its VERTICAL GRID -- an ARRAY of layer thicknesses here, not
# a scalar: the wave model's coupling kernels take a variable dz.
# ---------------------------------------------------------------------------
if color == 0:          # (color 1 below: the handshake only when coupled)
    if comm.Get_rank() == 0:
        intercomm.send(M.dt, dest=0, tag=1)
        ocean_dt_r = intercomm.recv(source=0, tag=2)
        ocean_dz_r = intercomm.recv(source=0, tag=3)
        ocean_nz_r = intercomm.recv(source=0, tag=4)
        if ocean_dt_r < M.dt or abs(round(ocean_dt_r/M.dt)*M.dt - ocean_dt_r) > 1e-9:
            raise RuntimeError('ocean dt (%g s) must be a positive integer '
                               'multiple of wave dt (%g s)' % (ocean_dt_r, M.dt))
        sub_cycle = int(round(ocean_dt_r/M.dt))
    else:
        ocean_dt_r = ocean_dz_r = ocean_nz_r = None
        sub_cycle = 0
    sub_cycle  = comm.bcast(sub_cycle,  root=0)
    ocean_dz_r = comm.bcast(ocean_dz_r, root=0)
    ocean_nz_r = comm.bcast(ocean_nz_r, root=0)
    if comm.Get_rank() == 0:
        print('wave model sub-cycles %d times per ocean step; ocean column '
              '%d levels, %.1f m surface cell, %.0f m total'
              % (sub_cycle, ocean_nz_r, ocean_dz_r[-1], ocean_dz_r.sum()),
              flush=True)
    cpl_writer = None
    if comm.Get_rank() == 0:
        cpl_writer = CouplingWriter('%s_coupling%s.cdf' % (exp_name, segment),
                                    nx_wave, ny_wave, ocean_nz_r,
                                    M.xt_gl, M.yt_gl,
                                    stress_is_3d=use_stress_work,
                                    dt_cpl=sub_cycle*M.dt)
elif color == 1 and waves:
    if M.my_pe == 0:
        wave_dt_r = intercomm.recv(source=0, tag=1)
        intercomm.send(M.dt, dest=0, tag=2)
        intercomm.send(np.asarray(M.dzt)[M.halo:-M.halo], dest=0, tag=3)
        intercomm.send(M.Nz - 2*M.halo, dest=0, tag=4)
        if M.dt < wave_dt_r or abs(round(M.dt/wave_dt_r)*wave_dt_r - M.dt) > 1e-9:
            raise RuntimeError('ocean dt (%g s) must be a positive integer '
                               'multiple of wave dt (%g s)' % (M.dt, wave_dt_r))
        sub_cycle = int(round(M.dt/wave_dt_r))
    else:
        sub_cycle = 0
    sub_cycle = comm.bcast(sub_cycle, root=0)
    ocean_dz_r = ocean_nz_r = None
    cpl_writer = None

world.Barrier()

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
if color < IDLE:   # idle ranks lack the branch-defined names (sub_cycle etc.)
    # the waves take their wind from the ocean when there is an atmosphere or
    # when the forced ocean forms the relative-wind stress (exp_ocean.py)
    wind_hub = bool(atmos) or str(getattr(setup, 'STRESS_MODE', 'era5')) == 'relwind'
    cfg = dict(steps=int(runlen/ocean_dt), sub_cycle=sub_cycle, rsteps=rsteps, wind_hub=wind_hub,
               seg_steps=int(segment_days*86400.0/ocean_dt),
               do_restart=do_restart, restart_wave_file=restart_wave_file,
               init_ocean=init_ocean_file if do_init else '',
               restart_ocean_file=restart_ocean_file, spinup_time=spinup_time,
               use_stress_work=use_stress_work, ocean_dz_r=ocean_dz_r,
               ocean_nz_r=ocean_nz_r, ocean_ab=ocean_ab, cmap=cmap,
               # each component's OWN grid: restart and gathers are sized on it
               ny=ny_wave if color == 0 else (ny_atm if color == 2 else ny),
               nx=nx_wave if color == 0 else (nx_atm if color == 2 else nx),
               cpl_write_every=cpl_write_every,
               cpl_writer=cpl_writer_a if color == 2 else cpl_writer,
               profile=profile, profile_every=profile_every,
               # the atmosphere
               atmos=atmos, intercomm_ao=intercomm_ao, cmap_a=cmap_a, atmos_sub=atmos_sub,
               restart_atmos_file=restart_atmos_file, cpl_dt=ocean_dt)

try:
    if color == 0:
        n_end = run_wave(M, cfg, comm, intercomm, active)
    elif color == 1 and waves:
        n_end = run_ocean(M, cfg, comm, intercomm, active)
    elif color == 1:
        n_end = run_ocean_only(M, cfg, comm, active)
    elif color == 2 and atmos:
        n_end = run_atmos(M, cfg, comm, intercomm_ao, active)
    else:
        n_end = None
except Exception:
    import traceback; traceback.print_exc()
    world.Abort(1)      # kill ALL ranks: no barrier deadlock, no wasted walltime

world.Barrier()
if my_pe == 0:
    if n_end >= cfg['steps']:
        open('RUN_DONE', 'w').write('completed %d coupling steps\n' % n_end)
        print('all done')
    else:
        print('segment done at coupling step %d / %d' % (n_end, cfg['steps']))
