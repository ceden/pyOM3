#!/bin/tcsh
#SBATCH --job-name=ic01_part
#SBATCH --account=uo0780_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --constraint=a100_80
#SBATCH --time=04:00:00
#SBATCH --output=log_part_%j.out
#SBATCH --error=log_part_%j.out

# icon_0.1deg with LAGRANGIAN PARTICLES in the North Atlantic, for the
# Lagrangian frequency analysis of draft1 against the band method on the
# window 40-25 W, 35-45 N. run_gpu.sh in everything that is not about the
# particles (the same physics, forcing, offsets and shard layout), as a
# BRANCH from the end of year 9 of icon01_spinup.
#
#   record   80 inertial periods at 40 N (f = 9.37e-5 1/s, 18.6 h): 62 days,
#            a record every 6 ocean steps (864 s; 6 divides the snapshot
#            interval in steps, so the snapshots are record steps)
#   snapshot day 0 and day 31 (the middle of the record, for the band method)
#   seeds    every T point of 45-20 W, 30-50 N at 50, 200, 500, 1000, 2000 m
#            (the window and a margin for 31 days of drift), about 330 000
#            moving particles; fixed points every second T point of the
#            window at the same depths, about 24 000 (the Eulerian series)
#   output   icon01_particles_seg0000000.nc (days 0-31) and
#            icon01_particles_seg0018600.nc (days 31-62), ~90 GB together
#
# Two segments of 31 days, the second through the run's own restart
# (icon01_restart_ocean.nc + icon01_restart_ocean_particles.npz).

set SCRIPT = run_particles_gpu.sh
# the spinup run directory: its input files, its ic_blocks, its year-9 restart
set SPINUP = ../icon01_spinup
set INIT   = $SPINUP/icon01_restart_ocean_y9.nc

set DT    = 144
set FRESH = 1
set QNET_SPINUP = 14.40 ; set EMP_SPINUP = "-0.3050"    # as run_gpu.sh: the spinup's numbers

setenv HDF5_USE_FILE_LOCKING FALSE
setenv TF_CPP_MIN_LOG_LEVEL 2
module purge
conda activate jax_sharding1
setenv JAX_COMPILATION_CACHE_DIR /work/uo0780/u241155/jaxcache
setenv JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0.5

setenv PYOM3_ROOT               /work/uo0780/u241155/pyOM3_current/
setenv PYOM3_BACKEND            jax
setenv XLA_PYTHON_CLIENT_MEM_FRACTION 0.95

# ---- the shard layout: the spinup's (the raw restart belongs to it) --------
if ($SLURM_NNODES != 12) then
  echo "run on 12 nodes (16 x 3 shards), as the spinup whose raw restart this branches from"
  exit 1
endif
set NPX = 16 ; set NPY = 3
if (! -e icon01_restart_ocean.nc) then
  # the first segment branches: the layout of the year-9 restart must be this one
  set LAY = `python3 -c "from netCDF4 import Dataset; f=Dataset('$INIT'); print(f.n_pes_x, f.n_pes_y)"`
  if ("$LAY" != "$NPX $NPY") then
    echo "$INIT was written on $LAY shards, this run has $NPX $NPY: change the node count"
    exit 1
  endif
endif
setenv EXP_NPX_OCEAN $NPX
setenv EXP_NPY_OCEAN $NPY

# ---- the configuration of run_gpu.sh ----------------------------------------
setenv EXP_SETUP                  icon01_setup
setenv EXP_INPUT                  $SPINUP
setenv EXP_FORCING_DIR            /work/uo0780/u241155/pyOM3_real/era5_icon033
setenv EXP_FORCING_TAG            icon033
setenv EXP_FRESHWATER             $FRESH
setenv EXP_SW_BOTTOM              1
setenv EXP_RESTART_RAW            1               # per shard
setenv EXP_RESTART_DAYS           10
setenv EXP_SSS_PISTON             0.17
setenv EXP_AVERAGE                month
setenv EXP_KE_BUDGET              0
setenv EXP_OCEAN_DT               $DT
setenv EXP_VADV_SUB_CYCLING       2
setenv EXP_OCEAN_AB               3
setenv EXP_FREE_SURFACE           explicit
setenv EXP_SUB_CYCLING            48
setenv EXP_FB_THETA               0.5
setenv EXP_PROFILE                2
setenv EXP_PROFILE_EVERY          200
setenv EXP_WAVES                  0
setenv EXP_FORCING_START          1990-05-01
setenv EXP_FORCING_CYCLE_DAYS     365
setenv EXP_QNET_OFFSET            $QNET_SPINUP
setenv EXP_EMP_OFFSET             "$EMP_SPINUP"
setenv EXP_STRESS_MODE            relwind
setenv EXP_STRESS_UO              lagrangian
setenv EXP_FORCING_WIND_TAG       icon033w
setenv EXP_CANARY_EVERY           100
setenv EXP_CANARY_CFL             1

# ---- the branch and the record ----------------------------------------------
# the year-9 state (3285 d, whole repeat years: the forcing continues without
# a jump); read only while this run has no restart of its own
setenv EXP_INIT_OCEAN             $INIT
setenv EXP_SNAPINT_OCEAN_DAYS     31
set RUNLEN = 62 ; set SEGMENT = 31

# ---- the particles (coupled_common/particle_diag.py) ------------------------
setenv EXP_PARTICLES              1
setenv EXP_PARTICLE_BOX           -45,-20,30,50
setenv EXP_PARTICLE_STRIDE        1
setenv EXP_PARTICLE_DEPTHS        50,200,500,1000,2000
setenv EXP_MOORING_BOX            -40,-25,35,45
setenv EXP_MOORING_STRIDE         2
setenv EXP_MOORING_DEPTHS         50,200,500,1000,2000
setenv EXP_PARTICLE_OUT_STEPS     6

# ---- the target of this job --------------------------------------------------
set RESTART = icon01_restart_ocean
if (-e ${RESTART}.nc) then
  set DAY = `python3 -c "from netCDF4 import Dataset; f=Dataset('${RESTART}.nc'); print(int(round(f.n_cpl*$DT/86400.)))"`
else
  set DAY = 0
endif
if ($DAY >= $RUNLEN) then
  echo "icon01 particles: model day $DAY has reached RUNLEN $RUNLEN, nothing to do"
  exit 0
endif
setenv EXP_RUNLEN_DAYS   $RUNLEN
setenv EXP_SEGMENT_DAYS  $SEGMENT
echo "icon01 particles GPU: day $DAY of $RUNLEN in segments of $SEGMENT, $SLURM_NNODES nodes, shards $NPX x $NPY"

# the initial conditions per shard (the state is then replaced by the branch,
# but run.py reads them first): the spinup's blocks, cut there if missing
set ICB = ic_blocks_${NPX}x${NPY}
if (! -e $SPINUP/$ICB/salt_0000.npy) then
  echo "cutting $SPINUP/initial_conditions.nc into $NPX x $NPY shard blocks ($SPINUP/$ICB/)"
  python3 ../setups/forcing_common/ic_blocks.py $SPINUP/initial_conditions.nc $NPX $NPY --out $SPINUP/$ICB
  if ($status != 0) exit 1
endif
setenv EXP_IC_BLOCKS $ICB

setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK
unsetenv SLURM_CPUS_PER_TASK

# ---- start watchdog (run_gpu.sh) --------------------------------------------
set WAIT = 3600
if (! $?WD_TRIES) set WD_TRIES = 0
set WD_TRIES = $WD_TRIES
set LOG = log_part_${SLURM_JOB_ID}.out
rm -f WATCHDOG_FIRED
sh -c "sleep $WAIT; grep -q 'ocean step' $LOG || { touch WATCHDOG_FIRED; scancel --signal=KILL ${SLURM_JOB_ID}.0; }" &
set WD = $!

srun -l --ntasks=$SLURM_NTASKS python3 -u ../setups/coupled_common/run.py
set st = $status
/bin/kill $WD >& /dev/null

if (-e WATCHDOG_FIRED) then
  rm -f WATCHDOG_FIRED
  @ WD_TRIES++
  if ($WD_TRIES > 3) then
    echo "no time step within $WAIT s, $WD_TRIES times in a row -- chain stopped"
    exit 1
  endif
  echo "no time step within $WAIT s: step killed, resubmitting (attempt $WD_TRIES of 3)"
  sbatch --nodes=$SLURM_NNODES --export=ALL,WD_TRIES=$WD_TRIES $SCRIPT
  exit 0
endif

if ($st != 0) then
  echo "segment failed with status $st, chain stopped"
  exit $st
endif

if (! -e RUN_DONE) then
  sbatch --nodes=$SLURM_NNODES --export=ALL,WD_TRIES=0 $SCRIPT
  exit 0
endif
echo "run complete: icon01_particles_seg*.nc, snapshots in raw_ocean_icon01_seg*/"
