#!/bin/tcsh
#SBATCH --job-name=ic033_gpu
#SBATCH --account=uo0780_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=log_gpu_%j.out
#SBATCH --error=log_gpu_%j.out

# icon_0.33deg, the OCEAN ALONE on the JAX BACKEND, on the GPUs of one or
# more Levante GPU nodes: ONE PROCESS PER GPU under srun (as gpu_timing.sh),
# jax.distributed initialised in pyOM3/__init__, the model's mesh spanning
# the processes (coupled_common/run.py with PYOM3_BACKEND=jax). Same setup
# module, same forcing, same restart chain as run.sh; what differs is below.
#
#   * EXP_NPX_OCEAN x EXP_NPY_OCEAN is the SHARD layout and must equal the
#     number of devices over ALL processes (run.py checks; one device per
#     process). A shard is an exact division of the grid: 1080 = 2^3 3^3 5,
#     689 = 13 x 53, so the shards are 4N x 1 for N nodes, and N must divide
#     1080/4 = 270: N = 1, 2, 3, 5, 6, 9, 10 ... (NOT 4, 7, 8). Set --nodes
#     above; the layout is computed from it below.
#
# TEST = 1 (default here): 0.2 model days in two 0.1 day segments from a cold
# start, so the restart path is used. TEST = 0 runs the production chain of
# run.sh (checkpoints, resubmission) with this script.
#

set SCRIPT = run_gpu.sh
if (! $?TEST) set TEST = 1

setenv HDF5_USE_FILE_LOCKING FALSE
setenv TF_CPP_MIN_LOG_LEVEL 2
module purge
conda activate jax_sharding1
setenv JAX_COMPILATION_CACHE_DIR /work/uo0780/u241155/jaxcache
setenv JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0.5

setenv PYOM3_ROOT               /work/uo0780/u241155/pyOM3_current/
setenv PYOM3_BACKEND            jax
# jax takes 75 % of each GPU's memory up front by default; the whole device
# is ours, so let it have all of it
setenv XLA_PYTHON_CLIENT_MEM_FRACTION 0.95

setenv EXP_SETUP                icon033_setup
setenv EXP_WAVES                0
setenv EXP_INPUT                .
setenv EXP_FORCING_DIR          /work/uo0780/u241155/pyOM3_real/era5_icon033
setenv EXP_FORCING_TAG          icon033
setenv EXP_FORCING_START        1990-05-01
setenv EXP_FORCING_CYCLE_DAYS   365
setenv EXP_OCEAN_DT             216
setenv EXP_SEGMENT_DAYS         60
@ NPX = $SLURM_NNODES * 4                  # one shard per GPU, 4 per node
@ REM = 1080 % $NPX
if ($REM != 0) then
  echo "$SLURM_NNODES nodes = $NPX shards in x do not divide nx = 1080; use 1, 2, 3, 5, 6, 9 or 10 nodes"
  exit 1
endif
setenv EXP_NPX_OCEAN            $NPX
setenv EXP_NPY_OCEAN            1
setenv EXP_RESTART_RAW          1          # per shard (see above); 0 = gathered
setenv EXP_RESTART_DAYS         10
# setenv EXP_INIT_OCEAN         ../icon033_spinup/icon033_restart_ocean.nc   # a GATHERED restart (see above)

setenv EXP_VADV_SUB_CYCLING     2
# the free surface: implicit (the CG, 27 % of the GPU step) or the
# split-explicit sub cycle, EXP_SUB_CYCLING sub steps (24) with forward-
# backward theta EXP_FB_THETA (0.5); see the FREE_SURFACE block of
# icon033_setup.py. A run keeps its method from the cold start.
setenv EXP_FREE_SURFACE         explicit
# the two offsets of run.sh: measure them for this grid and period and keep
# both scripts on the same numbers
setenv EXP_QNET_OFFSET          14.40
setenv EXP_SW_BOTTOM            1
setenv EXP_FRESHWATER           1
setenv EXP_EMP_OFFSET           "-0.3050"
setenv EXP_SSS_PISTON           0.17
setenv EXP_AVERAGE              month
setenv EXP_SNAPINT_OCEAN_DAYS   30
setenv EXP_KE_BUDGET            0
setenv EXP_PROFILE              2
setenv EXP_PROFILE_EVERY        200


# switch these 3 off to not using relative wind bulk formulae
setenv EXP_STRESS_MODE        relwind
setenv EXP_STRESS_UO          lagrangian        # u_o = u_L (eulerian: u_L - u_S)
setenv EXP_FORCING_WIND_TAG   icon033w          # era5_icon033w_YYYY-MM.nc in EXP_FORCING_DIR

setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------- checkpoints
set CHECKPOINTS = ( 1825 2190 2555 2920 3650 5475 7300 )
set DT          = 216
set RESTART     = icon033_restart_ocean

if ($TEST != 0) then
  setenv EXP_RUNLEN_DAYS          0.2
  setenv EXP_SEGMENT_DAYS         0.1
  setenv EXP_INIT_CHECK           0
  setenv EXP_AVERAGE              0.1d
  setenv EXP_SNAPINT_OCEAN_DAYS   0.1
  setenv EXP_PROFILE_EVERY        20
  set TARGET = 0
  echo "icon033 GPU TEST: 0.2 model days in two segments on $SLURM_NNODES node(s), $NPX GPU(s)"
else
  if (-e ${RESTART}.nc) then
    set DAY = `python3 -c "from netCDF4 import Dataset; f=Dataset('${RESTART}.nc'); print(int(round(f.n_cpl*$DT/86400.)))"`
  else
    set DAY = 0
  endif
  set TARGET = 0
  foreach c ( $CHECKPOINTS )
    if ( $c > $DAY && $TARGET == 0 ) set TARGET = $c
  end
  if ( $TARGET == 0 ) then
    echo "all checkpoints reached, model day $DAY -- nothing left to do"
    exit 0
  endif
  setenv EXP_RUNLEN_DAYS $TARGET
  @ TGTYEARS = $TARGET / 365
  echo "icon033 GPU: model day $DAY, running toward checkpoint $TARGET d = $TGTYEARS y"
endif


unsetenv SLURM_CPUS_PER_TASK


srun -l --ntasks=$SLURM_NTASKS python3 -u ../setups/coupled_common/run.py
set st = $status

if ($st != 0) then
  echo "segment failed with status $st, chain stopped"
  exit $st
endif

if (! -e RUN_DONE) then
  sbatch --nodes=$SLURM_NNODES --export=ALL,TEST=$TEST  $SCRIPT        # mid-chain, next segment
  exit 0
endif
if ($TEST != 0) then
  echo "GPU test complete"
  exit 0
endif

# a checkpoint is reached: archive the header AND its shard directory (raw),
# which belong together -- coupler_io derives <name>_pe/ from the header's name
if (-e ${RESTART}.nc) then
  if (-e ${RESTART}_y${TGTYEARS}.nc) then
    echo "${RESTART}_y${TGTYEARS}.nc exists already, not overwritten"
  else
    cp ${RESTART}.nc ${RESTART}_y${TGTYEARS}.nc
    if (-d ${RESTART}_pe) cp -r ${RESTART}_pe ${RESTART}_y${TGTYEARS}_pe
    echo "archived ${RESTART} at day $TARGET as ${RESTART}_y${TGTYEARS}"
  endif
endif

set LAST = $CHECKPOINTS[$#CHECKPOINTS]
if ( $TARGET == $LAST ) then
  echo "complete at day $TARGET"
else
  rm -f RUN_DONE
  sbatch --nodes=$SLURM_NNODES --export=ALL,TEST=$TEST $SCRIPT
endif
