#!/bin/tcsh
#SBATCH --job-name=ic01_gpu
#SBATCH --account=uo0780_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=log_gpu_%j.out
#SBATCH --error=log_gpu_%j.out

# icon_0.1deg, the OCEAN ALONE on the JAX BACKEND over several Levante GPU
# nodes: one process per GPU under srun, jax.distributed initialised in
# pyOM3/__init__, the model's mesh spanning the processes
# (coupled_common/run.py with PYOM3_BACKEND=jax). The GPU counterpart of
# run.sh for its ocean-alone modes:
#
# TEST = 1 (default): 0.2 model days in two 0.1 day segments from a cold
# start (120 steps, the restart path used), ~10 min on 6 nodes with the
# compilation. TEST = 0 is the production chain of run.sh (yearly
# checkpoints, resubmission).

set SCRIPT = run_gpu.sh
set DT    = 144                  # ocean step [s]; EXP_OCEAN_DT is set from it
set FRESH = 1
set QNET_SPINUP = 14.40 ; set EMP_SPINUP = "-0.3050"    # <- icon033's numbers; measure (run.sh)
if (! $?TEST) set TEST = 1

setenv HDF5_USE_FILE_LOCKING FALSE
setenv TF_CPP_MIN_LOG_LEVEL 2
module purge
conda activate jax_sharding1
setenv JAX_COMPILATION_CACHE_DIR /work/uo0780/u241155/jaxcache
setenv JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0.5

setenv PYOM3_ROOT               /work/uo0780/u241155/pyOM3_current/
setenv PYOM3_BACKEND            jax
setenv XLA_PYTHON_CLIENT_MEM_FRACTION 0.95

# ---- the shard layout from the node count ----------------------------------
switch ($SLURM_NNODES)
  case 1:  set NPX = 4  ; set NPY = 1  ; breaksw
  case 2:  set NPX = 8  ; set NPY = 1  ; breaksw
  case 3:  set NPX = 4  ; set NPY = 3  ; breaksw
  case 4:  set NPX = 16 ; set NPY = 1  ; breaksw
  case 5:  set NPX = 4  ; set NPY = 5  ; breaksw
  case 6:  set NPX = 8  ; set NPY = 3  ; breaksw
  case 9:  set NPX = 12 ; set NPY = 3  ; breaksw
  case 10: set NPX = 8  ; set NPY = 5  ; breaksw
  case 12: set NPX = 16 ; set NPY = 3  ; breaksw
  case 15: set NPX = 12 ; set NPY = 5  ; breaksw
  case 18: set NPX = 8  ; set NPY = 9  ; breaksw
  case 20: set NPX = 16 ; set NPY = 5  ; breaksw
  case 25: set NPX = 20 ; set NPY = 5  ; breaksw
  case 30: set NPX = 24 ; set NPY = 5  ; breaksw
  case 36: set NPX = 16 ; set NPY = 9  ; breaksw
  case 45: set NPX = 12 ; set NPY = 15 ; breaksw
  default:
    echo "no exact shard layout for $SLURM_NNODES nodes (npx | 3600, npy | 2295, npx*npy = 4 nodes); see the table"
    exit 1
endsw
setenv EXP_NPX_OCEAN $NPX
setenv EXP_NPY_OCEAN $NPY

# ---- the same in both modes ------------------------------------------------
setenv EXP_SETUP                  icon01_setup
setenv EXP_INPUT                  .
setenv EXP_FORCING_DIR            /work/uo0780/u241155/pyOM3_real/era5_icon033
setenv EXP_FORCING_TAG            icon033
setenv EXP_FRESHWATER             $FRESH
setenv EXP_SW_BOTTOM              1
setenv EXP_RESTART_RAW            1               # per shard
setenv EXP_RESTART_DAYS           10
setenv EXP_SSS_PISTON             0.17
setenv EXP_AVERAGE                month
setenv EXP_SNAPINT_OCEAN_DAYS     5
setenv EXP_KE_BUDGET              0               # must be 0 with the explicit free surface
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

set RUNLEN = 3650 ; set SEGMENT = 15

# set these three to have relative wind bulk formulae
setenv EXP_STRESS_MODE            relwind
setenv EXP_STRESS_UO              lagrangian
setenv EXP_FORCING_WIND_TAG       icon033w          # era5_icon033w_YYYY-MM.nc in EXP_FORCING_DIR

if ($TEST != 0) then
  set RUNLEN = 0.2 ; set SEGMENT = 0.1
  setenv EXP_INIT_CHECK             0
  setenv EXP_AVERAGE                0.1d
  setenv EXP_SNAPINT_OCEAN_DAYS     0.1
  setenv EXP_PROFILE_EVERY          20
endif
# ---- checkpoints: the target of THIS chain leg -----------------------------
set RESTART = icon01_restart_ocean
if ($TEST != 0) then
  set TARGET = $RUNLEN
else
  if (-e ${RESTART}.nc) then
    set DAY = `python3 -c "from netCDF4 import Dataset; f=Dataset('${RESTART}.nc'); print(int(round(f.n_cpl*$DT/86400.)))"`
  else
    set DAY = 0
  endif
  if ($DAY >= $RUNLEN) then
    echo "icon01: model day $DAY has reached RUNLEN $RUNLEN, nothing to do"
    exit 0
  endif
  @ TARGET = ( $DAY / 365 + 1 ) * 365
  if ($TARGET > $RUNLEN) set TARGET = $RUNLEN
  echo "icon01: model day $DAY, next checkpoint at day $TARGET"
endif
setenv EXP_RUNLEN_DAYS   $TARGET
setenv EXP_SEGMENT_DAYS  $SEGMENT
echo "icon01 GPU: TEST $TEST, $RUNLEN days in segments of $SEGMENT, $SLURM_NNODES nodes, shards $NPX x $NPY"

setenv EXP_CANARY_EVERY 100
setenv EXP_CANARY_CFL  1

setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK
unsetenv SLURM_CPUS_PER_TASK

srun -l --ntasks=$SLURM_NTASKS python3 -u ../setups/coupled_common/run.py
set st = $status

if ($st != 0) then
  echo "segment failed with status $st, chain stopped"
  exit $st
endif

if (! -e RUN_DONE) then                    # mid-chain: the next segment
  sbatch --nodes=$SLURM_NNODES --export=ALL,TEST=$TEST $SCRIPT
  exit 0
endif
if ($TEST != 0) then
  echo "run complete (test)"
  exit 0
endif

# a checkpoint is reached: archive the header AND its shard directory
@ YRS  = $TARGET / 365
@ BACK = $YRS * 365
if ($BACK == $TARGET) then
  set TAG = y$YRS
else
  set TAG = d$TARGET
endif
if (-e ${RESTART}.nc) then
  if (-e ${RESTART}_${TAG}.nc) then
    echo "${RESTART}_${TAG}.nc exists already, not overwritten"
  else
    cp    ${RESTART}.nc  ${RESTART}_${TAG}.nc
    cp -r ${RESTART}_pe  ${RESTART}_${TAG}_pe
    echo "archived ${RESTART} at day $TARGET as ${RESTART}_${TAG}"
  endif
endif

if ($TARGET >= $RUNLEN) then
  echo "run complete at day $TARGET"
else
  rm -f RUN_DONE                           # on to the next year
  sbatch --nodes=$SLURM_NNODES --export=ALL,TEST=$TEST $SCRIPT
endif
