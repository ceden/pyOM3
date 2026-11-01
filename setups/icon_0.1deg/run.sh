#!/bin/tcsh
#SBATCH --job-name=ic01
#SBATCH --account=uo0780
#SBATCH --partition=compute
#SBATCH --nodes=60
#SBATCH --ntasks=7680
#SBATCH --time=08:00:00
#SBATCH --output=log_ic01_%j.out
#SBATCH --error=log_ic01_%j.out

# icon_0.1deg with hourly ERA5, on the shared coupled_common / forcing_common.
# ONE SCRIPT, THREE RUNS, each in its OWN directory (the restart and output
# files carry no run name):
#
#   directory        MODE      what
#   icon01_spinup    spinup    ocean alone, repeat year from 1990-05-01
#   icon01_ocean     ocean     ocean alone, 1991-05-01 on, from the spinup
#   icon01_coupled   coupled   the same with the wave model
#
# MODE is the last part of the directory name, or --export=ALL,MODE=...
# Each directory holds: icon01_setup.py, this script, and the input files
# grid_topo.nc, initial_conditions.nc, ecmwf_surface_forcing.nc,
# woa_surface_forcing.nc (or EXP_INPUT points to them). The ERA5 files are
# icon033's set, era5_icon033_YYYY-MM.nc and the 10 m wind
# era5_icon033w_YYYY-MM.nc, on the ERA5 grid (the native files carry
# nothing about the model grid). The physics and the numerics are those of
# run_gpu.sh, so a CPU run and a GPU run are the same experiment.
#
# THE TIME STEP is 144 s with the tracer advection outside the AB step in
# two vertical sub-steps (EXP_VADV_SUB_CYCLING 2) and AB3 (EXP_OCEAN_AB 3).
# Without the split, 144 s crashed after 4-6 days (vertical CFL on the
# Antarctic shelf). EXP_CANARY_CFL 1 prints max w dt/dz with the canary.
#
# THE FREE SURFACE is split-explicit (FS = explicit below) in the two
# ocean-alone modes, as on the GPU. The coupled mode needs the implicit one
# (run.py stops otherwise), so it keeps it. A restart of one method is
# refused by a run with the other (coupler_io.py): an existing chain written
# with the implicit free surface continues only with FS = implicit, and the
# coupled branch cannot start from an explicit spinup.
#
# THE STRESS is the relative-wind bulk formula with ERA5's drag coefficient
# (EXP_STRESS_MODE relwind, the Lagrangian surface velocity), in all modes.
# In the coupled mode the waves take this stress from the ocean.
#
# COST, measured at dt 144 with the split on 7650 ranks (implicit free
# surface): 1.04 s per step, 10.4 minutes per model DAY, 2.6 days of wall
# clock per model YEAR; an 8 hour job advances about 45 model days.
#
# TEST = 1 is the short version to check the installation: 0.2 model days
# in two 0.1 day segments, so the restart path is used. TEST = 0 (default
# here) is production. EXP_INIT_CHECK = 0 lets the branches start from a
# spinup that is not on a whole repeat year; production has it 1.
#
# CHECKPOINTS. The chain stops at every whole model year (TARGET below).
# There the restart header AND its per-rank directory are copied to
# icon01_restart_ocean_y<N>, and the chain goes on to the next year.
#
# LAYOUTS. The ocean is 90 x 85 = 7650 ranks, blocks of exactly 40 x 27
# (3600 = 2^4 3^2 5^2, 2295 = 3^3 5 17); the header asks for the 60 nodes
# that holds. Other exact ones, if you want more or fewer cores:
#     80 x  51 =  4080, blocks 45 x 45, 32 nodes
#    100 x  51 =  5100, blocks 36 x 45, 40 nodes
#    120 x  85 = 10200, blocks 30 x 27, 80 nodes
#     72 x 135 =  9720, blocks 50 x 17, 76 nodes
# Ragged blocks cost a factor, so stay on divisors.
#
# THE WAVE GRID, MODE coupled only. C is the coarsening (EXP_WAVE_COARSEN):
#
#   C   wave grid     cells   wave dt  sub   wave PEs      ranks     nodes
#   3  1200 x  765   918 k     36 s     4    30 x 45    1350 + 7650    71
#   4   900 x  573   516 k     48 s     3    24 x 32     768 + 7650    66
# C = 3 is the default and the one to use: 3600 and 2295 both divide by 3, so
# no wave row or column is ragged and the wave cells are 6.7 km. At the
# ocean step of 144 s the wave model sub-cycles (column sub). C = 1 and C = 2 are listed in
# icon01_setup.py for completeness but are not affordable: on the ocean grid
# the wave model would need a 12 s step and three sub-steps per ocean step on
# 2.2 km cells.
# The wave PE counts are estimates scaled from the icon033 C = 2 measurement.
# Read the two idle percentages in the log ('load wave', 'load ocean') and
# move ranks between the groups. A run directory keeps ONE C, because a wave
# restart belongs to its wave grid. Submit the coupled mode with its own
# node count:
#   sbatch --nodes=71 --ntasks=9088 --export=ALL,TEST=0,C=3 run.sh
#   sbatch --nodes=66 --ntasks=8448 --export=ALL,TEST=0,C=4 run.sh
#
# THE OCEAN LAYOUT IS THE SAME IN ALL THREE MODES and EXP_RESTART_RAW = 1, so
# the restart is a set of per-rank blocks and a branch must keep the spinup's
# layout. To branch on another one, rerun the LAST spinup segment with
# EXP_RESTART_RAW = 0 and keep it 1 here.
#
# THE TWO OFFSETS are properties of the forcing over the wet cells and of the
# PERIOD. Measure each with
#   python3 ../forcing_common/measure_era5_offsets.py 1990-05 1991-04 \
#           --grid grid_topo.nc --dir $FORCING_DIR --tag icon033
# (and the branch period for the branches), and put the SAME branch numbers
# in both branches or the two runs are not comparable. The values below are
# icon033's and are NOT this grid's.

set SCRIPT = run.sh
set DT    = 144                  # ocean step [s]; EXP_OCEAN_DT is set from it
set FS    = explicit             # free surface of the ocean-alone modes; implicit to resume an implicit restart
set FORCING_DIR = /work/uo0780/u241155/pyOM3_real/era5_icon033
set FRESH = 1
set QNET_SPINUP = 14.40 ; set EMP_SPINUP = "-0.3050"    # <- measure, see above
set QNET_BRANCH = 14.40 ; set EMP_BRANCH = "-0.3050"    # <- measure, see above
if (! $?C) set C = 3
if (! $?TEST) set TEST = 0
if (! $?MODE) set MODE = `basename "$SLURM_SUBMIT_DIR" | sed 's/.*_//'`

module purge
conda activate pyOM3_wave
setenv OMPI_MCA_opal_common_ucx_opal_mem_hooks 1
cd "$SLURM_SUBMIT_DIR"

# ---- the same in all three runs -------------------------------------------
setenv EXP_SETUP                  icon01_setup
setenv EXP_INPUT                  .
setenv EXP_FORCING_DIR            $FORCING_DIR
setenv EXP_FORCING_TAG            icon033
setenv EXP_FORCING_WIND_TAG       icon033w          # era5_icon033w_YYYY-MM.nc in EXP_FORCING_DIR
setenv EXP_STRESS_MODE            relwind
setenv EXP_STRESS_UO              lagrangian
setenv EXP_FRESHWATER             $FRESH
setenv EXP_SW_BOTTOM              1
setenv EXP_RESTART_RAW            1
setenv EXP_RESTART_DAYS           10
# the top cell is 2.05 m, so the piston of the 4 degree runs would restore it
# in two days; 0.17 m/day is 12 days
setenv EXP_SSS_PISTON             0.17
# TKE: the defaults of icon01_setup.py are the production set (alpha_CB 100,
# NEMO length with beta 2e4, alpha_tke 2), nothing to set
setenv EXP_AVERAGE                month
setenv EXP_SNAPINT_OCEAN_DAYS     5
# the KE budget re-evaluates every momentum tendency from the state at every
# step, about 40 % on top of a step at this size
setenv EXP_KE_BUDGET              0
setenv EXP_OCEAN_DT               $DT
setenv EXP_VADV_SUB_CYCLING       2
setenv EXP_OCEAN_AB               3
setenv EXP_CANARY_EVERY           100
setenv EXP_CANARY_CFL             1
setenv EXP_PROFILE                2
setenv EXP_PROFILE_EVERY          200

# ---- per run ---------------------------------------------------------------
switch ($MODE)
  case spinup:
    setenv EXP_WAVES                  0
    setenv EXP_FORCING_START          1990-05-01
    setenv EXP_FORCING_CYCLE_DAYS     365
    setenv EXP_QNET_OFFSET            $QNET_SPINUP
    setenv EXP_EMP_OFFSET             "$EMP_SPINUP"
    setenv EXP_NPX_OCEAN 90 ; setenv EXP_NPY_OCEAN 85
    setenv EXP_FREE_SURFACE           $FS
    setenv EXP_SUB_CYCLING            48
    setenv EXP_FB_THETA               0.5
    set RUNLEN = 3650 ; set SEGMENT = 15              # 10 y, ~15 d per job
    breaksw
  case ocean:
    setenv EXP_WAVES                  0
    setenv EXP_FORCING_START          1991-05-01
    setenv EXP_FORCING_CYCLE_DAYS     0
    setenv EXP_QNET_OFFSET            $QNET_BRANCH
    setenv EXP_EMP_OFFSET             "$EMP_BRANCH"
    setenv EXP_INIT_OCEAN             ../icon01_spinup/icon01_restart_ocean.nc
    setenv EXP_NPX_OCEAN 90 ; setenv EXP_NPY_OCEAN 85
    setenv EXP_FREE_SURFACE           $FS
    setenv EXP_SUB_CYCLING            48
    setenv EXP_FB_THETA               0.5
    set RUNLEN = 365 ; set SEGMENT = 15
    breaksw
  case coupled:
    setenv EXP_WAVES                  1
    setenv EXP_FORCING_START          1991-05-01
    setenv EXP_FORCING_CYCLE_DAYS     0
    setenv EXP_QNET_OFFSET            $QNET_BRANCH
    setenv EXP_EMP_OFFSET             "$EMP_BRANCH"
    setenv EXP_INIT_OCEAN             ../icon01_spinup/icon01_restart_ocean.nc
    setenv EXP_WAVE_COARSEN           $C
    # the wave layout of this C (tcsh takes no command on a case line)
    if ($C == 3) then
      setenv EXP_NPX_WAVE 30 ; setenv EXP_NPY_WAVE 45
    else if ($C == 4) then
      setenv EXP_NPX_WAVE 24 ; setenv EXP_NPY_WAVE 32
    else
      echo "C = $C is not supported here; see the note above"
      exit 1
    endif
    setenv EXP_NPX_OCEAN 90 ; setenv EXP_NPY_OCEAN 85
    setenv EXP_FREE_SURFACE           implicit          # the coupled model needs it
    setenv EXP_SPINUP_DAYS            2
    setenv EXP_DIRECT_BRANCH          1
    setenv EXP_HS_FACTOR              1.6
    setenv EXP_SNAPINT_WAVE_DAYS      5
    # 9 fields of 33 MB per record on this grid: one per model day is 300 MB
    # per model month
    setenv EXP_CPL_WRITE_EVERY        2400            # one record per model day
    set RUNLEN = 365 ; set SEGMENT = 15
    breaksw
  default:
    echo "MODE = $MODE : use spinup, ocean or coupled" ; exit 1
endsw

if ($TEST != 0) then
  set RUNLEN = 0.2 ; set SEGMENT = 0.1
  setenv EXP_INIT_CHECK             0
  setenv EXP_AVERAGE                0.1d
  setenv EXP_SNAPINT_OCEAN_DAYS     0.1
  setenv EXP_PROFILE_EVERY          100
  if ($MODE == coupled) then
    setenv EXP_SPINUP_DAYS            0.05
    setenv EXP_SNAPINT_WAVE_DAYS      0.1
    setenv EXP_CPL_WRITE_EVERY        100
  endif
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
echo "icon01: MODE $MODE, TEST $TEST, $RUNLEN days in segments of $SEGMENT"
if ($MODE == coupled) echo "icon01: wave grid coarsened by $C"

setenv OMP_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1

srun --ntasks=$SLURM_NTASKS --cpu_bind=cores --distribution=block:cyclic  python3 -u ../coupled_common/run.py
set st = $status

if ($st != 0) then
  echo "segment failed with status $st, chain stopped"
  exit $st
endif
if (! -e RUN_DONE) then                    # mid-chain: the next segment
  sbatch --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS --export=ALL,MODE=$MODE,TEST=$TEST,C=$C $SCRIPT
  exit 0
endif
if ($TEST != 0) then
  echo "$MODE complete (test)"
  exit 0
endif

# a checkpoint is reached: archive the header AND its per-rank directory
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
  echo "$MODE complete at day $TARGET"
else
  rm -f RUN_DONE                           # on to the next year
  sbatch --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS --export=ALL,MODE=$MODE,TEST=$TEST,C=$C $SCRIPT
endif
