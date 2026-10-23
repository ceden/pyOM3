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
# era5_icon01_YYYY-MM.nc in ../era5_icon01, on the ERA5 grid.
#
# READ THE COST FIRST. 1058 million cells at dt = 36 s is 2400 ocean steps
# per model day. By the icon033 scaling (0.47 s per step at 101,760 cells per
# rank) this is about 0.6 s per step on the 7650 ranks below, i.e.
#     26 minutes per model DAY, 6 days of wall clock per model YEAR,
# and an 8 hour job advances about 18 model days. The run lengths below are
# therefore a PLAN, not a promise: run one segment, read the seconds per step
# off the profiler table, and decide the period before launching a chain.
#
# TEST = 1 (default) is the short version to check the installation: 0.2
# model days in two 0.1 day segments, so the restart path is used, about 5
# minutes of compute. TEST = 0 is production.
#
# THE TIME STEP IS WORTH TESTING FIRST. 36 s is setup1.f90's value and POP
# runs the same horizontal spacing at 216 s; the difference is this grid's
# 128 levels and 2.05 m top cell, not anything horizontal (see the note at
# ocean_dt in icon01_setup.py). Run the TEST three times with
#   setenv EXP_OCEAN_DT 36 / 72 / 108
# and compare the canary and the identity line. Every factor on the step is
# the same factor on the cost of the whole experiment.
# EXP_INIT_CHECK = 0 lets the branches start from a spinup that is not on a
# whole repeat year; production has it 1.
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
#   3  1200 x  765   918 k     36 s     1    30 x 45    1350 + 7650    71
#   4   900 x  573   516 k     36 s     1    24 x 32     768 + 7650    66
# C = 3 is the default and the one to use: 3600 and 2295 both divide by 3, so
# no wave row or column is ragged, the wave cells are 6.7 km, and the wave
# model steps ONCE per ocean step. C = 1 and C = 2 are listed in
# icon01_setup.py for completeness but are not affordable: on the ocean grid
# the wave model would need a 12 s step and three sub-steps per ocean step on
# 2.2 km cells.
# The wave PE counts are estimates scaled from the icon033 C = 2 measurement.
# Read the two idle percentages in the log ('load wave', 'load ocean') and
# move ranks between the groups. A run directory keeps ONE C, because a wave
# restart belongs to its wave grid. Submit the coupled mode with its own
# node count:
#   sbatch --nodes=71 --ntasks=9088 --export=ALL,TEST=0,C=3 run_icon01.sh
#   sbatch --nodes=66 --ntasks=8448 --export=ALL,TEST=0,C=4 run_icon01.sh
#
# THE OCEAN LAYOUT IS THE SAME IN ALL THREE MODES and EXP_RESTART_RAW = 1, so
# the restart is a set of per-rank blocks and a branch must keep the spinup's
# layout. To branch on another one, rerun the LAST spinup segment with
# EXP_RESTART_RAW = 0 and keep it 1 here.
#
# THE TWO OFFSETS are properties of the forcing over the wet cells and of the
# PERIOD. Measure each with
#   python3 ../forcing_common/measure_era5_offsets.py 1990-05 1991-04 \
#           --grid grid_topo.nc --dir ../era5_icon01 --tag icon01
# (and the branch period for the branches), and put the SAME branch numbers
# in both branches or the two runs are not comparable. The values below are
# icon033's and are NOT this grid's.

set SCRIPT = run.sh
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
setenv EXP_FORCING_DIR            ../era5_icon01
setenv EXP_FORCING_TAG            icon01
setenv EXP_FRESHWATER             $FRESH
setenv EXP_SW_BOTTOM              1
setenv EXP_RESTART_RAW            1
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
setenv EXP_OCEAN_DT               144
# EXP_OCEAN_DT is the setup module's default of 36 s unless set here; see
# the note above before changing it
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
    set RUNLEN = 365 ; set SEGMENT = 15               # 1 y, ~15 d per job
    breaksw
  case ocean:
    setenv EXP_WAVES                  0
    setenv EXP_FORCING_START          1991-05-01
    setenv EXP_FORCING_CYCLE_DAYS     0
    setenv EXP_QNET_OFFSET            $QNET_BRANCH
    setenv EXP_EMP_OFFSET             "$EMP_BRANCH"
    setenv EXP_INIT_OCEAN             ../icon01_spinup/icon01_restart_ocean.nc
    setenv EXP_NPX_OCEAN 90 ; setenv EXP_NPY_OCEAN 85
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
setenv EXP_RUNLEN_DAYS   $RUNLEN
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
if (-e RUN_DONE) then
  echo "$MODE complete"
else
  sbatch --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS --export=ALL,MODE=$MODE,TEST=$TEST,C=$C $SCRIPT
endif
