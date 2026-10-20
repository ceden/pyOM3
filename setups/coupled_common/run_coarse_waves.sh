#!/bin/tcsh
#SBATCH --job-name=ic033_cw
#SBATCH --account=uo0780
#SBATCH --partition=compute
#SBATCH --nodes=16
#SBATCH --ntasks=2048
#SBATCH --time=08:00:00
#SBATCH --output=log_cw_%j.out
#SBATCH --error=log_cw_%j.out

# icon_0.33deg COUPLED, WAVE MODEL ON A COARSER GRID: a 30 day test branch
# from the ocean state of icon033_spinup. Everything but the wave grid is
# run_branch_coupled.sh.
#
# ONE DIRECTORY PER c. The restart, snapshot and coupling files carry no c in
# their names, and a wave restart belongs to its wave grid.
#
#   c   wave grid   wave dt  sub   wave PEs     ranks        submit with
#   2   540 x 344    72 s     3    32 x 32   1024 + 936   sbatch run_coarse_waves.sh
#   3   360 x 229   108 s     2    16 x 16    256 + 936   sbatch --nodes=10 --ntasks=1280 --export=ALL,C=3 run_coarse_waves.sh
#   4   270 x 172   108 s     2    12 x 12    144 + 936   sbatch --nodes=9  --ntasks=1152 --export=ALL,C=4 run_coarse_waves.sh
#   1  1080 x 689    36 s     6    64 x 64   4096 + 936   sbatch --nodes=40 --ntasks=5120 --export=ALL,C=1 run_coarse_waves.sh
# c = 1 is the same-grid twin over the same 30 days, for the comparison.
#
# THE WAVE LAYOUTS ARE ESTIMATES, scaled from the measured same-grid run
# (4096 wave PEs, 0.85 s per coupling step, blocks of 17 x 11 cells): the
# wave work per coupling step falls as c*c times the sub-cycle ratio, 8, 27
# and 48 times, and the target is the ocean's own 0.47 s. Read the two idle
# percentages in the log ('load wave', 'load ocean') and move EXP_NPX_WAVE /
# EXP_NPY_WAVE until the ocean no longer waits.
#
# THE OCEAN LAYOUT MUST BE THE SPINUP'S if its restart is raw (per rank). Set
# the two numbers below to what icon033_spinup ran on.
#
# EXP_INIT_CHECK = 0: run.py stops a branch whose start is not on a whole
# repeat year, because the forcing jumps there. For this test, from a spinup
# that is still running, that does not matter, so it only warns. Take it out
# for a production branch.

if (! $?C) set C = 2
set SCRIPT = run_coarse_waves.sh

switch ($C)
  case 1:
    set NPXW = 64 ; set NPYW = 64 ; breaksw
  case 2:
    set NPXW = 32 ; set NPYW = 32 ; breaksw
  case 3:
    set NPXW = 16 ; set NPYW = 16 ; breaksw
  case 4:
    set NPXW = 12 ; set NPYW = 12 ; breaksw
  default:
    echo "C = $C is not supported" ; exit 1
endsw

module purge
conda activate pyOM3_wave
setenv OMPI_MCA_opal_common_ucx_opal_mem_hooks 1

cd "$SLURM_SUBMIT_DIR"

setenv EXP_SETUP                icon033_setup
setenv EXP_WAVES                1
setenv EXP_WAVE_COARSEN         $C
setenv EXP_INPUT                .
setenv EXP_FORCING_DIR          ../era5_icon033
setenv EXP_FORCING_START        1991-05-01
setenv EXP_FORCING_CYCLE_DAYS   0
setenv EXP_INIT_OCEAN           ../icon033_spinup/icon033_restart_ocean.nc
setenv EXP_INIT_CHECK           0
setenv EXP_RUNLEN_DAYS          30
setenv EXP_SEGMENT_DAYS         30
setenv EXP_SPINUP_DAYS          2
setenv EXP_NPX_WAVE             $NPXW
setenv EXP_NPY_WAVE             $NPYW
setenv EXP_NPX_OCEAN            72
setenv EXP_NPY_OCEAN            13
setenv EXP_RESTART_RAW          1
setenv EXP_FORCING_TAG          icon033
setenv EXP_QNET_OFFSET          14.40
setenv EXP_SW_BOTTOM            1
setenv EXP_FRESHWATER           1
setenv EXP_EMP_OFFSET           "-0.3050"
setenv EXP_SSS_PISTON           0.17
setenv EXP_DIRECT_BRANCH        1
setenv EXP_AVERAGE              month
setenv EXP_SNAPINT_OCEAN_DAYS   3
setenv EXP_SNAPINT_WAVE_DAYS    3
setenv EXP_CPL_WRITE_EVERY      400
setenv EXP_KE_BUDGET            0
setenv EXP_PROFILE              1
setenv EXP_PROFILE_EVERY        500

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
  echo "coarse-wave test complete"
else
  sbatch --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS --export=ALL,C=$C $SCRIPT
endif
