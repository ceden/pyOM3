#!/bin/tcsh
#SBATCH --job-name=ic033_spin
#SBATCH --account=uo0780
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks=936
#SBATCH --ntasks-per-node=117
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=log_spin_%j.out
#SBATCH --error=log_spin_%j.out

# icon_0.33deg SPINUP: the ocean ALONE, repeat-year ERA5 (1 May 1990 to 30
# April 1991) for EXP_RUNLEN_DAYS. The branch starts from its last restart.
#
# THE SHARED MACHINERY. The run directory holds icon033_setup.py, this script
# and the input files (grid_topo.nc, initial_conditions.nc,
# ecmwf_surface_forcing.nc, woa_surface_forcing.nc) and nothing else: the
# driver is ../coupled_common/run.py, which finds the package and
# ../forcing_common from its own location and puts THIS directory last on the
# path, so a stale run.py or coupler file left here cannot shadow the shared
# one. (The old monolithic icon033_setup.py and the local run.py it needed
# are gone; delete any copy of run.py, coupler_*.py, forcing_*.py or
# block_io.py still in this directory.)
#
# The run is chained: each job runs one segment, writes the restart and
# resubmits itself until run.py leaves RUN_DONE behind. A segment that fails
# does NOT resubmit, so a crash stops the chain instead of burning the
# allocation. Delete RUN_DONE and the restarts to start over.
#
# DECOMPOSITION 72 x 13 = 936 ranks on 8 nodes, blocks of exactly 15 x 53:
# 1080 = 2^3 3^3 5 and 689 = 13 x 53, so both divisions are exact. The ragged
# 64 x 16 = 1024 this started with was 12 % slower on 10 % more cores.
# EXP_RESTART_RAW BINDS THE RESTART TO THIS LAYOUT. The branches must use the
# same 72 x 13, or the raw restart cannot be read (run.py stops with that
# message). Change the layout only together with a cold start -- or rerun the
# last segment with EXP_RESTART_RAW=0 (GATHERED=1 below), which writes the
# gathered netCDF form that any layout, and the GPU run (run_gpu.sh), can
# start from through EXP_INIT_OCEAN.
#
# CHECKPOINTS. The chain does not run straight to the end: it stops at each
# day in CHECKPOINTS below, archives the restart as
# icon033_restart_ocean_y<NN>.nc plus its _pe directory, and resubmits toward
# the next one. That is why the targets are set here rather than copied out
# by hand -- restarts are written every 10 model days, 5 years is 1825 days
# and is not a multiple of 10, so an exact five-year state never exists to
# copy; and run.py refuses a branch start that does not sit on a whole repeat
# year, which 1825 does and 1830 does not. Each archive is the full ocean
# state, about 15 GB.
#
# ERA5 comes from this experiment's own set, era5_icon033_YYYY-MM.nc, built by
# prep_era5_icon033.sh and kept on the ERA5 grid; forcing_era5_native.py
# interpolates it to each rank's block while the run goes on.
#
# TEST = 1 (--export=ALL,TEST=1) is the short version to check the
# installation: 0.2 model days in two 0.1 day segments, so the restart path
# is used, a few minutes of compute. Default 0 = production.
#
# COST is not known yet. Set EXP_SEGMENT_DAYS from what the first job
# reports ("load ocean: x s/step", 400 steps per model day) before launching
# the chain: a segment must finish inside the walltime, restart write
# included.
#
# Every physical parameter is a default of icon033_setup.py and is printed by
# the CONFIG lines of the log. Only what this run does differently is set here.

set SCRIPT = run.sh                  # this file's name: it resubmits itself
if (! $?TEST)     set TEST = 0
if (! $?GATHERED) set GATHERED = 0   # 1: write the gathered restart (see above)

module purge
conda activate pyOM3_wave
setenv OMPI_MCA_opal_common_ucx_opal_mem_hooks 1

cd "$SLURM_SUBMIT_DIR"

setenv PYOM3_ROOT               /work/uo0780/u241155/pyOM3_current/

setenv EXP_SETUP                icon033_setup
setenv EXP_WAVES                0
setenv EXP_INPUT                .
setenv EXP_FORCING_DIR          /work/uo0780/u241155/pyOM3_real/era5_icon033
setenv EXP_FORCING_START        1990-05-01
setenv EXP_FORCING_CYCLE_DAYS   365
#                               EXP_RUNLEN_DAYS is set by the checkpoint
#                               block below, not here
setenv EXP_OCEAN_DT             216
setenv EXP_SEGMENT_DAYS         60
setenv EXP_NPX_OCEAN            72
setenv EXP_NPY_OCEAN            13
if ($GATHERED) then
  setenv EXP_RESTART_RAW        0
else
  setenv EXP_RESTART_RAW        1
endif
setenv EXP_FORCING_TAG          icon033

setenv EXP_VADV_SUB_CYCLING 2

# ERA5 runs on a prescribed SST, so its surface energy budget does not close;
# over the icon15 grid the annual mean of sshf+slhf+ssr+str was +14.28 W/m2
# into the ocean and the SST restoring can only remove it by holding a warm
# bias. THAT NUMBER IS NOT THIS GRID'S: the wet cells are different and the
# fields now come from the native ERA5 grid by bilinear interpolation, not
# remapped conservatively onto the model cells. Measure it over this grid's
# wet cells for this period and replace it:
#   python3 ../forcing_common/measure_era5_offsets.py 1990-05 1991-04 \
#           --grid grid_topo.nc --dir ../era5_icon033 --tag icon033
setenv EXP_QNET_OFFSET          14.40
setenv EXP_SW_BOTTOM            1
# freshwater flux: e and tp are in the same files as the flux fields
setenv EXP_FRESHWATER           1
# mm/day subtracted from P-E. Measure it over this grid's wet cells for this
# period, as for the heat flux; without it the salinity drifts because the
# rivers are missing.
setenv EXP_EMP_OFFSET           "-0.3050"
# m/day. The default here is dzt(top)/(30 d) = 0.068 m/day, five times weaker
# than the 4 degree runs' 0.367 because the top cell is 2.05 m instead of 10 m.
setenv EXP_SSS_PISTON           0.17
setenv EXP_AVERAGE              month
setenv EXP_SNAPINT_OCEAN_DAYS   30
setenv EXP_KE_BUDGET            0
setenv EXP_PROFILE              2
setenv EXP_PROFILE_EVERY        200


setenv OMP_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1

# ---------------------------------------------------------------- checkpoints
# model days at which the state is archived: 5, 8, 10, 15, 20 years of
# 365 days. Edit this line and nothing else.
set CHECKPOINTS = ( 1825 2190 2555 2920 3650 5475 7300 )
set DT          = 216                      # must equal EXP_OCEAN_DT above
set RESTART     = icon033_restart_ocean

if ($TEST != 0) then
  setenv EXP_RUNLEN_DAYS          0.2
  setenv EXP_SEGMENT_DAYS         0.1
  setenv EXP_INIT_CHECK           0
  setenv EXP_AVERAGE              0.1d
  setenv EXP_SNAPINT_OCEAN_DAYS   0.1
  setenv EXP_PROFILE_EVERY        100
  set TARGET = 0
  echo "icon033 spinup TEST: 0.2 model days in two segments"
else
  # where the chain is now: the restart header carries the coupling step
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
  echo "model day $DAY, running toward checkpoint $TARGET d = $TGTYEARS y"
endif

srun --ntasks=$SLURM_NTASKS --cpu_bind=cores --distribution=block:cyclic  python3 -u ../setups/coupled_common/run.py
set st = $status

if ($st != 0) then
  echo "segment failed with status $st, chain stopped"
  exit $st
endif

if (! -e RUN_DONE) then
  sbatch --export=ALL,TEST=$TEST,GATHERED=$GATHERED $SCRIPT   # mid-chain, next segment
  exit 0
endif
if ($TEST != 0) then
  echo "spinup test complete"
  exit 0
endif

# the checkpoint is reached: archive the header AND its block directory, which
# belong together -- coupler_io derives <name>_pe/ from the header's name
# (a gathered restart, GATHERED=1, has no _pe directory)
foreach R ( icon033_restart_ocean icon033_restart_wave )
  if (-e ${R}.nc) then
    if (-e ${R}_y${TGTYEARS}.nc) then
      echo "${R}_y${TGTYEARS}.nc exists already, not overwritten"
    else
      cp    ${R}.nc  ${R}_y${TGTYEARS}.nc
      if (-d ${R}_pe) cp -r ${R}_pe ${R}_y${TGTYEARS}_pe
      echo "archived ${R} at day $TARGET as ${R}_y${TGTYEARS}"
    endif
  endif
end

set LAST = $CHECKPOINTS[$#CHECKPOINTS]
if ( $TARGET == $LAST ) then
  echo "spinup complete at day $TARGET"
else
  rm -f RUN_DONE                      # on to the next checkpoint
  sbatch --export=ALL,TEST=$TEST,GATHERED=$GATHERED $SCRIPT
endif
