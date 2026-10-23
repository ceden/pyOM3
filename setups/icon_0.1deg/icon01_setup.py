"""
icon_0.1deg experiment physics: the model grid, the input files, the surface
forcing and the wave / ocean model classes.

The 1/10 degree, 128 level global configuration of pyOM2 setup1.f90 ("1/10
global with 128 levels, from Icon 10km version"), converted to pyOM3 and
forced by hourly ERA5. It is icon033_setup.py on the 1/10 degree grid: the
machinery, the coupler, the u_L formulation and the energy diagnostics are
the same, and so is the I/O a domain of this size needs -- raw per-rank
snapshots, restarts and means, and scattered input reads.

THE GRID is not written here. dxt(nx), dyt(ny), dzt(nz), the first U point
xu(1), yu(1) and kbot come from grid_topo.nc, the file the fortran setup
reads (its source is the Icon R2B8 10 km ocean grid), so there is one source
of truth for the grid. dxt is 1/10 degree everywhere, dyt is VARIABLE (0.020
to 0.100 degrees, the Icon grid keeps the cells near square), the domain runs
78.5 S to 69.5 N and is cyclic in x; the column is the 1/3 degree run's, 128
levels from 2.05 m at the surface to 197 m at 6100 m. kbot is the pyOM2
1-based index of the deepest wet level, which is also pe_realistic's padded
index at halo = 1, so nothing is remapped.

    3600 x 2295 x 128 = 1058 million cells, 11.1 times icon033
    5,153,349 wet surface cells
    cells 11.1 km at the equator, 2.22 km at the southern edge, in BOTH
    directions (dyt shrinks with dxt cos(lat), that is what "near square"
    means here)

WHAT IS DIFFERENT FROM icon033, and why

1. dt = 36 s, as setup1.f90 (dt_mom = 144/4). That is 2400 ocean steps per
   model day against 400 at 1/3 degree, so a model day costs 11.1 times the
   cells times 6 times the steps = 67 times an icon033 day at equal cores.
   READ THE COST NOTE at the end of this docstring before planning a run.
2. BIHARMONIC friction A_hbi = 2.7e10 with cosPower 1.5, as setup1.f90
   (icon033 runs 2e11 on its coarser cells). No harmonic friction, no
   isoneutral diffusion and no GM: this grid is eddy resolving.
3. Everything else is icon033's: TEOS-10, the implicit free surface with the
   surface tracer flux, superbee advection, the calibrated TKE set (alpha_CB
   100, NEMO mixing length with beta 2e4, alpha_tke 2), the kappaH_min floor
   and the linear bottom friction. IDEMIX is on in setup1.f90 and pe has no
   internal-wave module, so the interior mixing is the TKE closure above that
   floor and nothing else.

THE WAVE MODEL runs on a COARSENED grid here (EXP_WAVE_COARSEN, see the block
below). On the ocean grid its cells would be 2.22 km, the wave step 12 s and
the sub-cycle 3, which is not affordable beside an ocean that already costs
what it does. c = 3 gives a 1200 x 765 wave grid at 36 s, i.e. ONE wave step
per ocean step and no sub-cycling at all, for wave cells of 6.7 km -- still
finer than the 14.8 km the icon033 coupled test ran with.

COST, honestly. The ocean alone on 7650 ranks (90 x 85, blocks 40 x 27, 60
nodes) is about 0.6 s per step by the icon033 scaling, i.e. 26 minutes per
model DAY and six days of wall clock per model year. On 10200 ranks (120 x
85, 80 nodes) roughly 19 minutes per model day. A ten-year spinup is
therefore not a thing this configuration does: plan in model months, start
from an interpolated icon033 state rather than from rest if you can, and
measure one segment before committing an allocation.

Input files, in $EXP_INPUT (default '.'), all as the fortran setup reads
them:
    grid_topo.nc            dxt, dyt, dzt, xu, yu, kbot
    initial_conditions.nc   temp, salt (bottom first)
    ecmwf_surface_forcing.nc   dqdt
    woa_surface_forcing.nc     sss
ERA5 forcing in $EXP_FORCING_DIR: era5_<EXP_FORCING_TAG>_YYYY-MM.nc
(prep_era5_native.py), on the ERA5 grid, interpolated to each rank's block
while the run goes on.

ON THE GPUs (run_gpu.sh): the same module on the jax backend, one process
per GPU over several nodes, the split-explicit free surface
(EXP_FREE_SURFACE=explicit, see the block at FREE_SURFACE) and the per-shard
raw restart. Nothing here slices a padded array by hand: the initial
conditions go through block_io.local_block / pad_block, which is what the
MPI layout and the jax layout differ in.

The rest of this module is icon033_setup.py; see it for the reasoning behind
the forcing, the free-surface tracer flux and the coupled formulation.
"""
import os
import numpy as np
import pyOM3 as OM
from pyOM3.wim_realistic.snapshot import snap_raw as wave_base
from pyOM3.pe_realistic.snapshot  import snap_raw as ocean_base
from pyOM3.pe_realistic.tke       import tke      as tke_mixin
from netCDF4 import Dataset
import sys
from block_io   import local_block, read_scatter2, pad_block
from exp_common import (Cfg, parse_average, coarse_wave_grid, default_wave_dt,
                        wave_grid_report as _wave_grid_report,
                        spectral_grid_report as _spectral_grid_report)
from exp_ocean  import ocean_common
from exp_wave   import wave_common
from pyOM3.density import get_rho


# name of the experiment: every output file of run.py and of the coupler is
# built from it (snapshots, restarts, coupling file, energy file, means)
exp_name = 'icon01'

INPUT_DIR = os.environ.get('EXP_INPUT', '.')
GRID_FILE  = os.environ.get('EXP_GRID',  'grid_topo.nc')
IC_FILE    = os.environ.get('EXP_IC',    'initial_conditions.nc')
ECMWF_FILE = os.environ.get('EXP_ECMWF', 'ecmwf_surface_forcing.nc')   # dQ/dT
WOA_FILE   = os.environ.get('EXP_WOA',   'woa_surface_forcing.nc')    # SSS target

# ERA5 forcing: directory of the monthly files, the date of model time 0, and
# the repeat cycle in days (365 = repeat year, 0 = through the calendar)
FORCING_DIR        = os.environ.get('EXP_FORCING_DIR', '.')
FORCING_START      = os.environ.get('EXP_FORCING_START', '1990-05-01')
FORCING_CYCLE_DAYS = float(os.environ.get('EXP_FORCING_CYCLE_DAYS', '365'))
# name part of the files, era5_<tag>_YYYY-MM.nc, as given to prep_era5_native.py.
# They stay on the ERA5 grid; forcing_era5_native.py interpolates them.
FORCING_TAG        = os.environ.get('EXP_FORCING_TAG', 'icon01')
# the 10 m wind set (prep_era5_native.py --fields u10,v10 --tag <this>), needed
# by the relative-wind stress alone; empty = in the same files as the rest
FORCING_WIND_TAG   = os.environ.get('EXP_FORCING_WIND_TAG', '')
# ---------------------------------------------------------------------------
# THE SURFACE STRESS OF THE FORCED OCEAN (coupled_common/exp_ocean.py):
#   EXP_STRESS_MODE  era5     tau_ERA5 as it is (the runs so far)
#                    relwind  ERA5's drag coefficient C_D rho_a = |tau_E|/|u10|^2
#                             on the wind RELATIVE to the surface current,
#                             tau = C_D rho_a |u10 - u_o| (u10 - u_o); the
#                             interface is then a sink, (u10 - u_o).tau >= 0
#                             (energy_cycle.tex Sec. 6.2, 6.5), and coupled the
#                             waves take this stress from the ocean
#   EXP_STRESS_UO    eulerian (u_L - u_S) | lagrangian (u_L), as the coupled model
#   EXP_STRESS_U10_MIN  floor of |u10| in the denominator [m/s]           (1.0)
#   EXP_STRESS_CD_MAX   cap on C_D: the stress is the hourly mean, the wind the
#                       analysis at the stamp, so the quotient can explode (3e-3)
# ---------------------------------------------------------------------------
STRESS_MODE    = os.environ.get('EXP_STRESS_MODE', 'era5')
STRESS_UO      = os.environ.get('EXP_STRESS_UO', 'eulerian')
STRESS_U10_MIN = float(os.environ.get('EXP_STRESS_U10_MIN', '1.0'))
STRESS_CD_MAX  = float(os.environ.get('EXP_STRESS_CD_MAX', '3.0e-3'))   # cap on C_D (ERA5 hourly stress vs instantaneous wind)
if STRESS_MODE not in ('era5', 'relwind'):
    raise ValueError('EXP_STRESS_MODE must be era5 or relwind, not %r' % STRESS_MODE)
# Constant [W/m2] subtracted from the ERA5 net heat flux. ERA5 runs on a
# PRESCRIBED sea surface temperature, so its surface energy budget is free and
# does not close: over this grid the annual mean of sshf+slhf+ssr+str comes to
# +14.3 W/m2 into the ocean, where the real ocean absorbs about 0.5 W/m2. The
# model has no way to export that heat, so the SST restoring has to carry it
# back out, which it can only do by holding a global warm bias of
# offset/(dQ/dT) -- about +0.5 K here. Subtracting the measured offset removes
# the imbalance and leaves the restoring to work on the circulation error
# alone. It is deliberately NOT a default: measure it for the period the run
# uses (area weighted over the wet cells) and set it. The penetrating solar
# qsol is not touched, so the shortwave profile is unchanged.
QNET_OFFSET        = float(os.environ.get('EXP_QNET_OFFSET', '0.0'))
# Light that reaches the sea floor. divpen deposits the fraction pen(z_bottom)
# of the net solar in cells below the bottom, where the mask drops it, so a
# shallow column LOSES that heat (setup1.f90 does the same: the pen(nz) = 0
# compensation exists at the surface and has no counterpart at the bottom).
# On this grid the loss is 0.14 % of the net solar over the whole ocean but
# 2.8 % on a 62 m shelf column, always a cooling. Switched on, the remainder
# is absorbed in the DEEPEST WET CELL of each column instead, which is what
# the seabed does with it. Default off, so the runs made so far reproduce.
SW_BOTTOM          = os.environ.get('EXP_SW_BOTTOM', '0') not in ('0', '', 'no')
# FRESHWATER FLUX. Without it the only thing acting on the surface salinity is
# the restoring to the WOA climatology, which is a strong constraint and the
# one thing standing between the run and a wrong halocline. With it the ERA5
# evaporation and precipitation drive the salinity as a virtual salt flux
# S dz/dt = S (E - P), and the restoring can be weakened to what it is really
# for: there is no river runoff and no sea ice in this configuration, so the
# freshwater budget of the ocean is incomplete on its own.
#   EXP_FRESHWATER     1 turns it on (default off). e and tp are in the SAME
#                      files as the six flux fields and sst, so nothing else
#                      has to be prepared or pointed at -- prep_era5_native.py
#                      writes all nine when it is given them.
#   EXP_EMP_OFFSET     constant [mm/day] subtracted from P - E, the analogue of
#                      EXP_QNET_OFFSET. Over the ocean evaporation exceeds
#                      precipitation and the difference is returned by the
#                      rivers, which this configuration does not have, so the
#                      net of P - E over the wet cells is NOT zero and the
#                      salinity would drift. Measure it over the period the run
#                      uses and set it.
#   EXP_SSS_PISTON     restoring strength as a PISTON VELOCITY [m/day], which
#                      unlike a relaxation time does not depend on the top cell
#                      thickness. 0 keeps setup1.f90's dz/(30 days), which is
#                      0.068 m/day with the 2.05 m top cell of this grid --
#                      five times weaker than the 4 degree runs' 0.367 m/day
#                      at the same 30 days, because the relaxation time is
#                      converted with the top cell thickness. CORE and OMIP
#                      practice is around 0.17 m/day.
FRESHWATER         = os.environ.get('EXP_FRESHWATER', '0') not in ('0', '', 'no')
EMP_OFFSET         = float(os.environ.get('EXP_EMP_OFFSET', '0.0'))/86400.0/1000.0
SSS_PISTON         = float(os.environ.get('EXP_SSS_PISTON', '0.0'))/86400.0

# ---------------------------------------------------------------------------
# SURFACE TKE FLUX OF THE OCEAN ALONE (EXP_ALPHA_CB, default 100)
#     F = alpha_CB |tau_a/rho_0|^{3/2} = alpha_CB u*_w^3      [m^3/s^3]
# alpha_CB = 1 is the pyOM2 boundary condition, ~100 the breaking-wave flux of
# Craig and Banner (1994) (NEMO's rn_ebb = 67.83). With the wave model the flux
# is phi_oc and alpha_CB is not used. See ocean._surface_forcing.
# ---------------------------------------------------------------------------
ALPHA_CB = float(os.environ.get('EXP_ALPHA_CB', '100'))

# ---------------------------------------------------------------------------
# TKE CLOSURE PARAMETERS OF THE ENSEMBLE
#   EXP_ALPHA_TKE        TKE diffusivity alpha_tke * kappaM      (default 30)
#   EXP_MXL_SURF_FACTOR  mixing length at the surface W point in units of
#                            the top cell thickness dzt(top)          (default 1)
# pe/pyOM2 limit the surface mixing length to mxl_min + z0_surf + dzt(top)
# after the downward sweep (tke.f90 order, unchanged). The factor f sets
# z0_surf = (f - 1) dzt(top), so the limit is mxl_min + f dzt(top); f = 1 is
# the pyOM2 behaviour.
# ---------------------------------------------------------------------------
ALPHA_TKE = float(os.environ.get('EXP_ALPHA_TKE', '2'))
# mixing length: 1 = pyOM2 tke_mxl_choice 2 (default; EXP_MXL_SURF_FACTOR
# applies), 2 = NEMO nn_mxl 3 with the Charnock surface length
MXL_CHOICE = int(os.environ.get('EXP_MXL_CHOICE', '2'))
MXL_SURF_FACTOR = float(os.environ.get('EXP_MXL_SURF_FACTOR', '1'))
# mxl_choice = 2 only: Charnock coefficient of the surface length
# l0 = kappa beta |tau|/(g rho_0). NEMO uses beta = 2e5 (Stacey 1999), which
# gives 0.8 m at |tau| = 0.1 Pa; beta = 2.5e6 gives 10 m there, the value the
# pyOM2 limits use at this vertical resolution.
MXL_BETA = float(os.environ.get('EXP_MXL_BETA', '2e4'))
# COUPLED runs with mxl_choice = 2: the surface mixing length comes from the
# wave model's significant wave height instead of the stress,
#     z0 = HS_FACTOR * Hs ,  l0 = kappa z0      (NEMO ln_mxhsw, z0 = 1.6 Hs)
# 0 keeps the Charnock form. Only used when the coupler delivers hs.
HS_FACTOR = float(os.environ.get('EXP_HS_FACTOR', '1.6'))
# restrict the online means to these fields (comma list); empty = all
AVERAGE_FIELDS = [f.strip() for f in os.environ.get('EXP_AVERAGE_FIELDS', '').split(',') if f.strip()]


VADV_SUB_CYCLING = int(os.environ.get('EXP_VADV_SUB_CYCLING', '0'))

# ---------------------------------------------------------------------------
# THE FREE SURFACE, EXP_FREE_SURFACE = implicit (default, the MPI runs) |
# explicit, as in icon033_setup.py: the linear implicit free surface with the
# CG, or the SPLIT-EXPLICIT sub cycle (free_surface.py, Demange et al. 2019),
# which has no global reduction -- on the GPUs the CG was a quarter of the
# step at 1/3 degree and its iterations are latency bound.
#   EXP_SUB_CYCLING  M. c = sqrt(g H) = 245 m/s at 6100 m and the narrowest
#                    cell is 2.22 km, so c dt/dx = 245 x 144/2220 = 15.9 at
#                    dt = 144 s (4.0 at 36 s). For a sub-step CFL of ~0.3
#                    that is M = 48 at 144 s (default), 12 at 36 s; icon033
#                    runs 24 at c dt/dx = 7.1.
#   EXP_FB_THETA     dissipation of the forward-backward sub cycle, 0.5 as
#                    icon033 (0.14 left grid-scale gravity waves at this
#                    resolution, free_surface.py).
# A run keeps its method from the cold start (coupler_io refuses the other
# method's restart); ocean alone only; EXP_KE_BUDGET must be 0 with it.
# ---------------------------------------------------------------------------
FREE_SURFACE = os.environ.get('EXP_FREE_SURFACE', 'implicit').strip().lower()
if FREE_SURFACE not in ('implicit', 'explicit'):
    raise ValueError('EXP_FREE_SURFACE must be implicit or explicit, not %r' % FREE_SURFACE)
SUB_CYCLING = int(os.environ.get('EXP_SUB_CYCLING', '48'))
FB_THETA    = float(os.environ.get('EXP_FB_THETA', '0.5'))

# ---------------------------------------------------------------------------
# grid, read once from GRID_FILE (identical for both models -- the coupler
# exchanges full-domain fields on the shared horizontal grid). dxt/dyt are in
# degrees and dyt is variable in y; dzt is BOTTOM FIRST; kbot is the pyOM2
# 1-based index of the deepest wet level, 0 on land. The coordinates
# themselves are NOT read: both models build them from dxt/dyt and the first
# U point with the pyOM2 u-centred recurrence, and on a stretched grid that
# differs from the file's own yt by the rounding of its float32 spacings.
# ---------------------------------------------------------------------------
def _read_grid():
    with Dataset(os.path.join(INPUT_DIR, GRID_FILE)) as f:
        dxt = np.asarray(f.variables['dxt'][:], np.float64)
        dyt = np.asarray(f.variables['dyt'][:], np.float64)
        dzt = np.asarray(f.variables['dzt'][:], np.float64)
        x0  = float(np.asarray(f.variables['xu'][0]))
        y0  = float(np.asarray(f.variables['yu'][0]))
        kb  = np.asarray(f.variables['kbot'][:], np.int64)
    if kb.shape != (len(dyt), len(dxt)):
        raise ValueError('%s: kbot is %s, the grid is (%d, %d)'
                         % (GRID_FILE, kb.shape, len(dyt), len(dxt)))
    if kb.max() > len(dzt) or kb.min() < 0:
        raise ValueError('%s: kbot spans %d .. %d, the column has %d levels'
                         % (GRID_FILE, kb.min(), kb.max(), len(dzt)))
    if dzt.min() <= 0.0 or dxt.min() <= 0.0 or dyt.min() <= 0.0:
        raise ValueError('%s: non-positive grid spacing' % GRID_FILE)
    return dxt, dyt, dzt, x0, y0, kb

dxt_deg, dyt_deg, _dzt_gl, x_origin, y_origin, _kbot_gl = _read_grid()
nx, ny, nz = len(dxt_deg), len(dyt_deg), len(_dzt_gl)

radius  = 6370e3
degtom  = radius/180.0*np.pi
omega_e = np.pi/43082.0                 # as in the fortran setup
rho_0   = 1024.0
rho_air = 1.225
cp_0    = 3991.86795711963
t_rest  = 30*86400.0

# ---------------------------------------------------------------------------
# run control
# ---------------------------------------------------------------------------
# THE TIME STEP, 36 s, is setup1.f90's dt_mom = 144.0/4 and it is the whole
# cost of this configuration. It is NOT set by anything horizontal: POP runs
# the same 1/10 degree spacing (3600 x 2163, narrowest cell 2234 m against
# 2220 m here) at 216 s. At 3 m/s over 2.22 km the advective CFL is 0.29 at
# 216 s and 0.05 at 36 s, and the biharmonic stability number S (see
# get_ocean_parameter) is 0.010 at 36 s and would be 0.061 at 216 s. What
# differs from POP is the VERTICAL grid: 128 levels with 2.05, 2.15, 2.25 m
# at the top, 25 m of water in the first ten cells, against POP's 42 levels
# and a top cell about five times thicker. Two things scale with that -- the
# vertical advective CFL w dt/dz, which at w = 5 mm/s in the top cell is 0.53
# at 216 s and 0.09 at 36 s, and everything that divides a surface flux by
# the top cell thickness. The fortran line reads 144.0/4, i.e. the value was
# walked down rather than computed, so 36 s may well be conservative by a
# factor of two or three. It costs six days of wall clock per model year;
# 108 s would cost two. TEST IT (EXP_OCEAN_DT=72 or 108 over a few hundred
# steps from a cold start, watching the canary and the identity line) before
# committing an allocation. Coupled, the wave step follows: at 108 s with
# c = 3 the default wave step is 36 s and the sub-cycle 3, so only the ocean
# gets cheaper.
ocean_dt = float(os.environ.get('EXP_OCEAN_DT', '36.0'))

# ---------------------------------------------------------------------------
# A COARSER WAVE GRID, EXP_WAVE_COARSEN = c (1 = the ocean grid, as before).
# The wave grid is the ocean grid with c x c cells merged: its cell faces ARE
# ocean cell faces (coarse_edges; the remainder of n/c is absorbed by the last
# group, here the northernmost row: 2295 = 3 x 765 exactly, but
# 2295 = 2*1147 + 1 = 4*573 + 3),
# so the mapping between the two grids is an index map and nothing is
# interpolated. dxt and dyt of a wave cell are the sums over its group and the
# first U point is the last U point of the first group; the wave model then
# builds its own coordinates and metric from them as it always does.
#   c      wave grid      wet cells   narrowest   wave dt   sub-cycle
#   1     3600 x 2295     5,153,349      2.2 km      12 s          3
#   2     1800 x 1147     1,294,717      4.4 km      18 s          2
#   3     1200 x  765       578,167      6.7 km      36 s          1
#   4      900 x  573       326,468      8.9 km      36 s          1
# The wave step scales with the narrowest cell, c x 2.22 km, but only
# divisors of the 36 s ocean step exist (36, 18, 12, 9, 6, 4, 3, 2, 1), so
# from c = 3 the wave model steps ONCE per ocean step and there is no
# sub-cycling left to save. The default below is the largest divisor not
# above 12 c seconds, which keeps the CFL at or below 0.115 against the
# certified 0.125. c = 3 is the intended setting: 2295 = 3 x 765 and
# 3600 = 3 x 1200 divide exactly, so no wave row or column is ragged.
# WHAT THE EXCHANGE DOES with it is in coupler_mpi.py (BlockExchange); what
# this module does is below: wave.set_topography (the coarse land mask) and
# wave.set_initial / update_forcing (the wind as the group mean of tau_a).
# ---------------------------------------------------------------------------
wave_coarsen = int(os.environ.get('EXP_WAVE_COARSEN', '1'))
_wg = coarse_wave_grid(dxt_deg, dyt_deg, x_origin, y_origin, wave_coarsen)
_je_w, _ie_w     = _wg['je'], _wg['ie']
ny_wave, nx_wave = _wg['ny'], _wg['nx']
dxt_wave, dyt_wave = _wg['dxt'], _wg['dyt']
x_origin_wave, y_origin_wave = _wg['x_origin'], _wg['y_origin']

# default wave step: the largest divisor of the ocean step not above 12 c s
wave_dt  = float(os.environ.get('EXP_WAVE_DT',
                                '%g' % default_wave_dt(ocean_dt, 12.0, wave_coarsen)))


def wave_grid_report():
    return _wave_grid_report(_wg, wave_coarsen, f0, wave_dt, radius)


def spectral_grid_report():
    return _spectral_grid_report(nf, nd, f0, df0)


# ---------------------------------------------------------------------------
# SURFACE FORCING: which source, and its settings (forcing_common). To force
# this experiment with something else, change 'source' (EXP_FORCING) and give
# that source its settings here. Nothing in the model classes changes.
# ---------------------------------------------------------------------------
def forcing_settings():
    src = os.environ.get('EXP_FORCING', 'era5')
    if src == 'era5':
        return {'source': 'era5', 'grid': 'native',
                'dir': FORCING_DIR, 'tag': FORCING_TAG, 'wind_tag': FORCING_WIND_TAG or None,
                'start': FORCING_START, 'cycle_days': FORCING_CYCLE_DAYS,
                'qnet_offset': QNET_OFFSET, 'emp_offset': EMP_OFFSET,
                'freshwater': FRESHWATER,
                'clim': {'dqdt': (os.path.join(INPUT_DIR, ECMWF_FILE), 'dqdt'),
                         'sss':  (os.path.join(INPUT_DIR, WOA_FILE),   'sss')},
                'clim_scatter': True}
    if src == 'ecmwf_clim':          # the monthly climatology of setup1.f90
        return {'source': 'ecmwf_clim',
                'file': os.path.join(INPUT_DIR, ECMWF_FILE),
                'sss': (os.path.join(INPUT_DIR, WOA_FILE), 'sss'),
                'start': FORCING_START, 'cycle_days': 365.0,
                'clim_scatter': True}
    raise ValueError('EXP_FORCING=%s: no settings for that source in %s'
                     % (src, __name__))


# coupling happens once per ocean step; the wave model sub-cycles
# ocean_dt/wave_dt times (checked at the dt handshake in run.py), which is 1
# at the intended c = 3. Only the divisors of the 36 s ocean step exist:
# 36, 18, 12, 9, 6, 4, 3, 2, 1 s.
#
# WAVE CFL: c_g,max = g/(4 pi f0) = 21.2 m/s. This grid is near isotropic in
# the sense that dyt follows dxt cos(lat), so the narrowest cell is 2.22 km
# in BOTH directions at the southern edge, and the wave grid coarsened by c
# is c times that. The certified unsplit limit of the superbee scheme is
# 0.125; the defaults above sit at 0.115 or below.
#
# WHY NOT THE ECMWF STEP. ECWAM runs its global wave grid at 12 minutes. Two
# things buy that and neither is available here as the model stands: its
# irregular spherical grid keeps the zonal spacing near its equatorial value
# instead of letting it collapse as cos(lat); and it advects with first-order
# upwinding, which tolerates a CFL of order 1 where this superbee scheme is
# certified to 0.125. wim HAS an upwind option (f_limiter = 'upwind'), so the
# second factor is reachable -- but its CFL limit has not been measured, and
# the upwind branch currently bypasses the Fortran kernels. Measure before
# changing this. At 1/10 degree this is the lever that matters most.

snapint_ocean = float(os.environ.get('EXP_SNAPINT_OCEAN_DAYS', '30'))*86400.0

average_mode = parse_average(os.environ.get('EXP_AVERAGE', '0'))
snapint_wave  = float(os.environ.get('EXP_SNAPINT_WAVE_DAYS', '1'))*86400.0

# WAVE SNAPSHOT CONTENT. The wave model writes RAW per-PE blocks here
# (wim_realistic snap_raw), because snap_nc4 walks the PEs one at a time
# through one shared file and writes five 4-D fields -- 20 GB per record on
# the ocean grid, 2.2 GB on the c = 3 wave grid, which takes longer than the
# model run.
#   EXP_SNAP_WAVE_FIELDS  mean | spectra | full | an explicit comma list
#     mean     E_mean, om_mean, Hs                         11 MB per record
#     spectra  + E_f (integrated over direction) and
#              E_d (over frequency)                       184 MB   (default)
#     full     + the 4-D spectrum E                       2.2 GB
#   (on the c = 3 wave grid; 9 times those numbers on the ocean grid)
#     a list may name any 2-D or 4-D model attribute, ustar and the ray
#     rates xdot/ydot/fdot/ddot included
#   EXP_SNAPINT_WAVE_FULL_DAYS  interval for the 4-D E on its own, so the
#     marginals can come every 30 days and the full spectrum a few times a
#     year (0 = never, and 'full' above makes it every snapshot)
_WAVE_FIELD_SETS = {
    'mean':    ('E_mean', 'om_mean', 'Hs'),
    'spectra': ('E_mean', 'om_mean', 'Hs', 'E_f', 'E_d'),
    'full':    ('E_mean', 'om_mean', 'Hs', 'E_f', 'E_d', 'E'),
}
_wf = os.environ.get('EXP_SNAP_WAVE_FIELDS', 'spectra').strip()
snap_wave_fields = _WAVE_FIELD_SETS.get(_wf, tuple(
    f.strip() for f in _wf.split(',') if f.strip()))
snapint_wave_full = float(os.environ.get('EXP_SNAPINT_WAVE_FULL_DAYS', '0'))*86400.0

# KE budget (ocean.ke_budget): OFF by default here. It re-evaluates every
# momentum tendency from the state at every step -- advection fluxes,
# Coriolis, the hydrostatic pressure -- plus about 25 global reductions, and
# that is roughly 40 % on top of an ocean step at this size. Switch it on to
# check the discrete identities, not for production.
# 0 off, 1 the full budget with every identity check, 2 the LIGHT form
# (exp_ocean.ke_budget): the pathways and the leak alone, a tenth of that
KE_BUDGET = int(os.environ.get('EXP_KE_BUDGET', '0') or 0)
# switches of the shared classes (exp_common.Cfg): raw per-rank snapshots; the
# online means beyond the standard list (see icon033_setup.py)
RAW_SNAPSHOTS = True
AVERAGE_MLD         = os.environ.get('EXP_AVERAGE_MLD', '0') not in ('0', '', 'no')
AVERAGE_EDDY        = os.environ.get('EXP_AVERAGE_EDDY', '0') not in ('0', '', 'no')
AVERAGE_TRACER_FLUX = os.environ.get('EXP_AVERAGE_TRACER_FLUX', '0') not in ('0', '', 'no')

# ---------------------------------------------------------------------------
# spectral grid
# ---------------------------------------------------------------------------
# The frequency axis is GEOMETRIC by construction: f[n] = f0*(1+df0)^n, built
# by wim's set_grid from f0 and df0 alone. An arbitrary dft(f) is NOT a usable
# option: the DIA (Snonlin) derives its whole weight table from the single
# ratio co = f[1]/f[0] (snonlin_np._nlweigt), so a non-geometric axis would be
# silently misinterpreted rather than rejected. The three free knobs are
# therefore nf, df0 and f0 -- all settable here or from the environment.
#
# WHAT df0 COSTS. The DIA places its resonant lobes at f(1 +- 0.25) and
# converts that to an INTEGER bin shift, ISP = int(log10(1.25)/log10(1+df0)).
# ISP = 2 for df0 <= 0.1175 (the stencil the source terms are calibrated
# with), ISP = 1 from df0 = 0.12, and ISP = 0 from df0 = 0.29 -- where the
# up-shifted lobe lands on the centre bin and the DIA stops transferring
# upward at all. See the sweep table in README.md for the measured cost.
#
# WHAT nf COSTS. Only through the top of the grid: f_max = f0*(1+df0)^(nf-1)
# must stay near 1 Hz. Below it the imphftail cut-off max(2.5 f_mean, 3 f_pm)
# never falls inside the grid at moderate winds, the f^-5 tail is never
# imposed, the ghost anchor stays zero, and BOTH the resolved and the analytic
# tail input vanish -- the run completes with wind_work identically zero.
# Setting EXP_FMAX (default: keep it at the reference 0.9841 Hz) solves
# f0 for you so that reducing nf raises df0 rather than truncating the grid.

# nf=30 nd=20 df0=0.12 was the measured sweet spot.
# nf=16 nd=8 df0=0.25 is smallest possible
nf  = int(os.environ.get('EXP_NF',  '30'))
nd  = int(os.environ.get('EXP_ND',  '20'))
df0 = float(os.environ.get('EXP_DF0', '0.12'))
f_max_target = float(os.environ.get('EXP_FMAX', '0.9841'))
if f_max_target > 0:
    f0 = f_max_target/(1.0 + df0)**(nf - 1)
else:                                   # explicit low-frequency end instead
    f0 = float(os.environ.get('EXP_F0', '0.035'))
snonlin_backend = os.environ.get('EXP_SNONLIN', 'fortran')


if int(os.environ.get('EXP_DF0_GUARD', '1')) and (1.0 + df0) > 1.25 + 1e-12:
    raise ValueError(
        'df0 = %.3f puts the DIA up-shift lobe on the centre bin (ISP = 0, '
        'which happens for df0 > 0.25): the nonlinear transfer stops '
        'cascading upward. Use df0 <= 0.25, and preferably <= 0.1175 where '
        'ISP = 2 as calibrated. Set EXP_DF0_GUARD=0 to override '
        'deliberately.' % df0)

# wave -> ocean momentum transfer: 'stress' (2-D surface) or 'stress_work'
# (3-D, depth distributed). 'stress' as in acc1.
transfer_mode   = 'stress'
use_stress_work = (transfer_mode == 'stress_work')

# hand the direct (non-wave-supported) branch tau_a - tau_in to the ocean as
# well?  See the module docstring. OFF = acc1 behaviour.
direct_stress_branch = bool(int(os.environ.get('EXP_DIRECT_BRANCH', '0')))

# RAW wave snapshots: a directory per segment, as for the ocean. run.py
# splices the segment tag in front of '.cdf' and wave.setup strips it.
wave_snap_file  = 'raw_wave_%s.cdf'  % exp_name
# RAW ocean snapshots: run.py turns this into ocean_snap_file with the segment
# tag spliced in front of '.cdf', and ocean.prepare_for_run strips the '.cdf'
# again, so each segment writes into its own directory raw_ocean_<exp><seg>/
ocean_snap_file = 'raw_ocean_%s.cdf' % exp_name


# ===========================================================================
# input files
# ===========================================================================
def dzt_global():
    """T-cell thicknesses, BOTTOM FIRST, as the grid file gives them."""
    return _dzt_gl


def zw_pyom2():
    """pyOM2-convention face depths (negative, zero at the surface), used for
    the shortwave penetration profile."""
    d = dzt_global()
    return np.cumsum(d) - d.sum()


def kbot_global():
    """Global kbot(ny,nx) from the grid file. kbot = 0 is land, kbot = k means
    the bottom is below the k-th model level counted from the BOTTOM, 1-based
    -- the pyOM2 convention, which is also pe_realistic's PADDED index at
    halo = 1, so nothing is remapped here (set_topography shifts it for a
    wider halo). The depth-to-level mapping and the pothole sweep that
    global4 did on its raw bathymetry are already in this file."""
    return _kbot_gl


# ===========================================================================
# WAVE model: wim_realistic on the coarsened pseudo-spherical grid
# ===========================================================================
class wave(wave_common, wave_base):
    """The wind forcing is the hourly ERA5 turbulent surface stress, the same
    tau_a the ocean's direct branch uses, converted to an air-side friction
    velocity
        u*_air = sqrt(|tau| / rho_air)
    (tau in N/m^2; the ocean divides it by rho_0 for its own kinematic
    stress). The direction convention is wim's: theta = 0 -> north,
    theta = pi/2 -> east, so udir = atan2(tau_east, tau_north).

    The stress sits on T points here. The ocean averages the same T-point
    field to its U/V points, with the operator it applies to tau_in.

    Everything is in coupled_common/exp_wave.py (wave_common)."""
    cfg = Cfg(sys.modules[__name__])


def get_wave_parameter():
    # the wave grid: the ocean grid for EXP_WAVE_COARSEN = 1 (the reduceat of
    # single cells returns dxt_deg and dyt_deg themselves, the origins are
    # unchanged), its coarsening otherwise
    return {'nx': nx_wave, 'ny': ny_wave, 'nf': nf, 'nd': nd,
            'dxt': dxt_wave, 'dyt': dyt_wave, 'coord_degree': True,
            'x_origin': x_origin_wave, 'y_origin': y_origin_wave,
            'dt': wave_dt, 'f0': f0, 'df0': df0,
            'periodic_in_x': True, 'periodic_in_y': False,
            'mask_energy': True, 'limit_growth': True,
            'snonlin_backend': snonlin_backend,
            'snapint': snapint_wave,'input_scheme':'janssen',
            'snap_fields': snap_wave_fields,
            'snapint_full': snapint_wave_full}


# ===========================================================================
# OCEAN model: pe_realistic global 1/10 deg, wind delivered by the waves
#              when coupled, by ERA5 when alone
# ===========================================================================
class ocean(ocean_common, tke_mixin, ocean_base):
    # forcing, coupling terms, TKE hooks, energy diagnostics and online means
    # are in coupled_common/exp_ocean.py (ocean_common). Here: the snapshot
    # form, the initial conditions and the friction of THIS configuration.
    cfg = Cfg(sys.modules[__name__])

    # paper Sec. 5: the TKE closure must mix the EULERIAN velocity
    # u_bar = u_L - u_S, not the model velocity u_L.
    mix_eulerian = True

    # False = ocean alone (EXP_WAVES=0, set by run.py): tau_a is the
    # surface stress, no vortex force, no direct branch
    coupled = True


    def prepare_for_run(self, snap_file_name='raw_snap',
                        show_each_time_step=True):
        """RAW snapshots (snap_raw): every rank writes its own block as .npy
        into <dir>/pe=<rank>/n=<step>/. run.py hands a name ending in '.cdf'
        (it splices the segment tag in front of it); the suffix is stripped
        here, so each segment gets its own directory.

        The parent creates <dir>/pe=<rank> for EVERY rank on rank 0 and the
        other ranks mkdir inside them at the first diagnose, so a barrier
        must separate the two."""
        if snap_file_name.endswith('.cdf'):
            snap_file_name = snap_file_name[:-4]
        super().prepare_for_run(snap_file_name, show_each_time_step)
        self._ke_vfric = self._ke_vnum = self._ke_hfric = 0.0
        self.mpi_barrier()
        return


    # --- initial conditions, LEVEL BY LEVEL ---------------------------------
    # icon15 builds the full (nz, ny, nx) temp and salt on every rank and
    # hands them to the package's set_initial_conditions, which slices out
    # each rank's part. Here that is 762 MB per field per rank. Instead rank 0
    # reads one level at a time and scatters the blocks, so make_initial_fields
    # returns nothing and the state is installed by set_initial_conditions
    # below; run.py's call sequence is unchanged.
    def make_initial_fields(self):
        """Read temp and salt of IC_FILE level by level on rank 0 and keep
        THIS rank's block. The file is (nz, ny, nx) and BOTTOM FIRST, the
        model's own order. Dry cells are 0 in the file make_initial_conditions.py
        writes and -1e33 in the one the fortran setup used; both are replaced
        here, and the masks decide in any case."""
        comm = getattr(self, 'comm', None)
        if comm is None or comm.Get_rank() == 0:
            with Dataset(os.path.join(INPUT_DIR, IC_FILE)) as f:
                sh = f.variables['temp'].shape[-3:]
            if tuple(sh) != (nz, ny, nx):
                raise ValueError('%s: temp is %s, the grid is (%d, %d, %d)'
                                 % (IC_FILE, tuple(sh), nz, ny, nx))
        # THIS PE's block: under MPI the local block, on the jax backend the
        # whole grid (block_io.local_block). On jax every process holds the
        # global field, so it is kept in the file's own precision, float32:
        # 4.25 GB per field at this size instead of 8.5 (the MPI blocks stay
        # float64 as before)
        nyl, nxl = local_block(self)[2:]
        dt_ = np.float32 if OM.jax else np.float64
        T0 = np.empty((nz, nyl, nxl), dt_); S0 = np.empty((nz, nyl, nxl), dt_)
        for k in range(nz):
            T0[k] = read_scatter2(self, os.path.join(INPUT_DIR, IC_FILE), 'temp', (k,))
            S0[k] = read_scatter2(self, os.path.join(INPUT_DIR, IC_FILE), 'salt', (k,))
        self._T0 = np.where(np.isfinite(T0) & (T0 > -100.0), T0, dt_(0.0))
        self._S0 = np.where(np.isfinite(S0) & (S0 > 0.0),    S0, dt_(35.0))
        if self.my_pe == 0:
            print('initial conditions read from %s' % IC_FILE, flush=True)
        return None, None, None, None, None

    def set_initial_conditions(self, _u, _v, _temp, _salt, _p_s):
        """Install the blocks make_initial_fields kept. Same content as the
        package's set_initial_conditions_mpi, without the full-domain arrays:
        velocities and surface pressure start from rest, w and rho follow from
        the state. The arguments are the five None values above.

        The blocks become padded fields through block_io.pad_block: zeros +
        [h:-h] slices + apply_bc under MPI, the shard layout on jax (each
        process its own shards from the global field)."""
        self.u = self.apply_bc(OM.np.zeros((self.Nz, self.Ny, self.Nx), np.float64))
        self.v = self.apply_bc(OM.np.zeros((self.Nz, self.Ny, self.Nx), np.float64))
        self.w = self.vertical_velocity(self.u, self.v, self.w, self.maskW)
        self.temp = pad_block(self, self._T0)
        self.salt = pad_block(self, self._S0)
        del self._T0, self._S0
        self.rho = get_rho(self.salt, self.temp, self.p0, self.eq_of_state)
        if self.implicit_free_surface:
            for it in range(3):
                self.p_s = OM.modify_array(self.p_s, (it, slice(None), slice(None)),
                                           OM.np.zeros((self.Ny, self.Nx), np.float64))
        if self.explicit_free_surface:
            self.p_s = self.p_s*0.0          # keeps the package's (sharded) array
        return


    # --- d(u_S)/dt forcing of the u_L formulation (direct, non-AB) ----------
    # ------------------------------------------------------------------
    # KINETIC ENERGY BUDGET of u_L, the ocean counterpart of the atmosphere's
    # energy_budget in hs_setup.py (KE side only; the PE side is deferred).
    # Every tendency term is re-evaluated from the CURRENT state at the end of
    # the step, exactly as the model forms it, dotted with u_L and integrated;
    # the friction terms and the coupling terms are taken from their exact
    # bookings at the moment of application. Accumulated over eb_every_days,
    # then the spatial sum dE_sp is compared with the finite-difference change
    # of the reservoirs dE_fd; LEAK = dE_fd - dE_sp is the time-stepping error
    # (AB weighting, operator splitting). Units W/m^2 (rho_0 included), per
    # unit wet ocean area. Printed on rank 0 and appended to <exp_name>_energy.txt.
    # ------------------------------------------------------------------
    KE_COLUMNS = ('day', 'KE_adv', 'KE_cor', 'KE_ph', 'KE_ps', 'W_tau', 'W_dir', 'W_vortex',
                  'C', 'D_h', 'D_v', 'D_v_num', 'K_diss', 'PEs_dot', 'dE_sp', 'dE_fd', 'LEAK',
                  'KE', 'PEs', 'KE_div', 'D_bot')
    KE_LEGEND = ('KE budget legend [W/m2]: KE_adv u.(advection) | KE_cor u.(Coriolis) | '
                 'KE_ph u.(-grad p_h) = conversion | KE_ps u.(-grad p_s) | W_tau wave stress '
                 'work | W_dir direct-branch work | W_vortex vortex-force work (round-off) | '
                 'C exchange int u_L.d(u_S)/dt | D_h horizontal friction | D_bot bottom friction | '
                 'D_v vertical friction '
                 '= -K_diss - D_v_num | dE_sp sum of terms | dE_fd (E(t1)-E(t0))/T | '
                 'LEAK = dE_fd - dE_sp (time stepping)')
    eb_every_days = 1.0


    # --- biharmonic friction and its energetics -----------------------------
    # THE IDENTITY. The package applies u <- u - dt maskU L(L(u)), where L is
    # the single Laplacian pass below with visc = sqrt(A_hbi) already inside
    # its fluxes. Written with the U-cell area weight, L is SELF-ADJOINT:
    # summation by parts turns sum u L(w) area dz into
    #   -sum [ fe(w) (u(i+1) - u(i)) dyt + fn(w) (u(j+1) - u(j)) dxu ] dz ,
    # and fe(w) (u(i+1) - u(i)) = fe(u) (w(i+1) - w(i)) because the two share
    # the same coefficient visc fxa/(cost dxt(i+1)) times the mask pair. So
    #   sum u L(L(u)) area dz = sum L(u) L(u) area dz ,
    # and the work of the friction on u is
    #   D_h,stencil = -sum |L(u)|^2 area dz   (negative definite)
    # plus the forward-Euler part dt/2 sum |F|^2 of applying it explicitly.
    # The check below recomputes L(u) here and compares the two sides. It
    # therefore also checks that this copy of the operator still matches the
    # package's; if damping.py changes, the residual column says so.
    def _biharm_del2(self, w, Ahbi, mask, kind):
        """One Laplacian pass of pe_realistic/damping.py biharmonic_momentum,
        for the u rows (kind='u') or the v rows (kind='v')."""
        p    = getattr(self, 'biha_friction_cosPower', 0.0)
        visc = OM.np.sqrt(OM.np.abs(Ahbi))
        if kind == 'u':
            dxtp1 = OM.np.roll(self.dxt, -1)[None, None, :]
            fe = visc*self.cost_y**p*(self.rollx(w, -1) - w)/(self.cost_y*dxtp1) \
                 * mask*self.rollx(mask, -1)
            fn = visc*self.cosu_y**p*(self.rolly(w, -1) - w)/self.dyu_y*self.cosu_y \
                 * mask*self.rolly(mask, -1)
            fe, fn = self.apply_bc(fe), self.apply_bc(fn)
            return ((fe - self.rollx(fe, 1))/(self.cost_y*self.dxu_x)
                    + (fn - self.rolly(fn, 1))/(self.cost_y*self.dyt_y))
        dytp1  = OM.np.roll(self.dyt, -1)[None, :, None]
        costp1 = OM.np.roll(self.cost, -1)[None, :, None]
        fe = visc*self.cosu_y**p*(self.rollx(w, -1) - w)/(self.cosu_y*self.dxu_x) \
             * mask*self.rollx(mask, -1)
        fn = visc*costp1**p*(self.rolly(w, -1) - w)/dytp1*costp1 \
             * mask*self.rolly(mask, -1)
        fe, fn = self.apply_bc(fe), self.apply_bc(fn)
        return ((fe - self.rollx(fe, 1))/(self.cosu_y*self.dxt_x)
                + (fn - self.rolly(fn, 1))/(self.dyu_y*self.cosu_y))

    def biharmonic_momentum(self, u, v, Ahbi, maskU, maskV):
        """Book the exact KE change of the BIHARMONIC friction (A_hbi = 2e11,
        cos^1.5 per Laplacian pass, as setup1.f90), its adjoint identity
        -sum |L(u)|^2 and the forward-Euler part dt/2 |F|^2, at the moment it
        is applied. This configuration has no harmonic friction, so D_h in the
        KE budget is this term. The adjoint part is only recomputed when the
        KE budget is on; it costs one Laplacian pass per component."""
        un, vn = super().biharmonic_momentum(u, v, Ahbi, maskU, maskV)
        if KE_BUDGET:
            self._ke_hfric = self._ke_of(un, vn) - self._ke_of(u, v)
            dt = self.dt
            self._ke_hnum = dt*self._ke_of((un - u)/dt, (vn - v)/dt)
        if KE_BUDGET == 1:                      # the stencil identity, two Laplacian passes
            h = self.halo; sl = slice(h, -h)
            d2u = self._biharm_del2(u, Ahbi, maskU, 'u')
            d2v = self._biharm_del2(v, Ahbi, maskV, 'v')
            self._ke_hstencil = -float(OM.np.sum(
                (d2u*d2u*self.area_u + d2v*d2v*self.area_v)[sl, sl, sl]
                * self.dzt_z[sl]))
        return un, vn

    def bottom_friction(self, u, v, maskU, maskV):
        """Book the KE the package's bottom friction removes (D_bot). The
        friction itself is the package's implicit split step on the lowest
        wet cell (enable_bottom_friction, r_bot), which runs after the
        vertical friction -- icon15 still carries its own forward-Euler
        linear_bottom_friction from before the package had one, and with
        enable_bottom_friction set it therefore applies the drag TWICE."""
        ub, vb = super().bottom_friction(u, v, maskU, maskV)
        if KE_BUDGET:
            self._ke_bfric = self._ke_of(ub, vb) - self._ke_of(u, v)
        return ub, vb


def get_ocean_parameter():
    return dict(
        nx=nx, ny=ny, nz=nz,
        dxt=dxt_deg, dyt=dyt_deg, dzt=dzt_global(),
        coord_degree=True, x_origin=x_origin, y_origin=y_origin,
        dt=ocean_dt, snapint=snapint_ocean,
        periodic_in_x=True, periodic_in_y=False,
        eq_of_state=5,
        # The fortran uses the rigid lid. pe_realistic's rigid-lid surface
        # pressure Poisson is singular on the global cyclic domain with land
        # and its plain CG does not converge (pyOM2 avoids this by solving the
        # streamfunction problem with island integrals, which pe does not
        # have). The linear implicit free surface is used instead. It is not
        # the rigid lid: the external gravity wave is damped rather than
        # filtered, and p_s/g is a real sea surface height.
        explicit_free_surface=(FREE_SURFACE == 'explicit'),
        implicit_free_surface=(FREE_SURFACE == 'implicit'),
        # the split-explicit sub cycle (EXP_FREE_SURFACE=explicit, see the
        # block at FREE_SURFACE above); unused by the implicit method
        sub_cycling=SUB_CYCLING, fb_theta=FB_THETA,
        # THE TOP CELL MUST CONSERVE ITS TRACER. With pyOM2's form the
        # advective flux through the surface face is zeroed, so flux-form
        # advection changes the top cell by -var dt div_h(u) every step, i.e.
        # by -var dEta/dz: the relative error accumulates with the moving sea
        # surface, divided by the TOP CELL THICKNESS. That is tolerable on the
        # 10 m cells of the 4 degree runs and is not here -- with dzt(top) =
        # 2.05 m and a cold-start SSH range of several metres the factor is
        # of order one; at 1/3 degree three days of it drove 48 % of the surface outside
        # 32 to 38 psu (1.9 to 61.7), with SST reaching 46.7 C. Temperature
        # hides it better only because dQ/dT restores the top cell in 2.4 days
        # against the salinity piston's 12. This option gives the surface face
        # the flux -w_surf var_surf, so a uniform tracer stays uniform.
        free_surface_tracer_flux=True,
        superbee_advection=True,
        vmix_scheme='implicit',
        enable_tke=True,
        c_k=0.1, c_eps=0.7, alpha_tke=ALPHA_TKE, mxl_min=1e-8,
        # NO isoneutral diffusion and NO GM: 1/10 degree is eddy resolving
        # and setup1.f90 runs neither here.
        enable_neutral_diffusion=False,
        # background diffusivity below the TKE value. setup1.f90 gets the
        # interior mixing from IDEMIX, which pe has no module for, so this
        # floor is all there is below the mixed layer. The viscosity floor
        # kappaM_min stays at the tke.py default 0.
        kappaH_min=1e-5,
        # linear bottom friction (Sep 2026, pyOM2 names): damps the spurious
        # closed recirculation that grew in a deep dead-end pocket at 178E,
        # 26S-22S, where the free-slip walls and kappaM_min = 0 left it
        # without any friction
        enable_bottom_friction=True, r_bot=1e-5,
        enable_quadratic_bottom_friction=True, r_quad_bot=2.5e-3,
        # ocean alone: surface TKE flux alpha_CB |tau_a/rho_0|^{3/2} (Craig
        # and Banner 1994, alpha ~ 100, range 50-150); pyOM2 uses 1. Coupled
        # runs take phi_oc from the wave model instead (ocean._surface_forcing)
        alpha_CB=ALPHA_CB,
        # BIHARMONIC friction as setup1.f90: A_hbi = 2.7e10 m^4/s with
        # cosPower 1.5 applied per Laplacian pass, as pyOM2 does (so cos^3 in
        # total). There is no 'Ah' key here, and the package applies the
        # harmonic operator only when the attribute exists, so this
        # configuration has biharmonic friction and nothing else.
        # Forward-Euler stability, evaluated over every wet cell of this grid
        # with the cos^3 factor included,
        #     S = A cos^3 dt (4/dx^2 + 4/dy^2)^2 / 2 <= 1 ,
        # peaks at S = 0.010 at the southern edge, where the cells are 2.2 km
        # and cos^3 is 8e-3. At dt = 144 s (setup1.f90's unreduced dt_mom) it
        # would be 0.041, so the time step is NOT what the friction limits.
        # For comparison, icon033 runs at S = 0.004.
        Ahbi=2.7e10, biha_friction_cosPower=1.5,
        # CG of the implicit free surface: the error is max|delta p_s| per
        # iteration, so 1e-9 is the tolerance the 1/10 degree run uses
        # (1e-18 costs ~650 iterations per step there)
        # (implicit method only)
        epsilon=1e-9, max_iterations=2000,
    )
