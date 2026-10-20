"""
What the experiment modules (icon033_setup.py, icon15_setup.py,
global4_setup.py, ...) share that is neither a model class nor forcing:
the configuration view the shared classes read, the averaging-mode parser,
the spectral grid report and the COARSER WAVE GRID.
"""
import numpy as np
from coupler_mpi import coarse_edges


class Cfg:
    """The experiment module as the shared classes see it. exp_ocean.py and
    exp_wave.py never read the environment and never name an experiment:
    every number they need is an attribute of the module that defines the
    experiment, reached as self.cfg.NAME. Names an experiment does not define
    fall back to the defaults below, so an older module keeps working."""
    DEFAULTS = dict(KE_BUDGET=True, AVERAGE_MLD=True, AVERAGE_EDDY=False, AVERAGE_TRACER_FLUX=False,
                    RAW_SNAPSHOTS=False, SW_BOTTOM=False, SSS_PISTON=0.0,
                    AVERAGE_FIELDS=[], wave_coarsen=1, direct_stress_branch=False,
                    # the FULL vortex force -u_L x (nabla x u_S) in the hydrostatic
                    # ocean: the horizontal Stokes shear terms w dz(u_S) and the
                    # vertical component in the hydrostatic relation
                    # (exp_ocean.stokes_shear_force); False = the zeta_S part
                    # alone, the paper's App. C (as before)
                    STOKES_SHEAR=False,
                    # fraction of the surface stress that reaches the ocean
                    # under ice, and the freezing point the mask uses;
                    # 1.0 = the momentum is not masked at all (as before)
                    ICE_TAU_FRAC=1.0, ICE_T_FREEZE=-1.8,
                    # pe_realistic/vadv_split.py: tracer advection outside the
                    # AB step, M vertical sub-steps; 0 = off (as before)
                    VADV_SUB_CYCLING=0,
                    # the coupled ATMOSPHERE (exp_atmos.py, energy_cycle.tex Sec. 6):
                    # boundary layer 'kprofile' | 'rayleigh'; the ocean velocity in the
                    # relative wind 'eulerian' (u_L - u_S) | 'lagrangian' (u_L) and
                    # the factor on it; dissipation heating of the boundary layer
                    # and the interface, and of the biharmonic friction
                    atmos_coarsen=1, ATM_BL='kprofile', ATM_UO='eulerian', ATM_RWIND=1.0,
                    ATM_DISS_HEAT=True, ATM_BIH_HEAT=True,
                    # NEAR-INERTIAL energy flux diagnostic of the ocean (ni_diag.py,
                    # Rimac et al. 2016): off by default. NI_BAND is the relative
                    # half width of the band around f, NI_LEVELS the number of
                    # levels from the surface (0 = all), NI_LAT_MIN the latitude
                    # below which |f| is held fixed, NI_MLD_EVERY the steps between
                    # evaluations of the mixed layer depth, NI_DUMP_EVERY the steps
                    # between raw dumps of u, v, w, p of those levels (0 = none)
                    NI_DIAG=False, NI_BAND=0.2, NI_LEVELS=0, NI_LAT_MIN=5.0,
                    NI_MLD_EVERY=50, NI_DUMP_EVERY=0,
                    # the surface stress of the FORCED ocean (exp_ocean.py):
                    # 'era5' takes tau_ERA5 as it is; 'relwind' keeps ERA5's
                    # drag coefficient and forms the stress from the wind
                    # relative to the surface current (STRESS_UO 'eulerian' |
                    # 'lagrangian'), the forced counterpart of the coupled
                    # model's bulk formula; the waves then take it from the ocean
                    STRESS_MODE='era5', STRESS_UO='eulerian', STRESS_U10_MIN=1.0,
                    STRESS_CD_MAX=3.0e-3, rho_air=1.225)

    def __init__(self, module):
        self.__dict__['_m'] = module

    def __getattr__(self, name):
        m = self.__dict__['_m']
        if hasattr(m, name):
            return getattr(m, name)
        if name in Cfg.DEFAULTS:
            return Cfg.DEFAULTS[name]
        if name == 'exp_name':
            return m.__name__
        raise AttributeError('the experiment module %s does not define %s'
                             % (m.__name__, name))


def parse_average(s):
    s = str(s).strip().lower()
    if s in ('', '0', 'off', 'none'):
        return None
    if s in ('month', 'monthly'):
        return ('month', 0.0)
    if s in ('day', 'daily'):
        return ('interval', 86400.0)
    if s.endswith('d'):
        return ('interval', float(s[:-1])*86400.0)
    if s.endswith('h'):
        return ('interval', float(s[:-1])*3600.0)
    raise ValueError("the averaging mode must be 0, month, day, <N>d or <N>h, got %r" % s)


def spectral_grid_report(nf, nd, f0, df0):
    """One line describing the frequency grid and what the DIA makes of it.
    Printed by run.py so the choice is in the log of every run."""
    import math
    co  = 1.0 + df0
    isp = int(math.log10(1.25)/math.log10(co) + 1e-6)
    fmx = f0*co**(nf - 1)
    note = {2: 'DIA stencil as calibrated',
            1: 'DIA stencil coarsened (ISP 2->1)'}.get(
                isp, 'DIA BROKEN: +lambda lobe on the centre bin')
    return ('spectral grid: nf=%d nd=%d f0=%.5f df0=%.4f -> f = %.4f .. %.4f Hz '
            '(T = %.1f .. %.2f s), DIA ISP=%d [%s]'
            % (nf, nd, f0, df0, f0, fmx, 1./f0, 1./fmx, isp, note))


# ---------------------------------------------------------------------------
# A COARSER WAVE GRID, c = wave_coarsen (1 = the ocean grid).
# The wave grid is the ocean grid with c x c cells merged: its cell faces ARE
# ocean cell faces (coupler_mpi.coarse_edges; the remainder of n/c is absorbed
# by the last group), so the mapping between the two grids is an index map and
# nothing is interpolated. dxt and dyt of a wave cell are the sums over its
# group and the first U point is the last U point of the first group; the wave
# model then builds its own coordinates and metric from them as it always
# does. What the EXCHANGE does with it is in coupler_mpi.py (BlockExchange),
# the coarse land mask and the group-mean wind are in exp_wave.py.
# ---------------------------------------------------------------------------
def coarse_wave_grid(dxt_deg, dyt_deg, x_origin, y_origin, c):
    """dict: je, ie (group edges in ocean indices), nx, ny, dxt, dyt,
    x_origin, y_origin of the wave grid. x_origin is the U point (east face)
    of ocean cell 0; the wave grid's first U point is the east face of ocean
    cell ie[1]-1. For c = 1 everything is returned unchanged."""
    je, ie = coarse_edges(len(dyt_deg), c), coarse_edges(len(dxt_deg), c)
    return {'je': je, 'ie': ie, 'ny': len(je) - 1, 'nx': len(ie) - 1,
            'dxt': np.add.reduceat(dxt_deg, ie[:-1]),
            'dyt': np.add.reduceat(dyt_deg, je[:-1]),
            'x_origin': x_origin + float(np.asarray(dxt_deg)[1:ie[1]].sum()),
            'y_origin': y_origin + float(np.asarray(dyt_deg)[1:je[1]].sum())}


def default_wave_dt(ocean_dt, dt_same_grid, c):
    """Largest divisor of the ocean step not above c times the same-grid wave
    step: the wave step scales with the narrowest cell, but only divisors of
    the ocean step exist."""
    k = 1
    while ocean_dt/k > dt_same_grid*c + 1e-9:
        k += 1
    return ocean_dt/k


def wave_grid_report(wg, c, f0, wave_dt, radius=6370e3):
    """One line on the wave grid and its advective CFL, printed by run.py."""
    from pyOM3.wim_realistic.grid import u_centered_grid
    degtom = radius/180.0*np.pi
    yu, yt, _ = u_centered_grid(wg['dyt'])
    yt = yt + wg['y_origin'] - yu[0]
    dx = (wg['dxt'][None, :]*np.cos(np.deg2rad(yt))[:, None]).min()*degtom
    dy = wg['dyt'].min()*degtom
    cg = 9.81/(4.0*np.pi*f0)
    return ('wave grid: %d x %d, ocean grid coarsened by %d; narrowest cell '
            '%.1f km (x) %.1f km (y); c_g,max %.1f m/s, dt %g s -> CFL %.3f '
            '(certified 0.125)' % (wg['nx'], wg['ny'], c, dx/1e3, dy/1e3, cg,
                                   wave_dt, cg*wave_dt/min(dx, dy)))
