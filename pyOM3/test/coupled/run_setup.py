"""
The experiment module of test 22 (endtoend.py): a configuration for
setups/coupled_common/run.py in the form of icon15_setup.py / global4_setup.py,
on the small box of the coupled family with the synthetic forcing source,
so that run.py, the ocean-alone loop, the restart over segments and the
means can be run end to end without input files -- on numpy (one or more
MPI ranks) and on the jax backend (one process, the devices as shards).

Everything a configuration module must define is here: the grid, the
parameters, the forcing settings, the two model classes, exp_name, the
snapshot names, the averaging mode, the dt handshake values. The numbers
are those of test/coupled/config.py (the goldens of test 9 are on the same
box), the ocean class is config.small_ocean's with its make_initial_fields.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_raw as ocean_base
from pyOM3.wim_realistic.snapshot import snap_raw as wave_base
from pyOM3.pe_realistic.tke import tke as tke_mixin
import config as _c                               # registers the 'synthetic' source
from exp_common import Cfg, parse_average, spectral_grid_report as _sgr
from exp_ocean import ocean_common
from exp_wave import wave_common

exp_name = 'runtest'
nx, ny, nz = _c.NX, _c.NY, _c.NZ
dxt_deg, dyt_deg, x_origin, y_origin = _c.DXT, _c.DYT, _c.X0, _c.Y0
rho_0, cp_0, rho_air = _c.RHO_0, _c.CP_0, _c.RHO_AIR
t_rest = 30.0*86400.0
omega_e = _c.OMEGA
FORCING_START = '2000-01-01'
FORCING_CYCLE_DAYS = 0.0
HS_FACTOR = 1.6
MXL_CHOICE = 1
MXL_BETA = 1.0
MXL_SURF_FACTOR = 1.0
RAW_SNAPSHOTS = True
KE_BUDGET = int(os.environ.get('EXP_KE_BUDGET', '1') or 0)
AVERAGE_MLD = True
SW_BOTTOM = os.environ.get('EXP_SW_BOTTOM', '0') not in ('0', '', 'no')
STRESS_MODE = os.environ.get('EXP_STRESS_MODE', 'era5')
STRESS_UO = os.environ.get('EXP_STRESS_UO', 'eulerian')
SSS_PISTON = float(os.environ.get('EXP_SSS_PISTON', '0.0'))/86400.0
VADV_SUB_CYCLING = int(os.environ.get('EXP_VADV_SUB_CYCLING', '0'))
FREE_SURFACE = os.environ.get('EXP_FREE_SURFACE', 'implicit')        # | explicit (test 22e)
ICE_TAU_FRAC = float(os.environ.get('EXP_ICE_TAU_FRAC', '1.0'))
direct_stress_branch = bool(int(os.environ.get('EXP_DIRECT_BRANCH', '0')))
wave_coarsen = 1
ocean_dt = float(os.environ.get('EXP_OCEAN_DT', '%g' % _c.DT))
wave_dt = float(os.environ.get('EXP_WAVE_DT', '300'))
nf, nd, f0, df0 = 16, 12, 0.06, 0.13
snapint_ocean = float(os.environ.get('EXP_SNAPINT_OCEAN_DAYS', '30'))*86400.0
snapint_wave = float(os.environ.get('EXP_SNAPINT_WAVE_DAYS', '1'))*86400.0
average_mode = parse_average(os.environ.get('EXP_AVERAGE', '0'))
transfer_mode = 'stress'
use_stress_work = False
wave_snap_file = 'raw_wave_%s.cdf' % exp_name
ocean_snap_file = 'raw_ocean_%s.cdf' % exp_name
ALPHA_CB = 100.0


def spectral_grid_report():
    return _sgr(nf, nd, f0, df0)


def dzt_global():
    return _c.DZT.copy()


def zw_pyom2():
    return _c.zw_pyom2()


def kbot_global():
    return _c.kbot_global()


def forcing_settings():
    return {'source': 'synthetic', 'start': FORCING_START, 'cycle_days': FORCING_CYCLE_DAYS,
            'amp': 1.0, 'omit': ()}


class wave(wave_common, wave_base):
    cfg = Cfg(sys.modules[__name__])


def get_wave_parameter():
    return {'nx': nx, 'ny': ny, 'nf': nf, 'nd': nd, 'dxt': dxt_deg, 'dyt': dyt_deg,
            'coord_degree': True, 'x_origin': x_origin, 'y_origin': y_origin,
            'dt': wave_dt, 'f0': f0, 'df0': df0, 'periodic_in_x': False, 'periodic_in_y': False,
            'mask_energy': True, 'limit_growth': True, 'snonlin_backend': 'numpy',
            'propagation_backend': 'numpy', 'snapint': snapint_wave}


class ocean(ocean_common, tke_mixin, ocean_base):
    cfg = Cfg(sys.modules[__name__])
    mix_eulerian = True
    coupled = True

    def prepare_for_run(self, snap_file_name='raw_snap', show_each_time_step=True):
        if snap_file_name.endswith('.cdf'):
            snap_file_name = snap_file_name[:-4]
        super().prepare_for_run(snap_file_name, show_each_time_step)
        self._ke_vfric = self._ke_vnum = self._ke_hfric = 0.0
        self.mpi_barrier()
        return

    def make_initial_fields(self):
        """the initial state of config.small_ocean (global interior arrays)"""
        jg = np.arange(ny)[:, None]*np.ones((1, nx)); ig = np.arange(nx)[None, :]*np.ones((ny, 1))
        zt = np.cumsum(_c.DZT) - 0.5*_c.DZT; H = _c.DZT.sum()
        T = 4.0 + 16.0*(zt/H)[:, None, None]*np.ones((nz, ny, nx))
        T += 1.0*np.exp(-((ig - nx/2)/3.)**2 - ((jg - ny/2)/2.)**2)[None]*(zt/H)[:, None, None]
        S = 35.0*np.ones((nz, ny, nx)) + 0.3*(zt/H - 0.5)[:, None, None]
        return (np.zeros((nz, ny, nx)), np.zeros((nz, ny, nx)), T, S, np.zeros((ny, nx)))

    def harmonic_momentum(self, *a, **k):
        if not self.cfg.KE_BUDGET:
            return super().harmonic_momentum(*a, **k)
        u0, v0 = self.u*1.0, self.v*1.0
        k0 = self._ke_of(u0, v0)
        out = super().harmonic_momentum(*a, **k)
        self._ke_hfric = self._ke_of(self.u, self.v) - k0
        self._ke_hnum = self._ke_of(self.u - u0, self.v - v0)
        self._ke_hstencil = self._ke_hfric - self._ke_hnum
        return out


def get_ocean_parameter():
    return dict(nx=nx, ny=ny, nz=nz, dt=ocean_dt, snapint=snapint_ocean,
                dxt=dxt_deg, dyt=dyt_deg, dzt=dzt_global(), coord_degree=True,
                x_origin=x_origin, y_origin=y_origin, periodic_in_x=False, periodic_in_y=False,
                eq_of_state=5, explicit_free_surface=(FREE_SURFACE == 'explicit'),
                implicit_free_surface=(FREE_SURFACE == 'implicit'), sub_cycling=12, fb_theta=0.14,
                superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
                enable_tke=True, c_k=0.1, c_eps=0.7, alpha_tke=30.0, mxl_min=1e-8,
                kappa_T=1e-5, A_v=1e-4, Ahbi=1e11, biha_friction_cosPower=1.0,
                alpha_CB=ALPHA_CB, epsilon=1e-12, max_iterations=5000)
