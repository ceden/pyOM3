"""
acc2: a simple global model with a Southern Ocean and an Atlantic basin
pyOM2: for_config/acc2.f90, py_config/acc2.py

Spherical coordinates, 2 degree resolution, 15 levels, wind stress and
surface temperature restoring, isoneutral mixing with GM, TKE vertical mixing.

run:    python acc2.py
"""
import sys; sys.path.append('../..')
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_nc4
from pyOM3.pe_realistic.tke import tke

nx, ny, nz = 30, 42, 15
dx, dt     = 2.0, 4800.
dzt        = np.array([50., 70., 100., 140., 190., 240., 290., 340., 390., 440., 490., 540., 590., 640., 690.])[::-1]/2.5
y0         = -40.0                      # southern boundary
yt_start, yt_end, yu_start, yu_end = -39.0, 43.0, -40.0, 42.0
omega      = 7.2921e-5
t_rest     = 30*86400.
runlen     = 100*365*86400.
snapint    = 365*86400./12.
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=dx, dzt=dzt, dt=dt,
           coord_degree=True, x_origin=0.0, y_origin=y0, set_zero_at_surface=True,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=5, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-12, max_iterations=5000,
           Ah=(2*111e3)**3*2e-11, hor_friction_cosPower=1,
           enable_neutral_diffusion=True, enable_skew_diffusion=True,
           K_iso_0=1000., K_iso_steep=500., K_gm_0=1000., iso_dslope=0.005, iso_slopec=0.01,
           vmix_scheme='implicit', enable_tke=True, alpha_tke=30., mxl_min=1e-8,
           enable_quadratic_bottom_friction=True, r_quad_bot=2e-3,
           snapint=snapint)


class acc2(tke, snap_nc4):

    def set_coriolis(self):
        f = 2*omega*np.sin(self.yt/180.*np.pi)
        self.coriolis_t = OM.np.asarray(f[:, None] + 0*self.xt[None, :])

    def set_topography(self):
        X, Y = np.meshgrid(self.xt, self.yt)
        kbot = np.zeros((self.Ny, self.Nx))
        kbot[(Y < -20) | ((X > 1.0) & (X < 59.0))] = 1       # Southern Ocean and Atlantic basin
        kbot[(X > 25) & (X < 30)] = 5                        # mid Atlantic ridge
        self.kbot = OM.np.asarray(kbot)

    def set_initial_conditions(self):
        H = dzt.sum()
        temp = (1 + self.zt_gl/H)[:, None, None]*15 + np.zeros((nz, ny, nx))
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(zeros, zeros, temp, 35. + zeros, np.zeros((ny, nx)))
        ks = -self.halo - 1                     # surface level
        X, YU = np.meshgrid(self.xt, self.yu)
        X, YT = np.meshgrid(self.xt, self.yt)
        taux = np.where(YT < -20, 1e-4*np.sin(np.pi*(YU - yu_start)/(-20 - yt_start)), 0.)
        taux = np.where(YT > 10, 1e-4*(1 - np.cos(2*np.pi*(YU - 10)/(yu_end - 10))), taux)
        self.taux = OM.np.asarray(taux*self.maskU[ks])
        self.tauy = OM.np.zeros_like(self.taux)
        T_star = np.where(YT < -20, 15*(YT - yt_start)/(-20 - yt_start), 15.)
        T_star = np.where(YT > 20, 15*(1 - (YT - 20)/(yt_end - 20)), T_star)
        self.T_star = OM.np.asarray(T_star)

    def set_forcing(self, tau):
        ks = -self.halo - 1
        self.du = OM.add_to_array(self.du, (tau, ks), self.taux/self.dzt[ks])
        self.forc_temp_surface = (self.T_star - self.temp[ks])*self.dzt[ks]/t_rest*self.maskT[ks]
        self.dtemp = OM.add_to_array(self.dtemp, (tau, ks), self.forc_temp_surface/self.dzt[ks])


M = acc2(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='acc2.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.init_tke()
M.loop(int(runlen/dt))
