"""
acc3: Southern Ocean channel on the sphere with a meridional wall north of the channel
pyOM2: for_config/acc3.f90 (coarse, hres=1), for_config/acc_hi1.f90 (eddying, hres=5)

Wind stress and surface temperature restoring, TKE vertical mixing.
With eddies=False (acc3): harmonic friction, isoneutral mixing and GM.
With eddies=True (acc_hi1): biharmonic friction and superbee advection.

run:    python acc3.py
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_nc4
from pyOM3.pe_realistic.tke import tke

hres, vres = 1.0, 2.0
eddies     = False
nx, ny, nz = int(20*hres), int(20*hres), int(10*vres)
dx, dt     = 1.0/hres, 3600./hres
y0, y1, y2 = -40., -32., -28.             # southern boundary, end of channel, start of restoring zone
yt_1       = y0 - dx/2                    # first T point
yt_ny      = yt_1 + dx*ny                 # last T point plus one cell
omega      = 7.2921e-5
t_rest     = 30*86400.
runlen     = 50*365*86400.
snapint    = 86400./4
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=dx, dzt=1000./nz, dt=dt,
           coord_degree=True, x_origin=0.0, y_origin=y0, set_zero_at_surface=True,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=5, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-9, max_iterations=15000,
           vmix_scheme='implicit', enable_tke=True, alpha_tke=30., mxl_min=1e-8,
           enable_bottom_friction=True, r_bot=1e-5*vres,
           snapint=snapint)
if eddies:
    par.update(Ahbi=5e11, superbee_advection=True)
else:
    par.update(Ah=(111e3)**3*2e-11, hor_friction_cosPower=1,
               enable_neutral_diffusion=True, enable_skew_diffusion=True,
               K_iso_0=1000., K_iso_steep=1000., K_gm_0=1000., iso_dslope=4e-3, iso_slopec=4e-3)


class acc3(tke, snap_nc4):

    def set_coriolis(self):
        f = 2*omega*np.sin(self.yt/180.*np.pi)
        self.coriolis_t = OM.np.asarray(f[:, None] + 0*self.xt[None, :])

    def set_topography(self):
        X, Y = np.meshgrid(self.xt, self.yt)
        kbot = np.ones((self.Ny, self.Nx))
        kbot[(Y >= y1) & (abs(X - nx*dx/2) <= 1.0)] = 0      # wall, 2 degree wide
        self.kbot = OM.np.asarray(kbot)

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(self.zt_gl, self.yt_gl, self.xt_gl, indexing='ij')
        y = Y/yt_ny
        temp = (1 + Z/1000.)*15 + 2*(np.tanh((y - 0.5)/0.1) + 1) + 3*(np.tanh(-(y - 0.5)/0.2) + 1)
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(zeros, zeros, temp, 35. + zeros, np.zeros((ny, nx)))
        ks = -self.halo - 1
        X, YU = np.meshgrid(self.xt, self.yu)
        X, YT = np.meshgrid(self.xt, self.yt)
        taux = np.where(YT < y1, 1e-4*np.sin(np.pi*(YU - y0)/(y1 - yt_1)), 0.)
        self.taux = OM.np.asarray(taux*self.maskU[ks])
        self.tauy = OM.np.zeros_like(self.taux)
        T_star = np.where(YT < y1, 15*(YT - yt_1)/(y1 - yt_1), 15.)
        T_star = np.where(YT > y2, 15*(1 - (YT - y2)/(yt_ny - y2)), T_star)
        self.T_star = OM.np.asarray(T_star)

    def set_forcing(self, tau):
        ks = -self.halo - 1
        self.du = OM.add_to_array(self.du, (tau, ks), self.taux/self.dzt[ks])
        self.forc_temp_surface = (self.T_star - self.temp[ks])*self.dzt[ks]/t_rest*self.maskT[ks]
        self.dtemp = OM.add_to_array(self.dtemp, (tau, ks), self.forc_temp_surface/self.dzt[ks])


M = acc3(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='acc3.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.init_tke()
M.loop(int(runlen/dt))
