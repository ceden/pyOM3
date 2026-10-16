"""
acc1: idealised Southern Ocean channel after Viebahn and Eden (2010)
pyOM2: for_config/acc1.f90, py_config/acc1.py

Zonally periodic channel with a meridional wall in the northern half,
wind stress and surface temperature restoring, TKE vertical mixing.

run:    python acc1.py             (mpirun -np N python acc1.py with n_pes_x*n_pes_y = N)
"""
import sys; sys.path.append('../..')
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_nc4
from pyOM3.pe_realistic.tke import tke
from pyOM3.density import linear_eq_of_state_drhodT

hres   = 0.5           # horizontal resolution factor, 1 = 20 km
vres   = 1.0           # vertical resolution factor, 1 = 50 m
eddies = False         # True: biharmonic friction only, False: harmonic friction + GM

nx, ny, nz = int(128*hres), int(128*hres), int(18*vres)
dx, dz, dt = 20e3/hres, 50./vres, 1200./hres
Lx, Ly     = nx*dx, ny*dx
phi0       = -25./180.*np.pi
omega, radius, rho_0, grav = 7.2921e-5, 6370e3, 1024., 9.81
t_rest     = 30*86400.
runlen     = 5.5*365*86400.
snapint    = 3*86400.
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=dx, dzt=dz, dt=dt,
           coord_degree=False, set_zero_at_surface=True,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=1, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-6, max_iterations=5000, superbee_advection=True,
           vmix_scheme='implicit', enable_tke=True, alpha_tke=30., mxl_min=1e-8,
           enable_bottom_friction=True, r_bot=1e-5*vres,
           snapint=snapint)
if eddies:
    par.update(Ahbi=2e9/hres**4)
else:
    par.update(Ah=5e4, enable_neutral_diffusion=True, enable_skew_diffusion=True,
               K_iso_0=1000., K_gm_0=1000., iso_dslope=0.005, iso_slopec=0.01)


class acc1(tke, snap_nc4):

    def set_coriolis(self):
        f = 2*omega*np.sin(phi0) + 2*omega*np.cos(phi0)/radius*self.yt
        self.coriolis_t = OM.np.asarray(f[:, None] + 0*self.xt[None, :])

    def set_topography(self):
        X, Y = np.meshgrid(self.xt, self.yt)
        kbot = np.ones((self.Ny, self.Nx))
        kbot[(Y > Ly/2) & ((X > 0.75*Lx) | (X < 0.25*Lx))] = 0
        self.kbot = OM.np.asarray(kbot)

    def set_initial_conditions(self):
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(zeros, zeros, zeros, 35. + zeros, np.zeros((ny, nx)))
        # wind stress on U points and restoring temperature on T points
        ks = -self.halo - 1                     # surface level
        X, YU = np.meshgrid(self.xt, self.yu)
        X, YT = np.meshgrid(self.xt, self.yt)
        self.taux = OM.np.asarray(np.where(YT < Ly/2, 1e-4*np.sin(2*np.pi*YU/Ly), 0.)*self.maskU[ks])
        self.tauy = OM.np.zeros_like(self.taux)
        b_star = np.where(YT < Ly/2, -30e-3*YT/(Ly/2), -30e-3)
        b_star = np.where(YT > 0.75*Ly, -30e-3*(1 - (YT - 0.75*Ly)/(0.25*Ly)), b_star)
        self.T_star = OM.np.asarray(b_star*rho_0/grav/linear_eq_of_state_drhodT())

    def set_forcing(self, tau):
        ks = -self.halo - 1                     # surface level
        self.du = OM.add_to_array(self.du, (tau, ks), self.taux/self.dzt[ks])
        self.forc_temp_surface = (self.T_star - self.temp[ks])*self.dzt[ks]/t_rest*self.maskT[ks]
        self.dtemp = OM.add_to_array(self.dtemp, (tau, ks), self.forc_temp_surface/self.dzt[ks])


M = acc1(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='acc1.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.init_tke()
M.loop(int(runlen/dt))
