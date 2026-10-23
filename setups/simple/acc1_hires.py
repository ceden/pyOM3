"""
acc1_hires: idealised Southern Ocean channel after Viebahn and Eden (2010), eddy resolving
pyOM2: for_config/acc1.f90

Zonally periodic channel with a meridional wall in the northern half,
wind stress and surface buoyancy restoring, TKE vertical mixing,
biharmonic friction and superbee advection. pe_simple with buoyancy as the
prognostic tracer (eq_of_state=0). The coarse version is acc1_coarse.py.

run:    python acc1_hires.py       (mpirun -np N python acc1_hires.py with n_pes_x*n_pes_y = N)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_simple.snapshot import snap_nc4
from pyOM3.pe_simple.tke import tke

hres, vres = 1.0, 1.0        # resolution factors, 1 = 20 km and 50 m
nx, ny, nz = int(128*hres), int(128*hres), int(18*vres)
dx, dz, dt = 20e3/hres, 50./vres, 800./hres
Lx, Ly     = nx*dx, ny*dx
phi0       = -25./180.*np.pi
omega, radius = 7.2921e-5, 6370e3
t_rest     = 30*86400.
runlen     = 5.5*365*86400.
snapint    = 3*86400.
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz, dt=dt,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=0, implicit_free_surface=True, explicit_free_surface=False,
           epsilon=1e-8, max_iterations=5000, superbee_advection=True,
           Ahbi=5e11/hres**4,
           vmix_scheme='implicit', enable_tke=True, alpha_tke=30., mxl_min=1e-8,
           enable_bottom_friction=True, r_bot=1e-5*vres,
           snapint=snapint)


class acc1_hires(tke, snap_nc4):

    def set_grid(self):
        # coordinates of the T and U points of this PE, halo included
        i = np.arange(self.Nx) - self.halo + self.xs_pe - 1
        j = np.arange(self.Ny) - self.halo + self.ys_pe - 1
        self.xt, self.yt, self.yu = (i + 0.5)*dx, (j + 0.5)*dx, (j + 1.0)*dx

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
        super().set_initial_conditions(zeros, zeros, zeros, zeros, np.zeros((ny, nx)))
        # wind stress on U points and restoring buoyancy on T points
        ks = -self.halo - 1                     # surface level
        X, YU = np.meshgrid(self.xt, self.yu)
        X, YT = np.meshgrid(self.xt, self.yt)
        self.taux = OM.np.asarray(np.where(YT < Ly/2, 1e-4*np.sin(2*np.pi*YU/Ly), 0.)*self.maskU[ks])
        self.tauy = OM.np.zeros_like(self.taux)
        b_star = np.where(YT < Ly/2, 30e-3*YT/(Ly/2), 30e-3)
        b_star = np.where(YT > 0.75*Ly, 30e-3*(1 - (YT - 0.75*Ly)/(0.25*Ly)), b_star)
        self.b_star = OM.np.asarray(b_star)

    def set_forcing(self, tau):
        ks = -self.halo - 1                     # surface level
        self.du = OM.add_to_array(self.du, (tau, ks), self.taux/dz)
        self.forc_temp_surface = (self.b_star - self.temp[ks])*dz/t_rest*self.maskT[ks]
        self.dtemp = OM.add_to_array(self.dtemp, (tau, ks), self.forc_temp_surface/dz)


M = acc1_hires(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_grid()
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='acc1_hires.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.init_tke()
M.loop(int(runlen/dt))
