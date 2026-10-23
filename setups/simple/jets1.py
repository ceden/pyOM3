"""
jets1: zonal jets in a wide beta-plane channel
pyOM2: for_config/jets1.f90, py_config/jets1.py

Buoyancy is the prognostic tracer (eq_of_state=0), pe_simple. A meridional buoyancy
gradient is maintained by restoring in sponge layers at the northern and
southern walls, baroclinic instability of the mean state generates eddies and jets.

run:    python jets1.py
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_simple.snapshot import snap_nc4

hres, vres = 1.0, 0.5
nx, ny, nz = int(128*hres), int(128*hres), int(18*vres)
Lx, H      = 3800e3, 1800.
dx, dz, dt = Lx/nx, H/nz, 1800./hres
xt_gl, yt_gl, zt_gl = (np.arange(nx) + 0.5)*dx, (np.arange(ny) + 0.5)*dx, (np.arange(nz) + 0.5)*dz - H
M_0        = np.sqrt(1e-5*0.1/1024.*9.81)     # meridional buoyancy gradient sqrt(M_0^2)
N_0        = 0.004                            # stability frequency
r_ray      = 0.5e-7                           # Rayleigh friction
spg_width  = int(3*hres)                      # sponge width in grid cells
t_rest     = 1./(5*86400.)
omega, radius = 7.2921e-5, 6370e3
runlen     = 10*365*86400.
snapint    = 3*86400.
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz, dt=dt,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=0, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-12, max_iterations=5000,
           Ahbi=1e11/hres**4, Khbi=1e11/hres**4,
           vmix_scheme='explicit', kappa_T=1e-4/vres**2, A_v=1e-3/vres**2,
           snapint=snapint)


def b_star(Y, Z):
    return N_0**2*Z - M_0**2*Y


class jets1(snap_nc4):

    def set_grid(self):
        # coordinates of the T points of this PE, halo included
        i = np.arange(self.Nx) - self.halo + self.xs_pe - 1
        j = np.arange(self.Ny) - self.halo + self.ys_pe - 1
        k = np.arange(self.Nz) - self.halo
        self.xt, self.yt, self.zt = (i + 0.5)*dx, (j + 0.5)*dx, (k + 0.5)*dz - H

    def set_coriolis(self):
        phi0 = 10./180.*np.pi
        f = 2*omega*np.sin(phi0) + 2*omega*np.cos(phi0)/radius*self.yt
        self.coriolis_t = OM.np.asarray(f[:, None] + 0*self.xt[None, :])

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(zt_gl, yt_gl, xt_gl, indexing='ij')
        b = b_star(Y, Z) - 0.5e-3*np.sin(8.5*np.pi*X/Lx)
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(zeros, zeros, b, zeros, np.zeros((ny, nx)))
        # sponge layers: restoring rate decreases away from the walls
        Z, Y, X = np.meshgrid(self.zt, self.yt, self.xt, indexing='ij')
        j = np.round(Y/dx + 0.5)                        # row index as in pyOM2, 1 is the first row
        rate = np.where((j >= 2) & (j <= spg_width + 1), t_rest/np.maximum(j - 1, 1), 0.)
        rate = np.where((j >= ny - spg_width) & (j <= ny - 1), t_rest/np.maximum(ny - j, 1), rate)
        self.rate   = OM.np.asarray(rate*self.maskT)
        self.b_star = OM.np.asarray(b_star(Y, Z))

    def set_forcing(self, tau):
        self.dtemp = OM.add_to_array(self.dtemp, (tau,), self.rate*(self.b_star - self.temp))
        self.du    = OM.add_to_array(self.du, (tau,), -r_ray*self.u)
        self.dv    = OM.add_to_array(self.dv, (tau,), -r_ray*self.v)


M = jets1(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_grid()
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='jets1.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.loop(int(runlen/dt))
