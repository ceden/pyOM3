"""
roullet1: baroclinic jet in a beta-plane channel after Roullet et al. (2012)
pyOM2: for_config/roullet1.f90

Exponential stratification with a mixed layer, a surface intensified jet in thermal
wind balance, vertical grid stretched with the stratification. Buoyancy is the
prognostic tracer (eq_of_state=0). The zonal mean is restored to the initial state.

run:    python roullet1.py
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_nc4

gridpoints = 128                           # 256 in the original
Ld, H0     = 96e3, 4e3
f0, beta   = 1e-4, 1.56e-11
Ustar, alpha, Ek = 0.6, 1., 0.05
Ljet       = 1200e3
nx, ny, nz = gridpoints, 2*gridpoints, gridpoints//4
Lx         = 8*Ld
Ly         = 2*Lx
dx         = Lx/nx
dt         = 240.*(dx/3e3)
t_rest     = 1./(3*86400.)
runlen     = 5*365*86400.
snapint    = 2*86400.
n_pes_x, n_pes_y = 1, 1


def b_profile(z):                          # stratification, z/H0 in (-1,0)
    zbar = 1 + z/H0
    b1 = 0.5*(1 + np.tanh(8*(-0.95)))
    b0 = 1./(0.1 - b1 + 0.5*(1 + np.tanh(8*(1 - 0.95))))
    return b0*(0.1*zbar - b1 + 0.5*(1 + np.tanh(8*(zbar - 0.95))))


# vertical grid: dz proportional to 1/N
z   = (np.arange(nz) + 0.5)*H0/nz - H0
S   = np.sqrt(np.gradient(b_profile(z), z))
dzt = dx/S
dzt = dzt/dzt.sum()*H0

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=dx, dzt=dzt, dt=dt,
           coord_degree=False, set_zero_at_surface=True,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=0, implicit_free_surface=True, explicit_free_surface=False,
           epsilon=1e-6*(dx/3e3)**2, max_iterations=5000, superbee_advection=True,
           Ahbi=Ek*f0*dx**4, vmix_scheme='explicit', A_v=Ek*f0*(H0/nz)**2,
           enable_bottom_friction=True, r_bot=1./(30*86400.)*100./dzt[0],
           snapint=snapint)


class roullet1(snap_nc4):

    def set_coriolis(self):
        f = f0 + beta*(self.yt - Ly/2)
        self.coriolis_t = OM.np.asarray(f[:, None] + 0*self.xt[None, :])

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(self.zt_gl, self.yt_gl, self.xt_gl, indexing='ij')
        ZU, YU, XU = np.meshgrid(self.zt_gl, self.yu_gl, self.xt_gl, indexing='ij')
        b  = (f0*Ld)**2/H0*b_profile(Z)
        bn = b*H0/(f0*Ld)**2                                      # normalised b
        u0 = Ustar*beta*Ld**2*(2*(alpha - 1)*bn**3 + (3 - 2*alpha)*bn**2)
        u0 = np.where(abs(YU - Ly/2)/Ljet < 0.5, u0*np.cos((YU - Ly/2)/Ljet*np.pi), 0.)
        # thermal wind: b_y = -f u_z, integrated from the southern wall
        uz = np.gradient(u0, self.zt_gl, axis=0)
        b  = b - np.cumsum((f0 + beta*(Y - Ly/2))*uz, axis=1)*dx
        # perturbation of v: a wave and noise
        v0 = Ustar*beta*Ld**2/100.*np.sin(X/Lx*12*np.pi)*np.cos(YU/Ly*np.pi)*np.cos(Z/H0*np.pi)
        v0 = v0 + Ustar*beta*Ld**2/500.*(np.random.default_rng(1).random(v0.shape) - 0.5)
        super().set_initial_conditions(u0, v0, b, np.zeros((nz, ny, nx)), np.zeros((ny, nx)))
        self.u0 = self.u*1
        self.b0 = self.temp*1

    def set_forcing(self, tau):
        h  = self.halo                         # zonal mean over the interior, needs n_pes_x = 1
        um = OM.np.mean(self.u[:, :, h:-h], axis=2)[:, :, None]
        bm = OM.np.mean(self.temp[:, :, h:-h], axis=2)[:, :, None]
        self.du    = OM.add_to_array(self.du, (tau,), t_rest*(self.u0 - um)*self.maskU)
        self.dtemp = OM.add_to_array(self.dtemp, (tau,), t_rest*(self.b0 - bm)*self.maskT)


M = roullet1(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='roullet1.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.loop(int(runlen/dt))
