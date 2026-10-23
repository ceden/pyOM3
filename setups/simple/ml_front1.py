"""
ml_front1: mixed layer front, baroclinic instability in the mixed layer
pyOM2: for_config/ml_front.f90 (3-D version, without the mixed layer parameterisation)

A weakly stratified mixed layer of depth Hml above a strongly stratified
pycnocline, with a tanh front in thermal wind balance. Buoyancy is the
prognostic tracer (eq_of_state=0), pe_simple.

run:    python ml_front1.py
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.pe_simple.snapshot import snap_nc4

res_hor    = 0.5
Ri, alpha  = 31.6, 4.0              # Richardson number, slope parameter
H0, f0     = 300., 7e-5
N0         = alpha*np.sqrt(Ri)*f0   # mixed layer stratification
M0         = np.sqrt(alpha)*f0      # front strength
U0         = alpha*H0*f0
Npcl       = 120*f0                 # pycnocline stratification
Hml        = 200.                   # mixed layer depth
kmax       = np.sqrt(5./2)/np.sqrt(1 + Ri)*f0/U0
nx = ny    = int(120*res_hor)
nz, dz     = 60, 5.0
dx         = np.rint(2*np.pi/kmax/10/res_hor)
Ly         = ny*dx
Lf         = 0.33*Ly                # width of the front
dt         = np.rint(5e-3*dx/U0)
xt_gl, yt_gl, zt_gl = (np.arange(nx) + 0.5)*dx, (np.arange(ny) + 0.5)*dx, (np.arange(nz) + 0.5)*dz - nz*dz
b_amp      = 1e-3*9.81*1.67e-4      # random perturbation, 0.001 K
runlen     = 172800*res_hor*dt
snapint    = 216*res_hor*dt
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz, dt=dt,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=0, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-12, max_iterations=5000, superbee_advection=True,
           Ahbi=U0*dx**3*0.5, vmix_scheme='explicit', A_v=U0*dz*5e-3,
           snapint=snapint)


class ml_front1(snap_nc4):

    def set_coriolis(self):
        self.coriolis_t = OM.np.asarray(f0 + np.zeros((self.Ny, self.Nx)))

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(zt_gl, yt_gl, xt_gl, indexing='ij')
        N2 = np.where(Z > -Hml, N0**2, Npcl**2)
        b  = np.cumsum(N2, axis=0)*dz - Lf*M0**2/2*np.tanh(2*(Y - Ly/2)/Lf)
        # thermal wind: u_z = -b_y/f0, integrated from the bottom where u = 0
        by = np.gradient(b, yt_gl, axis=1)
        u  = np.cumsum(-by/f0, axis=0)*dz
        u  = u - u[0:1]
        b += b_amp*np.random.default_rng(1).random(b.shape)
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(u, zeros, b, zeros, np.zeros((ny, nx)))


M = ml_front1(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='ml_front1.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.loop(int(runlen/dt))
