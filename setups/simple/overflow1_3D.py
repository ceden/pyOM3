"""
overflow1_3D: dense water from a shelf through a gap down a slope, rotating tank
pyOM2: for_config/overflow1_3D.f90

Non-hydrostatic (nh_realistic), laboratory scale. Cold water on a shelf
(x <= slope) lies next to warmer water. A barrier along the shelf edge has
a gap in y. The cold water flows through the gap and down the slope,
deflected by the rotation of the tank.

run:    python overflow1_3D.py     (mpirun -np N python overflow1_3D.py with n_pes_x*n_pes_y = N)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.nh_realistic.snapshot import snap_nc4

fac        = 1.0
mix        = 2e-6
nx, ny, nz = int(60*fac), int(60*fac), int(20*fac)
dx         = 0.01/fac
dt         = 0.25/fac
L0, H0     = nx*dx, nz*dx
f0         = 0.4189                 # 4 rpm (1 rpm: 0.1047, 2 rpm: 0.2094, 8 rpm: 0.8378)
slope      = nx//4                  # last index of the shelf, the barrier is at this index
shelf      = H0/4                   # depth of the shelf
gap1, gap2 = nx*2//3, nx*5//6       # the gap in the barrier, in y index
runlen     = 600.
snapint    = 2.
n_pes_x, n_pes_y = 2, 3

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=dx, dzt=dx, dt=dt,
           coord_degree=False, set_zero_at_surface=True,
           periodic_in_x=False, periodic_in_y=False,
           eq_of_state=1, implicit_free_surface=False, explicit_free_surface=False,
           epsilon=1e-8, max_iterations=5000, epsilon_nh=1e-6, superbee_advection=True,
           Ah=mix/fac**2, vmix_scheme='explicit', A_v=mix/fac**2,
           enable_bottom_friction=True, r_bot=0.7,
           snapint=snapint)


class overflow1_3D(snap_nc4):

    def set_coriolis(self):
        self.coriolis_t = OM.np.asarray(f0 + 0*self.yt[:, None] + 0*self.xt[None, :])

    def set_topography(self):
        i = np.rint(self.xt/dx + 1.5)[None, :]      # pyOM2 index, xt = (i - 1.5) dx
        j = np.rint(self.yt/dx + 1.5)[:, None]
        b = (nx*shelf - H0*slope)/(nx - slope)
        a = (shelf - b)/slope
        depth = np.where(i <= slope, shelf, a*i + b) + 0*j
        kbot = np.ones((self.Ny, self.Nx))
        for k in range(nz):
            kbot[self.zt_gl[k] < -depth] = k + 1
        kbot[(i == slope) & ((j < gap1) | (j > gap2))] = 0
        self.kbot = OM.np.asarray(kbot)

    def set_initial_conditions(self):
        T = 12 + np.zeros((nz, ny, nx))
        T[:, :, :slope] = 10.
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(zeros, zeros, T, zeros, None)


M = overflow1_3D(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='overflow1_3D.cdf', show_each_time_step=False)
M.set_initial_conditions()
M.loop(int(runlen/dt))
