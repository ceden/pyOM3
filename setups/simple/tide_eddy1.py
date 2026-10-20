"""
tide_eddy1: an internal tide beam hits a balanced eddy
ps3D: tide_eddy1.f90

Non-hydrostatic (nh_simple) with rigid lids. A geostrophic eddy of the first
vertical mode, balanced to third order in the Rossby number with the eigenmode
decomposition of nh_simple/modes.py, and a wave maker at x = 0.9 Lx that forces
one gravity wave mode (first vertical mode, wavelength forcing_wavelength)
propagating in negative x direction through the eddy.

run:    python tide_eddy1.py          (one process)
"""
import os, sys
sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.snapshot import snap_nc4
from pyOM3.nh_simple.modes import modes

fac        = 1                      # 2 in ps3D
nx, ny, nz = 126*fac, 126*fac, 10*fac
Lx, Ly, Lz = 500e3, 500e3, 2000.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt         = 400./fac
f0         = 1e-4
N0         = 30*f0
forcing_wavelength = 20e3
wave_amplitude     = 0.5
eddy_velocity      = 0.2
runlen     = 15*86400.
snapint    = 86400.

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt,
           f0=f0, N0=N0, Ro=1., dsqr=1., periodic_in_z=False,
           Ahbi=dx**4/(30*86400.), Khbi=dx**4/(30*86400.),
           snapint=snapint)


class tide_eddy1(modes, snap_nc4):

    def set_forcing(self, tau):
        self.time += self.dt
        c, s = np.cos(self.omega_forc*self.time), np.sin(self.omega_forc*self.time)
        self.u = self.u + self.dt*(c*self.u_r + s*self.u_i)
        self.v = self.v + self.dt*(c*self.v_r + s*self.v_i)
        self.w = self.w + self.dt*(c*self.w_r + s*self.w_i)
        self.b = self.b + self.dt*(c*self.b_r + s*self.b_i)


M = tide_eddy1(par)
M.setup(snap_file_name='tide_eddy1.cdf', show_each_time_step=False)
M.init_modes()

# local coordinates with halo points, arrays are (x,y,z), z = 0 at the bottom
h = M.halo
x = (np.arange(M.Nx) - h + 0.5)*dx
y = (np.arange(M.Ny) - h + 0.5)*dy
z = (np.arange(M.Nz) - h + 0.5)*dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# the eddy: solid body rotation inside the radius Rad, decaying outside,
# first vertical mode; its buoyancy from the thermal wind, then balanced
Rad = 2*N0*Lz/np.pi/f0
def eddy(X, Y):
    r = np.hypot(X - Lx/2, Y - Ly/2)
    r = np.where(r > Rad, Rad*np.exp(-(r - Rad)/(Rad/2)), r)
    return r/Rad, np.arctan2(Y - Ly/2, X - Lx/2)
r, theta = eddy(X + dx/2, Y)                                # U points
u = -r*np.sin(theta)*np.cos(np.pi*Z/Lz)
r, theta = eddy(X, Y + dy/2)                                # V points
v =  r*np.cos(theta)*np.cos(np.pi*Z/Lz)
M.u = M.apply_bc(OM.np.asarray(u))
M.v = M.apply_bc(OM.np.asarray(v))
b = -f0*np.cumsum(np.gradient(u, dz, axis=2), axis=1)*dy    # b_y = -f u_z
M.b = M.apply_bc(OM.np.asarray(b))
M.balance(0)
scale = eddy_velocity/abs(M.u).max()
M.u, M.v, M.w, M.b = M.u*scale, M.v*scale, M.w*scale, M.b*scale
M.balance(3)

# the wave maker: one gravity wave mode, masked around x = 0.9 Lx and
# projected on the wave mode again
i = np.argmin((M.kx - 2*np.pi/forcing_wavelength)**2)
k = np.argmin((M.kz - np.pi/Lz)**2)
om = np.sqrt(M.omega2()[i, 0, k])
M.omega_forc = -om                                          # propagation in negative x direction
print('beam: wavelength %.1f km, omega/f = %.2f' % (2*np.pi/M.kx[i]/1e3, om/f0))
real, imag = M.mode_fields(i, 0, k, 1)
mask = np.exp(-(X - 0.9*Lx)**2/(2*forcing_wavelength)**2 - (Y - 0.5*Ly)**2/(2*forcing_wavelength)**2)
amp  = wave_amplitude*1e-6*om/f0/max(abs(real[0]).max(), abs(imag[0]).max())
M.u_r, M.v_r, M.w_r, M.b_r = M.project(*[amp*mask*a for a in real], 1)
M.u_i, M.v_i, M.w_i, M.b_i = M.project(*[amp*mask*a for a in imag], 1)
M.time = 0.
M.loop(int(runlen/dt))
