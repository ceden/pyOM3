"""
tide_vortical_spec1: an internal tide beam hits a spectrum of balanced eddies
ps3D: tide_vortical_spec1.f90

Non-hydrostatic (nh_simple) with rigid lids. A random phase spectrum of the
geostrophic mode, E(k) ~ 1/(k (1 + k^2 L^2)^1.5) in the first vertical mode,
scaled to the energy mean_flow_energy, localised in the centre of the domain
and balanced to third order in the Rossby number (nh_simple/modes.py). A wave
maker at x = 0.9 Lx forces one gravity wave mode propagating in negative x
direction through the eddies.

run:    python tide_vortical_spec1.py          (one process)
"""
import sys; sys.path.append('../..')
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.snapshot import snap_nc4
from pyOM3.nh_simple.modes import modes

fac        = 1
nx, ny, nz = 2*126*fac, 126*fac, 10*fac
Lx, Ly, Lz = 2*500e3, 500e3, 2000.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt         = 400./fac
f0         = 1e-4
N0         = 30*f0
forcing_wavelength = 20e3
wave_amplitude     = 0.5
mean_flow_energy   = 0.5**2/2
seed       = 1234
runlen     = 15*86400.
snapint    = 86400.

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt,
           f0=f0, N0=N0, Ro=1., dsqr=1., periodic_in_z=False,
           Ahbi=dx**4/(30*86400.), Khbi=dx**4/(30*86400.),
           snapint=snapint)


class tide_vortical_spec1(modes, snap_nc4):

    def set_forcing(self, tau):
        self.time += self.dt
        c, s = np.cos(self.omega_forc*self.time), np.sin(self.omega_forc*self.time)
        self.u = self.u + self.dt*(c*self.u_r + s*self.u_i)
        self.v = self.v + self.dt*(c*self.v_r + s*self.v_i)
        self.w = self.w + self.dt*(c*self.w_r + s*self.w_i)
        self.b = self.b + self.dt*(c*self.b_r + s*self.b_i)


M = tide_vortical_spec1(par)
M.setup(snap_file_name='tide_vortical_spec1.cdf', show_each_time_step=False)
M.init_modes()

# local coordinates with halo points, arrays are (x,y,z), z = 0 at the bottom
h = M.halo
x = (np.arange(M.Nx) - h + 0.5)*dx
y = (np.arange(M.Ny) - h + 0.5)*dy
z = (np.arange(M.Nz) - h + 0.5)*dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# the eddy spectrum in the geostrophic mode of the first vertical mode
L  = N0*Lz/np.pi/f0                                         # Rossby radius
k  = np.argmin((M.kz - np.pi/Lz)**2)
KH = np.hypot(M.KX, M.KY)[:, :, 0]
with np.errstate(divide='ignore'):
    Ep = np.where(KH > 0, 1./(2*np.pi*KH*(1 + KH**2*L**2)**1.5), 0.)
phase = np.exp(2j*np.pi*np.random.default_rng(seed).random(KH.shape))
A = np.sqrt(Ep*2*M.coeff_p(0)[:, :, k])*phase
F = []
for q in M.vec_q(0):
    Q = np.zeros((len(M.kx), ny, nz), complex); Q[:, :, k] = A*q[:, :, k]; F.append(Q)
M.u, M.v, M.w, M.b = M.backward(*F)

# scale to the mean energy, localise, balance
E = 0.5*np.mean(M.u[h:-h, h:-h, h:-h]**2 + M.v[h:-h, h:-h, h:-h]**2 + M.w[h:-h, h:-h, h:-h]**2
                + M.b[h:-h, h:-h, h:-h]**2/N0**2)
mask = np.exp(-(X - 0.5*Lx)**2/(0.4*Ly)**2 - (Y - 0.5*Ly)**2/(0.3*Ly)**2)
for name in ('u', 'v', 'w', 'b'):
    setattr(M, name, M.apply_bc(getattr(M, name)*np.sqrt(mean_flow_energy/E)*mask))
M.balance(3)

# the wave maker: one gravity wave mode, masked around x = 0.9 Lx and
# projected on the wave mode again
i = np.argmin((M.kx - 2*np.pi/forcing_wavelength)**2)
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
