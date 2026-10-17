"""
lee_wave1: an internal wave beam in a background current hits a critical layer
ps3D: for_config/lee_wave1.f90

Non-hydrostatic (nh_simple), triply periodic, scaled equations with Rossby
number Ro and aspect ratio dsqr. A wave maker forces one gravity wave mode
with intrinsic frequency omega_i. A constant background current u_back0
Doppler shifts it to zero frequency (a lee wave) and a geostrophic jet of
amplitude jet_amplitude above it brings the intrinsic frequency below f at
a critical layer. The background current enters the perturbation equations
through the linearised advection terms in set_forcing. The background state
is written to forcing.cdf.

run:    python lee_wave1.py          (one process)
"""
import sys; sys.path.append('../..')
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.snapshot import snap_nc4
from pyOM3.nh_simple.modes import modes
from netCDF4 import Dataset

fac        = 2.
nx, ny, nz = int(64*fac), int(64*fac), int(128*fac)
Lx, Ly, Lz = 0.05, 0.05, 2.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
dt         = 10e-3/fac
f0, N0     = 1., 1.
Ro, dsqr   = 0.05, 0.02**2
wavelength_x, wavelength_z = 0.005, 0.02
wave_amplitude = 0.01
jet_width  = 0.3
runlen     = 100.
snapint    = 0.1

par = dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt,
           f0=f0, N0=N0, Ro=Ro, dsqr=dsqr, periodic_in_z=True,
           snapint=snapint)


class lee_wave1(modes, snap_nc4):

    def set_forcing(self, tau):
        # advection of the perturbation by the background current U(y,z) and
        # of the background buoyancy B(y,z) by the perturbation
        U, B = self.u_back, self.b_back
        u, v, w, b = self.u, self.v, self.w, self.b
        rx, ry, rz = self.rollx, self.rolly, self.rollz
        fe = 0.5*(U + rx(U, -1))*(rx(u, -1) + u)
        fn = 0.25*(U + ry(U, -1))*(rx(v, -1) + v)
        ft = 0.25*(U + rz(U, -1))*(rx(w, -1) + w)
        self.du = OM.add_to_array(self.du, (tau,), -Ro*((fe - rx(fe, 1))/dx + (fn - ry(fn, 1))/dy + (ft - rz(ft, 1))/dz))
        fe = 0.25*(v + rx(v, -1))*(ry(U, -1) + U)
        self.dv = OM.add_to_array(self.dv, (tau,), -Ro*(fe - rx(fe, 1))/dx)
        fe = 0.25*(w + rx(w, -1))*(rz(U, -1) + U)
        self.dw = OM.add_to_array(self.dw, (tau,), -dsqr*Ro*(fe - rx(fe, 1))/dx)
        fe = 0.5*(b + rx(b, -1))*U + 0.5*(B + rx(B, -1))*u
        fn = 0.5*(B + ry(B, -1))*v
        ft = 0.5*(B + rz(B, -1))*w
        self.db = OM.add_to_array(self.db, (tau,), -Ro*((fe - rx(fe, 1))/dx + (fn - ry(fn, 1))/dy + (ft - rz(ft, 1))/dz))
        # the wave maker
        self.time += self.dt
        c, s = np.cos(-self.omega_e*self.time), np.sin(-self.omega_e*self.time)
        self.u = self.u + self.dt*(c*self.u_r + s*self.u_i)
        self.v = self.v + self.dt*(c*self.v_r + s*self.v_i)
        self.w = self.w + self.dt*(c*self.w_r + s*self.w_i)
        self.b = self.b + self.dt*(c*self.b_r + s*self.b_i)


M = lee_wave1(par)
M.setup(snap_file_name='lee_wave1.cdf', show_each_time_step=False)
M.init_modes()

# local coordinates with halo points, arrays are (x,y,z)
h = M.halo
x = (np.arange(M.Nx) - h + 0.5)*dx
y = (np.arange(M.Ny) - h + 0.5)*dy
z = (np.arange(M.Nz) - h + 0.5)*dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# the wave: intrinsic frequency, background current for zero absolute frequency
i = np.argmin((M.kx - 2*np.pi/wavelength_x)**2)
k = np.argmin((M.kz - 2*np.pi/wavelength_z)**2)
omega_i = np.sqrt(M.omega2()[i, 0, k])
kx_d    = np.sin(M.kx[i]*dx)/dx                             # discrete wavenumber
u_back0 = omega_i/kx_d/Ro
M.omega_e = omega_i - Ro*u_back0*kx_d
jet_amplitude = 1.1*(omega_i - f0)/kx_d/Ro
print('beam: omega/f = %.2f, angle %.1f deg, u_back0 = %.3f, jet amplitude = %.3f'
      % (omega_i/f0, np.degrees(np.arccos(M.kx[i]/np.hypot(M.kx[i], M.kz[k]))), u_back0, jet_amplitude))

# the wave maker: one gravity wave mode, masked and projected on the wave mode again
real, imag = M.mode_fields(i, 0, k, 1)
mask = np.exp(-(X - 0.5*Lx)**2/(Lx/10)**2 - (Y - 0.5*Ly)**2/(Lx/10)**2 - (Z - 0.25)**2/(Lz/20)**2)
amp  = wave_amplitude*f0/max(abs(real[0]).max(), abs(imag[0]).max())
M.u_r, M.v_r, M.w_r, M.b_r = M.project(*[amp*mask*a for a in real], 1)
M.u_i, M.v_i, M.w_i, M.b_i = M.project(*[amp*mask*a for a in imag], 1)

# the background jet U(y,z), B(y,z): geostrophic mode of a pressure field
p = np.exp(-(Z - 0.75*Lz)**2/jet_width**2)*np.sin(2*np.pi*Y/Ly)
u = (np.roll(p, -1, axis=1) - p)/(f0*dy)
zero = OM.np.zeros_like(M.u)
u, v, w, b = M.project(M.apply_bc(OM.np.asarray(u)), zero, zero, zero, 0)
scale = jet_amplitude/abs(u).max()
M.u_back = u*scale + u_back0
M.b_back = b*scale

f = Dataset('forcing.cdf', 'w')
for n, s in (('x', nx), ('y', ny), ('z', nz)): f.createDimension(n, s)
f.createVariable('u', 'f8', ('x', 'y', 'z'))[:] = M.u_back[h:-h, h:-h, h:-h]
f.createVariable('b', 'f8', ('x', 'y', 'z'))[:] = M.b_back[h:-h, h:-h, h:-h]
f.close()

M.time = 0.
M.loop(int(runlen/dt))
