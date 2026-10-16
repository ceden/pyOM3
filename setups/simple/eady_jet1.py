"""
eady_jet1: baroclinic instability of a zonal jet in a channel
pyOM2: for_config/eady_jet1.f90, for_config/eady_opv1.f90

A Gaussian jet with cos(pi z/H) vertical structure in thermal wind balance,
perturbed with an Eady mode of random phase. Buoyancy is the prognostic
tracer (eq_of_state=0). With balanced=True the initial state is balanced
with the optimal balance method (eady_opv1) before the integration.

run:    python eady_jet1.py
"""
import sys; sys.path.append('../..')
import numpy as np
import pyOM3 as OM
from pyOM3.pe_realistic.snapshot import snap_nc4
from pyOM3.pe_realistic.diag_opt_balance import opt_balance

Ri, delta  = 400., 0.01             # Richardson number, aspect ratio
Ek, CFL    = 0.05, 0.02             # Ekman number, CFL number
f0, H0     = 1e-4, 200.
N0         = f0/delta
U0         = N0*H0/np.sqrt(Ri)
M0         = np.sqrt(f0*U0/H0)
kmax       = 1./(np.sqrt(2./5)*np.sqrt(1 + Ri)*M0**2/f0**2*H0)   # Eady wavenumber of fastest growth
nx, ny, nz = 60, 120, 20
Lx, Ly     = 2*np.pi/kmax, 4*2*np.pi/kmax
dx, dz     = Lx/nx, H0/nz
dt         = CFL/U0*dx
jet_scale  = 0.05*Ly
b_amp      = 1e-5                   # amplitude of the perturbation
balanced   = False
runlen     = 45*86400.
snapint    = 86400.
n_pes_x, n_pes_y = 1, 1

par = dict(nx=nx, ny=ny, nz=nz, dxt=dx, dyt=Ly/ny, dzt=dz, dt=dt,
           coord_degree=False, set_zero_at_surface=True,
           periodic_in_x=True, periodic_in_y=False,
           eq_of_state=0, implicit_free_surface=True, explicit_free_surface=False,
           epsilon=1e-12*(dx/20e3)**2, max_iterations=5000, superbee_advection=True,
           Ahbi=Ek*f0*dx**4, vmix_scheme='explicit', A_v=Ek*f0*dz**2,
           snapint=snapint)


class eady_jet1(opt_balance, snap_nc4):

    def set_coriolis(self):
        self.coriolis_t = OM.np.asarray(f0 + 0*self.yt[:, None] + 0*self.xt[None, :])

    def set_initial_conditions(self):
        Z, Y, X = np.meshgrid(self.zt_gl, self.yt_gl, self.xt_gl, indexing='ij')
        u0 = U0*np.exp(-(Y - Ly/2)**2/jet_scale**2)*np.cos(np.pi*Z/H0)
        # thermal wind: b_y = -f0 u_z, integrated from the southern wall
        uz = np.gradient(u0, self.zt_gl, axis=0)
        b0 = N0**2*Z - f0*np.cumsum(uz, axis=1)*(Ly/ny)
        # Eady mode with random phase
        d    = f0/N0/kmax
        c1   = (1 + 0.25*(H0/d)**2 - H0/d/np.tanh(H0/d))*complex(1, 0)
        c1   = (np.sqrt(c1)*d/H0 + 0.5)*U0
        A    = (U0 - c1)/U0*H0/d
        phiz = abs(A/d*np.sinh(Z/d) + np.cosh(Z/d)/d)
        ampA, ampB = np.random.default_rng(1234).random(2)
        b = b0 + b_amp*phiz/phiz.max()*(ampA*np.sin(kmax*X) + ampB*np.cos(kmax*X))*np.exp(-(Y - Ly/2)**2/jet_scale**2)
        zeros = np.zeros((nz, ny, nx))
        super().set_initial_conditions(u0, zeros, b, zeros, np.zeros((ny, nx)))


M = eady_jet1(par, n_pes_x=n_pes_x, n_pes_y=n_pes_y)
M.set_coriolis()
M.set_topography()
M.prepare_for_run(snap_file_name='eady_jet1.cdf', show_each_time_step=False)
M.set_initial_conditions()
if balanced:
    M.balance_it(OB_period=7.5*86400., OB_average=15*86400., OB_average_times=5, OB_max_itts=5)
M.loop(int(runlen/dt))
