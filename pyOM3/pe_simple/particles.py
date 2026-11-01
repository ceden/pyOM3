"""
 Lagrangian particles for pe_simple, for the Lagrangian frequency analysis
 after Shakespeare and Hogg (2017). Port of the grav2D particle code
 (py_src/sw_particles.py) to three dimensions.

 Particles are advected with the model velocity (u, v, w) interpolated
 trilinearly from the staggered C grid at every model time step, with a
 fourth order Adam-Bashforth scheme started with Euler, AB2, AB3 steps.
 The domain is periodic in x and y. In the vertical the particles stay
 between the bottom z = -Lz and the surface z = 0, w vanishes on both by
 the rigid lid (or is the small free surface motion), and the position is
 clamped there.

 Only for pe_simple without topography, one process, numpy backend.

 Provides (class particles, add to the model class bases):
   init_particles(pxyz)        positions (3, npart) in m, x y z with z in (-Lz, 0)
   particles_uvwb()            u, v, w, b of the model state at the particles
   step_particles()            one AB4 time step, call once per model step
 The time stepping uses the velocities at the time of the call, so call
 step_particles AFTER the model time step, in diagnose, with the record of
 the positions taken BEFORE the velocities of the new step, as in grav2D:
 record positions, interpolate and step, record the interpolated values.
"""
import pyOM3                 as OM
from   pyOM3.pe_simple.model import model
import numpy                 as np


class particles(model):

    # ------------------------------------------------------------------
    def init_particles(self, pxyz):
        if self.n_pes > 1:
            raise RuntimeError('particles: one process only')
        self.pxyz = np.array(pxyz, dtype='f8')
        npart = self.pxyz.shape[1]
        self.p_duvw = np.zeros((4, 3, npart))      # AB4 history of dx/dt
        self.p_tau = 0                             # current level in the history
        self.p_nab = 0                             # number of steps done
        self.p_Lz = self.nz*self.dz
        return

    # ------------------------------------------------------------------
    def _interp(self, a, ox, oy, oz, zero_floor=False):
        """trilinear interpolation of interior field a (nz, ny, nx) at the
        particles, with the grid offsets o in cells, periodic in x and y.
        zero_floor: a vanishes at the bottom (for w), a zero layer is
        assumed below the first level and the top value is clamped."""
        nx, ny, nz = self.nx, self.ny, self.nz
        x, y, z = self.pxyz
        fi = x/self.dx - ox
        fj = y/self.dy - oy
        fk = (z + self.p_Lz)/self.dz - oz
        i0 = np.floor(fi).astype(int); ai = fi - i0
        j0 = np.floor(fj).astype(int); aj = fj - j0
        k0 = np.floor(fk).astype(int); ak = fk - k0
        i0 %= nx; i1 = (i0 + 1) % nx
        j0 %= ny; j1 = (j0 + 1) % ny
        if zero_floor:
            # a(k) with a(-1) = 0 below and clamped at the top
            kk0 = np.clip(k0, -1, nz - 1); kk1 = np.clip(k0 + 1, -1, nz - 1)
            def get(k, j, i):
                v = a[np.maximum(k, 0), j, i]
                return np.where(k < 0, 0., v)
        else:
            kk0 = np.clip(k0, 0, nz - 1); kk1 = np.clip(k0 + 1, 0, nz - 1)
            def get(k, j, i):
                return a[k, j, i]
        out = ((1-ak)*((1-aj)*((1-ai)*get(kk0, j0, i0) + ai*get(kk0, j0, i1))
                       +  aj *((1-ai)*get(kk0, j1, i0) + ai*get(kk0, j1, i1)))
               +  ak *((1-aj)*((1-ai)*get(kk1, j0, i0) + ai*get(kk1, j0, i1))
                       +  aj *((1-ai)*get(kk1, j1, i0) + ai*get(kk1, j1, i1))))
        return out

    def _interior(self, a):
        h = self.halo
        return np.asarray(a)[h:-h, h:-h, h:-h]

    def particles_uvwb(self, b=None):
        """u, v, w at the particles, and b when given (interior field or None)"""
        u = self._interp(self._interior(self.u), 1.0, 0.5, 0.5)
        v = self._interp(self._interior(self.v), 0.5, 1.0, 0.5)
        w = self._interp(self._interior(self.w), 0.5, 0.5, 1.0, zero_floor=True)
        if b is None:
            return u, v, w
        return u, v, w, self._interp(b, 0.5, 0.5, 0.5)

    # ------------------------------------------------------------------
    def step_particles(self):
        """one time step of the particles with the current model velocities"""
        u, v, w = self.particles_uvwb()
        self.p_duvw[self.p_tau] = (u, v, w)
        t0 = self.p_tau
        t1, t2, t3 = (t0 - 1) % 4, (t0 - 2) % 4, (t0 - 3) % 4
        D = self.p_duvw
        if   self.p_nab == 0: dd = D[t0]
        elif self.p_nab == 1: dd = 1.5*D[t0] - 0.5*D[t1]
        elif self.p_nab == 2: dd = (23*D[t0] - 16*D[t1] + 5*D[t2])/12.
        else:                 dd = (55*D[t0] - 59*D[t1] + 37*D[t2] - 9*D[t3])/24.
        self.pxyz += self.dt*dd
        self.pxyz[0] %= self.nx*self.dx
        self.pxyz[1] %= self.ny*self.dy
        eps = 1e-3*self.dz
        self.pxyz[2] = np.clip(self.pxyz[2], -self.p_Lz + eps, -eps)
        self.p_tau = (self.p_tau + 1) % 4
        self.p_nab += 1
        return
