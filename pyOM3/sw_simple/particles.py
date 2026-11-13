"""
Lagrangian particles for sw_simple, port of sw_particles.py of the grav2D
code (Nov 2026), with its time schemes and its interpolation.

The particles are advected with the model velocity interpolated
bilinearly from the C grid, after every model time step with the velocity
of the new state (as in the original code: diagnose, after the model step).
Each particle keeps the index (i, j) of its cell; the interpolation of u, v
and h uses the grid points around that cell, and the index is moved by one
cell after the step if the particle left it (the step must be shorter than
a cell). Periodic in x and y, positions in m in (0, Lx) x (0, Ly).

The schemes (particle_scheme): 'Euler', 'RK3', 'AB2', 'AB3', 'AB4'; the
Adams-Bashforth schemes start with Euler, AB2 (and AB3) steps. RK3 takes
its stages in the velocity field of the new state, frozen during the step,
and is only first order in time, as the other schemes: the field is that
of the end of the model step, not of the stage times.

Provides (class particles, add to the model class bases):
  init_particles(pxy, scheme='AB4')   positions (2, npart) in m
  particles_uvh()                     u, v, h of the model state at the particles
  step_particles()                    one time step, call once per model step
In loop, step_particles is called in diagnose; record the positions before
calling super().diagnose to have them at the beginning of the step.
"""
import pyOM3                  as OM
from   pyOM3.sw_simple.model  import model
import numpy                  as np


class particles(model):

    def init_particles(self, pxy, scheme='AB4'):
        if scheme not in ('Euler', 'RK3', 'AB2', 'AB3', 'AB4'):
            raise ValueError('particles: scheme Euler, RK3, AB2, AB3 or AB4, not %r' % (scheme,))
        self.particle_scheme = scheme
        self.pxy = np.array(pxy, dtype='f8')
        npart = self.pxy.shape[1]
        nlev = {'Euler': 1, 'RK3': 1, 'AB2': 2, 'AB3': 3, 'AB4': 4}[scheme]
        self.puvh = np.zeros((3, npart, nlev))
        self.p_lev = list(range(nlev))[::-1]          # current level first
        self.p_n = 0                                  # steps done
        # the domain, and the cells of the particles: the nearest centre, on
        # a face the western (southern) cell, as the original code
        self.pxy[0] = np.clip(self.pxy[0], 0., self.Lx)
        self.pxy[1] = np.clip(self.pxy[1], 0., self.Ly)
        self.pij = np.zeros((2, npart), dtype=np.int64)
        self.pij[0] = np.minimum(np.ceil(self.pxy[0]/self.dx - 1.), self.nx - 1).astype(np.int64)
        self.pij[1] = np.minimum(np.ceil(self.pxy[1]/self.dy - 1.), self.ny - 1).astype(np.int64)
        self._particles_periodic()
        self.puvh[:, :, self.p_lev[0]] = self.particles_uvh()
        return

    def _particles_periodic(self):
        pij, pxy = self.pij, self.pxy
        for d, n, L in ((0, self.nx, self.Lx), (1, self.ny, self.Ly)):
            hi, lo = pij[d] > n - 1, pij[d] < 0
            pxy[d] = np.where(hi, pxy[d] - L, np.where(lo, pxy[d] + L, pxy[d]))
            pij[d] = np.where(hi, pij[d] - n, np.where(lo, pij[d] + n, pij[d]))
        return

    def _bilinear(self, a, i, j, x0, x1, y0, y1):
        """a at the particles from the points (i-1, i) x (j-1, j) of a at
        x0 (of i-1), x1 (of i), y0 (of j-1), y1 (of j)"""
        x, y = self.pxy
        dx, dy = self.dx, self.dy
        xe, xw = x1 - x, x - x0
        yn, ys = y1 - y, y - y0
        im, ip = (i - 1) % self.nx, i % self.nx
        jm, jp = (j - 1) % self.ny, j % self.ny
        xeys, xwys = xe*ys/(dx*dy), xw*ys/(dx*dy)
        xeyn, xwyn = xe*yn/(dx*dy), xw*yn/(dx*dy)
        return a[jp, im]*xeys + a[jp, ip]*xwys + a[jm, im]*xeyn + a[jm, ip]*xwyn

    def particles_uvh(self):
        """u, v, h of the model state at the particles, (3, npart)"""
        u, v, h = (np.asarray(a) for a in (self.u, self.v, self.h))
        dx, dy = self.dx, self.dy
        x, y = self.pxy
        i, j = self.pij
        xt = lambda k: (k + 0.5)*dx; xu = lambda k: (k + 1.0)*dx
        yt = lambda k: (k + 0.5)*dy; yu = lambda k: (k + 1.0)*dy
        # u between the faces of the cell in x, the centres around in y
        ju = np.where(y > yt(j), j + 1, j)
        pu = self._bilinear(u, i, ju, xu(i - 1), xu(i), yt(ju - 1), yt(ju))
        # v between the centres around in x, the faces of the cell in y
        iv = np.where(x > xt(i), i + 1, i)
        pv = self._bilinear(v, iv, j, xt(iv - 1), xt(iv), yu(j - 1), yu(j))
        # h between the centres around
        pi = np.where(x > xt(i), i + 1, i)
        pj = np.where(y > yt(j), j + 1, j)
        ph = self._bilinear(h, pi, pj, xt(pi - 1), xt(pi), yt(pj - 1), yt(pj))
        return np.array([pu, pv, ph])

    def step_particles(self):
        """one time step of the particles with the current model state"""
        dt, l = self.dt, self.p_lev
        P = self.puvh
        P[:, :, l[0]] = self.particles_uvh()
        sch, n = self.particle_scheme, self.p_n
        if sch == 'Euler' or (sch != 'RK3' and n == 0):
            self.pxy += dt*P[:2, :, l[0]]
        elif sch == 'RK3':
            xy = self.pxy.copy()
            self.pxy = xy + 1./3.*dt*P[:2, :, l[0]]
            P[:, :, l[0]] = self.particles_uvh()
            self.pxy = xy + 1./2.*dt*P[:2, :, l[0]]
            P[:, :, l[0]] = self.particles_uvh()
            self.pxy = xy + dt*P[:2, :, l[0]]
        elif sch == 'AB2' or n == 1:
            self.pxy += dt*(1.5*P[:2, :, l[0]] - 0.5*P[:2, :, l[1]])
        elif sch == 'AB3' or n == 2:
            self.pxy += dt*(23./12.*P[:2, :, l[0]] - 16./12.*P[:2, :, l[1]] + 5./12.*P[:2, :, l[2]])
        else:
            self.pxy += dt*(55./24.*P[:2, :, l[0]] - 59./24.*P[:2, :, l[1]]
                            + 37./24.*P[:2, :, l[2]] - 9./24.*P[:2, :, l[3]])
        # the cell index, moved by one if the particle left the cell
        x, y = self.pxy
        i, j = self.pij
        self.pij[0] = np.where(x > (i + 1.)*self.dx, i + 1, np.where(x < i*self.dx, i - 1, i))
        self.pij[1] = np.where(y > (j + 1.)*self.dy, j + 1, np.where(y < j*self.dy, j - 1, j))
        self._particles_periodic()
        nlev = P.shape[2]
        self.p_lev = [(k + 1) % nlev for k in l]
        self.p_n += 1
        return

    def diagnose(self, n, tau):
        self.step_particles()
        super().diagnose(n, tau)
        return
