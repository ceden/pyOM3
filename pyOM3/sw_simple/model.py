"""
The shallow-water model sw_simple: one layer of fluid of mean thickness H0
on the f plane, doubly periodic, dimensional. Port of sw_model.py of the
grav2D code of the Lagrangian-filtering draft (Nov 2026).

The equations, with h the displacement of the surface (the thickness is
H0 + h) and the velocity (u, v),

  dh/dt = - div( (H0 + h) u )
  du/dt = + q (H0 + h) v + f0 v - dK/dx - g dh/dx + mixing
  dv/dt = - q (H0 + h) u - f0 u - dK/dy - g dh/dy + mixing

with the relative potential vorticity q = zeta/(H0 + h), zeta = dv/dx - du/dy,
and K = (u^2 + v^2)/2. The discretisation is that of the original code: the
C grid, the energy conserving form of the vorticity term (Sadourny) with the
thickness fluxes, the Coriolis term f0 by the four-point average, the
biharmonic friction - Ahbi del^4 u and diffusion - Khbi del^4 h.

The original code is nondimensional (Ro, Bu = c^2, f0 = 1). It is this model
with the scales L (length), 1/f0 (time), U = Ro f0 L (velocity), N = f0 U L/g
(the displacement h) and H0 = Bu f0^2 L^2/g. Its h is g h/(f0 U L), its c^2
is g H0/(f0 L)^2, its Ahbi is Ahbi/(f0 L^4).

The grid: nx x ny cells of size dx x dy, no halo points (periodic by
rolls). Arrays are (ny, nx), y first as in pyOM3. h is at the cell centres
x = (i + 1/2) dx, y = (j + 1/2) dy, u at the east faces x = (i + 1) dx, v at
the north faces y = (j + 1) dy, zeta at the north-east corners. One process,
numpy or JAX (OM.np).

Parameters (the dictionary of the constructor, set as attributes):
  nx, ny, dx, dy        grid
  dt                    time step (s)
  f0, g, H0             Coriolis parameter (1/s), gravity (m/s^2), mean thickness (m)
  Ahbi, Khbi            biharmonic friction and diffusion (m^4/s), default 0
  time_scheme           'AB3' (default), 'AB2' (with AB_eps, default 0), 'AB4', 'RK3'

The time schemes are those of the original code. AB3 starts with two RK3
steps, AB4 with two RK3 steps and one AB3 step; the RK3 steps store the
tendency of their last stage, which is the history of the following
Adams-Bashforth steps. The mixing is a forward step after the
Adams-Bashforth update, from the state at its beginning. The RK3 steps have
no mixing (also the start of AB3 and AB4), and RK3 with mixing is refused.

Provides (class model):
  set_initial_conditions(u, v, h)   the state, arrays (ny, nx)
  loop(max_steps)                   max_steps time steps, diagnose(n, tau) after each
  tendency(u, v, h, rho=1)          the tendency (du, dv, dh) without mixing;
                                    rho multiplies the non-linear terms (1: the
                                    model, 0: the linear model, the ramp of
                                    optimal balance)
  tendency_nonlin(u, v, h)          the non-linear part of the tendency
  tendency_linear(u, v, h)          the linear part
  biharmonic(a, A)                  - A del^4 a
  energy()                          the total energy per unit area, domain mean
"""
import pyOM3            as OM
import numpy            as np_
from   functools        import partial


class model:

    def __init__(self, parameter={}):
        for key in parameter: setattr(self, key, parameter[key])
        for key, val in (('Ahbi', 0.), ('Khbi', 0.), ('time_scheme', 'AB3'), ('AB_eps', 0.)):
            if not hasattr(self, key): setattr(self, key, val)
        if self.time_scheme not in ('AB2', 'AB3', 'AB4', 'RK3'):
            raise ValueError('sw_simple: time_scheme AB2, AB3, AB4 or RK3, not %r' % (self.time_scheme,))
        if self.time_scheme == 'RK3' and (self.Ahbi > 0. or self.Khbi > 0.):
            raise ValueError('sw_simple: no mixing with time_scheme RK3')
        self.my_pe, self.n_pes = 0, 1
        self.Lx, self.Ly = self.nx*self.dx, self.ny*self.dy
        self.xt = (np_.arange(self.nx) + 0.5)*self.dx
        self.yt = (np_.arange(self.ny) + 0.5)*self.dy
        self.xu = (np_.arange(self.nx) + 1.0)*self.dx
        self.yu = (np_.arange(self.ny) + 1.0)*self.dy
        self.allocate_workspace()
        return

    def allocate_workspace(self):
        nlev = {'AB2': 2, 'RK3': 2, 'AB3': 3, 'AB4': 4}[self.time_scheme]
        z2 = lambda: OM.np.zeros((self.ny, self.nx), OM.prec)
        self.u, self.v, self.h = z2(), z2(), z2()
        self.du = OM.np.zeros((nlev, self.ny, self.nx), OM.prec)
        self.dv = OM.np.zeros((nlev, self.ny, self.nx), OM.prec)
        self.dh = OM.np.zeros((nlev, self.ny, self.nx), OM.prec)
        self.du_mix, self.dv_mix, self.dh_mix = z2(), z2(), z2()
        return

    def set_initial_conditions(self, u, v, h):
        self.u, self.v, self.h = (OM.np.asarray(np_.array(a, dtype='f8')) for a in (u, v, h))
        return

    def prepare_for_run(self):
        return

    # ------------------------------------------------------------------
    # the tendencies
    @partial(OM.jaxjit, static_argnums=0)
    def tendency(self, u, v, h, rho=1.):
        """
        the tendency (du, dv, dh) of the state without the mixing, the
        non-linear terms multiplied by rho (the expressions and their order
        are those of the original code)
        """
        np = OM.np
        dx, dy, H0, g, f0 = self.dx, self.dy, self.H0, self.g, self.f0
        E, W = (lambda a: np.roll(a, -1, axis=1)), (lambda a: np.roll(a, 1, axis=1))
        N, S = (lambda a: np.roll(a, -1, axis=0)), (lambda a: np.roll(a, 1, axis=0))
        # the thickness: dh = - div( (H0 + rho h) u )
        dh = -H0*((u - W(u))/dx + (v - S(v))/dy)
        fe, fn = u*0.5*(h + E(h)), v*0.5*(h + N(h))
        dh = dh - rho*((fe - W(fe))/dx + (fn - S(fn))/dy)
        # the thickness fluxes and the potential vorticity rho zeta/(H0 + rho h)
        hc = rho*h + H0
        fe, fn = u*0.5*(hc + E(hc)), v*0.5*(hc + N(hc))
        q = rho*((E(v) - v)/dx - (N(u) - u)/dy)
        q = q/(0.25*(hc + E(hc) + N(hc) + E(N(hc))))
        du = +0.5*(q*0.5*(fn + E(fn)) + S(q)*0.5*(S(fn) + E(S(fn))))
        dv = -0.5*(q*0.5*(fe + N(fe)) + W(q)*0.5*(W(fe) + N(W(fe))))
        Ke = 0.5*(0.5*(u**2 + W(u)**2) + 0.5*(v**2 + S(v)**2))
        du = du - rho*(E(Ke) - Ke)/dx
        dv = dv - rho*(N(Ke) - Ke)/dy
        # the pressure gradient
        du = du - g*(E(h) - h)/dx
        dv = dv - g*(N(h) - h)/dy
        # the Coriolis term
        du = du + f0*0.25*(v + S(v) + E(v) + E(S(v)))
        dv = dv - f0*0.25*(u + W(u) + N(u) + N(W(u)))
        return du, dv, dh

    @partial(OM.jaxjit, static_argnums=0)
    def tendency_nonlin(self, u, v, h):
        """the non-linear part of the tendency, (du, dv, dh)"""
        np = OM.np
        dx, dy, H0 = self.dx, self.dy, self.H0
        E, W = (lambda a: np.roll(a, -1, axis=1)), (lambda a: np.roll(a, 1, axis=1))
        N, S = (lambda a: np.roll(a, -1, axis=0)), (lambda a: np.roll(a, 1, axis=0))
        fe, fn = u*0.5*(h + E(h)), v*0.5*(h + N(h))
        dh = -((fe - W(fe))/dx + (fn - S(fn))/dy)
        hc = h + H0
        fe, fn = u*0.5*(hc + E(hc)), v*0.5*(hc + N(hc))
        q = (E(v) - v)/dx - (N(u) - u)/dy
        q = q/(0.25*(hc + E(hc) + N(hc) + E(N(hc))))
        du = +0.5*(q*0.5*(fn + E(fn)) + S(q)*0.5*(S(fn) + E(S(fn))))
        dv = -0.5*(q*0.5*(fe + N(fe)) + W(q)*0.5*(W(fe) + N(W(fe))))
        Ke = 0.5*(0.5*(u**2 + W(u)**2) + 0.5*(v**2 + S(v)**2))
        du = du - (E(Ke) - Ke)/dx
        dv = dv - (N(Ke) - Ke)/dy
        return du, dv, dh

    @partial(OM.jaxjit, static_argnums=0)
    def tendency_linear(self, u, v, h):
        """the linear part of the tendency, (du, dv, dh)"""
        np = OM.np
        dx, dy, H0, g, f0 = self.dx, self.dy, self.H0, self.g, self.f0
        E, W = (lambda a: np.roll(a, -1, axis=1)), (lambda a: np.roll(a, 1, axis=1))
        N, S = (lambda a: np.roll(a, -1, axis=0)), (lambda a: np.roll(a, 1, axis=0))
        dh = -H0*((u - W(u))/dx + (v - S(v))/dy)
        du = -g*(E(h) - h)/dx + f0*0.25*(v + S(v) + E(v) + E(S(v)))
        dv = -g*(N(h) - h)/dy - f0*0.25*(u + W(u) + N(u) + N(W(u)))
        return du, dv, dh

    @partial(OM.jaxjit, static_argnums=0)
    def biharmonic(self, a, A):
        """- A del^4 a (the original code's form)"""
        np = OM.np
        dx, dy = self.dx, self.dy
        E, W = (lambda a: np.roll(a, -1, axis=1)), (lambda a: np.roll(a, 1, axis=1))
        N, S = (lambda a: np.roll(a, -1, axis=0)), (lambda a: np.roll(a, 1, axis=0))
        fe, fn = -A*(E(a) - a)/dx, -A*(N(a) - a)/dy
        L = (fe - W(fe))/dx + (fn - S(fn))/dy
        fe, fn = (E(L) - L)/dx, (N(L) - L)/dy
        return (fe - W(fe))/dx + (fn - S(fn))/dy

    def _mixing(self, u, v, h):
        if self.Ahbi > 0.:
            self.du_mix, self.dv_mix = self.biharmonic(u, self.Ahbi), self.biharmonic(v, self.Ahbi)
        if self.Khbi > 0.:
            self.dh_mix = self.biharmonic(h, self.Khbi)
        return

    def _store(self, tau, d):
        self.du = OM.modify_array(self.du, (tau,), d[0])
        self.dv = OM.modify_array(self.dv, (tau,), d[1])
        self.dh = OM.modify_array(self.dh, (tau,), d[2])
        return

    # ------------------------------------------------------------------
    # the time steps
    def rk3_time_step(self, tau, rho=1., dt=None):
        """
        y* = y_n + dt/3 f(y_n), y** = y_n + dt/2 f(y*), y_n+1 = y_n + dt f(y**);
        the tendency of the last stage is stored at tau
        """
        dt = self.dt if dt is None else dt
        un, vn, hn = self.u, self.v, self.h
        d = self.tendency(self.u, self.v, self.h, rho)
        self.u, self.v, self.h = un + dt/3.*d[0], vn + dt/3.*d[1], hn + dt/3.*d[2]
        d = self.tendency(self.u, self.v, self.h, rho)
        self.u, self.v, self.h = un + dt/2.*d[0], vn + dt/2.*d[1], hn + dt/2.*d[2]
        d = self.tendency(self.u, self.v, self.h, rho)
        self.u, self.v, self.h = un + dt*d[0], vn + dt*d[1], hn + dt*d[2]
        self._store(tau, d)
        return

    def ab_time_step(self, coef, levels, rho=1., dt=None, mixing=True):
        """
        an Adams-Bashforth step with the coefficients coef of the tendencies
        at the time levels (the first the current one), then the mixing
        """
        dt = self.dt if dt is None else dt
        d = self.tendency(self.u, self.v, self.h, rho)
        self._store(levels[0], d)
        if mixing: self._mixing(self.u, self.v, self.h)
        for X, D in (('u', self.du), ('v', self.dv), ('h', self.dh)):
            inc = coef[0]*D[levels[0]]
            for c, l in zip(coef[1:], levels[1:]):
                inc = inc + c*D[l]
            setattr(self, X, getattr(self, X) + dt*inc)
        if mixing and self.Ahbi > 0.:
            self.u, self.v = self.u + dt*self.du_mix, self.v + dt*self.dv_mix
        if mixing and self.Khbi > 0.:
            self.h = self.h + dt*self.dh_mix
        return

    def time_steps(self, max_steps, rho=None, dt=None, mixing=True, after_step=None):
        """
        max_steps time steps of the scheme from its start (the first steps
        of AB3 and AB4 are RK3 steps). rho(n): the factor of the non-linear
        terms in step n (default 1); dt: the time step (default self.dt,
        negative for a backward integration); after_step(n, tau) is called
        after each step
        """
        sch = self.time_scheme
        nlev = self.du.shape[0]
        lev = list(range(nlev))[::-1]          # tau, taum1, ... = nlev-1, ..., 0
        AB2 = (1.5 + self.AB_eps, -(0.5 + self.AB_eps))
        AB3 = (23./12., -16./12., 5./12.)
        AB4 = (55./24., -59./24., 37./24., -9./24.)
        for n in range(max_steps):
            r = 1. if rho is None else rho(n)
            tau = lev[0]
            if sch == 'RK3':
                self.rk3_time_step(0, r, dt)
            elif sch == 'AB2':
                self.ab_time_step((1., 0.) if n == 0 else AB2, lev[:2], r, dt, mixing)
            elif n < 2:
                self.rk3_time_step(tau, r, dt)
            elif sch == 'AB3' or n == 2:
                self.ab_time_step(AB3, lev[:3], r, dt, mixing)
            else:
                self.ab_time_step(AB4, lev[:4], r, dt, mixing)
            if after_step is not None: after_step(n, tau)
            if sch != 'RK3': lev = [(l + 1) % nlev for l in lev]
        return

    def loop(self, max_steps=1000000):
        """max_steps time steps, diagnose(n, tau) after each"""
        self.time_steps(max_steps, after_step=self.diagnose)
        return

    def diagnose(self, n, tau):
        return

    # ------------------------------------------------------------------
    def energy(self):
        """the domain mean of the energy per unit area / rho_0:
        (H0 + h)(u^2 + v^2)/2 + g h^2/2 (u^2, v^2 averaged to the centres)"""
        np = OM.np
        u2 = 0.5*(self.u**2 + np.roll(self.u, 1, axis=1)**2)
        v2 = 0.5*(self.v**2 + np.roll(self.v, 1, axis=0)**2)
        return float(np.mean((self.H0 + self.h)*(u2 + v2)/2 + self.g*self.h**2/2))
