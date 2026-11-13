"""
 Lagrangian particles for pe_realistic, for the Lagrangian frequency analysis
 after Shakespeare and Hogg (2017). The scheme of pe_simple/particles.py
 (trilinear interpolation from the C grid, fourth order Adams-Bashforth
 started with Euler, AB2, AB3) on the grid of pe_realistic: the
 pseudo-spherical metric or the Cartesian one, variable dx, dy, dz, the land
 mask, and the domain decomposition of both backends.

 THE POSITIONS are in the model's own coordinates. x and y are xt, yt
 (degrees with coord_degree, metres otherwise), z is the height above the
 bottom of the deepest level, as zt and zw (zw at the top of the column is
 the total depth H, so the depth below the surface is H - z). A periodic
 direction wraps, a closed one clamps the position to the outer faces of
 the domain, the vertical to 1e-3 of the outer cells inside. With
 coord_degree a particle moves with
     dx/dt = u/(a cos y) 180/pi ,  dy/dt = v/a 180/pi ,  dz/dt = w
 at its own latitude y, a = self.radius.

 THE INTERPOLATION is trilinear between the points of each field's own
 staggering: u at (xu, yt, zt), v at (xt, yu, zt), w at (xt, yt, zw) with
 w = 0 at the bottom face of the deepest level, the tracers at (xt, yt, zt).
 Outside the outermost points the value of the outermost point is used (in
 the vertical above the top zt and below the deepest zt, as pe_simple). The
 velocities are masked in the model, so they go to zero towards land. The
 tracers are not masked in the model, so they are interpolated with the
 weights times maskT, normalised by the sum of those weights.

 LAND. A step that would end in a dry T cell is not taken: the particle
 keeps its position and the step is counted in p_frozen. Its
 Adams-Bashforth history keeps the velocity of the old position.

 THE TIME STEP. step_particles uses the velocities at the time of the call.
 Call it AFTER the model time step, as pe_simple: the positions of the
 record are those BEFORE the step, the values those interpolated at them
 from the new model state, and the step uses the same values.

 THE DECOMPOSITION. Every PE holds every particle. Each PE interpolates the
 particles whose T cell lies in its own interior block, from its local
 fields (the stencil reaches one point into the halo, so halo_points >= 1;
 u, v, temp, salt leave the model step with their halos exchanged, w does
 not and is exchanged here, in a copy), and contributes zero for the others. A global sum then gives each particle
 exactly one nonzero term, so the result does not depend on the
 decomposition (jax: shard_map + psum over the shard mesh, any number of
 processes; MPI: Allreduce on the model's communicator; one PE: the
 identity). Each sum moves 3 numbers per particle per step (u, v, w), plus
 2 at the record steps (temp, salt) and 1 for the land check.

 On jax the positions and the history are replicated device arrays (the
 same on every device); host copies are made only for records and restarts
 (particles_state). The model fields are arguments of the jitted functions,
 never closed over (several processes).

 Provides (a mixin, add it to the model class bases):
   init_particles(x, y, z, moving=None)  positions (model coordinates); particles
                                         in dry cells are dropped; moving = 0 marks
                                         fixed points (moorings, Eulerian series)
   particle_depth_to_z(depth)            z of a depth below the surface [m]
   step_particles(record=False, host=True)  one AB4 step; with record=True returns the
                                         host dict x, y, z (before the step), u, v, w,
                                         temp, salt (at those positions); collective
   particles_state() / particles_restore(d)   the state for a restart (host numpy)
 Attributes: p_n (number), p_ids (index into the positions given to
 init_particles), p_moving, p_frozen (refused steps so far).
"""
import numpy as np__
import pyOM3 as OM
from functools import partial


# Adams-Bashforth weights of the levels t0 (newest), t1, t2, t3 for the
# number of steps already done: Euler, AB2, AB3, AB4 (as pe_simple)
_AB = ((1.0, 0.0, 0.0, 0.0),
       (1.5, -0.5, 0.0, 0.0),
       (23./12., -16./12., 5./12., 0.0),
       (55./24., -59./24., 37./24., -9./24.))


class particles:

    # ------------------------------------------------------------------
    # set up
    # ------------------------------------------------------------------
    def _p_grid(self):
        """host constants of the interpolation: the global coordinates, the
        periodic extensions, the domain bounds, the block of this PE"""
        nx, ny, nz, h = self.nx, self.ny, self.nz, self.halo
        if h < 1:
            raise ValueError('particles: halo_points >= 1 needed')
        g = {}
        xt, xu = np__.asarray(self.xt_gl, float), np__.asarray(self.xu_gl, float)
        yt, yu = np__.asarray(self.yt_gl, float), np__.asarray(self.yu_gl, float)
        zt, zw = np__.asarray(self.zt_gl, float), np__.asarray(self.zw_gl, float)
        # the outer faces of the domain (u_centered_grid: xt(0) = xu(0) - dxt(0)/2)
        g['x_west'] = float(xu[0] - 2.0*(xu[0] - xt[0])); g['Lx'] = float(xu[-1] - g['x_west'])
        g['y_south'] = float(yu[0] - 2.0*(yu[0] - yt[0])); g['y_north'] = float(yu[-1])
        g['z_bot'] = float(zw[0] - np__.asarray(self.dzt)[h]); g['z_top'] = float(zw[-1])
        g['eps_bot'] = 1e-3*float(np__.asarray(self.dzt)[h]); g['eps_top'] = 1e-3*float(np__.asarray(self.dzt)[-h-1])
        g['px'], g['py'] = bool(self.periodic_in_x), bool(self.periodic_in_y)
        L = g['Lx']
        ext = lambda c, L: np__.concatenate([[c[-1] - L], c, [c[0] + L]])
        g['xt'] = ext(xt, L) if g['px'] else xt
        g['xu'] = ext(xu, L) if g['px'] else xu
        Ly = g['y_north'] - g['y_south']
        g['yt'] = ext(yt, Ly) if g['py'] else yt
        g['yu_c'] = ext(yu, Ly) if g['py'] else yu
        g['Ly'] = Ly
        g['yu'] = yu; g['xu_cell'] = xu                    # cell faces for the owner search
        g['zt'] = zt
        g['zw0'] = np__.concatenate([[g['z_bot']], zw])    # w: the bottom face (w = 0) and the tops
        g['zw'] = zw
        g['nx'], g['ny'], g['nz'] = nx, ny, nz
        g['deg'] = bool(getattr(self, 'coord_degree', False))
        g['radius'] = float(self.radius)
        if OM.jax:
            g['bx'] = nx//self.n_pes_x; g['by'] = ny//self.n_pes_y     # interior points per shard
        else:
            g['ox'] = int(self._blk0_x); g['oy'] = int(self._blk0_y)
            g['bx'] = self.Nx - 2*h; g['by'] = self.Ny - 2*h
        return g

    def particle_depth_to_z(self, depth):
        """z (height above the bottom of the deepest level) of a depth below the surface"""
        return float(np__.asarray(self.zw_gl)[-1]) - np__.asarray(depth, float)

    def init_particles(self, x, y, z, moving=None, quiet=False):
        """positions in the model coordinates (1-D arrays of one length);
        moving = 0 for fixed points. Particles in dry cells are dropped."""
        self._pg = self._p_grid()
        self._p_fns = {}
        x = np__.asarray(x, float).ravel(); y = np__.asarray(y, float).ravel(); z = np__.asarray(z, float).ravel()
        if not (len(x) == len(y) == len(z)):
            raise ValueError('init_particles: x, y, z of different length')
        mv = np__.ones(len(x)) if moving is None else np__.asarray(moving, float).ravel()
        P = self._p_wrap_np(np__.stack([x, y, z]))
        wet = np__.asarray(self._p_host(self._p_wet_fn()(self.maskT, self._p_rep(P)))) > 0.5
        self.p_ids = np__.nonzero(wet)[0].astype(np__.int64)
        self.p_n = int(len(self.p_ids))
        if self.p_n == 0:
            raise ValueError('init_particles: no particle in a wet cell')
        self.p_pos = self._p_rep(P[:, wet])
        self.p_hist = self._p_rep(np__.zeros((4, 3, self.p_n)))
        self.p_moving_np = mv[wet]
        self.p_moving = self._p_rep(self.p_moving_np)
        self.p_tau = 0; self.p_nab = 0; self._p_fz = self._p_rep(np__.zeros(()))
        if self.my_pe == 0 and not quiet:
            print('particles: %d of %d positions in wet cells (%d moving, %d fixed)'
                  % (self.p_n, len(x), int(self.p_moving_np.sum()), self.p_n - int(self.p_moving_np.sum())),
                  flush=True)
        return

    def _p_wrap_np(self, P):
        g = self._pg; P = P.copy()
        if g['px']:
            P[0] = g['x_west'] + np__.mod(P[0] - g['x_west'], g['Lx'])
        else:
            P[0] = np__.clip(P[0], g['x_west'], g['x_west'] + g['Lx'])
        if g['py']:
            P[1] = g['y_south'] + np__.mod(P[1] - g['y_south'], g['Ly'])
        else:
            P[1] = np__.clip(P[1], g['y_south'], g['y_north'])
        P[2] = np__.clip(P[2], g['z_bot'] + g['eps_bot'], g['z_top'] - g['eps_top'])
        return P

    # ------------------------------------------------------------------
    # the backends: replicated arrays, host copies, the global sum
    # ------------------------------------------------------------------
    def _p_rep(self, a):
        """a host array as a replicated array of the backend"""
        a = np__.asarray(a, np__.float64)
        if not OM.jax:
            return a
        sh = OM.jax.sharding.NamedSharding(self.mesh, OM.jax.sharding.PartitionSpec())
        return OM.jax.make_array_from_callback(a.shape, sh, lambda idx: a[idx])

    def _p_host(self, a):
        """host copy of a replicated array (every process holds one)"""
        if OM.jax:
            return np__.asarray(a.addressable_data(0))
        return np__.asarray(a)

    # ------------------------------------------------------------------
    # the interpolation core, for both backends (xp = OM.np)
    # ------------------------------------------------------------------
    @staticmethod
    def _p_axis(xp, c, x, periodic):
        """bracket of x in the coordinates c: (g0, g1, a) with the unwrapped
        global indices of the two points (-1 and n are the periodic images)
        and the weight of the second; outside a closed axis the outermost
        point (a clipped to 0 or 1)"""
        n = c.shape[0]
        e = xp.clip(xp.searchsorted(c, x, side='right') - 1, 0, n - 2)
        a = xp.clip((x - c[e])/(c[e + 1] - c[e]), 0.0, 1.0)
        if periodic:                       # c has the image of the last point in front
            return e - 1, e, a
        return e, e + 1, a

    def _p_core(self, fields, maskT, P, ox, oy, nxl, nyl):
        """the contributions of this block: for every name in fields
        ((name, kind, array) with kind 'u', 'v', 'w' or 't') the value at the
        particles it owns, zero for the others; plus the owner mask"""
        xp = OM.np; g = self._pg; h = self.halo
        x, y, z = P[0], P[1], P[2]
        cst = lambda a: xp.asarray(a)
        # the owner: the T cell of the particle
        ic = xp.clip(xp.searchsorted(cst(g['xu_cell']), x, side='right'), 0, g['nx'] - 1)
        jc = xp.clip(xp.searchsorted(cst(g['yu']), y, side='right'), 0, g['ny'] - 1)
        own = (ic >= ox) & (ic < ox + nxl) & (jc >= oy) & (jc < oy + nyl)
        Nxl, Nyl, Nz = nxl + 2*h, nyl + 2*h, g['nz'] + 2*h
        lx = lambda gi: xp.clip(gi - ox + h, 0, Nxl - 1)
        ly = lambda gj: xp.clip(gj - oy + h, 0, Nyl - 1)
        lz = lambda gk: xp.clip(gk + h, 0, Nz - 1)
        ax = {'t': self._p_axis(xp, cst(g['xt']), x, g['px']),
              'u': self._p_axis(xp, cst(g['xu']), x, g['px'])}
        ay = {'t': self._p_axis(xp, cst(g['yt']), y, g['py']),
              'u': self._p_axis(xp, cst(g['yu_c']), y, g['py'])}
        az = {'t': self._p_axis(xp, cst(g['zt']), z, False),
              'w': self._p_axis(xp, cst(g['zw0']), z, False)}
        out = []
        for name, kind, a in fields:
            sx = 'u' if kind == 'u' else 't'
            sy = 'u' if kind == 'v' else 't'
            i0, i1, ai = ax[sx]; j0, j1, aj = ay[sy]
            if kind == 'w':
                e0, e1, ak = az['w']; k0, k1 = e0 - 1, e1 - 1          # face index, -1 = the bottom (w = 0)
            else:
                k0, k1, ak = az['t']
            I0, I1, J0, J1, K0, K1 = lx(i0), lx(i1), ly(j0), ly(j1), lz(k0), lz(k1)
            wts = (((1 - ak), K0), (ak, K1))
            num = 0.0; den = 0.0
            for wk, K in wts:
                for wj, J in (((1 - aj), J0), (aj, J1)):
                    for wi, I in (((1 - ai), I0), (ai, I1)):
                        wgt = wk*wj*wi
                        v = a[K, J, I]
                        if kind == 'w':
                            v = xp.where((K == K0) & (k0 < 0), 0.0, v)
                        if kind == 't':
                            m = maskT[K, J, I]
                            num = num + wgt*m*v; den = den + wgt*m
                        else:
                            num = num + wgt*v
            val = num/xp.where(den > 0, den, 1.0) if kind == 't' else num
            if kind == 't':
                val = xp.where(den > 0, val, 0.0)
            out.append(xp.where(own, val, 0.0))
        return xp.stack(out) if out else xp.zeros((0,) + x.shape), own

    def _p_wet_core(self, maskT, P, ox, oy, nxl, nyl):
        """1 where the T cell of the particle is wet, on the owner; 0 elsewhere"""
        xp = OM.np; g = self._pg; h = self.halo
        x, y, z = P[0], P[1], P[2]
        cst = lambda a: xp.asarray(a)
        ic = xp.clip(xp.searchsorted(cst(g['xu_cell']), x, side='right'), 0, g['nx'] - 1)
        jc = xp.clip(xp.searchsorted(cst(g['yu']), y, side='right'), 0, g['ny'] - 1)
        kc = xp.clip(xp.searchsorted(cst(g['zw']), z, side='right'), 0, g['nz'] - 1)
        own = (ic >= ox) & (ic < ox + nxl) & (jc >= oy) & (jc < oy + nyl)
        I = xp.clip(ic - ox + h, 0, nxl + 2*h - 1); J = xp.clip(jc - oy + h, 0, nyl + 2*h - 1)
        return xp.where(own, maskT[kc + h, J, I], 0.0)

    # ------------------------------------------------------------------
    # the wrapped functions: sample (u, v, w [, temp, salt]) and the land check
    # ------------------------------------------------------------------
    def _p_sample_fn(self, kinds):
        """f(arrays..., maskT, P) -> (len(kinds), N) replicated, the global sum"""
        key = ('sample', kinds)
        if key in self._p_fns:
            return self._p_fns[key]
        g = self._pg
        def core(arrs, maskT, P, ox, oy):
            return self._p_core([(str(i), k, a) for i, (k, a) in enumerate(zip(kinds, arrs))],
                                maskT, P, ox, oy, g['bx'], g['by'])[0]
        if OM.jax:
            jax = OM.jax; Pspec = jax.sharding.PartitionSpec
            bx, by = g['bx'], g['by']
            def local(*args):
                arrs, maskT, P = args[:-2], args[-2], args[-1]
                ox = jax.lax.axis_index('x')*bx; oy = jax.lax.axis_index('y')*by
                return jax.lax.psum(core(arrs, maskT, P, ox, oy), ('y', 'x'))
            sm = OM.shard_map(local, mesh=self.mesh,
                              in_specs=(self.grid_3D,)*(len(kinds) + 1) + (Pspec(),), out_specs=Pspec())
            fn = jax.jit(sm)
        else:
            def fn(*args):
                arrs, maskT, P = args[:-2], args[-2], args[-1]
                r = core([np__.asarray(a) for a in arrs], np__.asarray(maskT), P, g['ox'], g['oy'])
                return self.global_sum(np__.ascontiguousarray(r)) if self.n_pes > 1 else r
        self._p_fns[key] = fn
        return fn

    def _p_wet_fn(self):
        key = ('wet',)
        if key in self._p_fns:
            return self._p_fns[key]
        g = self._pg
        if OM.jax:
            jax = OM.jax; Pspec = jax.sharding.PartitionSpec
            bx, by = g['bx'], g['by']
            def local(maskT, P):
                ox = jax.lax.axis_index('x')*bx; oy = jax.lax.axis_index('y')*by
                return jax.lax.psum(self._p_wet_core(maskT, P, ox, oy, bx, by), ('y', 'x'))
            fn = jax.jit(OM.shard_map(local, mesh=self.mesh, in_specs=(self.grid_3D, Pspec()),
                                      out_specs=Pspec()))
        else:
            def fn(maskT, P):
                r = self._p_wet_core(np__.asarray(maskT), P, g['ox'], g['oy'], g['bx'], g['by'])
                return self.global_sum(np__.ascontiguousarray(r)) if self.n_pes > 1 else r
        self._p_fns[key] = fn
        return fn

    def _p_ab_fn(self):
        """(P, D, vel, moving, tau, nab) -> (P candidate, D): the AB4 step on
        the replicated arrays; tau and nab are static (seven variants)"""
        key = ('ab',)
        if key in self._p_fns:
            return self._p_fns[key]
        g = self._pg; dt = float(self.dt)
        def ab(P, D, vel, moving, tau, nab):
            xp = OM.np
            u, v, w = vel[0], vel[1], vel[2]
            if g['deg']:
                r2d = 180.0/np__.pi
                rate = xp.stack([u/(g['radius']*xp.cos(P[1]/r2d))*r2d, v/g['radius']*r2d, w])
            else:
                rate = xp.stack([u, v, w])
            if OM.jax:
                D = D.at[tau].set(rate)
            else:
                D = D.copy(); D[tau] = rate
            c = _AB[min(nab, 3)]
            dd = c[0]*D[tau]
            for l in (1, 2, 3):
                if c[l] != 0.0:
                    dd = dd + c[l]*D[(tau - l) % 4]
            Pn = P + dt*dd*moving[None, :]
            x = Pn[0]; y = Pn[1]; z = Pn[2]
            if g['px']:
                x = g['x_west'] + xp.mod(x - g['x_west'], g['Lx'])
            else:
                x = xp.clip(x, g['x_west'], g['x_west'] + g['Lx'])
            if g['py']:
                y = g['y_south'] + xp.mod(y - g['y_south'], g['Ly'])
            else:
                y = xp.clip(y, g['y_south'], g['y_north'])
            z = xp.clip(z, g['z_bot'] + g['eps_bot'], g['z_top'] - g['eps_top'])
            return xp.stack([x, y, z]), D
        if OM.jax:
            fn = OM.jax.jit(ab, static_argnums=(4, 5))
        else:
            fn = ab
        self._p_fns[key] = fn
        return fn

    def _p_accept_fn(self):
        key = ('accept',)
        if key in self._p_fns:
            return self._p_fns[key]
        def acc(P, Pc, wet, moving, fz):
            xp = OM.np
            ok = (wet > 0.5) | (moving < 0.5)
            return xp.where(ok[None, :], Pc, P), fz + xp.sum(xp.where(ok, 0.0, 1.0))
        fn = OM.jax.jit(acc) if OM.jax else acc
        self._p_fns[key] = fn
        return fn

    # ------------------------------------------------------------------
    # the step
    # ------------------------------------------------------------------
    def step_particles(self, record=False, host=True):
        """one time step of the particles with the current model velocities.
        record=True: returns the host dict x, y, z (the positions before the
        step), u, v, w, temp, salt (the values there). Every process must
        call it (the sums are collective); host=False skips the host copy of
        the record on this process (returns None), e.g. on all but PE 0"""
        kinds = ('u', 'v', 'w') + (('t', 't') if record else ())
        # w leaves the model's time step with a stale halo (vertical_velocity
        # integrates the divergence over the whole local array, and its outer
        # halo rows see the wrapped neighbours of roll); the model never reads
        # it there, the stencil here does: a copy is exchanged, self.w stays
        w = self.apply_bc(self.w*1.0)
        arrs = (self.u, self.v, w) + ((self.temp, self.salt) if record else ())
        S = self._p_sample_fn(kinds)(*arrs, self.maskT, self.p_pos)
        out = None
        if record and host:
            Sh = self._p_host(S); Ph = self._p_host(self.p_pos)
            out = dict(x=Ph[0], y=Ph[1], z=Ph[2], u=Sh[0], v=Sh[1], w=Sh[2], temp=Sh[3], salt=Sh[4])
        vel = S[:3]
        Pc, self.p_hist = self._p_ab_fn()(self.p_pos, self.p_hist, vel, self.p_moving,
                                          self.p_tau, min(self.p_nab, 3))
        wet = self._p_wet_fn()(self.maskT, Pc)
        self.p_pos, self._p_fz = self._p_accept_fn()(self.p_pos, Pc, wet, self.p_moving, self._p_fz)
        self.p_tau = (self.p_tau + 1) % 4
        self.p_nab += 1
        return out

    @property
    def p_frozen(self):
        """refused steps so far, summed over the particles (a host int; a device sync)"""
        return int(round(float(self._p_host(self._p_fz))))

    # ------------------------------------------------------------------
    # restart
    # ------------------------------------------------------------------
    def particles_state(self):
        """host numpy dict of everything a restart needs"""
        return dict(p_pos=self._p_host(self.p_pos), p_hist=self._p_host(self.p_hist),
                    p_moving=np__.asarray(self.p_moving_np), p_ids=np__.asarray(self.p_ids),
                    p_tau=np__.int64(self.p_tau), p_nab=np__.int64(self.p_nab),
                    p_frozen=np__.int64(self.p_frozen))

    def particles_restore(self, d):
        """the state of particles_state (init_particles must have run, for the grid)"""
        if not hasattr(self, '_pg'):
            self._pg = self._p_grid(); self._p_fns = {}
        self.p_pos = self._p_rep(d['p_pos']); self.p_hist = self._p_rep(d['p_hist'])
        self.p_moving_np = np__.asarray(d['p_moving'], float)
        self.p_moving = self._p_rep(self.p_moving_np)
        self.p_ids = np__.asarray(d['p_ids'], np__.int64); self.p_n = int(len(self.p_ids))
        self.p_tau = int(d['p_tau']); self.p_nab = int(d['p_nab'])
        self._p_fz = self._p_rep(np__.full((), float(d['p_frozen'])))
        return
