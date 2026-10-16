"""
 Variable horizontal grid spacings dx(x), dy(y) and the pseudo-spherical
 coordinate system for the wave model, ported from pe_realistic/grid.py
 (itself a port of pyOM2 for_src/main/numerics.f90: u_centered_grid,
 calc_grid). The spectral axes (frequency, direction) stay in
 main.set_grid and are untouched by this module.

 Input parameters (in the parameter dict):
   dxt, dyt        : T-cell sizes, scalar or 1-D array of length nx/ny
                     (global, without halo points). Scalars dx, dy are
                     accepted as a fallback for wim_simple-style configs.
   coord_degree    : if True, dxt/dyt are in degrees longitude/latitude and
                     the pseudo-spherical metric factors cost, cosu, tantr
                     are active. Default False (Cartesian, cost=cosu=1,
                     tantr=0, and the model reduces exactly to wim_simple).
   x_origin, y_origin : coordinate of the first U/V point (degrees if
                     coord_degree). Default 0.

 Grid arrays built here (pyOM2 names, 1-D arrays in the LOCAL halo
 layout, plus broadcast views):
   dxt, dxu, xt, xu             length Nx
   dyt, dyu, yt, yu             length Ny
   cost, cosu, tantr            length Ny  (pyOM2 calc_grid: cos(yt),
                                 cos(yu), tan(yt)/a; tantr in 1/m)

 Broadcast views (TRAILING-axis convention: the model fields are
 (Nd,Nf,Ny,Nx) and the sampled currents (Nf,Ny,Nx), so 2-D views shaped
 (Ny,1) / (1,Nx) / (Ny,Nx) broadcast against all of them from the right):
   dxt_x, dxu_x   (1,Nx)        dyt_y, dyu_y   (Ny,1)
   cost_y, cosu_y, tantr_y (Ny,1)
   area_t = dxt_x*cost_y*dyt_y  (Ny,Nx)   T-cell area (m^2)

 Halo filling of the grid arrays does NOT use apply_bc (which zero-fills
 closed boundaries and would put zeros into divisors): spacings are
 wrapped for a periodic direction and edge-replicated otherwise, then
 u_centered_grid runs on the extended array so the coordinates in the
 halo are proper linear extensions (exactly pyOM2 calc_grid). For a
 periodic direction the halo coordinates are afterwards overwritten with
 their periodic images, as in pyOM2, so cost/cosu are periodic.

 Layout: the wave model decomposes with boundary._local_range (even
 split, remainder spread over the first blocks), which differs from
 pe_realistic's nominal-block convention -- the local slice is therefore
 taken from the block indices xs_pe/ys_pe the decomposition already
 provides, not recomputed here.

 The poles are NOT treated (no reduced grid, no pole offset): a global
 configuration must stop short of them, as in pyOM2. The zonal CFL
 shrinks with cost towards high latitudes.
"""
import pyOM3     as OM
import numpy     as np__


def u_centered_grid(dyt):
    """
    pyOM2 numerics.f90 u_centered_grid, verbatim (0-based):
      yu_i = yu_{i-1} + dyt_i,  yt_{i+1} = 2 yu_i - yt_i,
      dyu_i = yt_{i+1} - yt_i,  dyu_n = 2 dyt_n - dyu_{n-1}.
    The recursion (not a cumsum of averages) is what keeps gradient and
    divergence adjoint on a stretched grid. Pure numpy, host side.
    """
    n   = len(dyt)
    yu  = np__.zeros(n); yt = np__.zeros(n); dyu = np__.zeros(n)
    yu[0] = 0.0
    for i in range(1, n):  yu[i] = yu[i-1] + dyt[i]
    yt[0] = yu[0] - dyt[0]*0.5
    for i in range(1, n):  yt[i] = 2*yu[i-1] - yt[i-1]
    for i in range(n-1):   dyu[i] = yt[i+1] - yt[i]
    dyu[n-1] = 2*dyt[n-1] - dyu[max(0, n-2)]
    return yu, yt, dyu


class grid:

      radius = 6370.0e3                       # Earth radius in m (pyOM2)

      def _dxt_input(self, name, fallback, n):
          """dxt/dyt from the parameter dict: array of length n, or a
          scalar, or the wim_simple-style scalar fallback (dx/dy)."""
          if hasattr(self, name):       val = getattr(self, name)
          elif hasattr(self, fallback): val = getattr(self, fallback)
          else:
              raise ValueError('grid: neither %s nor %s given' % (name, fallback))
          val = np__.asarray(val, dtype=np__.float64)
          if val.ndim == 0:  return np__.full(n, float(val))
          if val.shape != (n,):
              raise ValueError('grid: %s must be scalar or shape (%d,), got %s'
                               % (name, n, val.shape))
          return val.copy()

      def _to_layout(self, a_ext, halo, n, n_shards, blk0, nloc):
          """Global halo-extended 1-D array -> local layout, as an OM.np
          array. MPI: the local block [blk0, blk0+nloc) plus its halo (the
          decomposition's _local_range). JAX: the n_shards blocks of n/n_shards
          points, each with its own halo, concatenated (the inline layout of
          wim_simple/decomposition_jax.py)."""
          if OM.jax:
              if n % n_shards != 0:
                  raise ValueError('JAX layout needs n %% n_shards == 0')
              bx  = n // n_shards
              seg = [ a_ext[s*bx : (s+1)*bx + 2*halo] for s in range(n_shards) ]
              return OM.np.asarray(np__.concatenate(seg))
          return OM.np.asarray(a_ext[blk0 : blk0 + nloc + 2*halo])

      def _roll1(self, a, halo, n, n_shards):
          """np.roll(a, 1) of a 1-D grid array in the layout: per shard
          under JAX (each block of n/n_shards + 2 halo points rolled on its
          own, like decomposition_jax.rollx on the model fields), plain on
          numpy."""
          a = np__.asarray(a)
          if OM.jax and n_shards > 1:
              L = n // n_shards + 2*halo
              return OM.np.asarray(np__.concatenate(
                         [np__.roll(a[s*L:(s+1)*L], 1) for s in range(n_shards)]))
          return OM.np.asarray(np__.roll(a, 1))

      def _extend_1d(self, d, halo, periodic):
          """Extend global T spacings by halo ghost points on each side:
          wrap for a periodic direction, edge-replicate otherwise
          (pyOM2 calc_grid)."""
          if periodic:
              left  = d[-halo:] ;  right = d[:halo]
          else:
              left  = np__.full(halo, d[0]) ; right = np__.full(halo, d[-1])
          return np__.concatenate([left, d, right])

      def calc_grid(self):
          """
          Build all horizontal grid and metric arrays. Called from
          main.setup after the domain decomposition (block indices known)
          and before allocate_workspace / set_grid.
          """
          halo = self.halo
          nx, ny = self.nx, self.ny                   # GLOBAL sizes (input)
          degtom = self.radius/180.0*np__.pi
          self.degtom = degtom
          self.coord_degree = getattr(self, 'coord_degree', False)
          x0 = getattr(self, 'x_origin', 0.0)
          y0 = getattr(self, 'y_origin', 0.0)

          # ---- global T spacings, extended by halo ghosts ----------------
          dxt_gl = self._extend_1d(self._dxt_input('dxt','dx',nx), halo, self.periodic_in_x)
          dyt_gl = self._extend_1d(self._dxt_input('dyt','dy',ny), halo, self.periodic_in_y)

          # ---- coordinates + U spacings on the extended arrays -----------
          xu_gl, xt_gl, dxu_gl = u_centered_grid(dxt_gl)
          yu_gl, yt_gl, dyu_gl = u_centered_grid(dyt_gl)

          # origin: first INTERIOR U point at x_origin (pyOM2:
          # xt_gl = xt_gl - xu_gl(1) + x_origin, index 1 = first interior)
          xt_gl += x0 - xu_gl[halo];  xu_gl += x0 - xu_gl[halo]
          yt_gl += y0 - yu_gl[halo];  yu_gl += y0 - yu_gl[halo]

          # periodic directions: overwrite halo coords/spacings with their
          # periodic images (pyOM2 calc_grid; makes cost/cosu periodic).
          # NOTE pyOM2 copies the coordinate VALUES (not shifted by the
          # domain length) -- the halo coordinates are only consumed by the
          # metric factors, which must be periodic images.
          if self.periodic_in_x:
              for i in range(halo):
                  xt_gl[nx+halo+i]  = xt_gl[halo+i];  xt_gl[halo-1-i]  = xt_gl[nx+halo-1-i]
                  xu_gl[nx+halo+i]  = xu_gl[halo+i];  xu_gl[halo-1-i]  = xu_gl[nx+halo-1-i]
                  dxu_gl[nx+halo+i] = dxu_gl[halo+i]; dxu_gl[halo-1-i] = dxu_gl[nx+halo-1-i]
          if self.periodic_in_y:
              for j in range(halo):
                  yt_gl[ny+halo+j]  = yt_gl[halo+j];  yt_gl[halo-1-j]  = yt_gl[ny+halo-1-j]
                  yu_gl[ny+halo+j]  = yu_gl[halo+j];  yu_gl[halo-1-j]  = yu_gl[ny+halo-1-j]
                  dyu_gl[ny+halo+j] = dyu_gl[halo+j]; dyu_gl[halo-1-j] = dyu_gl[ny+halo-1-j]


          # ---- horizontal U spacings must be positive ---------------------
          # The pyOM2 recursion gives dxu <= 0 where dxt jumps too strongly;
          # in a periodic direction the seam counts as a jump (cyclic
          # dxt = [1,1,2,2] gives dxu = 0 at the first interior point). The
          # horizontal U spacings are checked here,
          # on the interior plus the halo (the halo spacings enter the
          # gradients at the block edges). NaN-safe: NaN compares False.
          if not (dxu_gl.min() > 0.0) or not (dyu_gl.min() > 0.0):
              raise ValueError('u-centered horizontal grid degenerate: dxu or '
                  'dyu <= 0 for the given dxt/dyt (pyOM2 recursion; a jump '
                  'in the spacing, or a periodic direction whose two ends '
                  'do not match). min dxu=%g min dyu=%g'
                  % (dxu_gl.min(), dyu_gl.min()))

          # ---- degrees -> metres, metric factors -------------------------
          # (pyOM2 calc_grid: convert spacings AFTER building coordinates,
          #  so xt/yt stay in degrees for the metric factors and snapshots)
          if self.coord_degree:
              dxt_gl *= degtom; dxu_gl *= degtom
              dyt_gl *= degtom; dyu_gl *= degtom
              cost_gl  = np__.cos(yt_gl/180.0*np__.pi)
              cosu_gl  = np__.cos(yu_gl/180.0*np__.pi)
              tantr_gl = np__.tan(yt_gl/180.0*np__.pi)/self.radius
          else:
              cost_gl  = np__.ones_like(yt_gl)
              cosu_gl  = np__.ones_like(yu_gl)
              tantr_gl = np__.zeros_like(yt_gl)

          # ---- global interior coordinates (for snapshots) ---------------
          self.xt_gl = xt_gl[halo:-halo].copy(); self.xu_gl = xu_gl[halo:-halo].copy()
          self.yt_gl = yt_gl[halo:-halo].copy(); self.yu_gl = yu_gl[halo:-halo].copy()

          # domain extents in metres (diagnostic; along a latitude circle the
          # physical zonal extent is Lx*cost)
          self.Lx = float(dxt_gl[halo:-halo].sum())
          self.Ly = float(dyt_gl[halo:-halo].sum())

          # ---- local layout (wave-model decomposition: xs_pe/ys_pe are the
          #      1-based inclusive global index range of this PE's block;
          #      JAX: every shard's block with its own halo, inline) -------
          def layout_x(a): return self._to_layout(a, halo, nx, self.n_pes_x, self.xs_pe-1, self.nx_loc)
          def layout_y(a): return self._to_layout(a, halo, ny, self.n_pes_y, self.ys_pe-1, self.ny_loc)

          self.dxt = layout_x(dxt_gl); self.dxu = layout_x(dxu_gl)
          self.xt  = layout_x(xt_gl) ; self.xu  = layout_x(xu_gl)
          self.dyt = layout_y(dyt_gl); self.dyu = layout_y(dyu_gl)
          self.yt  = layout_y(yt_gl) ; self.yu  = layout_y(yu_gl)
          self.cost  = layout_y(cost_gl)
          self.cosu  = layout_y(cosu_gl)
          self.tantr = layout_y(tantr_gl)

          # consistency with the decomposition sizes
          assert len(self.dxt) == self.Nx and len(self.dyt) == self.Ny, \
                 'grid layout does not match Nx/Ny'

          # ---- broadcast views (trailing (y,x) axes) ---------------------
          self.dxt_x  = self.dxt[None,:]; self.dxu_x  = self.dxu[None,:]
          self.dyt_y  = self.dyt[:,None]; self.dyu_y  = self.dyu[:,None]
          # the U spacings shifted by one point towards lower index
          # (rollx(dxu_x,1), rolly(dyu_y,1) of the gradient operators in
          # propagation.py), built here in the layout: under JAX the roll
          # is per shard (a (1,Nx) view cannot go through the sharded
          # roll), on numpy np.roll as before
          self.dxu_xm1 = self._roll1(self.dxu, halo, nx, self.n_pes_x)[None,:]
          self.dyu_ym1 = self._roll1(self.dyu, halo, ny, self.n_pes_y)[:,None]
          self.cost_y = self.cost[:,None]; self.cosu_y = self.cosu[:,None]
          self.tantr_y= self.tantr[:,None]

          # ---- T-cell area (pyOM2 calc_grid) -----------------------------
          self.area_t = self.dxt_x*self.cost_y*self.dyt_y      # (Ny,Nx)

          # tripwire: the scalar spacings of wim_simple must not survive as
          # silently-correct-on-uniform-grids fallbacks. Any leftover
          # self.dx use now raises immediately.
          self.dx = self.dy = None
          return
