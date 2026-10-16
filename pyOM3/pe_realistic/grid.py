"""
 Variable grid spacings dx(x), dy(y), dz(z) and the pseudo-spherical
 coordinate system, ported from pyOM2 (for_src/main/numerics.f90:
 u_centered_grid, calc_grid).

 Input parameters (in the parameter dict):
   dxt, dyt, dzt   : T-cell sizes, scalar or 1-D array of length nx/ny/nz
                     (global, without halo points). Scalars dx, dy, dz are
                     accepted as a fallback for pe_simple-style configs.
   coord_degree    : if True, dxt/dyt are in degrees longitude/latitude and
                     the pseudo-spherical metric factors cost, cosu, tantr
                     are active. Default False (Cartesian, cost=cosu=1).
   x_origin, y_origin : coordinate of the first U/V point (degrees if
                     coord_degree). Default 0.
   set_zero_at_surface : if True, zt/zw are measured from the surface
                     downward (pyOM2 ocean convention, z <= 0); default
                     False keeps pe_simple's height-above-bottom convention
                     (zt[0] = dzt[0]/2 > 0). For eq_of_state = 100 the
                     atmosphere offset of pe_simple applies regardless.

 Grid arrays built here (pyOM2 names, all stored as OM.np 1-D arrays in the
 LOCAL halo layout, plus broadcast views):
   dxt, dxu, xt, xu             length Nx
   dyt, dyu, yt, yu             length Ny
   dzt, dzw, zt, zw             length Nz
   cost, cosu, tantr            length Ny  (metric factors, pyOM2
                                 calc_grid: cos(yt), cos(yu), tan(yt)/a)
   area_t, area_u, area_v, area_z : (1,Ny,Nx) cell areas
                                 (dxt*cost*dyt etc., pyOM2 calc_grid)

 Broadcast views for the kernels (suffix = axis the array varies along):
   3-D fields: dxt_x, dxu_x (1,1,Nx); dyt_y, dyu_y, cost_y, cosu_y,
               tantr_y (1,Ny,1); dzt_z, dzw_z (Nz,1,1)
   2-D fields: use view[0] (e.g. self.dxt_x[0] has shape (1,Nx)).

 Halo filling of the grid arrays does NOT use apply_bc (which zero-fills
 closed boundaries and would put zeros into divisors): spacings are wrapped
 for a periodic direction and edge-replicated otherwise, then
 u_centered_grid runs on the extended array so the coordinates in the halo
 are proper linear extensions (exactly pyOM2 calc_grid). For a periodic
 direction the halo coordinates are afterwards overwritten with their
 periodic images, as in pyOM2, so cost/cosu are periodic.

 Layout subtlety: with JAX sharding the global array carries the per-shard
 halo points INLINE (Nx = nx + 2*halo*n_pes_x, see boundary.py), so the 1-D
 grid arrays are assembled shard by shard into that same layout; with MPI
 the layout is the plain local slice. Grid arrays are constants of the
 model (they are baked into the jitted kernels via self), so they are not
 sharded themselves.
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
          """dxt/dyt/dzt from the parameter dict: array of length n, or a
          scalar, or the pe_simple-style scalar fallback (dx/dy/dz)."""
          if hasattr(self, name):     val = getattr(self, name)
          elif hasattr(self, fallback): val = getattr(self, fallback)
          else:
              raise ValueError('grid: neither %s nor %s given' % (name, fallback))
          val = np__.asarray(val, dtype=np__.float64)
          if val.ndim == 0:  return np__.full(n, float(val))
          if val.shape != (n,):
              raise ValueError('grid: %s must be scalar or shape (%d,), got %s'
                               % (name, n, val.shape))
          return val.copy()

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
          Build all grid and metric arrays. Called from main.__init__ after
          the domain decomposition (block indices known) and before
          allocate_workspace.
          """
          halo = self.halo
          nx, ny, nz = self.nx, self.ny, self.nz      # GLOBAL sizes (input)
          degtom = self.radius/180.0*np__.pi
          self.coord_degree = getattr(self, 'coord_degree', False)
          x0 = getattr(self, 'x_origin', 0.0)
          y0 = getattr(self, 'y_origin', 0.0)

          # ---- global T spacings, extended by halo ghosts ----------------
          dxt_gl = self._extend_1d(self._dxt_input('dxt','dx',nx), halo, self.periodic_in_x)
          dyt_gl = self._extend_1d(self._dxt_input('dyt','dy',ny), halo, self.periodic_in_y)
          dzt_gl = self._extend_1d(self._dxt_input('dzt','dz',nz), halo, False)

          # ---- coordinates + U spacings on the extended arrays -----------
          xu_gl, xt_gl, dxu_gl = u_centered_grid(dxt_gl)
          yu_gl, yt_gl, dyu_gl = u_centered_grid(dyt_gl)
          # VERTICAL: pyOM2 has no z halo and runs u_centered_grid on the
          # nz interior only (numerics.f90: call u_centered_grid(dzt,dzw,
          # zt,zw,nz)).  Running the yt(k+1) = 2 zw(k) - zt(k) recurrence
          # across replicated halo spacings is NOT a linear extension and
          # can produce dzw = 0 at the surface face for non-uniform dzt
          # (divide-by-zero in every vertical operator).  Build the interior
          # exactly as pyOM2, then extend coordinates linearly into the
          # halos and replicate the edge spacings.
          zw_i, zt_i, dzw_i = u_centered_grid(dzt_gl[halo:halo+nz])
          zt_gl  = np__.empty(nz+2*halo); zw_gl  = np__.empty(nz+2*halo)
          dzw_gl = np__.empty(nz+2*halo)
          zt_gl[halo:halo+nz]  = zt_i
          zw_gl[halo:halo+nz]  = zw_i
          dzw_gl[halo:halo+nz] = dzw_i
          for k in range(halo):
              zt_gl[halo-1-k]  = zt_i[0]  - (k+1)*dzt_gl[halo]
              zw_gl[halo-1-k]  = zw_i[0]  - (k+1)*dzt_gl[halo]
              zt_gl[nz+halo+k] = zt_i[-1] + (k+1)*dzt_gl[nz+halo-1]
              zw_gl[nz+halo+k] = zw_i[-1] + (k+1)*dzt_gl[nz+halo-1]
              dzw_gl[halo-1-k]  = dzw_i[0]
              dzw_gl[nz+halo+k] = dzw_i[-1]
          if not (np__.isfinite(dzt_gl).all() and dzt_gl.min() > 0.0):
              raise ValueError('vertical grid: dzt contains non-finite or '
                  'non-positive values (wrong byte order of a binary '
                  'input file?)')
          if not (dzw_i.min() > 0.0):   # NaN-safe: NaN compares False
              raise ValueError('u-centered vertical grid degenerate: '
                  'dzw <= 0 for the given dzt (pyOM2 convention '
                  'dzw(nz) = 2*dzt(nz) - dzw(nz-1); dzt must vary smoothly)')

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
          # vertical grid is checked above (dzw), the horizontal one here,
          # on the interior plus the halo (the halo spacings enter the
          # gradients at the block edges). NaN-safe: NaN compares False.
          if not (dxu_gl.min() > 0.0) or not (dyu_gl.min() > 0.0):
              raise ValueError('u-centered horizontal grid degenerate: dxu or '
                  'dyu <= 0 for the given dxt/dyt (pyOM2 recursion; a jump '
                  'in the spacing, or a periodic direction whose two ends '
                  'do not match). min dxu=%g min dyu=%g'
                  % (dxu_gl.min(), dyu_gl.min()))

          # ---- vertical origin -------------------------------------------
          # default: height above bottom, bottom face of the first interior
          # cell at z=0 (pe_simple snapshot convention: zt[0]=dzt[0]/2,
          # zw[0]=dzt[0]); option: z=0 at the surface W face (pyOM2 ocean).
          if getattr(self, 'set_zero_at_surface', False):
              off = zw_gl[nz+halo-1]                 # last interior W face
          else:
              off = zw_gl[halo] - dzt_gl[halo]       # bottom face -> 0
          zt_gl -= off;  zw_gl -= off

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
          self.zt_gl = zt_gl[halo:-halo].copy(); self.zw_gl = zw_gl[halo:-halo].copy()

          # ---- local layout ----------------------------------------------
          def layout_x(a): return self._to_layout(a, halo, nx, self.n_pes_x, self._blk0_x)
          def layout_y(a): return self._to_layout(a, halo, ny, self.n_pes_y, self._blk0_y)

          # global interior offset of this PE's block (0-based), from the
          # SAME nominal block size as decomposition_mpi
          if OM.jax:
              self._blk0_x = self._blk0_y = 0        # unused in JAX layout
          else:
              x_blk = (nx-1)//self.n_pes_x + 1
              y_blk = (ny-1)//self.n_pes_y + 1
              self._blk0_x = (self.my_blk_x-1)*x_blk
              self._blk0_y = (self.my_blk_y-1)*y_blk

          self.dxt = layout_x(dxt_gl); self.dxu = layout_x(dxu_gl)
          self.xt  = layout_x(xt_gl) ; self.xu  = layout_x(xu_gl)
          self.dyt = layout_y(dyt_gl); self.dyu = layout_y(dyu_gl)
          self.yt  = layout_y(yt_gl) ; self.yu  = layout_y(yu_gl)
          self.cost  = layout_y(cost_gl)
          self.cosu  = layout_y(cosu_gl)
          self.tantr = layout_y(tantr_gl)
          # vertical: no decomposition, layout = extended global array
          self.dzt = OM.np.asarray(dzt_gl); self.dzw = OM.np.asarray(dzw_gl)
          self.zt  = OM.np.asarray(zt_gl) ; self.zw  = OM.np.asarray(zw_gl)

          # consistency with the decomposition sizes
          assert len(self.dxt) == self.Nx and len(self.dyt) == self.Ny \
                 and len(self.dzt) == self.Nz, 'grid layout does not match Nx/Ny/Nz'

          # ---- broadcast views -------------------------------------------
          self.dxt_x  = self.dxt[None,None,:]; self.dxu_x  = self.dxu[None,None,:]
          self.dyt_y  = self.dyt[None,:,None]; self.dyu_y  = self.dyu[None,:,None]
          self.cost_y = self.cost[None,:,None]; self.cosu_y = self.cosu[None,:,None]
          self.tantr_y= self.tantr[None,:,None]
          self.dzt_z  = self.dzt[:,None,None]; self.dzw_z  = self.dzw[:,None,None]

          # ---- cell areas (pyOM2 calc_grid) ------------------------------
          self.area_t = self.dxt_x*self.cost_y*self.dyt_y      # (1,Ny,Nx)
          self.area_u = self.dxu_x*self.cost_y*self.dyt_y
          self.area_v = self.dxt_x*self.cosu_y*self.dyu_y
          self.area_z = self.dxu_x*self.cosu_y*self.dyu_y

          # tripwire: the scalar spacings of pe_simple must not survive as
          # silently-correct-on-uniform-grids fallbacks. Any leftover
          # self.dx use now raises immediately.
          self.dx = self.dy = self.dz = None
          return

      def _to_layout(self, a_ext, halo, n, n_shards, blk0):
          """Global halo-extended 1-D array -> local layout (see module
          docstring), returned as an OM.np array."""
          if OM.jax:
              if n % n_shards != 0:
                  raise ValueError('JAX layout needs n %% n_shards == 0')
              bx  = n // n_shards
              seg = [ a_ext[s*bx : (s+1)*bx + 2*halo] for s in range(n_shards) ]
              return OM.np.asarray(np__.concatenate(seg))
          # MPI/single PE: local slice, interior points [blk0, blk0+nloc_int)
          # plus halo on each side (nominal block size as decomposition_mpi,
          # last block truncated)
          nloc_int = min(n - blk0, (n-1)//n_shards + 1)
          return OM.np.asarray(a_ext[blk0 : blk0 + nloc_int + 2*halo])
