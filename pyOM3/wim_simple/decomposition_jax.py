"""
 Domain decomposition of the WIM model using manual JAX sharding.
 Based on class setup. The counterpart of decomposition_mpi.

 The model arrays are C-ordered (Nd,Nf,Ny,Nx) (dA: (3,Nd,Nf,Ny,Nx), the
 exchange and mask fields (Nf,Ny,Nx) or (Ny,Nx)). Only the trailing two
 axes (y,x) are sharded, with the INLINE-HALO LAYOUT of pe_simple: every
 shard carries its own halo ring, so the global padded sizes are
    Nx = nx + 2*halo*n_pes_x ,  Ny = ny + 2*halo*n_pes_y
 and a slice [halo:-halo] on y or x strips the OUTER halo only. Interior
 operations therefore go through the mask maskI (main.py) or through the
 shard-aware unpad below, never through a slice on y,x. The spectral axes
 (d,f) are never sharded and keep their plain halo.

 Defines the mesh and the shardings, the shard_map wrappers by rank, and
 the helpers pad/unpad (trailing two axes), rollx/rolly (shift inside each
 shard, the halo exchange supplies the seam) and the neighbour tables of
 the ppermute halo exchange in boundary.apply_bc_jax.
"""
import pyOM3            as OM
from   pyOM3.wim_simple import setup
from   functools        import partial
import numpy            as np__


class decomposition_jax(setup):


      def domain_decomposition_jax(self, n_pes_x, n_pes_y):
          """
          Horizontal domain decomposition with JAX sharding: n_pes_x, n_pes_y
          shards in x, y. Mesh axes ('y','x'); the PartitionSpec of an array
          of rank r puts 'y','x' on its last two axes and None on the rest.
          """
          self.my_pe = OM.jax.process_index()
          self.n_pes_x, self.n_pes_y = n_pes_x, n_pes_y
          self.n_pes = n_pes_x*n_pes_y              # shards (the mesh uses the first n devices)
          if self.n_pes > OM.jax.device_count():
             raise ValueError('n_pes_x*n_pes_y (=%d) exceeds the device count (=%d)'
                              % (self.n_pes, OM.jax.device_count()))
          if self.nx % n_pes_x or self.ny % n_pes_y:
             raise ValueError('JAX layout needs nx, ny divisible by n_pes_x, n_pes_y')
          self.comm = None                     # no communicator: collectives are jit's
          # block coordinates (1-based) and the local range: the single
          # process holds the whole domain
          self.my_blk_x, self.my_blk_y = 1, 1
          self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe = 1, self.nx, 1, self.ny
          self.nx_loc, self.ny_loc = self.nx, self.ny

          # positional arguments: the keyword was renamed axis_shapes -> axis_sizes in jax 0.7.
          # Auto axis types: since jax 0.7 make_mesh defaults to Explicit axes, under which
          # shard_map no longer reshards its inputs to in_specs but raises instead.
          try:
              auto = (OM.jax.sharding.AxisType.Auto,)*2
              self.mesh = OM.jax.make_mesh((n_pes_y, n_pes_x), ('y','x'), axis_types = auto)
          except (AttributeError, TypeError):           # older jax: Auto is the only mode
              self.mesh = OM.jax.make_mesh((n_pes_y, n_pes_x), ('y','x'))
          P = OM.jax.sharding.PartitionSpec
          self.grid_spec = {r: P(*((None,)*(r-2) + ('y','x'))) for r in (2,3,4,5)}
          self.sharding  = {r: OM.jax.sharding.NamedSharding(self.mesh, self.grid_spec[r])
                            for r in (2,3,4,5)}
          self.grid_2D, self.grid_3D, self.grid_4D = (self.grid_spec[r] for r in (2,3,4))
          self.sharding_2D, self.sharding_3D, self.sharding_4D = (self.sharding[r] for r in (2,3,4))

          # ppermute neighbour tables; a non-periodic direction drops the wrap
          # pair, and ppermute fills a shard that receives nothing with zeros,
          # which is the closed boundary value of the halo
          self.eastward  = [(i, (i + 1) % n_pes_x) for i in range(n_pes_x)]
          self.westward  = [(i, (i - 1) % n_pes_x) for i in range(n_pes_x)]
          self.northward = [(i, (i + 1) % n_pes_y) for i in range(n_pes_y)]
          self.southward = [(i, (i - 1) % n_pes_y) for i in range(n_pes_y)]
          if not self.periodic_in_x:
              self.eastward = self.eastward[:-1];  self.westward = self.westward[1:]
          if not self.periodic_in_y:
              self.northward = self.northward[:-1]; self.southward = self.southward[1:]
          return


      def show_pe_layout(self):
          """print the layout (single process)"""
          print('global horizontal domain is nx=', self.nx, 'ny=', self.ny)
          print('spectral grid (local on every shard): nf=', self.nf, 'nd=', self.nd)
          print('JAX sharding on', self.n_pes, 'devices:  n_pes_x=', self.n_pes_x,
                ' n_pes_y=', self.n_pes_y, ' (inline halo layout, Nx=%d Ny=%d)' % (self.Nx, self.Ny))
          return


      # ---------------------------------------------------------------
      #  shard_map wrappers by rank
      # ---------------------------------------------------------------
      def shard_map_arr(self, func, ndim):
          spec = self.grid_spec[ndim]
          return OM.shard_map(func, mesh=self.mesh, in_specs=spec, out_specs=spec)


      def _xy_slices(self, ndim):
          """index tuple selecting all of the leading axes"""
          return (slice(None),)*(ndim-2)


      @partial(OM.jaxjit, static_argnums=(0,2))
      def pad(self, x, lead=0):
          """
          add halo points to an array: on the trailing two (sharded) axes
          inside every shard, on the first `lead` leading axes plainly
          (e.g. lead=1 for a (nf,ny,nx) field whose frequency axis is padded
          as well). The result has the inline-halo layout.
          """
          h = self.halo
          @partial(self.shard_map_arr, ndim=x.ndim)
          def pad_it(x):
              w = ((0,0),)*(x.ndim-2) + ((h,h),(h,h))
              return OM.np.pad(x, w)
          x = pad_it(x)
          if lead:
              w = ((h,h),)*lead + ((0,0),)*(x.ndim-lead)
              x = OM.np.pad(x, w)
          return x


      @partial(OM.jaxjit, static_argnums=(0,2))
      def unpad(self, x, lead=0):
          """
          remove the halo points of the trailing two axes shard by shard
          (and plainly of the first `lead` leading axes): the inverse of pad.
          The result is the global interior in shard order.
          """
          h = self.halo
          s = slice(h,-h)
          @partial(self.shard_map_arr, ndim=x.ndim)
          def unpad_it(x):
              return x[self._xy_slices(x.ndim) + (s,s)]
          x = unpad_it(x)
          if lead:
              x = x[(s,)*lead]
          return x


      @partial(OM.jaxjit, static_argnums=(0,2))
      def rollx(self, x, shift):
          """Shift last (x) axis inside every shard. shift=-1 -> positive direction."""
          @partial(self.shard_map_arr, ndim=x.ndim)
          def roll_it(x):
              return OM.np.roll(x, shift=shift, axis=-1)
          return roll_it(x)


      @partial(OM.jaxjit, static_argnums=(0,2))
      def rolly(self, x, shift):
          """Shift 2.-last (y) axis inside every shard."""
          @partial(self.shard_map_arr, ndim=x.ndim)
          def roll_it(x):
              return OM.np.roll(x, shift=shift, axis=-2)
          return roll_it(x)


      def rollf(self, a, shift):
          """Shift 3.-last (freq) axis; never sharded."""
          return OM.np.roll(a, shift=shift, axis=-3)

      def rolld(self, a, shift):
          """Shift 4.-last (direction) axis; never sharded."""
          return OM.np.roll(a, shift=shift, axis=-4)


      def gather_array(self, a):
          """full-domain interior as a numpy array (the MPI counterpart gathers on PE 0)"""
          return np__.asarray(self.unpad(a))


      def from_global(self, a):
          """
          a global numpy array (the interior, shape (..., ny, nx)) as a
          sharded device array: every process supplies its own shards
          (make_array_from_callback), so this works with one process per
          GPU as well as with all devices in one process.
          """
          a = np__.asarray(a)
          return OM.jax.make_array_from_callback(a.shape, self.sharding[a.ndim],
                                                 lambda idx: a[idx])


      def to_device(self, a):
          """
          put a full-size padded numpy array (global interior plus the plain
          outer halo, the layout of a single PE) into the inline-halo layout:
          strip the outer halo, pad shard by shard. For initial conditions
          and forcing fields given per global index.
          """
          h = self.halo
          s = slice(h,-h)
          a = np__.asarray(a)
          return self.pad(self.from_global(a[(slice(None),)*(a.ndim-2) + (s,s)]))
