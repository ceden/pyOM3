"""
 Setting halo points to boundary conditions for the WIM model,
 using either the MPI (mpi4py) or the JAX-sharding domain decomposition.

 Based on decomposition_mpi and decomposition_jax; the backend chosen when
 pyOM3 is imported (OM.jax) selects which one is in front of the MRO, so
 rollx/rolly/unpad/gather_array/show_pe_layout resolve to that backend's
 version without a branch per call.

 apply_bc applies everything to the WIM arrays: (Nd,Nf,Ny,Nx), (Nf,Ny,Nx) or
 (Ny,Nx). The two trailing axes (y,x) are decomposed, the leading axes are
 local on every PE / shard.
"""
import pyOM3                                 as OM
from   pyOM3.wim_simple.decomposition_mpi    import decomposition_mpi
from   pyOM3.wim_simple.decomposition_jax    import decomposition_jax
from   functools                             import partial

_bases = (decomposition_jax, decomposition_mpi) if OM.jax else (decomposition_mpi, decomposition_jax)


class boundary(*_bases):


      def __init__(self, parameter = {}, n_pes_x = 1, n_pes_y = 1,
                                         halo_points = 1, comm_in = None):
         """
         Initialize the x,y domain decomposition and add halo points.

         Input nx,ny is the GLOBAL horizontal shape without halo points;
         nf,nd are global and stay local on every PE / shard.
         """
         super().__init__(parameter)
         self.fortran_ordering = False   # WIM arrays are C-ordered (Nd,Nf,Ny,Nx)
         self.halo = halo_points         # number of halo points at each side
         self.domain_decomposition(n_pes_x, n_pes_y, comm_in)
         self.show_pe_layout()
         return


      def domain_decomposition(self, n_pes_x, n_pes_y, comm_in = None):
         """
         The decomposition and the local shapes WITH halo points. MPI: this
         PE's block plus one halo ring. JAX: the whole domain in the inline
         halo layout, every shard with its own ring (decomposition_jax).
         """
         if OM.jax:
            self.domain_decomposition_jax(n_pes_x, n_pes_y)
            self.Nx = self.nx + 2*self.halo*n_pes_x
            self.Ny = self.ny + 2*self.halo*n_pes_y
         else:
            self.domain_decomposition_mpi(n_pes_x, n_pes_y, comm_in)
            self.Nx = self.nx_loc + 2*self.halo
            self.Ny = self.ny_loc + 2*self.halo
         self.Nf = self.nf + 2*self.halo      # frequency: always local
         self.Nd = self.nd + 2*self.halo      # direction: always local
         if hasattr(self, 'dx') and hasattr(self, 'dy'):      # uniform grid (wim_realistic: calc_grid)
            self.Lx, self.Ly = self.nx*self.dx, self.ny*self.dy
         return


      def apply_bc(self, a, bc_value = 0, keep_f_ghosts = False):
          """
          Apply boundary conditions to the halo points of array a.

          Horizontal x,y: halo exchange between neighbouring blocks (periodic
          or closed depending on periodic_in_x / periodic_in_y), by MPI
          messages or by JAX collectives.
          Frequency f: always closed (set to bc_value), unless keep_f_ghosts
          (the halo refresh after the source update must leave the f^-5
          continuation that imphftail wrote into the ghost bins).
          Direction d: always periodic (local wrap).
          """
          if OM.jax: return self.apply_bc_jax(a, keep_f_ghosts = keep_f_ghosts)
          else:      return self.apply_bc_mpi(a, bc_value, keep_f_ghosts = keep_f_ghosts)


      @partial(OM.jaxjit, static_argnums=(0,), static_argnames=('keep_f_ghosts',))
      def apply_bc_jax(self, a, keep_f_ghosts = False):
          """
          apply_bc with JAX sharding: the x and y halo strips travel by
          ppermute between the shards (a shard without a neighbour in a
          closed direction receives zeros, the closed boundary value); the
          f and d halos are local on every shard and set as in apply_bc_mpi.
          """
          halo = self.halo
          lead = (slice(None),)*(a.ndim-2)

          @partial(self.shard_map_arr, ndim=a.ndim)
          def exchange(x):
              west = x[lead + (slice(None), slice(halo, 2*halo))]
              east = x[lead + (slice(None), slice(-2*halo, -halo))]
              rw = OM.jax.lax.ppermute(east, axis_name='x', perm=self.eastward)
              re = OM.jax.lax.ppermute(west, axis_name='x', perm=self.westward)
              x = x.at[lead + (slice(None), slice(0, halo))].set(rw)
              x = x.at[lead + (slice(None), slice(-halo, None))].set(re)
              south = x[lead + (slice(halo, 2*halo), slice(None))]
              north = x[lead + (slice(-2*halo, -halo), slice(None))]
              rs = OM.jax.lax.ppermute(north, axis_name='y', perm=self.northward)
              rn = OM.jax.lax.ppermute(south, axis_name='y', perm=self.southward)
              x = x.at[lead + (slice(0, halo), slice(None))].set(rs)
              x = x.at[lead + (slice(-halo, None), slice(None))].set(rn)
              return x

          a = exchange(a)

          # frequency: always closed (a lifted field with a singleton f axis is skipped)
          if a.ndim > 2 and a.shape[-3] > 2*halo and not keep_f_ghosts:
             for i in range(halo):
                 a = a.at[..., i,      :, :].set(0.0)
                 a = a.at[..., -1 - i, :, :].set(0.0)
          # direction: always periodic
          if a.ndim > 3 and a.shape[-4] > 2*halo:
             for i in range(halo):
                 a = a.at[i,         :, :, :].set(a[-2*halo + i, :, :, :])
                 a = a.at[-halo + i, :, :, :].set(a[ halo + i,   :, :, :])
          return a
