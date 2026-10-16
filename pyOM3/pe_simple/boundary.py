"""
 Setting halo points to boundary conditions or inner exchange
 using either JAX or MPI domain decomposition.

 Based on classes decomposition_mpi or decomposition_jax

 apply_bc applies everything to 2D or 3D array.
 
"""
import pyOM3                             as OM
from   pyOM3.pe_simple.decomposition_mpi import decomposition_mpi
from   pyOM3.pe_simple.decomposition_jax import decomposition_jax
from   functools                         import partial


class boundary(decomposition_mpi,decomposition_jax):


      def __init__(self,parameter = {},n_pes_x = 1,n_pes_y = 1, halo_points = 1, comm_in = None):  
         """
         Add domain decomposition to initialisation.
         """
         super().__init__(parameter) 
          
         self.halo = halo_points             # number of halo points at each side
         nx,ny,nz = self.nx,self.ny,self.nz  # from input, nx,ny,nz is shape without halo points 
         # calculate shape with halo points -> Nx,Ny,Nz
         if OM.jax:
             Nx, Ny, Nz = nx + 2*self.halo*n_pes_x, ny + 2*self.halo*n_pes_y, nz + 2*self.halo
         else: 
             Nx, Ny, Nz = nx + 2*self.halo, ny + 2*self.halo, nz + 2*self.halo 
         self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
 
         if OM.jax: self.domain_decomposition_jax(n_pes_x,n_pes_y) 
         else:      self.domain_decomposition_mpi(n_pes_x,n_pes_y,comm_in)
         return

    
      @partial(OM.jaxjit, static_argnums=(0,))    
      def get_full_domain_extent(self): 
          """
           returns full horizontal extent as integers using either MPI decomposition or JAX sharding
          """ 
          if OM.jax: return self.get_full_domain_extent_jax()
          else:      return self.get_full_domain_extent_mpi()

    
      @partial(OM.jaxjit, static_argnums=(0,))
      def apply_bc(self,x):
          """
          apply horizontally periodic or closed boundary condition to halo points
          of 2D/3D array for distributed domain and 
          exchange interior halo points using MPI or JAX sharding.
          """                    
          if OM.jax: return self.apply_bc_jax(x)
          else:      return self.apply_bc_mpi(x)
          

      @partial(OM.jaxjit, static_argnums=(0,))
      def apply_bc_jax(self,x):
          """
          apply horizontally periodic or closed boundary condition to halo points
          of 2D/3D array for distributed domain and 
          exchange interior halo points using JAX sharding.
          """          
          halo,n_pes_x,n_pes_y  = self.halo , self.n_pes_x, self.n_pes_y
          x_sl = slice(None) 
          y_sl = slice(None) if len(x.shape) > 1 else None
          z_sl = slice(None) if len(x.shape) > 2 else None  
          
          local_shard = self.shard_map_arr_3D if len(x.shape) == 3 else self.shard_map_arr_2D

          @local_shard
          def halo_exchange_x(x):                
              west_halo = x[z_sl,y_sl,halo : 2 * halo]
              east_halo = x[z_sl,y_sl,-(2 * halo) : -halo]
              received_west_halo  = OM.jax.lax.ppermute(east_halo , axis_name='x', perm=self.eastward)
              received_east_halo  = OM.jax.lax.ppermute(west_halo , axis_name='x', perm=self.westward)
              x = x.at[z_sl,y_sl,0:halo].set(received_west_halo)
              x = x.at[z_sl,y_sl,-halo:].set(received_east_halo)
              return x  

          @local_shard
          def halo_exchange_y(x):                
              south_halo = x[z_sl,halo : 2 * halo,x_sl]
              north_halo = x[z_sl,-(2 * halo) : -halo,x_sl]
              received_south_halo  = OM.jax.lax.ppermute(north_halo , axis_name='y', perm=self.northward)
              received_north_halo  = OM.jax.lax.ppermute(south_halo , axis_name='y', perm=self.southward)
              x = x.at[z_sl,0:halo,x_sl].set(received_south_halo)
              x = x.at[z_sl,-halo:,x_sl].set(received_north_halo)
              return x 

          # closed boundaries are automatically filled with zeros    
          x = halo_exchange_x(x)  
          x = halo_exchange_y(x)  
          return x
    
      
    
      def apply_bc_multi(self,*xs):
          """
          apply_bc for several 2D/3D arrays at once. Under JAX sharding the halo
          strips of all arrays travel in ONE ppermute per direction (4 collectives
          per call instead of 4 per array); the arrays themselves are not copied.
          Under MPI it is apply_bc on each array. Returns a tuple.
          """
          if OM.jax: return self.apply_bc_multi_jax(*xs)
          else:      return tuple(self.apply_bc_mpi(x) for x in xs)


      @partial(OM.jaxjit, static_argnums=(0,))
      def apply_bc_multi_jax(self,*xs):
          """
          Halo exchange of several arrays with one ppermute per direction. The
          strips of all arrays are flattened, concatenated, sent, and put back.
          Each array is 2D (Ny,Nx) or 3D (Nz,Ny,Nx) in the layout of apply_bc_jax.
          """
          halo  = self.halo
          specs = tuple(self.grid_3D if x.ndim == 3 else self.grid_2D for x in xs)
          shard = OM.shard_map(lambda *a: self._exchange_strips(*a), mesh=self.mesh,
                               in_specs=specs, out_specs=specs)
          return shard(*xs)


      def _exchange_strips(self,*xs):
          """body of apply_bc_multi_jax, runs per shard"""
          halo = self.halo
          def permute(strips, axis_name, perm):
              flat  = OM.np.concatenate([st.reshape(-1) for st in strips])
              recv  = OM.jax.lax.ppermute(flat, axis_name=axis_name, perm=perm)
              out, i = [], 0
              for st in strips:
                  out.append(recv[i:i+st.size].reshape(st.shape)); i += st.size
              return out
          # x direction: send east strip eastward (becomes west halo of neighbour) and v.v.
          east = [x[..., :, -(2*halo):-halo] for x in xs]
          west = [x[..., :,  halo:2*halo]    for x in xs]
          rw = permute(east, 'x', self.eastward)
          re = permute(west, 'x', self.westward)
          xs = [x.at[..., :, 0:halo].set(w).at[..., :, -halo:].set(e) for x, w, e in zip(xs, rw, re)]
          # y direction
          north = [x[..., -(2*halo):-halo, :] for x in xs]
          south = [x[...,  halo:2*halo,    :] for x in xs]
          rs = permute(north, 'y', self.northward)
          rn = permute(south, 'y', self.southward)
          xs = [x.at[..., 0:halo, :].set(s).at[..., -halo:, :].set(n) for x, s, n in zip(xs, rs, rn)]
          return tuple(xs)


      @partial(OM.jaxjit, static_argnums=(0,))
      def apply_bc_mpi_old(self,a):
          """
          ORIGINAL pe_simple halo exchange, kept for reference. Superseded by
          apply_bc_mpi below.

          This version uses a plain send-first ordering with a single tag pair
          (49/153) and hardcoded halo=1 indices. It DEADLOCKS for a periodic
          decomposed direction once a halo message exceeds the MPI eager-buffer
          limit: with n_pes >= 2 and periodicity, neighbouring blocks each issue
          a blocking Send before any Recv, and for large messages MPI switches
          from eager buffering to rendezvous, so both ranks block in Send. Small
          fields (2D / low-resolution) stay under the eager limit and pass, which
          is why this went unnoticed until a full-size 3D field was exchanged.
          It also only works for halo = 1. Do not use; here only for comparison.
          """
          z = slice(None) if len(a.shape) == 3 else None
          bc_value = 0

          if self.n_pes_y > 1:

             south = self.my_pe + self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == 1            else  \
                     self.my_pe - self.n_pes_x
             north = self.my_pe - self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == self.n_pes_y else \
                     self.my_pe + self.n_pes_x

             buf = OM.np.empty( (a.shape[0] if len(a.shape)==3 else 1,self.Nx), OM.prec)

             if self.periodic_in_y or self.my_blk_y > 1:
                self.send(  a[(z,1,slice(None))] ,south, 49)
             if self.periodic_in_y or self.my_blk_y < self.n_pes_y:
                a = OM.modify_array(a, (z,-1,slice(None)),  self.recv(buf,north, 49) )
             else:
                a = OM.modify_array(a, (z,-1,slice(None)) , bc_value )
             if self.periodic_in_y or self.my_blk_y < self.n_pes_y:
                self.send( a[(z,-2,slice(None))], north, 153)
             if self.periodic_in_y or self.my_blk_y > 1:
                a = OM.modify_array(a, (z,0,slice(None)),  self.recv(buf, south, 153)   )
             else:
                a = OM.modify_array(a, (z,0,slice(None)) , bc_value )
          else:
             if self.periodic_in_y:
                 a = OM.modify_array(a, (z, 0,slice(None)) , a[(z,-2,slice(None))] )
                 a = OM.modify_array(a, (z,-1,slice(None)) , a[(z, 1,slice(None))] )
             else:
                 a = OM.modify_array(a, (z, 0,slice(None)), bc_value )
                 a = OM.modify_array(a, (z,-1,slice(None)), bc_value )

          if self.n_pes_x > 1:

             west = self.my_pe + self.n_pes_x - 1   if self.my_blk_x == 1            else self.my_pe - 1
             east = self.my_pe - (self.n_pes_x - 1) if self.my_blk_x == self.n_pes_x else self.my_pe + 1

             buf = OM.np.empty( (a.shape[0] if len(a.shape)==3 else 1,self.Ny), OM.prec)

             if self.periodic_in_x or self.my_blk_x > 1:
                self.send(  a[(z,slice(None),1)]  , west, 49)
             if self.periodic_in_x or self.my_blk_x < self.n_pes_x:
                a = OM.modify_array(a, (z,slice(None),-1),  self.recv(buf, east, 49) )
             else:
                a = OM.modify_array(a, (z,slice(None),-1),  bc_value )
             if self.periodic_in_x or self.my_blk_x < self.n_pes_x:
                self.send( a[(z,slice(None),-2)] , east, 153)
             if self.periodic_in_x or self.my_blk_x > 1:
                a = OM.modify_array(a, (z,slice(None),0),  self.recv(buf, west, 153) )
             else:
                a = OM.modify_array(a, (z,slice(None),0),  bc_value )
          else:
             if self.periodic_in_x:
                 a = OM.modify_array(a, (z,slice(None), 0),  a[(z,slice(None),-2)]  )
                 a = OM.modify_array(a, (z,slice(None),-1),  a[(z,slice(None), 1)]  )
             else:
                 a = OM.modify_array(a, (z,slice(None), 0),  bc_value )
                 a = OM.modify_array(a, (z,slice(None),-1),  bc_value )
          return a


      @partial(OM.jaxjit, static_argnums=(0,))
      def apply_bc_mpi(self,a):
          """
          Apply horizontally periodic or closed boundary conditions to the halo
          points of a 2D/3D array and exchange interior halo points across the
          MPI domain decomposition (x,y only; z is always local in pe_simple).

          Arrays are C-ordered (Nz,Ny,Nx). The exchange uses the deadlock-free
          tag-splitting pattern from nh_simple: interior neighbour exchanges use
          tags 49/153, while the periodic-wrap seam (block 1 <-> block n_pes)
          uses separate tags 47/152. Splitting the seam onto its own tags means
          that with n_pes = 2 in a direction -- where a block's two neighbours
          are the same PE -- the sends and receives still pair up, so a blocking
          Send can never sit waiting on a Send (which is what deadlocks the
          plain send-first ordering once a message exceeds the MPI eager-buffer
          limit). It also loops over self.halo, so it works for any halo width.
          """
          z = slice(None) if len(a.shape) == 3 else None
          bc_value = 0

          # ------------------------- y exchange -------------------------
          if self.n_pes_y > 1:
             south = self.my_pe + self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == 1            else \
                     self.my_pe - self.n_pes_x
             north = self.my_pe - self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == self.n_pes_y else \
                     self.my_pe + self.n_pes_x
             buf = OM.np.empty( (a.shape[0] if len(a.shape)==3 else 1, self.Nx), OM.prec)

             for i in range(self.halo):
                 # send my southern interior row (halo+i) to south,
                 # receive my northern halo (i-halo) from north
                 sl_send = (z, self.halo+i, slice(None))
                 sl_recv = (z, i-self.halo,  slice(None))
                 if self.my_blk_y != 1:            self.send(a[sl_send], south, 49)
                 if self.my_blk_y != self.n_pes_y: a = OM.modify_array(a, sl_recv, self.recv(buf, north, 49))
                 if self.periodic_in_y:
                    if self.my_blk_y == 1:            self.send(a[sl_send], south, 47)
                    if self.my_blk_y == self.n_pes_y: a = OM.modify_array(a, sl_recv, self.recv(buf, north, 47))
                 else:
                    if self.my_blk_y == self.n_pes_y: a = OM.modify_array(a, sl_recv, bc_value)

                 # send my northern interior row (-halo-1-i) to north,
                 # receive my southern halo (halo-1-i) from south
                 sl_send = (z, -self.halo-1-i, slice(None))
                 sl_recv = (z,  self.halo-1-i, slice(None))
                 if self.my_blk_y != self.n_pes_y: self.send(a[sl_send], north, 153)
                 if self.my_blk_y != 1:            a = OM.modify_array(a, sl_recv, self.recv(buf, south, 153))
                 if self.periodic_in_y:
                    if self.my_blk_y == self.n_pes_y: self.send(a[sl_send], north, 152)
                    if self.my_blk_y == 1:            a = OM.modify_array(a, sl_recv, self.recv(buf, south, 152))
                 else:
                    if self.my_blk_y == 1:            a = OM.modify_array(a, sl_recv, bc_value)
          else:
             for i in range(self.halo):
                 sl_send = (z, -self.halo-1-i, slice(None)); sl_recv = (z, self.halo-1-i, slice(None))
                 if self.periodic_in_y: a = OM.modify_array(a, sl_recv, a[sl_send])
                 else:                  a = OM.modify_array(a, sl_recv, bc_value)
                 sl_send = (z, self.halo+i, slice(None));    sl_recv = (z, i-self.halo, slice(None))
                 if self.periodic_in_y: a = OM.modify_array(a, sl_recv, a[sl_send])
                 else:                  a = OM.modify_array(a, sl_recv, bc_value)

          # ------------------------- x exchange -------------------------
          if self.n_pes_x > 1:
             west = self.my_pe + self.n_pes_x - 1   if self.my_blk_x == 1            else self.my_pe - 1
             east = self.my_pe - (self.n_pes_x - 1) if self.my_blk_x == self.n_pes_x else self.my_pe + 1
             buf = OM.np.empty( (a.shape[0] if len(a.shape)==3 else 1, self.Ny), OM.prec)

             for i in range(self.halo):
                 # send my western interior column (halo+i) to west,
                 # receive my eastern halo (i-halo) from east
                 sl_send = (z, slice(None), self.halo+i)
                 sl_recv = (z, slice(None), i-self.halo)
                 if self.my_blk_x != 1:            self.send(a[sl_send], west, 49)
                 if self.my_blk_x != self.n_pes_x: a = OM.modify_array(a, sl_recv, self.recv(buf, east, 49))
                 if self.periodic_in_x:
                    if self.my_blk_x == 1:            self.send(a[sl_send], west, 47)
                    if self.my_blk_x == self.n_pes_x: a = OM.modify_array(a, sl_recv, self.recv(buf, east, 47))
                 else:
                    if self.my_blk_x == self.n_pes_x: a = OM.modify_array(a, sl_recv, bc_value)

                 # send my eastern interior column (-halo-1-i) to east,
                 # receive my western halo (halo-1-i) from west
                 sl_send = (z, slice(None), -self.halo-1-i)
                 sl_recv = (z, slice(None),  self.halo-1-i)
                 if self.my_blk_x != self.n_pes_x: self.send(a[sl_send], east, 153)
                 if self.my_blk_x != 1:            a = OM.modify_array(a, sl_recv, self.recv(buf, west, 153))
                 if self.periodic_in_x:
                    if self.my_blk_x == self.n_pes_x: self.send(a[sl_send], east, 152)
                    if self.my_blk_x == 1:            a = OM.modify_array(a, sl_recv, self.recv(buf, west, 152))
                 else:
                    if self.my_blk_x == 1:            a = OM.modify_array(a, sl_recv, bc_value)
          else:
             for i in range(self.halo):
                 sl_send = (z, slice(None), -self.halo-1-i); sl_recv = (z, slice(None), self.halo-1-i)
                 if self.periodic_in_x: a = OM.modify_array(a, sl_recv, a[sl_send])
                 else:                  a = OM.modify_array(a, sl_recv, bc_value)
                 sl_send = (z, slice(None), self.halo+i);    sl_recv = (z, slice(None), i-self.halo)
                 if self.periodic_in_x: a = OM.modify_array(a, sl_recv, a[sl_send])
                 else:                  a = OM.modify_array(a, sl_recv, bc_value)
          return a


      def gather_array_mpi(self,a):   # todo: account for other halo sizes
          """
          returns array of complete domain gathered by PE0 
          works for 2D and 3D distributed domains
          """
          z = slice(self.halo,-self.halo) if len(a.shape) == 3 else None
        
          nx, ny = self.get_full_domain_extent()
          nz = self.Nz-2
          
          # transfer local domain
          if self.my_pe==0:
             b = OM.np.empty( (nz if len(a.shape)==3 else 1,ny,nx), OM.prec)  
             b = OM.modify_array(b, (slice(None), slice(self.ys_pe-1,self.ye_pe), slice(self.xs_pe-1,self.xe_pe)),
                                 a[z,self.halo:-self.halo,self.halo:-self.halo] )  
          else:
             b = None

          # send and transfer non-local domains
          for pe in range(1,self.n_pes):       
              if self.my_pe == pe:   
                 self.send( OM.np.array( [self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe] ,OM.prec), 0, 49)         
                 self.send( a[z,self.halo:-self.halo,self.halo:-self.halo], 0, 153)                 
              if self.my_pe == 0:        
                 buf = OM.np.empty( (4,), OM.prec)  
                 xs, xe, ys, ye = ( int(v) for v in self.recv(buf, pe, 49) )         
                 buf = b[(slice(None),slice(ys-1,ye),slice(xs-1,xe))].copy()  
                 b = OM.modify_array(b, (slice(None),slice(ys-1,ye),slice(xs-1,xe)), 
                                        self.recv(buf, pe,  153))    
          return b

