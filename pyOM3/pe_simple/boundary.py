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


      def __init__(self,parameter = {},n_pes_x = 1,n_pes_y = 1, halo_points = 1):  
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
         else:      self.domain_decomposition_mpi(n_pes_x,n_pes_y)
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
    
      
    
      @partial(OM.jaxjit, static_argnums=(0,))  # todo: account for other halo sizes
      def apply_bc_mpi(self,a):
          """
          apply horizontally periodic or closed boundary condition to halo points
          of 2D/3D array for distributed domain and 
          exchange interior halo points using MPI decomposition.
          """          
          z = slice(None) if len(a.shape) == 3 else None
          bc_value = 0
              
          if self.n_pes_y > 1:   
             
             south = self.my_pe + self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == 1            else  \
                     self.my_pe - self.n_pes_x  
             north = self.my_pe - self.n_pes_x * (self.n_pes_y-1) if self.my_blk_y == self.n_pes_y else \
                     self.my_pe + self.n_pes_x
        
             buf = OM.np.empty( (self.Nz if len(a.shape)==3 else 1,self.Nx), OM.prec)
            
             if self.periodic_in_y or self.my_blk_y > 1:                 
                OM.send(  a[(z,1,slice(None))] ,south, 49)                
             if self.periodic_in_y or self.my_blk_y < self.n_pes_y:                    
                a = OM.modify_array(a, (z,-1,slice(None)),  OM.recv(buf,north, 49) )
             else: 
                a = OM.modify_array(a, (z,-1,slice(None)) , bc_value )               
             if self.periodic_in_y or self.my_blk_y < self.n_pes_y:                  
                OM.send( a[(z,-2,slice(None))], north, 153)
             if self.periodic_in_y or self.my_blk_y > 1:                  
                a = OM.modify_array(a, (z,0,slice(None)),  OM.recv(buf, south, 153)   ) 
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

             buf = OM.np.empty( (self.Nz if len(a.shape)==3 else 1,self.Ny), OM.prec)

             if self.periodic_in_x or self.my_blk_x > 1:  
                OM.send(  a[(z,slice(None),1)]  , west, 49)                
             if self.periodic_in_x or self.my_blk_x < self.n_pes_x:   
                a = OM.modify_array(a, (z,slice(None),-1),  OM.recv(buf, east, 49) )  
             else:
                a = OM.modify_array(a, (z,slice(None),-1),  bc_value )                  
             if self.periodic_in_x or self.my_blk_x < self.n_pes_x:              
                OM.send( a[(z,slice(None),-2)] , east, 153)
             if self.periodic_in_x or self.my_blk_x > 1:          
                a = OM.modify_array(a, (z,slice(None),0),  OM.recv(buf, west, 153) )  
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
                 OM.send( OM.np.array( self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe ,OM.prec), 0, 49)         
                 OM.send( a[z,self.halo:-self.halo,self.halo:-self.halo], 0, 153)                 
              if self.my_pe == 0:        
                 buf = OM.np.empty( (4,), OM.prec)  
                 xs, xe, ys, ye = OM.recv(buf, pe, 49)         
                 buf = b[(slice(None),slice(ys-1,ye),slice(xs-1,xe))].copy()  
                 b = OM.modify_array(b, (slice(None),slice(ys-1,ye),slice(xs-1,xe)), 
                                        OM.recv(buf, pe,  153))    
          return b

