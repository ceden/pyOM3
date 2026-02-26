
"""
 Domain decomposition using manual JAX sharding.
 Based on basic class setup.

 domain_decomposition_jax generates the mesh and shardings needed
 to deal with JAX sharding

 Defines decorator functions shard_map_arr_2/3D to deal 
 with jax.jit and sharding

 Defines several helper functions: 
 pad/unpad:         adds/removes halo points from arrays
 rollx,rolly,rollz: shift index in x/y/z by n points
                    (Note: shift=-1 denotes shift in positive i-direction)
                    
"""

import pyOM3           as OM
from   pyOM3.pe_simple import setup
from   functools       import partial
import numpy           as np__


class decomposition_jax(setup):

    
      def domain_decomposition_jax(self,n_pes_x,n_pes_y):    
          """
           Horizontal domain decomposition with JAX sharding.
           Input is number of shards in x,y: n_pes_x,n_pes_y.
           Defines what's needed to deal with JAX sharding.
          """
          self.my_pe, self.n_pes = OM.jax.process_index() ,   OM.jax.device_count()
          self.n_pes_x , self.n_pes_y = n_pes_x, n_pes_y
          
          self.mesh        = OM.jax.make_mesh( axis_shapes =  (self.n_pes_y,self.n_pes_x ),    axis_names = ('y','x') )
          self.grid_2D     = OM.jax.sharding.PartitionSpec('y','x')
          self.grid_3D     = OM.jax.sharding.PartitionSpec(None,'y','x')
          self.grid_4D     = OM.jax.sharding.PartitionSpec(None,None,'y','x')
          self.sharding_2D = OM.jax.sharding.NamedSharding(self.mesh, self.grid_2D)
          self.sharding_3D = OM.jax.sharding.NamedSharding(self.mesh, self.grid_3D)
          self.sharding_4D = OM.jax.sharding.NamedSharding(self.mesh, self.grid_4D)
                    
          self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe  =  1, self.Nx, 1, self.Ny   
          
          self.eastward  = [(i, (i + 1) % n_pes_x) for i in range(n_pes_x)]
          self.westward  = [(i, (i - 1) % n_pes_x) for i in range(n_pes_x)]
          self.northward = [(i, (i + 1) % n_pes_y) for i in range(n_pes_y)]
          self.southward = [(i, (i - 1) % n_pes_y) for i in range(n_pes_y)]
           
          if not self.periodic_in_x: 
              self.eastward = self.eastward[:-1]
              self.westward = self.westward[1:]
          if not self.periodic_in_y: 
              self.northward = self.northward[:-1]
              self.southward = self.southward[1:]
                      
          return

    
      def get_full_domain_extent_jax(self): 
          """
           returns full horizontal extent as integers using JAX sharding
          """ 
          nx,ny = self.Nx - 2*self.halo*self.n_pes_x , self.Ny - 2*self.halo*self.n_pes_y
          return  nx,ny      

    
      def shard_map_arr_2D(self, func: callable) -> callable:                     
          return OM.shard_map(func, mesh=self.mesh,  in_specs=self.grid_2D, out_specs=self.grid_2D) 
              
      def shard_map_arr_3D(self, func: callable) -> callable:          
          return OM.shard_map(func, mesh=self.mesh,  in_specs=self.grid_3D, out_specs=self.grid_3D) 

      def shard_map_arr_4D(self, func: callable) -> callable:          
          return OM.shard_map(func, mesh=self.mesh,  in_specs=self.grid_4D, out_specs=self.grid_4D)     
    
      def shard_map_reduce_2D(self, func: callable) -> callable:
          return OM.shard_map(func, mesh=self.mesh,  in_specs=self.grid_2D, out_specs=OM.jax.sharding.PartitionSpec()) 
          
      def shard_map_reduce_3D(self, func: callable) -> callable:
          return OM.shard_map(func, mesh=self.mesh,  in_specs=self.grid_3D, out_specs=OM.jax.sharding.PartitionSpec())


      @partial(OM.jaxjit, static_argnums=(0,))   
      def pad(self,x):
          """
           add halo points to 2D/3D array, account for JAX sharding 
          """ 
          p = (self.halo,self.halo)
          local_shard = self.shard_map_arr_3D if len(x.shape) == 3 else self.shard_map_arr_2D
          @local_shard
          def pad_it(x):
              x = OM.np.pad(x, (p,p) ) if len(x.shape) == 2 else OM.np.pad(x, (p,p,p) )
              return x
          return pad_it(x)

    
      @partial(OM.jaxjit, static_argnums=(0,))   
      def unpad(self,x):
          """
           remove halo points from 2D/3D array, account for JAX sharding 
          """           
          p = slice(self.halo,-self.halo)
          local_shard = self.shard_map_arr_3D if len(x.shape) == 3 else self.shard_map_arr_2D
          @local_shard
          def unpad_it(x):
              x = x[(p,p)] if len(x.shape) == 2 else x[(p,p,p)]
              return x
          return unpad_it(x)
    
          
      @partial(OM.jaxjit, static_argnums=(0,))          
      def rollx(self,x,shift):
          """
          Shift last (x) axis in 2D/3D array by "shift" points, account for JAX sharding.
          Note: shift=-1 shifts in positive direction.
          """
          local_shard = self.shard_map_arr_3D if len(x.shape) == 3 else self.shard_map_arr_2D
          z = slice(self.halo,-self.halo) if len(x.shape) > 2 else None  
          @local_shard
          def roll_it(x):  
              x = OM.np.roll(x, shift=shift,axis=-1) 
              return x
          return roll_it(x)

    
      @partial(OM.jaxjit, static_argnums=(0,))          
      def rolly(self,x,shift):
          """
          Shift second (y) axis in 2D/3D array by "shift" points, account for JAX sharding.
          Note: shift=-1 shifts in positive direction.
          """         
          local_shard = self.shard_map_arr_3D if len(x.shape) == 3 else self.shard_map_arr_2D
          z = slice(self.halo,-self.halo) if len(x.shape) > 2 else None  
          @local_shard
          def roll_it(x):  
              x = OM.np.roll(x, shift=shift,axis=-2)
              return x
          return roll_it(x)

    
      @partial(OM.jaxjit, static_argnums=(0,))          
      def rollz(self,x,shift):   
          """
          Shift first (z) axis in 2D/3D array by "shift" points, account for JAX sharding.
          Note: shift=-1 shifts in positive direction.
          """                   
          @self.shard_map_arr_3D 
          def roll_it(x):  
              x = OM.np.roll(x, shift=shift,axis=0)
              return x
          return roll_it(x)
         
          
  