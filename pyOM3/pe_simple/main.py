"""
 main class of PE model defining/allocating model variables 
 and setting topography masks.

 Based on class boundary dealing with domain decomposition.
""" 
import pyOM3                     as OM
from   pyOM3.pe_simple.boundary  import boundary
from   functools                 import partial


class main(boundary):

    
      def __init__(self,parameter = {}, n_pes_x = 1, n_pes_y = 1, halo_points = 1 ):  
         """
         add some constants and methods to initialisation.
         """
         super().__init__(parameter, n_pes_x, n_pes_y, halo_points = halo_points)
         self.grav ,  self.rho_0  = 9.81, 1024.     
         self.allocate_workspace()        
         self.set_mean_pressure()
         return

    
      def set_mean_pressure(self):
          """
          set mean pressure or depth levels for equation of state 
          TODO: p0 needs to be 3D
          """
          if   self.eq_of_state == 0 or self.eq_of_state == 1:  
             self.p0 = None
          elif self.eq_of_state == 100:

               zt = OM.np.arange(self.Nz-2*self.halo,dtype=OM.prec)*self.dz -self.dz/2
               zw = zt + self.dz/2
               #zt = zt + zw(2) 
               #zw = zw + zw(2)  # p=0 at zw(0) for atmosphere setup which is upside down
               zt += zw[1]
               zw += zw[1]
               for n in range(self.halo):
                 zt = OM.np.append( zt[0]-self.dz,zt )
                 zt = OM.np.append( zt, zt[-1]+self.dz )
               self.p0 = OM.modify_array( OM.np.zeros_like(self.u), (slice(None),slice(None),slice(None)),zt[:,None,None])
               
               
          else:
             raise pyOM3Error 
          return

         
      def allocate_workspace(self):
          """
          define all model variables compatible with JAX sharding.
          """
          @partial(OM.jaxjit, out_shardings=self.sharding_2D, static_argnums=(0,1))
          def zeros_2D(Ny,Nx):
              return OM.np.zeros((Ny,Nx), OM.prec ) 
          
          @partial(OM.jaxjit, out_shardings=self.sharding_3D, static_argnums=(0,1,2))
          def zeros_3D(Nz,Ny,Nx):
              return OM.np.zeros((Nz,Ny,Nx), OM.prec ) 
          
          @partial(OM.jaxjit, out_shardings=self.sharding_4D, static_argnums=(0,1,2,3))
          def zeros_4D(i,Nz,Ny,Nx):
              return OM.np.zeros((i,Nz,Ny,Nx), OM.prec ) 

          Nx, Ny, Nz = self.Nx, self.Ny, self.Nz 

          # allocate model variables with halo points
          self.u , self.du = zeros_3D(Nz,Ny,Nx) , zeros_4D(3,Nz,Ny,Nx)
          self.v , self.dv = zeros_3D(Nz,Ny,Nx) , zeros_4D(3,Nz,Ny,Nx)     
          self.w   = zeros_3D(Nz,Ny,Nx)
          self.rho = zeros_3D(Nz,Ny,Nx)  
          self.temp , self.dtemp =  zeros_3D(Nz,Ny,Nx) ,  zeros_4D(3,Nz,Ny,Nx)
          self.salt , self.dsalt =  zeros_3D(Nz,Ny,Nx) ,  zeros_4D(3,Nz,Ny,Nx)       
          
          if self.explicit_free_surface: 
              self.p_s  = zeros_2D(Ny,Nx)
              self.du_cor = zeros_4D(3,Nz,Ny,Nx)
              self.dv_cor = zeros_4D(3,Nz,Ny,Nx)
          else:
              self.p_s =  zeros_3D(3,Ny,Nx)
              
          self.p_h        = zeros_3D(Nz,Ny,Nx)
          self.coriolis_t = zeros_2D(Ny,Nx) 
          self.cf         = zeros_4D(3,3,Ny,Nx)
          self.kbot       = zeros_2D(Ny,Nx)          
          self.maskT      = zeros_3D(Nz,Ny,Nx)
          return

    
      def make_topo_masks(self):   
          """
          Define all topography masks.
          It is assumed that kbot is already defined.
          kbot=k denotes bottom is below k-th model level.
          """
          # close input field kbot at side walls
          self.kbot = self.apply_bc(self.kbot)    
          
          if not self.periodic_in_y: 
             if OM.jax or self.my_blk_y == 1:
                self.kbot = OM.modify_array(self.kbot, (slice( None,self.halo),slice(None)) ,0, out_sharding = self.sharding_2D )  
             if OM.jax or self.my_blk_y == self.n_pes_y:    
                self.kbot = OM.modify_array(self.kbot, (slice(-self.halo,None),slice(None)) ,0, out_sharding = self.sharding_2D )
          if not self.periodic_in_x: 
             if OM.jax or self.my_blk_x == 1:
                self.kbot = OM.modify_array(self.kbot, (self(None),slice( None,self.halo)) ,0, out_sharding = self.sharding_2D )
             if OM.jax or self.my_blk_x == self.n_pes_x:   
                self.kbot = OM.modify_array(self.kbot, (self(None),slice(-self.halo,None)) ,0, out_sharding = self.sharding_2D )
               
          # mask on T grid                     
          for k in range(self.halo,self.Nz-self.halo):  
             where =  OM.np.logical_and(self.kbot != 0, self.kbot <= k)
             self.maskT =  OM.modify_array(self.maskT, (k,slice(None),slice(None)),
                                           OM.np.where( where ,1.0,0.0) )
              
          # other masks         
          self.maskU = OM.np.minimum( self.rollx(self.maskT,-1), self.maskT )
          self.maskV = OM.np.minimum( self.rolly(self.maskT,-1), self.maskT )
          self.maskW = OM.np.minimum( self.rollz(self.maskT,-1), self.maskT )
          
          self.maskU = self.apply_bc(self.maskU)
          self.maskV = self.apply_bc(self.maskV)
          self.maskW = self.apply_bc(self.maskW)

          # open free surface in maskW
          if self.implicit_free_surface or self.explicit_free_surface:
              self.maskW = OM.modify_array(self.maskW,(-self.halo-1,slice(None),slice(None)),self.maskW[-self.halo-2,:,:])
              
          # total depth for pressure solver
          self.hu = OM.np.sum(self.maskU[self.halo:-self.halo,:,:]*self.dz,axis=0)
          self.hv = OM.np.sum(self.maskV[self.halo:-self.halo,:,:]*self.dz,axis=0)           
          return   

