"""
 Brings together all component for the PE model.
 Defines time steps for different options and a loops over them.

 Based on diverse other classes. 
"""
import pyOM3 as OM
from   pyOM3.pe_simple.thermodynamics import thermodynamics
from   pyOM3.pe_simple.momentum       import momentum
from   pyOM3.pe_simple.pressure       import pressure
from   pyOM3.pe_simple.free_surface   import free_surface

from   pyOM3.pe_simple.damping        import damping
from   pyOM3.density                  import get_rho
import numpy                          as np__
from   functools                      import partial


class model(thermodynamics,momentum,pressure,free_surface,damping):

      
      def set_coriolis(self,f0):
          """
          Set Coriolis frequency on T grid.
          Overload this method with different choices
          """
          self.coriolis_t = OM.modify_array(self.coriolis_t,(slice(None),slice(None)), f0, out_sharding = self.sharding_2D )
          return

      
      def set_topography(self):
          """
          Set topography. kbot=k denotes bottom is below k-th model level.
          Overload this method with different choices.
          """
          self.kbot = OM.modify_array(self.kbot,(slice(None),slice(None)), self.halo , out_sharding = self.sharding_2D  )           
          return


      def set_initial_conditions(self,_u,_v,_temp,_salt,_p_s):
          """
          Convenience function to define initial conditions based on the input.
          Surface pressure p_s can be None for rigid lid case.
          All input fields are full size, thus this method might request too much memory
          for large models.
          """
          if  OM.jax:  return self.set_initial_conditions_jax(_u,_v,_temp,_salt,_p_s)
          else:        return self.set_initial_conditions_mpi(_u,_v,_temp,_salt,_p_s)
         
    
      def set_initial_conditions_jax(self,_u,_v,_temp,_salt,_p_s):  
          """
          Convenience function to define initial conditions based on the input 
          for JAX sharding. Surface pressure p_s can be None for rigid lid case.
          All input fields are full size, thus this method might request too much memory
          for large models.
          """
          self.u = self.apply_bc(  self.pad(  OM.jax.device_put( _u , self.sharding_3D)  )  ) #*self.maskU
          self.v = self.apply_bc(  self.pad(  OM.jax.device_put( _v , self.sharding_3D)   )  ) #*self.maskV
          self.w    = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          self.temp = self.apply_bc( self.pad(  OM.jax.device_put( _temp , self.sharding_3D)  ) )#*self.maskT
          self.salt = self.apply_bc( self.pad(  OM.jax.device_put( _salt , self.sharding_3D)  ) )#*self.maskT
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if self.implicit_free_surface:
             p_s = self.pad(  OM.jax.device_put( _p_s , self.sharding_2D)  ) #*self.maskT[-self.halo-1,:,:]   
             self.p_s   = OM.modify_array(self.p_s ,(0,slice(None),slice(None)), self.apply_bc(p_s)  )   
             self.p_s   = OM.modify_array(self.p_s ,(1,slice(None),slice(None)), self.p_s[0,:,:])
             self.p_s   = OM.modify_array(self.p_s ,(2,slice(None),slice(None)), self.p_s[0,:,:]) 
          if self.explicit_free_surface:   
             self.p_s = self.apply_bc( self.pad(  OM.jax.device_put( _p_s , self.sharding_2D)  )  ) #*self.maskT[-self.halo-1,:,:]   
          return 

    
      def set_initial_conditions_mpi(self,_u,_v,_temp,_salt,_p_s):
          """
          Convenience function to define initial conditions based on the input 
          for MPI domain decomposition. Surface pressure p_s can be None for rigid lid case.
          All input fields are full size, thus this method might request too much memory
          for large models.
          """
          
          a,b = slice(None), slice(self.halo,-self.halo)
          ex = (a,slice(self.ys_pe-1,self.ye_pe),slice(self.xs_pe-1,self.xe_pe))          
          self.u    = self.apply_bc( OM.modify_array(self.u ,(b,b,b), _u[ex]) ) #*self.maskU
          self.v    = self.apply_bc( OM.modify_array(self.v ,(b,b,b), _v[ex]) ) #*self.maskV
          self.w    = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          self.temp = self.apply_bc( OM.modify_array(self.temp ,(b,b,b), _temp[ex]) ) #*self.maskT
          self.salt = self.apply_bc( OM.modify_array(self.salt ,(b,b,b), _salt[ex]) ) #*self.maskT         
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          ex = (slice(self.ys_pe-1,self.ye_pe),slice(self.xs_pe-1,self.xe_pe))          
          if self.implicit_free_surface:
             self.p_s   = OM.modify_array(self.p_s ,(0,b,b), _p_s[ex])    #*self.maskT[-self.halo-1,:,:] 
             self.p_s   = OM.modify_array(self.p_s ,(0,a,a), self.apply_bc( self.p_s[0,:,:])  )  
             self.p_s   = OM.modify_array(self.p_s ,(1,a,a), self.p_s[0,:,:])
             self.p_s   = OM.modify_array(self.p_s ,(2,a,a), self.p_s[0,:,:])
          if self.explicit_free_surface:
             self.p_s   = self.apply_bc(  OM.modify_array(self.p_s ,(b,b), _p_s[ex])   )   #*self.maskT[-self.halo-1,:,:]             
          return
    
    
      def prepare_for_run(self):
          """
          Call everything which is needed for the main time stepping loop
          """
          self.make_topo_masks()
          self.cf = self.make_coeff_surf_press(self.cf,self.hu,self.hv,self.maskT)
          return


      def time_step_explicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the split-explicit method after Demange et al (2019).
          Chosen with self.explicit_free_surface = True
          """
          self.du, self.dv,self.du_cor,self.dv_cor,self.p_h = self.momentum_tendency_expl(self.u,self.v,self.w,
                                                                                          self.du,self.dv,self.du_cor,self.dv_cor,
                                                                                          self.rho,
                                                                                          self.maskU,self.maskV,self.maskW,self.maskT,
                                                                                          self.coriolis_t,tau)
          
          self.u, self.v,  self.p_s = self.solve_expl_surface(self.u,self.v,
                                                              self.du,self.dv,self.du_cor,self.dv_cor,
                                                              self.p_h,self.p_s,
                                                              self.maskU,self.maskV,self.maskW,self.maskT,
                                                              self.hu,self.hv,self.coriolis_t,
                                                              A,B,C,tau,taum1,taum2)  
          
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          
          self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                             self.maskU,self.maskV,self.maskW,self.maskT,tau)  

          if hasattr(self,'Khbi'):  
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)
             
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                self.maskU,self.maskV,self.maskW,self.maskT,tau) 
             if hasattr(self,'Khbi'):  
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)   

          self.temp, self.salt, self.rho = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                         self.temp,self.salt,self.dtemp,self.dsalt,self.maskT)          
          return


          
      def time_step_implicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the rigid lid or implicit free surface method.
          """
          
          self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                             self.maskU,self.maskV,self.maskW,self.maskT,tau)  
              
          if hasattr(self,'Khbi'):  
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)
             
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                self.maskU,self.maskV,self.maskW,self.maskT,tau) 
             # background advection only if temperature is buoyancy, then there is no salt 
             if hasattr(self,'Khbi'):  
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)   
              
          self.du, self.dv, self.p_h = self.momentum_tendency_impl(self.u,self.v,self.w,
                                                                   self.du,self.dv,self.rho,self.p_h,
                                                                   self.maskU,self.maskV,self.maskW,self.maskT,
                                                                   self.coriolis_t,tau)               
          if hasattr(self,'Ahbi'):  
                       self.u = self.biharmonic_horizontal(self.u,self.Ahbi,self.maskU)
                       self.v = self.biharmonic_horizontal(self.v,self.Ahbi,self.maskV)          
             
          self.u, self.v = self.update_momentum_AB3(A,B,C,tau,taum1,taum2,
                                                    self.u,self.v,self.du,self.dv,self.maskU,self.maskV)
                                         
          self.u, self.v, self.p_s, \
               self.estimated_error, self.congr_itts  =  self.solve_for_pressure(self.u,self.v,self.p_s,
                                                                                 self.maskU,self.maskV,self.maskT,self.cf,
                                                                                 tau,taum1,taum2)
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
                         
          self.temp, self.salt, self.rho = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                         self.temp,self.salt,self.dtemp,self.dsalt,self.maskT)
          return

    
      def loop(self,max_steps=1000000):         
          """
          Loop max_steps time over time steps with Adam-Bashforth 3.order.
          """
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          for n in range(max_steps): # time step equations 
            if self.explicit_free_surface:  self.time_step_explicit_pressure(A,B,C,tau,taum1,taum2) 
            else:                           self.time_step_implicit_pressure(A,B,C,tau,taum1,taum2)  
            self.diagnose(n,tau)               
            tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)          
            if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
            elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.
          return    

    
      def diagnose(self,n,tau):  
          """
          Dummy routine for diagnostics.
          """
          t =  n*self.dt
          if self.my_pe == 0:      
                if hasattr(self,'congr_itts'):  print ("t=",t," n=",n," itts=",self.congr_itts)
                else:                           print ("t=",t," n=",n)
     