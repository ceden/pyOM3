"""
 Brings together all component for the PE model.
 Defines time steps for different options and a loops over them.

 Based on diverse other classes. 
"""
import pyOM3 as OM
from   pyOM3.pe_realistic.thermodynamics import thermodynamics
from   pyOM3.pe_realistic.momentum       import momentum
from   pyOM3.pe_realistic.pressure       import pressure
from   pyOM3.pe_realistic.free_surface   import free_surface

from   pyOM3.pe_realistic.damping        import damping
from   pyOM3.pe_realistic.implicit_vmix  import implicit_vmix
from   pyOM3.pe_realistic.isoneutral     import isoneutral
from   pyOM3.pe_realistic.vadv_split     import vadv_split
from   pyOM3.density                  import get_rho
import numpy                          as np__
from   functools                      import partial


class model(thermodynamics,momentum,pressure,free_surface,damping,implicit_vmix,isoneutral,vadv_split):

      
      def set_coriolis(self,f0):
          """
          Set Coriolis frequency on T grid.
          Overload this method with different choices
          """
          self.coriolis_t = OM.modify_array(self.coriolis_t,(slice(None),slice(None)), 
                                            f0, out_sharding = self.sharding_2D )
          return

      
      def set_topography(self):
          """
          Set topography. kbot=k denotes bottom is below k-th model level.
          Overload this method with different choices.
          """
          self.kbot = OM.modify_array(self.kbot,(slice(None),slice(None)), self.halo , 
                                      out_sharding = self.sharding_2D  )           
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
          self.u = self.apply_bc(  self.pad(  OM.jax.device_put( _u , self.sharding_3D)  )  )*self.maskU
          self.v = self.apply_bc(  self.pad(  OM.jax.device_put( _v , self.sharding_3D)   )  )*self.maskV
          self.w    = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          self.temp = self.apply_bc( self.pad(  OM.jax.device_put( _temp , self.sharding_3D)  ) )#*self.maskT
          self.salt = self.apply_bc( self.pad(  OM.jax.device_put( _salt , self.sharding_3D)  ) )#*self.maskT
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if self.implicit_free_surface:
             p_s = self.pad(  OM.jax.device_put( _p_s , self.sharding_2D)  )  
             self.p_s   = OM.modify_array(self.p_s ,(0,slice(None),slice(None)), self.apply_bc(p_s)  )   
             self.p_s   = OM.modify_array(self.p_s ,(1,slice(None),slice(None)), self.p_s[0,:,:])
             self.p_s   = OM.modify_array(self.p_s ,(2,slice(None),slice(None)), self.p_s[0,:,:]) 
          if self.explicit_free_surface:   
             self.p_s = self.apply_bc( self.pad(  OM.jax.device_put( _p_s , self.sharding_2D)  )  )
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
          self.u    = self.apply_bc( OM.modify_array(self.u ,(b,b,b), _u[ex]) )*self.maskU
          self.v    = self.apply_bc( OM.modify_array(self.v ,(b,b,b), _v[ex]) )*self.maskV
          self.w    = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          self.temp = self.apply_bc( OM.modify_array(self.temp ,(b,b,b), _temp[ex]) ) #*self.maskT
          self.salt = self.apply_bc( OM.modify_array(self.salt ,(b,b,b), _salt[ex]) ) #*self.maskT         
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          ex = (slice(self.ys_pe-1,self.ye_pe),slice(self.xs_pe-1,self.xe_pe))          
          if self.implicit_free_surface:
             self.p_s   = OM.modify_array(self.p_s ,(0,b,b), _p_s[ex])   
             self.p_s   = OM.modify_array(self.p_s ,(0,a,a), self.apply_bc( self.p_s[0,:,:])  )  
             self.p_s   = OM.modify_array(self.p_s ,(1,a,a), self.p_s[0,:,:])
             self.p_s   = OM.modify_array(self.p_s ,(2,a,a), self.p_s[0,:,:])
          if self.explicit_free_surface:
             self.p_s   = self.apply_bc(  OM.modify_array(self.p_s ,(b,b), _p_s[ex])   )  
          return
    
    
      def prepare_for_run(self):          
          """
          Call everything which is needed for the main time stepping loop
          """
          self.check_vmix_stability()   # explicit-scheme coefficient guard
          self.vadv_check()             # vadv_sub_cycling (vadv_split.py), no-op when 0
          self.make_topo_masks()
          self.cf = self.make_coeff_surf_press(self.cf,self.hu,self.hv,self.maskT)
          if self.enable_neutral_diffusion: self.init_isoneutral()
          return


      def time_step_explicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the split-explicit method after Demange et al (2019).
          Chosen with self.explicit_free_surface = True
          """
          if getattr(self,'enable_tke',False): self.set_tke_diffusivities()
          self.du, self.dv,self.du_cor,self.dv_cor,self.p_h = \
                         self.momentum_tendency_expl(self.u,self.v,self.w, self.du,self.dv,
                                                     self.du_cor,self.dv_cor,self.rho, 
                                                     self.maskU,self.maskV,self.maskW,self.maskT,
                                                     self.coriolis_t,tau)
          
          # allow for user defined momentum forcing, added to du[tau],dv[tau]
          # set_forcing must precede solve_expl_surface (momentum forcing enters
          # through du), but tracer_advection below OVERWRITES dtemp/dsalt at
          # level tau, which lost the surface heat/salt fluxes on this path
          # (found by test/identities.py, Sep 2026). Keep the tracer forcing
          # increments and add them back after the advection.
          if hasattr(self,'set_forcing'):
             _dtemp0, _dsalt0 = self.dtemp*1, self.dsalt*1
             self.set_forcing(tau)
             _ftemp, _fsalt = self.dtemp - _dtemp0, self.dsalt - _dsalt0
          else:
             _ftemp = _fsalt = 0.0

          # bottom friction on the 3-D bottom cell BEFORE the barotropic solve:
          # solve_expl_surface trims the column to the sub-cycle mean, so the
          # trimmed column is then consistent with the transport that was
          # damped inside the sub cycle. Applied after the trimming it left
          # sum(u dz) != Um, and the re-trim of the next step turned that into
          # a whole-column kick every step (vertical two-level instability in
          # shallow coastal columns, POP explicit run Sep 15 2026).
          self.u, self.v = self.bottom_friction(self.u,self.v,self.maskU,self.maskV)   # no-op unless enabled
          if not hasattr(self,'U_bt'):       # cold start (or restart without it): from the 3-D field
             self.U_bt, self.V_bt = self.init_barotropic_state(self.u,self.v,self.maskU,self.maskV)
          self.u, self.v,  self.p_s, self.U_bt, self.V_bt = self.solve_expl_surface(self.u,self.v,
                                                              self.du,self.dv,self.du_cor,self.dv_cor,
                                                              self.p_h,self.p_s,self.U_bt,self.V_bt,
                                                              self.maskU,self.maskV,self.maskW,self.maskT,
                                                              self.hu,self.hv,self.coriolis_t,
                                                              A,B,C,tau,taum1,taum2)  
          
          # vertical friction BEFORE w and the tracers: u is final after it, so
          # the tracers are advected with the final velocity and w stays
          # consistent with u into the next step (found by the KE_ph = CONV
          # identity of test/pe/identities.py: w was stale by the friction's
          # redistribution at the end of an explicit step, 1.6e-8)
          self.vertical_friction()        # under TKE also diagnoses K_diss_v
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          
          # OPTION vadv_sub_cycling > 0 (vadv_split.py), see the implicit step
          _split = getattr(self,'vadv_sub_cycling',0) > 0
          if _split: self.dtemp = self.vadv_zero_tendency(self.dtemp,tau)
          else:
            self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                               self.maskU,self.maskV,self.maskW,self.maskT,tau)  

          if hasattr(self,'Khbi'):  
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)

             
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             if _split: self.dsalt = self.vadv_zero_tendency(self.dsalt,tau)
             else:
               self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau) 
             if hasattr(self,'Khbi'):  
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)   


          self.dtemp, self.dsalt = self.dtemp + _ftemp, self.dsalt + _fsalt
          if _split: self.vadv_update_tracers(A,B,C,tau,taum1,taum2)
          else:
            self.temp, self.salt = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                            self.temp,self.salt,self.dtemp,self.dsalt,
                                                                            self.p0,self.maskT)
          self.isoneutral_diffusion()     # pyOM2 order: after advection, before vmix

          # optional implicit vertical diffusion + convective adjustment (new level)
          self.vertical_mixing()          # option hierarchy (see implicit_vmix.py)
          # density ONCE per step, of the fully updated tracers, before the TKE
          # (pyOM2: calc_eq_of_state(taup1) after vmix, then P_diss_v)
          self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if getattr(self,'enable_tke',False): self.integrate_tke()
          return


          
      def time_step_implicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the rigid lid or implicit free surface method.
          """
          if getattr(self,'enable_tke',False): self.set_tke_diffusivities()
          
          # OPTION vadv_sub_cycling > 0 (vadv_split.py): the tracer advection
          # leaves the AB step and is done forward in time at the END of the
          # step with the final u, v, w, the vertical direction sub-cycled.
          # dtemp, dsalt then hold the forcing only.
          _split = getattr(self,'vadv_sub_cycling',0) > 0
          if _split: self.dtemp = self.vadv_zero_tendency(self.dtemp,tau)
          else:
            self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                               self.maskU,self.maskV,self.maskW,self.maskT,tau)  
              
          if hasattr(self,'Khbi'):  
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)
             
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             if _split: self.dsalt = self.vadv_zero_tendency(self.dsalt,tau)
             else:
               self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau) 
             # background advection only if temperature is buoyancy, then there is no salt 
             if hasattr(self,'Khbi'):  
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)
              
          self.du, self.dv, self.p_h = self.momentum_tendency_impl(self.u,self.v,self.w,
                                                                   self.du,self.dv,self.rho,self.p_h,
                                                                   self.maskU,self.maskV,self.maskW,self.maskT,
                                                                   self.coriolis_t,tau)               

          # allow for user defined momentum forcing, added to du[tau],dv[tau]
          if hasattr(self,'set_forcing'): self.set_forcing(tau)

          if hasattr(self,'Ahbi'):  
                       # pyOM2-form momentum friction (proper U/V-row metrics)
                       self.u, self.v = self.biharmonic_momentum(self.u,self.v,
                                                self.Ahbi,self.maskU,self.maskV)          
          if hasattr(self,'Ah'):
                       # pyOM2-form harmonic momentum friction
                       self.u, self.v = self.harmonic_momentum(self.u,self.v,
                                                self.Ah,self.maskU,self.maskV)
             
          self.u, self.v = self.update_momentum_AB3(A,B,C,tau,taum1,taum2,
                                                    self.u,self.v,self.du,self.dv,self.maskU,self.maskV)

          # user-defined DIRECT momentum increment, bypassing the AB weights:
          # e.g. u_L += Delta u_S of the Lagrangian-mean (u_L) formulation,
          # which must be a plain forward increment (AB weighting would break
          # the exact telescoping u = u_L - u_S) and must PRECEDE the pressure
          # solve so the projection acts on its divergence.
          if hasattr(self,'set_forcing_direct'): self.set_forcing_direct()

          self.u, self.v, self.p_s, self.estimated_error, self.congr_itts  = \
                             self.solve_for_pressure(self.u,self.v,self.p_s,
                                                     self.maskU,self.maskV,self.maskT,self.cf,tau,taum1,taum2)
          # vertical friction (option hierarchy: explicit / implicit / TKE);
          # column-integrated momentum is unchanged (zero-flux BCs), so the
          # barotropic mode of the pressure solve is not perturbed; the
          # baroclinic w is re-diagnosed below.
          self.vertical_friction()
          self.u, self.v = self.bottom_friction(self.u,self.v,self.maskU,self.maskV)   # no-op unless enabled
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
                         
          if _split: self.vadv_update_tracers(A,B,C,tau,taum1,taum2)
          else:
            self.temp, self.salt = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                           self.temp,self.salt,self.dtemp,self.dsalt,
                                                                           self.p0,self.maskT)
          self.isoneutral_diffusion()     # pyOM2 order: after advection, before vmix

          # optional implicit vertical diffusion + convective adjustment (new level)
          self.vertical_mixing()          # option hierarchy (see implicit_vmix.py)
          # density ONCE per step, of the fully updated tracers, before the TKE
          # (pyOM2: calc_eq_of_state(taup1) after vmix, then P_diss_v)
          self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if getattr(self,'enable_tke',False): self.integrate_tke()
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
     