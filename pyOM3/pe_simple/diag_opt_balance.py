"""
 Optimal balance. 
 Only works with buoyancy as tracer and pressure method for external mode.

 Based on class model.
"""
import pyOM3                     as OM
from   pyOM3.pe_simple.model     import model
import numpy                     as np__
from   functools                 import partial


class opt_balance(model):

    
      def balance_it(self,OB_period,OB_average,OB_average_times,OB_max_itts=10,OB_tol=1e-5,
                     exchange_base_point = True):
          """
          Entry function for Optimal Balance diagnostics. 
          Only works with buoyancy as tracer and pressure method for external mode
          and with background stratification (parameter N0 in initialisations).

          Uses the model state as input and the following parameter:
          
          OB_period           : integration time for ramping from linear to non-linear end
          OB_average          : integration time for averaging to find linear mode
          OB_average_times    : number of chunks for averaging
          OB_max_itts         : (optional, default=10) maximal iterations if norm of changes stays above OB_tol
          exchange_base_point : (optional, default=True) save base point at start and exchange at every iteration
          
          Output is the balanced model state.
          """

          if self.my_pe == 0 : print('entering Optimal Balance diagnostics')
          for attr in ('OB_period','OB_average','OB_average_times','OB_max_itts','exchange_base_point'): 
              if self.my_pe == 0 : print(attr,locals().get(attr)  )
              setattr(self,attr,locals().get(attr)) 
              
          if self.explicit_free_surface:
              print('Error: ob_balance only works with pressure method for external mode.')
              raise ValueError

          if hasattr(self,'N0'): 
             print('Error: ob_balance does not work with background N0.')
             raise ValueError 
              
          # Calculate number of iteration steps for turning off/on the non linear terms and averaging
          n_end_ramp = int(OB_period / self.dt)  + 1
          n_end_ave  = int(OB_average / self.dt) + 1

          # remove horizontal mean of temperature
          self.temp, self.temp_back = self.OB_remove_mean_temp(self.temp,self.maskT)
          
          # store model state
          state = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )

          if exchange_base_point:    
            # save base point by time averaging in linear model
            print(' to save base point, integrating linear model for ',
                  n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')
         
            for m in range(OB_average_times):              
                print(' chunk ',m)
                mean = self.OB_time_ave(n_end_ave)              
                self.u, self.v, self.w, self.rho, self.temp, p = mean
                self.p_s = self.move_p_s(self.p_s,p)              

            # store the base point and restore model state
            base = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )
            self.u, self.v, self.w, self.rho, self.temp, p = state*1
            self.p_s = self.move_p_s(self.p_s,p)
          
          norm_diff = 1.
                            
          # Start optimal balance iteration
          for n in range(OB_max_itts):
 
              print(' starting optimal balance iteration:',n)

              # integrate backward to the linear end
              print(' integrating backward to linear end for ',n_end_ramp,' timesteps ')
              self.OB_backward_integration(n_end_ramp)
    
              # project on geostrophic mode at linear end
              print(' integrating linear model for ',n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')             
              for m in range(OB_average_times):
                  print(' chunk ',m)
                  mean = self.OB_time_ave(n_end_ave)
                  self.u, self.v, self.w, self.rho, self.temp, p = mean
                  self.p_s = self.move_p_s(self.p_s,p)                  
                 
              # integrate forward to the non linear end
              print(' integrating forward to non-linear end for ',n_end_ramp,' timesteps ')
              self.OB_forward_integration(n_end_ramp)

              if exchange_base_point: 
                # here we need to save the model state
                save = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )
              
                # apply boundary condition at non linear end
                print(' integrating linear model for ',n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')
                for m in range(OB_average_times):
                    print(' chunk ',m)
                    mean = self.OB_time_ave(n_end_ave)
                    self.u, self.v, self.w, self.rho, self.temp, p = mean
                    self.p_s = self.move_p_s(self.p_s,p)                  

                # exchange base point
                print('exchanging base point')
                u_save, v_save, w_save, rho_save, temp_save, p_save = save
                u_base, v_base, w_base, rho_base, temp_base, p_base = base
                self.u    = u_save    - self.u    + u_base
                self.v    = v_save    - self.v    + v_base
                self.w    = w_save    - self.w    + w_base  
                self.rho  = rho_save  - self.rho  + rho_base    
                self.temp = temp_save - self.temp + temp_base
                p         = p_save    - self.p_s[0,:,:] + p_base
                self.p_s = self.move_p_s(self.p_s,p)
              
              norm_diff = self.OB_norm(state[0],state[1],self.u,self.v)

              # update reference state
              state = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )
    
              # Check tolerance criterion
              if n > 0:
                 print(' norm of difference to n= ',n - 1,' is ', norm_diff,' / ',OB_tol)
                 if norm_diff < OB_tol: break 
                     
          # add horizontal mean to temperature
          self.temp = self.OB_add_mean_temp(self.temp,self.temp_back)
           
          return

          
      def move_p_s(self,p_s,p):
          """
          Copy p to the three time levels of p_s
          """
          p_s = OM.modify_array(p_s,(0,slice(None),slice(None)), p)
          p_s = OM.modify_array(p_s,(1,slice(None),slice(None)), p)
          p_s = OM.modify_array(p_s,(2,slice(None),slice(None)), p)
          return p_s


      def OB_backward_integration(self,_steps):
          """
          Backwards integration over _steps time steps with negative time step.
          Ramp OB_rho from 1 to 0. 
          """
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          self.dt = -OM.np.abs(self.dt)
          for n in range(_steps): # time step equations 
              OB_rho = 1. - ramp(n * self.dt /self.OB_period)
              self.OB_time_step(OB_rho,A,B,C,tau,taum1,taum2,0.,use_2nd_advection = True)  
              tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.          
          return

    
      def OB_forward_integration(self,_steps,ramp_reversed = False):
          """
          Forwards integration over _steps time steps with positive time step.
          Ramp OB_rho from 0 to 1. Use viscosity.
          """          
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          self.dt = +OM.np.abs(self.dt)
          for n in range(_steps): # time step equations                
              OB_rho = ramp(n * self.dt / self.OB_period) 
              if ramp_reversed: OB_rho = 1 - OB_rho
              self.OB_time_step(OB_rho,A,B,C,tau,taum1,taum2,self.Ahbi,use_2nd_advection = False)    
              tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.          
          return

        
      def OB_time_ave(self,_steps): 
          """
          Time averaging over _steps time steps with positive time step. 
          Use viscosity.
          """
          @OM.jaxjit  
          def _add(u,v,w,rho,temp,p_s,u_m,v_m,w_m,rho_m,temp_m,p_sm): 
              return u_m+u, v_m+v, w_m+w, rho_m+rho,temp_m+temp, p_s+p_sm

          @OM.jaxjit  
          def _divide(u,v,w,rho,temp,p_s,N):    
              return u/N, v/N, w/N, rho/N, temp/N, p_s/N    
          
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels 
          self.dt = +OM.np.abs(self.dt)
          u_m,v_m,w_m = OM.np.zeros_like(self.u),OM.np.zeros_like(self.v),OM.np.zeros_like(self.w)
          rho_m,temp_m, p_sm = OM.np.zeros_like(self.rho), OM.np.zeros_like(self.temp), OM.np.zeros_like(self.p_s[0,:,:])          
          for n in range(_steps):
              self.OB_time_step(0.,A,B,C,tau,taum1,taum2,self.Ahbi,use_2nd_advection = False)  
              u_m, v_m, w_m, rho_m, temp_m, p_sm = _add(self.u,self.v,self.w,
                                                        self.rho,self.temp,self.p_s[tau,:,:],
                                                        u_m,v_m,w_m,rho_m,temp_m,p_sm)
              tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.
          return _divide(u_m,v_m,w_m,rho_m,temp_m,p_sm,_steps) 
    
          
      def OB_time_step(self,OB_rho,A,B,C,tau,taum1,taum2,Ahbi_loc,use_2nd_advection = False):
          """
          Perform one time step given OB_rho scaling non-linear terms and viscosity Ahbi_loc. 
          """
          if hasattr(self,'superbee_advection') and not use_2nd_advection:
             self.dtemp = self.OB_tracer_linear_superbee(self.temp_back,self.dtemp,self.u,self.v,self.w,
                                             self.maskU,self.maskV,self.maskW,self.maskT,tau)
          else:
             self.dtemp = self.OB_tracer_linear_2nd(self.temp_back,self.dtemp,self.u,self.v,self.w,
                                             self.maskU,self.maskV,self.maskW,self.maskT,tau)
              
          if OB_rho>0:
            if hasattr(self,'superbee_advection') and not use_2nd_advection:  
              self.dtemp = self.OB_tracer_advection_superbee(self.temp,self.dtemp,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau,OB_rho)  
            else:    
              self.dtemp = self.OB_tracer_advection_2nd(self.temp,self.dtemp,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau,OB_rho)  
                    
          self.du, self.dv, self.p_h = self.OB_momentum_linear(self.u,self.v,self.du,self.dv,
                                                               self.maskU,self.maskV,self.maskT,
                                                               self.rho,self.coriolis_t,tau)
          if OB_rho>0:    
            self.du, self.dv = self.OB_momentum_advection(self.u,self.v,self.w,self.du,self.dv,
                                                          self.maskU,self.maskV,self.maskW,tau,OB_rho)
                       
          self.u, self.v = self.update_momentum_AB3(A,B,C,tau,taum1,taum2,
                                                    self.u,self.v,self.du,self.dv,self.maskU,self.maskV)

          if Ahbi_loc>0:  
            self.u = self.biharmonic_horizontal(self.u,Ahbi_loc,self.maskU)
            self.v = self.biharmonic_horizontal(self.v,Ahbi_loc,self.maskV)          
          
          self.u, self.v, self.p_s, \
               self.estimated_error, self.congr_itts  =  self.solve_for_pressure(self.u,self.v,self.p_s,
                                                                                 self.maskU,self.maskV,self.maskT,self.cf,
                                                                                 tau,taum1,taum2)
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
                         
          self.temp, self.salt, self.rho = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                          self.temp,self.salt,self.dtemp,self.dsalt,
                                                                          self.p0, self.maskT)
          return

    
      @partial(OM.jaxjit, static_argnums=0)    
      def OB_remove_mean_temp(self,temp,maskT):
          _t = OM.np.sum( OM.np.sum(temp*maskT,axis=-1),axis=-1) 
          _m = OM.np.sum( OM.np.sum(maskT,axis=-1),axis=-1) 
          _t = OM.np.where( _m>0, _t/(1e-32+_m), 0)
          temp_back = OM.np.zeros_like(temp) + _t[:,None,None]
          temp -= temp_back
          return temp,temp_back

    
      @partial(OM.jaxjit, static_argnums=0)    
      def OB_add_mean_temp(self,temp,temp_back):
          return temp+temp_back

    
      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_linear_2nd(self,Var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau):  
          """
          Tendency due to background advection of tracer.
          """
          fe,fn,ft = self.tracer_flux_2nd(Var,u,v,w,maskU,maskV,maskW)
          _dvar = (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                   + (ft-self.rollz(ft,1))/self.dz )*maskT 
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), _dvar, out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_linear_superbee(self,Var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau):  
          """
          Tendency due to background advection of tracer.
          """
          fe,fn,ft = self.tracer_flux_superbee(Var,u,v,w,maskU,maskV,maskW)
          _dvar = (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                   + (ft-self.rollz(ft,1))/self.dz )*maskT 
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), _dvar, out_sharding = self.sharding_4D)
    
      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_advection_2nd(self,var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau,ob_rho):  
          """
          Tendency due to perturbation advection of tracer.
          """              
          fe,fn,ft = self.tracer_flux_2nd(var,u,v,w,maskU,maskV,maskW)                 
          _dvar = (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                   + (ft-self.rollz(ft,1))/self.dz )*maskT 
          return OM.add_to_array(dvar, (tau,slice(None),slice(None),slice(None)), ob_rho*_dvar, out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_advection_superbee(self,var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau,ob_rho):  
          """
          Tendency due to perturbation advection of tracer.
          """              
          fe,fn,ft = self.tracer_flux_superbee(var,u,v,w,maskU,maskV,maskW)                 
          _dvar = (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                   + (ft-self.rollz(ft,1))/self.dz )*maskT 
          return OM.add_to_array(dvar, (tau,slice(None),slice(None),slice(None)), ob_rho*_dvar, out_sharding = self.sharding_4D)
    

      @partial(OM.jaxjit, static_argnums=0)
      def OB_momentum_advection(self,u,v,w,du,dv,maskU,maskV,maskW,tau,ob_rho):  
          """
          Tendency due to perturbation advection of momentum.
          """
          fe,fn,ft  = self.mom_flux_2nd_u(u,u,v,w,maskU,maskV,maskW)
          _du = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskU
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          _dv = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskV         
          du = OM.add_to_array(du, (tau,slice(None),slice(None),slice(None)), ob_rho*_du, out_sharding = self.sharding_4D)
          dv = OM.add_to_array(dv, (tau,slice(None),slice(None),slice(None)), ob_rho*_dv, out_sharding = self.sharding_4D)          
          return du,dv

    
      @partial(OM.jaxjit, static_argnums=0)
      def OB_momentum_linear(self,u,v,du,dv,maskU,maskV,maskT,rho,coriolis_t,tau):   
          """
          Tendency due to linear terms in momentum equation.
          """
          _du = + maskU*(           coriolis_t    [None,:,:] *           ( v + self.rolly(v,1) ) +
                         self.rollx(coriolis_t,-1)[None,:,:] * self.rollx( v + self.rolly(v,1) ,-1) )*0.25
            
          _dv = - maskV*(           coriolis_t    [None,:,:] *           ( u + self.rollx(u,1) ) +
                         self.rolly(coriolis_t,-1)[None,:,:] * self.rolly( u + self.rollx(u,1) ,-1) )*0.25  
          
          p_h = OM.np.zeros_like(rho)
          p_h = self.hydrostatic_pressure(rho,p_h,maskT)          
          _du +=  - (self.rollx(p_h,-1) - p_h)/self.dx*maskU
          _dv +=  - (self.rolly(p_h,-1) - p_h)/self.dy*maskV  
         
          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)          
          return du,dv,p_h


      @partial(OM.jaxjit, static_argnums=0)    
      def OB_norm(self,u1,v1,u2,v2):
          """
          Calculate a norm of differences in u1,v1 and u2,v2
          """
          a  = u1**2+v1**2
          e1 = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a) 
          a  = u2**2+v2**2
          e2 = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a)
          a  = (u1-u2)**2+(v1-v2)**2
          d  = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a)          
          return 2 * OM.np.sqrt(d) / (OM.np.sqrt(e1) + OM.np.sqrt(e2)) 



@OM.jaxjit
def ramp(theta):     
    """ 
    The ramp fucntion:  ramp = (1-cos(pi*theta))*0.5
    """   
    t1 = OM.np.where(theta   > 1e-32, 1./(1e-32+theta), 1e32)
    t2 = OM.np.where(1-theta > 1e-32, 1./(1e-32+1-theta), 1e32)
    return OM.np.exp(-t1)/(OM.np.exp(-t1) + OM.np.exp(-t2) )  



