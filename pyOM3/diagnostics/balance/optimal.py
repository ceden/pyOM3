"""
Optimal balance: optimal_pe_simple for pe_simple and optimal_pe_realistic
for pe_realistic (add the class to the model class bases). Only works with
buoyancy as tracer and pressure method for external mode.

The class _optimal holds the method (balance_it and the integrations), the
two family classes the parts that depend on the grid: the horizontal mean
of the buoyancy, the divergence of the tracer fluxes, the linear and the
advection terms of the momentum equation, the momentum friction, the norm,
and the projections at the linear end (kernel and fft: pe_simple only).

Changed with the move from pe_simple/diag_opt_balance.py and
pe_realistic/diag_opt_balance.py (Nov 2026): pe_realistic now uses the
method of pe_simple. Its backward integration ran with a negative time step:
the ramp of the non-linear terms stayed 0 there (OB_rho = 1 throughout, the
non-linear terms were never ramped down), and with JAX the integration ran
forward. Its time averaging used the friction, which damps the geostrophic
mode too (see OB_time_ave). The option remove_mean_u of pe_realistic is
removed, as before in pe_simple. pe_realistic gets the options
OB_friction and OB_window of pe_simple.
"""
import pyOM3                     as OM
from   pyOM3.pe_simple.model     import model as _pe_simple
from   pyOM3.pe_realistic.model  import model as _pe_realistic
from   pyOM3.density             import get_rho
from   pyOM3.diagnostics.balance.kernel import kernel_pe_simple
import numpy                     as np_
from   functools                 import partial


class _optimal:
      """
      the method, independent of the grid: balance_it, the backward and
      forward integrations, the time averaging and the time step
      """
      def balance_it(self,OB_period,OB_average,OB_average_times,OB_max_itts=10,OB_tol=1e-5,
                     exchange_base_point = True, OB_projection = 'average', OB_friction = True,
                     OB_window = 'box'):
          """
          Entry function for Optimal Balance diagnostics. 
          Only works with buoyancy as tracer and pressure method for external mode.

          Uses the model state as input and the following parameter:
          
          OB_period           : integration time for ramping from linear to non-linear end
          OB_average          : integration time for averaging to find linear mode
          OB_average_times    : number of chunks for averaging
          OB_max_itts         : (optional, default=10) maximal iterations if norm of changes stays above OB_tol
          exchange_base_point : (optional, default=True) save base point at start and exchange at every iteration
          OB_projection       : (optional, default='average') projection on the geostrophic mode at the
                                linear end. 'average' is the time averaging of the linear model over
                                OB_average with OB_average_times chunks. 'kernel' is the truncated kernel
                                inversion of the potential vorticity of kernel.py, which needs no time
                                stepping. 'fft' is the exact spectral projection on the geostrophic mode
                                of warn.py. Both only for pe_simple: the class must also inherit from
                                kernel_pe_simple of kernel.py, and the vertical modes are built from the
                                horizontal mean stratification of the model state (rigid lid, constant
                                f, one process). Both drop the domain mean velocity of each level, an
                                inertial oscillation without potential vorticity.
          OB_friction         : (optional, default=True) biharmonic friction and diffusion of the model
                                (Ahbi, Khbi, kappa_T) in the forward integration to the non-linear end.
                                The backward integration and the time averaging never use them. With
                                False the forward integration is frictionless as well, so that it is the
                                exact reverse of the backward integration.
          OB_window           : (optional, default='box') weight of the time average in each chunk of
                                OB_projection='average'. 'box' is the plain mean. 'hann' is the weight
                                sin^2(pi t/OB_average), which suppresses frequencies away from zero much
                                more strongly than the plain mean of the same length, but needs twice
                                the length to remove the inertial frequency f (OB_average >= 2 inertial
                                periods).
          
          Output is the balanced model state.
          """

          if self.my_pe == 0 : print('entering Optimal Balance diagnostics')
          for attr in ('OB_period','OB_average','OB_average_times','OB_max_itts','exchange_base_point','OB_projection',
                       'OB_friction','OB_window'): 
              if self.my_pe == 0 : print(attr,locals().get(attr)  )
              setattr(self,attr,locals().get(attr)) 
              
          if OB_window not in ('box','hann'):
              print('Error: OB_window must be box or hann.')
              raise ValueError

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

          # the family sets up the projection (kernel and fft: pe_simple only)
          self._OB_init_projection(OB_projection)
          last_ramp_tau = (n_end_ramp + 1) % 3

          def OB_project(last_tau):
             # projection on the geostrophic mode, time averaging or kernel inversion
             if OB_projection in ('kernel','fft'):
                if self.my_pe == 0 : print(' projecting with the %s method' % OB_projection)
                _project = self.OB_kernel_project if OB_projection == 'kernel' else self.OB_fft_project
                self.u, self.v, self.w, self.temp, p = _project(last_tau)
                self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
                self.p_s = self.move_p_s(self.p_s,p)
             else:
                for m in range(self.OB_average_times):
                    if self.my_pe == 0 : print(' chunk ',m)
                    mean = self.OB_time_ave(n_end_ave)
                    self.u, self.v, self.w, self.rho, self.temp, p = mean
                    self.p_s = self.move_p_s(self.p_s,p)
          
          # store model state
          state = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )

          if exchange_base_point:    
            # save base point by time averaging in linear model
            if self.my_pe == 0 : print(' to save base point, integrating linear model for ',
                  n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')
         
            OB_project(0)

            # store the base point and restore model state
            base = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )
            self.u, self.v, self.w, self.rho, self.temp, p = state*1
            self.p_s = self.move_p_s(self.p_s,p)
          
          norm_diff = 1.
                            
          # Start optimal balance iteration
          for n in range(OB_max_itts):
 
              if self.my_pe == 0 : print(' starting optimal balance iteration:',n)

              # integrate backward to the linear end
              if self.my_pe == 0 : print(' integrating backward to linear end for ',n_end_ramp,' timesteps ')
              self.OB_backward_integration(n_end_ramp)
    
              # project on geostrophic mode at linear end
              if self.my_pe == 0 : print(' integrating linear model for ',
                                         n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')             
              OB_project(last_ramp_tau)

              # integrate forward to the non linear end
              if self.my_pe == 0 : print(' integrating forward to non-linear end for ',n_end_ramp,' timesteps ')
              self.OB_forward_integration(n_end_ramp)

              if exchange_base_point: 
                # here we need to save the model state
                save = (self.u*1,self.v*1,self.w*1,self.rho*1,self.temp*1,self.p_s[0,:,:]*1 )
              
                # apply boundary condition at non linear end
                if self.my_pe == 0 : print(' integrating linear model for ',
                                           n_end_ave*OB_average_times,' timesteps in ',OB_average_times,' chunks')
                OB_project(last_ramp_tau)

                # exchange base point
                if self.my_pe == 0 : print('exchanging base point')
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
                 if self.my_pe == 0 : print(' norm of difference to n= ',n - 1,' is ', norm_diff,' / ',OB_tol)
                 if norm_diff < OB_tol: break 
                     
          # add horizontal mean to temperature
          self.temp = self.OB_add_mean_temp(self.temp,self.temp_back)
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          self.p_h = self.hydrostatic_pressure(self.rho,self.p_h,self.maskT)  
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
          Backwards integration over _steps time steps. Ramp OB_rho from 1 to 0.
          The time step stays positive and the time-reversed equations are
          integrated: all tendencies change sign (OB_time_step with backward=True).
          A negative self.dt does not work with JAX: the jitted methods take
          self as static argument and keep the dt of their first trace, so the
          integration ran forward in time. The negative dt also made the
          argument of the ramp negative, the ramp stayed 0 and OB_rho stayed 1,
          so the non-linear terms were never ramped down.
          With the implicit free surface the reversed equations are again of
          the model form for -p_s, so the sign of p_s is changed at the start
          and changed back at the end (for the rigid lid this does nothing).
          """
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          self.dt = OM.np.abs(self.dt)
          self.p_s = -self.p_s
          for n in range(_steps): # time step equations 
              OB_rho = 1. - ramp(n * self.dt /self.OB_period)
              self.OB_time_step(OB_rho,A,B,C,tau,taum1,taum2,0.,0.,0.,use_2nd_advection = True, backward = True)
              tau, taum1, taum2   = np_.mod(tau+1,3), np_.mod(taum1+1,3), np_.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.          
          self.p_s = -self.p_s
          return

      def OB_forward_integration(self,_steps,ramp_reversed = False):
          """
          Forwards integration over _steps time steps with positive time step.
          Ramp OB_rho from 0 to 1. Use viscosity unless OB_friction = False.
          """          
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          self.dt = +OM.np.abs(self.dt)
          
          Ahbi_loc = 0 if not hasattr(self,'Ahbi') else self.Ahbi
          Khbi_loc = 0 if not hasattr(self,'Khbi') else self.Khbi
          kappa_T_loc = 0 if not hasattr(self,'kappa_T') else self.kappa_T
          if not getattr(self,'OB_friction',True):
             Ahbi_loc, Khbi_loc, kappa_T_loc = 0., 0., 0.
          
          for n in range(_steps): # time step equations                
              OB_rho = ramp(n * self.dt / self.OB_period) 
              if ramp_reversed: OB_rho = 1 - OB_rho
              self.OB_time_step(OB_rho,A,B,C,tau,taum1,taum2,Ahbi_loc,Khbi_loc,kappa_T_loc,use_2nd_advection = False)    
              tau, taum1, taum2   = np_.mod(tau+1,3), np_.mod(taum1+1,3), np_.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.          
          return

      def OB_time_ave(self,_steps): 
          """
          Time averaging over _steps time steps with positive time step,
          with the weight OB_window: 'box' the plain mean, 'hann' the weight
          sin^2(pi (n + 1/2)/_steps), normalised to sum one.
          No friction or diffusion: the time average of the linear model must
          be a projection on the geostrophic mode (applied twice it gives the
          same). With the biharmonic friction it also damps the geostrophic
          mode, then the exchange of the base point does not keep the base
          point and the balanced state differs from the Warn et al. state by
          more than the geostrophic state does (jets_balance1 test, Nov 2026).
          """
          @OM.jaxjit  
          def _add(u,v,w,rho,temp,p_s,u_m,v_m,w_m,rho_m,temp_m,p_sm,wgt): 
              return u_m+wgt*u, v_m+wgt*v, w_m+wgt*w, rho_m+wgt*rho,temp_m+wgt*temp, p_sm+wgt*p_s

          if getattr(self,'OB_window','box') == 'hann':
             wgt = np_.sin(np_.pi*(np_.arange(_steps) + 0.5)/_steps)**2
          else:
             wgt = np_.ones(_steps)
          wgt = wgt/wgt.sum()
          
          A,B,C = 1.,0.,0.           # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels 
          self.dt = +OM.np.abs(self.dt)
          Ahbi_loc, Khbi_loc, kappa_T_loc = 0., 0., 0.
          
          u_m,v_m,w_m = OM.np.zeros_like(self.u),OM.np.zeros_like(self.v),OM.np.zeros_like(self.w)
          rho_m,temp_m, p_sm = OM.np.zeros_like(self.rho), OM.np.zeros_like(self.temp), OM.np.zeros_like(self.p_s[0,:,:])          
          for n in range(_steps):
              self.OB_time_step(0.,A,B,C,tau,taum1,taum2,Ahbi_loc,Khbi_loc,kappa_T_loc,use_2nd_advection = False)  
              u_m, v_m, w_m, rho_m, temp_m, p_sm = _add(self.u,self.v,self.w,
                                                        self.rho,self.temp,self.p_s[tau,:,:],
                                                        u_m,v_m,w_m,rho_m,temp_m,p_sm,float(wgt[n]))
              tau, taum1, taum2   = np_.mod(tau+1,3), np_.mod(taum1+1,3), np_.mod(taum2+1,3)          
              if A == 1. and B == 0. and C == 0:    A, B, C = 1.5,  -0.5, 0. 
              elif A==1.5 and B == -0.5 and C == 0: A, B, C =  23./12. , -16./12., 5./12.
          return u_m, v_m, w_m, rho_m, temp_m, p_sm

      def OB_time_step(self,OB_rho,A,B,C,tau,taum1,taum2,Ahbi_loc,Khbi_loc,kappa_T_loc,use_2nd_advection = False,
                       backward = False):
          """
          Perform one time step given OB_rho scaling non-linear terms and viscosity Ahbi_loc. 
          backward = True: the time-reversed equations, all tendencies change sign.
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

          if Khbi_loc>0: 
             self.temp = self.biharmonic_horizontal(self.temp,Khbi_loc,self.maskT)
          if kappa_T_loc>0: 
             self.temp = self.harmonic_vertical_expl(self.temp,kappa_T_loc,self.maskT) ## problem wih temp_back here

          
          self.du, self.dv, self.p_h = self.OB_momentum_linear(self.u,self.v,self.du,self.dv,
                                                               self.maskU,self.maskV,self.maskT,
                                                               self.rho,self.coriolis_t,tau)
          if OB_rho>0:    
            self.du, self.dv = self.OB_momentum_advection(self.u,self.v,self.w,self.du,self.dv,
                                                          self.maskU,self.maskV,self.maskW,tau,OB_rho)

          if backward:
            self.du, self.dv, self.dtemp = self.OB_reverse_tendencies(self.du,self.dv,self.dtemp,tau)
                       
          self.u, self.v = self.update_momentum_AB3(A,B,C,tau,taum1,taum2,
                                                    self.u,self.v,self.du,self.dv,self.maskU,self.maskV)

          if Ahbi_loc>0:  
            self.u, self.v = self._OB_friction(self.u,self.v,Ahbi_loc)
          
          self.u, self.v, self.p_s, \
               self.estimated_error, self.congr_itts  =  self.solve_for_pressure(self.u,self.v,self.p_s,
                                                                                 self.maskU,self.maskV,self.maskT,self.cf,
                                                                                 tau,taum1,taum2)
          self.w = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
                         
          self.temp, self.salt = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                self.temp,self.salt,self.dtemp,self.dsalt,
                                                                self.p0, self.maskT)
          self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          return

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_reverse_tendencies(self,du,dv,dtemp,tau):
          """
          Change the sign of the tendencies at time level tau (time-reversed equations).
          """
          s = (tau,slice(None),slice(None),slice(None))
          du    = OM.modify_array(du,    s, -du[tau],    out_sharding = self.sharding_4D)
          dv    = OM.modify_array(dv,    s, -dv[tau],    out_sharding = self.sharding_4D)
          dtemp = OM.modify_array(dtemp, s, -dtemp[tau], out_sharding = self.sharding_4D)
          return du,dv,dtemp

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_add_mean_temp(self,temp,temp_back):
          return temp+temp_back

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_linear_2nd(self,Var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau):  
          """
          Tendency due to background advection of tracer.
          """
          fe,fn,ft = self.tracer_flux_2nd(Var,u,v,w,maskU,maskV,maskW)
          _dvar = self._OB_div(fe,fn,ft)*maskT
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), _dvar, out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_linear_superbee(self,Var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau):  
          """
          Tendency due to background advection of tracer.
          """
          fe,fn,ft = self.tracer_flux_superbee(Var,u,v,w,maskU,maskV,maskW)
          _dvar = self._OB_div(fe,fn,ft)*maskT
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), _dvar, out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_advection_2nd(self,var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau,ob_rho):  
          """
          Tendency due to perturbation advection of tracer.
          """              
          fe,fn,ft = self.tracer_flux_2nd(var,u,v,w,maskU,maskV,maskW)                 
          _dvar = self._OB_div(fe,fn,ft)*maskT
          return OM.add_to_array(dvar, (tau,slice(None),slice(None),slice(None)), ob_rho*_dvar, out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_tracer_advection_superbee(self,var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau,ob_rho):  
          """
          Tendency due to perturbation advection of tracer.
          """              
          fe,fn,ft = self.tracer_flux_superbee(var,u,v,w,maskU,maskV,maskW)                 
          _dvar = self._OB_div(fe,fn,ft)*maskT
          return OM.add_to_array(dvar, (tau,slice(None),slice(None),slice(None)), ob_rho*_dvar, out_sharding = self.sharding_4D)


class optimal_pe_simple(_optimal, _pe_simple):
      """optimal balance for pe_simple, uniform Cartesian grid"""

      def _OB_init_projection(self,OB_projection):
          """
          OB_projection 'kernel' and 'fft': the vertical modes from the mean
          stratification and the kernels of kernel_pe_simple
          """
          if OB_projection not in ('kernel','fft'): return
          if not isinstance(self, kernel_pe_simple):
             print('Error: OB_projection %s needs the class kernel_pe_simple of kernel.py.' % OB_projection)
             raise ValueError
          h = self.halo
          Tb = np_.asarray(self.temp_back)[h:-h, h, h]
          N2w = np_.zeros(self.nz)
          N2w[:-1] = (Tb[1:] - Tb[:-1])/self.dz
          N2w[-1] = N2w[-2]
          if N2w.min() <= 0:
             print('Error: OB_projection kernel needs a stable mean stratification.')
             raise ValueError
          if N2w.max() - N2w.min() < 1e-6*N2w.max():
             self.init_balance(float(np_.sqrt(N2w.mean())), verbose = self.my_pe == 0)
          else:
             self.init_balance(N2w, verbose = self.my_pe == 0)
          if OB_projection == 'kernel':
             self.init_kernel(verbose = self.my_pe == 0)
          return

      def _OB_div(self,fe,fn,ft):
          """divergence of the fluxes fe, fn, ft on the uniform grid"""
          return (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                  + (ft-self.rollz(ft,1))/self.dz )

      def _OB_friction(self,u,v,Ahbi_loc):
          """biharmonic friction of u and v"""
          u = self.biharmonic_horizontal(u,Ahbi_loc,self.maskU)
          v = self.biharmonic_horizontal(v,Ahbi_loc,self.maskV)          
          return u,v

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_remove_mean_temp(self,temp,maskT):
          _t = OM.np.sum( OM.np.sum(temp*maskT,axis=-1),axis=-1) 
          _m = OM.np.sum( OM.np.sum(maskT,axis=-1),axis=-1) 
          _t = OM.np.where( _m>0, _t/(1e-32+_m), 0)
          temp_back = OM.np.zeros_like(temp) + _t[:,None,None]
          temp -= temp_back
          return temp,temp_back

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


class optimal_pe_realistic(_optimal, _pe_realistic):
      """optimal balance for pe_realistic, pseudo-spherical grid"""

      def _OB_init_projection(self,OB_projection):
          """only OB_projection = 'average' (kernel and fft: pe_simple only)"""
          if OB_projection != 'average':
             print('Error: OB_projection %s only for pe_simple, use average.' % OB_projection)
             raise ValueError
          return

      def _OB_div(self,fe,fn,ft):
          """divergence of the fluxes fe, fn, ft on the pseudo-spherical grid"""
          return (  (fe-self.rollx(fe,1))/(self.cost_y*self.dxt_x) 
                  + (fn-self.rolly(fn,1))/(self.cost_y*self.dyt_y) 
                  + (ft-self.rollz(ft,1))/self.dzt_z )

      def _OB_friction(self,u,v,Ahbi_loc):
          """
          the same metric-aware momentum friction as the model's time step
          (pyOM2 placement of the U/V-row metrics), not the generic tracer
          operator applied to u and v
          """
          return self.biharmonic_momentum(u,v,Ahbi_loc,self.maskU,self.maskV)

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_remove_mean_temp(self,temp,maskT):
          # horizontal mean per level, weighted with the T cell area (the
          # unweighted count is only right on a uniform Cartesian grid)
          w  = maskT*self.area_t
          _t = OM.np.sum( OM.np.sum(temp*w,axis=-1),axis=-1) 
          _m = OM.np.sum( OM.np.sum(w,axis=-1),axis=-1) 
          _t = OM.np.where( _m>0, _t/(1e-32+_m), 0)
          temp_back = OM.np.zeros_like(temp) + _t[:,None,None]
          temp -= temp_back
          return temp,temp_back

      @partial(OM.jaxjit, static_argnums=0)
      def OB_momentum_advection(self,u,v,w,du,dv,maskU,maskV,maskW,tau,ob_rho):  
          """
          Tendency due to perturbation advection of momentum.
          """
          fe,fn,ft  = self.mom_flux_2nd_u(u,u,v,w,maskU,maskV,maskW)
          _du = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskU/(self.area_u*self.dzt_z)
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          _dv = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskV/(self.area_v*self.dzt_z)         
          du = OM.add_to_array(du, (tau,slice(None),slice(None),slice(None)), ob_rho*_du, out_sharding = self.sharding_4D)
          dv = OM.add_to_array(dv, (tau,slice(None),slice(None),slice(None)), ob_rho*_dv, out_sharding = self.sharding_4D)          
          return du,dv

      @partial(OM.jaxjit, static_argnums=0)
      def OB_momentum_linear(self,u,v,du,dv,maskU,maskV,maskT,rho,coriolis_t,tau):   
          """
          Tendency due to linear terms in momentum equation.
          """
          # grid-weighted Coriolis + metric terms (momentum.py)
          _du, _dv = self.coriolis_and_metric(u,v,coriolis_t,maskU,maskV)  
          
          p_h = OM.np.zeros_like(rho)
          p_h = self.hydrostatic_pressure(rho,p_h,maskT)          
          _du +=  - (self.rollx(p_h,-1) - p_h)/(self.cost_y*self.dxu_x)*maskU
          _dv +=  - (self.rolly(p_h,-1) - p_h)/self.dyu_y*maskV  
         
          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)          
          return du,dv,p_h

      @partial(OM.jaxjit, static_argnums=0)    
      def OB_norm(self,u1,v1,u2,v2):
          """
          Calculate a norm of differences in u1,v1 and u2,v2
          (volume-weighted kinetic energy norm; on a uniform grid the
          weights cancel)
          """
          wu = self.maskU*self.area_u*self.dzt_z
          wv = self.maskV*self.area_v*self.dzt_z
          a  = u1**2*wu+v1**2*wv
          e1 = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a) 
          a  = u2**2*wu+v2**2*wv
          e2 = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a)
          a  = (u1-u2)**2*wu+(v1-v2)**2*wv
          d  = OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a)          
          return 2 * OM.np.sqrt(d) / (OM.np.sqrt(e1) + OM.np.sqrt(e2)) 



@OM.jaxjit
def ramp(theta):     
    """ 
    The ramp function: ramp = exp(-1/theta)/(exp(-1/theta) + exp(-1/(1-theta))),
    0 at theta <= 0, 1 at theta >= 1, smooth in between (all derivatives
    vanish at both ends)
    """   
    t1 = OM.np.where(theta   > 1e-32, 1./(1e-32+theta), 1e32)
    t2 = OM.np.where(1-theta > 1e-32, 1./(1e-32+(1-theta)), 1e32)   # (1e-32+1)-theta was 0 at theta = 1
    return OM.np.exp(-t1)/(OM.np.exp(-t1) + OM.np.exp(-t2) )
