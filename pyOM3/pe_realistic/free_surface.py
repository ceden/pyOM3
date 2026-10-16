
""" 
Explicit free surface formulation after Demange et al 2019
which is also used in FESOM2. 

Call first momentum_tendency_expl to calculate tendency terms,
then solve_expl_surface to sub cycle the external mode equations.

"""

import pyOM3                    as OM
from   pyOM3.pe_realistic.main     import main 
from   functools                import partial
import numpy                    as np__

class free_surface(main):


      @partial(OM.jaxjit, static_argnums=0)
      def momentum_tendency_expl(self,u,v,w,du,dv,du_cor,dv_cor,rho,
                                 maskU,maskV,maskW,maskT,coriolis_t,tau):    
          """
              Calculate momentum tendencies for the split-explicit formulation
          """
          
          fe,fn,ft  = self.mom_flux_2nd_u(u,u,v,w,maskU,maskV,maskW)
          _du = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskU/(self.area_u*self.dzt_z)
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          _dv = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskV/(self.area_v*self.dzt_z)
          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)

          # Coriolis + metric terms, grid-weighted (momentum.py)
          _du, _dv = self.coriolis_and_metric(u,v,coriolis_t,maskU,maskV)
          du_cor = OM.modify_array(du_cor, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv_cor = OM.modify_array(dv_cor, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)
         
          p_h = OM.np.zeros_like(rho)
          p_h = self.hydrostatic_pressure(rho,p_h,maskT)
          return du,dv,du_cor,dv_cor,p_h
          
    
      def init_barotropic_state(self,u,v,maskU,maskV):
          """barotropic transports from the 3-D velocities, for a cold start
          of the split-explicit sub cycle (afterwards U_bt, V_bt are carried
          as state and belong in the restart)"""
          return OM.np.sum( u*maskU*self.dzt_z, axis=0 ), OM.np.sum( v*maskV*self.dzt_z, axis=0 )

      @partial(OM.jaxjit, static_argnums=0)
      def solve_expl_surface(self,u,v,du,dv,du_cor,dv_cor,p_h,p_s,U_bt,V_bt,
                             maskU,maskV,maskW,maskT,hu,hv,coriolis_t,
                             A,B,C,tau,taum1,taum2):    
          """
              This is the split-explicit formulation after Demange et al 2019
              with sub cycling of the external mode equation.
              
              Number of sub cycling given by M=self.sub_cycling
              
              Barotropic system solved in the sub cycle (transports
              U = int u dz, V = int v dz over the fixed depth hu, hv):
              
                dU/dt = + f V + tantr U V/h - h/(a cos(phi)) dp_s/dlambda + R_U
                dV/dt = - f U - tantr U^2/h - h/a        dp_s/dphi     + R_V
                dp_s/dt = -g/(a cos(phi)) [ dU/dlambda + d(V cos(phi))/dphi ]
              
              with tantr = tan(phi)/a and R_U, R_V the depth integral of the
              advective, hydrostatic-pressure and friction tendencies ONLY:
              Coriolis and the metric terms are in du_cor/dv_cor, are kept out
              of _R_u/_R_v, and are re-evaluated inside the sub cycle from the
              barotropic transports.
              
          """

          @partial(OM.jaxjit, out_shardings=self.sharding_2D, static_argnums=(0,1))
          def zeros_2D(Ny,Nx):
              return OM.np.zeros((Ny,Nx), OM.prec ) 
                    
          _du_hyd  =  - (self.rollx(p_h,-1) - p_h)/(self.cost_y*self.dxu_x)*maskU
          _dv_hyd  =  - (self.rolly(p_h,-1) - p_h)/self.dyu_y*maskV  
          _du_surf =  - (self.rollx(p_s,-1) - p_s)/(self.cost_y[0]*self.dxu_x[0])*maskU[-self.halo-1,:,:]
          _dv_surf =  - (self.rolly(p_s,-1) - p_s)/self.dyu_y[0]*maskV[-self.halo-1,:,:] 
          
          if hasattr(self,'Ahbi'): 
             # pyOM2's biharmonic momentum friction (U/V-row metrics, cos^p
             # per Laplacian pass) as a tendency: the same operator the
             # implicit path applies as a split step, so both paths dissipate
             # identically. The T-point form used before had no cos scaling
             # (pe_simple keeps it: uniform Cartesian, the two coincide).
             _un, _vn = self.biharmonic_momentum(u,v,self.Ahbi,maskU,maskV)
             _du_mix, _dv_mix = (_un-u)/self.dt, (_vn-v)/self.dt
          else:
              _du_mix, _dv_mix = 0,0
              
          _R_u = A*du[tau,:,:,:] + B*du[taum1,:,:,:] + C*du[taum2,:,:,:] + _du_hyd + _du_mix
          _R_v = A*dv[tau,:,:,:] + B*dv[taum1,:,:,:] + C*dv[taum2,:,:,:] + _dv_hyd + _dv_mix
          
          # Barotropic state carried across baroclinic steps (Banerjee et al.
          # 2024, FESOM2 split-explicit solver, section 2.3): the sub cycle
          # starts from the END-of-cycle transports of the previous step,
          # U_bt, V_bt, together with the end-of-cycle p_s. The 3-D velocities
          # are trimmed to the sub-cycle AVERAGE <<U>> (eq. 13 there), so
          # rebuilding U from sum(u dz) -- what this routine did before --
          # pairs the averaged transport with the end-of-cycle p_s. That
          # mismatch is fed back every step and grows exponentially when the
          # external-wave CFL c dt/dx is large, the faster the larger M
          # (measured: 1-D toy and the pe bump harness, Sep 2026).
          U, V = U_bt*1, V_bt*1
          # bottom drag on the barotropic transport (0 unless a bottom friction
          # option is on), applied implicitly per sub step: U /(1 + dt/M r_bt)
          if (getattr(self,'enable_bottom_friction',False) or
              getattr(self,'enable_quadratic_bottom_friction',False)):
             _rbu, _rbv = self.bottom_friction_rate_bt(u,v,maskU,maskV,hu,hv)
             _fbu, _fbv = 1.0/(1.0 + self.dt/self.sub_cycling*_rbu), 1.0/(1.0 + self.dt/self.sub_cycling*_rbv)
          else:
             _fbu = _fbv = 1.0
          
          u += self.dt*(_R_u + A*du_cor[tau,:,:,:] + B*du_cor[taum1,:,:,:] + C*du_cor[taum2,:,:,:] + _du_surf[None,:,:])*maskU 
          v += self.dt*(_R_v + A*dv_cor[tau,:,:,:] + B*dv_cor[taum1,:,:,:] + C*dv_cor[taum2,:,:,:] + _dv_surf[None,:,:])*maskV 
          u, v = self.apply_bc(u), self.apply_bc(v)
          
          _R_u, _R_v = OM.np.sum( _R_u*maskU*self.dzt_z, axis=0 ) , OM.np.sum( _R_v*maskV*self.dzt_z, axis=0 )

          maskUU = maskU[-self.halo-1,:,:]
          maskVV = maskV[-self.halo-1,:,:]
          maskTT = maskT[-self.halo-1,:,:]
          M = self.sub_cycling
          # theta: dissipation of the forward-backward sub cycle (Demange et al
          # 2019 use 0.14). It damps the grid-scale barotropic mode; at 1/10 deg
          # with dt 216 s (c dt/dx ~ 5) 0.14 left grid-scale gravity waves whose
          # surface divergence drove w ~ 1e-2 m/s in 10 m cells and a tracer
          # runaway at a steep coast (Sep 2026, reproduced from a POP restart);
          # 0.5 held it. Configuration attribute fb_theta, default 0.14.
          AB2_a, AB2_b, theta = 1.5,  -0.5, getattr(self, 'fb_theta', 0.14)
          
          Um1, Vm1  = U*1,V*1
          Um, Vm  = zeros_2D(self.Ny,self.Nx), zeros_2D(self.Ny,self.Nx)
          U0, V0  = U*1, V*1
          
          def loop_it(m,state):
              U,V,p_s,Um,Vm,Um1,Vm1 = state
              _u , _v =  AB2_a*U + AB2_b*Um1, AB2_a*V + AB2_b*Vm1              
              # grid-weighted Coriolis on the barotropic transports (2-D
              # views of the metric arrays; same weights as
              # coriolis_and_metric)
              _fv = coriolis_t*( _v + self.rolly(_v,1) )*self.dxt_x[0]
              _du = + maskUU*( _fv + self.rollx(_fv,-1) )*0.25/self.dxu_x[0]
              _fu = coriolis_t*( _u + self.rollx(_u,1) )*self.dyt_y[0]*self.cost_y[0]
              _dv = - maskVV*( _fu + self.rolly(_fu,-1) )*0.25/(self.dyu_y[0]*self.cosu_y[0])

              if self.coord_degree:
                 # metric terms of the barotropic mode, the depth integral of
                 # +uv tan(phi)/a and -u^2 tan(phi)/a with the split's
                 # assumption u = U/h, v = V/h:  +tantr U V/h, -tantr U^2/h.
                 # Same stencil and weights as coriolis_and_metric (momentum.py),
                 # evaluated on the barotropic velocities and converted back to
                 # a transport tendency with hu, hv. Without them the barotropic
                 # mode keeps Coriolis but loses the metric terms that du_cor
                 # carries for the baroclinic part (they are excluded from _R_u).
                 _ub = _u/OM.np.maximum(1e-12,hu)
                 _vb = _v/OM.np.maximum(1e-12,hv)
                 _mv = ( _ub + self.rollx(_ub,1) )*( _vb + self.rolly(_vb,1) )*self.dxt_x[0]
                 _du += + maskUU*hu*self.tantr_y[0]*( _mv + self.rollx(_mv,-1) )*0.125/self.dxu_x[0]
                 _mu = self.tantr_y[0]*( _ub + self.rollx(_ub,1) )**2*self.dyt_y[0]*self.cost_y[0]
                 _dv += - maskVV*hv*( _mu + self.rolly(_mu,-1) )*0.125/(self.dyu_y[0]*self.cosu_y[0])

              Up1 = U + self.apply_bc( self.dt/M*( _du - hu*(self.rollx(p_s,-1) - p_s)
                                                   /(self.cost_y[0]*self.dxu_x[0])*maskUU + _R_u )*maskUU )
              Vp1 = V + self.apply_bc( self.dt/M*( _dv - hv*(self.rolly(p_s,-1) - p_s)
                                                   /self.dyu_y[0]*maskVV + _R_v )*maskVV )
              Up1 = Up1*_fbu; Vp1 = Vp1*_fbv          # bottom drag, implicit per sub step
              Um += Up1
              Vm += Vp1              
              Vc , Vp1c = V*self.cosu_y[0] , Vp1*self.cosu_y[0]
              p_s += self.dt/M*self.grav*( -(1+theta)*( (Up1-self.rollx(Up1,1))/(self.cost_y[0]*self.dxt_x[0]) 
                                                      + (Vp1c-self.rolly(Vp1c,1))/(self.cost_y[0]*self.dyt_y[0]) ) 
                                           +   theta *( (U-self.rollx(U,1))/(self.cost_y[0]*self.dxt_x[0]) 
                                                      + (Vc-self.rolly(Vc,1))/(self.cost_y[0]*self.dyt_y[0]) ) ) *maskTT
              p_s = self.apply_bc(p_s)
              Um1,Vm1 = U, V
              U, V = Up1, Vp1
              return (U,V,p_s,Um,Vm,Um1,Vm1)
              
          U,V,p_s,Um,Vm,Um1,Vm1 = OM.for_loop(0, M, loop_it, (U,V,p_s,Um,Vm,Um1,Vm1))
          
          Um, Vm = Um/M + theta/M*(U-U0), Vm/M + theta/M*(V-V0)          
          u -= OM.np.where( hu>0, ( OM.np.sum( u*maskU*self.dzt_z, axis=0 ) - Um)/OM.np.maximum(1e-12,hu), 0) *maskU
          v -= OM.np.where( hv>0, ( OM.np.sum( v*maskV*self.dzt_z, axis=0 ) - Vm)/OM.np.maximum(1e-12,hv), 0) *maskV                   
          return u,v,p_s,U,V




