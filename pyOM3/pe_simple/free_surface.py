
""" 
Explicit free surface formulation after Demange et al 2019
which is also used in FESOM2. 

Call first momentum_tendency_expl to calculate tendency terms,
then solve_expl_surface to sub cycle the external mode equations.

"""

import pyOM3                    as OM
from   pyOM3.pe_simple.main     import main 
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
          _du = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskU
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          _dv = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskV
          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)

          _du = + maskU*(           coriolis_t    [None,:,:] *           ( v + self.rolly(v,1) ) +
                         self.rollx(coriolis_t,-1)[None,:,:] * self.rollx( v + self.rolly(v,1) ,-1) )*0.25            
          _dv = - maskV*(           coriolis_t    [None,:,:] *           ( u + self.rollx(u,1) ) +
                         self.rolly(coriolis_t,-1)[None,:,:] * self.rolly( u + self.rollx(u,1) ,-1) )*0.25
          du_cor = OM.modify_array(du_cor, (tau,slice(None),slice(None),slice(None)), _du, out_sharding = self.sharding_4D)
          dv_cor = OM.modify_array(dv_cor, (tau,slice(None),slice(None),slice(None)), _dv, out_sharding = self.sharding_4D)
         
          factor = self.grav/self.rho_0*self.dz  
          p_h = OM.np.zeros_like(rho)
          p_h = self.hydrostatic_pressure(rho,p_h,maskT)
          return du,dv,du_cor,dv_cor,p_h
          
    
      @partial(OM.jaxjit, static_argnums=0)
      def solve_expl_surface(self,u,v,du,dv,du_cor,dv_cor,p_h,p_s,
                             maskU,maskV,maskW,maskT,hu,hv,coriolis_t,
                             A,B,C,tau,taum1,taum2):    
          """
              This is the split-explicit formulation after Demange et al 2019
              with sub cycling of the external mode equation.
              
              Number of sub cycling given by M=self.sub_cycling
              
          """

          @partial(OM.jaxjit, out_shardings=self.sharding_2D, static_argnums=(0,1))
          def zeros_2D(Ny,Nx):
              return OM.np.zeros((Ny,Nx), OM.prec ) 
                    
          _du_hyd  =  - (self.rollx(p_h,-1) - p_h)/self.dx*maskU
          _dv_hyd  =  - (self.rolly(p_h,-1) - p_h)/self.dy*maskV  
          _du_surf =  - (self.rollx(p_s,-1) - p_s)/self.dx*maskU[-self.halo-1,:,:]
          _dv_surf =  - (self.rolly(p_s,-1) - p_s)/self.dy*maskV[-self.halo-1,:,:] 
          
          if hasattr(self,'Ahbi'): 
             _du_mix = self.biharmonic_horizontal_tendency(u,self.Ahbi,maskU)
             _dv_mix = self.biharmonic_horizontal_tendency(v,self.Ahbi,maskV) 
          else:
              _du_mix, _dv_mix = 0,0
              
          _R_u = A*du[tau,:,:,:] + B*du[taum1,:,:,:] + C*du[taum2,:,:,:] + _du_hyd + _du_mix
          _R_v = A*dv[tau,:,:,:] + B*dv[taum1,:,:,:] + C*dv[taum2,:,:,:] + _dv_hyd + _dv_mix
          
          U, V = OM.np.sum( u*maskU, axis=0 )*self.dz, OM.np.sum( v*maskV, axis=0 )*self.dz
          
          u += self.dt*(_R_u + A*du_cor[tau,:,:,:] + B*du_cor[taum1,:,:,:] + C*du_cor[taum2,:,:,:] + _du_surf[None,:,:])*maskU 
          v += self.dt*(_R_v + A*dv_cor[tau,:,:,:] + B*dv_cor[taum1,:,:,:] + C*dv_cor[taum2,:,:,:] + _dv_surf[None,:,:])*maskV 
          u, v = self.apply_bc(u), self.apply_bc(v)
          
          _R_u, _R_v = OM.np.sum( _R_u*maskU, axis=0 )*self.dz , OM.np.sum( _R_v*maskV, axis=0 )*self.dz

          maskUU = maskU[-self.halo-1,:,:]
          maskVV = maskV[-self.halo-1,:,:]
          maskTT = maskT[-self.halo-1,:,:]
          M = self.sub_cycling
          AB2_a, AB2_b, theta = 1.5,  -0.5, 0.14
          
          Um1, Vm1  = U*1,V*1
          Um, Vm  = zeros_2D(self.Ny,self.Nx), zeros_2D(self.Ny,self.Nx)
          U0, V0  = U*1, V*1
          
          def loop_it(m,state):
              U,V,p_s,Um,Vm,Um1,Vm1 = state
              _u , _v =  AB2_a*U + AB2_b*Um1, AB2_a*V + AB2_b*Vm1              
              _du = + maskUU*(       coriolis_t *               ( _v + self.rolly(_v,1) ) +
                          self.rollx(coriolis_t,-1) * self.rollx( _v + self.rolly(_v,1) ,-1) )*0.25            
              _dv = - maskVV*(       coriolis_t *               ( _u + self.rollx(_u,1) ) +
                          self.rolly(coriolis_t,-1) * self.rolly( _u + self.rollx(_u,1) ,-1) )*0.25 

              Up1 = U + self.apply_bc( self.dt/M*( _du - hu*(self.rollx(p_s,-1) - p_s)/self.dx*maskUU + _R_u )*maskUU )
              Vp1 = V + self.apply_bc( self.dt/M*( _dv - hv*(self.rolly(p_s,-1) - p_s)/self.dy*maskVV + _R_v )*maskVV )
              Um += Up1
              Vm += Vp1              
              p_s += self.dt/M*self.grav*( -(1+theta)*( (Up1-self.rollx(Up1,1))/self.dx 
                                                      + (Vp1-self.rolly(Vp1,1))/self.dy ) 
                                           +   theta *( (U-self.rollx(U,1))/self.dx 
                                                      + (V-self.rolly(V,1))/self.dy ) ) *maskTT
              p_s = self.apply_bc(p_s)
              Um1,Vm1 = U, V
              U, V = Up1, Vp1
              return (U,V,p_s,Um,Vm,Um1,Vm1)
              
          U,V,p_s,Um,Vm,Um1,Vm1 = OM.for_loop(0, M, loop_it, (U,V,p_s,Um,Vm,Um1,Vm1))
          
          Um, Vm = Um/M + theta/M*(U-U0), Vm/M + theta/M*(V-V0)          
          u -= OM.np.where( hu>0, ( OM.np.sum( u*maskU, axis=0 )*self.dz - Um)/OM.np.maximum(1e-12,hu), 0) *maskU
          v -= OM.np.where( hv>0, ( OM.np.sum( v*maskV, axis=0 )*self.dz - Vm)/OM.np.maximum(1e-12,hv), 0) *maskV                   
          return u,v,p_s




