"""
 Momentum equation.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_simple.main     import main
from   functools                import partial


class momentum(main):

    
      @partial(OM.jaxjit, static_argnums=0)
      def update_momentum_AB3(self,A,B,C,tau,taum1,taum2,u,v,du,dv,maskU,maskV):
          """
          Update momentum based on their tendencies.
          """
          u += self.dt*(A*du[tau,:,:,:] + B*du[taum1,:,:,:] + C*du[taum2,:,:,:])*maskU 
          v += self.dt*(A*dv[tau,:,:,:] + B*dv[taum1,:,:,:] + C*dv[taum2,:,:,:])*maskV 
          #return self.apply_bc(u), self.apply_bc(v) # done later in pressure
          return u,v

          
      @partial(OM.jaxjit, static_argnums=0)
      def momentum_tendency_impl(self,u,v,w,du,dv,rho,p_h,maskU,maskV,maskW,maskT,coriolis_t,tau):    
          """
              Add advective, Coriolis, and hydrostatic pressure gradient to tendencies
              extrapolate with AB3 and calculate intermediate velocity.
              This is part of the rigid lid or implicit_free_surface formulation
          """
          fe,fn,ft  = self.mom_flux_2nd_u(u,u,v,w,maskU,maskV,maskW)
          du_ = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskU
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          dv_ = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy + (ft-self.rollz(ft,1))/self.dz )*maskV

          du_ += + maskU*(           coriolis_t    [None,:,:] *           ( v + self.rolly(v,1) ) +
                          self.rollx(coriolis_t,-1)[None,:,:] * self.rollx( v + self.rolly(v,1) ,-1) )*0.25
            
          dv_ += - maskV*(           coriolis_t    [None,:,:] *           ( u + self.rollx(u,1) ) +
                          self.rolly(coriolis_t,-1)[None,:,:] * self.rolly( u + self.rollx(u,1) ,-1) )*0.25
          
          p_h = self.hydrostatic_pressure(rho,p_h,maskT) 
          
          du_ +=  - (self.rollx(p_h,-1) - p_h)/self.dx*maskU
          dv_ +=  - (self.rolly(p_h,-1) - p_h)/self.dy*maskV  

          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), du_, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), dv_, out_sharding = self.sharding_4D)
          
          return du,dv,p_h


      @partial(OM.jaxjit, static_argnums=0)
      def hydrostatic_pressure(self,rho,p_h,maskT):                 
          factor = self.grav/self.rho_0*self.dz    
          p_h = OM.modify_array(p_h,(-(self.halo+1),slice(None),slice(None)), 
                                     0.5*rho[-(self.halo+1),:,:]*factor*maskT[-(self.halo+1),:,:] )    
          def loop_it(k, p_h):
              _k = self.Nz-k-1
              return OM.modify_array(p_h, (_k,slice(None),slice(None)), 
                                     maskT[_k,:,:]*(p_h[_k+1,:,:] + 0.5*(rho[_k+1,:,:]+rho[_k,:,:])*factor ) ) 
              
          return OM.for_loop(self.halo+1, self.Nz-self.halo, loop_it, p_h)

        
      @partial(OM.jaxjit, static_argnums=0)
      def mom_flux_2nd_u(self,u1,u,v,w,maskU,maskV,maskW):
          """
          Simple 2nd order zonal momentum fluxes. u1 is advected by u,v,w.
          u1 and u can be identical
          """
          fe = - ( u*maskU + self.rollx(u*maskU,-1) )*0.25*(u1 + self.rollx(u1,-1) )*maskU*self.rollx(maskU,-1)
          fn = - ( v*maskV + self.rollx(v*maskV,-1) )*0.25*(u1 + self.rolly(u1,-1) )*maskU*self.rolly(maskU,-1)
          ft = - ( w*maskW + self.rollx(w*maskW,-1) )*0.25*(u1 + self.rollz(u1,-1) )*maskU*self.rollz(maskU,-1)
          return self.apply_bc(fe), self.apply_bc(fn), self.apply_bc(ft)  

    
      @partial(OM.jaxjit, static_argnums=0)
      def mom_flux_2nd_v(self,v1,u,v,w,maskU,maskV,maskW):
          """
          Simple 2nd order meridional momentum fluxes. v1 is advected by u,v,w.
          v1 and v can be identical
          """           
          fe = - ( u*maskU + self.rolly(u*maskU,-1) )*0.25*(v1 + self.rollx(v1,-1) )*maskV*self.rollx(maskV,-1)
          fn = - ( v*maskV + self.rolly(v*maskV,-1) )*0.25*(v1 + self.rolly(v1,-1) )*maskV*self.rolly(maskV,-1)
          ft = - ( w*maskW + self.rolly(w*maskW,-1) )*0.25*(v1 + self.rollz(v1,-1) )*maskV*self.rollz(maskV,-1)
          return self.apply_bc(fe), self.apply_bc(fn), self.apply_bc(ft)  

    
      @partial(OM.jaxjit, static_argnums=0)          
      def vertical_velocity(self,u,v,w,maskW):
           """
            Calculate vertical velocity from horizontal divergence.
           """
           _div = (u-self.rollx(u,1))/self.dx  +(v-self.rolly(v,1))/self.dy
           #return -self.rollz(OM.np.cumsum( maskW*self.dz*_div  , axis=0) ,1)*maskW
           return -OM.np.cumsum( maskW*self.dz*_div  , axis=0)*maskW
 
    

 