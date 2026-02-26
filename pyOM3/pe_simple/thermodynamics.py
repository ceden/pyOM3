"""
 Tracer advection and diffusion.

 Based on class main.
"""
import pyOM3                       as OM
from   pyOM3.pe_simple.main        import main
from   pyOM3.density               import get_rho
from   functools                   import partial


class thermodynamics(main):

   
      @partial(OM.jaxjit, static_argnums=0)
      def update_thermodynamics_AB3(self,A,B,C,tau,taum1,taum2,temp,salt,dtemp,dsalt,maskT): 
          """
          Update temperature and salinity using their tendencies.
          """      
          temp += self.dt*(A*dtemp[tau,:,:,:] + B*dtemp[taum1,:,:,:] + C*dtemp[taum2,:,:,:])*maskT 
          temp = self.apply_bc(temp)
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             salt += self.dt*(A*dsalt[tau,:,:,:] + B*dsalt[taum1,:,:,:] + C*dsalt[taum2,:,:,:])*maskT 
             salt = self.apply_bc(salt) 
          rho = get_rho(salt,temp,self.p0,self.eq_of_state)    
          return temp,salt,rho

        
    
      @partial(OM.jaxjit, static_argnums=0)    
      def tracer_advection(self,var,dvar,u,v,w,maskU,maskV,maskW,maskT,tau):  
          """
           Calculates tendencies due to tracer advection for general tracer var.
           Stores tendency in dvar and overwrites at time level tau.
          """
          if hasattr(self,'superbee_advection'):
             if self.superbee_advection:
                fe,fn,ft   = self.tracer_flux_superbee(var,u,v,w,maskU,maskV,maskW)  
             else: raise ValueError    
          else:
                fe,fn,ft   = self.tracer_flux_2nd(var,u,v,w,maskU,maskV,maskW)   
              
          _dvar = (  (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy 
                   + (ft-self.rollz(ft,1))/self.dz )*maskT 
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), _dvar, out_sharding = self.sharding_4D)
          
                    
    
      @partial(OM.jaxjit, static_argnums=0)
      def tracer_flux_2nd(self,var,u,v,w,maskU,maskV,maskW):
          """ 
          Simple 2nd order tracer fluxes.
          """
          fe = - u*0.5*( var + self.rollx(var,-1) )*maskU
          fn = - v*0.5*( var + self.rolly(var,-1) )*maskV
          ft = - w*0.5*( var + self.rollz(var,-1) )*maskW      
          return self.apply_bc(fe), self.apply_bc(fn), self.apply_bc(ft)
          
   
    
      @partial(OM.jaxjit, static_argnums=0)
      def tracer_flux_superbee(self,var,u,v,w,maskU,maskV,maskW):
          """
          from MITgcm: 2nd oder scheme with Superbee flux limiter.
          """
          def _limiter(cr):
              return OM.np.maximum(OM.np.clip(2 * cr, 0, 1), OM.np.clip(cr, 0, 2))
              
          eps = 1e-20  
          uCFL = OM.np.abs(u * self.dt / self.dx)          
          rj  = self.apply_bc( (self.rollx(var,-1)-var)*maskU )
          rjm = self.rollx(rj, 1) 
          rjp = self.rollx(rj,-1)                     
          cr = _limiter( OM.np.where(u > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj)  )          
          fe = - (u * (self.rollx(var,-1) + var) * 0.5 - OM.np.abs(u) * ((1.0 - cr) + uCFL * cr) * rj * 0.5)*maskU

          uCFL = OM.np.abs(v * self.dt / self.dy)          
          rj  = self.apply_bc( (self.rolly(var,-1)-var)*maskV )
          rjm = self.rolly(rj, 1) 
          rjp = self.rolly(rj,-1)                     
          cr = _limiter( OM.np.where(v > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj)  )          
          fn = - (v * (self.rolly(var,-1) + var) * 0.5 - OM.np.abs(v) * ((1.0 - cr) + uCFL * cr) * rj * 0.5)*maskV
          
          uCFL = OM.np.abs(w * self.dt / self.dz)          
          rj  = self.apply_bc( (self.rollz(var,-1)-var)*maskW )
          rjm = self.rollz(rj, 1) 
          rjp = self.rollz(rj,-1)                     
          cr = _limiter( OM.np.where(w > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj)  )          
          ft = - (w * (self.rollz(var,-1) + var) * 0.5 - OM.np.abs(w) * ((1.0 - cr) + uCFL * cr) * rj * 0.5)*maskW
              
          return self.apply_bc(fe), self.apply_bc(fn), self.apply_bc(ft)
         
          

