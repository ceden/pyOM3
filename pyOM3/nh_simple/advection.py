"""
 advection of buoyancy and momentum

 Based on boundary class. 
"""
import pyOM3                    as OM
from   pyOM3.nh_simple.boundary import boundary


class advection(boundary):


      def buoyancy_advection_2nd(self):   
          """
          Advection of buoyancy with second order 
          """
          east  = 0.5*(self.b + self.rollx(self.b,-1) )*self.u
          north = 0.5*(self.b + self.rolly(self.b,-1) )*self.v
          top   = 0.5*(self.b + self.rollz(self.b,-1) )*self.w
          top = self.apply_vertical_flux_bc(top)          
          return -self.Ro*((east  - self.rollx( east,1)  )/self.dx 
                         + (north - self.rolly( north,1) )/self.dy 
                         + (top   - self.rollz( top,1)   )/self.dz )
          


      def buoyancy_advection_superbee(self):       
          """
          from MITgcm: 2nd oder scheme with Superbee flux limiter.
          """
          def _limiter(cr):
              return OM.np.maximum(OM.np.clip(2 * cr, 0, 1), OM.np.clip(cr, 0, 2))
           
          var = self.b           
          eps = 1e-20 
          
          CFL = OM.np.abs(self.u * self.dt / self.dx)          
          rj  = self.apply_bc( (self.rollx(var,-1) - var) )
          rjm = self.rollx(rj, 1) 
          rjp = self.rollx(rj,-1)                     
          cr = _limiter( OM.np.where(self.u>0.0, rjm, rjp)/OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fx = (self.u * (self.rollx(var,-1) + var) * 0.5 
               - OM.np.abs(self.u) * ((1.0 - cr) + CFL * cr) * rj * 0.5)
          
          CFL = OM.np.abs(self.v * self.dt / self.dy)          
          rj  = self.apply_bc( (self.rolly(var,-1) - var) )
          rjm = self.rolly(rj, 1) 
          rjp = self.rolly(rj,-1)                     
          cr = _limiter( OM.np.where(self.v>0.0, rjm, rjp)/OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fy =  (self.v * (self.rolly(var,-1) + var) * 0.5 
               - OM.np.abs(self.v) * ((1.0 - cr) + CFL * cr) * rj * 0.5)
        
          CFL = OM.np.abs(self.w * self.dt / self.dz)          
          rj  = self.apply_vertical_flux_bc( self.apply_bc( (self.rollz(var,-1) - var) ) )
          rjm = self.rollz(rj, 1) 
          rjp = self.rollz(rj,-1)                     
          cr = _limiter( OM.np.where(self.w>0.0, rjm, rjp)/OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fz_imp = - OM.np.abs(self.w) * ((1.0 - cr) + CFL * cr) * rj * 0.5
          fz =  self.w * (self.rollz(var,-1) + var) * 0.5 + fz_imp

          # The flux at the lower halo face (index 0) needs the slope one
          # face further down, which the single halo row does not hold
          # (the roll wraps to the opposite halo). Each face is therefore
          # taken from the block that owns it: the halo-face flux is
          # replaced by the neighbour's flux of the same face. Without this
          # the two blocks used different fluxes at their common face
          # (non-conservative, decomposition dependent), and the periodic
          # seam in x was wrong even on one PE.
          fx = self.apply_bc(fx)
          fy = self.apply_bc(fy)
          fz = self.apply_bc(fz)
          fz = self.apply_vertical_flux_bc(fz)          
          # vertical implicit subgrid buoyancy flux of the limiter (the part
          # beyond the centred flux), <w'b'> on the w-grid (interior faces
          # are exact, the halo is not used); the LES TKE closure uses it as
          # buoyancy production when les_mix_b is False
          self.b_flux_implicit_z = self.Ro*self.apply_vertical_flux_bc(fz_imp)
          return  - self.Ro*((fx  - self.rollx( fx,1) )/self.dx 
                           + (fy  - self.rolly( fy,1) )/self.dy 
                           + (fz  - self.rollz( fz,1) )/self.dz )

              
      def buoyancy_advection_upwind3(self):   
          """
          Advection of buoyancy with third order upwind from MITgcm
          """
          var = self.b        
          oneSixth=1./6.
              
          rj  = self.apply_bc( (self.rollx(var,-1) - var) )
          rjm = self.rollx(rj, 1) 
          rjp = self.rollx(rj,-1) 
          rjjp = rjp-rj 
          rjjm = rj-rjm
          fx = self.u*(self.rollx(var,-1) + var - oneSixth*(rjjp+rjjm ))*0.5   \
                     +OM.np.abs(self.u)*0.5*oneSixth*(rjjp-rjjm ) 
          
          rj  = self.apply_bc( (self.rolly(var,-1) - var) )
          rjm = self.rolly(rj, 1) 
          rjp = self.rolly(rj,-1) 
          rjjp = rjp-rj 
          rjjm = rj-rjm
          fy = self.v*(self.rolly(var,-1) + var - oneSixth*(rjjp+rjjm ))*0.5   \
                     +OM.np.abs(self.v)*0.5*oneSixth*(rjjp-rjjm )

          rj  = self.apply_vertical_flux_bc( self.apply_bc( (self.rollz(var,-1) - var) ) )
          rjm = self.rollz(rj, 1) 
          rjp = self.rollz(rj,-1) 
          rjjp = rjp-rj 
          rjjm = rj-rjm
          fz = self.w*(self.rollz(var,-1) + var - oneSixth*(rjjp+rjjm ))*0.5   \
                     +OM.np.abs(self.w)*0.5*oneSixth*(rjjp-rjjm )
         
          # halo-face fluxes from the owning block, see superbee
          fx = self.apply_bc(fx)
          fy = self.apply_bc(fy)
          fz = self.apply_bc(fz)
          fz = self.apply_vertical_flux_bc(fz)          
          return  - self.Ro*((fx  - self.rollx( fx,1) )/self.dx 
                           + (fy  - self.rolly( fy,1) )/self.dy 
                           + (fz  - self.rollz( fz,1) )/self.dz )
                   

    
      def momentum_advection_u(self):
          """
          Advection of u-momentum with second order which is energy conserving
          """
          u,v,w = self.u,self.v,self.w
          east  = 0.25*(u + self.rollx(u,-1))*(self.rollx(u,-1) + u)
          north = 0.25*(u + self.rolly(u,-1))*(self.rollx(v,-1) + v)
          top   = 0.25*(u + self.rollz(u,-1))*(self.rollx(w,-1) + w)
          top = self.apply_vertical_flux_bc(top) 
          return -self.Ro*( (east  - self.rollx(east,1)  )/self.dx 
                           +(north - self.rolly(north,1) )/self.dy 
                           +(top   - self.rollz(top,1)   )/self.dz )

    
      def momentum_advection_v(self):
          """
          Advection of v-momentum with second order which is energy conserving
          """
          u,v,w = self.u,self.v,self.w          
          east  = 0.25*(v + self.rollx(v,-1))*(self.rolly(u,-1) + u)
          north = 0.25*(v + self.rolly(v,-1))*(self.rolly(v,-1) + v)
          top   = 0.25*(v + self.rollz(v,-1))*(self.rolly(w,-1) + w)
          top = self.apply_vertical_flux_bc(top)
          return -self.Ro*( (east  - self.rollx(east,1)  )/self.dx 
                           +(north - self.rolly(north,1) )/self.dy 
                           +(top   - self.rollz(top,1)   )/self.dz )

    
      def momentum_advection_w(self):
          """
          Advection of w-momentum with second order which is energy conserving
          """  
          u,v,w = self.u,self.v,self.w 
          east  = 0.25*(w + self.rollx(w,-1))*(self.rollz(u,-1) + u)
          north = 0.25*(w + self.rolly(w,-1))*(self.rollz(v,-1) + v)
          top   = 0.25*(w + self.rollz(w,-1))*(self.rollz(w,-1) + w)
          return -self.dsqr*self.Ro*( (east  - self.rollx(east,1)  )/self.dx 
                                     +(north - self.rolly(north,1) )/self.dy 
                                     +(top   - self.rollz(top,1)   )/self.dz )





          