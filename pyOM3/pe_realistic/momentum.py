"""
 Momentum equation, generalised to variable grid spacings and the
 pseudo-spherical coordinate system (pyOM2: momentum.f90,
 momentum_advection.f90).

 Same 2nd-order scheme and the SAME mask conventions as pe_simple; the
 geometry enters through (a) transports instead of velocities in the
 advective fluxes (MITgcm form: utr = dzt*dyt*u etc., divergence over
 area*dzt) -- on a uniform Cartesian grid the constant factors cancel and
 this reproduces pe_simple to round-off, (b) grid-weighted Coriolis
 averaging, (c) the metric terms uv*tan(phi)/a, -u^2*tan(phi)/a active
 under coord_degree, (d) metric factors in the pressure gradients and the
 continuity divergence.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_realistic.main     import main
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
      def coriolis_and_metric(self,u,v,coriolis_t,maskU,maskV):
          """
          Coriolis tendencies with grid-weighted averaging (pyOM2
          momentum.f90: dxt/dxu and dyt*cost/(dyu*cosu) weights; reduces to
          the plain 0.25 average on a uniform Cartesian grid) plus the
          metric terms +uv*tan(phi)/a and -u^2*tan(phi)/a (tantr = 0 for
          coord_degree = False).
          Pattern: weight each T-row term BEFORE the shift, divide by the
          U/V-row spacing at the destination point.
          """
          _fv = coriolis_t[None,:,:]*( v + self.rolly(v,1) )*self.dxt_x
          du_ = + maskU*( _fv + self.rollx(_fv,-1) )*0.25/self.dxu_x

          _fu = coriolis_t[None,:,:]*( u + self.rollx(u,1) )*self.dyt_y*self.cost_y
          dv_ = - maskV*( _fu + self.rolly(_fu,-1) )*0.25/(self.dyu_y*self.cosu_y)

          if self.coord_degree:
             _mv = ( u + self.rollx(u,1) )*( v + self.rolly(v,1) )*self.dxt_x
             du_ += + maskU*self.tantr_y*( _mv + self.rollx(_mv,-1) )*0.125/self.dxu_x
             _mu = self.tantr_y*( u + self.rollx(u,1) )**2*self.dyt_y*self.cost_y
             dv_ += - maskV*( _mu + self.rolly(_mu,-1) )*0.125/(self.dyu_y*self.cosu_y)
          return du_, dv_

          
      @partial(OM.jaxjit, static_argnums=0)
      def momentum_tendency_impl(self,u,v,w,du,dv,rho,p_h,maskU,maskV,maskW,maskT,coriolis_t,tau):    
          """
              Add advective, Coriolis, and hydrostatic pressure gradient to tendencies
              extrapolate with AB3 and calculate intermediate velocity.
              This is part of the rigid lid or implicit_free_surface formulation
          """
          fe,fn,ft  = self.mom_flux_2nd_u(u,u,v,w,maskU,maskV,maskW)
          du_ = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskU/(self.area_u*self.dzt_z)
          fe,fn,ft  = self.mom_flux_2nd_v(v,u,v,w,maskU,maskV,maskW)  
          dv_ = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskV/(self.area_v*self.dzt_z)

          _du_c, _dv_c = self.coriolis_and_metric(u,v,coriolis_t,maskU,maskV)
          du_ += _du_c
          dv_ += _dv_c
          
          p_h = self.hydrostatic_pressure(rho,p_h,maskT) 
          
          du_ +=  - (self.rollx(p_h,-1) - p_h)/(self.cost_y*self.dxu_x)*maskU
          dv_ +=  - (self.rolly(p_h,-1) - p_h)/self.dyu_y*maskV  

          du = OM.modify_array(du, (tau,slice(None),slice(None),slice(None)), du_, out_sharding = self.sharding_4D)
          dv = OM.modify_array(dv, (tau,slice(None),slice(None),slice(None)), dv_, out_sharding = self.sharding_4D)
          
          return du,dv,p_h


      @partial(OM.jaxjit, static_argnums=0)
      def hydrostatic_pressure(self,rho,p_h,maskT):                 
          """
          Downward integration with the W spacing dzw (distance between T
          points; pyOM2 solve_pressure.f90). Surface half cell: 0.5*rho*dzw
          at the surface level, interior p(k) = p(k+1)
          + 0.5*(rho(k+1)+rho(k))*g/rho_0*dzw(k).
          """
          factor = self.grav/self.rho_0    
          p_h = OM.modify_array(p_h,(-(self.halo+1),slice(None),slice(None)), 
                                     0.5*rho[-(self.halo+1),:,:]*factor
                                     *self.dzw[-(self.halo+1)]*maskT[-(self.halo+1),:,:] )    
          def loop_it(k, p_h):
              _k = self.Nz-k-1
              return OM.modify_array(p_h, (_k,slice(None),slice(None)), 
                                     maskT[_k,:,:]*(p_h[_k+1,:,:] + 0.5*(rho[_k+1,:,:]+rho[_k,:,:])
                                                    *factor*self.dzw[_k] ) ) 
              
          return OM.for_loop(self.halo+1, self.Nz-self.halo, loop_it, p_h)

        
      @partial(OM.jaxjit, static_argnums=0)
      def mom_flux_2nd_u(self,u1,u,v,w,maskU,maskV,maskW):
          """
          Simple 2nd order zonal momentum fluxes as volume transports
          (pyOM2 momentum_advection.f90: utr = dzt*dyt*u, vtr =
          dzt*cosu*dxt*v, wtr = area_t*w). u1 is advected by u,v,w; u1 and
          u can be identical. Divergence is over area_u*dzt (see
          momentum_tendency_impl).
          """
          utr = u*maskU*self.dzt_z*self.dyt_y
          vtr = v*maskV*self.dzt_z*self.cosu_y*self.dxt_x
          wtr = w*maskW*self.area_t
          # pyOM2/MITgcm convention: fluxes are NOT masked by U/V-cell pairs;
          # the transports (masked) carry the boundary condition, keeping the
          # energy-conserving self-flux at the last wet face next to walls.
          # Only the top face is closed explicitly (pyOM2: flux_top(:,:,nz)=0),
          # since maskW is opened at the surface for the free-surface solver.
          fe = - ( utr + self.rollx(utr,-1) )*0.25*(u1 + self.rollx(u1,-1) )
          fn = - ( vtr + self.rollx(vtr,-1) )*0.25*(u1 + self.rolly(u1,-1) )
          ft = - ( wtr + self.rollx(wtr,-1) )*0.25*(u1 + self.rollz(u1,-1) )
          ft = OM.modify_array(ft, (-(self.halo+1), slice(None), slice(None)),
                                OM.np.zeros_like(ft[-(self.halo+1)]))
          # no apply_bc on the fluxes: the halo-slot fluxes (e.g. the western
          # wall face fe[-1]) are legitimate face fluxes computed from already
          # halo-exchanged inputs, and the divergence at the first interior
          # point needs them (pyOM2 computes flux_east from is_pe-1 and keeps
          # it); apply_bc would zero them for non-cyclic domains.
          return fe, fn, ft  

    
      @partial(OM.jaxjit, static_argnums=0)
      def mom_flux_2nd_v(self,v1,u,v,w,maskU,maskV,maskW):
          """
          Simple 2nd order meridional momentum fluxes as volume transports.
          v1 is advected by u,v,w; v1 and v can be identical.
          Divergence is over area_v*dzt.
          """           
          utr = u*maskU*self.dzt_z*self.dyt_y
          vtr = v*maskV*self.dzt_z*self.cosu_y*self.dxt_x
          wtr = w*maskW*self.area_t
          # pyOM2/MITgcm convention: fluxes are NOT masked by U/V-cell pairs;
          # the transports (masked) carry the boundary condition, keeping the
          # energy-conserving self-flux at the last wet face next to walls.
          # Only the top face is closed explicitly (pyOM2: flux_top(:,:,nz)=0),
          # since maskW is opened at the surface for the free-surface solver.
          fe = - ( utr + self.rolly(utr,-1) )*0.25*(v1 + self.rollx(v1,-1) )
          fn = - ( vtr + self.rolly(vtr,-1) )*0.25*(v1 + self.rolly(v1,-1) )
          ft = - ( wtr + self.rolly(wtr,-1) )*0.25*(v1 + self.rollz(v1,-1) )
          ft = OM.modify_array(ft, (-(self.halo+1), slice(None), slice(None)),
                                OM.np.zeros_like(ft[-(self.halo+1)]))
          # no apply_bc on the fluxes: the halo-slot fluxes (e.g. the western
          # wall face fe[-1]) are legitimate face fluxes computed from already
          # halo-exchanged inputs, and the divergence at the first interior
          # point needs them (pyOM2 computes flux_east from is_pe-1 and keeps
          # it); apply_bc would zero them for non-cyclic domains.
          return fe, fn, ft  

    
      @partial(OM.jaxjit, static_argnums=0)          
      def vertical_velocity(self,u,v,w,maskW):
           """
            Calculate vertical velocity from horizontal divergence
            (spherical form, pyOM2 vertical_velocity.f90):
              div = (u_i - u_{i-1})/(cost*dxt)
                  + (cosu_j v_j - cosu_{j-1} v_{j-1})/(cost*dyt).
           """
           vc   = v*self.cosu_y
           _div = (u-self.rollx(u,1))/(self.cost_y*self.dxt_x) \
                 +(vc-self.rolly(vc,1))/(self.cost_y*self.dyt_y)
           #return -self.rollz(OM.np.cumsum( maskW*self.dzt_z*_div  , axis=0) ,1)*maskW
           return -OM.np.cumsum( maskW*self.dzt_z*_div  , axis=0)*maskW
 
    

 