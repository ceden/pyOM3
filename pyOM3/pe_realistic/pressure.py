"""
 Implicit surface pressure formulation as in MITgcm.
 
 Rigid lid or optionally a linear implicit free surface by using self.implicit_free_surface=True
 
 Based on class main and solver which contains JAX compatible iterative solvers.
 
"""
import pyOM3                    as OM
from   pyOM3.pe_realistic.main     import main 
from   pyOM3.pe_simple.solver   import solver   # grid-metric free, shared with pe_simple
from   functools                import partial


class pressure(main,solver):

    
      @partial(OM.jaxjit, static_argnums=0)
      def solve_for_pressure(self,u,v,p_s,maskU,maskV,maskT,cf,tau,taum1,taum2):

          # forcing for surface pressure and first guess (spherical
          # divergence: pyOM2 solve_pressure.f90:57)
          fpx = OM.np.sum( u*maskU*self.dzt_z, axis=0 )/self.dt 
          fpy = OM.np.sum( v*maskV*self.dzt_z, axis=0 )/self.dt 
          fpx, fpy = self.apply_bc(fpx), self.apply_bc(fpy)
          fpyc = fpy*self.cosu_y[0]
          fc = (  (fpx - self.rollx(fpx ,1))/(self.cost_y[0]*self.dxt_x[0]) 
                + (fpyc- self.rolly(fpyc,1))/(self.cost_y[0]*self.dyt_y[0]) )*maskT[-self.halo-1,:,:]
          
          if self.implicit_free_surface: 
              fc += - p_s[taum1,:,:]/(self.grav*self.dt**2)*maskT[-self.halo-1,:,:] 
          sol =  2*p_s[taum1,:,:] - p_s[taum2,:,:]

          #solve for surface pressure, different choices for solver (does not matter much). 
          result,estimated_error,congr_itts = self.cg(fc,sol,cf)
          #result,estimated_error,congr_itts = self.cg_pre(fc,sol,cf)
          #result,estimated_error,congr_itts = self.cg_pre_nojax(fc,sol,cf)
          #result,estimated_error,congr_itts = self.bicgstab(fc,sol,cf)
          #result,estimated_error,congr_itts = self.bicgstab_pre(fc,sol,cf)
          result = self.apply_bc(result)

          # transfer result to surface pressure and remove surface pressure gradient
          p_s = OM.modify_array(p_s, (tau,slice(None),slice(None)), result , out_sharding = self.sharding_3D)         
          u +=  - self.dt*( self.rollx(p_s[tau,None,:,:],-1)  - p_s[tau,None,:,:] )/(self.cost_y*self.dxu_x)*maskU
          v +=  - self.dt*( self.rolly(p_s[tau,None,:,:],-1)  - p_s[tau,None,:,:] )/self.dyu_y*maskV
          u, v = self.apply_bc(u), self.apply_bc(v)   
          
          return u,v,p_s,estimated_error,congr_itts

    
      @partial(OM.jaxjit, static_argnums=0)
      def make_coeff_surf_press(self,cf,hu,hv,maskT):
          """
          5-point coefficients of the surface pressure operator
             1/cost^2 (h p_x)_x + 1/cost (cosu h p_y)_y
          on the variable grid (pyOM2 solve_pressure.f90
          make_coeff_surf_press):
             x pairs:  -+ hu(i)  /(dxu(i)  *dxt(i)*cost(j)^2)
                       -+ hu(i-1)/(dxu(i-1)*dxt(i)*cost(j)^2)
             y pairs:  -+ hv(j)  *cosu(j)  /(dyu(j)  *dyt(j)*cost(j))
                       -+ hv(j-1)*cosu(j-1)/(dyu(j-1)*dyt(j)*cost(j))
          The flux factor h/dxu (h*cosu/dyu) is built BEFORE the shift so
          the shifted term carries its own face's spacing.
          """
          maskM = maskT[-self.halo-1,:,:]
          a = slice(None)
          gx = hu/self.dxu_x[0]
          wx = 1.0/(self.dxt_x[0]*self.cost_y[0]**2)
          mp = maskM * self.rollx(maskM,-1) # i+1
          mm = maskM * self.rollx(maskM, 1) 
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mp*gx*wx )
          cf = OM.add_to_array(cf,( 0+1, 1+1,a,a), + mp*gx*wx )
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mm*self.rollx(gx,1)*wx )
          cf = OM.add_to_array(cf,( 0+1,-1+1,a,a), + mm*self.rollx(gx,1)*wx )
          
          gy = hv*self.cosu_y[0]/self.dyu_y[0]
          wy = 1.0/(self.dyt_y[0]*self.cost_y[0])
          mp = maskM * self.rolly(maskM,-1) # j+1
          mm = maskM * self.rolly(maskM, 1) 
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mp*gy*wy )
          cf = OM.add_to_array(cf,( 1+1, 0+1,a,a), + mp*gy*wy )
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mm*self.rolly(gy,1)*wy )
          cf = OM.add_to_array(cf,(-1+1, 0+1,a,a), + mm*self.rolly(gy,1)*wy )
          
          if self.implicit_free_surface:     
             cf = OM.add_to_array(cf,(0+1,0+1,a,a), - 1./(self.grav*self.dt**2) *maskM ) 
          return cf

      
