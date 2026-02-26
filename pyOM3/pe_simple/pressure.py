"""
 Implicit surface pressure formulation as in MITgcm.
 
 Rigid lid or optionally a linear implicit free surface by using self.implicit_free_surface=True
 
 Based on class main and solver which contains JAX compatible iterative solvers.
 
"""
import pyOM3                    as OM
from   pyOM3.pe_simple.main     import main 
from   pyOM3.pe_simple.solver   import solver
from   functools                import partial


class pressure(main,solver):

    
      @partial(OM.jaxjit, static_argnums=0)
      def solve_for_pressure(self,u,v,p_s,maskU,maskV,maskT,cf,tau,taum1,taum2):

          # forcing for surface pressure and first guess
          fpx = OM.np.sum( u*maskU, axis=0 )*self.dz/self.dt 
          fpy = OM.np.sum( v*maskV, axis=0 )*self.dz/self.dt 
          fpx, fpy = self.apply_bc(fpx), self.apply_bc(fpy)
          fc = ( (fpx - self.rollx(fpx,1))/self.dx + (fpy - self.rolly(fpy,1))/self.dy )*maskT[self.Nz-2*self.halo,:,:]
          
          if self.implicit_free_surface: 
              fc += - p_s[taum1,:,:]/(self.grav*self.dt**2)*maskT[self.Nz-2*self.halo,:,:] 
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
          u +=  - self.dt*( self.rollx(p_s[tau,None,:,:],-1)  - p_s[tau,None,:,:] )/self.dx*maskU
          v +=  - self.dt*( self.rolly(p_s[tau,None,:,:],-1)  - p_s[tau,None,:,:] )/self.dy*maskV
          u, v = self.apply_bc(u), self.apply_bc(v)   
          
          return u,v,p_s,estimated_error,congr_itts

    
      @partial(OM.jaxjit, static_argnums=0)
      def make_coeff_surf_press(self,cf,hu,hv,maskT):
          maskM = maskT[self.Nz-2*self.halo,:,:]
          a = slice(None)
          mp = maskM * self.rollx(maskM,-1) # i+1
          mm = maskM * self.rollx(maskM, 1) 
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mp*hu/self.dx**2 )
          cf = OM.add_to_array(cf,( 0+1, 1+1,a,a), + mp*hu/self.dx**2 )
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mm*self.rollx(hu,1)/self.dx**2 )
          cf = OM.add_to_array(cf,( 0+1,-1+1,a,a), + mm*self.rollx(hu,1)/self.dx**2 )
          
          mp = maskM * self.rolly(maskM,-1) # i+1
          mm = maskM * self.rolly(maskM, 1) 
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mp*hv/self.dx**2 )
          cf = OM.add_to_array(cf,( 1+1, 0+1,a,a), + mp*hv/self.dx**2 )
          cf = OM.add_to_array(cf,( 0+1, 0+1,a,a), - mm*self.rolly(hv,1)/self.dy**2 )
          cf = OM.add_to_array(cf,(-1+1, 0+1,a,a), + mm*self.rolly(hv,1)/self.dy**2 )
          
          if self.implicit_free_surface:     
             cf = OM.add_to_array(cf,(0+1,0+1,a,a), - 1./(self.grav*self.dt**2) *maskM ) 
          return cf

      
