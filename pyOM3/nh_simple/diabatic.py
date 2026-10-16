"""
 mixing and friction of buoyancy and momentum

 Based on boundary class. 
"""
import pyOM3                    as OM
from   pyOM3.nh_simple.boundary import boundary


class diabatic(boundary):


      def harmonic(self,a,Ah,Av,on_wgrid=False):
          """
          horizontal and vertical harmonic operator
          returns tendency
          """
          fx = Ah*(self.rollx(a,-1) - a)/self.dx
          fy = Ah*(self.rolly(a,-1) - a)/self.dy
          fz = Av*(self.rollz(a,-1) - a)/self.dz
          if not on_wgrid: fz = self.apply_vertical_flux_bc(fz)          
          return   ((fx - self.rollx( fx,1) )/self.dx 
                  + (fy - self.rolly( fy,1) )/self.dy 
                  + (fz - self.rollz( fz,1) )/self.dz )

    
      def harmonic_v_only(self,a,Av,on_wgrid=False):
          """
          vertical harmonic operator
          returns tendency
          """
          fz = Av*(self.rollz(a,-1) - a)/self.dz
          if not on_wgrid: fz = self.apply_vertical_flux_bc(fz)          
          return   (fz - self.rollz( fz,1) )/self.dz 

    
      def harmonic_h_only(self,a,Ah):
          """
          horizontal harmonic operator
          returns tendency
          """
          fx = Ah*(self.rollx(a,-1) - a)/self.dx
          fy = Ah*(self.rolly(a,-1) - a)/self.dy                
          return   ((fx - self.rollx( fx,1) )/self.dx 
                  + (fy - self.rolly( fy,1) )/self.dy)


    

      def biharmonic(self,a,Ahbi,Avbi,on_wgrid=False):
          """
          horizontal and vertical biharmonic operator
          returns tendency
          """
          fx = -Ahbi*(self.rollx(a,-1) - a)/self.dx
          fy = -Ahbi*(self.rolly(a,-1) - a)/self.dy
          fz = -Avbi*(self.rollz(a,-1) - a)/self.dz
          if not on_wgrid: fz = self.apply_vertical_flux_bc(fz)          
          _del =   ((fx - self.rollx( fx,1) )/self.dx 
                  + (fy - self.rolly( fy,1) )/self.dy 
                  + (fz - self.rollz( fz,1) )/self.dz )
          _del = self.apply_bc(_del)              
          if on_wgrid: _del = self.apply_vertical_flux_bc(_del)   # the Laplacian of w vanishes at the lids (w odd)
          fx = (self.rollx(_del,-1) - _del)/self.dx
          fy = (self.rolly(_del,-1) - _del)/self.dy
          fz = (self.rollz(_del,-1) - _del)/self.dz
          if not on_wgrid: fz = self.apply_vertical_flux_bc(fz)          
          return  ((fx - self.rollx( fx,1) )/self.dx 
                 + (fy - self.rolly( fy,1) )/self.dy 
                 + (fz - self.rollz( fz,1) )/self.dz )



          
 