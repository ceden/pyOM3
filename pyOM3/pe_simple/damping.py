"""
 Damping terms like biharmonic friction.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_simple.main     import main
from   functools                import partial

class damping(main):

          
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_horizontal(self,p1,mixC,mask):          
          fe = -mixC*(self.rollx(p1,-1)-p1)/self.dx*self.rollx(mask,-1)*mask
          fn = -mixC*(self.rolly(p1,-1)-p1)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)   
          
          del2 = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy )*mask
          del2 = self.apply_bc(del2)
          
          fe = (self.rollx(del2,-1)-del2)/self.dx*self.rollx(mask,-1)*mask
          fn = (self.rolly(del2,-1)-del2)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
 
          return p1 + self.dt*( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy  )*mask


    
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_horizontal_tendency(self,p1,mixC,mask):          
          fe = -mixC*(self.rollx(p1,-1)-p1)/self.dx*self.rollx(mask,-1)*mask
          fn = -mixC*(self.rolly(p1,-1)-p1)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)   
          
          del2 = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy )*mask
          del2 = self.apply_bc(del2)
          
          fe = (self.rollx(del2,-1)-del2)/self.dx*self.rollx(mask,-1)*mask
          fn = (self.rolly(del2,-1)-del2)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
 
          return ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy  )*mask

