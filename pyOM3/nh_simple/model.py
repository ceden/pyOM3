"""
 Brings together all component for the model.
 Defines time steps for different options and a loops over them.

 Based on diverse other classes. 
"""
import pyOM3                      as OM
from   pyOM3.nh_simple.main       import main
from   pyOM3.nh_simple.advection  import advection
from   pyOM3.nh_simple.diabatic   import diabatic
from   pyOM3.nh_simple.pressure   import pressure
import numpy                      as np__


class model(main,advection,diabatic,pressure):

    
      def time_step(self,tA,tB,tC,tau,taum1,taum2):
          """
          one time step of model equations
          """ 

          if    getattr(self,'buoyancy_superbee_advection',False):
             self.db = OM.modify_array(self.db, (tau,None) , self.buoyancy_advection_superbee() )  
          elif  getattr(self,'buoyancy_upwind3_advection',False):    
             self.db = OM.modify_array(self.db, (tau,None) , self.buoyancy_advection_upwind3() )  
          else:    
             self.db = OM.modify_array(self.db, (tau,None) , self.buoyancy_advection_2nd() )
          
          
          # advection of background stratification
          self.db = OM.add_to_array(self.db, (tau,None) ,
                                   - 0.5*(self.w + self.rollz(self.w,1))*self.N0**2 )
 
          # momentum advection
          self.du = OM.modify_array(self.du, (tau,None) ,self.momentum_advection_u() )
          self.dv = OM.modify_array(self.dv, (tau,None) ,self.momentum_advection_v() )
          self.dw = OM.modify_array(self.dw, (tau,None) ,self.momentum_advection_w() )
                    
          #  add time tendency due to Coriolis force 
          self.du = OM.add_to_array(self.du, (tau,None) ,
                                  + (self.v + self.rolly(self.v,1) + 
                          self.rollx(self.v + self.rolly(self.v,1),-1)  )*0.25*self.f0 )
          self.dv = OM.add_to_array(self.dv, (tau,None) ,
                              - (self.u + self.rollx(self.u,1) +  
                      self.rolly(self.u + self.rollx(self.u,1) ,-1) )*0.25*self.f0 )

          # add time tendency due to buoyancy force
          self.dw = OM.add_to_array(self.dw, (tau,None) ,
                                            0.5*(self.b + self.rollz(self.b,-1)) )
          
          self.dw = OM.modify_array(self.dw, (tau,None) , 
                                    self.apply_vertical_flux_bc(self.dw[tau,...]) )
              
          # add friction and mixing 
          if hasattr(self,'Ah') and hasattr(self,'Av'):
             self.u += self.dt*self.harmonic(self.u,self.Ah,self.Av)
             self.v += self.dt*self.harmonic(self.v,self.Ah,self.Av)
             self.w += self.dt*self.harmonic(self.w,self.Ah,self.Av,on_wgrid=True) 
          if hasattr(self,'Ah') and not hasattr(self,'Av'):
             self.u += self.dt*self.harmonic_h_only(self.u,self.Ah)
             self.v += self.dt*self.harmonic_h_only(self.v,self.Ah)
             self.w += self.dt*self.harmonic_h_only(self.w,self.Ah)   
          if not hasattr(self,'Ah') and hasattr(self,'Av'): 
             self.u += self.dt*self.harmonic_v_only(self.u,self.Av)
             self.v += self.dt*self.harmonic_v_only(self.v,self.Av)
             self.w += self.dt*self.harmonic_v_only(self.w,self.Av,on_wgrid=True) 
 
          if hasattr(self,'Kh') and hasattr(self,'Kv'):
             self.b += self.dt*self.harmonic(self.b,self.Kh,self.Kv)
              
          if hasattr(self,'Ahbi') and hasattr(self,'Avbi'):
             self.u += self.dt*self.biharmonic(self.u,self.Ahbi,self.Avbi)
             self.v += self.dt*self.biharmonic(self.v,self.Ahbi,self.Avbi)
             self.w += self.dt*self.biharmonic(self.w,self.Ahbi,self.Avbi,on_wgrid=True) 
          if hasattr(self,'Khbi') and hasattr(self,'Kvbi'):
             self.b += self.dt*self.biharmonic(self.b,self.Khbi,self.Kvbi)  

          # the explicit friction and mixing increments above also changed the
          # halo points (with wrapped data): refresh them before anything
          # reads a neighbour (the LES closure in set_forcing does)
          self.u = self.apply_bc(self.u)
          self.v = self.apply_bc(self.v)
          self.w = self.apply_vertical_flux_bc( self.apply_bc(self.w) )
          self.b = self.apply_bc(self.b)

          # allow for user defined forcing 
          if hasattr(self,'set_forcing'): self.set_forcing(tau)
          
          # intermediate result with Adams Bashforth time stepping
          f1, f2 = self.dt, self.dt/self.dsqr
          self.u += f1*( tA*self.du[tau,...] + tB*self.du[taum1,...] + tC*self.du[taum2,...])
          self.v += f1*( tA*self.dv[tau,...] + tB*self.dv[taum1,...] + tC*self.dv[taum2,...]) 
          self.w += f2*( tA*self.dw[tau,...] + tB*self.dw[taum1,...] + tC*self.dw[taum2,...]) 
          self.b += f1*( tA*self.db[tau,...] + tB*self.db[taum1,...] + tC*self.db[taum2,...]) 

          # user-defined DIRECT momentum increment, bypassing the AB weights:
          # e.g. u_L += Delta u_S of the Lagrangian-mean (u_L) formulation,
          # which must be a plain forward increment (AB weighting would break
          # the exact telescoping u = u_L - u_S) and must PRECEDE the pressure
          # solve so the projection acts on its divergence; placed before
          # apply_bc so the halo exchange covers the increment.
          if hasattr(self,'set_forcing_direct'): self.set_forcing_direct()

          self.u = self.apply_bc(self.u)
          self.v = self.apply_bc(self.v)  
          self.w = self.apply_vertical_flux_bc( self.apply_bc(self.w) )
          self.b = self.apply_bc(self.b)
          
          self.pressure_solver()          
          return
    
    
      def loop(self,max_steps=1000000):         
          """
          Loop max_steps time over time steps with Adam-Bashforth 3.order.
          """
          tA,tB,tC = 1.,0.,0.        # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          for n in range(max_steps): # time step equations 
            self.time_step(tA,tB,tC,tau,taum1,taum2)  
            # the state is updated in place, so after the call it is at
            # t = (n+1) dt; diagnose gets the number of completed steps
            # and stamps t = (n+1) dt (before Oct 16 2026: n, one dt early)
            self.diagnose(n+1,tau)               
            tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)
            if tA == 1. and tB == 0. and tC == 0:    tA, tB, tC = 1.5,  -0.5, 0. 
            elif tA==1.5 and tB == -0.5 and tC == 0: tA, tB, tC =  23./12. , -16./12., 5./12.
          return    

    
      def diagnose(self,n,tau):  
          """
          Dummy routine for diagnostics. n is the number of completed
          time steps, the state is at t = n dt.
          """
          t =  n*self.dt
          if self.my_pe == 0: print ("t=",t," n=",n)

