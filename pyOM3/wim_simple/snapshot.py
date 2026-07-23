"""
Snapshot diagnostics, based on class model. 

class snap_nc4 is based on the netCDF4 module.

"""
import pyOM3           as OM
from   pyOM3.wim_simple.model import model
from   netCDF4         import Dataset as NF
import numpy           as np__


class snap_nc4(model):

    
      def setup(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          """
          Initialisation of the netcdf file.
          """
          if hasattr(super(),'setup'): super().setup()
          
          self.snap_file_name      = snap_file_name
          self.show_each_time_step = show_each_time_step
          
          if self.my_pe == 0:
             id = NF(snap_file_name,mode='w')    
             id.createDimension('x',self.nx)
             id.createDimension('y',self.ny)
             id.createDimension('d',self.nd) 
             id.createDimension('f',self.nf)  
             id.createDimension('time',None)

             x,y = (0.5+np__.arange(self.nx))*self.dx, (0.5+np__.arange(self.ny))*self.dy 
              
             id.createVariable('time', OM.prec ,('time',) )
             id.createVariable('x',   OM.prec ,('x',) )[:] = x
             id.createVariable('y',   OM.prec ,('y',) )[:] = y
             id.createVariable('f',   OM.prec ,('f',) )[:] = self.ft[self.halo:-self.halo] 
             id.createVariable('d',   OM.prec ,('d',) )[:] = self.theta[self.halo:-self.halo] 
             id.createVariable('E_mean', OM.prec ,('time','y','x') )
             id.createVariable('om_mean', OM.prec ,('time','y','x') )
             id.createVariable('E', OM.prec ,('time','d','f','y','x') ) 
             id.createVariable('xdot', OM.prec ,('time','d','f','y','x') )
             id.createVariable('ydot', OM.prec ,('time','d','f','y','x') )
             id.createVariable('fdot', OM.prec ,('time','d','f','y','x') )
             id.createVariable('ddot', OM.prec ,('time','d','f','y','x') )
             id.close()    
          return


      def diagnose(self,n,tau):  
          """
          Writing a snapshot into the netcdf file.
          """
          def _show_time(t):
              if     t <=86400.: return "%6.3f s"%t
              else:   return "%8.1f d"%(t/86400.)

          def _print_text():  print ("t=",_show_time(t)," n=",n)

                  
          t = n*self.dt 
          if self.my_pe==0 and self.show_each_time_step: _print_text()
                     
          if n==0 or np__.mod(n,int(self.snapint/self.dt))==0. : # writing to file
             if self.my_pe == 0: 
                 if not self.show_each_time_step: _print_text()
                 print(" --> writing to file ",self.snap_file_name)
             write = True 
          else: write = False

          if write: 
             
             self.comm.Barrier()  
             if self.my_pe == 0:
                fid = NF(self.snap_file_name,mode='r+')
                m = len(fid.dimensions['time'])
                fid.variables['time'][m] = t
                fid.close() 
             else: m=0
                 
             m = self.comm.bcast(m,root=0)
              
             for pe in range(self.n_pes): 
                 
                 self.comm.Barrier()  

                 sl = slice(self.halo,-self.halo)
                 sl1 = (m,slice(self.ys_pe-1,self.ye_pe), 
                          slice(self.xs_pe-1,self.xe_pe))    
                 sl2 = (m,slice(None),slice(None),slice(self.ys_pe-1,self.ye_pe), 
                          slice(self.xs_pe-1,self.xe_pe))  
                 if self.my_pe == pe:
                    fid = NF(self.snap_file_name,mode='r+')
                    fid.variables['E_mean'][sl1]  = self.e_mean[sl,sl] 
                    fid.variables['om_mean'][sl1] = self.om_mean[sl,sl] 
                    fid.variables['E'][sl2]       = self.E[sl,sl,sl,sl]
                    fid.variables['xdot'][sl2]    = self.xdot[sl,sl,sl,sl]
                    fid.variables['ydot'][sl2]    = self.ydot[sl,sl,sl,sl]
                    fid.variables['fdot'][sl2]    = self.fdot[sl,sl,sl,sl]
                    fid.variables['ddot'][sl2]    = self.ddot[sl,sl,sl,sl]
                    fid.close() 
                 
       