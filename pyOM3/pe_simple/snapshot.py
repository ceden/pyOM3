"""
Snapshot diagnostics, based on class model. 

class snap_nc4 is based on the netCDF4 module.
Works for MPI domain decomposition, and single processor JAX,
but not for multihost JAX environment.

class snap_raw writes npy files per processor and time step in a directory.
Works for MPI domain decomposition and single/multihost JAX,
Can be converted later to e.g. zarr format. 

"""
import pyOM3                 as OM
from   pyOM3.pe_simple.model import model
from   netCDF4               import Dataset as NF
import numpy                 as np__
import os,shutil


class snap_nc4(model):

      def _snap_coords(self, nx, ny, nz):
          """coordinates written to the snapshot file: (xt,yt,zt,xu,yu,zu),
          cell centres / faces in metres on the uniform grid; pe_realistic
          overrides with the grid coordinates of grid.calc_grid. nx,ny,nz
          are the global extents (get_full_domain_extent is collective, so
          it is called by the caller on all PEs)."""
          return ((0.5+np__.arange(nx))*self.dx, (0.5+np__.arange(ny))*self.dy,
                  (0.5+np__.arange(nz))*self.dz, (1.0+np__.arange(nx))*self.dx,
                  (1.0+np__.arange(ny))*self.dy, (1.0+np__.arange(nz))*self.dz)

      def prepare_for_run(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          """
          Initialisation of the netcdf file. The body lives in the
          module-level _snap_nc4_prepare so pe_realistic can share it: a
          zero-argument super() inside a shared method binds to the class
          it was defined in, so this chaining line must stay per package.
          """
          super().prepare_for_run()
          return _snap_nc4_prepare(self, snap_file_name, show_each_time_step)

    
      def diagnose(self,n,tau):  
          """
          Writing a snapshot into the netcdf file.
          Works for MPI domain decomposition, and single processor JAX,
          but not for multihost JAX environment.
          """
          def show_time(t):
              if     t <=86400.: return "%6.3f s"%t
              else:   return "%8.1f d"%(t/86400.)

          def print_text():
              if hasattr(self,'congr_itts'):     print ("t=",show_time(t),"n=",n,
                                                       "itts=",self.congr_itts,"error=%3.2e"%self.estimated_error)
              else:                              print ("t=",show_time(t)," n=",n)

          t = n*self.dt 
          if self.my_pe==0 and self.show_each_time_step: print_text()
             
          if hasattr(self,'congr_itts'): self.tot_congr_itts += self.congr_itts
          
          if n==0 or np__.mod(n,int(self.snapint/self.dt))==0. : # writing to file
             if self.my_pe == 0: 
                 if not self.show_each_time_step: print_text()
                 print(" --> writing to file ",self.snap_file_name)
                 if hasattr(self,'congr_itts'):  print(" total itts in solver ",self.tot_congr_itts)
             write = True 
             self.tot_congr_itts = 0 
          else: write = False

          if write: 
             
             self.mpi_barrier()  
             if self.my_pe == 0:
                fid = NF(self.snap_file_name,mode='r+')
                m = len(fid.dimensions['time'])
                fid.variables['time'][m] = t
                fid.close() 
             else: m=0
                 
             m = self.pe0_bcast(m)   
                           
             for pe in range(self.n_pes): 
                 
                 self.mpi_barrier()  
                 
                 if self.my_pe == pe:
                    fid = NF(self.snap_file_name,mode='r+')
                    fid.variables['u'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.u)
                    fid.variables['v'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.v)
                    fid.variables['w'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.w)
                    fid.variables['temp'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.temp) 
                    if self.eq_of_state > 0 and self.eq_of_state != 100:  
                       fid.variables['salt'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                             slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.salt) 
                    fid.variables['p_h'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.p_h) 
                    fid.variables['rho'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe), 
                                                          slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.rho)
                    
                    if self.explicit_free_surface: 
                       fid.variables['p_s'][(m,slice(self.ys_pe-1,self.ye_pe), 
                                               slice(self.xs_pe-1,self.xe_pe))] = self.unpad( self.p_s)                     
                   
                    else:                 
                       fid.variables['p_s'][(m,slice(self.ys_pe-1,self.ye_pe), 
                                               slice(self.xs_pe-1,self.xe_pe))] = self.unpad( self.p_s[tau,:,:])                      
                    
                    fid.close() 
             if self.my_pe == 0:   print(" --> done") 


class snap_raw(model):

      _snap_coords = snap_nc4._snap_coords

      def prepare_for_run(self, snap_file_name = 'raw_snap', show_each_time_step = True ):
          """
          Initialise the output directory and write grid files (body in the
          module-level _snap_raw_prepare, shared with pe_realistic).
          """
          super().prepare_for_run()
          return _snap_raw_prepare(self, snap_file_name, show_each_time_step)


      def diagnose(self,n,tau):  
          """
          Writing a snapshot per process and time step into the directory system.
          Works for MPI domain decomposition and single/multihost JAX environment.
          """
          def show_time(t):
              if     t <=86400.: return "%6.3f s"%t
              else:   return "%8.1f d"%(t/86400.)

          def print_text():
              if hasattr(self,'congr_itts'):     print ("t=",show_time(t),"n=",n,
                                                       "itts=",self.congr_itts,"error=%3.2e"%self.estimated_error)
              else:                              print ("t=",show_time(t)," n=",n)

          t =  n*self.dt     
          if self.my_pe==0 and self.show_each_time_step: print_text()
          if hasattr(self,'congr_itts'): self.tot_congr_itts += self.congr_itts
              
          if n==0 or np__.mod(n,int(self.snapint/self.dt))==0.: # writing to file
              
             if self.my_pe==0 and not self.show_each_time_step: print_text() 
             if self.my_pe==0: 
                  print(" --> writing to raw output to ",self.snap_file_name)  
                  if hasattr(self,'congr_itts'):  print(" total itts in solver ",self.tot_congr_itts)
             self.tot_congr_itts = 0
              
             d =  self.snap_file_name + '/pe=%i/n=%i/'%(self.my_pe,n)  
             os.mkdir(d)   
             OM.np.save(d+'t.npy', t) 
             for var in ('u','v','w','temp','salt','p_h','p_s'):                                   
                    data = getattr(self,var)
                    if var == 'p_s' and not self.explicit_free_surface: data = self.p_s[tau,:,:] 
                    # JAX: the global interior from ONE process (all its devices,
                    # decomposition [1,1] below); several processes (multihost)
                    # write their own shard each, as the MPI ranks do
                    if hasattr(data,'addressable_shards') and OM.jax.process_count() > 1:
                          OM.np.save(d+'%s.npy'%var,   self.unpad(data).addressable_shards[0].data) 
                    else:    
                          OM.np.save(d+'%s.npy'%var, np__.asarray(self.unpad(data)))


 


def _snap_nc4_prepare(self, snap_file_name, show_each_time_step):
    """netCDF body of snap_nc4.prepare_for_run, shared with pe_realistic"""
    
    nx,ny = self.get_full_domain_extent()
    nz = self.Nz - 2*self.halo
    self.snap_file_name = snap_file_name
    self.show_each_time_step = show_each_time_step
    self.tot_congr_itts = 0
    
    if self.my_pe == 0:
       id = NF(snap_file_name,mode='w')    
       id.createDimension('xt',nx)
       id.createDimension('yt',ny)
       id.createDimension('zt',nz)
       id.createDimension('xu',nx)
       id.createDimension('yu',ny)
       id.createDimension('zu',nz)
       id.createDimension('time',None)
    
       id.createVariable('time', OM.prec ,('time',) )
       xt,yt,zt,xu,yu,zu = self._snap_coords(nx,ny,nz)
       id.createVariable('xt',   OM.prec ,('xt',) )[:] = np__.asarray(xt)
       id.createVariable('yt',   OM.prec ,('yt',) )[:] = np__.asarray(yt)
       id.createVariable('zt',   OM.prec ,('zt',) )[:] = np__.asarray(zt)
       id.createVariable('xu',   OM.prec ,('xu',) )[:] = np__.asarray(xu)
       id.createVariable('yu',   OM.prec ,('yu',) )[:] = np__.asarray(yu)
       id.createVariable('zu',   OM.prec ,('zu',) )[:] = np__.asarray(zu)
    
       id.createVariable('temp', OM.prec ,('time','zt','yt','xt') )
       if self.eq_of_state > 0 and self.eq_of_state != 100:  
          id.createVariable('salt', OM.prec ,('time','zt','yt','xt') )
       id.createVariable('u',    OM.prec ,('time','zt','yt','xu') )
       id.createVariable('v',    OM.prec ,('time','zt','yu','xt') )
       id.createVariable('w',    OM.prec ,('time','zu','yt','xt') )
       id.createVariable('p_h',  OM.prec ,('time','zt','yt','xt') )
       id.createVariable('rho',  OM.prec ,('time','zt','yt','xt') )  
       id.createVariable('p_s',  OM.prec ,('time','yt','xt') )  
       id.close()    
    return

    


def _snap_raw_prepare(self, snap_file_name, show_each_time_step):
    """directory body of snap_raw.prepare_for_run, shared with pe_realistic"""
    nx,ny = self.get_full_domain_extent()
    nz = self.Nz - 2*self.halo     
    self.snap_file_name = snap_file_name
    self.show_each_time_step = show_each_time_step
    self.tot_congr_itts = 0
    
    if self.my_pe == 0:             
       if os.path.isdir(snap_file_name):  shutil.rmtree(snap_file_name, ignore_errors=False)
       os.mkdir(snap_file_name)              
       # JAX in one process: one block, the global interior (see diagnose);
       # several processes (multihost) or MPI: one block per process/rank
       single = OM.jax and OM.jax.process_count() == 1
       for pe in range(1 if single else self.n_pes_x*self.n_pes_y): os.mkdir(snap_file_name+'/pe=%i'%pe)  
       for name, c in zip(('xt','yt','zt','xu','yu','zu'), self._snap_coords(nx,ny,nz)):
           OM.np.save(snap_file_name+'/%s.npy'%name, OM.np.asarray(c))
       OM.np.save(snap_file_name+'/decomposition.npy',  OM.np.array([1,1] if single else [self.n_pes_x,self.n_pes_y]) )   
    return
