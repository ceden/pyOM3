"""
 Setting halo points to boundary conditions 

 at the moment only mpi decomposition

"""
import pyOM3                as OM
from   pyOM3.nh_simple      import setup
from   mpi4py               import MPI
import pyOM3.p3dfft.fortran as FFT
import numpy                as np__


class boundary(setup):    

    
      def __init__(self,parameter = {}, n_pes_x = 1, n_pes_y = 1, 
                                        halo_points = 1, comm_in = None):  
         """
         initialize decomposition and add halo points
         """
         super().__init__(parameter) 

         self.fortran_ordering       = True
         self.p3dfft3_pe_grid_layout = False

         self.comm = MPI.COMM_WORLD if comm_in == None else comm_in
         self.my_pe = self.comm.Get_rank() 
         self.n_pes = self.comm.Get_size() 
         
         FFT.setup(n_pes_x,n_pes_y,self.nx,self.ny,self.nz,self.comm.py2f() )
          
         self.istart,self.iend,self.isize   = FFT.get_dims_physical()
         self.fstart,self.fend,self.fsize   = FFT.get_dims_spectral()          
         self.xs_pe, self.ys_pe, self.zs_pe = self.istart 
         self.xe_pe, self.ye_pe, self.ze_pe = self.iend
         self.n_pes_x, self.n_pes_y, self.n_pes_z = 1, n_pes_x, n_pes_y 
         self.my_blk_x = 1
         self.my_blk_y = np__.mod(self.my_pe,self.n_pes_y) + 1
         self.my_blk_z = np__.mod(self.my_pe//self.n_pes_y,self.n_pes_z) + 1
          
         self.show_pe_layout()
          
         # calculate local shape with halo points -> Nx,Ny,..
         # input nx, ny, nz is global shape without halo points 
          
         self.halo = halo_points              # number of halo points at each side     
         self.Nx, self.Ny, self.Nz = self.isize[0] + 2*self.halo, \
                                     self.isize[1] + 2*self.halo, \
                                     self.isize[2] + 2*self.halo   
         self.Lx, self.Ly, self.Lz = self.nx*self.dx, self.ny*self.dy, self.nz*self.dz 
         return

    
      def apply_vertical_flux_bc(self,a, flux_on_wgrid = False): 
          """
          set vertical fluxes at top and bottom rigid lid to zero.
          for on_wgrid, zero out two uppermost points 
          """
          if not self.periodic_in_z:  
             if flux_on_wgrid: 
               sl_s, sl_e = (slice(0,self.halo,None),), (slice(-(self.halo+2),-1,None),)    
             else: 
                sl_s, sl_e = (slice(0,self.halo,None),), (slice(-(self.halo+1),-1,None),)  
             if self.fortran_ordering:
               sl_s, sl_e = 2*(slice(None),) + sl_s, 2*(slice(None),) + sl_e 
             else:    
               sl_s, sl_e = sl_s + 2*(slice(None),), sl_e + 2*(slice(None),) 
                 
             if self.my_blk_z == 1           : a = OM.modify_array(a,sl_s,0)
             if self.my_blk_z == self.n_pes_z: a = OM.modify_array(a,sl_e,0)
          return a

    
      def apply_bc(self,a,bc_value = 0): 
          """
          apply horizontally periodic or closed boundary condition to halo points
          works for 2D (horizontal) and 3D arrays
          """              
          my_pe, n_pes_x, n_pes_y, n_pes_z = self.my_pe, self.n_pes_x, self.n_pes_y, self.n_pes_z
          my_blk_x, my_blk_y, my_blk_z     = self.my_blk_x, self.my_blk_y, self.my_blk_z
          x = slice(None) 
          y = slice(None) 
          z = slice(None) if len(a.shape) > 2 else None  
                    
          if n_pes_x > 1: 
              
             if self.p3dfft3_pe_grid_layout:
               west, east = my_pe, my_pe   #  x dim is local anyways  
             else:    
               west = my_pe + n_pes_x - 1   if my_blk_x == 1        else my_pe - 1
               east = my_pe - (n_pes_x - 1) if my_blk_x == n_pes_x  else my_pe + 1

             if self.fortran_ordering:
                 buf = OM.np.empty( (self.Ny, self.Nz if len(a.shape)==3 else 1), OM.prec)
             else: 
                 buf = OM.np.empty( (self.Nz if len(a.shape)==3 else 1, self.Ny), OM.prec)   

             for i in range(self.halo): 
               # halo+i,'->',i-halo  for halo=1: 1 -> -1  
               sl_send = ( self.halo+i,y,z) if self.fortran_ordering else (z,y, self.halo+i) 
               sl_recv = ( i-self.halo,y,z) if self.fortran_ordering else (z,y, i-self.halo)   
               if my_blk_x !=1 :       self.send(  a[sl_send]  , west, 49)               
               if my_blk_x != n_pes_x: a = OM.modify_array(a, sl_recv,
                                                           self.recv(buf, east, 49)) 
               if my_blk_x ==1 :       self.send(  a[sl_send]  , west, 47)               
               if my_blk_x == n_pes_x: a = OM.modify_array(a, sl_recv,
                                                           self.recv(buf, east, 47)) 

               # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0
               sl_send = (-self.halo-1-i,y,z) if self.fortran_ordering else (z,y,-self.halo-1-i) 
               sl_recv = ( self.halo-1-i,y,z) if self.fortran_ordering else (z,y, self.halo-1-i)   
               if my_blk_x != n_pes_x: self.send( a[sl_send] , east, 153)                
               if my_blk_x !=1 :       a = OM.modify_array(a, sl_recv,
                                                           self.recv(buf, west, 153))  
               if my_blk_x == n_pes_x: self.send( a[sl_send] , east, 152)                
               if my_blk_x ==1 :       a = OM.modify_array(a, sl_recv,
                                                           self.recv(buf, west, 152))  
          
          else:
             for i in range(self.halo): 
                # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0 
                sl_send = (-self.halo-1-i,y,z) if self.fortran_ordering \
                                               else (z,y,-self.halo-1-i) 
                sl_recv = ( self.halo-1-i,y,z) if self.fortran_ordering \
                                               else (z,y, self.halo-1-i) 
                a = OM.modify_array(a, sl_recv,  a[sl_send]  ) 
                # halo+i,'->',i-halo  for halo=1: 1 -> -1   
                sl_send = ( self.halo+i,y,z) if self.fortran_ordering else (z,y, self.halo+i) 
                sl_recv = ( i-self.halo,y,z) if self.fortran_ordering else (z,y, i-self.halo) 
                a = OM.modify_array(a, sl_recv,  a[sl_send]  )             
                                    
          if n_pes_y > 1:  
              
             if self.p3dfft3_pe_grid_layout:
               south = my_pe + n_pes_z*(n_pes_y-1) if my_blk_y == 1       else my_pe - n_pes_z  
               north = my_pe - n_pes_z*(n_pes_y-1) if my_blk_y == n_pes_y else my_pe + n_pes_z
             else:    
               south = my_pe + n_pes_x*(n_pes_y-1) if my_blk_y == 1       else my_pe - n_pes_x  
               north = my_pe - n_pes_x*(n_pes_y-1) if my_blk_y == n_pes_y else my_pe + n_pes_x
 
             if self.fortran_ordering: 
                buf = OM.np.empty( (self.Nx,self.Nz if len(a.shape)==3 else 1), OM.prec)   
             else:    
                buf = OM.np.empty( (self.Nz if len(a.shape)==3 else 1,self.Nx), OM.prec) 
                 
             for i in range(self.halo):
                 # halo+i,'->',i-halo  for halo=1: 1 -> -1  
                 sl_send = (x,self.halo+i,z) if self.fortran_ordering else (z,self.halo+i,x) 
                 sl_recv = (x,i-self.halo,z) if self.fortran_ordering else (z,i-self.halo,x) 
                 if my_blk_y !=1 :       self.send(  a[sl_send] ,south, 49)  
                 if my_blk_y != n_pes_y: a = OM.modify_array(a, sl_recv,
                                                             self.recv(buf,north, 49) )
                 if my_blk_y ==1 :       self.send(  a[sl_send] ,south, 47)  
                 if my_blk_y == n_pes_y: a = OM.modify_array(a, sl_recv,
                                                             self.recv(buf,north, 47) )

                 # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0
                 sl_send = (x,-self.halo-1-i,z) if self.fortran_ordering \
                                                else (z,-self.halo-1-i,x) 
                 sl_recv = (x, self.halo-1-i,z) if self.fortran_ordering \
                                                else (z, self.halo-1-i,x) 
                 if my_blk_y != n_pes_y: self.send( a[sl_send], north, 153)             
                 if my_blk_y !=1 :       a = OM.modify_array(a, sl_recv,
                                                             self.recv(buf, south, 153) ) 
                 if my_blk_y == n_pes_y: self.send( a[sl_send], north, 152)
                 if my_blk_y ==1 :       a = OM.modify_array(a, sl_recv,
                                                             self.recv(buf, south, 152) ) 
                 
          else: 
              
             for i in range(self.halo):              
                 # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0 
                 sl_send = (x,-self.halo-1-i,z) if self.fortran_ordering \
                                                else (z,-self.halo-1-i,x) 
                 sl_recv = (x, self.halo-1-i,z) if self.fortran_ordering \
                                                else (z, self.halo-1-i,x)
                 a = OM.modify_array(a, sl_recv , a[sl_send] )
                 # halo+i,'->',i-halo  for halo=1: 1 -> -1   
                 sl_send = (x, self.halo+i,z) if self.fortran_ordering else (z,self.halo+i,x) 
                 sl_recv = (x, i-self.halo,z) if self.fortran_ordering else (z,i-self.halo,x) 
                 a = OM.modify_array(a, sl_recv , a[sl_send] )
          
          if n_pes_z > 1 and len(a.shape) > 2 :   
              
             if self.p3dfft3_pe_grid_layout:
                down = my_pe + (n_pes_z-1) if my_blk_z==1       else  my_pe - 1 
                up   = my_pe - (n_pes_z-1) if my_blk_z==n_pes_z else  my_pe + 1
             else:   
                down = my_pe + n_pes_y*n_pes_x*(n_pes_z-1) if my_blk_z==1    \
                                                           else my_pe - n_pes_x*n_pes_y    
                up   = my_pe - n_pes_y*n_pes_x*(n_pes_z-1) if my_blk_z==n_pes_z \
                                                           else my_pe + n_pes_x*n_pes_y
             if self.fortran_ordering: 
                buf = OM.np.empty( (self.Nx,self.Ny), OM.prec)   
             else:    
                buf = OM.np.empty( (self.Ny,self.Nx), OM.prec) 
                 
             for i in range(self.halo): 
                # halo+i,'->',i-halo  for halo=1: 1 -> -1  
                sl_send = (x,y,self.halo+i) if self.fortran_ordering else (self.halo+i,y,x) 
                sl_recv = (x,y,i-self.halo) if self.fortran_ordering else (i-self.halo,y,x)
                if my_blk_z !=1 :       self.send(  a[sl_send] ,down, 49)  
                if my_blk_z != n_pes_z: a = OM.modify_array(a, sl_recv,
                                                            self.recv(buf,up, 49) )
                if self.periodic_in_z:    
                   if my_blk_z ==1 :       self.send(  a[sl_send] ,down, 47)  
                   if my_blk_z == n_pes_z: a = OM.modify_array(a, sl_recv,
                                                               self.recv(buf,up, 47) )
                else:       
                   if my_blk_z == n_pes_z: a = OM.modify_array(a, sl_recv,  bc_value ) 
                 
                # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0
                sl_send = (x,y,-self.halo-1-i) if self.fortran_ordering \
                                               else (-self.halo-1-i,y,x) 
                sl_recv = (x,y, self.halo-1-i) if self.fortran_ordering \
                                               else ( self.halo-1-i,y,x)
                if my_blk_z != n_pes_z: self.send( a[sl_send], up, 153)             
                if my_blk_z !=1 :       a = OM.modify_array(a, sl_recv, 
                                                            self.recv(buf, down, 153) ) 
                if self.periodic_in_z:                   
                   if my_blk_z == n_pes_z: self.send( a[sl_send], up, 152)
                   if my_blk_z ==1 :       a = OM.modify_array(a, sl_recv,
                                                               self.recv(buf,down,152) ) 
                else:  
                   if my_blk_z ==1 :       a = OM.modify_array(a, sl_recv,  bc_value )  
             
          elif  n_pes_z == 1 and len(a.shape) > 2 :    
              
             for i in range(self.halo):               
               if self.periodic_in_z: 
                 # -halo-1-i,'->',halo-1-i) for halo=1: -2 -> 0 
                 sl_send = (x,y,-self.halo-1-i) if self.fortran_ordering \
                                                else (-self.halo-1-i,y,x) 
                 sl_recv = (x,y, self.halo-1-i) if self.fortran_ordering \
                                                else ( self.halo-1-i,y,x)
                 a = OM.modify_array(a, sl_recv,  a[sl_send]  )
                 # halo+i,'->',i-halo  for halo=1: 1 -> -1   
                 sl_send = (x,y, self.halo+i) if self.fortran_ordering else (self.halo+i,y,x) 
                 sl_recv = (x,y, i-self.halo) if self.fortran_ordering else (i-self.halo,y,x)
                 a = OM.modify_array(a, sl_recv,  a[sl_send]  )
               else:   
                 sl_recv = (x,y, self.halo-1-i) if self.fortran_ordering \
                                                else ( self.halo-1-i,y,x)  
                 a = OM.modify_array(a, sl_recv ,  bc_value )
                 sl_recv = (x,y, i-self.halo) if self.fortran_ordering \
                                              else (i-self.halo,y,x)  
                 a = OM.modify_array(a, sl_recv ,  bc_value )                            
          return a



      def gather_array_mpi(self,a):  
          """
          returns array of complete domain with shape (nx,ny,nz) or
          (nz,ny,nx) gathered by PE0 
          works for 2D and 3D distributed domains
          """
          
          xs_pe, xe_pe, ys_pe, ye_pe = self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe
          nx, ny, nz, zs_pe, ze_pe   = self.nx, self.ny, self.nz, self.zs_pe, self.ze_pe
          
          # transfer local domain
          
          if self.fortran_ordering:
              sl_return =  (nx, ny, nz if len(a.shape)==3 else 1) 
              sl_recv = (slice(xs_pe-1,xe_pe), slice(ys_pe-1,ye_pe), 
                         slice(zs_pe-1,ze_pe) if len(a.shape)==3 else None)
              sl_send = (slice(self.halo,-self.halo), slice(self.halo,-self.halo), 
                         slice(self.halo,-self.halo) if len(a.shape) == 3 else None) 
          else:                     
              sl_return  = (nz if len(a.shape)==3 else 1, ny, nx)
              sl_recv = (slice(zs_pe-1,ze_pe) if len(a.shape)==3 else None, 
                         slice(ys_pe-1,ye_pe), slice(xs_pe-1,xe_pe))
              sl_send = (slice(self.halo,-self.halo) if len(a.shape) == 3 else None, 
                         slice(self.halo,-self.halo), slice(self.halo,-self.halo) ) 
              
          if self.my_pe==0:       
              b = OM.np.empty( sl_return, OM.prec)  
              b = OM.modify_array(b, sl_recv, a[sl_send] )  
          else:
             b = None

          # send and transfer non-local domains
          for pe in range(1,self.n_pes):       
              if self.my_pe == pe:   
                 dims = [xs_pe, xe_pe, ys_pe, ye_pe, zs_pe, ze_pe] 
                 self.comm.Send( [OM.np.array( dims, dtype='i'),MPI.INT],0,49) 
                 self.send( a[sl_send], 0, 153)                 
              if self.my_pe == 0:        
                 dims = OM.np.zeros( (6,), dtype='i')  
                 self.comm.Recv([dims, MPI.INT], source=pe, tag=49) 
                 xs, xe, ys, ye, zs, ze = dims                  
                 if self.fortran_ordering: 
                    sl_recv = (slice(xs-1,xe), slice(ys-1,ye),
                               slice(zs-1,ze) if len(a.shape)==3 else None)  
                 else: 
                    sl_recv = (slice(zs-1,ze) if len(a.shape)==3 else None,
                               slice(ys-1,ye), slice(xs-1,xe))
                     
                 buf = b[sl_recv].copy()  
                 b = OM.modify_array(b, sl_recv, self.recv(buf, pe,  153))    
          return b



          
    
      def show_pe_layout(self):
         """
         print pe layout 
         """
         if self.my_pe==0: 
            print('global domain is nx=',self.nx,'ny=',self.ny,'nz=',self.nz)
            print('using',self.n_pes,'PEs for domain decomposition')
            print('fortran_ordering = ',self.fortran_ordering)
         self.comm.Barrier()  
          
         if self.my_pe==0: print('Layout of processor mesh ')           
         for pe in range(self.n_pes):
             if pe == self.my_pe: 
                  print('PE#',pe,': my_blk_x = ',self.my_blk_x,
                                  ' my_blk_y = ',self.my_blk_y,
                                  ' my_blk_z = ',self.my_blk_z)   
             self.comm.Barrier()  
              
         if self.my_pe==0: print('Layout of physical domain decomposition ')
         for pe in range(self.n_pes):
             if pe == self.my_pe: 
                print('PE#',self.my_pe,': i=',self.xs_pe,':',self.xe_pe, 
                                        ' j=',self.ys_pe,':',self.ye_pe,
                                        ' k=',self.zs_pe,':',self.ze_pe )
             self.comm.Barrier()  
             
         if self.my_pe==0: print('Layout of spectral domain decomposition ')
         for pe in range(self.n_pes):
             if pe == self.my_pe: 
                print('PE#',self.my_pe,': i=',self.fstart[0],':',self.fend[0], 
                                        ' j=',self.fstart[1],':',self.fend[1],
                                        ' k=',self.fstart[2],':',self.fend[2] ) 
             self.comm.Barrier()    
         return   
    
    
      def rollx(self,a,shift):
          """
          Shift last or first (x) axis in array a by "shift" points
          Note: shift=-1 shifts in positive direction.
          """
          if self.fortran_ordering:
             return OM.np.roll(a, shift=shift,axis= 0)  
          else:    
             return OM.np.roll(a, shift=shift,axis=-1)
      
      def rolly(self,a,shift):
          """
          Shift 2. last or second (y) axis in array a by "shift" points
          Note: shift=-1 shifts in positive direction.
          """
          if self.fortran_ordering:
             return OM.np.roll(a, shift=shift,axis= 1)  
          else:    
             return OM.np.roll(a, shift=shift,axis=-2) 
        
      def rollz(self,a,shift):
          """
          Shift 3. last or third (z) axis in array a by "shift" points
          Note: shift=-1 shifts in positive direction.
          """
          if self.fortran_ordering:
             return OM.np.roll(a, shift=shift,axis= 2)  
          else:    
             return OM.np.roll(a, shift=shift,axis=-3) 


      def unpad(self,a):
          s = slice(self.halo,-self.halo)
          sl = (s,s,s) if len(a.shape) > 2 else (s,s)
          return a[sl]
          
      def send(self,buf, dest, tag):
          self.comm.Send(  OM.np.ascontiguousarray(buf) , dest = dest, tag = tag )

      def recv(self,buf, source, tag):
          self.comm.Recv(buf,source = source, tag = tag)
          return buf
  
    