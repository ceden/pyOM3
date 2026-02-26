"""
 Domain decomposition using MPI libray and mpi4py.
 Based on basic class setup.

 domain_decomposition_mpi generates the mesh.
 
"""

import pyOM3           as OM
from   pyOM3.pe_simple import setup
from   functools       import partial
import numpy           as np_


class decomposition_mpi(setup):

    
      def domain_decomposition_mpi(self,n_pes_x,n_pes_y):
          """
           Horizontal domain decomposition with MPI.
           Input is number of processors in x,y: n_pes_x,n_pes_y.
           Defines what's needed to deal with MPI.
          """
          
          self.my_pe , self.n_pes = OM.my_pe, OM.n_pes
          # null space
          self.mesh = None
          self.grid_2D, self.grid_3D, self.grid_4D = None , None, None
          self.sharding_2D, self.sharding_3D, self.sharding_4D = None, None, None

          if self.n_pes != n_pes_x * n_pes_y:
             print("Error: number of available PEs not equal to input n_pes_x x n_pes_y")
             raise IOError 

          if self.n_pes == 1:
             self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe  =  1, self.Nx, 1, self.Ny
             self.n_pes_x, self.n_pes_y =  1,1
             return 
        
          self.my_blk_x = np_.mod(self.my_pe,         n_pes_x)+1      # pe rank in x direction, starts with 1
          self.my_blk_y = np_.mod(self.my_pe//n_pes_x,n_pes_y)+1      # pe rank in y direction, ends with n_pes_y
        
          for pe in range(1):#self.n_pes):
              if pe == self.my_pe: print('PE#',pe,': my_blk_x = ',self.my_blk_x,' my_blk_y = ',self.my_blk_y)
                
          nx, ny = self.Nx - 2*self.halo, self.Ny - 2*self.halo  # actual grid points of total domain without boundary cells
          x_blk, y_blk  = (nx-1)//n_pes_x + 1 ,  (ny-1)//n_pes_y + 1 # grid points of each block
          # indices from fortran versions, also useful here
          xs_pe, xe_pe = (self.my_blk_x-1)*x_blk + self.halo , np_.minimum(self.my_blk_x*x_blk,nx) 
          ys_pe, ye_pe = (self.my_blk_y-1)*y_blk + self.halo , np_.minimum(self.my_blk_y*y_blk,ny)
          x_blk, y_blk = xe_pe-xs_pe+1 , ye_pe-ys_pe+1        #last block might have been truncated    
                        
          for pe in range(1):#self.n_pes):
              if pe == self.my_pe: print('PE#',pe,': i=',xs_pe,':',xe_pe,' j=',ys_pe,':',ye_pe)
        
          # check for incorrect domain decomposition
          if self.my_blk_y==n_pes_y and ys_pe>ye_pe:
             print(' domain decompositon impossible in j-direction')
             print(' choose other number of PEs in j-direction')
             raise IOError
      
          if self.my_blk_x==n_pes_x and xs_pe>xe_pe:     
             print(' domain decompositon impossible in i-direction')
             print(' choose other number of PEs in i-direction')
             raise IOError

          self.xs_pe, self.xe_pe, self.ys_pe, self.ye_pe  = xs_pe, xe_pe, ys_pe, ye_pe
          self.n_pes_x, self.n_pes_y = n_pes_x, n_pes_y
          self.Nx, self.Ny = x_blk + 2*self.halo, y_blk + 2*self.halo # add boundary cells   
          return

  
      def get_full_domain_extent_mpi(self):  
          """
           returns full horizontal extent as integers using MPI decomposition
          """ 
          nx,ny = self.Nx-2*self.halo,self.Ny-2*self.halo
          buf = OM.np.empty( (1,), OM.prec)
          for pe in range( 1, self.n_pes_x ):        
              if self.my_pe == pe:             OM.send( OM.np.array(self.Nx-2*self.halo,OM.prec),  0 , 49)
              if self.my_pe == 0 :   nx += int(  OM.recv(buf,  pe, 49)[0] )     
          for pe in range( self.n_pes_x , self.n_pes_y * self.n_pes_x , self.n_pes_x ):        
              if self.my_pe == pe:             OM.send( OM.np.array(self.Ny-2*self.halo,OM.prec) , 0   , 49 )
              if self.my_pe == 0 :   ny += int( OM.recv( buf, pe,  49 )[0] )     
          return  OM.pe0_bcast(nx), OM.pe0_bcast(ny)       
             
