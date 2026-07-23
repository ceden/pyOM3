"""
 main class of model defining/allocating model variables 

 Based on class boundary dealing with domain decomposition
""" 
import pyOM3                     as OM
from   pyOM3.nh_simple.boundary  import boundary


class main(boundary):

    
      def setup(self):  
          """
          setup everything for the model run
          """
          self.allocate_workspace()   
          self.setup_wavenumber_grid()
          return
   

      def allocate_workspace(self):
          """
          define all model variables 
          """

          Nx, Ny, Nz = self.Nx, self.Ny, self.Nz 

          # allocate model variables with halo points
          if self.fortran_ordering:
             buf_3D = OM.np.zeros(  (Nx,Ny,Nz), OM.prec) 
             buf_4D = OM.np.zeros(  (3,Nx,Ny,Nz), OM.prec)  
          else:   
             buf_3D = OM.np.zeros(  (Nz,Ny,Nx), OM.prec) 
             buf_4D = OM.np.zeros(  (3,Nz,Ny,Nx), OM.prec)  
              
          self.b   = buf_3D.copy() 
          self.u   = buf_3D.copy()
          self.v   = buf_3D.copy()
          self.w   = buf_3D.copy()
          self.p   = buf_3D.copy()
          self.db   = buf_4D.copy() 
          self.du   = buf_4D.copy()
          self.dv   = buf_4D.copy()
          self.dw   = buf_4D.copy()
          
          return



      def setup_wavenumber_grid(self):
          """
          wavenumber related stuff
          """
          def hat_ksqr(k,dx):
              return 2.*(1.- OM.np.cos(k*dx))/dx**2

          pi = OM.np.pi
          kx = OM.np.zeros( (self.nx//2+1,) , OM.prec)  
          ky = OM.np.zeros( (self.ny     ,) , OM.prec)
          kz = OM.np.zeros( (self.nz     ,) , OM.prec)
          
          for i in range(self.nx//2+1):  kx[i] = 2*i*pi/self.Lx              
          for j in range( (self.ny+1)//2    ):      ky[j] =  2*j          *pi/self.Ly
          for j in range( (self.ny+1)//2,self.ny ): ky[j] = -2*(self.ny-j)*pi/self.Ly
          for k in range( (self.nz+1)//2    ):      kz[k] =  2*k          *pi/self.Lz
          for k in range( (self.nz+1)//2,self.nz ): kz[k] = -2*(self.nz-k)*pi/self.Lz

          x, y = hat_ksqr(kx,self.dx), hat_ksqr(ky,self.dy)
          z    = hat_ksqr(kz,self.dz) if self.periodic_in_z else  OM.np.zeros( (self.nz,))    
          ksqr = x[None,None,:] + y[None,:,None] + z[:,None,None]/self.dsqr               
          if self.fortran_ordering: ksqr = ksqr.transpose()
              
          self.ksqr = ksqr[self.fstart[0]-1:self.fend[0],
                           self.fstart[1]-1:self.fend[1],
                           self.fstart[2]-1:self.fend[2]]
          
          self.kx = kx[self.fstart[0]-1:self.fend[0]]
          self.ky = ky[self.fstart[1]-1:self.fend[1]]
          self.kz = kz[self.fstart[2]-1:self.fend[2]]          
          return

