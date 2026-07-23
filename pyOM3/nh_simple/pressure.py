"""
 pressure solver with FFT

 Based on boundary class and P3FFT module
"""
import pyOM3                    as OM
from   pyOM3.nh_simple.boundary import boundary
import pyOM3.p3dfft.fortran     as FFT


class pressure(boundary):


      def pressure_solver(self): 
          """
          solve for non-hydrostatic pressure with FFT 
          only for shapes (x,y,z)
          """
          if not self.fortran_ordering:   
              raise NotImplementedError("pressure_solver requires fortran_ordering")
              
          cprec = OM.np.result_type(OM.prec, OM.np.complex64)    
          
          aloc = (( self.u - self.rollx(self.u,1) )/self.dx  + 
                  ( self.v - self.rolly(self.v,1) )/self.dy  +
                  ( self.w - self.rollz(self.w,1) )/self.dz )/self.dt
                  
          sl = slice(self.halo,-self.halo,None)
          aloc = aloc[sl,sl,sl] # unpad
          aloc = aloc if aloc.flags['F_CONTIGUOUS'] else OM.np.asfortranarray(aloc)
          
          if self.periodic_in_z:  
                
             A = OM.np.zeros( self.fsize , cprec, order='F' )
             FFT.fft_3d(aloc,A)                 
             A = A*self.dx*self.dy*self.dz  
             A = OM.np.divide( - A, self.ksqr,  out = OM.np.zeros_like(A), 
                               where = self.ksqr > 0.,order='F') 
             FFT.ifft_3d(A,aloc) 
             aloc = aloc/(self.Lx*self.Ly*self.Lz) 
              
          else:
             #p1 = OM.np.zeros( self.fsize, cprec, order='F'  )
             #p2 = OM.np.zeros( self.fsize, cprec, order='F'  ) 
             
             #FFT.fft_2d(aloc,p1)
             #p1 = p1*self.dx*self.dy                 
                           
             #p2[:,:,0] = 0.    
             #p2[:,:,1] = p1[:,:,0]*self.dz**2*self.dsqr
             #for k in range(1,self.nz-1):
             #     p2[:,:,k+1] = 2*p2[:,:,k] - p2[:,:,k-1] + p1[:,:,k]*self.dz**2*self.dsqr
                 
             #p1 = solve_tridiag(p1,self.dz,self.dsqr,self.ksqr)    
             #p1 = OM.np.where( self.ksqr > 0, p1 , p2)
              
             #FFT.ifft_2d(p1 if p1.flags['F_CONTIGUOUS'] \
             # else OM.np.asfortranarray(p1),aloc)   
             #aloc = aloc/(self.Lx*self.Ly)             

             p1 = OM.np.zeros( self.fsize, cprec, order='F'  )
             FFT.fft_2d(aloc, p1)
             p1 = p1 * self.dx * self.dy

             fac = self.dz**2 * self.dsqr

             # ksqr==0 mode: integrate d2p/dz2 = rhs directly via a z-recurrence.
             # full-plane, static shapes -> jit/vmap friendly.
             # body operates on the whole (nx,ny) plane at once for each k.
             def _body(k, p2):
                 nxt = 2*p2[:, :, k] - p2[:, :, k-1] + p1[:, :, k]*fac
                 return OM.modify_array(p2, (slice(None), slice(None), k+1), nxt)

             p2 = OM.np.zeros(self.fsize, cprec, order='F')
             p2 = OM.modify_array(p2, (slice(None), slice(None), 1), p1[:, :, 0]*fac)
             p2 = OM.for_loop(1, self.nz-1, _body, p2)

             p1 = solve_tridiag(p1, self.dz, self.dsqr, self.ksqr)
             p1 = OM.np.where(self.ksqr > 0, p1, p2)

             FFT.ifft_2d(p1 if p1.flags['F_CONTIGUOUS'] else OM.np.asfortranarray(p1), aloc)
             aloc = aloc / (self.Lx*self.Ly)
        
          self.p = self.apply_bc( OM.np.pad(aloc,self.halo) )
              
          self.u += - self.dt          *( self.rollx(self.p,-1) - self.p)/self.dx  
          self.v += - self.dt          *( self.rolly(self.p,-1) - self.p)/self.dy 
          self.w += - self.dt/self.dsqr*( self.rollz(self.p,-1) - self.p)/self.dz 
          self.w = self.apply_vertical_flux_bc(self.w)          

          self.u = self.apply_bc(self.u)
          self.v = self.apply_bc(self.v)
          self.w = self.apply_bc(self.w)
          #self.w = self.apply_vertical_flux_bc(self.w)  
          
          return



def solve_tridiag_old(x,dz,dsqr,ksqr):  
    """
          Thomas algorithm to solve tridiagonal matrix
          but only for shape (...,z)
          this is the original version, below is what Claude made for jax
    """
    with OM.np.errstate(divide='ignore',invalid='ignore'): 
    # division by zero possible for ksqr=0 

             fxb = 1./dz**2/dsqr 
             cp = OM.np.zeros( x.shape, OM.prec)
             dp = OM.np.zeros( x.shape, OM.np.result_type(x, OM.np.complex64)) 
             b  = OM.np.zeros( x.shape, OM.prec) 
             b[...,   0] = -ksqr[...,   0] - fxb
             b[...,1:-1] = -ksqr[...,1:-1] - 2*fxb
             b[...,  -1] = -ksqr[...,  -1] - fxb
        
             # initialize c-prime and d-prime
             #a_tri[..., 1:] = fxb
             #c_tri[...,:-1] = fxb
                          
             cp[...,0] = fxb/b[...,0]
             dp[...,0] = x[...,0]/b[...,0]
             # solve for vectors c-prime and d-prime
             for i in range(1,x.shape[-1]):
                  m = b[...,i]-cp[...,i-1]*fxb
                  fxa = 1./m 
                  cp[...,i] = fxb*fxa
                  dp[...,i] = (x[...,i]-dp[...,i-1]*fxb)*fxa
             #cp[...,-1] = 0.    
             # initialize x
             x[...,-1] = dp[...,-1] 
             # solve for x from the vectors c-prime and d-prime
             for i in range(x.shape[-1]-2,-1,-1):
                  x[...,i] = dp[...,i]-cp[...,i]*x[...,i+1]  
             return x


def solve_tridiag(x, dz, dsqr, ksqr):
    """
    Thomas algorithm to solve tridiagonal matrix, only along the last (z) axis.
    JAX/NumPy compatible: sequential z-recurrences use OM.for_loop and
    functional updates via OM.modify_array.
    """
    with OM.np.errstate(divide='ignore', invalid='ignore'):
        # division by zero possible for ksqr==0 (overwritten by caller)

        N   = x.shape[-1]
        fxb = 1./dz**2/dsqr

        cp = OM.np.zeros(x.shape, OM.prec)
        dp = OM.np.zeros(x.shape, OM.np.result_type(x, OM.np.complex64))

        b  = OM.np.zeros(x.shape, OM.prec)
        b  = OM.modify_array(b, (Ellipsis, 0),            -ksqr[..., 0]    - fxb)
        b  = OM.modify_array(b, (Ellipsis, slice(1, -1)), -ksqr[..., 1:-1] - 2*fxb)
        b  = OM.modify_array(b, (Ellipsis, -1),           -ksqr[..., -1]   - fxb)

        cp = OM.modify_array(cp, (Ellipsis, 0), fxb / b[..., 0])
        dp = OM.modify_array(dp, (Ellipsis, 0), x[..., 0] / b[..., 0])

        # forward sweep: carry (cp, dp), i = 1 .. N-1
        def _fwd(i, state):
            cp, dp = state
            m   = b[..., i] - cp[..., i-1]*fxb
            fxa = 1./m
            cp  = OM.modify_array(cp, (Ellipsis, i), fxb*fxa)
            dp  = OM.modify_array(dp, (Ellipsis, i), (x[..., i] - dp[..., i-1]*fxb)*fxa)
            return (cp, dp)

        cp, dp = OM.for_loop(1, N, _fwd, (cp, dp))

        # back substitution: x[N-1] = dp[N-1]; then i = N-2 .. 0
        x = OM.modify_array(x, (Ellipsis, -1), dp[..., -1])

        def _bwd(i, x):
            j = N - 2 - i                      # descend: i=0 -> j=N-2, ... i=N-2 -> j=0
            return OM.modify_array(x, (Ellipsis, j), dp[..., j] - cp[..., j]*x[..., j+1])

        x = OM.for_loop(0, N-1, _bwd, x)
        return x

