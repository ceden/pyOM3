"""
 sources of wave action

 Based on class main. 
 
 fortran extension for discrete interaction approximation
 is in a fortran extension module with original code
"""

import pyOM3                 as OM
from   pyOM3.wim_simple.main        import main
import pyOM3.wim_simple.wam_source  as WAM

class source(main):


  def Sdiss(self,E):
      #return self.Sdiss_wam(E)
      return self.Sdiss_wamdi(E)

  def Sinput(self,ustar,udir, z0=0.0002):
      #return self.Sinput_wam(ustar,udir,z0=z0)
      return self.Sinput_wamdi(ustar,udir)
    
    
  def Snonlin(self,E):
      """
      Discrete interaction approximation from original WAM fortran code
      also numpy or jax version are available, see main.py
      """
      # fortran DIA needs unpadded shape (nx,ny,nd,nf) but input is padded (Nd,Nf,Ny,Nx)
      sl = slice(self.halo,-self.halo)
      E = E[sl,sl,sl,sl] # unpad
      E = OM.np.transpose(E,axes=(3,2,0,1)) # shape is now (self.nx,self.ny,self.nd, self.nf)
      sl,fl = self.Snonlin_backend.source(E)
      sl = OM.np.transpose(sl,axes=(2,3,1,0)) # shape is now (self.nd,self.nf,self.ny, self.nx)
      sl = OM.np.pad(sl,self.halo) # add halo points
      fl = OM.np.transpose(fl,axes=(2,3,1,0))
      fl = OM.np.pad(fl,self.halo) 
      return sl,fl

    
  def Sinput_wam(self,ustar,udir, z0 = 0.0002):    
      """
      growth rate for input from WAM
      uses module wam_source
      """
      if not len(OM.np.array(ustar).shape):
          ustar = OM.np.ones( (self.Ny,self.Nx), OM.prec)*ustar
      if not len(OM.np.array(udir).shape):
          udir = OM.np.ones( (self.Ny,self.Nx), OM.prec)*udir
      if not len(OM.np.array(z0).shape):
          z0 = OM.np.ones( (self.Ny,self.Nx), OM.prec)*z0
            
      roairn = OM.np.zeros_like(ustar) + 1.225
      wstar  = OM.np.zeros_like(ustar) 
      gamma  = WAM.sinput(ustar, udir, z0, roairn, wstar, self.theta, self.ft)
      return gamma



  def Sdiss_wam(self,E):
      """
      growth rate for dissipation from WAM
      uses module wam_source
      """
      sl = slice(self.halo,-self.halo)
      E = E[sl,sl,sl,sl] # unpad
      th = self.theta[sl]   # direction bins (emean_fmean only uses th.size)
 
      e_mean,f_mean =  WAM.emean_fmean(E, self.ft[sl], th)               
      gamma = WAM.sdissip(e_mean, f_mean, self.ft[sl])
      
      gamma = OM.np.pad(gamma,self.halo) # add halo points
      e_mean = OM.np.pad(e_mean,self.halo)
      f_mean = OM.np.pad(f_mean,self.halo)
      return e_mean, f_mean*2*OM.np.pi, gamma
 

  def Sdiss_wam_github(self,E):
      """
      Sink by dissipation from WAM github, independent of direction   
      returns growth rate which is independent of direction
      should be identical to Sdiss_wam
      """      
      cdis, delta = 1.33, 0.5
      e_mean,om_mean = self.calc_emean_wam(E)
      consd = -cdis/self.grav**4    
      temp1 = (self.omega_t[:,None,None]/OM.np.maximum(1e-32, om_mean[None,:,:] ) )**2      
      tmp =  om_mean**9
      gamma = consd * e_mean[None,:,:]**2 * tmp[None,:,:]* ((1.-delta)*temp1 + delta*temp1**2)     
      return e_mean, om_mean, gamma
     
    
  def Sinput_ecmwf(self,ustar,udir, z0 = 0.0002):
      """
      Source by wind 
      ustar : FRICTION VELOCITY
      udir  : WIND DIRECTION
      z0    : ROUGHNESS LENGTH 
      returns source 
      from IFS DOCUMENTATION – Cy46r1, before June 2019
      """

      if not len(OM.np.array(ustar).shape):
          ustar = OM.np.ones( (self.Ny,self.Nx), OM.prec)*ustar
      if not len(OM.np.array(udir).shape):
          udir = OM.np.ones( (self.Ny,self.Nx), OM.prec)*udir
      if not len(OM.np.array(z0).shape):
          z0 = OM.np.ones( (self.Ny,self.Nx), OM.prec)*z0

      eps   = 1.225/1000.    
      zalp , betamax, kappa = 0.0080, 1.20, 0.40       # VON KARMAN CONSTANT.         
    
      c = self.grav / self.omega_t  
      cos = OM.np.cos(self.theta[:,None,None,None] - udir[None,None,:,:]) 
      uc = ustar[None,None,:,:]/c[None,:,None,None]  
      x =  uc * OM.np.maximum(cos, 0)  
      Omega_m = self.grav * kappa**2 * z0 / ustar**2 # shape is ny x ny
       
      #xhat = ( uc +zalp ) * cos
      xhat = OM.np.maximum( ( uc +zalp ) * cos , 5e-3 ) 
      # exp(0.4/5e-3) is still defined, smaller is not
      
      mu = 1/kappa**2 * uc**2 * Omega_m[None,None,:,:] * OM.np.exp(kappa/xhat)    
      beta = OM.np.where( mu<=1, betamax/kappa**2*mu*OM.np.log(mu)**2 , 0)      
      gamma = self.omega_t[None,:,None,None]*eps*beta*x**2 
      return gamma  
    
        
  def Sinput_wamdi(self,ustar,udir):
      """
      simple growth rate from Wamdi paper
      """
      if not len(OM.np.array(ustar).shape):
          ustar = OM.np.ones( (self.Ny,self.Nx), OM.prec)*ustar
      if not len(OM.np.array(udir).shape):
          udir = OM.np.ones( (self.Ny,self.Nx), OM.prec)*udir
          
      eps   = 1.225/1000.  
      cos = OM.np.cos(self.theta[:,None,None,None] - udir[None,None,:,:])
      c = self.grav/self.omega_t 
      uu = 28*ustar[None,None,:,:]/c[None,:,None,None]*cos
      gamma = OM.np.maximum(0,0.25*eps*(uu-1) )*self.omega_t[None,:,None,None]
      return gamma

    
  def Sdiss_wamdi(self,E):
      """
      dissipation growth rate from Wamdi paper
      returns growth rate which is independent of direction
      """
      e_mean,om_mean = self.calc_emean_wamdi(E)
      alpha = e_mean*om_mean**4/self.grav**2
      alpha_pm = 0.66*4.57e-3
      gamma = -2.33e-5*om_mean[None,:,:]*(self.omega_t[:,None,None]/ \
                                  OM.np.maximum(1e-32,om_mean[None,:,:]))**2
      gamma = gamma*(alpha[None,:,:]/alpha_pm)**2  
      #gamma = OM.np.minimum(gamma, 0.0)
      return e_mean, om_mean, gamma

      
      
      
  def calc_emean_wamdi(self,E):
      """
      calculate total energy and mean omega as in Wamdi paper
      """
      sl = slice(self.halo,-self.halo,None)
      e_mean_dir = OM.np.sum(E[sl,:,:,:]*self.dd,axis=0)
      e_mean     = OM.np.sum(e_mean_dir[sl,:,:]*self.dft[sl,None,None],axis=0)    
      e_mean     = OM.np.maximum(e_mean, 0.0)
      
      om_mean    = OM.np.sum(e_mean_dir[sl,:,:]/self.omega_t[sl,None,None]
                             *self.dft[sl,None,None],axis=0)
      om_mean    =  e_mean/OM.np.maximum(1e-32,om_mean)     
      om_mean    = OM.np.clip(om_mean, self.omega_t[self.halo], self.omega_t[-self.halo-1])
      return e_mean,om_mean
    
    
  def calc_emean_wam(self,E):
      """
      calculate total energy and mean omega as in ECMWF documentation
      """
      sl = slice(self.halo,-self.halo,None)
      e_mean_dir = OM.np.sum(E[sl,:,:,:]*self.dd,axis=0)
      e_mean     = OM.np.sum(e_mean_dir[sl,:,:]*self.dft[sl,None,None],axis=0) 
      e_mean     = OM.np.maximum(e_mean, 0.0)
      
      om_mean    = OM.np.sum(e_mean_dir[sl,:,:]*self.omega_t[sl,None,None]
                             *self.dft[sl,None,None],axis=0)
      om_mean    = om_mean/OM.np.maximum(1e-32,e_mean)  
      om_mean    = OM.np.clip(om_mean, self.omega_t[self.halo], self.omega_t[-self.halo-1])
      return e_mean,om_mean


      