"""
 sources of wave action

 Based on class main. 
 
 fortran extension for discrete interaction approximation
 is in a fortran extension module with original code
"""

import pyOM3                 as OM
from   pyOM3.wim_simple.main        import main

class source(main):


  # wim_simple carries the SIMPLEST source pair only (WAMDI 1988): Sinput and
  # Sdiss dispatch to Sinput_wamdi / Sdiss_wamdi. The Janssen input and the
  # WAM4 dissipation live in wim_realistic as an option (input_scheme).
  # Sep 9 2026: the unused variants Sinput_wam / Sinput_ecmwf / Sdiss_wam /
  # Sdiss_wam_github and the module wam_source.py were removed from here.
  def Sdiss(self,E):
      return self.Sdiss_wamdi(E)

  # ---------------------------------------------------------------
  # SOURCE STATE HOOKS of model._source_update. The WAMDI pair carries no
  # state between steps and nothing between its terms, so the hooks are
  # the public Sinput/Sdiss/frcutindex (which tests and diagnostics call
  # directly, and may replace on an instance). A scheme with state (the
  # Janssen input of wim_realistic: the wave stress tau_w carried between
  # steps, z0 and the active-input mask passed from the input to the
  # dissipation and the cut-off within the step) overrides the hooks with
  # pure functions of (state, aux): under JAX _source_update is jitted
  # with self static, so an attribute written inside it would be a leaked
  # tracer and an attribute read inside it a trace-time constant.
  #   get_source_state()         -> tuple of arrays carried between steps
  #   source_input(ustar,udir,E,state) -> gamma_in, state, aux (dict)
  #   source_diss(E,aux)         -> e_mean, om_mean, gamma_diss, aux
  #   source_cutoff(f_mean,ustar,aux) -> freq_mask
  #   set_source_state(state,aux)      stores the returned state and the
  #                                    per-step fields as attributes
  # ---------------------------------------------------------------
  def get_source_state(self):
      return ()

  def source_input(self, ustar, udir, E, state):
      return self.Sinput(ustar, udir, E), state, {}

  def source_diss(self, E, aux):
      e_mean, om_mean, gamma_diss = self.Sdiss(E)
      return e_mean, om_mean, gamma_diss, aux

  def source_cutoff(self, f_mean, ustar, aux):
      return self.frcutindex(f_mean, ustar)

  def set_source_state(self, state, aux):
      return

  def Sinput(self,ustar,udir,E=None):
      """growth rate of the wind input; E is the spectrum of this step (the
      Janssen input of wim_realistic needs it, the WAMDI input does not)"""
      return self.Sinput_wamdi(ustar,udir)
    
    
  def Snonlin(self,E):
      """
      Discrete interaction approximation from original WAM fortran code
      also numpy or jax version are available, see main.py
      """
      # fortran DIA needs unpadded shape (nx,ny,nd,nf) but input is padded (Nd,Nf,Ny,Nx)
      sl = slice(self.halo,-self.halo)
      if OM.jax:
          # the DIA is pointwise in (y,x): run it on the padded columns too
          # (a slice [halo:-halo] on y,x is no interior in the inline shard
          # layout), unpad and pad the spectral axes only
          h = self.halo
          E = OM.np.transpose(E[sl,sl,:,:],axes=(3,2,0,1))    # (Nx,Ny,nd,nf)
          sl_,fl = self.Snonlin_backend.source(E)
          w = ((h,h),(h,h),(0,0),(0,0))
          sl_ = OM.np.pad(OM.np.transpose(sl_,axes=(2,3,1,0)), w)
          fl  = OM.np.pad(OM.np.transpose(fl, axes=(2,3,1,0)), w)
          return sl_,fl
      E = E[sl,sl,sl,sl] # unpad
      E = OM.np.transpose(E,axes=(3,2,0,1)) # shape is now (self.nx,self.ny,self.nd, self.nf)
      sl,fl = self.Snonlin_backend.source(E)
      sl = OM.np.transpose(sl,axes=(2,3,1,0)) # shape is now (self.nd,self.nf,self.ny, self.nx)
      sl = OM.np.pad(sl,self.halo) # add halo points
      fl = OM.np.transpose(fl,axes=(2,3,1,0))
      fl = OM.np.pad(fl,self.halo) 
      return sl,fl

    
  def Sinput_wamdi(self,ustar,udir):
      """
      simple growth rate from Wamdi paper
      """
      if not len(OM.np.array(ustar).shape):
          ustar = OM.np.ones( (self.Ny,self.Nx), OM.prec)*ustar
      if not len(OM.np.array(udir).shape):
          udir = OM.np.ones( (self.Ny,self.Nx), OM.prec)*udir
          
      eps   = self.rho_a/self.rho_w
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
      # beta_o = 2.33e-5/alpha_pm^2 with alpha_pm = 0.66*4.57e-3 (= 2.5612);
      # the same coefficient is used for the analytic dissipation tails in
      # interface.py (calc_stress_work and the WAMDI tail of calc_stress)
      gamma = -self.beta_o*om_mean[None,:,:]*(self.omega_t[:,None,None]/ \
                                  OM.np.maximum(1e-32,om_mean[None,:,:]))**2
      gamma = gamma*alpha[None,:,:]**2
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
