"""
 Source terms of wim_realistic.

 wim_simple carries the simplest source pair only (WAMDI 1988). This class
 adds the pre-2019 ecWAM pair as an OPTION:

     input_scheme = 'wamdi'    (default)  Sinput_wamdi / Sdiss_wamdi of
                                          wim_simple, nothing changes
     input_scheme = 'janssen'             Janssen (1991) wind input with the
                                          wave-induced roughness z0 and the
                                          WAM4 dissipation of Bidlot et al.
                                          (2007) in wavenumber form

 The 'janssen' pair is the IPHYS=0 configuration of the ecwam repository
 (github.com/ecmwf-ifs/ecwam, routines sinput_jan, sdissip_jan, stresso,
 tau_phi_hf, z0wave, fkmean, femeanws, frcutindex): LLNORMAGAM, LLGCBZ0,
 LLCAPCHNK all false, NGST=1 (no gustiness), LLSNEG false (no negative
 input), TAUWSHELTER=0. Constants from ecwam setwavphys.F90 for IPHYS=0.
 The port is validated against those routines compiled with f2py
 (development/janssen_ref_test.py).

 Physics of the option
 ---------------------
 Input:  gamma = (beta_max/kappa^2) (rho_a/rho_w) sigma x^2 mu ln^4(mu),
         x = (u_*/c + z_alp) cos(theta - theta_u),
         ln(mu) = ln(k z0) + kappa/((u_*/c + z_alp) cos(theta - theta_u)),
         active where cos(theta - theta_u) > 0.01 and ln(mu) < 0.
 Roughness: z0 = (alpha/g) u_*^3 / sqrt(u_*^2 - tau_w), with tau_w the
         kinematic (per rho_a) momentum the waves absorb from the wind,
         tau_w = sum over all resolved bins of gamma E omega (rho_w/rho_a)
         plus the analytic tail beyond the last real bin (see _janssen_tail),
         capped at u_*^2/(1 + eps1). u_* is PRESCRIBED (the coupler hands the
         wave model the atmospheric stress), so there is no u_*(U10) iteration
         as in ecwam's airsea; the z0 <-> tau_w feedback is iterated
         janssen_ncall times inside one source step (ecwam: 2), starting
         from the tau_w of the previous step.
 Dissipation (direction independent):
         gamma_d = c_dis 2 pi f1 E_tot^2 kbar^4 (k/kbar) [(1-delta) + delta k/kbar]
         with E_tot, f1 (energy-weighted mean frequency), kbar (mean
         wavenumber from the sqrt(k) moment) of ecwam's fkmean, including
         its analytic f^-5 tail beyond the last real bin.
 Prognostic cut-off (frcutindex): f_max = 2.5 max(fm, fm_ws) with fm_ws the
         windsea mean frequency (bins with active input), nearest bin in
         log(f) as ecwam; the Pierson-Moskowitz term of wim_simple's rule
         is off (ecwam IPHYS=0 sets TAILFACTOR_PM=0).
 Tail (beyond the anchor bin, spectrum E_c (f_c/f)^5): stress and energy
         flux by Simpson's rule over ln(omega sqrt(z0/g)) from the anchor
         to omega = sqrt(g/z0), exactly ecwam tau_phi_hf. Direction: the
         tail growth is taken as cos^2(theta - theta_u) times the
         wind-aligned rate (cos not inside the logarithm, as in ecwam), and
         the stress is booked per direction along the wave direction s(theta)
         -- its projection on the wind equals ecwam's TAUHF, ecwam drops the
         cross-wind part. The same tail closes calc_stress / calc_phioc
         (through _tail_input_factors, anchor = frequency ghost bin, wim
         convention) and tau_w (anchor = last real bin, because the ghost
         is zero when Sinput runs), and calc_stress_work deposits it per
         quadrature node with 2 k_j e^{2 k_j z} (_janssen_tail_depth_weights).

 What is NOT ported: gustiness (NGST=2), the negative input LLSNEG, the
 growth-rate renormalisation LLNORMAGAM, the gravity-capillary z0 model
 LLGCBZ0, the sheltering TAUWSHELTER, viscous dissipation (cdisvis = 0 for
 IPHYS=0; the term is in place), the airsea u_*(U10, z0) solver.

 State carried between steps (janssen): tau_w (Ny,Nx) kinematic per rho_a,
 z0 (Ny,Nx). Not in the restart files; a restart begins from Charnock
 z0 = alpha u_*^2/g (tau_w = 0), which the ncall iteration corrects within
 the first step.

 MRO: this class sits before wim_simple.model in wim_realistic.model, so
 Sinput / Sdiss / frcutindex / _tail_input_factors override wim_simple's
 while everything else is inherited; wim_simple stays frozen.
"""
import pyOM3                       as OM
from   pyOM3.wim_simple.source     import source as _source_simple
import numpy                       as np__


class source(_source_simple):

  # ecwam setwavphys, IPHYS=0, LLGCBZ0=false
  _janssen_defaults = dict(betamax=1.20, zalp=0.008, alpha_charnock=0.0065,
                           kappa=0.40, cdis=-1.33, delta_sdis=0.5,
                           cdisvis=0.0, rnu=1.5e-6, eps1=1e-5,
                           tailfactor=2.5, janssen_ncall=2, jtot_tauhf=19,
                           rho_a=1.225, rho_w=1000.)

  def _jp(self, name):
      """Janssen parameter: attribute if set, else the ecwam default."""
      return getattr(self, name, self._janssen_defaults[name])

  # ---------------------------------------------------------------- dispatch
  # The public Sinput/Sdiss/frcutindex keep the signatures of wim_simple
  # (tests and diagnostics call them directly); for the Janssen scheme
  # they wrap the PURE bodies (_sinput_janssen, _sdiss_janssen,
  # _frcutindex_janssen), which take the state they need as arguments
  # and return what they produce: tau_w carried between steps, z0, the
  # active-input mask and the windsea mean frequency within the step. The
  # source-state hooks of wim_simple.source route model._source_update
  # (jitted under JAX) through the pure bodies; the public wrappers keep
  # the attributes tau_w, z0, _xllws, _fmws, tau_in_x/y up to date.
  def _janssen(self):
      return getattr(self, 'input_scheme', 'wamdi') == 'janssen'

  def Sinput(self, ustar, udir, E=None):
      if not self._janssen():
          return self.Sinput_wamdi(ustar, udir)
      if E is None: E = self.E                          # the spectrum of this step
      gam, state, aux = self._sinput_janssen(ustar, udir, E, self._janssen_state())
      self.set_source_state(state, aux)
      return gam

  def Sdiss(self, E):
      if not self._janssen():
          return self.Sdiss_wamdi(E)
      e_mean, om_mean, gamma, fmws = self._sdiss_janssen(E, getattr(self, '_xllws', None))
      if fmws is not None: self._fmws = fmws
      return e_mean, om_mean, gamma

  def frcutindex(self, f_mean, ustar):
      if not self._janssen():
          return super().frcutindex(f_mean, ustar)
      return self._frcutindex_janssen(f_mean, ustar, getattr(self, '_fmws', None))

  # the state hooks of model._source_update (see wim_simple.source)
  def _janssen_state(self):
      tau_w = getattr(self, 'tau_w', None)
      if tau_w is None or OM.np.shape(tau_w) != (self.Ny, self.Nx):
          tau_w = OM.np.zeros((self.Ny, self.Nx), OM.prec)
      return (tau_w,)

  def get_source_state(self):
      return self._janssen_state() if self._janssen() else ()

  def source_input(self, ustar, udir, E, state):
      if not self._janssen():
          return self.Sinput_wamdi(ustar, udir), state, {}
      return self._sinput_janssen(ustar, udir, E, state)

  def source_diss(self, E, aux):
      if not self._janssen():
          e_mean, om_mean, gamma_diss = self.Sdiss(E)
          return e_mean, om_mean, gamma_diss, aux
      e_mean, om_mean, gamma, fmws = self._sdiss_janssen(E, aux['act'])
      aux = dict(aux, fmws=fmws)
      return e_mean, om_mean, gamma, aux

  def source_cutoff(self, f_mean, ustar, aux):
      if not self._janssen():
          return self.frcutindex(f_mean, ustar)
      return self._frcutindex_janssen(f_mean, ustar, aux.get('fmws', None))

  def set_source_state(self, state, aux):
      if not self._janssen():
          return
      (self.tau_w,) = state
      self.z0     = aux['z0']
      self._xllws = aux['act']
      self.tau_in_x, self.tau_in_y = aux['tau_in_x'], aux['tau_in_y']   # diagnostic, per rho_w
      if 'fmws' in aux: self._fmws = aux['fmws']
      return

  def _frcutindex_janssen(self, f_mean, ustar, fmws):
      # ecwam frcutindex, IPHYS=0: f_max = TAILFACTOR*max(fm_ws, fm),
      # TAILFACTOR_PM = 0 (no Pierson-Moskowitz term), MIJ = nearest bin in
      # log(f), clamped so the lowest real bin is always prognostic.
      if fmws is not None:
          f_mean = OM.np.maximum(f_mean, fmws)
      f_max = self._jp('tailfactor')*f_mean
      ratio = OM.np.sqrt(1.0 + self.df0)             # half a bin in log(f)
      mask = self.ft[:, None, None] <= f_max*ratio
      mask = OM.modify_array(mask, (self.halo, slice(None), slice(None)), True)
      return mask

  # ---------------------------------------------------------------- helpers
  def _as_field(self, a):
      if not len(OM.np.array(a).shape):
          a = OM.np.ones((self.Ny, self.Nx), OM.prec)*a
      return a

  def _x0tauhf(self):
      """
      ecwam init_x0tauhf: x0 with alpha x0^2 exp(kappa/(x0 + z_alp)) = 1,
      the u_*/c below which the growth rate vanishes for Charnock z0. The
      tail integral of _janssen_tail starts at max(omega_c, x0 g/u_*).
      """
      if hasattr(self, '_x0tauhf_val'):
          return self._x0tauhf_val
      import math                          # a scalar constant: host arithmetic
      alph, zalp, kappa = (float(self._jp(k)) for k in ('alpha_charnock', 'zalp', 'kappa'))
      x0 = 0.005
      for _ in range(30):
          ff = math.exp(kappa/(x0 + zalp))
          f  = alph*x0**2*ff - 1.0
          if f == 0.0:
              break
          df = alph*ff*(2.0*x0 - kappa*(x0/(x0 + zalp))**2)
          x0 = x0 - f/df
      self._x0tauhf_val = float(x0)
      return self._x0tauhf_val

  def z0wave(self, ustar, tau_w):
      """
      ecwam z0wave: z0 = (alpha/g) u_*^3 / sqrt(u_*^2 - tau_w), tau_w kinematic
      per rho_a. The floor 1e-10 m keeps log(k z0) finite where u_* = 0
      (there the input vanishes through ln(mu) > 0 anyway).
      """
      arg = OM.np.maximum(ustar**2 - tau_w, self._jp('eps1'))
      z0  = self._jp('alpha_charnock')/self.grav*ustar**3/OM.np.sqrt(arg)
      return OM.np.maximum(z0, 1e-10)

  def _janssen_gamma(self, ustar, udir, z0):
      """
      ecwam sinput_jan (NGST=1, LLSNEG=false, LLNORMAGAM=false, deep water).
      Returns the growth rate gamma (Nd,Nf,Ny,Nx) [1/s] and the mask xllws
      (Nd,Nf,Ny,Nx) of bins with active input (ecwam XLLWS).
      """
      kappa  = self._jp('kappa')
      const1 = self._jp('betamax')/kappa**2
      eps    = self._jp('rho_a')/self._jp('rho_w')
      zalp   = self._jp('zalp')
      om   = self.omega_t                                   # (Nf,)
      k    = om**2/self.grav
      cinv = om/self.grav
      cosD = OM.np.cos(self.theta[:, None, None] - udir[None, :, :])   # (Nd,Ny,Nx)
      lz   = cosD > 0.01
      cosD_s = OM.np.where(lz, cosD, 1.0)                   # safe divisor
      ucn  = ustar[None, :, :]*cinv[:, None, None] + zalp   # (Nf,Ny,Nx)
      zcn  = OM.np.log(k[:, None, None]*z0[None, :, :])     # (Nf,Ny,Nx)
      zlog = zcn[None] + kappa/(cosD_s[:, None]*ucn[None])  # (Nd,Nf,Ny,Nx)
      act  = lz[:, None] & (zlog < 0.0)
      x    = cosD[:, None]*ucn[None]
      zlog2x = zlog*zlog*x
      cnsn = om*const1*eps                                  # sigma beta_max/kappa^2 rho_a/rho_w
      gam  = OM.np.where(act, zlog2x*zlog2x*OM.np.exp(OM.np.minimum(zlog, 0.0))
                              *cnsn[None, :, None, None], 0.0)
      return gam, act

  def _janssen_tail(self, fc, ustar, udir, z0, nodes=False):
      """
      ecwam tau_phi_hf for the f^-5 tail anchored at f_c (per direction,
      to be multiplied by E_c(theta) and summed with dd), kinematic per
      rho_w with wim's g convention:
         fac_tau(theta) = (rho_a/rho_w)(beta_max/kappa^2)(2 pi)^4 f_c^5 u_*^2
                          cos^2(theta-theta_u)/g^2 * sum_j w_j B_j dz
         fac_phi(theta) = the same with 1/g sqrt(z0/g) and B_j/Y_j
      where B = ln^4(mu) mu, ln(mu) = ln(g z0) + 2 ln(omega/g) + kappa/(u_* omega/g + z_alp)
      (min 0), Y = omega sqrt(z0/g), Simpson nodes Y_j = exp(z_inf + j dz)
      from z_inf = ln(max(omega_c, x0 g/u_*) sqrt(z0/g)) to 0, j = 0..jtot-1,
      weights (1,4,2,4,...,4,1)/3.
      Along the wind the summed stress equals ecwam's TAUHF*(rho_a/rho_w);
      the energy flux equals PHIHF/rho_w.
      With nodes=True also returns the per-node stress factors
      (jtot,Nd,Ny,Nx) whose sum over j is fac_tau, and k_j (jtot,Ny,Nx).
      """
      jtot   = self._jp('jtot_tauhf')
      kappa  = self._jp('kappa')
      const1 = self._jp('betamax')/kappa**2
      eps    = self._jp('rho_a')/self._jp('rho_w')
      zalp   = self._jp('zalp')
      g      = self.grav
      omc    = 2*OM.np.pi*fc
      us     = OM.np.maximum(ustar, 1e-6)                   # ecwam EPSUS guard
      omegacc = OM.np.maximum(omc, self._x0tauhf()*g/us)
      sqrtz0og = OM.np.sqrt(z0/g)
      zinf  = OM.np.log(omegacc*sqrtz0og)
      delz  = OM.np.maximum(-zinf/(jtot - 1), 0.0)           # (Ny,Nx)
      xloggz0 = OM.np.log(g*z0)
      w = np__.ones(jtot)*const1/3.0                       # Simpson weights (host)
      w[1:-1:2] *= 4.0
      w[2:-1:2] *= 2.0
      w = OM.np.asarray(w)
      j = OM.np.arange(jtot)
      y     = OM.np.exp(zinf[None] + j[:, None, None]*delz[None])    # (jtot,Ny,Nx)
      omega = y/sqrtz0og[None]
      cm1   = omega/g
      zx    = us[None]*cm1 + zalp
      zlog  = OM.np.minimum(xloggz0[None] + 2*OM.np.log(cm1) + kappa/zx, 0.0)
      zbeta = zlog**4*OM.np.exp(zlog)                        # (jtot,Ny,Nx)
      cosD  = OM.np.maximum(OM.np.cos(self.theta[:, None, None] - udir[None]), 0.0)
      cos2  = cosD**2                                        # (Nd,Ny,Nx)
      ctau  = eps*(2*OM.np.pi)**4*fc**5/g**2*ustar**2*delz   # (Ny,Nx)
      fac_tau = OM.np.sum(w[:, None, None]*zbeta, axis=0)[None]*ctau[None]*cos2
      cphi  = eps*(2*OM.np.pi)**4*fc**5/g*sqrtz0og*ustar**2*delz
      fac_phi = OM.np.sum(w[:, None, None]*zbeta/y, axis=0)[None]*cphi[None]*cos2
      if not nodes:
          return fac_tau, fac_phi
      fac_j = (w[:, None, None]*zbeta*ctau[None])[:, None]*cos2[None]   # (jtot,Nd,Ny,Nx)
      k_j   = omega**2/g
      return fac_tau, fac_phi, fac_j, k_j

  def _tail_input_factors(self):
      """
      Tail factors for calc_stress / calc_phioc / calc_stress_work. For the
      janssen scheme the ecwam tail anchored at the frequency ghost bin (wim
      convention, see wim_simple.interface); a and fL of the WAMDI closed
      form do not exist here (None), calc_stress_work branches on the scheme.
      """
      if getattr(self, 'input_scheme', 'wamdi') != 'janssen':
          return super()._tail_input_factors()
      fc = self.ft[-self.halo]
      z0 = getattr(self, 'z0', None)
      if z0 is None:
          z0 = self.z0wave(self._as_field(self.ustar), 0.0)
      fac_tau, fac_phi = self._janssen_tail(fc, self._as_field(self.ustar),
                                            self._as_field(self.udir), z0)
      return fac_tau, fac_phi, None, None

  def _janssen_tail_depth_weights(self, zt, zb):
      """
      Cell weights w (nz,Nd,Ny,Nx) of the tail stress for calc_stress_work:
      quadrature node j is deposited with 2 k_j e^{2 k_j z}, cell-integrated,
      so sum_k w_k dz_k = fac_tau exactly (with closed_bottom the bottom cell
      takes the remainder e^{-2 k_j D}). zt, zb are the cell faces (top = 0).
      """
      fc = self.ft[-self.halo]
      _, _, fac_j, k_j = self._janssen_tail(fc, self._as_field(self.ustar),
                                            self._as_field(self.udir),
                                            self.z0, nodes=True)
      et = OM.np.exp(2*k_j[None]*zt[:, None, None, None])   # (nz,jtot,Ny,Nx)
      eb = OM.np.exp(2*k_j[None]*zb[:, None, None, None])
      if getattr(self, 'closed_bottom', False):
          eb = OM.modify_array(eb, (0,), 0.0)
      V = OM.np.einsum('zjyx,jdyx->zdyx', et - eb, fac_j)  # int over cell
      return V/self._dzc[:, None, None, None]

  # ---------------------------------------------------------------- input
  def Sinput_janssen(self, ustar, udir, E=None):
      """Janssen input through the public wrapper (see the dispatch above)."""
      return self.Sinput(ustar, udir, E)

  def _sinput_janssen(self, ustar, udir, E, state):
      """
      Janssen input: growth rate gamma (Nd,Nf,Ny,Nx). PURE: takes the state
      (tau_w,) of the previous step, returns gamma, the new state (tau_w
      after janssen_ncall passes of gamma -> tau_w -> z0 as ecwam's two
      sinflx calls) and the per-step fields z0, the active-input mask act
      (for the windsea mean frequency of the dissipation) and tau_in (per
      rho_w, x/y, a diagnostic) in a dict.
      """
      ustar = self._as_field(ustar)
      udir  = self._as_field(udir)
      sl = slice(self.halo, -self.halo)
      eps = self._jp('rho_a')/self._jp('rho_w')
      (tau_w,) = state
      fc = self.ft[-self.halo - 1]                      # last REAL bin: the
      Ec = E[:, -self.halo - 1, :, :]                   # ghost is zero here
      om_s = self.omega_t[sl]
      for _ in range(int(self._jp('janssen_ncall'))):
          z0 = self.z0wave(ustar, tau_w)
          gam, act = self._janssen_gamma(ustar, udir, z0)
          # resolved input momentum, kinematic per rho_w: sum gamma E omega
          # over the real bins (prognostic and imposed alike)
          S = (gam*E)[sl, sl, :, :]*om_s[None, :, None, None]
          tx = OM.np.sum(OM.np.sum(S*self.sinth[sl, None, None, None], axis=0)
                         *self.dft[sl, None, None], axis=0)*self.dd
          ty = OM.np.sum(OM.np.sum(S*self.costh[sl, None, None, None], axis=0)
                         *self.dft[sl, None, None], axis=0)*self.dd
          fac_tau, _ = self._janssen_tail(fc, ustar, udir, z0)
          tx = tx + OM.np.sum(Ec[sl]*fac_tau[sl]*self.sinth[sl, None, None], axis=0)*self.dd
          ty = ty + OM.np.sum(Ec[sl]*fac_tau[sl]*self.costh[sl, None, None], axis=0)*self.dd
          tau_w = OM.np.sqrt(tx**2 + ty**2)/eps          # per rho_a, as u_*^2
          tau_w = OM.np.minimum(tau_w, ustar**2/(1.0 + self._jp('eps1')))
      return gam, (tau_w,), dict(z0=z0, act=act, tau_in_x=tx, tau_in_y=ty)

  # ---------------------------------------------------------------- dissipation
  def calc_fkmean(self, E):
      """
      ecwam fkmean: total energy, the f^-1 and f^1 weighted mean frequencies
      and the mean wavenumber from the sqrt(k) moment, over the real bins
      with the analytic f^-5 tail beyond the last real bin f_N,
         int_fN^inf F_N (f_N/f)^5 {1, 1/f, f, sqrt(k)} df
           = F_N {f_N/4, 1/5, f_N^2/3, (2 pi/sqrt g) f_N^2/3}.
      Returns e_mean, fm1 (= e/int F/f), f1 (= int f F/e), kbar (= (int sqrt(k) F/e)^2),
      all (Ny,Nx), deep water.
      """
      sl = slice(self.halo, -self.halo)
      g  = self.grav
      ft, dft = self.ft[sl], self.dft[sl]
      Ed = OM.np.sum(E[sl, sl, :, :], axis=0)*self.dd         # (nf,Ny,Nx)
      EN = Ed[-1]
      fN = ft[-1]
      sqk = 2*OM.np.pi*ft/OM.np.sqrt(g)                        # sqrt(k) = omega/sqrt(g)
      em  = OM.np.sum(Ed*dft[:, None, None], axis=0)      + 0.25*fN*EN
      fm1 = OM.np.sum(Ed*(dft/ft)[:, None, None], axis=0) + 0.2*EN
      f1  = OM.np.sum(Ed*(dft*ft)[:, None, None], axis=0) + fN**2/3.0*EN
      xk  = OM.np.sum(Ed*(dft*sqk)[:, None, None], axis=0) + (2*OM.np.pi/OM.np.sqrt(g))*fN**2/3.0*EN
      tiny = 1e-33
      em  = OM.np.maximum(em, tiny)
      fm1 = em/OM.np.maximum(fm1, tiny)
      f1  = f1/em
      xk  = OM.np.maximum((xk/em)**2, tiny)     # floor keeps k/kbar finite on empty columns
      return em, fm1, f1, xk

  def calc_femeanws(self, E, act):
      """ecwam femeanws: f^-1 mean frequency of the windsea (bins with active input)."""
      sl = slice(self.halo, -self.halo)
      ft, dft = self.ft[sl], self.dft[sl]
      Ed = OM.np.sum(OM.np.where(act, E, 0.0)[sl, sl, :, :], axis=0)*self.dd
      EN = Ed[-1]
      em  = OM.np.sum(Ed*dft[:, None, None], axis=0)      + 0.25*ft[-1]*EN
      fm  = OM.np.sum(Ed*(dft/ft)[:, None, None], axis=0) + 0.2*EN
      tiny = 1e-33
      return OM.np.maximum(em, tiny)/OM.np.maximum(fm, tiny)

  def Sdiss_janssen(self, E):
      """the WAM4 dissipation through the public wrapper (dispatch above)"""
      return self.Sdiss(E)

  def _sdiss_janssen(self, E, act):
      """
      ecwam sdissip_jan (WAM4 / Bidlot et al. 2007 wavenumber form):
        gamma_d = c_dis 2 pi f1 E_tot^2 kbar^4 X [(1 - delta) + delta X]
                  + rnu c_disvis k^2,   X = k/kbar,
      direction independent, (Nf,Ny,Nx) on the padded frequency axis.
      Returns e_mean, om_mean (= 2 pi fm1, the same f^-1 mean as wamdi's,
      used by frcutindex) and gamma_d, like Sdiss_wamdi.
      """
      em, fm1, f1, xk = self.calc_fkmean(E)
      fmws = self.calc_femeanws(E, act) if act is not None else None
      k   = self.omega_t**2/self.grav                          # (Nf,)
      sds = self._jp('cdis')*2*OM.np.pi*f1*em**2*xk**4         # (Ny,Nx)
      X   = k[:, None, None]/xk[None]
      d   = self._jp('delta_sdis')
      gamma = sds[None]*X*((1.0 - d) + d*X) \
              + self._jp('rnu')*self._jp('cdisvis')*k[:, None, None]**2
      om_mean = OM.np.clip(2*OM.np.pi*fm1, self.omega_t[self.halo],
                           self.omega_t[-self.halo - 1])
      return em, om_mean, gamma, fmws
