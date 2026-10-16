"""
 Source terms for the wim_idemix model.

 The prognostic pair sees four processes, all as moments of the
 corresponding spectral source over the shape ansatz:

   Sinput    WAM-type wind input, shape-weighted over the PEAK
             component only (x <= x_win): the tail input belongs to the
             tail budget, which is routed straight to the ocean
   Sdiss     whitecapping, returned as (base, enhanced); the enhanced
             part is the saturation-excess route and acts at the FLANK
             frequency, so it downshifts instead of merely annihilating
             energy at fixed sigbar
   Scascade  the S_nl analogue: the exact spectral flux across the FIXED
             surface x = x_win, calibrated against the exact S_nl as
             c_r_hat(xi), r_c(xi) and tabulated in main; it removes
             energy from the pair at nu_nl and deposits it at
             sigma_c = r_c sigbar, whence it is routed to the ocean
   directional relaxation
             the remaining S_nl action across theta: relaxation toward a
             reference distribution at the MEASURED narrowness rate
             0.20 Nn^5.4, so the equilibrium spread is a prediction

 The stress on the ocean is NOT assembled from these terms but taken as
 the wave-momentum BUDGET residual tau_o = tau_in - dM_w/dt, which is
 exactly conservative: whatever the waves absorb and do not retain
 reaches the ocean.  No cap is applied -- clamping the delivered stress
 while the waves absorb tau_in would break momentum conservation.  The
 atmospheric supply rho_a/rho_o u*^2 is carried as a diagnostic only.
"""
import numpy as np__


def _ipow(x, p):
    """x**p for the reference-distribution exponent. When p is a scalar
    NON-NEGATIVE INTEGER (the default ref_spread_s gives 12) the power is
    formed by binary exponentiation, which is several times faster than
    the library pow and differs from it only in the last bit; otherwise
    (dyn_target makes p an array) the library power is used."""
    try:
        n = int(p)
        ok = (float(n) == float(p)) and n >= 0 and np__.isscalar(p)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return x**p
    r = None
    b = x
    while n:
        if n & 1:
            r = b if r is None else r*b
        n >>= 1
        if n:
            b = b*b
    return np__.ones_like(x) if r is None else r
from   pyOM3.wim_idemix.main import main


class source(main):

      def _cos_rel(self, udir):
          """max(cos(theta - udir), 0) on the nd REAL bins, built from the
          angle-difference identity cos(th-u) = cos th cos u + sin th sin u
          so the transcendentals are evaluated on the (Ny,Nx) wind field
          instead of the full direction-resolved array (the direction
          tables sinth/costh are precomputed). Differs from the direct
          cosine only in the last bit."""
          cu = np__.cos(udir)
          su = np__.sin(udir)
          sl = self.dsl
          return np__.maximum(self.s_y[sl]*cu + self.s_x[sl]*su, 0.0)

      def Sinput(self, ustar, udir):
          """Shape-weighted WAM-type input moments (gamma_E, gamma_A)."""
          sb   = self.sigbar()[self.dsl]
          cosd = self._cos_rel(udir)
          w    = 28.0*ustar*sb*cosd/(self.grav*self.s_half)
          q    = self.c_in*0.25*(self.rho_a/self.rho_o)*sb/self.s_half
          gE   = q*np__.interp(w, self._wg, self._IE)
          gA   = q*np__.interp(w, self._wg, self._IA)
          return gE, gA


      def Sdiss(self):
          """
          Whitecapping rate (negative), steepness-squared at sigbar_g,
          returned as (base, enhanced) with shape (1,Ny,Nx).
          """
          sg     = self.sigbar_g()
          E      = self.Ed[self.dsl].sum(0)*self.dth
          steep2 = (E*sg**4/self.grav**2)**2
          kbar   = (sg/self.s_half)**2/self.grav
          Atot   = self.Ad[self.dsl].sum(0)*self.dth
          satX   = np__.minimum(2.0*self.N_p*np__.sqrt(self.grav)*Atot
                                *kbar**2.5/self.B_sat, self.sat_cap)
          self._satX = satX               # kept for the stress tail level
          enh  = self.c_sat*np__.maximum(0.0, satX - 1.0)**2
          # paper Eq. (27): gammabar_out = -beta_o (s^c_1/s_1/2^2)
          #                 Ebar^2 sigbar_g^9 / g^4
          base = (-self.beta_o*(self.s1_c/self.s_half**2)
                  *sg*steep2)[None,:,:]
          return base, base*enh[None,:,:]


      def cascade_constants(self, ustar):
          """(c_r_hat, r_c) from the calibrated xi-table; see main."""
          if not self.use_xi_table:
             return self.c_r_hat, self.r_c
          sg   = self.sigbar_g()
          kbar = (sg/self.s_half)**2/self.grav
          Atot = self.Ad[self.dsl].sum(0)*self.dth
          xi   = ustar/np__.maximum(self.grav*Atot*kbar**2, 1e-30)
          self._xi = xi                   # kept for the tail level, paper eq:toba
          return (np__.interp(xi, self.xi_tab, self.cr_tab),
                  np__.interp(xi, self.xi_tab, self.rc_tab))


      def Scascade(self, ustar=0.0):
          """Cascade rate nu_nl and removal frequency sigma_c, (1,Ny,Nx)."""
          c_r, r_c = self.cascade_constants(ustar)
          sg = self.sigbar_g()
          M2 = 2*np__.pi*np__.sum(self.Ed[self.dsl]**2, 0)*self.dth
          nu = c_r*self.grav**-4*sg**9*M2
          return nu[None,:,:], (r_c*sg)[None,:,:]


      def source_substeps(self, ustar, udir):
          """
          Number of source sub-steps: the stiffest source rate times dt
          over the target per-step change source_cfl; 1 outside the
          growth turnover (see model.time_step for why this is needed).
          """
          if not self.adaptive_sources:
             return 1
          ginE, _ = self.Sinput(ustar, udir)
          gb, ge  = self.Sdiss()
          nu, _   = self.Scascade(ustar)
          r = max(float(np__.abs(ginE).max()),
                  float(np__.abs(self.d_out*gb + ge).max()),
                  float(np__.abs(nu).max()))
          return int(min(max(1.0, np__.ceil(r*self.dt/self.source_cfl)),
                         self.max_substeps))


      def add_sources(self, ustar, udir, dt):
          """
          Semi-implicit source update of (Acal,Ecal), house pattern
          X += S dt/(1 - dt dSdX) with the dissipative part implicit,
          followed by the directional relaxation and the stress budget.
          """
          # wave momentum before the sources (variance units)
          # THE SOURCES RUN ON THE REAL BINS ONLY.  The ring halo is a
          # copy of the interior, so computing it here would repeat
          # 2*halo/nd of every source term (17% at nd = 24) for a result
          # that apply_bc overwrites at the top of the next propagate().
          # Ed/Ad are read and written through the interior slice, and
          # the halo is left to the boundary condition.
          sl  = self.dsl
          A, E = self.Ad[sl], self.Ed[sl]
          kb0 = self.sigbar()[sl]**2/(self.grav*self.s_half**2)
          m0x = (kb0*A*self.s_x[sl]).sum(0)*self.dth
          m0y = (kb0*A*self.s_y[sl]).sum(0)*self.dth

          # energy budget booking (mirror of the stress budget): column
          # energy and physical input rate BEFORE the update; phi_oc is the
          # residual P_in - dE/dt across the sources, so at steady state the
          # flux to the ocean equals the input identically (no term-wise
          # closure can overdraw). Kinematic (m^2/s); calc_phioc adds g.
          e0    = E.sum(0)*self.dth
          ginE, ginA = self.Sinput(ustar, udir)
          gout, gout_e = self.Sdiss()
          nu, sigc = self.Scascade(ustar)
          r_fl = self.s_half/np__.sqrt(self.x_fl)   # sigbar/sigma_fl < 1

          # energy:  dEcal/dt = (gin + d_out gout + gout_e - nu) Ecal
          gE   = ginE + self.d_out*gout + gout_e - nu
          SE   = gE*E
          dSdE = np__.minimum(gE, 0.0) if self.implicit_dissipation_only else gE
          self.Ed[sl] = E + SE*dt/(1.0 - dt*dSdE)

          # action:  dAcal/dt = (gin + gout + r_fl gout_e) Acal
          #                     - nu Ecal/sigma_c
          SA   = (ginA + gout + r_fl*gout_e)*A - nu*self.Ed[sl]/sigc
          gA   = ginA + gout + r_fl*gout_e - nu*self.sigbar()[sl]/sigc
          dSdA = np__.minimum(gA, 0.0) if self.implicit_dissipation_only else gA
          self.Ad[sl] = A + SA*dt/(1.0 - dt*dSdA)

          pin_r = (ginE*self.Ed[sl]).sum(0)*self.dth      # after update: input
                                                        # rate at current state
          # THE INPUT'S REALIZED MOMENTUM CHANGE. The momentum below k_c
          # is slaved to the pair, M = E^2 s /(A g s_half^2) per bin, so
          # the input moves M only through its (dA, dE) -- its I_M moment
          # never acts on the state. Its share of the fused semi-implicit
          # increment goes through the same implicit denominators; the
          # attribution is linear in dt, the order of the splitting
          # itself. The k s-weighted moment belongs to the tail alone.
          c_    = self.grav*self.s_half**2
          dE_in = ginE*E*dt/(1.0 - dt*dSdE)
          dA_in = ginA*A*dt/(1.0 - dt*dSdA)
          # derivatives OF THE SAME M THAT IS ACCUMULATED: M = sb^2 A/c
          # with sb the GUARDED sigbar (floored where empty). Where the
          # guard binds, sb does not respond to (dA, dE) and M ~ sb_b^2 A:
          # dM/dE = 0, dM/dA = +sb_b^2/c. Using the raw ratio there
          # mismatches the accumulator and wrecked the fetch stress.
          Eg    = np__.maximum(E, 1e-30)
          Ag    = np__.maximum(A, self.Ad_floor)
          sb_   = Eg/Ag
          un    = (E > 1e-30) & (A > self.Ad_floor)
          dMdE  = np__.where(un, 2.0*sb_/c_, 0.0)
          dMdA  = np__.where(un, -sb_**2/c_, sb_**2/c_)
          dm    = dMdE*dE_in + dMdA*dA_in
          self._dMin_x += (dm*self.s_x[sl]).sum(0)*self.dth
          self._dMin_y += (dm*self.s_y[sl]).sum(0)*self.dth

          self.add_directional_relaxation(nu, dt)
          self.apply_floors()

          # the WHOLE local cycle's realized momentum change (m0x, m0y
          # were taken at entry), relaxation and floors included -- over
          # the restricted range neither is momentum-neutral, and the
          # relaxation IS the restricted sigma S_nl of eq. 28.
          kb1 = self.sigbar()[sl]**2/(self.grav*self.s_half**2)
          self._dMloc_x += (kb1*self.Ad[sl]*self.s_x[sl]).sum(0)*self.dth - m0x
          self._dMloc_y += (kb1*self.Ad[sl]*self.s_y[sl]).sum(0)*self.dth - m0y
          self._t_acc   += dt

          e1 = self.Ed[sl].sum(0)*self.dth
          self._win_sub  = pin_r                        # (Ny,Nx) m^2/s
          self._phi_sub  = pin_r - (e1 - e0)/dt         # (Ny,Nx) m^2/s
          return


      def tau_tail(self, ustar, udir):
          """Momentum input to the diagnostic range k > k_c (eq:tautail):
          varphi_tail = min(varphi_equi, varphi_limit), closed form in
          J1 (equilibrium branch) and J2 (limit branch), split at x_s and
          cut on at x = 1/w^2. This is the ONE place the k s-weighted
          input moment belongs: the tail is not slaved to the pair.
          Also refreshes tau_in_v / tau_in_g / tau_atm (diagnostics for
          the writer and Fig-3 comparisons; NOT part of tau_o).
          Returns the tail vector (2, Ny, Nx), kinematic m2/s2."""
          sl   = self.dsl
          sb   = self.sigbar()[sl]
          cosd = self._cos_rel(udir)
          w    = 28.0*ustar*sb*cosd/(self.grav*self.s_half)
          q    = self.c_in*0.25*(self.rho_a/self.rho_o)*sb/self.s_half
          kb   = sb**2/(self.grav*self.s_half**2)
          a    = 0.5*self.alpha_e*self._xi[None,:,:]
          b    = (self.N_p/np__.maximum(self._satX, 1e-30))[None,:,:]
          x_s  = np__.maximum(np__.minimum(b/np__.maximum(a, 1e-30),
                                           1e15)**2, self.x_win)
          x_w  = 1.0/np__.maximum(w, 1e-30)**2          # input cut-on
          x1   = np__.maximum(self.x_win, x_w)          # equi part lower
          x1e  = np__.minimum(x1, x_s)
          J1   = np__.where(x_s > x1,
                            w*np__.log(x_s/x1e) + 2.0*(x_s**-0.5 - x1e**-0.5),
                            0.0)
          x2   = np__.maximum(x_s, x1)                  # limit part lower
          J2   = np__.maximum(2.0*w/np__.sqrt(x2) - 1.0/x2, 0.0)
          ti   = q*kb*self.Ad[sl]*(a*J1 + b*J2)
          tx   = (ti*self.s_x[sl]).sum(0)*self.dth
          ty   = (ti*self.s_y[sl]).sum(0)*self.dth
          IM   = np__.interp(w, self._wg, self._IM)
          tp   = q*kb*self.Ad[sl]*IM
          self.tau_in_v = self.grav*np__.stack(
              (tx + (tp*self.s_x[sl]).sum(0)*self.dth,
               ty + (tp*self.s_y[sl]).sum(0)*self.dth))
          self.tau_in_g = np__.sqrt(self.tau_in_v[0]**2 + self.tau_in_v[1]**2)
          self.tau_atm  = (self.rho_a/self.rho_o)*ustar**2
          return self.grav*np__.stack((tx, ty))

      def stress_couple(self, ustar, udir):
          """The stress on the ocean, ONCE PER COUPLING INTERVAL:

              tau_o = - dM|_sinks / Dt + tau_tail
                    = (dM_input - dM_local)/Dt + tau_tail

          dM_local is the realized momentum change of the pair over the
          whole local cycle since the last call (accumulated in
          add_sources: sources, cascade, relaxation, floors), dM_input
          the input's realized share of it. Their difference is what the
          sinks took -- the model's own realization of the restricted
          gamma_out E + sigma S_nl of eq. 28; the tail adds eq:tautail
          (eq. 29's surviving term). The resolved-range I_M moment
          appears nowhere: below k_c the momentum is slaved to the pair
          and that moment never acts on the state.

          In steady state dM_local = 0 and tau_o = realized input + tail.
          Resets the accumulators; the call cadence defines the interval.
          """
          if self._t_acc <= 0.0:
              # no steps since the last call: KEEP the last coupled
              # stress. Recomputing here would silently deliver a
              # tail-only tau_o (0.29 tau_a where 0.62 is right).
              return
          rx = (self._dMin_x - self._dMloc_x)/self._t_acc
          ry = (self._dMin_y - self._dMloc_y)/self._t_acc
          self.tau_o_v = self.grav*np__.stack((rx, ry)) \
              + self.tau_tail(ustar, udir)
          self.tau_o_g = np__.sqrt(self.tau_o_v[0]**2 + self.tau_o_v[1]**2)
          self._dMin_x[:] = 0.0
          self._dMin_y[:] = 0.0
          self._dMloc_x[:] = 0.0
          self._dMloc_y[:] = 0.0
          self._t_acc = 0.0
          return

      def add_directional_relaxation(self, nu, dt):
          """
          Directional redistribution: bounded relaxation toward a
          reference distribution, at the rate extracted from the exact
          S_nl, |rate|/nu = 0.20 Nn^5.4 with Nn = 2 pi int D^2 dtheta.
          Conservative by construction (the sum over theta is unchanged),
          self-limiting (broad seas relax slower), and it makes the
          equilibrium spread a PREDICTION rather than a tuned number.

          Pure d^2/dtheta^2 diffusion was ruled out (structurally
          unbounded and a poor fit to the exact residual).  The fully
          extracted companion-kernel target is available as dyn_target
          but is NOT the default: it superimposes unphysical damped
          oscillations on duration-limited growth.
          """
          if self.c_theta <= 0.0:
             return
          sl = self.dsl
          th = self.theta[sl][:,None,None]
          Ei = self.Ed[sl]
          s  = (Ei*self.s_x[sl]).sum(0)
          c  = (Ei*self.s_y[sl]).sum(0)
          thm = np__.arctan2(s, c)
          # |cos|: wrap-immune.  atan2 flips thm by 2 pi at +-pi, which
          # flips the sign of cos(Delta/2); clipping at zero would then
          # blank the whole reference for that step.
          cd = np__.abs(np__.cos(0.5*(th - thm[None,:,:])))
          Etot = Ei.sum(0)[None,:,:]
          Atot = self.Ad[sl].sum(0)[None,:,:]
          if self.narrowness_rate:
             D  = Ei/np__.maximum(Etot*self.dth, 1e-30)
             Nn = np__.minimum(2*np__.pi*np__.sum(D**2, 0)*self.dth,
                               self.Nn_cap)
             if self.dyn_target:
                # variance-matched mirror target (decision record)
                s_state = np__.maximum(((Nn - 0.65)/0.85)**2, 0.5)
                s_star  = 6.0*Nn
                inv_sB  = np__.maximum(2.0/s_state - 1.0/s_star, 0.05)
                s_B = np__.clip(1.0/inv_sB, 0.5, 24.0)[None,:,:]
             else:
                s_B = float(self.ref_spread_s)
             # clean-protocol extraction (calib_operator.py / paper notebook):
             # LSQ fit of the full-range per-direction energy tendency of the
             # exact S_nl against the relaxation form, states cos^{2s}(th/2)
             rate = self.a_theta*Nn[None,:,:]**self.b_theta*nu
          else:
             s_B  = float(self.ref_spread_s)
             rate = self.c_theta*nu
          Dref = _ipow(cd, 2.0*s_B)
          Dref = Dref/np__.maximum(Dref.sum(0)[None,:,:]*self.dth, 1e-30)
          r = 1.0 - np__.exp(-dt*rate)
          self.Ed[sl] = Ei + r*(Etot*self.dth*Dref - Ei)
          self.Ad[sl] = self.Ad[sl] + r*(Atot*self.dth*Dref - self.Ad[sl])
          return
