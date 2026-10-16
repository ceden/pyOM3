"""
 Brings together all components of the wim_idemix model.
 Defines the time step and a loop over time steps.

 Based on class propagation and class source.
"""
import numpy as np__
from   pyOM3.wim_idemix.propagation import propagation
from   pyOM3.wim_idemix.source      import source
from   pyOM3.wim_idemix.interface   import interface


class model(propagation, source, interface):


      def time_step(self, ustar, udir):
          """
          One model time step: propagation, then the source update.

          The SOURCE UPDATE IS SUB-CYCLED so that the solution does not
          depend on dt.  The growth turnover of a young sea is stiff and
          self-triggering: the wind input is explicit
          (implicit_dissipation_only) while the saturation-excess sink is
          QUADRATIC in the excess, so a step that does not resolve the
          turnover overshoots into strong over-saturation, the enhanced
          sink fires, and the column collapses into a high-sigbar state
          it cannot leave (energy flat for hours, stress meaningless).
          The number of sub-steps is sized from the actual source rates,
          so outside the turnover it is 1 and costs nothing.

          ustar, udir: friction velocity and wind direction, either
          scalars or arrays broadcastable onto (Ny,Nx).  udir is in the
          wim/WAM convention (k_hat = (sin theta, cos theta)), so wim
          wind directions are handed over unconverted.
          """
          self._last_ustar = ustar
          self._last_udir  = udir
          self._grad_cache = None    # the state moved: sampled gradients
          self._wr_cache   = None    # and the cell weights carry kbar too
          if (self.nx > 1 or self.ny > 1 or self.theta_dot is not None
              or self.theta_dot_A is not None
              or self.gamma_ex is not None):
             self.propagate()

          n_sub  = self.source_substeps(ustar, udir)
          dt_sub = self.dt/n_sub
          tin = np__.zeros((2, self.Ny, self.Nx))
          tou = np__.zeros((2, self.Ny, self.Nx))
          win = np__.zeros((self.Ny, self.Nx))
          phi = np__.zeros((self.Ny, self.Nx))
          for _ in range(n_sub):
              self.add_sources(ustar, udir, dt_sub)
              tin += self.tau_in_v/n_sub      # time average over the
              tou += self.tau_o_v /n_sub      # sub-cycle, as vectors
              win += self._win_sub/n_sub
              phi += self._phi_sub/n_sub
          self.tau_in_v, self.tau_o_v = tin, tou
          self.win_g, self.phi_g = win, phi     # kinematic energy fluxes m^2/s
          # The source cycle writes the REAL bins only, so refresh the ring
          # halo from them: purely local (no MPI, no x/y exchange), and it
          # keeps the padded state self-consistent even when propagate()
          # was skipped, so anything reading Ad/Ed between steps sees a
          # ring that agrees with the interior.
          self.Ad = self.wrap_theta(self.Ad)
          self.Ed = self.wrap_theta(self.Ed)
          # waves do not exist on land (mirror of wim_simple's maskTp step)
          if hasattr(self, 'maskTp'):
              self.Ad = self.Ad*self.maskTp[None, :, :]
              self.Ed = self.Ed*self.maskTp[None, :, :]
          self.tau_in_g = np__.sqrt(tin[0]**2 + tin[1]**2)
          self.tau_o_g  = np__.sqrt(tou[0]**2 + tou[1]**2)
          self.n_sub = n_sub
          return


      def loop(self, ustar, udir, max_steps = 1, diagnose = True):
          """
          Loop over max_steps time steps, calling self.diagnose(n) after
          each step when diagnose is set (snapshot classes override it).
          """
          for n in range(max_steps):
              self.time_step(ustar, udir)
              self.itt = getattr(self, 'itt', 0) + 1
              if diagnose and hasattr(self, 'diagnose'):
                 self.diagnose(self.itt)
          return
