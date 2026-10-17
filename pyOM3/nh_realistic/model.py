"""
 Brings together all components of the non-hydrostatic model and defines
 the time steps.

 The two time steps are pe_realistic's time_step_implicit_pressure and
 time_step_explicit_pressure with the vertical momentum equation and the
 3-D projection inserted; the inserted lines are marked "nh:". Everything
 else is kept line by line, so that enable_nonhydrostatic = False
 reproduces pe_realistic to round-off (test/nh/realistic.py checks this).

 Order of one step (implicit path):
   tracer tendencies (with the divergence-free w of the last step)
   u, v tendencies (advection, Coriolis, -grad p_h)       pe_realistic
   nh: w tendency (advection)
   set_forcing, horizontal friction of u, v (nh: and of w)
   AB3 update of u, v (nh: and of w), set_forcing_direct
   surface pressure solve (rigid lid / implicit free surface)
   vertical friction of u, v (nh: and of w), bottom friction
   nh: surface w from the column divergence (free surface; zero under
       the rigid lid), then the 3-D solve for p_nh and the correction
       of u, v, w on the interior faces -- instead of the diagnostic w
   tracer update, isoneutral mixing, vertical mixing, density, TKE

 Based on diverse other classes.
"""
import pyOM3 as OM
from   pyOM3.nh_realistic.momentum       import momentum
from   pyOM3.nh_realistic.pressure       import pressure
from   pyOM3.nh_realistic.vmix           import vmix_w
from   pyOM3.pe_realistic.model          import model as model_pe
from   pyOM3.density                     import get_rho
import numpy                             as np__


class model(momentum,pressure,vmix_w,model_pe):
      # thermodynamics, free_surface, damping, implicit_vmix, isoneutral,
      # vadv_split come with model_pe; momentum and pressure are the
      # nh_realistic ones (which inherit pe_realistic's)

      # set_coriolis, set_topography, set_initial_conditions*, loop: pe_realistic's

      def prepare_for_run(self):
          """
          pe_realistic's preparation plus the coefficients of the
          non-hydrostatic operator, the number of wet cells for the gauge
          and the solver parameters: epsilon_nh, the largest divergence
          [1/s] the solve leaves (default 1e-12), max_iterations_nh
          (default: max_iterations of the surface solve).
          """
          super().prepare_for_run()
          if not hasattr(self,'epsilon_nh'):        self.epsilon_nh        = 1e-12
          if not hasattr(self,'max_iterations_nh'): self.max_iterations_nh = self.max_iterations
          self.cf_nh = self.make_coeff_non_hydro(self.cf_nh, self.maskU, self.maskV,
                                                 self.maskWc, self.maskT)
          self.n_wet = self._dot(self.maskT, self.maskT)
          self.congr_itts_nh, self.estimated_error_nh = 0, 0.0
          return

      def nh_projection(self):
          """
          nh: the vertical velocity at the end of the step. Without the
          non-hydrostatic option, the diagnostic w of pe_realistic. With
          it, the surface face gets the free-surface w from the column
          divergence (zero under the rigid lid), and the 3-D solve makes
          (u,v,w) divergence-free on the interior faces.
          """
          w_diag = self.vertical_velocity(self.u,self.v,self.w,self.maskW)
          if not self.enable_nonhydrostatic:
             self.w = w_diag
             return
          ks = -(self.halo+1)
          self.w = OM.modify_array(self.w, (ks,slice(None),slice(None)), w_diag[ks])
          self.u, self.v, self.w, self.p_nh, self.estimated_error_nh, self.congr_itts_nh = \
                 self.solve_for_nh_pressure(self.u,self.v,self.w,self.p_nh,
                                            self.maskU,self.maskV,self.maskWc,self.maskT,
                                            self.cf_nh,self.n_wet)
          return

      def time_step_explicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the split-explicit method after Demange et al (2019).
          Chosen with self.explicit_free_surface = True
          """
          if getattr(self,'enable_tke',False): self.set_tke_diffusivities()
          self.du, self.dv,self.du_cor,self.dv_cor,self.p_h = \
                         self.momentum_tendency_expl(self.u,self.v,self.w, self.du,self.dv,
                                                     self.du_cor,self.dv_cor,self.rho,
                                                     self.maskU,self.maskV,self.maskW,self.maskT,
                                                     self.coriolis_t,tau)
          # nh: advective tendency of w
          if self.enable_nonhydrostatic:
             self.dw = self.w_tendency(self.u,self.v,self.w,self.dw,self.maskU,self.maskV,self.maskWc,tau)

          # allow for user defined momentum forcing, added to du[tau],dv[tau]
          # (see pe_realistic/model.py for the tracer forcing bookkeeping)
          if hasattr(self,'set_forcing'):
             _dtemp0, _dsalt0 = self.dtemp*1, self.dsalt*1
             self.set_forcing(tau)
             _ftemp, _fsalt = self.dtemp - _dtemp0, self.dsalt - _dsalt0
          else:
             _ftemp = _fsalt = 0.0

          # nh: horizontal friction and AB3 update of w, before the surface
          # solve as for u and v on the implicit path
          if self.enable_nonhydrostatic:
             self.horizontal_friction_w()
             self.w = self.update_w_AB3(A,B,C,tau,taum1,taum2,self.w,self.dw,self.maskWc)

          self.u, self.v = self.bottom_friction(self.u,self.v,self.maskU,self.maskV)   # no-op unless enabled
          if not hasattr(self,'U_bt'):       # cold start (or restart without it): from the 3-D field
             self.U_bt, self.V_bt = self.init_barotropic_state(self.u,self.v,self.maskU,self.maskV)
          self.u, self.v,  self.p_s, self.U_bt, self.V_bt = self.solve_expl_surface(self.u,self.v,
                                                              self.du,self.dv,self.du_cor,self.dv_cor,
                                                              self.p_h,self.p_s,self.U_bt,self.V_bt,
                                                              self.maskU,self.maskV,self.maskW,self.maskT,
                                                              self.hu,self.hv,self.coriolis_t,
                                                              A,B,C,tau,taum1,taum2)

          self.vertical_friction()        # under TKE also diagnoses K_diss_v
          if self.enable_nonhydrostatic: self.vertical_friction_w()      # nh
          self.nh_projection()                                           # nh: replaces vertical_velocity

          _split = getattr(self,'vadv_sub_cycling',0) > 0
          if _split: self.dtemp = self.vadv_zero_tendency(self.dtemp,tau)
          else:
            self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                               self.maskU,self.maskV,self.maskW,self.maskT,tau)
          if hasattr(self,'Khbi'):
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             if _split: self.dsalt = self.vadv_zero_tendency(self.dsalt,tau)
             else:
               self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau)
             if hasattr(self,'Khbi'):
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)

          self.dtemp, self.dsalt = self.dtemp + _ftemp, self.dsalt + _fsalt
          if _split: self.vadv_update_tracers(A,B,C,tau,taum1,taum2)
          else:
            self.temp, self.salt = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                            self.temp,self.salt,self.dtemp,self.dsalt,
                                                                            self.p0,self.maskT)
          self.isoneutral_diffusion()
          self.vertical_mixing()
          self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if getattr(self,'enable_tke',False): self.integrate_tke()
          return


      def time_step_implicit_pressure(self,A,B,C,tau,taum1,taum2):
          """
          A single time step for the rigid lid or implicit free surface method.
          """
          if getattr(self,'enable_tke',False): self.set_tke_diffusivities()

          _split = getattr(self,'vadv_sub_cycling',0) > 0
          if _split: self.dtemp = self.vadv_zero_tendency(self.dtemp,tau)
          else:
            self.dtemp = self.tracer_advection(self.temp,self.dtemp,self.u,self.v,self.w,
                                               self.maskU,self.maskV,self.maskW,self.maskT,tau)
          if hasattr(self,'Khbi'):
             self.temp = self.biharmonic_horizontal(self.temp,self.Khbi,self.maskT)
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             if _split: self.dsalt = self.vadv_zero_tendency(self.dsalt,tau)
             else:
               self.dsalt = self.tracer_advection(self.salt,self.dsalt,self.u,self.v,self.w,
                                                  self.maskU,self.maskV,self.maskW,self.maskT,tau)
             if hasattr(self,'Khbi'):
                self.salt = self.biharmonic_horizontal(self.salt,self.Khbi,self.maskT)

          self.du, self.dv, self.p_h = self.momentum_tendency_impl(self.u,self.v,self.w,
                                                                   self.du,self.dv,self.rho,self.p_h,
                                                                   self.maskU,self.maskV,self.maskW,self.maskT,
                                                                   self.coriolis_t,tau)
          # nh: advective tendency of w
          if self.enable_nonhydrostatic:
             self.dw = self.w_tendency(self.u,self.v,self.w,self.dw,self.maskU,self.maskV,self.maskWc,tau)

          # allow for user defined momentum forcing, added to du[tau],dv[tau] (nh: and dw[tau])
          if hasattr(self,'set_forcing'): self.set_forcing(tau)

          if hasattr(self,'Ahbi'):
                       self.u, self.v = self.biharmonic_momentum(self.u,self.v,
                                                self.Ahbi,self.maskU,self.maskV)
          if hasattr(self,'Ah'):
                       self.u, self.v = self.harmonic_momentum(self.u,self.v,
                                                self.Ah,self.maskU,self.maskV)
          if self.enable_nonhydrostatic: self.horizontal_friction_w()    # nh

          self.u, self.v = self.update_momentum_AB3(A,B,C,tau,taum1,taum2,
                                                    self.u,self.v,self.du,self.dv,self.maskU,self.maskV)
          if self.enable_nonhydrostatic:                                  # nh
             self.w = self.update_w_AB3(A,B,C,tau,taum1,taum2,self.w,self.dw,self.maskWc)

          # user-defined DIRECT momentum increment, bypassing the AB weights
          # (must precede the pressure solves, see pe_realistic/model.py)
          if hasattr(self,'set_forcing_direct'): self.set_forcing_direct()

          self.u, self.v, self.p_s, self.estimated_error, self.congr_itts  = \
                             self.solve_for_pressure(self.u,self.v,self.p_s,
                                                     self.maskU,self.maskV,self.maskT,self.cf,tau,taum1,taum2)
          self.vertical_friction()
          if self.enable_nonhydrostatic: self.vertical_friction_w()      # nh
          self.u, self.v = self.bottom_friction(self.u,self.v,self.maskU,self.maskV)   # no-op unless enabled
          self.nh_projection()                                           # nh: replaces vertical_velocity

          if _split: self.vadv_update_tracers(A,B,C,tau,taum1,taum2)
          else:
            self.temp, self.salt = self.update_thermodynamics_AB3(A,B,C,tau,taum1,taum2,
                                                                           self.temp,self.salt,self.dtemp,self.dsalt,
                                                                           self.p0,self.maskT)
          self.isoneutral_diffusion()
          self.vertical_mixing()
          self.rho = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          if getattr(self,'enable_tke',False): self.integrate_tke()
          return

      def diagnose(self,n,tau):
          """
          Dummy routine for diagnostics.
          """
          t =  n*self.dt
          if self.my_pe == 0:
                if hasattr(self,'congr_itts'):
                     print ("t=",t," n=",n," itts=",self.congr_itts," itts_nh=",self.congr_itts_nh)
                else: print ("t=",t," n=",n)
