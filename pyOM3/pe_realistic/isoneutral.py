"""
 Isoneutral (isopycnal) diffusion of temperature and salinity plus the
 optional skew-diffusive (Gent-McWilliams) eddy-driven advection, ported
 1:1 from pyOM2 for_src/isoneutral/isoneutral.f90 (isoneutral_diffusion_pre,
 dm_taper, check_isoneutral_slope_crit) and
 isoneutral_diffusion.f90 (isoneutral_diffusion_trac), i.e. the functional
 (triad) formulation of Griffies et al. as adopted from MOM2.1.

 NOT ported (out of scope for now): the dissipation diagnostics
 (P_diss_iso / P_diss_skew, enable_conserve_energy), the mixed layer
 parameterisation (isoneutral_ml.f90), the TEM formulation
 (isoneutral_friction.f90, enable_TEM_friction) and the diagnostic
 streamfunction B1_gm/B2_gm.

 Options (parameter dict / attributes, pyOM2 names):
   enable_neutral_diffusion : bool, default False. Switches the module on.
   enable_skew_diffusion    : bool, default False. Adds the GM skew flux
                              (antisymmetric part) with diffusivity K_gm.
   K_iso_0     : constant isopycnal diffusivity in m^2/s (default 0)
   K_gm_0      : constant GM (skew) diffusivity in m^2/s (default 0)
   K_iso_steep : lateral diffusivity in steep-slope regions (default 0)
   iso_dslope, iso_slopec : slope taper parameters (default 8e-4, 1e-3)
 K_iso and K_gm are 3-D fields (Nz,Ny,Nx); they are filled with K_iso_0 /
 K_gm_0 at prepare_for_run unless already set as arrays by the setup, and
 may be overwritten at any time (e.g. by a future EKE closure). They are
 halo-filled here with apply_bc, so only the interior needs to be set.

 Time stepping as in pyOM2 thermodynamics.f90: the isoneutral operator is
 applied AFTER the (Adams-Bashforth) advective update and BEFORE the
 vertical mixing, as a forward-Euler increment of the explicit part
 (K_11, K_22, K_13, K_23, K_31, K_32 and the skew terms) followed by a
 backward-Euler solve of the K_33 part (implicit_vdiff of implicit_vmix.py,
 the same tridiagonal operator as pyOM2's solve_tridiag block). pyOM2
 evaluates the explicit fluxes at level tau and adds them to taup1; pe
 holds a single tracer level, so the fluxes are evaluated on the advected
 (new) level. Both are first-order splittings.

 Discrete layout: arrays are (Nz,Ny,Nx) with halo points, k = halo is the
 array bottom, k = -halo-1 the surface cell, rollx(a,-1) = a(i+1),
 rollx(a,1) = a(i-1). The triad arrays Ai_ez etc. are stored stacked as
 (2,2,Nz,Ny,Nx) indexed [ip or jp, kr] exactly as pyOM2's (0:1,0:1).
 pyOM2's bottom special case (k=1: only kr=1 triads, K_iso(k-1) replaced
 by K_iso(k)) is reproduced by replicating K_iso/K_gm into the lower
 z-halo slot and dropping the kr=0 sum at k = halo; the surface face
 (ddzt(nz)=0, K_33(nz)=0, flux_top(nz)=0) is closed explicitly since
 maskW is opened there for the free-surface solver.

 Equation of state: drho/dT and drho/dS at T points as in pyOM2. For
 eq_of_state = 0 the tracer "temp" IS the buoyancy b with
 rho = -b rho_0/g (density.py), so drho/db = -rho_0/g and drho/dS = 0
 (pyOM2 returns 0 there, which would switch the whole operator off);
 eq_of_state = 1 uses the linear constants, 5 the GSW derivatives at
 pressure p0; 100 (ideal-gas atmosphere) is refused.

 Based on class main (needs implicit_vdiff from implicit_vmix).
"""
import pyOM3                    as OM
from   pyOM3.pe_realistic.main     import main
from   pyOM3.density            import (linear_eq_of_state_drhodT,
                                        linear_eq_of_state_drhodS,
                                        gsw_drhodTS)
from   functools                import partial


class isoneutral(main):

      enable_neutral_diffusion = False
      enable_skew_diffusion    = False
      K_iso_0     = 0.0
      K_gm_0      = 0.0
      K_iso_steep = 0.0
      iso_dslope  = 0.0008
      iso_slopec  = 0.001
      _iso_epsln  = 1.0e-20                  # pyOM2 epsln

      # ---------------------------------------------------------------
      def init_isoneutral(self):
          """
          Allocate K_iso / K_gm (unless the setup provided arrays), build
          the constant masks of the operator and run pyOM2's linear
          stability check. Called from prepare_for_run.
          """
          if self.eq_of_state == 100:
              raise NotImplementedError('isoneutral diffusion: eq_of_state=100 '
                                        '(ideal gas atmosphere) has no drho/dT')
          if not hasattr(self, 'K_iso') or OM.np.ndim(self.K_iso) == 0:
              self.K_iso = self.K_iso_0 * OM.np.ones_like(self.temp)
          if not hasattr(self, 'K_gm') or OM.np.ndim(self.K_gm) == 0:
              self.K_gm  = self.K_gm_0  * OM.np.ones_like(self.temp)
          h, Nz = self.halo, self.Nz
          # W-face mask with the surface face and the z-halo closed
          # (pyOM2: ddzt(:,:,nz)=0, K_33(:,:,nz)=0, flux_top(:,:,nz)=0)
          wm = OM.np.zeros_like(self.maskW)
          wm = OM.modify_array(wm, (slice(h, Nz-h-1), slice(None), slice(None)),
                               self.maskW[h:Nz-h-1])
          self._iso_maskW = wm
          # (Nz,1,1) factor dropping the kr=0 triads at the array bottom
          # (pyOM2 k=1 special case)
          bz = OM.np.ones((Nz, 1, 1), OM.prec)
          bz = OM.modify_array(bz, (h, 0, 0), 0.0)
          self._iso_notbot_z = bz
          self.check_isoneutral_slope_crit()
          return

      # ---------------------------------------------------------------
      def check_isoneutral_slope_crit(self):
          """
          pyOM2 check_isoneutral_slope_crit: linear stability criterion
          (Griffies et al.) for the explicit horizontal part with K_iso_0.
          """
          if not self.enable_neutral_diffusion:  return
          h   = self.halo
          ft1 = 1.0/(4.0*self.K_iso_0*self.dt + self._iso_epsln)
          dxt = self.dxt_x[:, :, h:-h]; dyt = self.dyt_y[:, h:-h, :]
          cost = OM.np.abs(self.cost_y[:, h:-h, :]); dzt = self.dzt_z[h:-h]
          d1a = dxt*cost*dzt*ft1
          d1b = dyt*dzt*ft1
          delta_iso1 = min(float(OM.np.min(d1a)), float(OM.np.min(d1b)))
          delta_iso1 = -float(self.global_max(-OM.np.asarray(delta_iso1)))   # model communicator, not COMM_WORLD
          if self.my_pe == 0:
              print(' diffusion grid factor delta_iso1 = %14.7e' % delta_iso1)
          if delta_iso1 < self.iso_slopec:
              raise RuntimeError(
                  'isoneutral diffusion: delta_iso1 = %.4e is the steepest '
                  'isoneutral slope available for linear stability of Redi '
                  'and GM, but iso_slopec = %.4e; integration will be '
                  'unstable (reduce K_iso_0*dt or iso_slopec)'
                  % (delta_iso1, self.iso_slopec))
          return

      # ---------------------------------------------------------------
      @partial(OM.jaxjit, static_argnums=0)
      def _drhodTS(self, temp, salt, p0, maskT):
          """drho/dT and drho/dS at T points times maskT (pyOM2 drdTS)."""
          if   self.eq_of_state == 0:
               drdT = -self.rho_0/self.grav * OM.np.ones_like(temp)
               drdS = OM.np.zeros_like(temp)
          elif self.eq_of_state == 1:
               drdT = linear_eq_of_state_drhodT() * OM.np.ones_like(temp)
               drdS = linear_eq_of_state_drhodS() * OM.np.ones_like(temp)
          elif self.eq_of_state == 5:
               drdT, drdS = gsw_drhodTS(salt, temp, p0)
          else:
               raise NotImplementedError
          return drdT*maskT, drdS*maskT

      def _dm_taper(self, sx):
          """pyOM2 dm_taper: tapering function for isopycnal slopes."""
          return 0.5*(1.0 + OM.np.tanh((-OM.np.abs(sx) + self.iso_slopec)/self.iso_dslope))

      def _fill_lower_zhalo(self, K):
          """K(max(1,k-1)) semantics of pyOM2: replicate the bottom level
          into the lower z-halo slot(s)."""
          h = self.halo
          for k in range(h):
              K = OM.modify_array(K, (k, slice(None), slice(None)), K[h])
          return K

      # ---------------------------------------------------------------
      @partial(OM.jaxjit, static_argnums=0)
      def isoneutral_diffusion_pre(self, temp, salt, p0, K_iso,
                                   maskT, maskU, maskV, maskW_iso):
          """
          pyOM2 isoneutral_diffusion_pre: density triads, tapered slopes
          Ai_ez, Ai_nz (east/north faces) and Ai_bx, Ai_by (top face) and
          the diagonal tensor components K_11, K_22, K_33.
          Returns K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by with the
          Ai arrays stacked as (2,2,Nz,Ny,Nx) = [ip/jp, kr].
          """
          eps  = self._iso_epsln
          rz, rx, ry = self.rollz, self.rollx, self.rolly
          dzw_z = self.dzw_z
          dzw_zm1 = OM.np.roll(self.dzw, 1)[:, None, None]       # dzw(k-1)
          dxu_xm1 = OM.np.roll(self.dxu, 1)[None, None, :]       # dxu(i-1)
          facty_j   = self.cosu_y*self.dyu_y                     # cosu(j) dyu(j)
          facty_jm1 = (OM.np.roll(self.cosu, 1)*OM.np.roll(self.dyu, 1))[None, :, None]

          # drho/dT, drho/dS at T cells
          drdT, drdS = self._drhodTS(temp, salt, p0, maskT)

          # gradients at the top face (closed at the surface), east and
          # north faces of T cells
          ddzt = maskW_iso*(rz(temp, -1) - temp)/dzw_z
          ddzs = maskW_iso*(rz(salt, -1) - salt)/dzw_z
          ddxt = maskU*(rx(temp, -1) - temp)/(self.dxu_x*self.cost_y)
          ddxs = maskU*(rx(salt, -1) - salt)/(self.dxu_x*self.cost_y)
          ddyt = maskV*(ry(temp, -1) - temp)/self.dyu_y
          ddys = maskV*(ry(salt, -1) - salt)/self.dyu_y

          K_iso = self._fill_lower_zhalo(K_iso)
          K_iso_km1 = rz(K_iso, 1)
          notbot = self._iso_notbot_z
          shape5 = (2, 2) + temp.shape
          # shifts by 0 are identities: avoid the copy that roll would make
          def sx(a, n): return a if n == 0 else rx(a, n)
          def sy(a, n): return a if n == 0 else ry(a, n)
          def sz(a, n): return a if n == 0 else rz(a, n)

          # ---- Ai_ez and K_11 on the east face -------------------------
          diffloc = 0.25*(K_iso + K_iso_km1 + rx(K_iso, -1) + rx(K_iso_km1, -1))
          Ai_ez = OM.np.empty(shape5, OM.prec); sumz = 0.0
          for ip in range(2):
              drdT_e, drdS_e = sx(drdT, -ip), sx(drdS, -ip)
              drodxe = drdT_e*ddxt + drdS_e*ddxs
              ddzt_e, ddzs_e = sx(ddzt, -ip), sx(ddzs, -ip)
              for kr in range(2):
                  # ddzt(i+ip, k+kr-1): kr=0 -> face below (k-1), kr=1 -> k
                  drodze = drdT_e*sz(ddzt_e, 1-kr) + drdS_e*sz(ddzs_e, 1-kr)
                  sxe    = -drodxe/(OM.np.minimum(0.0, drodze) - eps)
                  taper  = self._dm_taper(sxe)
                  dzw_kr = dzw_zm1 if kr == 0 else dzw_z
                  fac    = notbot if kr == 0 else 1.0
                  sumz   = sumz + fac*dzw_kr*maskU*OM.np.maximum(self.K_iso_steep, diffloc*taper)
                  Ai_ez  = OM.modify_array(Ai_ez, (ip, kr), taper*sxe*maskU)
          K_11  = sumz/(4.0*self.dzt_z)

          # ---- Ai_nz and K_22 on the north face ------------------------
          diffloc = 0.25*(K_iso + K_iso_km1 + ry(K_iso, -1) + ry(K_iso_km1, -1))
          Ai_nz = OM.np.empty(shape5, OM.prec); sumz = 0.0
          for jp in range(2):
              drdT_n, drdS_n = sy(drdT, -jp), sy(drdS, -jp)
              drodyn = drdT_n*ddyt + drdS_n*ddys
              ddzt_n, ddzs_n = sy(ddzt, -jp), sy(ddzs, -jp)
              for kr in range(2):
                  drodzn = drdT_n*sz(ddzt_n, 1-kr) + drdS_n*sz(ddzs_n, 1-kr)
                  syn    = -drodyn/(OM.np.minimum(0.0, drodzn) - eps)
                  taper  = self._dm_taper(syn)
                  dzw_kr = dzw_zm1 if kr == 0 else dzw_z
                  fac    = notbot if kr == 0 else 1.0
                  sumz   = sumz + fac*dzw_kr*maskV*OM.np.maximum(self.K_iso_steep, diffloc*taper)
                  Ai_nz  = OM.modify_array(Ai_nz, (jp, kr), taper*syn*maskV)
          K_22  = sumz/(4.0*self.dzt_z)

          # ---- Ai_bx, Ai_by and K_33 on the top face -------------------
          Ai_bx = OM.np.empty(shape5, OM.prec); Ai_by = OM.np.empty(shape5, OM.prec)
          sumx = 0.0; sumy = 0.0
          KW = K_iso*maskW_iso
          for kr in range(2):
              drdT_b, drdS_b = sz(drdT, -kr), sz(drdS, -kr)     # (k+kr)
              drodzb = drdT_b*ddzt + drdS_b*ddzs
              rden   = 1.0/(OM.np.minimum(0.0, drodzb) - eps)
              ddxt_b, ddxs_b = sz(ddxt, -kr), sz(ddxs, -kr)
              ddyt_b, ddys_b = sz(ddyt, -kr), sz(ddys, -kr)
              for ip in range(2):
                  # face i-1+ip: ip=0 -> west face (i-1), ip=1 -> east face (i)
                  drodxb = drdT_b*sx(ddxt_b, 1-ip) + drdS_b*sx(ddxs_b, 1-ip)
                  sxb    = -drodxb*rden
                  taper  = self._dm_taper(sxb)
                  dxu_ip = dxu_xm1 if ip == 0 else self.dxu_x
                  tsxb   = taper*sxb
                  sumx   = sumx + dxu_ip*KW*tsxb*sxb
                  Ai_bx  = OM.modify_array(Ai_bx, (ip, kr), tsxb*maskW_iso)
                  # face j-1+jp
                  drodyb = drdT_b*sy(ddyt_b, 1-ip) + drdS_b*sy(ddys_b, 1-ip)
                  syb    = -drodyb*rden
                  taper  = self._dm_taper(syb)
                  facty  = facty_jm1 if ip == 0 else facty_j
                  tsyb   = taper*syb
                  sumy   = sumy + facty*KW*tsyb*syb
                  Ai_by  = OM.modify_array(Ai_by, (ip, kr), tsyb*maskW_iso)
          K_33  = sumx/(4.0*self.dxt_x) + sumy/(4.0*self.dyt_y*self.cost_y)
          return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by

      # ---------------------------------------------------------------
      @partial(OM.jaxjit, static_argnums=0)
      def isoneutral_diffusion_tendency(self, tr, K_iso, K_gm, K_11, K_22,
                                        Ai_ez, Ai_nz, Ai_bx, Ai_by,
                                        maskT, maskU, maskV, maskW_iso):
          """
          Explicit part of pyOM2 isoneutral_diffusion_trac: the total
          (isoneutral plus skew) tracer fluxes at the east, north and top
          faces of T cells and their divergence. K_gm must be zero when
          skew diffusion is off (the caller passes zeros). The K_33 part
          is treated implicitly by the caller.
          """
          rz, rx, ry = self.rollz, self.rollx, self.rolly
          K_iso = self._fill_lower_zhalo(K_iso)
          K_gm  = self._fill_lower_zhalo(K_gm)
          K_iso_km1, K_gm_km1 = rz(K_iso, 1), rz(K_gm, 1)

          # vertical tracer differences at the top face (kr=1) and the
          # face below (kr=0); zero at the array bottom / surface as
          # pyOM2's max(1,k-1) / min(k+kr,nz) clamps
          dz1 = (rz(tr, -1) - tr)*maskW_iso
          dz0 = rz(dz1, 1)
          # horizontal differences at face i (ip=1) / i-1 (ip=0) etc.
          dx1 = (rx(tr, -1) - tr)*maskU;  dx0 = rx(dx1, 1)
          dy1 = (ry(tr, -1) - tr)*maskV;  dy0 = ry(dy1, 1)

          # ---- east face ----------------------------------------------
          diffloc = 0.25*(K_iso + K_iso_km1 + rx(K_iso, -1) + rx(K_iso_km1, -1)) \
                  - 0.25*(K_gm  + K_gm_km1  + rx(K_gm,  -1) + rx(K_gm_km1,  -1))
          sumz = 0.0
          for ip in range(2):
              dz1_e, dz0_e = rx(dz1, -ip), rx(dz0, -ip)
              sumz = sumz + diffloc*(Ai_ez[ip, 0]*dz0_e + Ai_ez[ip, 1]*dz1_e)
          fe = sumz/(4.0*self.dzt_z) + dx1/(self.cost_y*self.dxu_x)*K_11

          # ---- north face ---------------------------------------------
          diffloc = 0.25*(K_iso + K_iso_km1 + ry(K_iso, -1) + ry(K_iso_km1, -1)) \
                  - 0.25*(K_gm  + K_gm_km1  + ry(K_gm,  -1) + ry(K_gm_km1,  -1))
          sumz = 0.0
          for jp in range(2):
              dz1_n, dz0_n = ry(dz1, -jp), ry(dz0, -jp)
              sumz = sumz + diffloc*(Ai_nz[jp, 0]*dz0_n + Ai_nz[jp, 1]*dz1_n)
          fn = self.cosu_y*(sumz/(4.0*self.dzt_z) + dy1/self.dyu_y*K_22)

          # ---- top face (K_31, K_32 and skew parts, explicit) ---------
          diffloc = K_iso + K_gm
          cosu_jm1 = OM.np.roll(self.cosu, 1)[None, :, None]
          sumx = 0.0; sumy = 0.0
          for kr in range(2):
              dx0_b, dx1_b = rz(dx0, -kr), rz(dx1, -kr)          # (k+kr)
              dy0_b, dy1_b = rz(dy0, -kr), rz(dy1, -kr)
              sumx = sumx + diffloc/self.cost_y*(Ai_bx[0, kr]*dx0_b + Ai_bx[1, kr]*dx1_b)
              sumy = sumy + diffloc*(Ai_by[0, kr]*cosu_jm1*dy0_b + Ai_by[1, kr]*self.cosu_y*dy1_b)
          ft = (sumx/(4.0*self.dxt_x) + sumy/(4.0*self.dyt_y*self.cost_y))*maskW_iso

          return maskT*( (fe - rx(fe, 1))/(self.cost_y*self.dxt_x)
                        +(fn - ry(fn, 1))/(self.cost_y*self.dyt_y)
                        +(ft - rz(ft, 1))/self.dzt_z )

      # ---------------------------------------------------------------
      def isoneutral_diffusion(self):
          """
          Apply isoneutral (and optionally skew) diffusion to temp and
          salt: pyOM2 isoneutral_diffusion_pre followed by
          isoneutral_diffusion_trac for each tracer (explicit forward-Euler
          increment, then the implicit K_33 solve).
          """
          if not self.enable_neutral_diffusion:  return
          K_iso = self.apply_bc(self.K_iso)
          K_gm  = self.apply_bc(self.K_gm) if self.enable_skew_diffusion \
                  else OM.np.zeros_like(K_iso)
          self.K_11, self.K_22, self.K_33, self.Ai_ez, self.Ai_nz, self.Ai_bx, self.Ai_by = \
                 self.isoneutral_diffusion_pre(self.temp, self.salt, self.p0, K_iso,
                                               self.maskT, self.maskU, self.maskV,
                                               self._iso_maskW)
          def _step(tr):
              tend = self.isoneutral_diffusion_tendency(tr, K_iso, K_gm,
                                 self.K_11, self.K_22,
                                 self.Ai_ez, self.Ai_nz, self.Ai_bx, self.Ai_by,
                                 self.maskT, self.maskU, self.maskV, self._iso_maskW)
              tr = self.apply_bc(tr + self.dt*tend)
              return self.implicit_vdiff(tr, self.K_33, self.maskT)
          self.temp = _step(self.temp)
          if self.eq_of_state > 0 and self.eq_of_state != 100:
              self.salt = _step(self.salt)
          return
