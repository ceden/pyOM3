"""
 Interface of the wim_idemix model to the ocean.

 Mirrors pyOM3.wim_simple.interface: the ocean supplies the Lagrangian
 velocity profile (UL, VL) on its own uniform vertical grid (dz, nz,
 index 0 = bottom, top face at z = 0, as in wim_simple._wave_kernel),
 and the wave model projects it with its vertical kernels.

 The simplified model does not carry U(k).  It needs the kernel-weighted
 samplings of the paper (eq:Aeq, eq:Eeq),

     U_A = 2 kbar int u^L q_A(2 kbar z) dz    (action equation)
     U_E = 2 kbar int u^L q_E(2 kbar z) dz    (energy equation)

 which are fields per direction, since kbar = sigbar^2/(g s_1/2^2) is
 per direction.  NOTE the state dependence: in wim_simple U(f) depends
 on the ocean state only, here the sampling depth depends on the sea
 state, so calc_doppler_vel must be called again when the wave state
 has evolved (every coupling step).

 Each ocean cell receives its exact cell integral of the kernel,
 r_X(zeta_top) - r_X(zeta_bot) with the antiderivatives r_X tabulated
 in shape.py (zeta = 2 kbar z).  The column sum telescopes, so a
 barotropic current is projected exactly: U_A = U_E = u^L, independent
 of dz.  With the attribute ``closed_bottom`` the bottom cell is
 extended to z -> -infinity (r_X(zeta_bot) -> 0) as in wim_simple, so
 the column weights sum to one exactly; otherwise the fraction
 r_X(2 kbar z_floor) is lost below the floor.

 Based on class main.
"""
import numpy as np__
from   pyOM3.wim_idemix.main import main


class interface(main):


      def _zeta_faces(self, dz, nz):
          """Ocean cell faces (index 0 = bottom, top face = 0), as in
          wim_simple._wave_kernel."""
          z  = np__.arange(nz)*dz
          z  = z - z[-1] - dz/2
          return z + dz/2, z - dz/2               # zt, zb

      def calc_doppler_vel(self, dz, nz, UL, VL):
          """
          Kernel-weighted samplings of the ocean current.

          UL, VL: Lagrangian velocity components on the ocean grid,
          shape (nz, ny_loc, nx_loc), unpadded, index 0 = bottom.

          Returns (U_A, U_E), each of shape (2, nd, Ny, Nx) with the
          component axis first (U_A[0] = x, U_A[1] = y), padded with
          halos applied, ready to assign to model.U_A and model.U_E,
          which switches the current advection on in propagate().
          """
          h  = self.halo
          mc = self._z_cut_level(dz, nz)       # top levels that matter
          out = []
          for ws in self._cell_weights_r(dz, nz):
              wA = np__.zeros((2, self.nd, self.ny_loc, self.nx_loc))
              for k_, m_ in enumerate(range(nz - mc, nz)):  # 0 = bottom
                  w = ws[k_]                        # exact cell integral
                  wA[0] += UL[m_][None,:,:]*w
                  wA[1] += VL[m_][None,:,:]*w
              out.append(wA)
          UA, UE = out
          pad = ((0,0), (h,h), (h,h), (h,h))   # ring halo padded here,
          UA = np__.pad(UA, pad); UE = np__.pad(UE, pad)   # filled below
          UA[0] = self.apply_bc(UA[0]); UA[1] = self.apply_bc(UA[1])
          UE[0] = self.apply_bc(UE[0]); UE[1] = self.apply_bc(UE[1])
          return UA, UE

      def calc_doppler_disp(self, dz, nz, UL, VL):
          """
          Dispersion samplings of the ocean current (paper eq:Aeq,
          eq:Eeq),

              V_X = 2 kbar int u^L p_X(2 kbar z) dz ,   X = A, E

          with the dispersion kernels p_X = d/dzeta (zeta q_X).  The
          transport velocity gains s (s . V_X) along the wave
          direction.  Same call convention as calc_doppler_vel; the
          companion of wim_simple.calc_doppler_vel_df, which feeds the
          k-dependence of the Doppler velocity there.

          The cell integral of p_X is exact with no extra table,
              int_cell p_X dzeta = zeta_t q_X(zeta_t) - zeta_b q_X(zeta_b) ,
          and it telescopes to zero over the column (zeta q_X vanishes
          at both ends), so a depth-uniform current gives V_X = 0
          exactly, independent of dz.  With ``closed_bottom`` the
          bottom face is taken at zeta -> -infinity.

          Returns (V_A, V_E), each (2, nd, Ny, Nx), padded with halos
          applied, ready to assign to model.V_A and model.V_E.
          """
          h  = self.halo
          zt, zb = self._zeta_faces(dz, nz)
          kb = self.sigbar()[self.dsl, h:-h, h:-h]**2 \
             / (self.grav*self.s_half**2)
          mc = self._z_cut_level(dz, nz)       # top levels that matter
          out = []
          for q_tab in (self._qA_tab, self._qE_tab):
              wA = np__.zeros((2, self.nd, self.ny_loc, self.nx_loc))
              for m_ in range(nz - mc, nz):    # index 0 = bottom
                  ztt = 2*kb*zt[m_]
                  ft  = ztt*np__.interp(ztt, self._zeta_tab, q_tab,
                                        left=0.0)
                  if getattr(self, 'closed_bottom', False) and m_ == 0:
                     fb = 0.0*ft
                  else:
                     zbb = 2*kb*zb[m_]
                     fb  = zbb*np__.interp(zbb, self._zeta_tab, q_tab,
                                           left=0.0)
                  w = ft - fb                   # exact cell integral of p_X
                  wA[0] += UL[m_][None,:,:]*w
                  wA[1] += VL[m_][None,:,:]*w
              out.append(wA)
          VA, VE = out
          pad = ((0,0), (h,h), (h,h), (h,h))   # ring halo padded here,
          VA = np__.pad(VA, pad); VE = np__.pad(VE, pad)   # filled below
          VA[0] = self.apply_bc(VA[0]); VA[1] = self.apply_bc(VA[1])
          VE[0] = self.apply_bc(VE[0]); VE[1] = self.apply_bc(VE[1])
          return VA, VE

      def _cell_weights_r(self, dz, nz):
          """Exact vertical cell integrals of q_A and q_E, w = r(zeta_t) -
          r(zeta_b), for every level in the z_cut range.

          calc_doppler_vel and _sampled_gradients need the SAME weights --
          same faces, same kbar, same two tables -- and each used to build
          them from scratch, which is two of the level loops that dominate
          the sampling cost. Cached for the coupling interval and cleared
          by time_step, like the gradients: the weights carry kbar.
          Returns a list [w_A, w_E], each (mc, nd, ny_pad, nx_pad), in the
          level order the callers loop over.
          """
          key = (dz, nz, self.Ad.shape)
          hit = getattr(self, '_wr_cache', None)
          if hit is not None and hit[0] == key:
              return hit[1]
          h  = self.halo
          kb = self.sigbar()[self.dsl, h:-h, h:-h]**2 \
             / (self.grav*self.s_half**2)          # (nd, ny_loc, nx_loc)
          mc = self._z_cut_level(dz, nz)
          zt, zb = self._zeta_faces(dz, nz)
          out = []
          for r_tab in (self._rA_tab, self._rE_tab):
              ws = []
              for m_ in range(nz - mc, nz):
                  rt = np__.interp(2*kb*zt[m_], self._zeta_tab, r_tab,
                                   left=0.0)
                  if getattr(self, 'closed_bottom', False) and m_ == 0:
                     rb = 0.0*rt
                  else:
                     rb = np__.interp(2*kb*zb[m_], self._zeta_tab, r_tab,
                                      left=0.0)
                  ws.append(rt - rb)
              out.append(ws)
          self._wr_cache = (key, out)
          return out

      def _sampled_gradients(self, dz, nz, UL, VL):
          """
          Kernel-weighted vertical samplings of the horizontal current
          gradients,

              G_X[i,j] = 2 kbar int (d u^L_j / d x_i) q_X(2 kbar z) dz

          for X = A, E, with i, j in (x, y).  The gradients are central
          differences of UL, VL on the wave grid (one halo update on
          the padded level block), the vertical integral is the exact
          cell integral via the antiderivatives r_X, as in
          calc_doppler_vel.  Returns (G_A, G_E), each (2, 2, nd,
          ny_loc, nx_loc), unpadded, first index i, second j.
          """
          # CACHED WITHIN ONE COUPLING INTERVAL. calc_refraction and
          # calc_exchange both need these gradients and the coupler always
          # calls them back to back on the same UL, VL: computing them
          # twice cost 4.95 ms of the 17.95 ms sampling time plus two halo
          # exchanges on the largest array in the operation.
          # The cache is cleared by every time_step, because the vertical
          # weights depend on kbar through the state. That is what makes
          # the key safe: id() alone would not be, since a freed array's
          # address can be handed to the next one of the same size.
          key = (id(UL), id(VL), dz, nz, np__.shape(UL))
          hit = getattr(self, '_grad_cache', None)
          if hit is not None and hit[0] == key:
              return hit[1]
          h = self.halo
          pad = ((0,0), (h,h), (h,h))
          U = self.apply_bc(np__.pad(np__.asarray(UL, float), pad),
                            wrap_d=False)
          V = self.apply_bc(np__.pad(np__.asarray(VL, float), pad),
                            wrap_d=False)
          il = slice(h, -h)
          grads = []                          # (dxu, dyu, dxv, dyv)
          for F in (U, V):
              gx = ((np__.roll(F, -1, -1) - np__.roll(F, 1, -1))
                    /(2*self.dx))[:, il, il]
              gy = ((np__.roll(F, -1, -2) - np__.roll(F, 1, -2))
                    /(2*self.dy))[:, il, il]
              grads.append(gx); grads.append(gy)
          zt, zb = self._zeta_faces(dz, nz)
          mc = self._z_cut_level(dz, nz)       # top levels that matter
          out = []
          # the SAME cell weights calc_doppler_vel uses -- same faces,
          # same kbar, same two tables -- so they are built once
          for ws in self._cell_weights_r(dz, nz):
              G = np__.zeros((2, 2, self.nd, self.ny_loc, self.nx_loc))
              for k_, m_ in enumerate(range(nz - mc, nz)):  # 0 = bottom
                  w = ws[k_]
                  G[0,0] += grads[0][m_][None,:,:]*w    # dx u
                  G[1,0] += grads[1][m_][None,:,:]*w    # dy u
                  G[0,1] += grads[2][m_][None,:,:]*w    # dx v
                  G[1,1] += grads[3][m_][None,:,:]*w    # dy v
              out.append(G)
          self._grad_cache = (key, (out[0], out[1]))
          return out[0], out[1]

      def calc_refraction(self, dz, nz, UL, VL):
          """
          Sampled-shear refraction rates of the coupling (paper
          eq:Aeq, eq:Eeq),

              theta_dot_X = - 2 kbar s_j int (s_perp . grad) u^L_j
                                            q_X(2 kbar z) dz

          with s = (sin th, cos th) and s_perp = (cos th, -sin th).
          For a depth-uniform current both rates reduce to the ray rate
          - s_j (s_perp . grad) U_j exactly; with vertical shear they
          differ, the energy rate sampling the current closer to the
          surface than the action rate.

          Same call convention as calc_doppler_vel.  Returns
          (theta_dot_A, theta_dot_E), each (nd, Ny, Nx), padded with
          halos applied, ready to assign to model.theta_dot_A and
          model.theta_dot_E.
          """
          h = self.halo
          GA, GE = self._sampled_gradients(dz, nz, UL, VL)
          sn = self.s_x[self.dsl]
          cs = self.s_y[self.dsl]
          out = []
          for G in (GA, GE):
              # (s_perp . grad) u_j = cos th dx u_j - sin th dy u_j
              du = cs*G[0,0] - sn*G[1,0]
              dv = cs*G[0,1] - sn*G[1,1]
              td = -(sn*du + cs*dv)
              td = np__.pad(td, ((h,h),(h,h),(h,h)))
              out.append(self.apply_bc(td))
          return out[0], out[1]

      def calc_exchange(self, dz, nz, UL, VL):
          """
          Conservative energy exchange rate with the mean flow (paper
          eq:gamma-ex),

              gamma_ex = - kbar s_i s_j int (d u^L_j / d x_i)
                                          q_E(2 kbar z) dz ,

          the moment of the transport in k: it changes the energy of
          the pair at fixed action, which is the frequency refraction
          of the simplified model.  The same energy appears in the mean
          flow with opposite sign.  For a jet U(y) and a wave at angle
          theta the rate is - (1/2) sin th cos th dU/dy, which
          reproduces the ray result sigma = Omega - U k_x to first
          order in U/c.

          Same call convention as calc_doppler_vel.  Returns gamma_ex,
          shape (nd, Ny, Nx), padded with halos applied, ready to
          assign to model.gamma_ex.
          """
          h = self.halo
          GA, GE = self._sampled_gradients(dz, nz, UL, VL)
          sn = self.s_x[self.dsl]
          cs = self.s_y[self.dsl]
          g = -0.5*(sn*sn*GE[0,0] + sn*cs*GE[1,0]
                    + cs*sn*GE[0,1] + cs*cs*GE[1,1])
          g = np__.pad(g, ((h,h),(h,h),(h,h)))
          return self.apply_bc(g)

      def calc_mean_exchange(self, dz, nz, UL, VL):
          """
          Paper-form momentum exchange rates with the mean flow
          (paper eq:Mex, eq:Mrot),

              gamma_M     = - 2 kbar s_i s_j int (d_i u^L_j)
                                              q_M(2 kbar z) dz
              thetadot_M  = - 2 kbar s_j int (s_perp . grad) u^L_j
                                              q_M(2 kbar z) dz ,

          the q_M-sampled companions of gamma_ex and theta_dot_X.
          These are DIAGNOSTIC: the delivered coupling books the
          momentum exchange from the realized change of M across the
          adiabatic operators (propagate, Tmean_v), which keeps the
          coupled budget exactly conservative; this method gives the
          paper expressions for comparison.  For a depth-uniform
          current the two agree identically.

          The antiderivative of q_M is q_A (r_M = m_1), so the exact
          cell integrals use the q_A point table and nothing new.

          Returns (gamma_M, thetadot_M), each (nd, Ny, Nx), padded
          with halos applied.
          """
          h  = self.halo
          il = slice(h, -h)
          pad3 = ((0,0), (h,h), (h,h))
          U = self.apply_bc(np__.pad(np__.asarray(UL, float), pad3),
                            wrap_d=False)
          V = self.apply_bc(np__.pad(np__.asarray(VL, float), pad3),
                            wrap_d=False)
          grads = []
          for F in (U, V):
              gx = ((np__.roll(F, -1, -1) - np__.roll(F, 1, -1))
                    /(2*self.dx))[:, il, il]
              gy = ((np__.roll(F, -1, -2) - np__.roll(F, 1, -2))
                    /(2*self.dy))[:, il, il]
              grads.append(gx); grads.append(gy)
          zt, zb = self._zeta_faces(dz, nz)
          kb = self.sigbar()[self.dsl, il, il]**2/(self.grav*self.s_half**2)
          mc = self._z_cut_level(dz, nz)       # top levels that matter
          G = np__.zeros((2, 2, self.nd, self.ny_loc, self.nx_loc))
          for m_ in range(nz - mc, nz):    # index 0 = bottom
              rt = np__.interp(2*kb*zt[m_], self._zeta_tab, self._qA_tab,
                               left=0.0)
              if getattr(self, 'closed_bottom', False) and m_ == 0:
                 rb = 0.0*rt
              else:
                 rb = np__.interp(2*kb*zb[m_], self._zeta_tab, self._qA_tab,
                                  left=0.0)
              w = rt - rb
              G[0,0] += grads[0][m_][None,:,:]*w
              G[1,0] += grads[1][m_][None,:,:]*w
              G[0,1] += grads[2][m_][None,:,:]*w
              G[1,1] += grads[3][m_][None,:,:]*w
          sn = self.s_x[self.dsl]
          cs = self.s_y[self.dsl]
          gM = -(sn*sn*G[0,0] + sn*cs*G[1,0] + cs*sn*G[0,1] + cs*cs*G[1,1])
          du = cs*G[0,0] - sn*G[1,0]
          dv = cs*G[0,1] - sn*G[1,1]
          tM = -(sn*du + cs*dv)
          padd = ((h,h), (h,h), (h,h))     # pad the ring too, then fill it
          gM = self.apply_bc(np__.pad(gM, padd))
          tM = self.apply_bc(np__.pad(tM, padd))
          return gM, tM
