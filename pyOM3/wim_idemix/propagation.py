"""
 Propagation for the wim_idemix model.

 Phase-space advection of the pair (Acal,Ecal), dimension split into an
 x sweep, a y sweep and a theta sweep (refraction).  Fluxes are in flux
 form with a limited second order (superbee or van Leer) reconstruction,
 as in wim_simple.

 Two exact properties of the reduction are used:
   * the ENERGY flux is (g/2) k_hat Acal with NO shape factor -- exact
     under the ansatz, because int c_g sigma A~ dk = (g/2) int A~ dk;
   * the ACTION speed is beta_A g/(2 sigbar) k_hat, with the shape
     constant beta_A = s_{1/2} s_{-1/2} (full range, paper Eq. 12).
 Both fluxes therefore carry Acal and share sign(k_hat), so a SINGLE
 limited reconstruction of Acal serves both.

 The face speed uses the ACTION-WEIGHTED face frequency
 (E_lo+E_hi)/(A_lo+A_hi): the wave content sets the speed, and empty
 neighbours contribute nothing (an arithmetic average of cell speeds
 inflates a front travelling into empty water).

 Lateral boundaries are entirely in the halo values (see boundary.py),
 so the stencil is uniform and the scheme telescopes everywhere -- which
 is what makes the total conserved to round-off in the periodic case.
"""
import numpy as np__
from   pyOM3.wim_idemix.main import main


class propagation(main):

      def limited_slope(self, q, ax, mf = None):
          """
          Limited slope of q along axis ax, computed with rolls so that
          every cell (including halo cells) gets the same formula.

          NOTE the slope MUST be single-valued on a wrapped face: taking
          it from a padded array with zeroed end slopes gives the same
          face two different reconstructions, the flux stops
          telescoping, and the total drifts (this was a real bug, found
          by the Bickley-jet conservation check).
          """
          a = q - np__.roll(q,  1, ax)          # gradient at face j-1/2
          b = np__.roll(q, -1, ax) - q          # gradient at face j+1/2
          if mf is not None:
              # MASK THE FACE GRADIENTS, EACH WITH ITS OWN FACE, exactly as
              # wim_simple masks rj before the limiter. Masking the
              # resulting CELL slope with the j+1/2 mask instead (what this
              # did before) protects the cell below a coast and not the one
              # above it, so a coast facing north and the same coast facing
              # south are treated differently.
              a = a*np__.roll(mf, 1, ax)
              b = b*mf
          if self.limiter == 'vanleer':
             ab = a*b
             return np__.where(ab > 0.0, 2.0*ab/(a + b + 1e-300), 0.0)
          mm = lambda u, v: np__.where(u*v > 0.0,
                                       np__.sign(u)*np__.minimum(np__.abs(u),
                                                                 np__.abs(v)),
                                       0.0)
          s1, s2 = mm(2.0*a, b), mm(a, 2.0*b)
          return np__.where(np__.abs(s1) > np__.abs(s2), s1, s2)


      # ------------------------------------------------------------------
      # Transport, in the four stages of wim_simple: velocities -> fluxes
      # -> boundary conditions on the fluxes -> one divergence.  x, y and
      # theta are treated alike; the scheme is UNSPLIT, i.e. every flux is
      # built from the SAME state and a single summed divergence updates
      # the pair once per step (wim_simple.add_propagation).
      # ------------------------------------------------------------------
      def transport_velocities(self):
          """All advection velocities of the pair system, on the faces.

          Returns (vA, vE) with vX = (x, y, theta) face velocities for the
          action and the energy.  The intrinsic part is the pair speed:
          the energy flux is (g/2) Acal (= c_g Ecal with c_g = g/2 sigbar)
          and the action flux beta_A (g/2 sigbar) Acal, so BOTH intrinsic
          fluxes are carried by the reconstructed ACTION -- that is the
          pair structure, and the coefficients differ, not the tracer.
          The current part adds U_X + s (s . V_X) per component, each with
          its own face average.  The theta rates come from the sampled
          refraction; their face value is built from the ring mean plus
          the harmonic part divided by cos(dth), which is exact for any
          rate that is a quadratic form in (sin th, cos th).
          """
          g2  = 0.5*self.grav
          A, E = self.Ad, self.Ed
          out = {-1: None, -2: None}
          # a degenerate axis (nx = 1 or ny = 1) carries no transport: the
          # old split form skipped that sweep entirely and so must this one
          axes = [(ax, dl) for ax, dl in ((-1, self.dx), (-2, self.dy))
                  if (self.nx if ax == -1 else self.ny) > 1]
          for ax, dl in axes:
              cdir = self.s_x if ax == -1 else self.s_y
              Ar = np__.roll(A, -1, ax)
              Er = np__.roll(E, -1, ax)
              sbf = np__.clip((E + Er)/np__.maximum(A + Ar, 1e-30),
                              self.sig_min, self.sig_max)
              cA = self.beta_A*g2/sbf*cdir            # action, intrinsic
              cE = g2*cdir + 0.0*sbf                  # energy, intrinsic
              uA = uE = None
              if self.U_A is not None:
                  comp = 0 if ax == -1 else 1
                  uA = self.U_A[comp]
                  uE = self.U_E[comp]
                  if self.V_A is not None:
                      sn, cs = self.s_x, self.s_y
                      sc = sn if comp == 0 else cs
                      uA = uA + sc*(sn*self.V_A[0] + cs*self.V_A[1])
                      uE = uE + sc*(sn*self.V_E[0] + cs*self.V_E[1])
                  uA = 0.5*(uA + np__.roll(uA, -1, ax))     # face j+1/2
                  uE = 0.5*(uE + np__.roll(uE, -1, ax))
              out[ax] = (cA, cE, uA, uE)
          # theta: same treatment, on the periodic ring
          TA = self.theta_dot_A if self.theta_dot_A is not None \
               else self.theta_dot
          TE = self.theta_dot_E if self.theta_dot_E is not None \
               else self.theta_dot
          if TA is None and TE is None:
              out[0] = (None, None)
          else:
              def face_rate(T0):
                  T  = np__.broadcast_to(T0, self.Ad.shape)
                  Tb = T[self.dsl].mean(0, keepdims=True)
                  return Tb + (0.5*(T + np__.roll(T, -1, 0)) - Tb)/np__.cos(self.dth)
              out[0] = (face_rate(TA) if TA is not None else None,
                        face_rate(TE) if TE is not None else None)
          return out

      def tracer_fluxes(self, vel):
          """Face fluxes of (Acal, Ecal) in x, y and theta, ALL built from
          the same state with the limited reconstruction upwinded on the
          sign of each face velocity (the intrinsic and the current part
          can run against each other, so they get separate upwinding).

          Returns {axis: (FA, FE)} with axis -1 = x, -2 = y, 0 = theta.
          """
          A, E = self.Ad, self.Ed
          F = {-1: (None, None), -2: (None, None)}
          for ax in (-1, -2):
              if vel[ax] is None:            # degenerate axis, no transport
                  continue
              cA, cE, uA, uE = vel[ax]
              mf = (self._mU3 if ax == -1 else self._mV3) if self._land else None
              open_c = bool(getattr(self, 'open_coasts', False)) and self._land
              mfo = mf if open_c else None
              sA = self.limited_slope(A, ax, mf)
              Ar, sAr = np__.roll(A, -1, ax), np__.roll(sA, -1, ax)
              cdir = self.s_x if ax == -1 else self.s_y
              AF  = np__.where(cdir > 0, A + 0.5*sA, Ar - 0.5*sAr)
              if open_c:
                  # PURE UPWIND AT THE BOUNDARY FACE.  The interior keeps
                  # the limited second-order reconstruction, but a face
                  # that opens onto land must not use a slope: the slope
                  # is built from both sides, so the reconstruction would
                  # take information from outside the domain.  First-order
                  # upwind uses the wet cell alone, which is the only
                  # defensible outflow value.  For an INcoming face the
                  # upwind cell is land, where A = E = 0, so nothing
                  # enters -- the radiating condition falls out.
                  AF = np__.where(mfo > 0, AF, np__.where(cdir > 0, A, Ar))
              FA = cA*AF
              FE = cE*AF
              if uA is not None:
                  sE = self.limited_slope(E, ax, mf)
                  Er, sEr = np__.roll(E, -1, ax), np__.roll(sE, -1, ax)
                  AF2 = np__.where(uA > 0, A + 0.5*sA, Ar - 0.5*sAr)
                  EF  = np__.where(uE > 0, E + 0.5*sE, Er - 0.5*sEr)
                  if open_c:
                      AF2 = np__.where(mfo > 0, AF2,
                                       np__.where(uA > 0, A, Ar))
                      EF  = np__.where(mfo > 0, EF,
                                       np__.where(uE > 0, E, Er))
                  FA = FA + uA*AF2
                  FE = FE + uE*EF
              if self._land and not open_c:
                  # MASK THE FLUX ITSELF, as wim_simple does ('no flux
                  # through a dry face'). The reflection below then adds
                  # the neighbouring face's flux onto a wall face that is
                  # ZERO; masking only the slope leaves a flux there and
                  # the pyOM2 rule double-counts it, which is what broke
                  # the conservation of the coast and budget tests.
                  mf = self._mU3 if ax == -1 else self._mV3
                  FA = FA*mf
                  FE = FE*mf
              F[ax] = (FA, FE)
          TA, TE = vel[0]
          if TA is None and TE is None:
              F[0] = (None, None)
          else:
              def th_flux(q, Tf):
                  s  = self.limited_slope(q, 0)
                  qF = np__.where(Tf > 0, q + 0.5*s,
                                  np__.roll(q, -1, 0) - 0.5*np__.roll(s, -1, 0))
                  return Tf*qF
              F[0] = (th_flux(A, TA) if TA is not None else None,
                      th_flux(E, TE) if TE is not None else None)
          return F

      def apply_flux_bc(self, F):
          """Boundary conditions ON THE FLUXES, one mechanism for every
          wall: specular reflection at each land edge, and the closed
          domain walls are land (see _reflect_tables), so they go through
          the same four branches. The theta ring is periodic and needs
          none. Action and energy flux are reflected alike.
          """
          if getattr(self, 'open_coasts', False):
              return F                  # the flux already left the domain
          if not (getattr(self, 'enable_reflection', False) and self._land):
              return F
          if F[-1][0] is None or F[-2][0] is None:
              return F                      # a degenerate axis: no walls
          FA = list(self.reflect_flux(F[-1][0], F[-2][0]))     # action
          FE = list(self.reflect_flux(F[-1][1], F[-2][1]))     # energy
          F[-1] = (FA[0], FE[0])
          F[-2] = (FA[1], FE[1])
          return F

      def add_divergence(self, F):
          """One summed divergence for all three directions, one masked
          tendency, one update of the pair (wim_simple.add_propagation).
          Land cells are never updated.
          """
          h = self.halo
          inner = (self.dsl, slice(h, -h), slice(h, -h))
          dA = np__.zeros_like(self.Ad)
          dE = np__.zeros_like(self.Ed)
          for ax, dl in ((-1, self.dx), (-2, self.dy), (0, self.dth)):
              FA, FE = F[ax]
              if FA is not None:
                  dA += (FA - np__.roll(FA, 1, ax))/dl
              if FE is not None:
                  dE += (FE - np__.roll(FE, 1, ax))/dl
          if self._land:
              dA = dA*self._mT3
              dE = dE*self._mT3
          if self._land and getattr(self, 'open_coasts', False):
              # ACCOUNT THE FLUX THAT LEFT THE DOMAIN, over the coastal
              # faces tabulated in _land_tables (weight and indices fixed
              # by the mask).  d = the direction slice: land faces outside
              # the halo would double count across PEs.
              d = self.dsl
              for ax in (-2, -1):
                  FA, FE = F[ax]
                  if FA is None:
                      continue
                  jj, ii, wg = self._loss_idx[ax]
                  self.loss_A += self.dt*float(
                      (FA[d][:, jj, ii]*wg[None, :]).sum())
                  self.loss_E += self.dt*float(
                      (FE[d][:, jj, ii]*wg[None, :]).sum())
          self.Ad[inner] -= self.dt*dA[inner]
          self.Ed[inner] -= self.dt*dE[inner]
          return

      def _reflect_tables(self):
          """Reflection index tables, ported from wim_simple.reflect_ini,
          which is itself the port of pyOM2 idemix_spectral.f90.

          There is ONE mechanism and four symmetric branches. A closed
          DOMAIN WALL is simply land: its halo is set to land here, so a
          wall is a coastline like any other and needs no condition of
          its own. For every direction bin, the bin is assigned to
          exactly ONE of {south, north} and one of {west, east} by the
          half-plane it travels into, and reflects into the mirror bin:
            y walls (south/north)  theta -> pi - theta   (flips cos)
            x walls (west/east)    theta -> 2 pi - theta (flips sin)
          Rebuilt when maskTp changes identity.
          """
          m = np__.array(self.maskTp, dtype=np__.float64)
          h = self.halo
          # A closed domain wall is land -- but ONLY where reflection is
          # asked for. Without enable_reflection a non-periodic edge keeps
          # its lateral condition (bc_west/bc_east/bc_south/bc_north:
          # 'coast' absorbing, 'open' zero-gradient outflow); making those
          # land would close the face and turn an outflow boundary into a
          # reflecting one (it inflated the fetch run to 23 E_PM).
          if getattr(self, 'enable_reflection', False):
            if not self.periodic_in_x:
                if self.xs_pe == 1:        m[:, :h]  = 0.0
                if self.xe_pe == self.nx:  m[:, -h:] = 0.0
            if not self.periodic_in_y:
                if self.ys_pe == 1:        m[:h, :]  = 0.0
                if self.ye_pe == self.ny:  m[-h:, :] = 0.0
          self._mask_wall = m                    # land incl. the walls
          nd, two_pi = self.nd, 2*np__.pi
          th = self.theta[self.dsl] % two_pi     # the real bins
          k  = np__.arange(self.Nd) - h
          # mirror permutations on the PADDED ring, mapping into the real bins
          self._mir_y = h + (nd//2 - k) % nd     # theta -> pi - theta
          self._mir_x = h + (-k) % nd            # theta -> 2 pi - theta
          # half-plane assignment, exactly wim_simple.reflect_ini:
          # south if pi/2 < theta <= 3 pi/2, else north;
          # west  if pi   < theta <= 2 pi,   else east
          # Half-plane assignment. pyOM2 puts EVERY bin in exactly one of
          # {south,north} via pi/2 < theta <= 3pi/2, which hands the two
          # wall-PARALLEL bins (cos theta = 0) to a fixed branch. Those
          # bins are their own y-mirror, so that tie-break is not
          # mirror-symmetric and shows up as a 1e-1 defect in
          # mirror_test. They carry no flux through a y face anyway, so
          # they are excluded from both branches instead; same in x.
          cs, sn = np__.cos(th), np__.sin(th)
          self._sel_s = cs < 0.0
          self._sel_n = cs > 0.0
          self._sel_w = sn < 0.0
          self._sel_e = sn > 0.0
          # the mirror bin of each REAL bin, as a plain index (one value
          # per bin, exactly the kk of reflect_ini)
          kr = np__.arange(nd)
          self._kk_y = (nd//2 - kr) % nd          # theta -> pi - theta
          self._kk_x = (-kr) % nd                 # theta -> 2 pi - theta
          my1 = np__.roll(m, -1, 0)              # cell j+1
          mx1 = np__.roll(m, -1, 1)              # cell i+1
          # face masks (wim_simple/pe_simple maskUp/maskVp)
          # OUTFLOW BOOK-KEEPING for open_coasts.  The weight of a face
          # in the domain budget is (m[i] - m[i+1]): +1 where the domain
          # ends going up-index, -1 where it begins, 0 in the interior and
          # inside land.  It depends only on the mask and the geometry,
          # so it is built ONCE here, and only the handful of coastal
          # faces are stored -- the per-step cost is then a gather over
          # those faces instead of a multiply over the whole (nd,Ny,Nx)
          # array.  Index 0 is y (axis -2), index 1 is x (axis -1).
          vol = self.dx*self.dy*self.dth
          self._loss_idx = {}
          for ax, m1, dl in ((-2, my1, self.dy), (-1, mx1, self.dx)):
              wgt = (m - m1)*(vol/dl)
              own = np__.zeros_like(wgt, dtype=bool)      # this PE's own
              own[h:-h, h:-h] = True                      # interior only:
              wgt = np__.where(own, wgt, 0.0)             # a halo face is
              jj, ii = np__.nonzero(wgt)                  # the neighbour's
              self._loss_idx[ax] = (jj, ii, wgt[jj, ii])
          self.maskUp = np__.minimum(mx1, m)
          self.maskVp = np__.minimum(my1, m)
          self._mU3 = self.maskUp[None, :, :]
          self._mV3 = self.maskVp[None, :, :]
          self._mT3 = m[None, :, :]
          self._has_land = bool((m == 0.0).any())
          # wall edges, as in reflect_ini
          self._edge_s = ((m == 0.0) & (my1 == 1.0))[None]   # land j, ocean j+1
          self._edge_n = ((m == 1.0) & (my1 == 0.0))[None]   # ocean j, land j+1
          self._edge_w = ((m == 0.0) & (mx1 == 1.0))[None]   # land i, ocean i+1
          self._edge_e = ((m == 1.0) & (mx1 == 0.0))[None]   # ocean i, land i+1
          self._rfl_mask_id = id(self.maskTp)
          return

      def _land_tables(self):
          """Ensure the land-derived tables (face masks, mirror
          permutations, into-land selections) match the current maskTp.
          They are built ONCE in _reflect_tables and only rebuilt if the
          mask object itself is replaced, so the per-sweep cost is one
          identity comparison."""
          if getattr(self, '_rfl_mask_id', None) != id(self.maskTp):
              self._reflect_tables()
          return

      def reflect_flux(self, FX, FY):
          """Reflection of the face fluxes, a LITERAL transcription of
          wim_simple.reflect_flux, which is the port of pyOM2
          idemix_spectral.f90.  Four branches, one per wall type; the
          closed domain walls are land (see _reflect_tables) and go
          through the same code, so there is no separate wall condition.

          At each wall the flux of the incoming bin is taken from the
          NEIGHBOURING face on the ocean side, ADDED to the wall face in
          bin k and SUBTRACTED from it in the mirror bin kk.

          ORDER MATTERS and is preserved: the loop runs over bins in
          order and mutates the flux arrays IN PLACE, so when it reaches
          bin kk that bin already carries what bin k gave it.  A form
          that reads every bin from the unmodified array is NOT the same
          operation -- that mistake cost the conservation of the coast
          test once already.
          """
          self._land_tables()
          h = self.halo
          es, en = self._edge_s[0], self._edge_n[0]     # (Ny,Nx) bool
          ew, ee = self._edge_w[0], self._edge_e[0]
          for k in range(self.nd):
              kd = h + k
              if self._sel_s[k]:            # south wall: flux from j+1
                  if es.any():
                      f = np__.where(es, np__.roll(FY[kd], -1, 0), 0.0)
                      FY[kd] += f
                      FY[h + self._kk_y[k]] -= f
              if self._sel_n[k]:            # north wall: flux from j-1
                  if en.any():
                      f = np__.where(en, np__.roll(FY[kd], 1, 0), 0.0)
                      FY[kd] += f
                      FY[h + self._kk_y[k]] -= f
              if self._sel_w[k]:            # west wall: flux from i+1
                  if ew.any():
                      f = np__.where(ew, np__.roll(FX[kd], -1, 1), 0.0)
                      FX[kd] += f
                      FX[h + self._kk_x[k]] -= f
              if self._sel_e[k]:            # east wall: flux from i-1
                  if ee.any():
                      f = np__.where(ee, np__.roll(FX[kd], 1, 1), 0.0)
                      FX[kd] += f
                      FX[h + self._kk_x[k]] -= f
          return FX, FY

      def propagate(self):
          """One transport step in the four stages of wim_simple:

              velocities -> fluxes -> boundary conditions -> divergence

          UNSPLIT: every flux is built from the SAME state and one summed
          divergence updates the pair, so x, y and theta are on an equal
          footing.  The mean-flow energy exchange that follows is a local
          source (the exact exponential at fixed action), not a flux, and
          therefore stays outside the transport.
          """
          if not hasattr(self, 'maskTp') and (not self.periodic_in_x
                                              or not self.periodic_in_y):
              self.maskTp = np__.ones((self.Ny, self.Nx))   # walls only
          self._land = hasattr(self, 'maskTp')
          if self._land:
              self._land_tables()
              self._land = self._has_land
          # halos once, then the closed walls imposed by mirroring the
          # state into the wall halo (the theta ring is periodic)
          self.Ad = self.apply_bc(self.Ad)
          self.Ed = self.apply_bc(self.Ed)
          if self._land:
              # land holds nothing, walls included: apply_bc may have
              # written a zero-gradient copy into the wall halo
              self.Ad = self.Ad*self._mT3
              self.Ed = self.Ed*self._mT3

          mean_flow = (self.theta_dot_A is not None
                       or self.gamma_ex is not None
                       or self.theta_dot is not None)
          if mean_flow:
              m0 = self._wave_momentum()

          vel = self.transport_velocities()
          F   = self.tracer_fluxes(vel)
          F   = self.apply_flux_bc(F)
          self.add_divergence(F)

          if self.gamma_ex is not None:
              self.add_exchange()
          if mean_flow:
              # momentum handed to the mean flow this step (paper
              # eq:tauocU), booked from the REALIZED change of
              # M = int kbar Acal s dtheta across the rotation and the
              # exchange, so the coupled system conserves momentum
              # exactly. Positive INTO the waves, kinematic m2/s2.
              self.Tmean_v = self.grav*(self._wave_momentum() - m0)/self.dt
          self.apply_floors()
          return

      def _wave_momentum(self):
          """Column wave momentum components int kbar Acal s dtheta
          (variance units, x first), shape (2, Ny, Nx)."""
          sl = self.dsl
          kb = self.sigbar()[sl]**2/(self.grav*self.s_half**2)
          mx = (kb*self.Ad[sl]*self.s_x[sl]).sum(0)*self.dth
          my = (kb*self.Ad[sl]*self.s_y[sl]).sum(0)*self.dth
          return np__.stack((mx, my))


      def add_exchange(self):
          """
          Energy exchange with the mean flow (paper eq:gamma-ex): the
          energy equation gains gamma_ex Ecal at fixed action, the
          frequency refraction of the pair.  Applied as the exact
          exponential of the constant-rate problem (rates are of the
          order of the shear, so gamma_ex dt << 1 and the update is
          unconditionally stable).  exch_g records the column energy
          received from the mean flow this step, positive into the
          waves, for the coupled energy budget.
          """
          fac = np__.exp(self.dt*self.gamma_ex)
          dE  = (fac - 1.0)*self.Ed
          self.exch_g = dE[self.dsl].sum(0)*self.dth/self.dt
          self.Ed = self.Ed + dE
          return


      def max_dt(self, cfl = 0.125):
          """
          Largest stable time step for the CURRENT state, in seconds.

          The pair is a nonlinear hyperbolic SYSTEM: energy travels at
          c = g/(2 sigbar), action at beta_A c, and the two are coupled
          through sigbar = Ecal/Acal.  The flux Jacobian of
          (Acal,Ecal) has eigenvalues
              lambda_pm = c (beta_A +- sqrt(beta_A^2 - beta_A)) ,
          so the fastest characteristic is lambda_+ = 1.370 c for the
          full-range beta_A = 1.0787 -- 37% faster than the group
          velocity, which is why a CFL based on c_g alone is optimistic.
          (beta_A = s_1/2 s_-1/2 >= 1 by Cauchy-Schwarz, so the root is
          real and the system is hyperbolic for any admissible shape.)

          Returned: cfl*min(dx,dy)/lambda_+ evaluated with the SMALLEST
          sigbar in the domain (the longest, fastest waves), and the
          refraction limit cfl*dtheta/max|theta_dot| when refraction is
          active.

          THE DEFAULT cfl IS 0.125, NOT the 0.25 of the dimension-split
          scheme.  Measured on a travelling packet with the floors off
          (64x64, dx = 10 km, 800 steps, undershoot relative to the
          packet peak): nu = lambda_+ dt/dx of 0.125 and 0.143 stay at a
          few percent, 0.160 reaches 0.16, and 0.169 runs away to 8.5.
          The unsplit form therefore loses stability at nu ~ 0.165 where
          the split form held to 0.34 -- one summed divergence updates
          the pair from a single state, so the axes no longer stabilise
          each other in sequence.  Energy stays conserved to round-off
          well past that point (1.6e-16 at nu = 0.178, where the
          solution is already unusable), so conservation is NOT a
          stability diagnostic here; the undershoot is.
          """
          # smallest sigbar among cells that actually carry waves --
          # empty cells sit at the sigbar floor and would otherwise set
          # a needlessly strict limit
          E  = self.unpad(self.Ed[self.dsl].sum(0))
          sg = self.unpad(self.sigbar_g())
          wet = E > 1e-6*max(float(E.max()), 1e-30)
          sig_min_state = float(sg[wet].min()) if wet.any() else self.sig_max
          # lambda_+/c from beta_A itself: hard-coding 1.354 (its value
          # at beta_A = 1.0735) AND multiplying by beta_A counted the
          # shape constant twice and returned a step ~6% short.
          lam_fac = self.beta_A + np__.sqrt(max(self.beta_A**2
                                                - self.beta_A, 0.0))
          lam = lam_fac*0.5*self.grav \
              / max(sig_min_state, self.sig_min)
          # current advection adds its own face speed on top of the
          # intrinsic characteristic (paper eq:Aeq, eq:Eeq), including
          # the dispersion velocity when set
          if self.U_A is not None:
             umax = max(float(np__.abs(self.U_A).max()),
                        float(np__.abs(self.U_E).max()))
             if self.V_A is not None:
                umax = umax + max(float(np__.abs(self.V_A).max()),
                                  float(np__.abs(self.V_E).max()))
             lam = lam + umax
          dt = cfl*min(self.dx if self.nx > 1 else 1e30,
                       self.dy if self.ny > 1 else 1e30)/lam
          tmax = 0.0
          for T in (self.theta_dot, self.theta_dot_A, self.theta_dot_E):
              if T is not None:
                 tmax = max(tmax, float(np__.abs(T).max()))
          if tmax > 0.0:
             dt = min(dt, cfl*self.dth/tmax)
          return dt


      def apply_floors(self):
          """
          Positivity and physical-band floors.  Acal >= Ad_floor keeps
          the ratio finite; sigbar is held inside [sig_min, sig_max],
          which bounds the transport speed g/(2 sigbar) and the
          saturation diagnostic satX ~ sigbar^5.
          """
          if not self.floor_action:
             return
          sl = self.dsl                    # real bins; the ring halo is
          A, E = self.Ad[sl], self.Ed[sl]  # refilled by apply_bc
          self.Ad[sl] = np__.maximum(A, self.Ad_floor)
          self.Ed[sl] = np__.maximum(E, self.Ad[sl]*self.sig_min)
          self.Ad[sl] = np__.maximum(self.Ad[sl], E/self.sig_max)
          return
