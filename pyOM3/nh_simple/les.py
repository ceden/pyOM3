"""
 LES closures ported from ps3D (LES_clean/for_src/les_closures.f90)

 Provides Smagorinsky (3D), Ducros et al. (1996) filtered-structure-
 function, and a Deardorff-type prognostic TKE closure, plus an optional
 Peclet bound on the horizontal viscosities. The closures update u, v, w, b in place with a forward Euler
 step (outside the Adams-Bashforth weights), exactly as in ps3D.

 Grid conventions are identical to ps3D (Arakawa C):
   str11,str22,str33, visc_00, kappah/kappav : T-cell centers
   str12, visc_12                            : cell corners (i+1/2, j+1/2)
   str13, visc_13                            : above u      (i+1/2, k+1/2)
   str23, visc_23                            : above v      (i, j+1/2, k+1/2)
   tke, diff, les_len, Nsqr, shear/buoy_prod : w-grid       (k+1/2)

 Translation rules from Fortran indices to roll operators (roll shift=-1
 accesses index +1):
   a(i+1)-a(i) -> rollx(a,-1) - a         a(i)-a(i-1) -> a - rollx(a,1)

 Halo handling: instead of computing over extended index ranges as in
 Fortran, strain and viscosity fields get an apply_bc() after they are
 assembled, so that all subsequent +-1 point stencils see valid halo
 values (sufficient for halo_points = 1).

 Note: the closure is formulated dimensionally; use it in setups with
 Ro = 1 and dsqr = 1 (the usual LES configuration). The stability caps
 use self.dt directly.

 For coupled wave-ocean setups with the Lagrangian-mean formulation
 (prognostic u_L in self.u), set les_mix_eulerian = True: strains and
 mixing fluxes are then computed from the Eulerian velocity
 u = u_L - u_S (self.u_S/v_S, and w_S if present), and the tendency is
 added to the prognostic u_L, i.e. friction acts on u, not on u_L.
 les_freeslip_strain (default True) zeroes the vertical shear seen by
 the closure at the rigid lids, consistent with the zero-flux BC.

 Based on boundary class, mix into the model class or hook via
 set_forcing, see suggestions at the bottom of this file.
"""
import pyOM3                    as OM
try:
    from pyOM3.nh_simple.boundary import boundary as _base
except ImportError:
    # offline analysis environments (notebooks) without mpi4py or the
    # compiled p3dfft extension: fall back to an empty base class; the
    # subclass must then provide rollx/rolly/rollz, apply_bc and
    # apply_vertical_flux_bc itself (see the offline stub in the TKE
    # diagnostics cell of plot_ocean.ipynb)
    class _base: pass


class les(_base):

      # ------------------------------------------------------------------
      # setup
      # ------------------------------------------------------------------
      def init_les(self):
          """
          set default parameters (only if not already given in the
          parameter dictionary) and allocate persistent LES variables
          """
          # the closures are formulated dimensionally and update the
          # fields with plain dt increments, non-unit scaling parameters
          # would be inconsistent
          if getattr(self,'Ro',1.0) != 1.0 or getattr(self,'dsqr',1.0) != 1.0:
             raise ValueError('les closure requires Ro = 1 and dsqr = 1, '
                              'but Ro = %r, dsqr = %r'
                              % (getattr(self,'Ro',1.0),
                                 getattr(self,'dsqr',1.0)) )

          defaults = dict(
             enable_smagorinsky        = False,
             enable_ducros_96          = False,
             enable_tke_closure        = False,
             enable_three_level_filter = False,  # for ducros_96
             enable_peclet_bound       = False,
             les_mix_eulerian          = False,  # mix u = u_L - u_S,
                                                 # needs self.u_S/v_S
             les_freeslip_strain       = True,   # zero str13/str23 at
                                                 # the rigid lids
             les_mix_b                 = True,   # explicit SGS buoyancy
                                                 # diffusion; set False when
                                                 # a flux-limited advection
                                                 # scheme (superbee) provides
                                                 # the scalar SGS mixing
             smag_coeff    = 0.148,   # Smagorinsky coefficient
             prandtl       = 0.47,    # turbulent Prandtl number
             min_smag      = 0.0,     # minimum Smagorinsky viscosity
             Peclet_limit  = 50.0,
             c_m           = 0.1,     # TKE closure constants
             c_eke         = 2.0,
             tke_min       = 1e-8,
             les_epsln     = 1e-20,
          )
          for key in defaults:
              if not hasattr(self,key): setattr(self,key,defaults[key])

          buf = OM.np.zeros( (self.Nx,self.Ny,self.Nz) if self.fortran_ordering
                        else (self.Nz,self.Ny,self.Nx), OM.prec)
          # viscosities/diffusivities, kept for diagnostics
          self.visc_00, self.visc_12 = buf.copy(), buf.copy()
          self.visc_13, self.visc_23 = buf.copy(), buf.copy()
          self.kappah,  self.kappav  = buf.copy(), buf.copy()

          if self.enable_tke_closure:
             self.tke        = buf.copy()   # w-grid
             self.diff       = buf.copy()   # w-grid, sqrt(tke)*les_len
             self.les_len    = buf.copy()   # w-grid
             self.tke_diss   = buf.copy()
             self.shear_prod = buf.copy()   # w-grid
             self.buoy_prod  = buf.copy()   # w-grid

          if self.enable_ducros_96:
             self.filteru = buf.copy()
             self.filterv = buf.copy()
             self.filterw = buf.copy()

          self._les_ready = True
          return


      # ------------------------------------------------------------------
      # driver, corresponds to subroutine les_closure of ps3D
      # ------------------------------------------------------------------
      def les_closure(self):
          """
          one forward Euler step of the chosen LES closure(s),
          updates u, v, w, b in place. Call after harmonic friction and
          before the Adams-Bashforth step (as in ps3D integrate.f90).
          """
          if not hasattr(self,'_les_ready'): self.init_les()

          # velocity the closure acts on: for the Lagrangian-mean (u_L)
          # formulation of coupled setups, friction should act on the
          # Eulerian velocity u = u_L - u_S; the tendency is added to
          # the prognostic u_L, which is exactly the friction term of
          # the u_L equation
          if self.les_mix_eulerian:
             if not (hasattr(self,'u_S') and hasattr(self,'v_S')):
                raise ValueError('les_mix_eulerian requires self.u_S '
                                 'and self.v_S')
             u = self.apply_bc(self.u - self.u_S)
             v = self.apply_bc(self.v - self.v_S)
             w = self.apply_bc(self.w - self.w_S) if hasattr(self,'w_S') \
                 else self.apply_bc(self.w)
          else:
             u, v, w = self.u, self.v, self.w

          if self.enable_smagorinsky or self.enable_tke_closure:
             strains = self.les_strain(u,v,w)
          else:
             strains = None

          if self.enable_ducros_96:      self.les_ducros96(u,v,w)
          if self.enable_smagorinsky:    self.les_smagorinsky(strains)
          if self.enable_tke_closure:    self.les_tke_closure(strains)
          if self.enable_peclet_bound:   self.les_peclet_bound()

          # valid halo values for the +-1 stencils of the mixing fluxes
          self.visc_00 = self.apply_bc(self.visc_00)
          self.visc_12 = self.apply_bc(self.visc_12)
          self.visc_13 = self.apply_bc(self.visc_13)
          self.visc_23 = self.apply_bc(self.visc_23)
          self.kappah  = self.apply_bc(self.kappah)
          self.kappav  = self.apply_bc(self.kappav)

          self.les_mix_momentum(u,v,w)
          if self.les_mix_b:
             self.les_mix_buoyancy()
          elif self.enable_tke_closure:
             # no explicit buoyancy flux -> no resolved buoyancy production
             self.buoy_prod = 0.0*self.buoy_prod
          return


      # ------------------------------------------------------------------
      # strain rate tensor
      # ------------------------------------------------------------------
      def les_strain(self,u,v,w):
          """
          strain rate tensor components at their natural grid positions,
          halos are exchanged so that all components can be used in
          +-1 point stencils
          """
          dx, dy, dz = self.dx, self.dy, self.dz

          str11 = (u - self.rollx(u,1))/dx                 # U_x at T point
          str22 = (v - self.rolly(v,1))/dy                 # V_y at T point
          str33 = (w - self.rollz(w,1))/dz                 # W_z at T point
          str12 = 0.5*( (self.rolly(u,-1) - u)/dy          # corner
                      + (self.rollx(v,-1) - v)/dx )

          str13 = 0.5*( (self.rollz(u,-1) - u)/dz          # above u
                      + (self.rollx(w,-1) - w)/dx )
          str23 = 0.5*( (self.rollz(v,-1) - v)/dz          # above v
                      + (self.rolly(w,-1) - w)/dy )

          if self.les_freeslip_strain:
             # zero the vertical shear seen by the closure at the rigid
             # lids, consistent with the free-slip (zero flux) boundary
             # condition of the mixing; otherwise the zero halo above
             # the lid produces a spurious shear (0 - u)/dz there which
             # inflates the viscosity in the top and bottom cells
             str13 = self.apply_vertical_flux_bc(str13)
             str23 = self.apply_vertical_flux_bc(str23)

          str11 = self.apply_bc(str11)
          str22 = self.apply_bc(str22)
          str33 = self.apply_bc(str33)
          str12 = self.apply_bc(str12)
          str13 = self.apply_bc(str13)
          str23 = self.apply_bc(str23)
          return (str11,str22,str33,str12,str13,str23)


      # ------------------------------------------------------------------
      # Smagorinsky closure
      # ------------------------------------------------------------------
      def les_smagorinsky(self,strains):
          """
          visc_smag = (smag_coeff*delta)^2 * sqrt(2 S_ij S_ij)
          """
          str11,str22,str33,str12,str13,str23 = strains
          dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt

          tmpfac = OM.np.sqrt(2.0)*(self.smag_coeff*(dx*dy*dz)**(1./3.))**2
          s66 = str11**2 + str22**2 + str33**2
          s12, s13, s23 = str12**2, str13**2, str23**2

          self.visc_00 = tmpfac*OM.np.sqrt( s66
             + 0.5*(s12 + self.rollx(s12,1) + self.rolly(s12,1)
                        + self.rollx(self.rolly(s12,1),1))
             + 0.5*(s13 + self.rollx(s13,1) + self.rollz(s13,1)
                        + self.rollx(self.rollz(s13,1),1))
             + 0.5*(s23 + self.rolly(s23,1) + self.rollz(s23,1)
                        + self.rolly(self.rollz(s23,1),1)) )

          self.visc_12 = tmpfac*OM.np.sqrt( 2.0*s12
             + 0.25*(s66 + self.rollx(s66,-1) + self.rolly(s66,-1)
                         + self.rollx(self.rolly(s66,-1),-1))
             + 0.5*(s13 + self.rolly(s13,-1) + self.rollz(s13,1)
                        + self.rolly(self.rollz(s13,1),-1))
             + 0.5*(s23 + self.rollx(s23,-1) + self.rollz(s23,1)
                        + self.rollx(self.rollz(s23,1),-1)) )

          self.visc_13 = tmpfac*OM.np.sqrt( 2.0*s13
             + 0.25*(s66 + self.rollx(s66,-1) + self.rollz(s66,-1)
                         + self.rollx(self.rollz(s66,-1),-1))
             + 0.5*(s12 + self.rolly(s12,1) + self.rollz(s12,-1)
                        + self.rolly(self.rollz(s12,-1),1))
             + 0.5*(s23 + self.rollx(s23,-1) + self.rolly(s23,1)
                        + self.rollx(self.rolly(s23,1),-1)) )

          self.visc_23 = tmpfac*OM.np.sqrt( 2.0*s23
             + 0.25*(s66 + self.rolly(s66,-1) + self.rollz(s66,-1)
                         + self.rolly(self.rollz(s66,-1),-1))
             + 0.5*(s12 + self.rollx(s12,1) + self.rollz(s12,-1)
                        + self.rollx(self.rollz(s12,-1),1))
             + 0.5*(s13 + self.rolly(s13,-1) + self.rollx(s13,1)
                        + self.rolly(self.rollx(s13,1),-1)) )

          # floors and diffusivity (cap order as in ps3D)
          self.visc_00 = OM.np.maximum(self.visc_00, self.min_smag)
          self.visc_12 = OM.np.maximum(self.visc_12, self.min_smag)
          self.visc_13 = OM.np.maximum(self.visc_13, self.min_smag)
          self.visc_23 = OM.np.maximum(self.visc_23, self.min_smag)
          # stability caps, direction-aware: each viscosity is capped by
          # the most restrictive spacing among the directions its fluxes
          # act in (visc_00: x-flux of u, y-flux of v, z-flux of w;
          # visc_12: x/y; visc_13: z-flux of u, x-flux of w; visc_23: y/z).
          # The original ps3D caps used geometric means, valid only for
          # near-isotropic grids: for dx >> dz they admit vertically
          # UNSTABLE viscosities (factor dx/dz too large).
          kappa        = self.visc_00/self.prandtl
          self.visc_00 = OM.np.minimum(self.visc_00, min(dx,dy,dz)**2/(4.*dt))
          self.visc_12 = OM.np.minimum(self.visc_12, min(dx,dy)**2/(4.*dt))
          self.visc_13 = OM.np.minimum(self.visc_13, min(dx,dz)**2/(4.*dt))
          self.visc_23 = OM.np.minimum(self.visc_23, min(dy,dz)**2/(4.*dt))
          self.kappah  = OM.np.minimum(kappa, min(dx,dy)**2/(8.*dt))
          self.kappav  = OM.np.minimum(kappa, dz**2/(4.*dt))
          return


      # ------------------------------------------------------------------
      # Ducros et al. (1996) closure
      # ------------------------------------------------------------------
      def les_ducros96(self,u,v,w):
          """
          viscosity from the structure function of high-pass filtered
          velocities (two or three iterated horizontal Laplace filters)
          """
          dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt

          def lap_h(a):   # horizontal 5-point Laplace filter
              return ( self.rollx(a,-1) + self.rollx(a,1)
                     + self.rolly(a,-1) + self.rolly(a,1) - 4.0*a )

          ducu, ducv, ducw = lap_h(u), lap_h(v), lap_h(w)
          ducu = self.apply_bc(ducu)
          ducv = self.apply_bc(ducv)
          ducw = self.apply_bc(ducw)

          self.filteru = self.apply_bc( lap_h(ducu) )
          self.filterv = self.apply_bc( lap_h(ducv) )
          self.filterw = self.apply_bc( lap_h(ducw) )

          if self.enable_three_level_filter:
             self.filteru = self.apply_bc( lap_h(self.filteru) )
             self.filterv = self.apply_bc( lap_h(self.filterv) )
             self.filterw = self.apply_bc( lap_h(self.filterw) )

          def struct(a):  # structure function of filtered velocity
              return OM.np.sqrt(0.25*( (self.rollx(a,-1)-a)**2
                                   + (self.rolly(a,-1)-a)**2
                                   + (self.rollx(a, 1)-a)**2
                                   + (self.rolly(a, 1)-a)**2 ))

          str11 = self.apply_bc( struct(self.filteru) )
          str22 = self.apply_bc( struct(self.filterv) )
          str33 = self.apply_bc( struct(self.filterw) )

          tmpfac = 0.0014*0.5**(-3./2.)
          self.visc_00 = tmpfac*(1./6.)*( dx*(str11 + self.rollx(str11,1))
                                        + dy*(str22 + self.rolly(str22,1))
                                        + dz*(str33 + self.rollz(str33,1)) )
          self.visc_12 = tmpfac*(1./6.)*( dx*(str11 + self.rolly(str11,-1))
                                        + dy*(str22 + self.rollx(str22,-1))
             + 0.25*dz*( str33 + self.rollx(str33,-1) + self.rolly(str33,-1)
                              + self.rollx(self.rolly(str33,-1),-1) )
             + 0.25*dz*self.rollz( str33 + self.rollx(str33,-1)
                                 + self.rolly(str33,-1)
                                 + self.rollx(self.rolly(str33,-1),-1) ,1) )
          self.visc_13 = tmpfac*(1./6.)*( dx*(str11 + self.rollz(str11,-1))
                                        + dz*(str33 + self.rollx(str33,-1))
             + 0.25*dy*( str22 + self.rollx(str22,-1) + self.rolly(str22,1)
                              + self.rollx(self.rolly(str22,1),-1) )
             + 0.25*dy*self.rollz( str22 + self.rollx(str22,-1)
                                 + self.rolly(str22,1)
                                 + self.rollx(self.rolly(str22,1),-1) ,-1) )
          self.visc_23 = tmpfac*(1./6.)*( dy*(str22 + self.rollz(str22,-1))
                                        + dz*(str33 + self.rolly(str33,-1))
             + 0.25*dx*( str11 + self.rollx(str11,1) + self.rolly(str11,-1)
                              + self.rollx(self.rolly(str11,-1),1) )
             + 0.25*dx*self.rollz( str11 + self.rollx(str11,1)
                                 + self.rolly(str11,-1)
                                 + self.rollx(self.rolly(str11,-1),1) ,-1) )

          self.visc_00 = OM.np.maximum(self.visc_00, self.min_smag)
          self.visc_12 = OM.np.maximum(self.visc_12, self.min_smag)
          self.visc_13 = OM.np.maximum(self.visc_13, self.min_smag)
          self.visc_23 = OM.np.maximum(self.visc_23, self.min_smag)
          # stability caps, direction-aware: each viscosity is capped by
          # the most restrictive spacing among the directions its fluxes
          # act in (visc_00: x-flux of u, y-flux of v, z-flux of w;
          # visc_12: x/y; visc_13: z-flux of u, x-flux of w; visc_23: y/z).
          # The original ps3D caps used geometric means, valid only for
          # near-isotropic grids: for dx >> dz they admit vertically
          # UNSTABLE viscosities (factor dx/dz too large).
          kappa        = self.visc_00/self.prandtl
          self.visc_00 = OM.np.minimum(self.visc_00, min(dx,dy,dz)**2/(4.*dt))
          self.visc_12 = OM.np.minimum(self.visc_12, min(dx,dy)**2/(4.*dt))
          self.visc_13 = OM.np.minimum(self.visc_13, min(dx,dz)**2/(4.*dt))
          self.visc_23 = OM.np.minimum(self.visc_23, min(dy,dz)**2/(4.*dt))
          self.kappah  = OM.np.minimum(kappa, min(dx,dy)**2/(8.*dt))
          self.kappav  = OM.np.minimum(kappa, dz**2/(4.*dt))
          return


      # ------------------------------------------------------------------
      # prognostic TKE closure (Deardorff type)
      # ------------------------------------------------------------------
      def les_tke_closure(self,strains):
          """
          prognostic TKE with stratification limited length scale,
          shear and buoyancy production from the previous time step
          """
          str11,str22,str33,str12,str13,str23 = strains
          dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt
          eps = self.les_epsln

          # N^2 on w-grid; the resolved b in nh_simple is the deviation
          # from the background stratification N0**2 * z, which is
          # therefore always included here
          Nsqr = self.N0**2 + (self.rollz(self.b,-1) - self.b)/dz

          delta = (dx*dy*dz)**(1./3.)
          self.les_len = OM.np.where( Nsqr > 0,
               OM.np.minimum(delta, 0.76*OM.np.sqrt( OM.np.maximum(0.,self.tke)
                                                /OM.np.maximum(eps,Nsqr) )),
               delta )

          # dissipation of TKE
          self.tke_diss = ( (0.19 + 0.51*self.les_len/delta)
                           *OM.np.sqrt(OM.np.maximum(0.,self.tke))
                           *OM.np.maximum(0.,self.tke)
                           /OM.np.maximum(eps,self.les_len) )

          # diffusive fluxes of TKE, diff is known from last time step
          fx = self.c_eke*0.5*(self.diff + self.rollx(self.diff,-1)) \
                              *(self.rollx(self.tke,-1) - self.tke)/dx
          fy = self.c_eke*0.5*(self.diff + self.rolly(self.diff,-1)) \
                              *(self.rolly(self.tke,-1) - self.tke)/dy
          fz = self.c_eke*0.5*(self.diff + self.rollz(self.diff,-1)) \
                              *(self.rollz(self.tke,-1) - self.tke)/dz
          fz = self.apply_vertical_flux_bc(fz)

          # time tendency of tke: diffusion, production, dissipation
          self.tke = self.tke + dt*( (fx - self.rollx(fx,1))/dx
                                   + (fy - self.rolly(fy,1))/dy
                                   + (fz - self.rollz(fz,1))/dz
                                   - self.tke_diss
                                   + self.shear_prod + self.buoy_prod )
          self.tke = OM.np.maximum(self.tke, self.tke_min)
          self.tke     = self.apply_bc(self.tke)      # also zeros vertical
          self.les_len = self.apply_bc(self.les_len)  # halos if closed in z

          # new diffusivity (diff and tke are on w grid)
          self.diff = OM.np.sqrt(OM.np.maximum(0.,self.tke))*self.les_len

          self.visc_00 = self.c_m*0.5*(self.diff + self.rollz(self.diff,1))
          self.visc_12 = 0.25*( self.visc_00 + self.rollx(self.visc_00,-1)
                              + self.rolly(self.visc_00,-1)
                              + self.rollx(self.rolly(self.visc_00,-1),-1) )
          self.visc_13 = self.c_m*0.5*(self.diff + self.rollx(self.diff,-1))
          self.visc_23 = self.c_m*0.5*(self.diff + self.rolly(self.diff,-1))
          self.kappah  = ( (1 + 2*self.les_len/delta)*self.c_m
                          *0.5*(self.diff + self.rollz(self.diff,1)) )
          self.kappav  = self.kappah

          # halos needed for the -1 point stencils of the shear production
          self.visc_12 = self.apply_bc(self.visc_12)
          self.visc_13 = self.apply_bc(self.visc_13)
          self.visc_23 = self.apply_bc(self.visc_23)

          # shear production; first part at T points (aloc), with the
          # deviatoric -2/3 tke S_ii contribution
          tke_c  = 0.5*(self.tke + self.rollz(self.tke,1))
          s12sum = ( str12 + self.rollx(str12,1) + self.rolly(str12,1)
                          + self.rollx(self.rolly(str12,1),1) )
          v12avg = 0.25*( self.visc_12 + self.rollx(self.visc_12,1)
                        + self.rolly(self.visc_12,1)
                        + self.rollx(self.rolly(self.visc_12,1),1) )
          aloc = ( 2.0*self.visc_00*str11**2 - (2./3.)*tke_c*str11
                 + 2.0*self.visc_00*str22**2 - (2./3.)*tke_c*str22
                 + 2.0*self.visc_00*str33**2 - (2./3.)*tke_c*str33
                 + s12sum**2*v12avg )

          s13sum = str13 + self.rollx(str13,1)
          s23sum = str23 + self.rolly(str23,1)
          v13avg = 0.5*(self.visc_13 + self.rollx(self.visc_13,1))
          v23avg = 0.5*(self.visc_23 + self.rolly(self.visc_23,1))
          self.shear_prod = ( 0.5*(self.rollz(aloc,-1) + aloc)
                            + s13sum**2*v13avg + s23sum**2*v23avg )

          # buoy_prod is diagnosed from the buoyancy flux in
          # les_mix_buoyancy and used in the next time step
          return


      # ------------------------------------------------------------------
      # Peclet bound
      # ------------------------------------------------------------------
      def les_peclet_bound(self):
          """
          raise horizontal viscosities where the grid Peclet number
          exceeds Peclet_limit; uses the prognostic (advecting) velocity,
          i.e. u_L in the Lagrangian-mean formulation
          """
          eps = self.les_epsln
          spd  = OM.np.sqrt(self.u**2 + self.v**2)

          # at T points for visc_00
          spd_w = OM.np.sqrt(self.rollx(self.u,1)**2 + self.rolly(self.v,1)**2)
          Pe = 0.5*(spd + spd_w)*OM.np.sqrt(self.dx*self.dy) \
                   /OM.np.maximum(self.visc_00,eps)
          self.visc_00 = OM.np.where( Pe > self.Peclet_limit,
                 Pe/self.Peclet_limit*OM.np.maximum(self.visc_00,eps),
                 self.visc_00 )

          # at corner points for visc_12
          spd_c = OM.np.sqrt(self.rolly(self.u,-1)**2 + self.rollx(self.v,-1)**2)
          Pe = 0.5*(spd + spd_c)*OM.np.sqrt(self.dx*self.dy) \
                   /OM.np.maximum(self.visc_12,eps)
          self.visc_12 = OM.np.where( Pe > self.Peclet_limit,
                 Pe/self.Peclet_limit*OM.np.maximum(self.visc_12,eps),
                 self.visc_12 )
          return


      # ------------------------------------------------------------------
      # apply the mixing to momentum and buoyancy
      # ------------------------------------------------------------------
      def les_mix_momentum(self,u,v,w):
          """
          divergence of the anisotropic viscous fluxes of the passed
          velocities (Eulerian for les_mix_eulerian), forward Euler
          update of the prognostic self.u, self.v, self.w
          """
          dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt

          # zonal velocity
          fx = self.rollx(self.visc_00,-1)*(self.rollx(u,-1) - u)/dx
          fy = self.visc_12*(self.rolly(u,-1) - u)/dy
          fz = self.visc_13*(self.rollz(u,-1) - u)/dz
          fz = self.apply_vertical_flux_bc(fz)
          self.u = self.u + dt*( (fx - self.rollx(fx,1))/dx
                               + (fy - self.rolly(fy,1))/dy
                               + (fz - self.rollz(fz,1))/dz )

          # meridional velocity
          fx = self.visc_12*(self.rollx(v,-1) - v)/dx
          fy = self.rolly(self.visc_00,-1)*(self.rolly(v,-1) - v)/dy
          fz = self.visc_23*(self.rollz(v,-1) - v)/dz
          fz = self.apply_vertical_flux_bc(fz)
          self.v = self.v + dt*( (fx - self.rollx(fx,1))/dx
                               + (fy - self.rolly(fy,1))/dy
                               + (fz - self.rollz(fz,1))/dz )

          # vertical velocity
          fx = self.visc_13*(self.rollx(w,-1) - w)/dx
          fy = self.visc_23*(self.rolly(w,-1) - w)/dy
          fz = self.rollz(self.visc_00,-1)*(self.rollz(w,-1) - w)/dz
          fz = self.apply_vertical_flux_bc(fz, flux_on_wgrid = True)
          self.w = self.w + dt*( (fx - self.rollx(fx,1))/dx
                               + (fy - self.rolly(fy,1))/dy
                               + (fz - self.rollz(fz,1))/dz )
          self.w = self.apply_vertical_flux_bc(self.w)
          return


      def les_mix_buoyancy(self):
          """
          divergence of the diffusive buoyancy fluxes, forward Euler
          update of b; also diagnoses the buoyancy production of TKE
          """
          dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt

          fx = 0.5*(self.kappah + self.rollx(self.kappah,-1)) \
                  *(self.rollx(self.b,-1) - self.b)/dx
          fy = 0.5*(self.kappah + self.rolly(self.kappah,-1)) \
                  *(self.rolly(self.b,-1) - self.b)/dy
          fz = 0.5*(self.kappav + self.rollz(self.kappav,-1)) \
                  *(self.rollz(self.b,-1) - self.b)/dz
          fz = self.apply_vertical_flux_bc(fz)

          self.b = self.b + dt*( (fx - self.rollx(fx,1))/dx
                               + (fy - self.rolly(fy,1))/dy
                               + (fz - self.rollz(fz,1))/dz )

          if self.enable_tke_closure: self.buoy_prod = -fz
          return
