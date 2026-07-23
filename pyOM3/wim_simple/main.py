"""
 main class of WIM model defining/allocating model variables 

 Based on class boundary dealing with domain decomposition
 and fortran extension for discrete interaction approximation.
""" 
import pyOM3               as OM
from   pyOM3.wim_simple.boundary  import boundary

class main(boundary):

    
      def setup(self):  
          """
          add some constants and methods to initialisation.
          """   
          self.allocate_workspace() 
          self.set_grid()

          if not hasattr(self,'snonlin_backend'):  self.snonlin_backend = 'fortran'
 
          if self.snonlin_backend == 'fortran':
            if OM.prec != OM.np.float64:
              raise NotImplementedError(
                  "Fortran code for Snonlin assumes double precision "
                  "(OM.prec must be np.float64)")
            from pyOM3.wim_simple.Snonlin import Snonlin as Snonlin_backend
          elif self.snonlin_backend == 'jax':
            from pyOM3.wim_simple.snonlin_jax import Snonlin as Snonlin_backend
          elif self.snonlin_backend == 'numpy':
            from pyOM3.wim_simple.snonlin_np  import Snonlin as Snonlin_backend    
          else:
            raise ValueError("snonlin_backend must be 'numpy','jax', or 'fortran', "
                             "got %r" % (self.snonlin_backend,))
 
          self.Snonlin_backend = Snonlin_backend
          self.Snonlin_backend.setup(self.nx_loc, self.ny_loc, self.nd, self.ft[1:-1])
              
          
          if not hasattr(self,'grav'):   self.grav  = 9.81
          if not hasattr(self,'rho_o'):  self.rho_o = 1024.
          if not hasattr(self,'limit_growth'):   self.limit_growth = False
          if not hasattr(self,'mask_energy' ):   self.mask_energy  = False    

          # horizontal land/sea mask. set_topography defines kbot (default: all
          # ocean; overload for a real coastline), make_topo_masks builds the
          # T/U/V masks and closes the outer ring for non-periodic boundaries.
          self.set_topography()
          self.make_topo_masks()
          
          #cfl = self.dt * self.grav/(4*OM.np.pi*0.035) / min(self.dx, self.dy)     # cg_max = g/(2*omega_min)
          #assert cfl < 0.28, 'wave CFL %.2f exceeds certified limit (see wim.tex)' % cfl
          return


      def set_topography(self):
          """
          Define the horizontal land/sea distribution: kbot = 1 in ocean cells,
          kbot = 0 on land. Default is all ocean. Overload this method to read a
          real coastline/topography (e.g. from the coupled ocean's kbot) for a
          non-box configuration.

          Note: the closing of a non-periodic outer boundary into a land ring is
          done in make_topo_masks (driven by periodic_in_x / periodic_in_y), so a
          plain box does not need to set any land here.
          """
          self.kbot = OM.modify_array(self.kbot, (slice(None), slice(None)), 1.0)
          return


      def make_topo_masks(self):
          """
          Build the horizontal masks from kbot. Mirrors pe_simple.make_topo_masks
          but in 2D (waves have no vertical level; d,f are never masked).

          maskTp = 1.0 in ocean cells (kbot != 0), 0.0 on land.
          maskUp/maskVp are the x/y face masks: a face is wet only if the two
          cells it connects are both wet (min of neighbouring T-cells), so fluxes
          through a land face are zeroed.

          For a closed (non-periodic) boundary the outermost ring of cells is set
          to land, so the domain edge acts as a wall. For a periodic direction the
          ring is left ocean and apply_bc wraps it.
          """
          # halo-fill kbot, then close the outer ring where the boundary is closed
          self.kbot = self.apply_bc(self.kbot)

          if not self.periodic_in_y:
             if self.my_blk_y == 1:
                self.kbot = OM.modify_array(self.kbot, (slice(None, self.halo),  slice(None)), 0.0)
             if self.my_blk_y == self.n_pes_y:
                self.kbot = OM.modify_array(self.kbot, (slice(-self.halo, None), slice(None)), 0.0)
          if not self.periodic_in_x:
             if self.my_blk_x == 1:
                self.kbot = OM.modify_array(self.kbot, (slice(None), slice(None, self.halo)),  0.0)
             if self.my_blk_x == self.n_pes_x:
                self.kbot = OM.modify_array(self.kbot, (slice(None), slice(-self.halo, None)), 0.0)

          # T mask: wet where kbot != 0
          self.maskTp = OM.np.where(self.kbot != 0, 1.0, 0.0)

          # face masks: wet only if both adjacent T-cells are wet
          self.maskUp = OM.np.minimum(self.rollx(self.maskTp, -1), self.maskTp)
          self.maskVp = OM.np.minimum(self.rolly(self.maskTp, -1), self.maskTp)

          self.maskTp = self.apply_bc(self.maskTp)
          self.maskUp = self.apply_bc(self.maskUp)
          self.maskVp = self.apply_bc(self.maskVp)

          # precompute the directional-reflection index tables (no-op cost if the
          # domain has no walls; reflection itself is only applied if enabled)
          if hasattr(self, 'reflect_ini'):
             self.reflect_ini()
          return
   

      def set_grid(self):
          """
          define grid variables
          dx,dy,dd are constants, while df is varying
          setting up u-centered axis for frequency 
          """         
          # WAM freq. grid: f_min = 0.035 1/s , delta f/f = 0.1  (geometric).
          f  = OM.np.zeros( (self.Nf-2*self.halo,), OM.prec)
          df = OM.np.zeros( (self.Nf-2*self.halo,), OM.prec)            
          f[0] = self.f0 
          df[0] = self.df0*f[0] 
          for n in range(1,self.Nf-2*self.halo):
             f[n]  = f[n-1] + df[n-1]
             df[n] = self.df0*f[n]

          # setup u-centered grid: 
          # ft = f, dfu = df, 
          # dfu_i = ft_(i+1)-ft_i
          # fu_i = 0.5(ft_i+ft_(i+1)) , 
          # dft_i = fu_i - fu_i-1 
          
          self.ft,self.dfu = f,df
          self.fu = OM.np.zeros_like(self.ft)
          self.fu[:-1] = (self.ft[:-1] + self.ft[1:])/2
          self.fu[-1]  = self.fu[-2] + self.dfu[-1]
          self.dft = OM.np.zeros_like(self.ft)
          self.dft[1:] = self.fu[1:] - self.fu[:-1]
          self.dft[0]  = self.dfu[0]
          
          self.dft = OM.np.pad(self.dft, pad_width=self.halo)  
          self.dft = OM.modify_array(self.dft, 0,  self.dft[ 1]  )
          self.dft = OM.modify_array(self.dft,-1,  self.dft[-2]  ) 

          self.dfu = OM.np.pad(self.dfu, pad_width=self.halo)  
          self.dfu = OM.modify_array(self.dfu, 0,  self.dfu[ 1]  )
          self.dfu = OM.modify_array(self.dfu,-1,  self.dfu[-2]  ) 
          
          self.ft  = OM.np.pad(self.ft, pad_width = self.halo) 
          self.ft = OM.modify_array(self.ft, 0,  self.ft[ 1] - self.dft[ 0] )
          self.ft = OM.modify_array(self.ft,-1,  self.ft[-2] + self.dft[-1] ) 

          self.fu  = OM.np.pad(self.fu, pad_width = self.halo) 
          self.fu = OM.modify_array(self.fu, 0,  self.fu[ 1] - self.dfu[ 0] )
          self.fu = OM.modify_array(self.fu,-1,  self.fu[-2] + self.dfu[-1] )   
                     
          self.omega_t = 2*OM.np.pi*self.ft
          self.omega_u = 2*OM.np.pi*self.fu

          # set directional axis
          self.dd = 2*OM.np.pi/self.nd
          self.theta = OM.np.arange(self.nd)*self.dd + self.dd/2
          self.theta = OM.np.pad(self.theta, pad_width = self.halo)  
          self.theta = OM.modify_array(self.theta, 0,  self.theta[-2]  )
          self.theta = OM.modify_array(self.theta,-1,  self.theta[ 1]  )                          
          self.costh = OM.np.cos(self.theta)
          self.sinth = OM.np.sin(self.theta)  
          return 


      def allocate_workspace(self):
          """
          define all model variables 
          """

          Nx, Ny, Nf, Nd = self.Nx, self.Ny, self.Nf, self.Nd 

          # allocate model variables with halo points
          self.A   = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec) 
          self.dA  = OM.np.zeros((3,Nd,Nf,Ny,Nx), OM.prec)
          self.E   = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec) 
          
          self.xdot = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec)
          self.ydot = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec) 
          self.fdot = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec) 
          self.ddot = OM.np.zeros(  (Nd,Nf,Ny,Nx), OM.prec)  

          # horizontal land/sea mask (x,y only; broadcast over d,f). kbot=0 land,
          # kbot=1 ocean. maskTp is 1.0 in ocean cells, 0.0 on land; maskUp/maskVp
          # are the x/y face masks (a face is wet only if both adjacent cells are).
          self.kbot   = OM.np.zeros( (Ny,Nx), OM.prec)
          self.maskTp = OM.np.zeros( (Ny,Nx), OM.prec)
          self.maskUp = OM.np.zeros( (Ny,Nx), OM.prec)
          self.maskVp = OM.np.zeros( (Ny,Nx), OM.prec)

          return

   