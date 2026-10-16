"""
 main class of WIM model defining/allocating model variables 

 Based on class boundary dealing with domain decomposition
 and fortran extension for discrete interaction approximation.
""" 
import pyOM3               as OM
from   pyOM3.wim_simple.boundary  import boundary
from   functools                  import partial
import numpy                      as np__

class main(boundary):

    
      def setup(self):  
          """
          add some constants and methods to initialisation.
          """   
          self.allocate_workspace() 
          self.set_grid()

          # DIA backend: 'fortran' (f2py module wim_simple/Snonlin) on numpy,
          # 'jax' under the jax backend (the Fortran kernel needs numpy arrays)
          if not hasattr(self,'snonlin_backend'):
              self.snonlin_backend = 'jax' if OM.jax else 'fortran'
          if OM.jax and self.snonlin_backend == 'fortran':
              raise ValueError("snonlin_backend='fortran' needs the numpy backend")
          if OM.jax and self.snonlin_backend == 'numpy':
              self.snonlin_backend = 'jax'      # the same DIA, written for both backends
 
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
          elif self.snonlin_backend == 'exact':
            from pyOM3.wim_simple.snl_exact.snonlin_exact import Snonlin as Snonlin_backend 
          else:
            raise ValueError("snonlin_backend must be 'numpy','jax', or 'fortran', "
                             "got %r" % (self.snonlin_backend,))
 
          self.Snonlin_backend = Snonlin_backend

          # propagation kernels: 'fortran' (default, f2py module
          # wim_simple/Propagation, one memory pass per flux) or 'numpy' (the
          # np.roll form). Bit-identical results. Under jax the arrays are not
          # numpy arrays, so the numpy/jax form is used.
          if not hasattr(self,'propagation_backend'):
              self.propagation_backend = 'numpy' if OM.jax else 'fortran'
          if self.propagation_backend == 'fortran':
              if OM.jax:
                  raise ValueError("propagation_backend='fortran' needs the numpy backend")
              if OM.prec != OM.np.float64:
                  raise NotImplementedError("Fortran propagation kernels assume double precision")
              from pyOM3.wim_simple.Propagation import Propagation as _P
              self._prop_f = _P
          elif self.propagation_backend != 'numpy':
              raise ValueError("propagation_backend must be 'numpy' or 'fortran', "
                               "got %r" % (self.propagation_backend,))
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

          # (JAX: the single process holds every edge of the global array;
          # the inline shard halos of a closed direction are already zero
          # from apply_bc, so zeroing the outer ring closes the domain)
          if not self.periodic_in_y:
             if OM.jax or self.my_blk_y == 1:
                self.kbot = OM.modify_array(self.kbot, (slice(None, self.halo),  slice(None)), 0.0)
             if OM.jax or self.my_blk_y == self.n_pes_y:
                self.kbot = OM.modify_array(self.kbot, (slice(-self.halo, None), slice(None)), 0.0)
          if not self.periodic_in_x:
             if OM.jax or self.my_blk_x == 1:
                self.kbot = OM.modify_array(self.kbot, (slice(None), slice(None, self.halo)),  0.0)
             if OM.jax or self.my_blk_x == self.n_pes_x:
                self.kbot = OM.modify_array(self.kbot, (slice(None), slice(-self.halo, None)), 0.0)

          # T mask: wet where kbot != 0
          self.maskTp = OM.np.where(self.kbot != 0, 1.0, 0.0)

          # face masks: wet only if both adjacent T-cells are wet
          self.maskUp = OM.np.minimum(self.rollx(self.maskTp, -1), self.maskTp)
          self.maskVp = OM.np.minimum(self.rolly(self.maskTp, -1), self.maskTp)

          self.maskTp = self.apply_bc(self.maskTp)
          self.maskUp = self.apply_bc(self.maskUp)
          self.maskVp = self.apply_bc(self.maskVp)

          # COASTAL-FACE INDICATORS for the radiating boundary option
          # (open_coasts). A face is coastal when exactly one of the two cells
          # it connects is wet; the absolute value covers both orientations,
          # wet|dry and dry|wet, and the upwind flux itself picks the side from
          # the sign of the velocity, so no orientation logic is needed.
          # Built from the already halo-filled maskTp and deliberately NOT
          # passed through apply_bc: that would zero index halo-1, which is the
          # west/south domain-edge face and IS consumed by the divergence of
          # the first real cell.
          mT = self.maskTp
          self.maskOx = OM.np.abs(mT - self.rollx(mT, -1))
          self.maskOy = OM.np.abs(mT - self.rolly(mT, -1))
          self.open_tables()

          # precompute the directional-reflection index tables (no-op cost if the
          # domain has no walls; reflection itself is only applied if enabled)
          if hasattr(self, 'reflect_ini'):
             self.reflect_ini()
          return


      def open_tables(self):
          """
          Face tables for the outflow book-keeping of open_coasts.

          The weight of a face in the domain budget is the SIGNED difference
          (m - m_next): +1 where the domain ends going up-index, -1 where it
          begins, 0 in the interior and inside land. With the flux convention
          of this model (fx = minus the physical flux) an outflow then counts
          positive at both ends of the domain.

          OWNERSHIP: a coastal face is owned by the PE that owns its WET cell,
          not by the PE whose interior index range contains the face. For a
          face of weight -1 the wet cell is the NEXT one, so the face sits at
          index halo-1 on that PE -- formally a halo index, but the only PE
          that may count it. Counting owned faces instead of interior faces is
          what makes the parallel sum independent of the decomposition.
          """
          h = self.halo
          m = self.maskTp
          inside = self.maskI > 0                        # real cells of this PE / every shard
          self._loss_idx, self._loss_wgt = {}, {}
          for ax, m1, w_vol in ((-2, self.rolly(m, -1), self.dx*self.dd),
                                (-1, self.rollx(m, -1), self.dy*self.dd)):
              w   = m - m1
              roll = self.rolly if ax == -2 else self.rollx
              nxt = roll(inside, -1)                     # cell i+1 is interior
              own = ((w > 0) & inside) | ((w < 0) & nxt)
              wgt = OM.np.where(own, w, 0.0)*w_vol
              self._loss_wgt[ax] = wgt                   # dense (Ny,Nx): the JAX path
              if not OM.jax:                             # sparse tables: the numpy path
                  jj, ii = OM.np.nonzero(wgt)
                  self._loss_idx[ax] = (jj, ii, wgt[jj, ii])
          return


      def coastal_loss(self):
          """
          Action, INTRINSIC energy (omega A) and ABSOLUTE energy (Omega A,
          Omega = omega + k.U) that have left the domain through open coasts
          since the start of the run, summed over all PEs. Returns the triple
          (loss_A, loss_E, loss_Eabs); the third is the one that closes
          against energy_fix_flux, which conserves Omega A, and equals the
          second when there is no current.

          The local accumulators are added up here rather than every step:
          the sum is a collective and the loss is read at the output cadence,
          not at the time step. Reduces over THIS MODEL'S communicator, not
          COMM_WORLD, so it is safe under a coupler that splits the ranks.
          """
          if OM.jax or self.n_pes <= 1:       # JAX: the sums are global already
              return float(self.loss_A), float(self.loss_E), float(self.loss_Eabs)
          from mpi4py import MPI as MPI_
          v   = OM.np.array([self.loss_A, self.loss_E, self.loss_Eabs], dtype=float)
          out = OM.np.empty_like(v)
          self.comm.Allreduce(v, out, op=MPI_.SUM)
          return float(out[0]), float(out[1]), float(out[2])
   

      # RUN CONSTANTS OF THE JITTED BODIES. The jitted step bodies
      # (_mk_advection, _add_propagation, _ab_update, _source_update) read
      # the land masks, the interior mask, the outflow weights and the
      # reflection tables from self. With one process per GPU these arrays
      # are sharded across processes, and JAX does not allow a jitted
      # function to close over such an array ("Closing over jax.Array that
      # spans non-addressable devices"); they must be arguments. The
      # wrappers therefore pass shard_consts() as one argument, and the
      # bodies bind it to self while they are traced (bind_consts), so the
      # same attribute reads become inputs of the compiled function. On
      # numpy this is a no-op (the bound values are the attributes).
      _CONST_NAMES = ('kbot', 'maskTp', 'maskUp', 'maskVp', 'maskOx', 'maskOy', 'maskI',
                      'bc_south', 'bc_north', 'bc_west', 'bc_east', '_loss_wgt')

      def shard_consts(self):
          """the sharded run constants as one pytree (dict)"""
          return {k: getattr(self, k) for k in self._CONST_NAMES if hasattr(self, k)}

      def bind_consts(self, consts):
          """context manager: the attributes of shard_consts replaced by the
          given (traced) values, restored afterwards"""
          import contextlib
          @contextlib.contextmanager
          def _bind():
              saved = {k: getattr(self, k) for k in consts}
              for k, v in consts.items(): setattr(self, k, v)
              try:     yield
              finally:
                  for k, v in saved.items(): setattr(self, k, v)
          return _bind()


      def interior_xy(self, a):
          """
          a restricted to the real (y,x) cells, for reductions over the
          interior. numpy: a view of the interior box. JAX: a times maskI,
          so the halo cells (inline per shard) contribute zero.
          """
          if OM.jax: return a*self.maskI
          s = slice(self.halo, -self.halo)
          return a[..., s, s]


      def interior_add(self, a, b):
          """
          a + b on the real (d,f,y,x) bins and cells, unchanged elsewhere.
          numpy: in place on the interior slice (as before). JAX: masked.
          """
          if OM.jax:
              return a + b*(self.maskI_d[:,None,None,None]*self.maskI_f[None,:,None,None]
                            *self.maskI[None,None,:,:])
          s = slice(self.halo, -self.halo)
          return OM.add_to_array(a, (s,s,s,s), b[s,s,s,s])


      def set_grid(self):
          """
          define grid variables
          dx,dy,dd are constants, while df is varying
          setting up u-centered axis for frequency 
          """         
          # WAM freq. grid: f_min = 0.035 1/s , delta f/f = 0.1  (geometric).
          # Built in host numpy (1-D, never sharded), moved to OM.np at the end.
          np_ = np__
          f  = np_.zeros( (self.Nf-2*self.halo,), OM.prec)
          df = np_.zeros( (self.Nf-2*self.halo,), OM.prec)            
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
          self.fu = np_.zeros_like(self.ft)
          self.fu[:-1] = (self.ft[:-1] + self.ft[1:])/2
          self.fu[-1]  = self.fu[-2] + self.dfu[-1]
          self.dft = np_.zeros_like(self.ft)
          self.dft[1:] = self.fu[1:] - self.fu[:-1]
          self.dft[0]  = self.dfu[0]
          
          self.dft = OM.np.pad(OM.np.asarray(self.dft), pad_width=self.halo)  
          self.dft = OM.modify_array(self.dft, 0,  self.dft[ 1]  )
          self.dft = OM.modify_array(self.dft,-1,  self.dft[-2]  ) 

          self.dfu = OM.np.pad(OM.np.asarray(self.dfu), pad_width=self.halo)  
          self.dfu = OM.modify_array(self.dfu, 0,  self.dfu[ 1]  )
          self.dfu = OM.modify_array(self.dfu,-1,  self.dfu[-2]  ) 
          
          self.ft  = OM.np.pad(OM.np.asarray(self.ft), pad_width = self.halo) 
          self.ft = OM.modify_array(self.ft, 0,  self.ft[ 1] - self.dft[ 0] )
          self.ft = OM.modify_array(self.ft,-1,  self.ft[-2] + self.dft[-1] ) 

          self.fu  = OM.np.pad(OM.np.asarray(self.fu), pad_width = self.halo) 
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
          # face angles theta + dd/2 (the upper face of bin k, where the
          # theta flux fd_k lives); sin/cos are periodic, so the wrapped
          # halo entries of theta give the correct wrapped face angles.
          self.costhu = OM.np.cos(self.theta + self.dd/2)
          self.sinthu = OM.np.sin(self.theta + self.dd/2)
          return 


      def allocate_workspace(self):
          """
          define all model variables 
          """

          Nx, Ny, Nf, Nd = self.Nx, self.Ny, self.Nf, self.Nd 

          # allocate model variables with halo points; under JAX sharded on
          # (y,x) from the start (out_shardings), so no array is ever
          # replicated on every device
          def zeros(shape):
              if OM.jax:
                  return OM.jax.jit(lambda: OM.np.zeros(shape, OM.prec),
                                    out_shardings=self.sharding[len(shape)])()
              return OM.np.zeros(shape, OM.prec)

          self.A   = zeros(  (Nd,Nf,Ny,Nx)) 
          self.dA  = zeros((3,Nd,Nf,Ny,Nx))
          self.E   = zeros(  (Nd,Nf,Ny,Nx)) 
          
          self.xdot = zeros(  (Nd,Nf,Ny,Nx))
          self.ydot = zeros(  (Nd,Nf,Ny,Nx)) 
          self.fdot = zeros(  (Nd,Nf,Ny,Nx)) 
          self.ddot = zeros(  (Nd,Nf,Ny,Nx))  

          # horizontal land/sea mask (x,y only; broadcast over d,f). kbot=0 land,
          # kbot=1 ocean. maskTp is 1.0 in ocean cells, 0.0 on land; maskUp/maskVp
          # are the x/y face masks (a face is wet only if both adjacent cells are).
          self.kbot   = zeros( (Ny,Nx))
          self.maskTp = zeros( (Ny,Nx))
          self.maskUp = zeros( (Ny,Nx))
          self.maskVp = zeros( (Ny,Nx))

          # INTERIOR MASK: 1.0 on the real cells, 0.0 on every halo cell.
          # MPI: the interior box of this PE. JAX: the interior of every
          # shard in the inline layout (pad of ones). Interior sums and the
          # interior source increment go through this mask under JAX, where
          # a slice [halo:-halo] on y,x would keep the inner shard halos.
          # The 1-D spectral indicators serve the same purpose on d and f.
          h = self.halo
          if OM.jax:
              ones = OM.jax.jit(lambda: OM.np.ones((self.ny, self.nx), OM.prec),
                                out_shardings=self.sharding[2])()
              self.maskI = self.pad(ones)
          else:
              self.maskI = OM.np.zeros((Ny,Nx), OM.prec)
              self.maskI[h:-h, h:-h] = 1.0
          self.maskI_d = OM.np.pad(OM.np.ones(Nd-2*h, OM.prec), h)
          self.maskI_f = OM.np.pad(OM.np.ones(Nf-2*h, OM.prec), h)

          # open_coasts book-keeping: action and energy that left the domain,
          # accumulated locally (each PE counts only the faces it owns) and
          # summed over the PEs by coastal_loss(). The per-time-level rates
          # are combined with the Adams-Bashforth weights in model.time_step,
          # so the budget closes at the order the state is actually stepped.
          # Under JAX these are device scalars / (3,) device arrays (no host
          # round trip per step); coastal_loss turns them into floats.
          self.loss_A    = OM.np.zeros(())
          self.loss_E    = OM.np.zeros(())
          self.loss_Eabs = OM.np.zeros(())
          self._lossA_lv = OM.np.zeros((3,))
          self._lossE_lv = OM.np.zeros((3,))
          self._lossO_lv = OM.np.zeros((3,))

          return

   