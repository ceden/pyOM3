"""
 main class of WIM model defining/allocating model variables 

 Based on class boundary dealing with domain decomposition
 and fortran extension for discrete interaction approximation.
""" 
import pyOM3               as OM
from   pyOM3.wim_realistic.boundary  import boundary
from   pyOM3.wim_realistic.grid      import grid
from   pyOM3.wim_simple.main         import main as _main_simple

class main(boundary,grid):

    
      def setup(self):  
          """
          add some constants and methods to initialisation: the variable
          grid and metric factors first (grid.py; needs the domain
          decomposition), then wim_simple's setup (workspace, spectral
          grid, DIA and propagation backends, masks).
          """   
          self.calc_grid()
          # the Janssen input closes the wind momentum with its own tail
          # (source.py _janssen_tail); the ocean stress must use the same tail,
          # i.e. tail_flux='input'. With the default 'wamdi' the stress to
          # the ocean would use the WAMDI dissipation continuation while
          # tau_w uses the Janssen tail (two closures, momentum not closed).
          if getattr(self, 'input_scheme', 'wamdi') == 'janssen' \
                  and getattr(self, 'tail_flux', 'wamdi') != 'input':
              raise ValueError("input_scheme='janssen' needs tail_flux='input' "
                               "(got %r)" % (getattr(self, 'tail_flux', 'wamdi'),))
          return _main_simple.setup(self)


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
          inside = self.maskI > 0
          self._loss_idx, self._loss_wgt = {}, {}
          # face weights on the variable/pseudo-spherical grid: the y-face
          # flux fy carries cosu (see tracer_flux_superbee), so its
          # face-integrated transport is fy*dxt*dd; the x-face length is the
          # meridian arc dyt, weight fx*dyt*dd. Both are (Ny,Nx) via the
          # trailing-axis broadcast views of grid.py.
          wy = OM.np.broadcast_to(self.dxt_x*self.dd, (self.Ny, self.Nx))
          wx = OM.np.broadcast_to(self.dyt_y*self.dd, (self.Ny, self.Nx))
          for ax, m1, w_vol in ((-2, self.rolly(m, -1), wy),
                                (-1, self.rollx(m, -1), wx)):
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

      # grid-metric free, shared with wim_simple (same pattern as model.py)
      set_topography     = _main_simple.set_topography
      make_topo_masks    = _main_simple.make_topo_masks
      coastal_loss       = _main_simple.coastal_loss
      set_grid           = _main_simple.set_grid
      allocate_workspace = _main_simple.allocate_workspace
      interior_xy        = _main_simple.interior_xy
      interior_add       = _main_simple.interior_add
