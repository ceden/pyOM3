"""
 Domain decomposition and halo points for the wim_idemix model.

 Decomposition is in x and y only; the direction dimension (d) is always
 local, so every PE holds the full directional grid for its horizontal
 sub-block.  Model arrays are C-ordered with shape (nd,Ny,Nx), i.e. x is
 the last (fastest) axis and y the second-last -- the same trailing-axis
 convention as wim_simple, one dimension shorter because the reduced
 model integrates over frequency.

 The direction axis carries NO halo: theta is periodic and local, so the
 theta sweep wraps with a local roll (simpler, and exact).

 Lateral boundary conditions are applied by SETTING HALO VALUES, which
 keeps the advection stencil uniform everywhere:
   periodic_in_x/y = True    wrap (MPI exchange between neighbours)
   otherwise, at the global edges:
     west  (low x)  'coast'  absorbing: halo action/energy = 0, so no
                             wave enters from land while outgoing waves
                             leave through the wall face unimpeded
     east  (high x) 'open'   zero gradient: halo = edge value, i.e. free
                             outflow
   the same options exist in y (bc_south / bc_north).

 PE layout (block coordinates, 1-based):
   my_blk_x in [1, n_pes_x]   (varies fastest)
   my_blk_y in [1, n_pes_y]
   my_pe = (my_blk_y-1)*n_pes_x + (my_blk_x-1)
"""
import pyOM3     as OM
from   pyOM3.wim_idemix import setup
from   mpi4py    import MPI
import numpy     as np__


class boundary(setup):

      def __init__(self, parameter = {}, n_pes_x = 1, n_pes_y = 1,
                                         halo_points = 2, comm_in = None):
         """
         Initialize the x,y domain decomposition and add halo points.

         Input nx,ny is the GLOBAL horizontal shape without halo points;
         nd is global and stays local on every PE.  halo_points = 2 is
         required by the limited second order advection (the face
         reconstruction needs a slope in the first halo cell).
         """
         super().__init__(parameter)

         self.fortran_ordering = False   # arrays are C-ordered (nd,Ny,Nx)

         self.comm  = MPI.COMM_WORLD if comm_in is None else comm_in
         self.my_pe = self.comm.Get_rank()
         self.n_pes = self.comm.Get_size()

         self.halo  = halo_points

         if not hasattr(self,'nx'): self.nx = 1
         if not hasattr(self,'ny'): self.ny = 1
         if not hasattr(self,'nd'): self.nd = 24
         if not hasattr(self,'dx'): self.dx = 5e3
         if not hasattr(self,'dy'): self.dy = 5e3
         if not hasattr(self,'periodic_in_x'): self.periodic_in_x = False
         if not hasattr(self,'periodic_in_y'): self.periodic_in_y = True
         # global-edge conditions when not periodic
         if not hasattr(self,'bc_west'):  self.bc_west  = 'coast'
         if not hasattr(self,'bc_east'):  self.bc_east  = 'open'
         if not hasattr(self,'bc_south'): self.bc_south = 'open'
         if not hasattr(self,'bc_north'): self.bc_north = 'open'

         self.n_pes_x, self.n_pes_y = n_pes_x, n_pes_y
         if self.n_pes_x*self.n_pes_y != self.n_pes:
            raise ValueError(
               'n_pes_x*n_pes_y (=%d) must equal communicator size (=%d)'
               % (self.n_pes_x*self.n_pes_y, self.n_pes))

         self.my_blk_x = np__.mod(self.my_pe, self.n_pes_x) + 1
         self.my_blk_y = self.my_pe//self.n_pes_x        + 1

         self.xs_pe, self.xe_pe = self._local_range(self.nx, self.n_pes_x,
                                                    self.my_blk_x)
         self.ys_pe, self.ye_pe = self._local_range(self.ny, self.n_pes_y,
                                                    self.my_blk_y)

         self.nx_loc = self.xe_pe - self.xs_pe + 1
         self.ny_loc = self.ye_pe - self.ys_pe + 1

         self.Nx = self.nx_loc + 2*self.halo
         self.Ny = self.ny_loc + 2*self.halo
         # direction: always local, but haloed like x and y so that the
         # ring is a halo condition rather than a special case (wim_simple
         # carries the same halo on its direction and frequency axes)
         self.Nd = self.nd  + 2*self.halo

         self.Lx, self.Ly = self.nx*self.dx, self.ny*self.dy

         self.show_pe_layout()
         return


      @staticmethod
      def _local_range(n, n_pes, blk):
         """
         Distribute n global points over n_pes blocks as evenly as
         possible.  Returns the (start,end) global indices (1-based,
         inclusive) of block blk (1-based); the first (n mod n_pes)
         blocks get one extra point.
         """
         base, rem = divmod(n, n_pes)
         b = blk - 1
         start = b*base + min(b, rem) + 1
         cnt   = base + (1 if b < rem else 0)
         return start, start + cnt - 1


      def _neighbours_x(self):
          """(west,east) PE ranks, None where there is no neighbour."""
          west = east = None
          if self.my_blk_x > 1:
             west = self.my_pe - 1
          elif self.periodic_in_x and self.n_pes_x > 1:
             west = self.my_pe + (self.n_pes_x - 1)
          if self.my_blk_x < self.n_pes_x:
             east = self.my_pe + 1
          elif self.periodic_in_x and self.n_pes_x > 1:
             east = self.my_pe - (self.n_pes_x - 1)
          return west, east


      def _neighbours_y(self):
          """(south,north) PE ranks, None where there is no neighbour."""
          south = north = None
          if self.my_blk_y > 1:
             south = self.my_pe - self.n_pes_x
          elif self.periodic_in_y and self.n_pes_y > 1:
             south = self.my_pe + (self.n_pes_y - 1)*self.n_pes_x
          if self.my_blk_y < self.n_pes_y:
             north = self.my_pe + self.n_pes_x
          elif self.periodic_in_y and self.n_pes_y > 1:
             north = self.my_pe - (self.n_pes_y - 1)*self.n_pes_x
          return south, north


      def apply_bc(self, a, wrap_d = None):
          """
          Fill the halo points of array a (..., Nd, Ny, Nx).

          x,y: MPI halo exchange between neighbouring blocks, wrapping
          locally when the direction is periodic and not decomposed;
          global edges get the physical condition (coast = 0, open =
          zero gradient).

          Direction: ALWAYS periodic, filled by a local wrap, exactly as
          wim_simple fills its direction halo.  The theta ring is the one
          axis whose boundary condition can never be anything else, which
          is why it is safe to do here and why the transport can then
          treat theta like x and y.

          wrap_d selects that fill.  The default detects a direction axis
          by its length (axis -3 of size Nd), which is what every
          direction-resolved field carries; pass wrap_d = False for an
          array whose third-last axis is something else, e.g. a vertical
          current profile (nz, Ny, Nx).
          """
          if wrap_d is None:
             wrap_d = a.ndim >= 3 and a.shape[-3] == self.Nd
          if wrap_d:
             a = self.wrap_theta(a)
          a = self._exchange(a, axis=-1, n_pes=self.n_pes_x,
                             blk=self.my_blk_x,
                             periodic=self.periodic_in_x,
                             nbr=self._neighbours_x(),
                             bc_lo=self.bc_west, bc_hi=self.bc_east)
          a = self._exchange(a, axis=-2, n_pes=self.n_pes_y,
                             blk=self.my_blk_y,
                             periodic=self.periodic_in_y,
                             nbr=self._neighbours_y(),
                             bc_lo=self.bc_south, bc_hi=self.bc_north)
          return a


      def wrap_theta(self, a):
          """Fill the direction halo of a (..., Nd, Ny, Nx) array from the
          real bins.  Purely LOCAL (the ring is never decomposed), so this
          is the cheap part of apply_bc and can be used on its own wherever
          only the ring needs refreshing -- model.time_step does that after
          the source update, which writes the real bins only."""
          h = self.halo
          sl = (slice(None), slice(None))
          for i in range(h):
              # leading halo  <- real bins at the far (high) end
              a = OM.modify_array(a, (Ellipsis, i) + sl,
                                  a[..., -2*h + i, :, :])
              # trailing halo <- real bins at the near (low) end
              a = OM.modify_array(a, (Ellipsis, -h + i) + sl,
                                  a[..., h + i, :, :])
          return a


      def _exchange(self, a, axis, n_pes, blk, periodic, nbr, bc_lo, bc_hi):
          """One-dimensional halo fill along axis (-1 = x, -2 = y)."""
          h  = self.halo
          lo, hi = nbr
          sl = lambda s: tuple(s if k == (a.ndim + axis) else slice(None)
                               for k in range(a.ndim))

          if n_pes == 1:
             if periodic:
                a[sl(slice(0, h))]  = a[sl(slice(-2*h, -h))]
                a[sl(slice(-h, None))] = a[sl(slice(h, 2*h))]
                return a
          else:
             # send my inner edge, receive into my halo
             if lo is not None:
                buf = np__.ascontiguousarray(a[sl(slice(h, 2*h))])
                self.send(buf, lo, tag=1)
             if hi is not None:
                buf = np__.empty_like(np__.ascontiguousarray(
                                      a[sl(slice(-h, None))]))
                a[sl(slice(-h, None))] = self.recv(buf, hi, tag=1)
             if hi is not None:
                buf = np__.ascontiguousarray(a[sl(slice(-2*h, -h))])
                self.send(buf, hi, tag=2)
             if lo is not None:
                buf = np__.empty_like(np__.ascontiguousarray(
                                      a[sl(slice(0, h))]))
                a[sl(slice(0, h))] = self.recv(buf, lo, tag=2)
             if periodic or (lo is not None and hi is not None):
                return a

          # global edges (only reached when this PE owns them)
          if lo is None:
             if bc_lo == 'coast':
                a[sl(slice(0, h))] = 0.0
             else:                                  # 'open': zero gradient
                a[sl(slice(0, h))] = a[sl(slice(h, h+1))]
          if hi is None:
             if bc_hi == 'coast':
                a[sl(slice(-h, None))] = 0.0
             else:
                a[sl(slice(-h, None))] = a[sl(slice(-h-1, -h))]
          return a


      def unpad(self, a):
          """Strip the halo from the two trailing (y,x) axes."""
          s = slice(self.halo, -self.halo)
          return a[..., s, s]


      def gather_array(self, a):
          """
          Gather the unpadded trailing (y,x) block of a onto PE 0.
          Returns the global array on PE 0 and None elsewhere; a plain
          copy for a single PE.
          """
          loc = np__.ascontiguousarray(self.unpad(a))
          if self.n_pes == 1:
             return loc
          shp = loc.shape[:-2]
          out = None
          if self.my_pe == 0:
             out = np__.zeros(shp + (self.ny, self.nx), dtype=loc.dtype)
             out[..., self.ys_pe-1:self.ye_pe, self.xs_pe-1:self.xe_pe] = loc
             for pe in range(1, self.n_pes):
                info = self.comm.recv(source=pe, tag=10)
                buf  = np__.empty(shp + (info['ny'], info['nx']),
                                  dtype=loc.dtype)
                self.recv(buf, pe, tag=11)
                out[..., info['ys']-1:info['ye'],
                         info['xs']-1:info['xe']] = buf
          else:
             self.comm.send(dict(xs=self.xs_pe, xe=self.xe_pe,
                                 ys=self.ys_pe, ye=self.ye_pe,
                                 nx=self.nx_loc, ny=self.ny_loc),
                            dest=0, tag=10)
             self.send(loc, 0, tag=11)
          return out


      def show_pe_layout(self):
          """Print the processor mesh and the local index ranges."""
          if self.my_pe == 0:
             print(' global horizontal domain is nx=',self.nx,' ny=',self.ny)
             print(' directional grid (local on every PE): nd=',self.nd)
             print(' using',self.n_pes,'PEs:  n_pes_x=',self.n_pes_x,
                   ' n_pes_y=',self.n_pes_y)
          self.comm.Barrier()
          for pe in range(self.n_pes):
             if pe == self.my_pe:
                print(' PE#',self.my_pe,': i=',self.xs_pe,':',self.xe_pe,
                      ' j=',self.ys_pe,':',self.ye_pe)
             self.comm.Barrier()
          return


      def send(self, buf, dest, tag):
          self.comm.Send(np__.ascontiguousarray(buf), dest=dest, tag=tag)

      def recv(self, buf, source, tag):
          self.comm.Recv(buf, source=source, tag=tag)
          return buf
