"""
 Setting halo points to boundary conditions for the WIM model.

 Domain decomposition is in x and y only. The direction (d) and frequency (f)
 dimensions are always local (never split across PEs), so every PE holds the
 full spectral grid for its horizontal sub-block.

 The model arrays are C-ordered with shape (Nd,Nf,Ny,Nx), i.e. x is the last
 (fastest) axis and y the second-last. This mirrors the nh_simple decomposition
 (which is Fortran-ordered with x first), but here the halo exchanges always act
 on the trailing two axes (...,y,x).

 PE layout (block coordinates, 1-based):
   my_blk_x in [1, n_pes_x]   (varies fastest)
   my_blk_y in [1, n_pes_y]
   my_pe = (my_blk_y-1)*n_pes_x + (my_blk_x-1)
"""
import pyOM3     as OM
from   pyOM3.wim_simple import setup
from   mpi4py    import MPI
import numpy     as np__


class boundary(setup):


      def __init__(self, parameter = {}, n_pes_x = 1, n_pes_y = 1,
                                         halo_points = 1, comm_in = None):
         """
         Initialize the x,y domain decomposition and add halo points.

         Input nx,ny is the GLOBAL horizontal shape without halo points;
         nf,nd are global and stay local on every PE.
         """
         super().__init__(parameter)

         self.fortran_ordering = False   # WIM arrays are C-ordered (Nd,Nf,Ny,Nx)

         self.comm  = MPI.COMM_WORLD if comm_in is None else comm_in
         self.my_pe = self.comm.Get_rank()
         self.n_pes = self.comm.Get_size()

         self.halo  = halo_points        # number of halo points at each side

         # ------------------------------------------------------------------
         # set up the 2D processor grid (x,y only)
         # ------------------------------------------------------------------
         self.n_pes_x, self.n_pes_y = n_pes_x, n_pes_y
         if self.n_pes_x*self.n_pes_y != self.n_pes:
            raise ValueError(
               'n_pes_x*n_pes_y (=%d) must equal communicator size (=%d)'
               % (self.n_pes_x*self.n_pes_y, self.n_pes))

         # block coordinates (1-based), x varies fastest
         self.my_blk_x = np__.mod(self.my_pe, self.n_pes_x) + 1
         self.my_blk_y = self.my_pe//self.n_pes_x        + 1

         # ------------------------------------------------------------------
         # local horizontal extent (global index range, 1-based inclusive)
         # split nx over n_pes_x and ny over n_pes_y as evenly as possible
         # ------------------------------------------------------------------
         self.xs_pe, self.xe_pe = self._local_range(self.nx, self.n_pes_x, self.my_blk_x)
         self.ys_pe, self.ye_pe = self._local_range(self.ny, self.n_pes_y, self.my_blk_y)

         # local block sizes (number of real, non-halo points on this PE)
         self.nx_loc = self.xe_pe - self.xs_pe + 1
         self.ny_loc = self.ye_pe - self.ys_pe + 1

         # local shapes WITH halo points
         self.Nx = self.nx_loc + 2*self.halo
         self.Ny = self.ny_loc + 2*self.halo
         self.Nf = self.nf  + 2*self.halo      # frequency: always local
         self.Nd = self.nd  + 2*self.halo      # direction: always local

         self.Lx, self.Ly = self.nx*self.dx, self.ny*self.dy

         self.show_pe_layout()
         return


      @staticmethod
      def _local_range(n, n_pes, blk):
         """
         Distribute n global points over n_pes blocks as evenly as possible.
         Returns the (start,end) global indices (1-based, inclusive) of block
         number blk (1-based). The first (n mod n_pes) blocks get one extra
         point.
         """
         base, rem = divmod(n, n_pes)
         b = blk - 1                         # 0-based block index
         start = b*base + min(b, rem)
         size  = base + (1 if b < rem else 0)
         return start + 1, start + size


      # ---------------------------------------------------------------------
      #  neighbour ranks in the 2D processor grid (periodic wrap)
      # ---------------------------------------------------------------------
      def _neighbours_x(self):
         """west/east neighbour ranks."""
         my_pe, n_pes_x = self.my_pe, self.n_pes_x
         west = my_pe + n_pes_x - 1   if self.my_blk_x == 1        else my_pe - 1
         east = my_pe - (n_pes_x - 1) if self.my_blk_x == n_pes_x  else my_pe + 1
         return west, east

      def _neighbours_y(self):
         """south/north neighbour ranks."""
         my_pe, n_pes_x, n_pes_y = self.my_pe, self.n_pes_x, self.n_pes_y
         south = my_pe + n_pes_x*(n_pes_y-1) if self.my_blk_y == 1       else my_pe - n_pes_x
         north = my_pe - n_pes_x*(n_pes_y-1) if self.my_blk_y == n_pes_y else my_pe + n_pes_x
         return south, north


      def apply_bc(self, a, bc_value = 0):
          """
          Apply boundary conditions to the halo points of array a.

          Horizontal x,y: MPI halo exchange between neighbouring blocks
          (periodic or closed depending on periodic_in_x / periodic_in_y).
          Frequency f: always closed (set to bc_value).
          Direction d: always periodic (local wrap).

          Works on the WIM 4D arrays of shape (Nd,Nf,Ny,Nx). The two leading
          axes (d,f) are local; the two trailing axes (y,x) are decomposed.
          """
          halo = self.halo
          x = slice(None)
          y = slice(None) if len(a.shape) > 1 else None
          f = slice(None) if len(a.shape) > 2 else None
          d = slice(None) if len(a.shape) > 3 else None

          # ---------------- x direction (last axis) ----------------
          a = self._exchange_x(a, d, f, y, bc_value)

          # ---------------- y direction (second-last axis) ---------
          a = self._exchange_y(a, d, f, x, bc_value)

          # ---------------- frequency: always closed ---------------
          # only meaningful when a real frequency axis is present, i.e. the
          # array is 4D with a frequency axis large enough to hold halo cells.
          # (lifted 2D forcing fields with a singleton f-axis are skipped.)
          if len(a.shape) > 2 and a.shape[-3] > 2*halo:
             for i in range(halo):
                 a = OM.modify_array(a, (d,  i,     y, x), bc_value)
                 a = OM.modify_array(a, (d, -1 - i, y, x), bc_value)

          # ---------------- direction: always periodic -------------
          # only when a real direction axis is present (4D, large enough).
          if len(a.shape) > 3 and a.shape[-4] > 2*halo:
             for i in range(halo):
                 # leading halo  <- real cells at the far (high) end
                 a = OM.modify_array(a, (i,         f, y, x), a[(-2*halo + i, f, y, x)])
                 # trailing halo <- real cells at the near (low) end
                 a = OM.modify_array(a, (-halo + i, f, y, x), a[( halo + i,   f, y, x)])
          return a


      def _exchange_x(self, a, d, f, y, bc_value):
          """halo exchange / boundary in x (last axis)."""
          halo, n_pes_x = self.halo, self.n_pes_x

          if n_pes_x > 1:
             west, east = self._neighbours_x()
             buf = OM.np.empty(a.shape[:-1] + (1,), OM.prec)   # one x-plane

             for i in range(halo):
                 # send my first real column to west, recv into my east halo
                 sl_send = (d, f, y, halo + i)
                 sl_recv = (d, f, y, i - halo)
                 if self.my_blk_x != 1:        self.send(a[sl_send][..., None], west, 49)
                 if self.my_blk_x != n_pes_x:  a = OM.modify_array(
                                                   a, sl_recv,
                                                   self.recv(buf, east, 49)[..., 0])
                 if self.periodic_in_x:
                    if self.my_blk_x == 1:        self.send(a[sl_send][..., None], west, 47)
                    if self.my_blk_x == n_pes_x:  a = OM.modify_array(
                                                      a, sl_recv,
                                                      self.recv(buf, east, 47)[..., 0])
                 else:
                    if self.my_blk_x == n_pes_x:  a = OM.modify_array(a, sl_recv, bc_value)

                 # send my last real column to east, recv into my west halo
                 sl_send = (d, f, y, -halo - 1 - i)
                 sl_recv = (d, f, y,  halo - 1 - i)
                 if self.my_blk_x != n_pes_x:  self.send(a[sl_send][..., None], east, 153)
                 if self.my_blk_x != 1:        a = OM.modify_array(
                                                   a, sl_recv,
                                                   self.recv(buf, west, 153)[..., 0])
                 if self.periodic_in_x:
                    if self.my_blk_x == n_pes_x:  self.send(a[sl_send][..., None], east, 152)
                    if self.my_blk_x == 1:        a = OM.modify_array(
                                                      a, sl_recv,
                                                      self.recv(buf, west, 152)[..., 0])
                 else:
                    if self.my_blk_x == 1:        a = OM.modify_array(a, sl_recv, bc_value)
          else:
             # single block in x: purely local
             for i in range(halo):
                 if self.periodic_in_x:
                    a = OM.modify_array(a, (d, f, y, halo - 1 - i), a[(d, f, y, -halo - 1 - i)])
                    a = OM.modify_array(a, (d, f, y, i - halo),     a[(d, f, y,  halo + i)])
                 else:
                    a = OM.modify_array(a, (d, f, y, halo - 1 - i), bc_value)
                    a = OM.modify_array(a, (d, f, y, i - halo),     bc_value)
          return a


      def _exchange_y(self, a, d, f, x, bc_value):
          """halo exchange / boundary in y (second-last axis)."""
          halo, n_pes_y = self.halo, self.n_pes_y

          if n_pes_y > 1:
             south, north = self._neighbours_y()
             buf = OM.np.empty(a.shape[:-2] + (1,) + a.shape[-1:], OM.prec)  # one y-plane

             for i in range(halo):
                 # send my first real row to south, recv into my north halo
                 sl_send = (d, f, halo + i,      x)
                 sl_recv = (d, f, i - halo,      x)
                 if self.my_blk_y != 1:        self.send(a[sl_send][..., None, :], south, 49)
                 if self.my_blk_y != n_pes_y:  a = OM.modify_array(
                                                   a, sl_recv,
                                                   self.recv(buf, north, 49)[..., 0, :])
                 if self.periodic_in_y:
                    if self.my_blk_y == 1:        self.send(a[sl_send][..., None, :], south, 47)
                    if self.my_blk_y == n_pes_y:  a = OM.modify_array(
                                                      a, sl_recv,
                                                      self.recv(buf, north, 47)[..., 0, :])
                 else:
                    if self.my_blk_y == n_pes_y:  a = OM.modify_array(a, sl_recv, bc_value)

                 # send my last real row to north, recv into my south halo
                 sl_send = (d, f, -halo - 1 - i, x)
                 sl_recv = (d, f,  halo - 1 - i, x)
                 if self.my_blk_y != n_pes_y:  self.send(a[sl_send][..., None, :], north, 153)
                 if self.my_blk_y != 1:        a = OM.modify_array(
                                                   a, sl_recv,
                                                   self.recv(buf, south, 153)[..., 0, :])
                 if self.periodic_in_y:
                    if self.my_blk_y == n_pes_y:  self.send(a[sl_send][..., None, :], north, 152)
                    if self.my_blk_y == 1:        a = OM.modify_array(
                                                      a, sl_recv,
                                                      self.recv(buf, south, 152)[..., 0, :])
                 else:
                    if self.my_blk_y == 1:        a = OM.modify_array(a, sl_recv, bc_value)
          else:
             # single block in y: purely local
             for i in range(halo):
                 if self.periodic_in_y:
                    a = OM.modify_array(a, (d, f, halo - 1 - i, x), a[(d, f, -halo - 1 - i, x)])
                    a = OM.modify_array(a, (d, f, i - halo,     x), a[(d, f,  halo + i,     x)])
                 else:
                    a = OM.modify_array(a, (d, f, halo - 1 - i, x), bc_value)
                    a = OM.modify_array(a, (d, f, i - halo,     x), bc_value)
          return a


      def gather_array_mpi(self, a):
          """
          Gather a distributed WIM array onto PE0.

          Returns the full-domain array with the halo removed. The decomposed
          (trailing) y,x axes are reassembled; the local leading axes (d,f or
          just f, etc.) are kept. Works for any array whose last two axes are
          (y,x): e.g. (Nd,Nf,Ny,Nx), (Nf,Ny,Nx) or (Ny,Nx).

          Only PE0 returns a meaningful array; other PEs return None.
          """
          halo = self.halo
          lead = a.shape[:-2]                 # local, non-decomposed axes

          sl_send = tuple(slice(None) for _ in lead) + \
                    (slice(halo, -halo), slice(halo, -halo))

          if self.my_pe == 0:
             b = OM.np.empty(lead + (self.ny, self.nx), OM.prec)
             sl_recv = tuple(slice(None) for _ in lead) + \
                       (slice(self.ys_pe - 1, self.ye_pe),
                        slice(self.xs_pe - 1, self.xe_pe))
             b = OM.modify_array(b, sl_recv, a[sl_send])
          else:
             b = None

          for pe in range(1, self.n_pes):
              if self.my_pe == pe:
                 dims = OM.np.array([self.xs_pe, self.xe_pe,
                                     self.ys_pe, self.ye_pe], dtype='i')
                 self.comm.Send([dims, MPI.INT], 0, 49)
                 self.send(a[sl_send], 0, 153)
              if self.my_pe == 0:
                 dims = OM.np.zeros((4,), dtype='i')
                 self.comm.Recv([dims, MPI.INT], source=pe, tag=49)
                 xs, xe, ys, ye = dims
                 sl_recv = tuple(slice(None) for _ in lead) + \
                           (slice(ys - 1, ye), slice(xs - 1, xe))
                 buf = OM.np.ascontiguousarray(b[sl_recv]).copy()
                 b = OM.modify_array(b, sl_recv, self.recv(buf, pe, 153))
          return b


      def show_pe_layout(self):
         """print pe layout"""
         if self.my_pe == 0:
            print('global horizontal domain is nx=', self.nx, 'ny=', self.ny)
            print('spectral grid (local on every PE): nf=', self.nf, 'nd=', self.nd)
            print('using', self.n_pes, 'PEs:  n_pes_x=', self.n_pes_x,
                  ' n_pes_y=', self.n_pes_y)
            print('fortran_ordering = ', self.fortran_ordering)
         self.comm.Barrier()

         if self.my_pe == 0: print('Layout of processor mesh')
         for pe in range(self.n_pes):
             if pe == self.my_pe:
                print('PE#', pe, ': my_blk_x = ', self.my_blk_x,
                                  ' my_blk_y = ', self.my_blk_y)
             self.comm.Barrier()

         if self.my_pe == 0: print('Layout of physical domain decomposition')
         for pe in range(self.n_pes):
             if pe == self.my_pe:
                print('PE#', self.my_pe, ': i=', self.xs_pe, ':', self.xe_pe,
                                         ' j=', self.ys_pe, ':', self.ye_pe)
             self.comm.Barrier()
         return


      def rollx(self, a, shift):
          """Shift last (x) axis by shift points. shift=-1 -> positive direction."""
          return OM.np.roll(a, shift=shift, axis=-1)

      def rolly(self, a, shift):
          """Shift 2.-last (y) axis by shift points."""
          return OM.np.roll(a, shift=shift, axis=-2)

      def rollf(self, a, shift):
          """Shift 3.-last (freq) axis by shift points."""
          return OM.np.roll(a, shift=shift, axis=-3)

      def rolld(self, a, shift):
          """Shift 4.-last (direction) axis by shift points."""
          return OM.np.roll(a, shift=shift, axis=-4)


      def unpad(self, a):
          s = slice(self.halo, -self.halo)
          if   len(a.shape) == 4: sl = (s, s, s, s)
          elif len(a.shape) == 3: sl = (s, s, s)
          else:                   sl = (s, s)
          return a[sl]

      def send(self, buf, dest, tag):
          self.comm.Send(OM.np.ascontiguousarray(buf), dest=dest, tag=tag)

      def recv(self, buf, source, tag):
          self.comm.Recv(buf, source=source, tag=tag)
          return buf
