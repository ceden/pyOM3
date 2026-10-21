"""
Block I/O shared by the forcing sources and the experiment classes: where a
rank's block sits in the global grid, and RANK 0 READS, THE OTHERS RECEIVE.

The input files are contiguous float32. A hyperslab read [k, jsl, isl] by
every rank is, per level and rank, dozens of strided fragments of a few
hundred bytes on one shared file, and there are ~1000 ranks: latency bound,
minutes of setup. Reading the whole field on every rank instead does not fit
at 1/3 degree. So rank 0 reads one full 2-D record (one contiguous read),
packs the blocks in rank order and scatters them with a single MPI_Scatterv.

The communicator is the MODEL's, not COMM_WORLD: in a coupled run the ocean
sits on a sub-communicator next to the wave ranks and a COMM_WORLD collective
would deadlock.
"""
import numpy as np
import pyOM3 as OM
from netCDF4 import Dataset


# ---------------------------------------------------------------------------
# THE TWO LAYOUTS. Under MPI a process holds ITS block of the grid with one
# halo ring, (Ny, Nx) = (ny_loc + 2h, nx_loc + 2h), and the block's position
# is ys_pe/xs_pe. Under JAX one process holds the WHOLE grid, and the
# sharded array carries every shard's halo INLINE: (Ny, Nx) =
# (ny + 2h n_pes_y, nx + 2h n_pes_x), with model.pad/unpad converting
# between the global interior and that layout. The three helpers below are
# the only place the shared code needs to know which it is: a block is the
# whole grid under JAX, a padded field is made with pad (not with a zero
# array and a slice), and an interior is taken with unpad (not with a
# slice, which would keep the inline halos).
# ---------------------------------------------------------------------------
def local_block(M):
    """(j0, i0, ny_loc, nx_loc) of THIS PE's interior block, 0-based global
    start indices. MPI: derived from Ny/Nx and ys_pe/xs_pe only (pe_realistic
    sets xe_pe = Nx, the PADDED extent, in the serial case, so xe_pe/ye_pe
    must not be used to size anything). JAX: the whole grid."""
    h = M.halo
    if OM.jax:                            # pe: Nx - 2h n_pes_x; wim keeps nx/ny as the global size
        ny, nx = M.Ny - 2*h*M.n_pes_y, M.Nx - 2*h*M.n_pes_x
        return 0, 0, ny, nx
    return M.ys_pe - 1, M.xs_pe - 1, M.Ny - 2*h, M.Nx - 2*h


def pad_block(M, a, exchange=True):
    """The interior block `a` ((ny_loc, nx_loc) or (nz, ny_loc, nx_loc),
    numpy) as a padded model field with the halo exchanged (apply_bc), i.e.
    what `b = zeros(Ny, Nx); b[h:-h, h:-h] = a; apply_bc(b)` did under MPI.
    exchange=False leaves the halo at zero (an interior mask, say)."""
    a = np.asarray(a, np.float64)
    if OM.jax:
        b = M.pad(OM.np.asarray(a))
    else:
        h = M.halo
        shape = (M.Nz, M.Ny, M.Nx) if a.ndim == 3 else (M.Ny, M.Nx)
        b = np.zeros(shape); b[(slice(h, -h),)*a.ndim] = a
        b = OM.np.asarray(b)
    return M.apply_bc(b) if exchange else b


def interior(M, a, lead=0):
    """The interior of a padded field as a numpy array: MPI the local block,
    JAX the global interior (every inline halo removed). `lead` = number of
    leading axes that carry a halo as well (the wave model's 4-D fields:
    lead=2 for (nd, nf, ny, nx)); the trailing two are y and x, a 3-D field
    with lead=0 is (nz, ny, nx) with the z halo removed too."""
    a = OM.np.asarray(a) if OM.jax else np.asarray(a)
    if OM.jax:
        try:
            return np.asarray(M.unpad(a, lead) if lead else M.unpad(a))
        except TypeError:                 # pe's unpad has no lead argument
            return np.asarray(M.unpad(a))
    h = M.halo; s = slice(h, -h)
    n = a.ndim
    sl = (s,)*lead + (slice(None),)*(n - lead - 2) + (s, s) if lead else (s,)*n
    if not lead and n == 2:
        sl = (s, s)
    return a[sl]


class BlockScatter:
    """Scatter a full (ny, nx) field held on rank 0 of `comm` into each
    rank's block (j0, i0, ny_loc, nx_loc). The constructor is COLLECTIVE."""

    def __init__(self, comm, block):
        self.comm = comm
        j0, i0, self.nyl, self.nxl = (int(v) for v in block)
        if self.comm is None or self.comm.Get_size() == 1:
            self.ext = [(j0, i0, self.nyl, self.nxl)]
            self.counts = self.displs = None
            return
        ext = self.comm.gather((j0, i0, self.nyl, self.nxl), root=0)
        if self.comm.Get_rank() == 0:
            self.ext = ext
            self.counts = np.array([e[2]*e[3] for e in ext], np.int64)
            self.displs = np.concatenate(([0], np.cumsum(self.counts)[:-1]))
        else:
            self.ext = self.counts = self.displs = None

    def scatter(self, full):
        """full: (ny, nx) on rank 0, ignored elsewhere -> this rank's block."""
        from mpi4py import MPI
        if self.comm is None or self.comm.Get_size() == 1:
            j0, i0, nyl, nxl = self.ext[0]
            return np.asarray(full, np.float64)[j0:j0+nyl, i0:i0+nxl].copy()
        if self.comm.Get_rank() == 0:
            full = np.asarray(full, np.float64)
            send = np.concatenate([full[j0:j0+nyl, i0:i0+nxl].ravel()
                                   for j0, i0, nyl, nxl in self.ext])
            sendbuf = [send, self.counts, self.displs, MPI.DOUBLE]
        else:
            sendbuf = None
        recv = np.empty(self.nyl*self.nxl, np.float64)
        self.comm.Scatterv(sendbuf, recv, root=0)
        return recv.reshape(self.nyl, self.nxl)

    def read(self, path, vname, index=()):
        """rank 0 reads variable[index + (:, :)] of one file, every rank gets
        its own block. COLLECTIVE."""
        if self.comm is None or self.comm.Get_rank() == 0:
            with Dataset(path) as f:
                f.set_auto_mask(False)
                full = np.asarray(f.variables[vname][tuple(index) + (slice(None),
                                                                     slice(None))],
                                  np.float64)
        else:
            full = None
        return self.scatter(full)


def read_scatter2(M, path, vname, index=()):
    """BlockScatter.read for a model instance; the scatter object is built on
    first use and cached on the instance. COLLECTIVE over M.comm."""
    if not hasattr(M, '_scat'):
        M._scat = BlockScatter(getattr(M, 'comm', None), local_block(M))
    return M._scat.read(path, vname, index)
