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
from netCDF4 import Dataset


def local_block(M):
    """(j0, i0, ny_loc, nx_loc) of THIS PE's interior block, 0-based global
    start indices. Derived from Ny/Nx and ys_pe/xs_pe only: pe_realistic sets
    xe_pe = Nx (the PADDED extent) in the serial case, so xe_pe/ye_pe must
    not be used to size anything."""
    h = M.halo
    return M.ys_pe - 1, M.xs_pe - 1, M.Ny - 2*h, M.Nx - 2*h


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
