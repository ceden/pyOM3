"""
Configuration-independent MPI plumbing for the WIM <-> pe_simple coupler:
world split into wave/ocean groups with an intercommunicator, and generic
gather/scatter between per-PE blocks and rank-0 full-domain fields.
"""
import numpy as np
from mpi4py import MPI


def split_world(n_pes_wave, n_pes_ocean):
    """Split COMM_WORLD into wave (color 0: ranks < n_pes_wave), ocean
    (color 1: next n_pes_ocean ranks) and IDLE (color 2: any surplus ranks,
    so the job can be launched on whole nodes without matching the active
    rank count exactly). Builds the wave<->ocean intercommunicator between
    the two rank 0s, and an ACTIVE communicator spanning wave+ocean for the
    per-coupling-step barriers (idle ranks must never participate in those,
    they only join the top-level setup/shutdown collectives on world).
    Returns (world, my_pe, color, comm, intercomm, active)."""
    world = MPI.COMM_WORLD
    my_pe = world.Get_rank()
    n_active = n_pes_wave + n_pes_ocean
    color = 0 if my_pe < n_pes_wave else (1 if my_pe < n_active else 2)
    comm = world.Split(color, key=0)
    # active communicator: wave+ocean together; idle ranks get their own
    # (unused) communicator -- Split is collective on world, all must call
    active = world.Split(0 if color < 2 else 1, key=0)
    if color == 0:
        intercomm = comm.Create_intercomm(0, world, n_pes_wave, tag=42)
    elif color == 1:
        intercomm = comm.Create_intercomm(0, world, 0, tag=42)
    else:
        intercomm = None
    return world, my_pe, color, comm, intercomm, active


# ---------------------------------------------------------------------------
# gather for UNPADDED per-PE fields (..., ny_loc, nx_loc), as returned by
# calc_stokes / calc_stress / calc_stress_work (already halo-stripped, "goes
# to ocean model"). Assembles the full (..., ny, nx) domain on PE0 (None
# elsewhere). Mirror of scatter_array_mpi. (wim's own gather_array_mpi assumes
# a PADDED model array and would strip a halo that isn't there.)
# ---------------------------------------------------------------------------
def gather_unpadded_mpi(M, a, ny, nx):
    import pyOM3 as OM
    lead = a.shape[:-2]
    if M.my_pe == 0:
        b = OM.np.empty(lead + (ny, nx), OM.prec)
        b = OM.modify_array(b, tuple(slice(None) for _ in lead) +
                            (slice(M.ys_pe-1, M.ye_pe), slice(M.xs_pe-1, M.xe_pe)), a)
    else:
        b = None
    for pe in range(1, M.n_pes):
        if M.my_pe == pe:
            M.send(OM.np.array([M.xs_pe, M.xe_pe, M.ys_pe, M.ye_pe], OM.prec), 0, 63)
            M.send(OM.np.ascontiguousarray(a), 0, 64)
        if M.my_pe == 0:
            dims = OM.np.empty((4,), OM.prec)
            xs, xe, ys, ye = (int(v) for v in M.recv(dims, pe, 63))
            buf = OM.np.empty(lead + (ye-ys+1, xe-xs+1), OM.prec)
            blk = M.recv(buf, pe, 64)
            b = OM.modify_array(b, tuple(slice(None) for _ in lead) +
                                (slice(ys-1, ye), slice(xs-1, xe)), blk)
    return b


# ---------------------------------------------------------------------------
# generic scatter: PE0 holds a full-domain field (..., ny, nx); send each PE
# its [ys_pe-1:ye_pe, xs_pe-1:xe_pe] block. Mirror of the models' gather.
# `full` is meaningful only on PE0 (None elsewhere). Returns each PE's block
# (..., ny_loc, nx_loc), halo NOT included. Works for wave or ocean (both
# expose xs_pe/xe_pe/ys_pe/ye_pe, n_pes, and a comm-aware self.send/recv).
# ---------------------------------------------------------------------------
def scatter_array_mpi(M, full, lead_shape):
    import pyOM3 as OM
    ny_loc = M.ye_pe - M.ys_pe + 1
    nx_loc = M.xe_pe - M.xs_pe + 1
    # PE0 keeps its own block directly
    if M.my_pe == 0:
        my_block = OM.np.ascontiguousarray(
            full[..., M.ys_pe-1:M.ye_pe, M.xs_pe-1:M.xe_pe]).copy()
    else:
        my_block = None

    for pe in range(1, M.n_pes):
        if M.my_pe == 0:
            # ask pe for its extents, then send it its block
            dims = OM.np.empty((4,), OM.prec)
            dims = M.recv(dims, pe, 61)
            xs, xe, ys, ye = (int(v) for v in dims)
            block = OM.np.ascontiguousarray(full[..., ys-1:ye, xs-1:xe]).copy()
            M.send(block, pe, 62)
        if M.my_pe == pe:
            M.send(OM.np.array([M.xs_pe, M.xe_pe, M.ys_pe, M.ye_pe], OM.prec), 0, 61)
            buf = OM.np.empty(tuple(lead_shape) + (ny_loc, nx_loc), OM.prec)
            my_block = M.recv(buf, 0, 62)
    return my_block

