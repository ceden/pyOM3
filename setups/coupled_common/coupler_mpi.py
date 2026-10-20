"""
Configuration-independent MPI plumbing for the WIM <-> pe coupler: world
split into wave/ocean groups with an intercommunicator, and generic
gather/scatter between per-PE blocks and rank-0 full-domain fields.

The block exchange with rank 0 (gather_unpadded_mpi, scatter_array_mpi) is
one Gatherv / Scatterv rather than a loop of point-to-point messages; see the
comment above those two routines.

Difference to the acc1 version: the local block extent is taken from
Ny/Nx and ys_pe/xs_pe (pe_block below) instead of ye_pe/xe_pe. pe_simple
and pe_realistic set xe_pe = Nx, ye_pe = Ny -- the PADDED extents -- when
they run on a SINGLE PE, so xe_pe - xs_pe + 1 is nx + 2*halo there. The
acc1 form survives that (numpy clamps the over-long slice, and the buffer
sized from it is only used in the n_pes > 1 branch, where the expression
is right), so this is not a bug fix -- it just makes the extents explicit
instead of accidentally correct. The wave model's own even-split
decomposition is unaffected either way.
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
    n_pes_wave = 0 is the ocean alone: no wave group, intercomm is None.
    Returns (world, my_pe, color, comm, intercomm, active)."""
    world = MPI.COMM_WORLD
    my_pe = world.Get_rank()
    n_active = n_pes_wave + n_pes_ocean
    color = 0 if my_pe < n_pes_wave else (1 if my_pe < n_active else 2)
    comm = world.Split(color, key=0)
    # active communicator: wave+ocean together; idle ranks get their own
    # (unused) communicator -- Split is collective on world, all must call
    active = world.Split(0 if color < 2 else 1, key=0)
    if n_pes_wave == 0:         # ocean alone: nothing to connect
        intercomm = None
    elif color == 0:
        intercomm = comm.Create_intercomm(0, world, n_pes_wave, tag=42)
    elif color == 1:
        intercomm = comm.Create_intercomm(0, world, 0, tag=42)
    else:
        intercomm = None
    return world, my_pe, color, comm, intercomm, active


def split_world3(n_pes_wave, n_pes_ocean, n_pes_atmos):
    """Three components: wave (colour 0), ocean (1), ATMOSPHERE (2), idle (3).
    Two intercommunicators, wave<->ocean as split_world builds it and
    atmosphere<->ocean; the atmosphere and the waves do not talk to each
    other, the ocean is the hub (coupler_loop: the wave wind comes from the
    ocean's copy of the atmosphere's stress, the wave input p_in reaches the
    atmosphere through the ocean). n_pes_wave = 0 is atmosphere + ocean
    alone. Returns (world, my_pe, color, comm, intercomm_wo, active,
    intercomm_ao); `active` spans the three components."""
    world = MPI.COMM_WORLD
    my_pe = world.Get_rank()
    n_wo = n_pes_wave + n_pes_ocean
    n_active = n_wo + n_pes_atmos
    if my_pe < n_pes_wave:   color = 0
    elif my_pe < n_wo:       color = 1
    elif my_pe < n_active:   color = 2
    else:                    color = 3
    comm = world.Split(color, key=0)
    active = world.Split(0 if color < 3 else 1, key=0)
    # intercomm creation is collective over the two groups involved, so the
    # calls must be ordered identically on every rank: wave-ocean first
    inter_wo = inter_ao = None
    if n_pes_wave > 0:
        if color == 0:   inter_wo = comm.Create_intercomm(0, world, n_pes_wave, tag=42)
        elif color == 1: inter_wo = comm.Create_intercomm(0, world, 0, tag=42)
    if n_pes_atmos > 0:
        if color == 2:   inter_ao = comm.Create_intercomm(0, world, n_pes_wave, tag=43)
        elif color == 1: inter_ao = comm.Create_intercomm(0, world, n_wo, tag=43)
    return world, my_pe, color, comm, inter_wo, active, inter_ao


def pe_block(M):
    """(j0, i0, ny_loc, nx_loc) of this PE's interior block: 0-based global
    start indices and sizes. See the module docstring for why ye_pe/xe_pe
    are not used."""
    h = M.halo
    return M.ys_pe - 1, M.xs_pe - 1, M.Ny - 2*h, M.Nx - 2*h


# ---------------------------------------------------------------------------
# BLOCK EXCHANGE WITH RANK 0, AS ONE COLLECTIVE.
#
# Both routines below used to move the data with a Python loop of
# point-to-point send/recv, one rank at a time, plus a two-message handshake
# per rank to learn its extents. That is fine at 3600 cells on 16 ranks and
# it is the run at 1/3 degree: on 4096 wave ranks the loop carries 1.5 GB
# through one process in 4095 sequential messages, and the coupled step spent
# 4.1 s (ocean) and 5.1 s (wave) of its 6.5 s in exactly these two calls,
# against about 1 s of physics in both models together.
#
# They are now one Gatherv / Scatterv on the model communicator, with the
# counts and displacements built once from the extents (allgather, cached on
# the instance). Rank 0 packs and unpacks, because a full-domain array is not
# the concatenation of the blocks -- that is a local memory copy at several
# GB/s instead of a network round trip per rank.
#
# The data moved, and the result, are unchanged.
# ---------------------------------------------------------------------------
_MPI_OF = {8: MPI.DOUBLE, 4: MPI.FLOAT}


def _mpi_type(dtype):
    t = _MPI_OF.get(np.dtype(dtype).itemsize)
    if t is None:
        raise TypeError('coupler_mpi: no MPI type for %s' % np.dtype(dtype))
    return t


def _extents(M):
    """[(j0, i0, ny_loc, nx_loc)] of every rank of M.comm, in rank order.
    One allgather, cached: the decomposition does not change during a run."""
    if not hasattr(M, '_cpl_ext'):
        M._cpl_ext = M.comm.allgather(tuple(int(v) for v in pe_block(M)))
    return M._cpl_ext


def _counts(ext, nlead):
    counts = np.array([nlead*e[2]*e[3] for e in ext], np.int64)
    return counts, np.concatenate(([0], np.cumsum(counts)[:-1]))


# ---------------------------------------------------------------------------
# gather for UNPADDED per-PE fields (..., ny_loc, nx_loc), as returned by
# calc_stokes / calc_stress / calc_stress_work (already halo-stripped, "goes
# to ocean model"). Assembles the full (..., ny, nx) domain on PE0 (None
# elsewhere). Mirror of scatter_array_mpi. (the models' own gather_array_mpi
# assumes a PADDED model array and would strip a halo that isn't there.)
# ---------------------------------------------------------------------------
def gather_unpadded_mpi(M, a, ny, nx):
    import pyOM3 as OM
    a = np.ascontiguousarray(np.asarray(a), OM.prec)
    lead = a.shape[:-2]
    nlead = int(np.prod(lead, dtype=np.int64)) if lead else 1
    j0, i0, ny_loc, nx_loc = pe_block(M)
    if M.n_pes == 1:
        b = np.empty(lead + (ny, nx), OM.prec)
        b[..., j0:j0+ny_loc, i0:i0+nx_loc] = a
        return b
    ext = _extents(M)
    counts, displs = _counts(ext, nlead)
    if M.my_pe == 0:
        flat = np.empty(int(counts.sum()), OM.prec)
        recvbuf = [flat, counts, displs, _mpi_type(OM.prec)]
    else:
        flat, recvbuf = None, None
    M.comm.Gatherv(a, recvbuf, root=0)
    if M.my_pe != 0:
        return None
    b = np.empty(lead + (ny, nx), OM.prec)
    for pe, (j0r, i0r, nyr, nxr) in enumerate(ext):
        blk = flat[displs[pe]:displs[pe]+counts[pe]].reshape(lead + (nyr, nxr))
        b[..., j0r:j0r+nyr, i0r:i0r+nxr] = blk
    return b


# ---------------------------------------------------------------------------
# generic scatter: PE0 holds a full-domain field (..., ny, nx); send each PE
# its block. Mirror of gather_unpadded_mpi. `full` is meaningful only on PE0
# (None elsewhere). Returns each PE's block (..., ny_loc, nx_loc), halo NOT
# included. Works for wave or ocean.
# ---------------------------------------------------------------------------
def scatter_array_mpi(M, full, lead_shape):
    import pyOM3 as OM
    lead = tuple(int(v) for v in lead_shape)
    nlead = int(np.prod(lead, dtype=np.int64)) if lead else 1
    j0, i0, ny_loc, nx_loc = pe_block(M)
    if M.n_pes == 1:
        return np.ascontiguousarray(
            np.asarray(full)[..., j0:j0+ny_loc, i0:i0+nx_loc]).copy()
    ext = _extents(M)
    counts, displs = _counts(ext, nlead)
    if M.my_pe == 0:
        full = np.ascontiguousarray(np.asarray(full), OM.prec)
        flat = np.empty(int(counts.sum()), OM.prec)
        for pe, (j0r, i0r, nyr, nxr) in enumerate(ext):
            flat[displs[pe]:displs[pe]+counts[pe]] = \
                full[..., j0r:j0r+nyr, i0r:i0r+nxr].ravel()
        sendbuf = [flat, counts, displs, _mpi_type(OM.prec)]
    else:
        sendbuf = None
    recv = np.empty(nlead*ny_loc*nx_loc, OM.prec)
    M.comm.Scatterv(sendbuf, recv, root=0)
    return recv.reshape(lead + (ny_loc, nx_loc))


# ---------------------------------------------------------------------------
# DIRECT BLOCK-TO-BLOCK EXCHANGE BETWEEN THE TWO COMPONENTS.
#
# Why this exists. The routines above assemble a full-domain field on rank 0,
# the two rank 0s ship it over the intercommunicator, and the other side
# scatters it. Per coupling step that is u and v out (1.5 GB at 1/3 degree)
# and u_S back (another 1.5 GB), through two processes. Measured on 40 nodes:
# 4.6 s on the ocean and 6.4 s on the wave side of a 7.5 s coupling step,
# against about 1 s of physics in both models together. It is insensitive to
# how the bytes move -- a loop of 4095 messages and a single Scatterv cost the
# same, because the root's strided pack of 136-byte fragments is the floor.
#
# What this does instead. The two models share the horizontal grid, so a
# local block of one overlaps a handful of blocks of the other, and the
# overlap is a rectangle. An MPI INTERCOMMUNICATOR spans both groups, so any
# rank of one can address any rank of the other by its remote rank number.
# Each rank therefore sends its own block straight to the ranks that own
# those cells and receives its own part back. Nothing is assembled, nothing
# is packed from a full-domain array, and the 1.5 GB is spread over the
# ranks: about 300 kB each with 936 ocean and 4096 wave PEs.
#
# The overlap map is fixed for the run (the decompositions do not change), so
# it is built once, from the extents both groups exchange at setup.
#
# The gather/scatter routines above stay: the coupling-file writer genuinely
# wants a full-domain field on rank 0, it just does not need one every step.
#
# A COARSER WAVE GRID (EXP_WAVE_COARSEN = c > 1). The wave grid is then an
# INTEGER COARSENING of the ocean grid: a wave cell is a group of c x c ocean
# cells and its faces are ocean cell faces (coarse_edges below; the remainder
# of n/c goes into the LAST group, so no wave cell is narrower than c ocean
# cells and the wave CFL is not set by a sliver). Every block is described in
# FINE (ocean) indices, so the overlap of a wave block and an ocean block is
# still a rectangle and the exchange stays block to block. What changes is
# what travels:
#   wave -> ocean  the wave PE sends the coarse cells that cover the overlap
#                  and the ocean PE copies each value into the ocean cells of
#                  that group (piecewise constant: u_S, tau, Phi_oc, tau_in
#                  and Hs are all per unit sea area, so the wave cell's value
#                  IS the ocean cell's value);
#   ocean -> wave  the ocean PE sends, per coarse cell, the WEIGHTED SUM of
#                  its part of the group, sum(w a), and the wave PE adds the
#                  parts of all ocean PEs and divides by sum(w), which is
#                  static and exchanged once. With w = area * mask this is the
#                  area mean over the WET ocean cells of the group, level by
#                  level. The reduction happens on the ocean side, so the
#                  message is c*c times smaller as well.
# c = 1 takes the original code path, slice for slice.
# ---------------------------------------------------------------------------
def coarse_edges(n, c):
    """Group edges of an integer coarsening by c of n fine cells, in fine
    indices: group g is the fine cells e[g] .. e[g+1]-1. n//c groups; the
    remainder is absorbed by the LAST one (c .. 2c-1 cells)."""
    c = int(c)
    if c < 1 or n//c < 1:
        raise ValueError('coarse_edges: cannot coarsen %d cells by %d' % (n, c))
    e = np.arange(n//c + 1, dtype=np.int64)*c
    e[-1] = n
    return e


class CoarseMap:
    """The wave grid as an integer coarsening of the (ny, nx) ocean grid."""

    def __init__(self, ny, nx, c):
        self.c = int(c)
        self.je, self.ie = coarse_edges(ny, c), coarse_edges(nx, c)
        self.ny_c, self.nx_c = len(self.je) - 1, len(self.ie) - 1
        # fine index -> coarse index
        self.cj = np.repeat(np.arange(self.ny_c), np.diff(self.je))
        self.ci = np.repeat(np.arange(self.nx_c), np.diff(self.ie))


def _group_offsets(cidx):
    """reduceat offsets of a run of fine cells with coarse indices cidx."""
    return np.concatenate(([0], 1 + np.flatnonzero(np.diff(cidx)))).astype(np.intp)


class BlockExchange:
    """Block-to-block transfer over the intercommunicator. One instance per
    component, built once; `comm` is the component's own communicator.

    cmap   None or a CoarseMap: the wave grid is a coarsening of the ocean's
    role   'wave' or 'ocean', needed only with a cmap: which of the two grids
           this component's block lives on."""

    def __init__(self, M, comm, intercomm, cmap=None, role=None):
        import pyOM3 as OM
        self.comm  = comm
        self.inter = intercomm
        self.prec  = OM.prec
        self.type  = _mpi_type(OM.prec)
        self.cmap  = cmap if (cmap is not None and cmap.c > 1) else None
        self.role  = role
        if self.cmap is not None and role not in ('wave', 'ocean'):
            raise ValueError("BlockExchange: role must be 'wave' or 'ocean'")
        j0, i0, ny_loc, nx_loc = (int(v) for v in pe_block(M))
        self.shape = (ny_loc, nx_loc)
        if self.cmap is not None and role == 'wave':
            # the wave block in FINE indices
            je, ie = self.cmap.je, self.cmap.ie
            if j0 + ny_loc > self.cmap.ny_c or i0 + nx_loc > self.cmap.nx_c:
                raise ValueError('BlockExchange: the wave block does not fit '
                                 'the %d x %d coarse grid'
                                 % (self.cmap.nx_c, self.cmap.ny_c))
            self.mine = (int(je[j0]), int(ie[i0]),
                         int(je[j0 + ny_loc] - je[j0]),
                         int(ie[i0 + nx_loc] - ie[i0]))
        else:
            self.mine = (j0, i0, ny_loc, nx_loc)
        # both groups learn the other's block extents: allgather inside each
        # group, one small exchange between the two rank 0s, broadcast back
        local = comm.allgather(self.mine)
        if comm.Get_rank() == 0:
            req = intercomm.isend(local, dest=0, tag=71)
            remote = intercomm.recv(source=0, tag=71)
            req.wait()
        else:
            remote = None
        self.remote = comm.bcast(remote, root=0)
        # overlaps of MY block with the remote blocks. c = 1: local slices,
        # as before. c > 1: the overlap in fine indices, and from it
        #   wave   the slice of my coarse block that covers it
        #   ocean  my fine slice, the index of each of its cells in the
        #          covering coarse rectangle (rj, ri) and the reduceat
        #          offsets of the groups (oj, oi)
        fj0, fi0, fny, fnx = self.mine
        self.ov = []
        for r, (J0, I0, NY, NX) in enumerate(self.remote):
            a0, a1 = max(fj0, J0), min(fj0 + fny, J0 + NY)
            b0, b1 = max(fi0, I0), min(fi0 + fnx, I0 + NX)
            if not (a1 > a0 and b1 > b0):
                continue
            if self.cmap is None:
                self.ov.append((r, slice(a0 - j0, a1 - j0),
                                   slice(b0 - i0, b1 - i0), a1 - a0, b1 - b0))
                continue
            cj, ci = self.cmap.cj, self.cmap.ci
            cj0, cj1 = int(cj[a0]), int(cj[a1 - 1]) + 1
            ci0, ci1 = int(ci[b0]), int(ci[b1 - 1]) + 1
            if role == 'wave':
                self.ov.append((r, slice(cj0 - j0, cj1 - j0),
                                   slice(ci0 - i0, ci1 - i0),
                                cj1 - cj0, ci1 - ci0))
            else:
                self.ov.append((r, slice(a0 - j0, a1 - j0),
                                   slice(b0 - i0, b1 - i0),
                                cj1 - cj0, ci1 - ci0,
                                cj[a0:a1] - cj0, ci[b0:b1] - ci0,
                                _group_offsets(cj[a0:a1]),
                                _group_offsets(ci[b0:b1])))
        self._sends = []

    # ------------------------------------------------------------------
    def send_blocks(self, a, tag):
        """Post the pieces of this rank's block `a` (..., ny_loc, nx_loc) to
        the remote ranks that own them. Non-blocking: the buffers are held
        until wait_sends(). With a coarser wave grid this is the WAVE side's
        send (its coarse cells as they are); the ocean uses send_sums."""
        if self.cmap is not None and self.role != 'wave':
            raise RuntimeError('send_blocks: the ocean sends with send_sums '
                               'when the wave grid is coarser')
        a = np.asarray(a)
        for o in self.ov:
            buf = np.ascontiguousarray(a[..., o[1], o[2]], self.prec)
            self._sends.append((self.inter.Isend([buf, self.type], dest=o[0],
                                                 tag=tag), buf))
        return

    def send_sums(self, a, w, tag):
        """OCEAN side, coarser wave grid: post sum(w*a) over this rank's part
        of every coarse cell. `w` must broadcast against `a`. a = None sends
        sum(w) itself, the static denominator."""
        w = np.asarray(w, self.prec)
        wa = w if a is None else w*np.asarray(a)
        for o in self.ov:
            s = np.add.reduceat(np.add.reduceat(wa[..., o[1], o[2]], o[7],
                                                axis=-2), o[8], axis=-1)
            buf = np.ascontiguousarray(s, self.prec)
            self._sends.append((self.inter.Isend([buf, self.type], dest=o[0],
                                                 tag=tag), buf))
        return

    def post_recv(self, lead_shape, tag):
        """Post the receives for this rank's block and return a handle. POST
        EVERY RECEIVE BEFORE ANY SEND, on both sides: then no ordering of the
        two components can deadlock, whatever the MPI protocol does. It
        matters. A large message goes through rendezvous, so an Isend does
        not complete until the matching receive exists -- and the first
        version of this exchange had each side wait for its own sends before
        the other side had posted anything, which hung the 1/3 degree coupled
        run at the first coupling step. Small messages are sent eagerly and
        complete anyway, so a small test cannot find it."""
        lead = tuple(int(v) for v in lead_shape)
        reqs, parts = [], []
        for o in self.ov:
            buf = np.empty(lead + (o[3], o[4]), self.prec)
            reqs.append(self.inter.Irecv([buf, self.type], source=o[0], tag=tag))
            parts.append((o, buf))
        return (lead, reqs, parts)

    def complete_recv(self, handle):
        """Wait for a posted receive and assemble this rank's block. On the
        ocean side of a coarser wave grid every coarse value is copied into
        the ocean cells of its group."""
        if self.cmap is not None and self.role == 'wave':
            raise RuntimeError('complete_recv: the wave side receives sums, '
                               'use complete_recv_sums')
        lead, reqs, parts = handle
        self._waitall(reqs, parts)
        out = np.empty(lead + self.shape, self.prec)
        for o, buf in parts:
            if self.cmap is None:
                out[..., o[1], o[2]] = buf
            else:
                out[..., o[1], o[2]] = buf[..., o[5], :][..., o[6]]
        return out

    def complete_recv_sums(self, handle):
        """WAVE side, coarser wave grid: wait and ADD the parts the ocean PEs
        sent for each coarse cell (a cell can straddle several ocean blocks).
        Returns the sums on this rank's block; dividing by the static sum of
        the weights gives the mean."""
        lead, reqs, parts = handle
        self._waitall(reqs, parts)
        out = np.zeros(lead + self.shape, self.prec)
        for o, buf in parts:
            out[..., o[1], o[2]] += buf
        return out

    def _waitall(self, reqs, parts):
        """Wait, and check that every message had the size of its buffer:
        a size mismatch between a send and its receive corrupts the heap
        silently (MPI writes past the buffer) and is found here instead."""
        st = [MPI.Status() for _ in reqs]
        MPI.Request.Waitall(reqs, st)
        for (o, buf), s in zip(parts, st):
            n = s.Get_count(self.type)
            if n != buf.size and not (s.Get_source() == -1 and s.Get_tag() == -1):
                raise RuntimeError('BlockExchange: received %d values for a buffer of %d '
                                   '(from remote rank %d, tag %d, shape %s)'
                                   % (n, buf.size, s.Get_source(), s.Get_tag(), buf.shape))
        return

    def recv_blocks(self, lead_shape, tag):
        """post_recv + complete_recv in one call. Only safe where nothing of
        this rank's own sending is still outstanding -- prefer the two-step
        form in a coupled loop."""
        return self.complete_recv(self.post_recv(lead_shape, tag))

    def wait_sends(self):
        """Complete the posted sends and release their buffers. Must be called
        once per coupling step, before the tags are reused."""
        if self._sends:
            MPI.Request.Waitall([r for r, _ in self._sends])
            self._sends = []
        return


def block_exchange(M, comm, intercomm, cmap=None, role=None, key='wave'):
    """The component's BlockExchange with one partner, built on first use
    and cached under `key` (a component can have two partners: the ocean
    talks to the waves and to the atmosphere)."""
    if not hasattr(M, '_cpl_bx'):
        M._cpl_bx = {}
    if key not in M._cpl_bx:
        M._cpl_bx[key] = BlockExchange(M, comm, intercomm, cmap, role)
        bx = M._cpl_bx[key]
        if comm.Get_rank() == 0:
            print('block exchange [%s]: %d local blocks vs %d remote, this rank '
                  'overlaps %d of them%s' % (key, comm.Get_size(),
                  len(bx.remote), len(bx.ov),
                  '' if bx.cmap is None else
                  '; coarse grid = fine grid coarsened by %d' % bx.cmap.c), flush=True)
    return M._cpl_bx[key]
