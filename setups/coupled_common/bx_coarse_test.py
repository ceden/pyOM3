"""
Test of the block-to-block exchange with a COARSER wave grid
(coupler_mpi.BlockExchange with a CoarseMap). No model is built: two fake
components with ragged decompositions exchange analytic fields and every rank
compares what it received with the serial answer.

    mpirun -np 13 python3 bx_coarse_test.py [c] [ny] [nx]      (default 3 53 60)

The first 6 ranks (3 x 2) are the wave group, the rest the ocean group, laid
out as close to square as the count allows. ny and nx are the OCEAN grid;
choose them NOT divisible by c and by the block counts, that is the case
worth testing. Checked, on every rank:

  wave -> ocean   each ocean cell holds the value of the wave cell it lies
                  in, exactly (it is a copy)
  ocean -> wave   sum(w a)/sum(w) over the wet ocean cells of each wave cell,
                  per level with a level-dependent mask, against the serial
                  sum (round-off: the parts are added in a different order)
  c = 1           both directions are plain copies, exactly

As bx_test.py, this checks CORRECTNESS. The messages are small and go eagerly,
so it cannot reproduce a rendezvous deadlock; the receive-before-send order
of the coupled loop is what protects against that.
"""
import sys
import numpy as np
from mpi4py import MPI
from coupler_mpi import split_world, BlockExchange, CoarseMap


class FakeModel:
    """The four attributes pe_block reads, for an even-split decomposition."""
    halo = 1

    def __init__(self, ny, nx, npy, npx, rank):
        def rng(n, p, b):
            q, r = divmod(n, p)
            s = b*q + min(b, r)
            return s, q + (1 if b < r else 0)
        by, bx = divmod(rank, npx)
        j0, nyl = rng(ny, npy, by)
        i0, nxl = rng(nx, npx, bx)
        self.ys_pe, self.xs_pe = j0 + 1, i0 + 1
        self.Ny, self.Nx = nyl + 2, nxl + 2


def near_square(n):
    a = int(np.sqrt(n))
    while n % a:
        a -= 1
    return n//a, a


def main():
    c  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 53
    nx = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    nz = 4
    size = MPI.COMM_WORLD.Get_size()
    n_wave = 6
    npx_w, npy_w = 3, 2
    n_ocean = size - n_wave
    if n_ocean < 1:
        raise SystemExit('need more than %d ranks' % n_wave)
    npx_o, npy_o = near_square(n_ocean)
    world, my_pe, color, comm, inter, active = split_world(n_wave, n_ocean)
    cm = CoarseMap(ny, nx, c)

    # serial truth, the same on every rank
    rs = np.random.RandomState(7)
    fine = rs.standard_normal((nz, ny, nx))               # ocean field
    mask = (rs.uniform(size=(nz, ny, nx)) > 0.3).astype(float)
    mask[:, :3*c, :2*c] = 0.0                             # some all-dry groups
    area = (1.0 + 0.3*np.cos(np.linspace(-1.3, 1.2, ny)))[None, :, None] \
           * np.ones((1, 1, nx))
    w = area*mask
    coarse = rs.standard_normal((2, cm.ny_c, cm.nx_c))    # wave field
    red = lambda a: np.add.reduceat(np.add.reduceat(a, cm.je[:-1], axis=-2),
                                    cm.ie[:-1], axis=-1)
    num_ref, den_ref = red(w*fine), red(w)

    err = 0.0
    if color == 0:                                        # ---- wave ----
        M = FakeModel(cm.ny_c, cm.nx_c, npy_w, npx_w, comm.Get_rank())
        bx = BlockExchange(M, comm, inter, cm, 'wave')
        j0, i0 = M.ys_pe - 1, M.xs_pe - 1
        nyl, nxl = bx.shape
        sl = (slice(j0, j0 + nyl), slice(i0, i0 + nxl))
        if c > 1:
            hD = bx.post_recv((nz,), 11)
            den = bx.complete_recv_sums(hD)
            err = max(err, np.abs(den - den_ref[(slice(None),) + sl]).max())
        hU = bx.post_recv((nz,), 1)
        bx.send_blocks(coarse[(slice(None),) + sl], 3)
        if c > 1:
            num = bx.complete_recv_sums(hU)
            got = np.where(den > 0, num/np.where(den > 0, den, 1.0), 0.0)
            ref = np.where(den_ref > 0, num_ref/np.where(den_ref > 0, den_ref, 1.0),
                           0.0)[(slice(None),) + sl]
            err = max(err, np.abs(got - ref).max())
        else:
            got = bx.complete_recv(hU)
            err = max(err, np.abs(got - fine[(slice(None),) + sl]).max())
        bx.wait_sends()
    elif color == 1:                                      # ---- ocean ----
        M = FakeModel(ny, nx, npy_o, npx_o, comm.Get_rank())
        bx = BlockExchange(M, comm, inter, cm, 'ocean')
        j0, i0 = M.ys_pe - 1, M.xs_pe - 1
        nyl, nxl = bx.shape
        sl = (slice(None), slice(j0, j0 + nyl), slice(i0, i0 + nxl))
        if c > 1:
            bx.send_sums(None, w[sl], 11)
            bx.wait_sends()
        hS = bx.post_recv((2,), 3)
        if c > 1:
            bx.send_sums(fine[sl], w[sl], 1)
        else:
            bx.send_blocks(fine[sl], 1)
        got = bx.complete_recv(hS)
        bx.wait_sends()
        ref = coarse[:, cm.cj, :][:, :, cm.ci][sl]
        err = max(err, np.abs(got - ref).max())           # a copy: exactly 0

    errs = world.gather((color, err), root=0)
    if my_pe == 0:
        ew = max(e for k, e in errs if k == 0)
        eo = max(e for k, e in errs if k == 1)
        print('c=%d, ocean %d x %d on %d x %d PEs, wave %d x %d on %d x %d PEs'
              % (c, nx, ny, npx_o, npy_o, cm.nx_c, cm.ny_c, npx_w, npy_w))
        print('  ocean -> wave  max error %.2e   (%s)'
              % (ew, 'round-off' if c > 1 else 'copy, must be 0'))
        print('  wave -> ocean  max error %.2e   (copy, must be 0)' % eo)
        ok = eo == 0.0 and (ew < 1e-13 if c > 1 else ew == 0.0)
        print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
