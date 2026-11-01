#!/usr/bin/env python3
"""
ic_blocks.py -- cut initial_conditions.nc into ONE CONTIGUOUS FILE PER SHARD
for a multi-process jax run.

    python3 ic_blocks.py initial_conditions.nc NPX NPY [--fields temp,salt] [--out DIR]

writes DIR/<field>_<shard>.npy (default DIR = ic_blocks_<NPX>x<NPY>), each
(nz, ny/NPY, nx/NPX) float32, shard = sy*NPX + sx numbered as block_io.shard_id
(row major in the shard grid, sy the block row from the south). The file is
read ONCE here, level by level (one contiguous record per read), and each
process of the run then reads nothing but its own ~350 MB file (1/10 degree,
24 shards) in one go -- instead of 24 processes reading the full 8.5 GB
field each through the netCDF chunking, which took the first icon01 job
longer than 20 minutes on a loaded Lustre.

NPX, NPY must divide nx, ny exactly (the shard layout of run_gpu.sh). The
blocks hold the file's values as they are; the setup module does the dry-cell
replacement when it reads them, as it does for the netCDF path.
"""
import os, sys, argparse
import numpy as np
from netCDF4 import Dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('icfile'); ap.add_argument('npx', type=int); ap.add_argument('npy', type=int)
    ap.add_argument('--fields', default='temp,salt')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or 'ic_blocks_%dx%d' % (a.npx, a.npy)
    fields = a.fields.split(',')
    with Dataset(a.icfile) as f:
        f.set_auto_mask(False)
        nz, ny, nx = f.variables[fields[0]].shape[-3:]
        if ny % a.npy or nx % a.npx:
            sys.exit('%d x %d shards do not divide the grid (ny %d, nx %d)' % (a.npx, a.npy, ny, nx))
        nyl, nxl = ny//a.npy, nx//a.npx
        os.makedirs(out, exist_ok=True)
        for v in fields:
            var = f.variables[v]
            mm = [np.lib.format.open_memmap(os.path.join(out, '%s_%04d.npy' % (v, sy*a.npx + sx)),
                                            mode='w+', dtype=np.float32, shape=(nz, nyl, nxl))
                  for sy in range(a.npy) for sx in range(a.npx)]
            for k in range(nz):
                rec = np.asarray(var[(k,) if var.ndim == 3 else (0, k)], np.float32)   # one level, contiguous
                for sy in range(a.npy):
                    for sx in range(a.npx):
                        mm[sy*a.npx + sx][k] = rec[sy*nyl:(sy+1)*nyl, sx*nxl:(sx+1)*nxl]
                if k % 16 == 0:
                    print('%s: level %d / %d' % (v, k, nz), flush=True)
            for m in mm:
                m.flush()
            del mm
    print('wrote %d blocks per field, (%d, %d, %d) float32 each, to %s/' % (a.npx*a.npy, nz, nyl, nxl, out))


if __name__ == '__main__':
    main()
