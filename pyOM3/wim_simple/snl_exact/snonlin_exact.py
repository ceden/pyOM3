"""snonlin_exact.py -- exact WRT S_nl as a wim_simple Snonlin backend.

Drop-in replacement for pyOM3.wim_simple.Snonlin / snonlin_np / snonlin_jax,
wrapping the exact four-wave scattering integral (kernel.py + locus.py +
assembly.py).  Same contract as the other backends:

    Snonlin.setup(nx, ny, kl, fr)          # fr: 1-D interior frequency axis
    sl, fl = Snonlin.source(F)             # F: (nx, ny, kl, ml) = E(f,theta)

    sl : exact nonlinear transfer dE/dt        (nx, ny, kl, ml)
    fl : diagonal functional derivative dS/dE  (nx, ny, kl, ml)

so source.py's Snonlin() wrapper (halo unpadding, transposes) works
unchanged.  Wiring in main.setup():

    elif self.snonlin_backend == 'exact':
        from pyOM3.wim_simple.snl_exact.snonlin_exact import Snonlin \
            as Snonlin_backend

with the snl_exact/ folder placed inside wim_simple/ (an __init__.py is
included; any other sys.path location works too).

Cost & intended use
-------------------
This is a REFERENCE evaluator: cost is per spatial point and scales as
nf^2 * nd * npts table entries.  At WIM's production grid (nf=62, nd=80)
expect minutes per spectrum.  It is meant for offline truth evaluations
and for comparing a parameterisation against the exact transfer at
selected points of a running simulation -- not for full-grid stepping.
Guards:

    Snonlin.points = [(ix, iy), ...]   # evaluate only here (zeros + fl=0
                                       # elsewhere; the semi-implicit
                                       # update is then a no-op there)
    Snonlin.max_cells = 64             # refuse silently huge evaluations

Physics/consistency notes
-------------------------
* fr must be geometric (WIM's WAM grid is); checked at setup.
* `pad_hi` tail rings (parametric f^-5 continuation matched per direction
  at the last band ring) are attached before evaluation and cropped
  after, following WW3 WRT practice; interactions with the diagnostic
  tail are thereby retained.  The band<->tail action exchange of the last
  call is reported in Snonlin.diag.
* mode='conservative' closes the total action budget of sl on the
  extended grid to machine precision.
* fl is the role-1 diagonal derivative (WAM semantics); it is invariant
  under the E <-> n conversion, so it can be used directly in
  E += S dt / (1 - dt fl).  In mode='conservative' it is the derivative of
  the DIRECT rate at k1, not of the returned (redistributed) transfer --
  about a factor 2 apart at and above the peak. It only serves as the
  semi-implicit factor.
* With pad_hi > 0 an equilibrium band spectrum (constant A) gives a
  non-zero band transfer through the tail interactions (order 0.2 of the
  gross rate); the operator itself (pad_hi = 0) keeps the equilibrium to
  machine precision.
* g: set Snonlin.g before setup(); wim_simple.main hands over its grav.
"""

import os
import numpy as np

from pyOM3.wim_simple.snl_exact.assembly import PolarGrid, build_table, snl
#try:                                    # package placement inside wim_simple
#    from pyOM3.wim_simple.snl_exact.assembly import PolarGrid, build_table, snl
#except ImportError:                     # plain sys.path placement
#    from assembly import PolarGrid, build_table, snl


class _SnonlinExact:
    # tunables (set before setup(), or between setup() and source())
    g = 9.81
    npts = 24
    npts_line = None
    pad_hi = 8
    dfilt = 0.025
    cache = None                        # e.g. "snl_table.npz"
    mode = 'conservative'
    points = None                       # None: all (guarded by max_cells)
    max_cells = 64
    verbose = True

    def setup(self, nx, ny, kl, fr):
        fr = np.asarray(fr, float)
        r = fr[1:] / fr[:-1]
        if not np.allclose(r, r[0], rtol=1e-8):
            raise ValueError("snonlin_exact: frequency axis must be "
                             "geometric (WAM grid)")
        self.nx, self.ny, self.kl, self.ml = nx, ny, kl, len(fr)
        ratio = float(r[0])
        nf_ext = self.ml + self.pad_hi
        self.grid = PolarGrid(f0=fr[0], ratio=ratio, nf=nf_ext, nd=kl,
                              g=self.g)
        om = 2 * np.pi * self.grid.f
        dkdf = 8 * np.pi ** 2 * self.grid.f / self.g
        self._M = (om * self.grid.k * dkdf / self.g)[:, None]  # E = M n
        self.diag = {}

        meta = np.array([fr[0], ratio, nf_ext, kl, self.g, self.npts,
                         self.npts_line or 4 * self.npts, self.dfilt])
        self.table = None
        if self.cache and os.path.exists(self.cache):
            z = np.load(self.cache)
            if 'meta' in z.files and np.allclose(z['meta'], meta):
                self.table = {k: z[k] for k in z.files if k != 'meta'}
                self.table['npts'] = self.npts
                if self.verbose:
                    print(f"snonlin_exact: table loaded from {self.cache}")
        if self.table is None:
            if self.verbose:
                print(f"snonlin_exact: building table nf={nf_ext} nd={kl} "
                      f"npts={self.npts} (one-off, cached)" )
            self.table = build_table(self.grid, npts=self.npts,
                                     npts_line=self.npts_line,
                                     triv_filter=self.dfilt, edge='strict',
                                     verbose=self.verbose)
            if self.cache:
                save = {k: v for k, v in self.table.items()
                        if isinstance(v, np.ndarray)}
                save['meta'] = meta
                np.savez_compressed(self.cache, **save)
        return self

    # -----------------------------------------------------------------
    def _pad(self, E):
        if self.pad_hi == 0:
            return E
        f = self.grid.f
        tail = (f[self.ml:] / f[self.ml - 1])[:, None] ** -5.0
        return np.concatenate([E, E[self.ml - 1][None, :] * tail], axis=0)

    def source(self, F):
        """F: (nx, ny, kl, ml) E(f,theta).  Returns (sl, fl), same shape."""
        F = np.asarray(F, float)
        nx, ny, kl, ml = F.shape
        assert (kl, ml) == (self.kl, self.ml), (F.shape, self.kl, self.ml)
        if self.points is None:
            pts = [(i, j) for i in range(nx) for j in range(ny)]
        else:
            pts = list(self.points)
        if len(pts) > self.max_cells:
            raise RuntimeError(
                f"snonlin_exact: {len(pts)} spectra requested "
                f"(max_cells={self.max_cells}). This is an offline "
                "reference evaluator; set Snonlin.points to selected "
                "cells, or raise Snonlin.max_cells knowingly.")
        SL = np.zeros_like(F)
        FL = np.zeros_like(F)
        band = float('nan')
        for (i, j) in pts:
            E = F[i, j].T                       # -> (ml, kl) = (nf, nd)
            n = self._pad(E) / self._M
            S_n, diag = snl(n, self.grid, self.table, mode=self.mode,
                            deriv=True)
            S_E = (S_n * self._M)[:self.ml]
            SL[i, j] = S_E.T
            FL[i, j] = diag['dSdn'][:self.ml].T
            act = self.grid.area[:, None] * S_n
            band = float(np.sum(act[:self.ml]))
            self.diag = dict(diag, band_action_rate=band,
                             tail_action_rate=float(np.sum(act[self.ml:])))
            self.diag.pop('dSdn', None)
        return SL, FL


Snonlin = _SnonlinExact()
