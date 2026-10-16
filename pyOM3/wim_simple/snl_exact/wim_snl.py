"""Adapter: exact WRT S_nl for WIM (offline reference / research source term).

Usage
-----
    from wim_snl import SnlExact
    snl = SnlExact(f=wim.f, nd=wim.nd, g=wim.grav, cache="snl_table.npz")
    S_E = snl.tendency_E(E)      # E(f,theta) [m^2/Hz/rad]  -> dE/dt
    S_A = snl.tendency_A(A)      # A(f,theta) = E/omega     -> dA/dt

Requirements and caveats
------------------------
* f must be geometric (WIM's df0-ratio grid; the `uniform_f` diagnostic
  switch must be OFF).  theta is assumed uniform over 2 pi with nd bins.
* Cost: the interaction table scales as nf^2 * nd * npts entries.  At
  nf=62, nd=80 expect several million entries (~1 GB, build tens of
  minutes, evaluation minutes).  This is an offline truth evaluator; for
  in-loop use build it on a coarsened grid or wait for the fortran port.
* Tail: `pad_hi` extends the grid by geometric rings above the WIM band
  and fills them with a parametric E ~ f^-5 continuation matched per
  direction to the last band ring (WW3 WRT practice).  Interactions with
  the tail then contribute to the band; the tail rings' own tendency is
  discarded on crop.  `pad_hi=0` reproduces plain strict truncation.
* Conservation: mode='conservative' makes the total-action budget of the
  returned tendency close to machine precision on the *extended* grid;
  after cropping, the band budget differs by the (physical) action
  exchanged with the tail, reported in the `diag` attribute.
"""

import os
import numpy as np
from assembly import PolarGrid, build_table, snl


class SnlExact:
    def __init__(self, f, nd, g=9.81, npts=24, npts_line=None, pad_hi=8,
                 dfilt=0.025, cache=None, verbose=True):
        f = np.asarray(f, float)
        r = f[1:] / f[:-1]
        if not np.allclose(r, r[0], rtol=1e-8):
            raise ValueError("WIM frequency grid must be geometric "
                             "(disable uniform_f)")
        self.ratio = float(r[0])
        self.nf_band = len(f)
        self.pad_hi = int(pad_hi)
        f_ext = f[0] * self.ratio ** np.arange(self.nf_band + self.pad_hi)
        self.grid = PolarGrid(f0=f[0], ratio=self.ratio, nf=len(f_ext),
                              nd=nd, g=g)
        self.diag = {}

        meta = dict(f0=f[0], ratio=self.ratio, nf=len(f_ext), nd=nd, g=g,
                    npts=npts, npts_line=npts_line or 4 * npts,
                    dfilt=dfilt, edge="strict")
        self.table = None
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            if all(np.isclose(z[f"meta_{k}"], v) if isinstance(v, float)
                   else z[f"meta_{k}"] == v for k, v in meta.items()
                   if not isinstance(v, str)):
                self.table = {k: z[k] for k in z.files
                              if not k.startswith("meta_")}
                self.table["npts"] = npts
                if verbose:
                    print(f"SnlExact: table loaded from {cache} "
                          f"({len(self.table['base'])} entries)")
        if self.table is None:
            if verbose:
                print(f"SnlExact: building table nf={len(f_ext)} nd={nd} "
                      f"npts={npts} ...")
            self.table = build_table(self.grid, npts=npts,
                                     npts_line=npts_line,
                                     triv_filter=dfilt, edge="strict",
                                     verbose=verbose)
            if cache:
                save = {k: v for k, v in self.table.items()
                        if isinstance(v, np.ndarray)}
                for k, v in meta.items():
                    if not isinstance(v, str):
                        save[f"meta_{k}"] = np.asarray(v)
                np.savez_compressed(cache, **save)
                if verbose:
                    print(f"SnlExact: table cached to {cache}")

    # ---------------------------------------------------------- padding
    def _pad(self, E):
        if self.pad_hi == 0:
            return np.asarray(E, float)
        nb = self.nf_band
        f = self.grid.f
        tail = (f[nb:] / f[nb - 1])[:, None] ** -5.0
        return np.concatenate([E, E[nb - 1][None, :] * tail], axis=0)

    # ------------------------------------------------------- tendencies
    def tendency_E(self, E, mode="conservative"):
        """dE/dt (f,theta) on the WIM band from E(f,theta) [m^2/Hz/rad]."""
        g, grid = self.grid.g, self.grid
        Ex = self._pad(E)
        om = 2 * np.pi * grid.f[:, None]
        dkdf = 8 * np.pi ** 2 * grid.f[:, None] / g
        n = g * Ex / (om * grid.k[:, None] * dkdf)
        S_n, diag = snl(n, grid, self.table, mode=mode)
        S_E = S_n * om * grid.k[:, None] * dkdf / g
        nb = self.nf_band
        # band action budget vs tail exchange (diagnostics)
        act = grid.area[:, None] * S_n
        self.diag = dict(diag, band_action_rate=float(np.sum(act[:nb])),
                         tail_action_rate=float(np.sum(act[nb:])))
        return S_E[:nb]

    def tendency_A(self, A, mode="conservative"):
        """dA/dt from action density A(f,theta) = E/omega [m^2/Hz/rad *s]."""
        om_band = 2 * np.pi * self.grid.f[:self.nf_band, None]
        return self.tendency_E(A * om_band, mode=mode) / om_band
