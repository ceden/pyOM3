"""
snonlin_jax.py
--------------
The DIA of snonlin_np.py in the fully vectorised form for the JAX backend
(no scatter, see source() and _vec_tables()). The setup tables are those
of snonlin_np.py. Below the original module docstring of the loop form.

Vectorized NumPy translation of the WAM discrete-interaction-approximation
(DIA) for the nonlinear 4-wave transfer, i.e. a drop-in replacement for the
compiled Fortran extension pyOM3.wim_simple.Snonlin (Snonlin.f90).

Public interface (matches the f2py extension used by source.py):

    Snonlin.setup(nx, ny, kl, fr)      # fr is the 1-D frequency axis (length ml)
    sl, fl = Snonlin.source(F)         # F shape (nx, ny, kl, ml)

    sl : nonlinear transfer source term  S_nl              (nx, ny, kl, ml)
    fl : diagonal of the functional derivative dS_nl/dF    (nx, ny, kl, ml)

Vectorization
-------------
The original Fortran vectorizes the two spatial axes (nx, ny) via array-section
assignment and loops over the centre frequency MC, the two mirror halves KH,
and the direction K. This translation keeps the small MC and KH loops (length
~ML and 2) but additionally vectorizes over the direction axis K using
fancy-indexed scatter-add (np.add.at). The spatial axes stay vectorized exactly
as in the Fortran. Net: only O(ML) python-level iterations regardless of nx, ny,
kl.

Index conventions
-----------------
Fortran is 1-based and several weight tables are dimensioned MFRSTLW:MLSTHG with
MFRSTLW possibly <= 0. Here everything is 0-based. Frequency-grid quantities that
the Fortran indexes by the (possibly negative) "long" frequency index M are
stored in arrays offset by -MFRSTLW, accessed through self._lo. The precomputed
per-MC bundles (INLCOEF, RNLCOEF, AF11, ...) are stored 0-based over MC = 0 ..
MLSTHG-1.

Correctness notes
-----------------
* JAFU, NLWEIGT and INIT_SNONLIN are setup-time table builders; they are small
  (1-D over directions / frequencies) and kept as explicit loops for a faithful,
  readable port.
* In source(), the destination frequency indices (IC, IP, IP1, IM, IM1) and the
  per-MC tail clamping are exactly those precomputed by INIT_SNONLIN, so the
  three Fortran branches (interior / high-frequency / low-frequency) collapse to
  a single uniform scatter once out-of-range destinations are masked. We follow
  the Fortran's clamped destination indices and skip contributions whose
  destination frequency falls outside 1..ML (the .LE.ML / .GE.1 guards).
"""

import numpy as np            # host-side table construction (setup runs once)
import pyOM3 as OM            # OM.np is numpy or jax.numpy; OM.add_to_array is
                             # the backend-agnostic scatter-add (.at[].add for jax)


def _jafu(cl, j, ian):
    """WAM JAFU: angular index wrap. j is 1-based, returns 1-based index."""
    v = j + int(cl)
    if v <= 0:
        v += ian
    if v > ian:
        v -= ian
    return v


class _Snonlin:

    # ----------------------------------------------------------------- setup
    def setup(self, nx, ny, kl, fr):
        self.nx = int(nx)
        self.ny = int(ny)
        self.kl = int(kl)          # number of directions KL
        self.fr = np.asarray(fr, dtype=float).copy()
        self.ml = self.fr.size     # number of frequencies ML
        self.co = self.fr[1] / self.fr[0]
        # WAM sets DELTH = ZPI/KL in MAKE_FRE_DIR, with ZPI = 2*PI and the
        # truncated PI = 3.1415927 used throughout wam_general_module. NLWEIGT
        # then forms DELTHA = DELTH*DEG with DEG = 180/PI (same truncated PI),
        # so the PI cancels exactly and DELTHA = 360/KL. We use the SAME
        # truncated PI here so the cancellation is identical to the Fortran;
        # using np.pi for DELTH but truncated PI for DEG would leave a ~1.5e-8
        # mismatch that can break tight validation tolerances.
        _PI = 3.1415927
        self.delth = 2.0 * _PI / self.kl
        self._nlweigt()
        self._init_snonlin()
        self._vec_tables()               # the vectorised tables (host numpy, outside any trace)
        return None

    # ------------------------------------------------------------- nlweigt
    def _nlweigt(self):
        ALAMD = 0.25
        CON = 3000.0
        PI = 3.1415927
        DEG = 180.0 / PI
        ml, kl, co = self.ml, self.kl, self.co

        F1P1 = np.log10(co)
        ISP = int(np.log10(1.0 + ALAMD) / F1P1 + 1e-6)
        ISM = int(np.floor(np.log10(1.0 - ALAMD) / F1P1 + 1e-7))
        self.mfrstlw = 1 + ISM            # first long (low) freq index (<=0 typically)
        self.mlsthg = ml - ISM            # last long (high) freq index
        self.kfrh = -ISM + ISP + 2
        self._isp, self._ism = ISP, ISM

        # ---- 1. angular grid -------------------------------------------------
        XF = ((1.0 + ALAMD) / (1.0 - ALAMD)) ** 4
        COSTH3 = (1.0 + 2.0 * ALAMD + 2.0 * ALAMD ** 3) / (1.0 + ALAMD) ** 2
        DELPHI1 = -180.0 / PI * np.arccos(COSTH3)
        COSTH4 = np.sqrt(1.0 - XF + XF * COSTH3 ** 2)
        DELPHI2 = 180.0 / PI * np.arccos(COSTH4)
        DELTHA = self.delth * DEG
        CL1 = DELPHI1 / DELTHA
        CL2 = DELPHI2 / DELTHA

        KLP1 = kl + 1
        # JA1, JA2 are (KL,2) 1-based; store 0-based rows but keep K 1-based logic
        JA1 = np.zeros((kl + 1, 2), dtype=int)   # index 1..KL used
        JA2 = np.zeros((kl + 1, 2), dtype=int)
        IC = 1
        for KH in (1, 2):
            KLH = kl if KH == 1 else KLP1
            for K in range(1, KLH + 1):
                KS = K if KH == 1 else KLP1 - K + 1
                if KS > kl:
                    continue
                JA1[KS, KH - 1] = _jafu(IC * CL1, K, kl)
                JA2[KS, KH - 1] = _jafu(IC * CL2, K, kl)
            IC = -1

        # angular weights
        CL1 = CL1 - int(CL1)
        CL2 = CL2 - int(CL2)
        self.acl1 = abs(CL1)
        self.acl2 = abs(CL2)
        self.cl11 = 1.0 - self.acl1
        self.cl21 = 1.0 - self.acl2
        AL11 = (1.0 + ALAMD) ** 4
        AL12 = (1.0 - ALAMD) ** 4
        self.dal1 = 1.0 / AL11
        self.dal2 = 1.0 / AL12

        # angular index tables K1W,K2W,K11W,K21W : (KL,2), values are 1-based dirs
        K1W = np.zeros((kl + 1, 2), dtype=int)
        K2W = np.zeros((kl + 1, 2), dtype=int)
        K11W = np.zeros((kl + 1, 2), dtype=int)
        K21W = np.zeros((kl + 1, 2), dtype=int)
        ISG = 1
        for KH in (1, 2):
            CL1H = ISG * CL1
            CL2H = ISG * CL2
            for K in range(1, kl + 1):
                if KH == 2:
                    KS = kl - K + 2
                else:
                    KS = K
                if K == 1:
                    KS = 1
                K1 = JA1[K, KH - 1]
                K1W[KS, KH - 1] = K1
                if CL1H < 0.0:
                    K11 = K1 - 1
                    if K11 < 1:
                        K11 = kl
                else:
                    K11 = K1 + 1
                    if K11 > kl:
                        K11 = 1
                K11W[KS, KH - 1] = K11
                K2 = JA2[K, KH - 1]
                K2W[KS, KH - 1] = K2
                if CL2H < 0.0:
                    K21 = K2 - 1
                    if K21 < 1:
                        K21 = kl
                else:
                    K21 = K2 + 1
                    if K21 > kl:
                        K21 = 1
                K21W[KS, KH - 1] = K21
            ISG = -1
        # store 0-based direction indices (subtract 1), shape (KL,2)
        self.k1w = K1W[1:kl + 1, :] - 1
        self.k2w = K2W[1:kl + 1, :] - 1
        self.k11w = K11W[1:kl + 1, :] - 1
        self.k21w = K21W[1:kl + 1, :] - 1

        # ---- 2. frequency grid ----------------------------------------------
        # FRLON indexed MFRSTLW .. ML+KFRH ; store with offset _lo = -MFRSTLW
        self._lo = -self.mfrstlw                # so FRLON[M] -> frlon[M+_lo]
        lo = self._lo
        n_frlon = (ml + self.kfrh) - self.mfrstlw + 1
        frlon = np.zeros(n_frlon)
        for M in range(1, ml + 1):
            frlon[M + lo] = self.fr[M - 1]
        for M in range(0, self.mfrstlw - 1, -1):
            frlon[M + lo] = frlon[M + 1 + lo] / co
        for M in range(ml + 1, ml + self.kfrh + 1):
            frlon[M + lo] = co * frlon[M - 1 + lo]
        self._frlon = frlon

        # weight tables over M = MFRSTLW .. MLSTHG ; offset by _lo as well
        nM = self.mlsthg - self.mfrstlw + 1
        self.af11 = np.zeros(nM)
        self.ikp = np.zeros(nM, dtype=int)
        self.ikp1 = np.zeros(nM, dtype=int)
        self.ikm = np.zeros(nM, dtype=int)
        self.ikm1 = np.zeros(nM, dtype=int)
        self.fklap = np.zeros(nM)
        self.fklap1 = np.zeros(nM)
        self.fklam = np.zeros(nM)
        self.fklam1 = np.zeros(nM)

        def Midx(M):           # map long freq index M to 0-based table row
            return M - self.mfrstlw

        for M in range(self.mfrstlw, self.mlsthg + 1):
            r = Midx(M)
            FRG = frlon[M + lo]
            self.af11[r] = CON * FRG ** 11
            FLP = FRG * (1.0 + ALAMD)
            FLM = FRG * (1.0 - ALAMD)
            IKN = M + ISP
            self.ikp[r] = IKN
            FKP = frlon[IKN + lo]
            self.ikp1[r] = IKN + 1
            self.fklap[r] = (FLP - FKP) / (frlon[IKN + 1 + lo] - FKP)
            self.fklap1[r] = 1.0 - self.fklap[r]
            if frlon[self.mfrstlw + lo] >= FLM:
                self.ikm[r] = 1
                self.ikm1[r] = 1
                self.fklam[r] = 0.0
                self.fklam1[r] = 0.0
            else:
                IKN = M + ISM
                self.ikm[r] = IKN
                FKM = frlon[IKN + lo]
                self.ikm1[r] = IKN + 1
                self.fklam[r] = (FLM - FKM) / (frlon[IKN + 1 + lo] - FKM)
                self.fklam1[r] = 1.0 - self.fklam[r]
                if IKN < self.mfrstlw:
                    self.ikm[r] = 1
                    self.fklam1[r] = 0.0

        # ---- 3. tail frequency ratios ---------------------------------------
        self.frh = np.zeros(self.kfrh)
        for I in range(1, self.kfrh + 1):
            M = ml + I - 1
            self.frh[I - 1] = (frlon[ml + lo] / frlon[M + lo]) ** 5

    # --------------------------------------------------------- init_snonlin
    def _init_snonlin(self):
        ml = self.ml
        co = self.co

        def epmma(x):
            return np.exp(-min(1.25 * x ** 4, 50.0)) * (x ** 5)

        # FTRF dimensioned MFRSTLW:1
        self._ftrf_lo = -self.mfrstlw          # FTRF[M] -> ftrf[M+_ftrf_lo], M in MFRSTLW..1
        ftrf = np.zeros(1 - self.mfrstlw + 1)
        ALPH = 1.0 / epmma(1.0)
        FRR = 1.0
        for MC in range(1, self.mfrstlw - 1, -1):
            ftrf[MC + self._ftrf_lo] = ALPH * epmma(FRR)
            FRR = FRR * co
        self._ftrf = ftrf

        def Midx(M):
            return M - self.mfrstlw

        nMC = self.mlsthg                       # MC = 1..MLSTHG -> rows 0..MLSTHG-1
        # destination frequency indices (clamped, 1-based as in Fortran)
        self.d_ic = np.zeros(nMC, dtype=int)
        self.d_ip = np.zeros(nMC, dtype=int)
        self.d_ip1 = np.zeros(nMC, dtype=int)
        self.d_im = np.zeros(nMC, dtype=int)
        self.d_im1 = np.zeros(nMC, dtype=int)
        # coefficient bundle
        self.c_ftail = np.zeros(nMC)
        self.c_gw = np.zeros((8, nMC))          # GW1..GW8
        self.c_fklampa = np.zeros(nMC); self.c_fklampb = np.zeros(nMC)
        self.c_fklamp2 = np.zeros(nMC); self.c_fklamp1 = np.zeros(nMC)
        self.c_fklapa2 = np.zeros(nMC); self.c_fklapb2 = np.zeros(nMC)
        self.c_fklap12 = np.zeros(nMC); self.c_fklap22 = np.zeros(nMC)
        self.c_fklamma = np.zeros(nMC); self.c_fklammb = np.zeros(nMC)
        self.c_fklamm2 = np.zeros(nMC); self.c_fklamm1 = np.zeros(nMC)
        self.c_fklama2 = np.zeros(nMC); self.c_fklamb2 = np.zeros(nMC)
        self.c_fklam12 = np.zeros(nMC); self.c_fklam22 = np.zeros(nMC)
        self.c_af11 = np.zeros(nMC)
        # destination tables MP,MP1,MM,MM1 (clamped, 1-based) and validity masks
        self.d_mp = np.zeros(nMC, dtype=int)
        self.d_mp1 = np.zeros(nMC, dtype=int)
        self.d_mm = np.zeros(nMC, dtype=int)
        self.d_mm1 = np.zeros(nMC, dtype=int)

        for MC in range(1, self.mlsthg + 1):
            r = MC - 1
            rM = Midx(MC)
            MP = self.ikp[rM]; MP1 = self.ikp1[rM]
            MM = self.ikm[rM]; MM1 = self.ikm1[rM]
            FFACP = FFACP1 = FFACM = FFACM1 = FTAIL = 1.0
            IC = MC; IP = MP; IP1 = MP1; IM = MM; IM1 = MM1

            # low frequency front tail
            if IM < 1:
                FFACM = self._ftrf[IM + self._ftrf_lo]
                IM = 1
                if IM1 < 1:
                    FFACM1 = self._ftrf[IM1 + self._ftrf_lo]
                    IM1 = 1
            # high frequency tail
            if IP1 > ml:
                ITEMP = IP1 - ml + 1
                if ITEMP > self.frh.size:
                    ITEMP = self.frh.size
                FFACP1 = self.frh[ITEMP - 1]
                IP1 = ml
                if IP > ml:
                    FFACP = self.frh[IP - ml + 1 - 1]
                    IP = ml
                    if IC > ml:
                        FTAIL = self.frh[IC - ml + 1 - 1]
                        IC = ml
                        if IM1 > ml:
                            FFACM1 = self.frh[IM1 - ml + 1 - 1]
                            IM1 = ml

            self.d_ic[r] = IC; self.d_ip[r] = IP; self.d_ip1[r] = IP1
            self.d_im[r] = IM; self.d_im1[r] = IM1
            self.d_mp[r] = MP; self.d_mp1[r] = MP1
            self.d_mm[r] = MM; self.d_mm1[r] = MM1

            FKLAMP = self.fklap[rM]; FKLAMP1 = self.fklap1[rM]
            GW2 = FKLAMP1 * FFACP * self.dal1
            GW1 = GW2 * self.cl11
            GW2 = GW2 * self.acl1
            GW4 = FKLAMP * FFACP1 * self.dal1
            GW3 = GW4 * self.cl11
            GW4 = GW4 * self.acl1
            FKLAMPA = FKLAMP * self.cl11
            FKLAMPB = FKLAMP * self.acl1
            FKLAMP2 = FKLAMP1 * self.acl1
            FKLAMP1 = FKLAMP1 * self.cl11
            FKLAPA2 = FKLAMPA ** 2
            FKLAPB2 = FKLAMPB ** 2
            FKLAP12 = FKLAMP1 ** 2
            FKLAP22 = FKLAMP2 ** 2

            FKLAMM = self.fklam[rM]; FKLAMM1 = self.fklam1[rM]
            GW6 = FKLAMM1 * FFACM * self.dal2
            GW5 = GW6 * self.cl21
            GW6 = GW6 * self.acl2
            GW8 = FKLAMM * FFACM1 * self.dal2
            GW7 = GW8 * self.cl21
            GW8 = GW8 * self.acl2
            FKLAMMA = FKLAMM * self.cl21
            FKLAMMB = FKLAMM * self.acl2
            FKLAMM2 = FKLAMM1 * self.acl2
            FKLAMM1 = FKLAMM1 * self.cl21
            FKLAMA2 = FKLAMMA ** 2
            FKLAMB2 = FKLAMMB ** 2
            FKLAM12 = FKLAMM1 ** 2
            FKLAM22 = FKLAMM2 ** 2

            self.c_ftail[r] = FTAIL
            self.c_gw[:, r] = (GW1, GW2, GW3, GW4, GW5, GW6, GW7, GW8)
            self.c_fklampa[r] = FKLAMPA; self.c_fklampb[r] = FKLAMPB
            self.c_fklamp2[r] = FKLAMP2; self.c_fklamp1[r] = FKLAMP1
            self.c_fklapa2[r] = FKLAPA2; self.c_fklapb2[r] = FKLAPB2
            self.c_fklap12[r] = FKLAP12; self.c_fklap22[r] = FKLAP22
            self.c_fklamma[r] = FKLAMMA; self.c_fklammb[r] = FKLAMMB
            self.c_fklamm2[r] = FKLAMM2; self.c_fklamm1[r] = FKLAMM1
            self.c_fklama2[r] = FKLAMA2; self.c_fklamb2[r] = FKLAMB2
            self.c_fklam12[r] = FKLAM12; self.c_fklam22[r] = FKLAM22
            self.c_af11[r] = self.af11[rM]

    # ------------------------------------------------------- vectorised tables
    def _vec_tables(self):
        """
        Tables of the vectorised source (built once, host numpy). The
        Fortran/numpy loop over MC (centre frequency) and KH (mirror half)
        becomes one group axis g = 2*(MC-1) + KH of length G = 2*MLSTHG:

          gather tables (G,KL) of directions and (G,1) of frequencies for
          the eight interpolation points F[K1,IP] ... F[K21,IM1] and the
          centre F[K,IC], the per-group coefficients GW1..8, AF11, the tail
          factor, and the DESTINATION of every contribution as a weight
          matrix W (G,ML) per (direction table, quantity): the contribution
          of group g to output frequency m is Q_perm[g] * W[g,m], where
          Q_perm is the quantity with the directions permuted into their
          destination bins (K -> K2[g,K] etc.). The three branches of the
          loop (interior / high / low frequency) are the validity masks
          folded into W (zero where the Fortran skips the term), and the
          accumulation over g is a matrix product instead of 2 x 9 scatters
          per group.
        """
        kl, ml = self.kl, self.ml
        self._vec_bad = None
        G = 2*self.mlsthg
        MFR1STFR = -self.mfrstlw + 1
        MFRLSTFR = ml - self.kfrh + MFR1STFR
        Kall = np.arange(kl)
        T = {}
        T['K1'] = np.zeros((G, kl), int); T['K11'] = np.zeros((G, kl), int)
        T['K2'] = np.zeros((G, kl), int); T['K21'] = np.zeros((G, kl), int)
        T['Kall'] = np.tile(Kall, (G, 1))
        for nm in ('IC', 'IP', 'IP1', 'IM', 'IM1'):
            T[nm] = np.zeros((G, 1), int)
        gw = np.zeros((8, G)); af = np.zeros(G); tail = np.ones(G)
        # weight matrices: (direction table, quantity) -> (G, ML)
        W = {k: np.zeros((G, ml)) for k in ('all_AD', 'all_DELAD', 'K2_AD', 'K21_AD', 'K1_AD',
                                            'K11_AD', 'K2_DELAM', 'K21_DELAM', 'K1_DELAP', 'K11_DELAP')}
        for MC in range(1, self.mlsthg + 1):
            r = MC - 1
            interior = (MC > MFR1STFR) and (MC < MFRLSTFR)
            high = (MC >= MFRLSTFR)
            low = not (interior or high)
            MCi = MC - 1; MP = self.d_mp[r] - 1; MP1 = self.d_mp1[r] - 1
            MM = self.d_mm[r] - 1; MM1 = self.d_mm1[r] - 1
            # the branch logic of source(): which destinations receive
            ok_mm  = interior or high
            ok_mm1 = interior or (high and MM1 + 1 <= ml) or (low and MM1 + 1 >= 1)
            ok_c   = interior or low or (high and MM1 + 1 <= ml and MCi + 1 <= ml)
            ok_mp  = interior or low or (high and MM1 + 1 <= ml and MCi + 1 <= ml and MP + 1 <= ml)
            ok_mp1 = ok_mp and (interior or low or MP1 + 1 <= ml)
            for KH in (0, 1):
                g = 2*r + KH
                T['K1'][g] = self.k1w[:, KH]; T['K11'][g] = self.k11w[:, KH]
                T['K2'][g] = self.k2w[:, KH]; T['K21'][g] = self.k21w[:, KH]
                T['IC'][g] = self.d_ic[r] - 1; T['IP'][g] = self.d_ip[r] - 1
                T['IP1'][g] = self.d_ip1[r] - 1; T['IM'][g] = self.d_im[r] - 1
                T['IM1'][g] = self.d_im1[r] - 1
                gw[:, g] = self.c_gw[:, r]; af[g] = self.c_af11[r]
                tail[g] = 1.0 if interior else self.c_ftail[r]
                def put(key, m, c):
                    # A destination outside 0..ML-1 happens only on a
                    # frequency grid too short for the DIA (ML <= KFRH, e.g.
                    # the nf = 2 grids of the propagation tests), where the
                    # Fortran writes out of bounds and the numpy loop fails
                    # with an IndexError in source(). The tables are then
                    # marked unusable and source() refuses, so that a model
                    # on such a grid can still be built and run without the
                    # nonlinear transfer, as with the numpy backend.
                    if not 0 <= m < ml:
                        if self._vec_bad is None:
                            self._vec_bad = (key, MC, int(m))
                        return
                    W[key][g, m] += c
                if ok_c:
                    put('all_AD', MCi, -2.0); put('all_DELAD', MCi, -2.0)
                if ok_mm:
                    put('K2_AD', MM, self.c_fklamm1[r]);  put('K2_DELAM', MM, self.c_fklam12[r])
                    put('K21_AD', MM, self.c_fklamm2[r]); put('K21_DELAM', MM, self.c_fklam22[r])
                if ok_mm1:
                    put('K2_AD', MM1, self.c_fklamma[r]);  put('K2_DELAM', MM1, self.c_fklama2[r])
                    put('K21_AD', MM1, self.c_fklammb[r]); put('K21_DELAM', MM1, self.c_fklamb2[r])
                if ok_mp:
                    put('K1_AD', MP, self.c_fklamp1[r]);  put('K1_DELAP', MP, self.c_fklap12[r])
                    put('K11_AD', MP, self.c_fklamp2[r]); put('K11_DELAP', MP, self.c_fklap22[r])
                if ok_mp1:
                    put('K1_AD', MP1, self.c_fklampa[r]);  put('K1_DELAP', MP1, self.c_fklapa2[r])
                    put('K11_AD', MP1, self.c_fklampb[r]); put('K11_DELAP', MP1, self.c_fklapb2[r])
        # inverse permutations: X_perm[g, D[g,k]] = X[g,k]  <=>  X_perm = X[g, Dinv[g]]
        Dinv = {k: np.argsort(T[k], axis=1) for k in ('K1', 'K11', 'K2', 'K21')}
        self._vec = dict(T=T, gw=gw, af=af, tail=tail, W=W, Dinv=Dinv, G=G)
        return self._vec

    # ---------------------------------------------------------------- source
    def source(self, F):
        """
        F : (nx, ny, kl, ml). Returns SL, FL with same shape.

        Vectorised over (nx, ny), the direction K AND the (MC, KH) groups
        (see _vec_tables): nine gathers of F with 2-D index tables, the
        quadruplet arithmetic on (nx,ny,G,kl) arrays, eight permutation
        gathers along the direction axis and ten matrix products over the
        group axis that place every contribution in its destination
        frequency. No scatter: the result is deterministic
        on the GPU and the memory traffic is a few passes over the
        G-expanded array (2.5 x F) instead of two scatter-adds per group
        (the loop form, ~280 full-array passes per step at 25 frequencies).
        On a CPU the loop form is faster (its per-group slices stay in the
        cache); the JAX backend is for the GPU.

        Same arithmetic as the Fortran/numpy loop (snonlin_np.py), the sums
        over the groups in a different order (round-off).
        """
        if self._vec_bad is not None:
            raise ValueError('Snonlin (jax): the frequency grid has %d frequencies, too few for '
                             'the DIA (it needs more than KFRH = %d); first destination outside '
                             'the grid: %s' % (self.ml, self.kfrh, self._vec_bad))
        F = OM.np.asarray(F, dtype=OM.prec)
        nx, ny, kl, ml = F.shape
        assert kl == self.kl and ml == self.ml, (F.shape, self.kl, self.ml)
        V = self._vec
        T, W, Dinv = V['T'], V['W'], V['Dinv']
        gw = V['gw'][:, :, None]                                # (8, G, 1)
        af, tail = V['af'][:, None], V['tail'][:, None]        # (G, 1)
        gi = np.arange(V['G'])[:, None]                         # (G, 1) group index

        def g(K, M):                                            # F[:, :, K[g,k], M[g]] -> (nx,ny,G,kl)
            return F[:, :, T[K], T[M]]

        SAP = gw[0]*g('K1', 'IP') + gw[1]*g('K11', 'IP') + gw[2]*g('K1', 'IP1') + gw[3]*g('K11', 'IP1')
        SAM = gw[4]*g('K2', 'IM') + gw[5]*g('K21', 'IM') + gw[6]*g('K2', 'IM1') + gw[7]*g('K21', 'IM1')
        FIJ = g('Kall', 'IC')*tail
        FAD1 = FIJ*(SAP + SAM)
        FAD2 = FAD1 - 2.0*SAP*SAM
        FAD1 = FAD1 + FAD2
        FCEN = af*FIJ
        AD    = FAD2*FCEN
        DELAD = FAD1*af
        DELAP = (FIJ - 2.0*SAM)*self.dal1*FCEN
        DELAM = (FIJ - 2.0*SAP)*self.dal2*FCEN

        def perm(X, K):                                         # X[g, k] -> destination bin K[g, k]
            return X[:, :, gi, Dinv[K]]

        # every (direction table, quantity) pair placed into its destination
        # frequencies by a matrix product over the group axis (contraction
        # over G), (nx,ny,kl,G) x (G,ml), summed: ten small GEMMs, one
        # permuted array live at a time (a single stacked GEMM over 5G
        # would be a 2.5 x larger temporary, 2 GB per GPU at 384^2)
        def place(X, key):
            return OM.np.einsum('xygk,gm->xykm', X, OM.np.asarray(W[key]))

        SL = (place(AD, 'all_AD') + place(perm(AD, 'K2'), 'K2_AD') + place(perm(AD, 'K21'), 'K21_AD')
              + place(perm(AD, 'K1'), 'K1_AD') + place(perm(AD, 'K11'), 'K11_AD'))
        FL = (place(DELAD, 'all_DELAD') + place(perm(DELAM, 'K2'), 'K2_DELAM')
              + place(perm(DELAM, 'K21'), 'K21_DELAM') + place(perm(DELAP, 'K1'), 'K1_DELAP')
              + place(perm(DELAP, 'K11'), 'K11_DELAP'))
        return SL, FL


# module-level object, like the f2py extension exposes
Snonlin = _Snonlin()
