"""
lin_stab: linear stability of a parallel shear flow U(z) with stratification B(z)

Non-hydrostatic, two dimensional (x,z), free slip rigid lids at top and bottom.
The perturbation streamfunction psi = phi(z) exp(i k (x - c t)) with u = psi_z,
w = -psi_x and buoyancy b = bh(z) exp(i k (x - c t)) satisfy

    c L phi = U L phi - U'' phi + bh + i nu/k L L phi        L = d^2/dz^2 - k^2
    c bh    = U bh - N^2 phi + i kappa/k L bh

which is solved as a matrix eigenvalue problem for c on the given z levels.
The growth rate is k Im(c). Without viscosity nu the discrete critical layers
produce spurious growing modes at large k, so use the viscosity of the model.
"""
import numpy as np


def taylor_goldstein(U, B, z, k, nu=0., kappa=0.):
    """
    U, B, z : mean velocity, mean buoyancy and depth on N levels (uniform spacing)
    k       : array of zonal wavenumbers
    nu, kappa : viscosity and diffusivity
    returns c (complex phase speed of the fastest growing mode for each k) and
    for the overall fastest mode: k_max, c_max, u(z), w(z), b(z) (complex amplitudes)
    """
    N, dz = len(z), z[1] - z[0]
    Uzz = np.gradient(np.gradient(U, dz), dz)
    N2  = np.gradient(B, dz)
    D2  = (np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1) - 2*np.eye(N))/dz**2   # phi = 0 at the lids
    c   = np.zeros(len(k), complex)
    best = None
    for i in range(len(k)):
        L  = D2 - k[i]**2*np.eye(N)
        A  = np.block([[np.diag(U) @ L - np.diag(Uzz) + 1j*nu/k[i]*L @ L, np.eye(N)],
                       [-np.diag(N2), np.diag(U) + 1j*kappa/k[i]*L]])
        Bm = np.block([[L, np.zeros((N, N))], [np.zeros((N, N)), np.eye(N)]])
        ev, vec = np.linalg.eig(np.linalg.solve(Bm, A))
        n = np.argmax(ev.imag)
        c[i] = ev[n]
        if best is None or k[i]*ev[n].imag > best[0]:
            best = (k[i]*ev[n].imag, k[i], ev[n], vec[:, n])
    sigma, k_max, c_max, vec = best
    phi, b = vec[:N], vec[N:]
    u = np.gradient(phi, dz)
    w = -1j*k_max*phi
    s = abs(u).max()
    return c, k_max, c_max, u/s, w/s, b/s
