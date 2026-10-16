"""2-D version of toy_split: C-grid, closed box (U=0 on the walls), anisotropic
dx, dy as the POP-like box, Demange FB loop as coded in pyOM3, 'end' restart
(barotropic state carried) vs 'average' restart. Energy ratio (potential +
kinetic) after nsteps baroclinic steps, and p_s-variance ratio."""
import numpy as np
def run(restart, M, nsteps=50, nx=30, ny=30, dx=3528., dy=11119., H=5000., dt=216., theta=0.14, g=9.81):
    dtb = dt/M
    eta = np.zeros((ny, nx)); U = np.zeros((ny, nx+1)); V = np.zeros((ny+1, nx))   # U at x-faces, V at y-faces
    j, i = np.mgrid[0:ny, 0:nx]
    eta[:] = 0.5*np.exp(-(((i-nx/2)/4.)**2 + ((j-ny/2)/4.)**2))
    E = lambda eta, U, V: 0.5*g*np.sum(eta**2)*dx*dy + 0.5/H*(np.sum(U[:,1:-1]**2) + np.sum(V[1:-1,:]**2))*dx*dy
    E0 = E(eta, U, V); P0 = np.sum(eta**2)
    Uc, Vc = U.copy(), V.copy()
    for n in range(nsteps):
        Uk, Vk = (Uc, Vc) if restart == 'end' else (U, V)
        Uk, Vk = Uk.copy(), Vk.copy(); U0, V0 = Uk.copy(), Vk.copy()
        Um = np.zeros_like(Uk); Vm = np.zeros_like(Vk)
        for m in range(M):
            Up1 = Uk.copy(); Vp1 = Vk.copy()
            Up1[:, 1:-1] = Uk[:, 1:-1] - dtb*H*g*(eta[:, 1:] - eta[:, :-1])/dx      # walls: U[:,0]=U[:,-1]=0
            Vp1[1:-1, :] = Vk[1:-1, :] - dtb*H*g*(eta[1:, :] - eta[:-1, :])/dy
            Um += Up1; Vm += Vp1
            divp1 = (Up1[:, 1:] - Up1[:, :-1])/dx + (Vp1[1:, :] - Vp1[:-1, :])/dy
            divk  = (Uk[:, 1:]  - Uk[:, :-1])/dx  + (Vk[1:, :]  - Vk[:-1, :])/dy
            eta = eta - dtb*((1+theta)*divp1 - theta*divk)
            Uk, Vk = Up1, Vp1
        Um = Um/M + theta/M*(Uk - U0); Vm = Vm/M + theta/M*(Vk - V0)
        Uc, Vc = Uk, Vk
        U, V = Um, Vm
    Ue, Ve = (Uc, Vc) if restart == 'end' else (U, V)
    return E(eta, Ue, Ve)/E0, np.sum(eta**2)/P0
for M in (16, 31, 62, 124):
    a = run('average', M); e = run('end', M)
    print('M %4d: average-restart energy %8.3f (p_s var %8.3f) | end-restart energy %8.3f (p_s var %8.3f)'
          % (M, a[0], a[1], e[0], e[1]))
print('isotropic dx=dy=3528 m, end restart:', run('end', 31, dy=3528.))
print('periodic-free check, H 1250 (single-layer case):', run('end', 28, H=1250.))
