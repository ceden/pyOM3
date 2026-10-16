"""1-D linear shallow water, staggered grid, Demange forward-backward sub-cycle
exactly as in pyOM3 loop_it (f = 0, no forcing), periodic. Three ways to start
the next baroclinic step:
  'average' : U0 = <<U>> (pyOM3: u trimmed to the average, U rebuilt from u)
  'end'     : U0 = U_end (FESOM: barotropic state carried across steps)
  'continue': the loop just continues (same as 'end' here, sanity)
Growth = energy after nsteps baroclinic steps / initial energy."""
import numpy as np
def run(restart, M, nsteps=50, N=120, cfl_bc=13.6, theta=0.14, seed=0):
    dx = 1.0; c = 1.0; dt = cfl_bc*dx/c; dtb = dt/M; g = 1.0; H = 1.0    # c = sqrt(gH) = 1
    rng = np.random.default_rng(seed)
    x = np.arange(N)*dx
    eta = np.exp(-((x-N/2)/4.)**2); U = np.zeros(N)
    E0 = np.sum(eta**2)
    Uc = U.copy()                                   # carried barotropic state
    for n in range(nsteps):
        U0 = Uc if restart == 'end' else U           # 'average': U holds <<U>>
        Um1 = U0.copy(); Uk = U0.copy(); Umean = np.zeros(N)
        for m in range(M):
            Up1 = Uk - dtb*H*(np.roll(eta,-1) - eta)/dx           # U at i+1/2, eta at i
            Umean += Up1
            eta = eta - dtb*g*( (1+theta)*(Up1 - np.roll(Up1,1)) - theta*(Uk - np.roll(Uk,1)) )/dx
            Um1, Uk = Uk, Up1
        Umean = Umean/M + theta/M*(Uk - U0)
        Uc = Uk.copy()                               # end-of-cycle transport
        U = Umean.copy()                             # what the trimmed 3-D field carries
    return (np.sum(eta**2) + np.sum(U**2))/E0, (np.sum(eta**2) + np.sum(Uc**2))/E0
for cfl in (0.7, 13.6):
    print('baroclinic CFL c dt/dx = %.1f' % cfl)
    for M in (16, 31, 62, 124):
        ga = run('average', M, cfl_bc=cfl); ge = run('end', M, cfl_bc=cfl)
        print('  M %4d (sub-step CFL %.2f): restart from average -> energy ratio %10.3f ; from end value -> %10.3f'
              % (M, cfl/M, ga[0], ge[1]))
