"""
 Non-hydrostatic model on the pe_realistic infrastructure: topography
 mask (kbot), variable grid spacings of grid.py (Cartesian only, the
 pseudo-spherical grid is refused), MPI or JAX domain decomposition, temp/salt and the
 equation of state, the option hierarchy of vertical mixing (constant,
 convective adjustment, TKE closure), biharmonic and harmonic friction.

 What is added to pe_realistic (and only that):
   momentum.py   prognostic vertical velocity w (advection in transport
                 form, AB3 update, horizontal friction)
   vmix.py       vertical friction of w (explicit, implicit, TKE viscosity)
   pressure.py   7-point non-hydrostatic pressure operator and solve
   solver.py     preconditioned conjugate gradient for that operator
   main.py       the extra fields (w tendency, p_nh, its coefficients,
                 the closed-top W mask)
   model.py      the time step: pe_realistic's with the w equation and
                 the 3-D projection inserted

 Formulation (pyOM2 enable_hydrostatic = .false., Oceananigans
 NonhydrostaticModel): p = p_h + p_s + p_nh. p_h is the hydrostatic
 pressure of pe_realistic, p_s the surface pressure of its 2-D solve
 (rigid lid, implicit or split-explicit free surface, unchanged), p_nh the
 non-hydrostatic pressure from the 3-D solve. The horizontal momentum
 equations are those of pe_realistic plus -grad_h p_nh. The vertical
 momentum equation carries advection, friction and -d p_nh/dz; the
 buoyancy cancels exactly against -d p_h/dz because p_h is integrated
 with dzw and 0.5 (rho_k + rho_k+1), so p_nh = 0 is pe_realistic to
 round-off (enable_nonhydrostatic = False gives exactly that path).

 Pressure solve per time step: 2-D surface solve as in pe_realistic, then
 the 3-D solve for p_nh with the volume-weighted operator
   V div grad p = sum over the six faces of  A_face (p_nb - p)/d_face
 (face area times inverse spacing, zero on masked faces, closed at the
 surface), which is symmetric, so the conjugate gradient is on safe
 ground; the right hand side is V div(u*,v*,w*)/dt. Preconditioner: the
 exact vertical tridiagonal solve of each column (Marshall et al. 1997,
 the Oceananigans ColumnwiseTridiagonalPreconditioner), the right choice
 for (dz/dx)^2 nz^2 << 1. Zero-mean gauge over the wet cells.

 Not included (regional runs): the pseudo-spherical grid, since the w
 equation would lack its metric terms (-(u^2+v^2)/a) and the
 non-traditional Coriolis terms; main.__init__ raises for coord_degree =
 True. The new kernels carry no metric factors; the inherited
 pe_realistic kernels keep theirs, which are 1. The dissipation of the w
 friction is not booked into the TKE closure.
"""
