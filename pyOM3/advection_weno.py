"""
 WENO5 reconstruction of a tracer at cell faces, shared by the nh and pe
 families (nh_simple/advection.py, pe_simple/thermodynamics.py, ...).

 Fifth-order weighted essentially non-oscillatory reconstruction with the
 Z weights of Borges et al. (2008), as implemented in Oceananigans.jl
 (src/Advection/weno_interpolants.jl, Oct 2026). Finite-volume form: the
 face value is built from the cell values of the five cells around the
 face, three candidate parabolas are combined with weights that fall to
 zero where a candidate stencil crosses a jump.

 Index convention. Everything is written with the model's roll functions,
 so it works for the (Nz,Ny,Nx) arrays of pe and the fortran ordered
 (Nx,Ny,Nz) arrays of nh_simple, under numpy, MPI halos and JAX sharding
 alike. roll(a,-1)[i] = a[i+1] and roll(a,1)[i] = a[i-1]. The face with
 index i is the one between cell i and cell i+1, as for the fluxes fe, fn,
 ft of the models.

 Stencil and halo. The left-biased reconstruction at face i (used when the
 velocity through the face is positive) reads the cells i-2 .. i+2, the
 right-biased one the cells i-1 .. i+3. The flux divergence at the first
 interior cell needs the face in the halo slot, so three halo points are
 required (halo_points = 3).

 Next to land. The order is lowered where the stencil of the direction in
 use reaches a dry cell: fifth order where its five cells are wet, third
 order where the three cells of the WENO3 stencil are, first-order upwind
 otherwise (for the left-biased value: i-2 .. i+2, i-1 .. i+1, else the
 value of cell i). Oceananigans lowers the order with the union of the
 left and right stencils and ends at the centred second-order value. The
 centred value is not used here: at the faces next to a wall it produced
 under- and overshoots of half the jump at a sharp front in overflow1_3D
 (nh_realistic, 10 and 12 degC, a barrier and a shelf edge), where
 superbee stays bounded; the upwind value removes them. The flux is still
 one value per face, so the scheme stays conservative, and the order
 depends on the geometry and the sign of the velocity at that face. The
 wet test is made with the T-cell mask of the tracer (maskT), NOT with the
 face masks: maskW is opened at the surface for the free surface solver,
 maskT is not.

 Regularisation. The Z weights contain eps in the denominator beta + eps.
 beta is a squared difference of tracer values, so eps = 1e-8 is a
 statement about the SIZE of the tracer. For a tracer with values of
 order one (temperature in degC, salinity) eps = 1e-8 is right. For a
 tracer of order 1e-4 (buoyancy in m/s^2) beta is below 1e-8 everywhere
 and the weights degenerate to the linear fifth-order upwind scheme,
 which oscillates at jumps. eps is therefore scaled, eps = 1e-8 ref^2,
 with ref the typical size of the tracer (a difference across the domain
 is a good value). The models pass their weno_ref_* parameters here.
"""
import pyOM3 as OM


EPS0 = 1.0e-8       # Oceananigans' epsilon for a tracer of order one


def face_value_weno5(q, roll, eps):
    """
    Left- and right-biased WENO5-Z reconstructions of q at the face i
    (between cell i and cell i+1). Returns (qL, qR) as arrays of the shape
    of q. qL is the value seen by a flow in the positive direction (from
    cell i into cell i+1), qR by a flow in the negative direction.
    """
    qm2, qm1, q0, qp1, qp2, qp3 = (roll(q, 2), roll(q, 1), q,
                                   roll(q, -1), roll(q, -2), roll(q, -3))
    qL = _weno5_biased(qm2, qm1, q0, qp1, qp2, eps)
    qR = _weno5_biased(qp3, qp2, qp1, q0, qm1, eps)     # the mirror image
    return qL, qR


def face_value_weno3(q, roll, eps):
    """
    Third-order counterpart of face_value_weno5: cells i-1 .. i+1 for qL,
    i .. i+2 for qR.
    """
    qm1, q0, qp1, qp2 = roll(q, 1), q, roll(q, -1), roll(q, -2)
    qL = _weno3_biased(qm1, q0, qp1, eps)
    qR = _weno3_biased(qp2, qp1, q0, eps)
    return qL, qR


def face_value_centred(q, roll):
    """Second-order centred value at the face i."""
    return 0.5*(q + roll(q, -1))


def _weno5_biased(a, b, c, d, e, eps):
    """
    Biased WENO5-Z reconstruction at the right face of the cell with value
    c, from the five cell values a, b, c, d, e in flow direction (a is
    furthest upstream, e furthest downstream). Jiang and Shu smoothness
    indicators, Borges et al. (2008) Z weights with the global indicator
    tau = |beta_0 - beta_2|, optimal linear weights 1/10, 6/10, 3/10.
    """
    p0 = (2.0*a - 7.0*b + 11.0*c)/6.0
    p1 = (-b + 5.0*c + 2.0*d)/6.0
    p2 = (2.0*c + 5.0*d - e)/6.0
    beta0 = 13.0/12.0*(a - 2.0*b + c)**2 + 0.25*(a - 4.0*b + 3.0*c)**2
    beta1 = 13.0/12.0*(b - 2.0*c + d)**2 + 0.25*(b - d)**2
    beta2 = 13.0/12.0*(c - 2.0*d + e)**2 + 0.25*(3.0*c - 4.0*d + e)**2
    tau = OM.np.abs(beta0 - beta2)
    alpha0 = 0.1*(1.0 + (tau/(beta0 + eps))**2)
    alpha1 = 0.6*(1.0 + (tau/(beta1 + eps))**2)
    alpha2 = 0.3*(1.0 + (tau/(beta2 + eps))**2)
    s = alpha0 + alpha1 + alpha2
    return (alpha0*p0 + alpha1*p1 + alpha2*p2)/s


def _weno3_biased(a, b, c, eps):
    """
    Biased WENO3-Z reconstruction at the right face of the cell with value
    b, from the three cell values a, b, c in flow direction. Optimal linear
    weights 1/3, 2/3, tau = |beta_0 - beta_1|.
    """
    p0 = (-a + 3.0*b)/2.0
    p1 = (b + c)/2.0
    beta0 = (b - a)**2
    beta1 = (c - b)**2
    tau = OM.np.abs(beta0 - beta1)
    alpha0 = (1.0/3.0)*(1.0 + (tau/(beta0 + eps))**2)
    alpha1 = (2.0/3.0)*(1.0 + (tau/(beta1 + eps))**2)
    return (alpha0*p0 + alpha1*p1)/(alpha0 + alpha1)


def stencil_masks(maskT, roll):
    """
    The wet masks of the face i for the four biased stencils, computed once
    per direction at setup from the T mask (pe) or from a mask with zero
    halo rows in the bounded direction (nh lid):
      wet5L  cells i-2 .. i+2 wet (left-biased WENO5)
      wet5R  cells i-1 .. i+3 wet (right-biased WENO5)
      wet3L  cells i-1 .. i+1 wet (left-biased WENO3)
      wet3R  cells i   .. i+2 wet (right-biased WENO3)
    """
    m0, mp1 = maskT, roll(maskT, -1)
    wet3L = roll(maskT, 1)*m0*mp1
    wet3R = m0*mp1*roll(maskT, -2)
    wet5L = wet3L*roll(maskT, 2)*roll(maskT, -2)
    wet5R = wet3R*roll(maskT, 1)*roll(maskT, -3)
    return wet5L, wet5R, wet3L, wet3R


def face_value(q, vel, roll, wet, eps):
    """
    Upwind face value of q for the velocity vel through the face i. wet is
    the tuple of stencil_masks. vel > 0 selects the left-biased values:
    WENO5 where wet5L, WENO3 where wet3L only, the upwind cell value q[i]
    otherwise; vel <= 0 the right-biased ones with q[i+1]. The caller
    multiplies by the velocity (or the transport) and the face mask.
    """
    wet5L, wet5R, wet3L, wet3R = wet
    qL5, qR5 = face_value_weno5(q, roll, eps)
    qL3, qR3 = face_value_weno3(q, roll, eps)
    qL = OM.np.where(wet5L > 0.0, qL5, OM.np.where(wet3L > 0.0, qL3, q))
    qR = OM.np.where(wet5R > 0.0, qR5, OM.np.where(wet3R > 0.0, qR3, roll(q, -1)))
    return OM.np.where(vel > 0.0, qL, qR)


def check_halo(halo):
    """WENO5 needs three halo points; refuse anything less."""
    if halo < 3:
        raise ValueError('WENO5 advection needs halo_points >= 3, got %d' % halo)
