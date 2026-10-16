import sympy as sp

# Canonical ray equations on the sphere. Coordinates q=(lam,phi) with
# conjugate momenta p=(m,n) where the local wavenumber components are
#   k_e = m/(R cos phi)   (east),   k_n = n/R   (north).
# Omega(q,p) = sigma(k) + k.U with k=|k|, U=(Ue(lam,phi), Un(lam,phi)).
# These are exact: lamdot=dOm/dm, phidot=dOm/dn, mdot=-dOm/dlam, ndot=-dOm/dphi.

lam, phi, m, n, R, g = sp.symbols('lambda phi m n R g', real=True, positive=True)
Ue = sp.Function('Ue')(lam,phi); Un = sp.Function('Un')(lam,phi)

ke = m/(R*sp.cos(phi)); kn = n/R
k  = sp.sqrt(ke**2+kn**2)
sig = sp.sqrt(g*k)                       # deep water
Om  = sig + ke*Ue + kn*Un

lamdot = sp.diff(Om,m); phidot = sp.diff(Om,n)
mdot   = -sp.diff(Om,lam); ndot = -sp.diff(Om,phi)

# direction theta clockwise from north: ke = k sin th, kn = k cos th
th = sp.atan2(ke,kn)
# thdot along the ray = total derivative of th(lam,phi,m,n)
thdot = (sp.diff(th,lam)*lamdot + sp.diff(th,phi)*phidot
        +sp.diff(th,m)*mdot + sp.diff(th,n)*ndot)
sigdot = (sp.diff(sig,lam)*lamdot + sp.diff(sig,phi)*phidot
         +sp.diff(sig,m)*mdot + sp.diff(sig,n)*ndot)

# substitute polar form: m = k R cos phi sin th0, n = k R cos th0
k0, th0 = sp.symbols('k0 theta0', positive=True, real=True)
sub = {m: k0*R*sp.cos(phi)*sp.sin(th0), n: k0*R*sp.cos(th0)}
cg = sp.sqrt(g/k0)/2

thdot_s = sp.simplify(sp.trigsimp(thdot.subs(sub)))
sigdot_s = sp.simplify(sp.trigsimp(sigdot.subs(sub)))

# Reference forms:
s, c = sp.sin(th0), sp.cos(th0)
ddx = lambda F: sp.diff(F,lam)/(R*sp.cos(phi))   # (R cos phi)^-1 d/dlam
ddy = lambda F: sp.diff(F,phi)/R                 # R^-1 d/dphi
# Cartesian-form current refraction with spherical gradients (wim's ddot):
ddot_cart = -(s*c*ddx(Ue) + c*c*ddx(Un)) + (s*s*ddy(Ue) + s*c*ddy(Un))
# frame rotation with FULL advective eastward velocity:
frame_full = (cg*s + Ue)*sp.tan(phi)/R
# frame rotation cg only (ECMWF 2.25c):
frame_cg   = cg*s*sp.tan(phi)/R

res_full = sp.simplify(sp.trigsimp(thdot_s - (ddot_cart + frame_full)))
res_cg   = sp.simplify(sp.trigsimp(thdot_s - (ddot_cart + frame_cg)))
print('thdot - [cart-gradient refr + (cg sin th + Ue) tanphi/R] =', res_full)
print('thdot - [cart-gradient refr + cg sin th tanphi/R]        =', res_cg)

# sigdot reference: cg * s.kdot/|k| Cartesian form with spherical gradients
sigdot_cart = -cg*k0*( s*(s*ddx(Ue)+c*ddx(Un)) + c*(s*ddy(Ue)+c*ddy(Un)) )
res_sig = sp.simplify(sp.trigsimp(sigdot_s - sigdot_cart))
print('sigdot - cart-gradient form =', sp.simplify(res_sig))

# Compact conjecture:
#   thdot  = cart-gradient refr + (cg + khat.U) sin th tan phi / R
#   sigdot = cart-gradient form - cg k (U.khat_perp) sin th tan phi / R
# with khat=(s,c) east-north, khat_perp = d khat/d th = (c,-s)
kU  = s*Ue + c*Un
kUp = c*Ue - s*Un
print('check thdot :', sp.simplify(sp.trigsimp(thdot_s - (ddot_cart + (cg+kU)*s*sp.tan(phi)/R))))
print('check sigdot:', sp.simplify(sp.trigsimp(sigdot_s - (sigdot_cart - cg*k0*kUp*s*sp.tan(phi)/R))))

# and xdot/ydot (the advective velocities) in local east/north components:
xdot_phys = lamdot.subs(sub)*R*sp.cos(phi)
ydot_phys = phidot.subs(sub)*R
print('xdot:', sp.simplify(sp.trigsimp(xdot_phys - (cg*s + Ue))))
print('ydot:', sp.simplify(sp.trigsimp(ydot_phys - (cg*c + Un))))
