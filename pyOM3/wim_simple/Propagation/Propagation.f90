! Fortran kernels for the propagation operator of wim_simple / wim_realistic
! (superbee fluxes on the four phase-space axes and the flux divergence).
!
! Same arithmetic as tracer_flux_superbee / add_propagation in propagation.py,
! operation by operation and in the same order, so the result is bit-identical
! to the numpy path (build without FMA contraction and without fast-math, see
! the Makefile). The gain is memory traffic: numpy evaluates the flux in ~15
! full-array passes per axis, the kernels in one.
!
! ARRAY LAYOUT: the python state is C-ordered (Nd,Nf,Ny,Nx). Every array is
! declared here with the dimensions REVERSED, (nx,ny,nf,nd), and the python
! side passes the transposed view a.T -- a Fortran-contiguous view of the same
! memory -- so f2py makes NO copy. Passing the C-ordered array itself would
! silently transpose-copy every argument on entry and exit.
!
! HALOS: the gradient rj and the fluxes are computed on the interior box only
! (h+1..N-h on every axis); apply_bc fills the halos afterwards exactly as in
! the numpy path (which computes wrapped halo values and then overwrites them).
! The divergence is computed on the full array with the periodic index wrap of
! np.roll, so dA is identical everywhere, halos included.
!
! Metric: den is the CFL denominator (dx, or cost*dxt on the sphere), pref the
! y-flux prefactor (1, or cosu on the sphere), m the face/land mask (1 without
! land). Multiplication by 1.0 is exact, so the Cartesian model uses the same
! kernels with unit tables.

module sb_kernel
  implicit none
  real*8, parameter :: eps = 1d-20
contains

  pure function sbflux(a0, a1, r, rm, rp, v, pref, den, dt, m) result(f)
    ! superbee flux at the face between a0 and a1 (MITgcm form), identical to
    !   CFL = |pref*v*dt/den|
    !   cr  = max(clip(2q,0,1), clip(q,0,2)),  q = where(v>0,rjm,rjp)/where(|rj|<eps,eps,rj)
    !   f   = -pref*( v*(a1+a0)*0.5 - |v|*((1-cr)+CFL*cr)*rj*0.5 )*m
    real*8, intent(in) :: a0, a1, r, rm, rp, v, pref, den, dt, m
    real*8 :: f, cfl, dn, q, c1, c2, cr, x
    cfl = abs(((pref*v)*dt)/den)
    if (abs(r) < eps) then
       dn = eps
    else
       dn = r
    end if
    if (v > 0d0) then
       q = rm/dn
    else
       q = rp/dn
    end if
    c1 = min(max(2d0*q, 0d0), 1d0)
    c2 = min(max(q,     0d0), 2d0)
    cr = max(c1, c2)
    x  = ((v*(a1+a0))*0.5d0) - (((abs(v)*((1d0-cr)+cfl*cr))*r)*0.5d0)
    f  = (pref*(-x))*m
  end function sbflux

end module sb_kernel


! ---------------------------------------------------------------- gradients
!  rj = (var(+1) - var)*mask  on the interior box

subroutine grad_x(var, m, rj, h, nx, ny, nf, nd)
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), m(nx,ny)
  real*8, intent(inout) :: rj(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       rj(i,j,f,d) = (var(i+1,j,f,d) - var(i,j,f,d))*m(i,j)
     end do
    end do
   end do
  end do
end subroutine grad_x

subroutine grad_y(var, m, rj, h, nx, ny, nf, nd)
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), m(nx,ny)
  real*8, intent(inout) :: rj(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       rj(i,j,f,d) = (var(i,j+1,f,d) - var(i,j,f,d))*m(i,j)
     end do
    end do
   end do
  end do
end subroutine grad_y

subroutine grad_f(var, rj, h, nx, ny, nf, nd)
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd)
  real*8, intent(inout) :: rj(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       rj(i,j,f,d) = var(i,j,f+1,d) - var(i,j,f,d)
     end do
    end do
   end do
  end do
end subroutine grad_f

subroutine grad_d(var, rj, h, nx, ny, nf, nd)
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd)
  real*8, intent(inout) :: rj(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       rj(i,j,f,d) = var(i,j,f,d+1) - var(i,j,f,d)
     end do
    end do
   end do
  end do
end subroutine grad_d


! ------------------------------------------------------------------- fluxes
!  interior box; rj must already carry its halo (apply_bc)

subroutine flux_x(var, rj, vel, den, m, dt, fx, h, nx, ny, nf, nd)
  use sb_kernel
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), rj(nx,ny,nf,nd), vel(nx,ny,nf,nd)
  real*8, intent(in)  :: den(nx,ny), m(nx,ny), dt
  real*8, intent(inout) :: fx(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       fx(i,j,f,d) = sbflux(var(i,j,f,d), var(i+1,j,f,d), rj(i,j,f,d),      &
                            rj(i-1,j,f,d), rj(i+1,j,f,d), vel(i,j,f,d),     &
                            1d0, den(i,j), dt, m(i,j))
     end do
    end do
   end do
  end do
end subroutine flux_x

subroutine flux_y(var, rj, vel, den, pref, m, dt, fy, h, nx, ny, nf, nd)
  use sb_kernel
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), rj(nx,ny,nf,nd), vel(nx,ny,nf,nd)
  real*8, intent(in)  :: den(nx,ny), pref(ny), m(nx,ny), dt
  real*8, intent(inout) :: fy(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       fy(i,j,f,d) = sbflux(var(i,j,f,d), var(i,j+1,f,d), rj(i,j,f,d),      &
                            rj(i,j-1,f,d), rj(i,j+1,f,d), vel(i,j,f,d),     &
                            pref(j), den(i,j), dt, m(i,j))
     end do
    end do
   end do
  end do
end subroutine flux_y

subroutine flux_f(var, rj, vel, den, dt, ff, h, nx, ny, nf, nd)
  use sb_kernel
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), rj(nx,ny,nf,nd), vel(nx,ny,nf,nd)
  real*8, intent(in)  :: den(nf), dt
  real*8, intent(inout) :: ff(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       ff(i,j,f,d) = sbflux(var(i,j,f,d), var(i,j,f+1,d), rj(i,j,f,d),      &
                            rj(i,j,f-1,d), rj(i,j,f+1,d), vel(i,j,f,d),     &
                            1d0, den(f), dt, 1d0)
     end do
    end do
   end do
  end do
end subroutine flux_f

subroutine flux_d(var, rj, vel, den, dt, fd, h, nx, ny, nf, nd)
  use sb_kernel
  implicit none
  integer, intent(in) :: h, nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), rj(nx,ny,nf,nd), vel(nx,ny,nf,nd)
  real*8, intent(in)  :: den, dt
  real*8, intent(inout) :: fd(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = h+1, nd-h
   do f = h+1, nf-h
    do j = h+1, ny-h
     do i = h+1, nx-h
       fd(i,j,f,d) = sbflux(var(i,j,f,d), var(i,j,f,d+1), rj(i,j,f,d),      &
                            rj(i,j,f,d-1), rj(i,j,f,d+1), vel(i,j,f,d),     &
                            1d0, den, dt, 1d0)
     end do
    end do
   end do
  end do
end subroutine flux_d


! --------------------------------------------------------------- divergence
!  dA = ( (fx-fx(-1))/denx + (fy-fy(-1))/deny + (ff-ff(-1))/dft + (fd-fd(-1))/dd )*mt
!  on the full array, periodic index wrap as np.roll

subroutine divergence(fx, fy, ff, fd, denx, deny, dft, dd, mt, da, nx, ny, nf, nd)
  implicit none
  integer, intent(in) :: nx, ny, nf, nd
  real*8, intent(in)  :: fx(nx,ny,nf,nd), fy(nx,ny,nf,nd), ff(nx,ny,nf,nd), fd(nx,ny,nf,nd)
  real*8, intent(in)  :: denx(nx,ny), deny(nx,ny), dft(nf), dd, mt(nx,ny)
  real*8, intent(inout) :: da(nx,ny,nf,nd)
  integer :: i, j, f, d, im, jm, fm, dm
  real*8 :: s
  do d = 1, nd
   dm = d-1; if (d == 1) dm = nd
   do f = 1, nf
    fm = f-1; if (f == 1) fm = nf
    do j = 1, ny
     jm = j-1; if (j == 1) jm = ny
     do i = 1, nx
       im = i-1; if (i == 1) im = nx
       s = (fx(i,j,f,d) - fx(im,j,f,d))/denx(i,j)
       s = s + (fy(i,j,f,d) - fy(i,jm,f,d))/deny(i,j)
       s = s + (ff(i,j,f,d) - ff(i,j,fm,d))/dft(f)
       s = s + (fd(i,j,f,d) - fd(i,j,f,dm))/dd
       da(i,j,f,d) = s*mt(i,j)
     end do
    end do
   end do
  end do
end subroutine divergence


! ===========================================================================
! kernels for the remaining propagation-side routines: mk_advection,
! energy_fix_flux, open_coast_flux. Same bit-identity contract and the same
! reversed-dimension no-copy convention as above. Reductions (the column
! sums of the energy fixer) deliberately stay in numpy: np.sum's pairwise
! order is part of the bit pattern, so the kernels only produce the 4-D
! integrand fields and python sums them exactly as before.
! ===========================================================================

subroutine mkadv_vel(xdot, ydot, u, v, dudf, dvdf, cg, ft, sth, cth,        &
                     hascur, hasdf, nx, ny, nf, nd)
  ! xdot = cg*s (+ U + R*s),  ydot = cg*c (+ V + R*c),
  ! R = (0.5*ft)*(s*dUdf + c*dVdf); full arrays, as the numpy broadcasts
  implicit none
  integer, intent(in) :: hascur, hasdf, nx, ny, nf, nd
  real*8, intent(in)  :: u(nx,ny,nf), v(nx,ny,nf), dudf(nx,ny,nf), dvdf(nx,ny,nf)
  real*8, intent(in)  :: cg(nf), ft(nf), sth(nd), cth(nd)
  real*8, intent(inout) :: xdot(nx,ny,nf,nd), ydot(nx,ny,nf,nd)
  integer :: i, j, f, d
  real*8 :: x, y, r
  do d = 1, nd
   do f = 1, nf
    do j = 1, ny
     do i = 1, nx
       x = cg(f)*sth(d)
       y = cg(f)*cth(d)
       if (hascur == 1) then
          x = x + u(i,j,f)
          y = y + v(i,j,f)
          r = (0.5d0*ft(f))*((sth(d)*dudf(i,j,f)) + (cth(d)*dvdf(i,j,f)))
          x = x + (r*sth(d))
          y = y + (r*cth(d))
       end if
       xdot(i,j,f,d) = x
       ydot(i,j,f,d) = y
     end do
    end do
   end do
  end do
end subroutine mkadv_vel


subroutine mkadv_rates(dd_out, fd_out, dudx, dvdx, dudy, dvdy, u, v,        &
                       cg, ft, sth, cth, sthu, cthu, tantr, sph,            &
                       nx, ny, nf, nd)
  ! raw ddot (theta faces) and fdot (centres) of the currents branch, before
  ! apply_bc; sph=1 adds the exact spherical turning terms of wim_realistic
  implicit none
  integer, intent(in) :: sph, nx, ny, nf, nd
  real*8, intent(in)  :: dudx(nx,ny,nf), dvdx(nx,ny,nf), dudy(nx,ny,nf), dvdy(nx,ny,nf)
  real*8, intent(in)  :: u(nx,ny,nf), v(nx,ny,nf)
  real*8, intent(in)  :: cg(nf), ft(nf), sth(nd), cth(nd), sthu(nd), cthu(nd), tantr(ny)
  real*8, intent(inout) :: dd_out(nx,ny,nf,nd), fd_out(nx,ny,nf,nd)
  integer :: i, j, f, d
  real*8 :: s, c, su, cu, kx, ky, t1, t2, t3, t4, ddv, fdv
  do d = 1, nd
   s  = sth(d);  c  = cth(d)
   su = sthu(d); cu = cthu(d)
   do f = 1, nf
    do j = 1, ny
     do i = 1, nx
       kx = -((s*dudx(i,j,f)) + (c*dvdx(i,j,f)))
       ky = -((s*dudy(i,j,f)) + (c*dvdy(i,j,f)))
       t1 = ((-su)*cu)*dudx(i,j,f)
       t2 = (cu*cu)*dvdx(i,j,f)
       t3 = (su*su)*dudy(i,j,f)
       t4 = (su*cu)*dvdy(i,j,f)
       ddv = ((t1 - t2) + t3) + t4
       fdv = (0.5d0*ft(f))*((s*kx) + (c*ky))
       if (sph == 1) then
          ddv = ddv + (((cg(f) + (su*u(i,j,f))) + (cu*v(i,j,f)))*su)*tantr(j)
          fdv = fdv - ((((0.5d0*ft(f))*((c*u(i,j,f)) - (s*v(i,j,f))))*s)*tantr(j))
       end if
       dd_out(i,j,f,d) = ddv
       fd_out(i,j,f,d) = fdv
     end do
    end do
   end do
  end do
end subroutine mkadv_rates


subroutine mkadv_post(fd, dd, fbnd, dbnd, mt, doclip, domask, nx, ny, nf, nd)
  ! after apply_bc: np.clip(fdot,-fbnd,fbnd), np.clip(ddot,-dbnd,dbnd),
  ! then the land mask on both -- one pass, in place
  implicit none
  integer, intent(in) :: doclip, domask, nx, ny, nf, nd
  real*8, intent(in)  :: fbnd(nf), dbnd, mt(nx,ny)
  real*8, intent(inout) :: fd(nx,ny,nf,nd), dd(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = 1, nd
   do f = 1, nf
    do j = 1, ny
     do i = 1, nx
       if (doclip == 1) then
          fd(i,j,f,d) = min(max(fd(i,j,f,d), -fbnd(f)), fbnd(f))
          dd(i,j,f,d) = min(max(dd(i,j,f,d), -dbnd),    dbnd)
       end if
       if (domask == 1) then
          dd(i,j,f,d) = dd(i,j,f,d)*mt(i,j)
          fd(i,j,f,d) = fd(i,j,f,d)*mt(i,j)
       end if
     end do
    end do
   end do
  end do
end subroutine mkadv_post


subroutine efix_terms(a, fx, fy, ff, fd, u, v, om, kf, sth, cth, dft, dd,   &
                      wx, wy, divx, divy, area, mox, moy, do_open, h,       &
                      p, q, dn, w, nx, ny, nf, nd)
  ! one pass producing the 4-D fields of energy_fix_flux:
  !   p = tf + td, q = tx + tx(i-1) + ty + ty(j-1) (np.roll wrap),
  !   dn = wgt*dOf*dd, w = wgt (zeroed at the closed f faces).
  ! Omega recomputed pointwise from the tables (cheaper than storing it);
  ! divx/divy select division (wim_simple /dx) vs multiplication
  ! (wim_realistic *dyt) so both stay bit-identical to their numpy lines.
  implicit none
  integer, intent(in) :: divx, divy, do_open, h, nx, ny, nf, nd
  real*8, intent(in)  :: a(nx,ny,nf,nd), fx(nx,ny,nf,nd), fy(nx,ny,nf,nd)
  real*8, intent(in)  :: ff(nx,ny,nf,nd), fd(nx,ny,nf,nd)
  real*8, intent(in)  :: u(nx,ny,nf), v(nx,ny,nf)
  real*8, intent(in)  :: om(nf), kf(nf), sth(nd), cth(nd), dft(nf), dd
  real*8, intent(in)  :: wx(nx,ny), wy(nx,ny), area(nx,ny), mox(nx,ny), moy(nx,ny)
  real*8, intent(inout) :: p(nx,ny,nf,nd), q(nx,ny,nf,nd), dn(nx,ny,nf,nd), w(nx,ny,nf,nd)
  integer :: i, j, f, d, ip, im, jp, jm, fp, dp
  real*8 :: om0, dof, tf, td, txi, txm, tyj, tym, wgt

  do d = 1, nd
   dp = d+1; if (d == nd) dp = 1
   do f = 1, nf
    fp = f+1; if (f == nf) fp = 1
    do j = 1, ny
     jp = j+1; if (j == ny) jp = 1
     jm = j-1; if (j == 1)  jm = ny
     do i = 1, nx
       ip = i+1; if (i == nx) ip = 1
       im = i-1; if (i == 1)  im = nx

       om0 = omg(i ,j ,f ,d )
       dof = omg(i ,j ,fp,d ) - om0

       txi = flx(fx(i ,j,f,d), omg(ip,j,f,d) - om0, f, wx(i ,j), divx)
       txm = flx(fx(im,j,f,d), om0 - omg(im,j,f,d), f, wx(im,j), divx)
       if (do_open == 1) then
          txi = txi*(1.0d0 - mox(i ,j))
          txm = txm*(1.0d0 - mox(im,j))
       end if
       tyj = flx(fy(i,j ,f,d), omg(i,jp,f,d) - om0, f, wy(i,j ), divy)
       tym = flx(fy(i,jm,f,d), om0 - omg(i,jm,f,d), f, wy(i,jm), divy)
       if (do_open == 1) then
          tyj = tyj*(1.0d0 - moy(i,j ))
          tym = tym*(1.0d0 - moy(i,jm))
       end if

       tf = ((ff(i,j,f,d)*dof)*dd)*area(i,j)
       td = ((fd(i,j,f,d)*(omg(i,j,f,dp) - om0))*dft(f))*area(i,j)

       p(i,j,f,d) = tf + td
       q(i,j,f,d) = ((txi + txm) + tyj) + tym

       wgt = (0.5d0*(a(i,j,f,d) + a(i,j,fp,d)))*dof
       if (f == h .or. f == nf-h) wgt = 0.0d0
       w(i,j,f,d)  = wgt
       dn(i,j,f,d) = ((wgt*dof)*dd)
     end do
    end do
   end do
  end do

contains

  pure function omg(ii, jj, kf_, dd_) result(o)
    ! Omega = omega + kf*(U*s + V*c) at (x=ii, y=jj, f=kf_, theta=dd_)
    integer, intent(in) :: ii, jj, kf_, dd_
    real*8 :: o
    o = om(kf_) + (kf(kf_)*((u(ii,jj,kf_)*sth(dd_)) + (v(ii,jj,kf_)*cth(dd_))))
  end function omg

  pure function flx(fv, dom, kf_, wv, dv) result(t)
    ! per-face defect term: (((F*dOmega)*dft)*dd) / w  or  * w
    integer, intent(in) :: kf_, dv
    real*8, intent(in)  :: fv, dom, wv
    real*8 :: t
    t = ((fv*dom)*dft(kf_))*dd
    if (dv == 1) then
       t = t/wv
    else
       t = t*wv
    end if
  end function flx

end subroutine efix_terms


subroutine efix_apply(ff, w, lam, mt, nx, ny, nf, nd)
  ! ff = ff + (lam*wgt)*maskT, in place
  implicit none
  integer, intent(in) :: nx, ny, nf, nd
  real*8, intent(in)  :: w(nx,ny,nf,nd), lam(nx,ny), mt(nx,ny)
  real*8, intent(inout) :: ff(nx,ny,nf,nd)
  integer :: i, j, f, d
  do d = 1, nd
   do f = 1, nf
    do j = 1, ny
     do i = 1, nx
       ff(i,j,f,d) = ff(i,j,f,d) + ((lam(i,j)*w(i,j,f,d))*mt(i,j))
     end do
    end do
   end do
  end do
end subroutine efix_apply


subroutine coastflux(fx, fy, var, xdot, ydot, mox, moy, pref, nx, ny, nf, nd)
  ! open_coast_flux: first-order upwind replacement on coastal faces,
  ! in place; pref = cosu on the sphere, 1 on the Cartesian grid
  implicit none
  integer, intent(in) :: nx, ny, nf, nd
  real*8, intent(in)  :: var(nx,ny,nf,nd), xdot(nx,ny,nf,nd), ydot(nx,ny,nf,nd)
  real*8, intent(in)  :: mox(nx,ny), moy(nx,ny), pref(ny)
  real*8, intent(inout) :: fx(nx,ny,nf,nd), fy(nx,ny,nf,nd)
  integer :: i, j, f, d, ip, jp
  real*8 :: v0, vn, xd, yd, inner
  do d = 1, nd
   do f = 1, nf
    do j = 1, ny
     jp = j+1; if (j == ny) jp = 1
     do i = 1, nx
       ip = i+1; if (i == nx) ip = 1
       v0 = var(i,j,f,d)
       if (mox(i,j) > 0.0d0) then
          vn = var(ip,j,f,d)
          xd = xdot(i,j,f,d)
          inner = (xd*(vn + v0)) - (abs(xd)*(vn - v0))
          fx(i,j,f,d) = (-inner)*0.5d0
       end if
       if (moy(i,j) > 0.0d0) then
          vn = var(i,jp,f,d)
          yd = ydot(i,j,f,d)
          inner = (yd*(vn + v0)) - (abs(yd)*(vn - v0))
          fy(i,j,f,d) = ((-pref(j))*inner)*0.5d0
       end if
     end do
    end do
   end do
  end do
end subroutine coastflux
