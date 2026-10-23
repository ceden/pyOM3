
!=======================================================================
!  1/3 global with 128 levels, from Icon 20km version
!======================================================================= 

module config_module
 ! use this module only locally in this file
 implicit none
 real*8, allocatable :: t_clim(:,:,:)
 real*8, allocatable :: s_clim(:,:,:)
 real*8, allocatable :: t_rest(:,:,:)
 real*8, allocatable :: s_rest(:,:,:)
 real*8, allocatable :: taux(:,:,:),tauy(:,:,:)
 real*8, allocatable :: divpen_shortwave(:),qsol(:,:,:)
end module config_module
 


subroutine set_parameter
 ! ----------------------------------
 !       set here main parameter
 ! ----------------------------------
 use main_module   
 use config_module
 use tke_module  
 use idemix_module  
 use diagnostics_module   
 implicit none

  nx   = 1080; ny   = 689; nz   = 128
  dt_mom    = 216.0
  dt_tracer = dt_mom

  coord_degree     = .true.
  runlen = 365*86400./4. !dt_tracer
  enable_cyclic_x  = .true.
  
  enable_diag_ts_monitor = .true.; ts_monint = dt_tracer
  enable_diag_snapshots  = .true.; snapint  =  86400.*3
  enable_diag_energy     = .true.; energint = 86400.; energfreq = energint
  enable_diag_averages   = .true.; aveint  = 365*86400./12. ; avefreq=86400.0
  enable_diag_variances  = .true.; varint = 365*86400./4; varfreq = 86400.0
 
  congr_max_iterations = 2000
  congr_epsilon = 1d-11
  enable_streamfunction = .false.
  enable_free_surface = .true.

  enable_conserve_energy             = .true.
  enable_store_cabbeling_heat        = .true.
  enable_take_p_diss_adv_from_tke    = .false.
  enable_store_lateral_friction_heat = .true.
  enable_store_bottom_friction_tke   = .true.
  enable_store_gm_diss_in_idemix     = .false.

  enable_biharmonic_friction  = .true.  
  A_hbi  = 2e11
  biha_friction_cosPower = 1.5
  !enable_biharmonic_thickness_bolus_skew_mom = .true.
  !A_thkbi = A_hbi
  !mld0_thk = 0d0
  enable_superbee_advection = .true.
  
  enable_tempsalt_sources = .true. ! for penetrative solar radiation
 
  enable_implicit_vert_friction = .true.
  enable_tke = .true.
  c_k = 0.1
  c_eps = 0.7
  alpha_tke = 30.0
  mxl_min = 1d-8
  tke_mxl_choice = 2

  enable_idemix = .true.
  enable_idemix_hor_diffusion_iter = .true.;
  idemix_hor_diffusion_iter  = 5

  ! optimized idemix parameter
  jstar=5.
  mu0= 1./3.
  tau_v = 2.*86400
  tau_h=15.0*86400.0

  eq_of_state_type = 5
end subroutine set_parameter


subroutine set_grid
 use main_module   
 use config_module   
 implicit none
 include "netcdf.inc"
 integer :: iret,id,ncid

 if (my_pe==0) print*,' reading grid '
 iret=nf_open('grid_topo.nc',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'dxt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,is_pe, ie_pe-is_pe+1, dxt(is_pe:ie_pe) )
 iret=nf_inq_varid(ncid,'dyt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,js_pe, je_pe-js_pe+1, dyt(js_pe:je_pe) )
 iret=nf_inq_varid(ncid,'dzt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,nz, dzt )

 iret=nf_inq_varid(ncid,'xu',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,1, x_origin )
 iret=nf_inq_varid(ncid,'yu',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,1,1, y_origin )
 iret = nf_close (ncid)
 if (my_pe==0) print*,' x_origin= ',x_origin,' y_origin= ',y_origin
 if (my_pe==0) print*,' done '

end subroutine set_grid


subroutine set_coriolis
 use main_module   
 use config_module   
 implicit none
 integer :: j
 do j=js_pe-onx,je_pe+onx
   coriolis_t(:,j) = 2*omega*sin( yt(j)/180.*pi ) 
 enddo
end subroutine set_coriolis


subroutine set_topography
 use main_module   
 implicit none
 include "netcdf.inc"
 integer :: iret,id,ncid
 
 if (my_pe==0) print*,' reading topography '

 iret=nf_open('grid_topo.nc',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_inq_varid(ncid,'kbot',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_int(ncid,id,(/is_pe,js_pe/),(/i_blk,j_blk/),kbot(is_pe:ie_pe,js_pe:je_pe))
 iret = nf_close (ncid)
end subroutine set_topography


subroutine set_phi
end subroutine set_phi
subroutine set_topographic_spectrum
end subroutine set_topographic_spectrum


subroutine set_initial_conditions
 use main_module   
 use tke_module  
 use idemix_module   
 use config_module   
 implicit none
 include "netcdf.inc"
 integer :: iret,id,ncid
 integer :: i,j,k,n
 real*8 :: cp_0 = 3991.86795711963
 real*8 :: qnet(is_pe:ie_pe,js_pe:je_pe,12) 
 real*8 :: dqdt(is_pe:ie_pe,js_pe:je_pe,12) 
 real*8 :: sst(is_pe:ie_pe,js_pe:je_pe,12) 
 real*8 :: sss(is_pe:ie_pe,js_pe:je_pe,12) 
 real*8 :: lam_s(is_pe:ie_pe,js_pe:je_pe)
 real*8 :: lam_n(is_pe:ie_pe,js_pe:je_pe)

 real*8 :: pen(0:nz),swarg1,swarg2
 real*8 ::  rpart_shortwave  = 0.58
 real*8 ::  efold1_shortwave = 0.35
 real*8 ::  efold2_shortwave = 23.0
 
 allocate( t_clim(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_clim=0.0
 allocate( s_clim(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_clim=0.0
 allocate( t_rest(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); t_rest=0.0
 allocate( s_rest(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); s_rest=0.0
 allocate( taux(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); taux=0.0
 allocate( tauy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); tauy=0.0
 allocate( divpen_shortwave(nz) ); divpen_shortwave=0.0
 allocate( qsol(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,12) ); qsol=0.0
 
 iret=nf_open('initial_conditions.nc',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_inq_varid(ncid,'temp',id); 
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,nz/),temp(is_pe:ie_pe,js_pe:je_pe,1:nz,tau) )
 iret=nf_inq_varid(ncid,'salt',id); 
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,nz/),salt(is_pe:ie_pe,js_pe:je_pe,1:nz,tau) )
 iret = nf_close (ncid) 
 temp(:,:,:,tau) = temp(:,:,:,tau)*maskT
 salt(:,:,:,tau) = salt(:,:,:,tau)*maskT
 temp(:,:,:,taum1) = temp(:,:,:,tau)
 salt(:,:,:,taum1) = salt(:,:,:,tau)


 iret=nf_open('ecmwf_surface_forcing.nc',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 ! wind stress
 iret=nf_inq_varid(ncid,'taux',id);  if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), taux(is_pe:ie_pe,js_pe:je_pe,:) )
 
 iret=nf_inq_varid(ncid,'tauy',id);  if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret=nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), tauy(is_pe:ie_pe,js_pe:je_pe,:) )

 iret=nf_inq_varid(ncid,'qnet',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), qnet )

 iret=nf_inq_varid(ncid,'dqdt',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), dqdt )

 iret=nf_inq_varid(ncid,'swf',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), qsol(is_pe:ie_pe,js_pe:je_pe,:) )
 iret = nf_close (ncid)


 iret=nf_open('woa_surface_forcing.nc',NF_NOWRITE,ncid)
 if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif

 iret=nf_inq_varid(ncid,'sst',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), sst )

 iret=nf_inq_varid(ncid,'sss',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
 iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), sss )
 iret = nf_close (ncid)


 ! rho c_d U^2 = kg/(ms^2) , N = kg m /s^2 ,  N/m^2 = kg/(m s^2)
 do n=1,12 
  taux(:,:,n) = taux(:,:,n)*maskU(:,:,nz) /rho_0
  tauy(:,:,n) = tauy(:,:,n)*maskV(:,:,nz) /rho_0
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,taux(:,:,n))
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,tauy(:,:,n))
 enddo

 !  q = qnet + dqdt*(SST-temp) = dqdt*(SST - temp + qnet/dqdt) 
 where( dqdt /= 0.) SST = SST+qnet/dqdt
 ! dqdt = W/m^2/K , qnet = W/m^2 
 ! W/m^2 = kg m^2 /s^3 /m^2 = kg/s^3 
 ! cp  =  J/(  kg K)  
 ! W = J/s
 ! W/m^2/cp/rho =  m K /s
 do n=1,12
  t_clim(is_pe:ie_pe,js_pe:je_pe,n) = sst(:,:,n)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
  t_rest(is_pe:ie_pe,js_pe:je_pe,n) = dqdt(:,:,n)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)/rho_0 /cp_0
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,t_clim(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,t_clim(:,:,n))
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,t_rest(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,t_rest(:,:,n))
  s_clim(is_pe:ie_pe,js_pe:je_pe,n) = sss(:,:,n)*maskT(is_pe:ie_pe,js_pe:je_pe,nz)
  s_rest(:,:,n) = dzt(nz)/(30.*86400.)*maskT(:,:,nz)
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,s_clim(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,s_clim(:,:,n))
  
  qsol(:,:,n) = qsol(:,:,n)*maskT(:,:,nz)
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,qsol(:,:,n)); 
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,qsol(:,:,n))  
enddo 
 

 if (enable_tke ) then
   tke(:,:,:,tau)   = 1d-12 
   tke(:,:,:,taum1) = 1d-12 
 endif

 if (enable_idemix ) then
 
  iret=nf_open('idemix_surface_forcing.nc',NF_NOWRITE,ncid)
  if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
  iret=nf_inq_varid(ncid,'diss',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
  iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), &
                           forc_iw_surface(is_pe:ie_pe,js_pe:je_pe)  )
  iret = nf_close (ncid)

  iret=nf_open('idemix_bottom_forcing.nc',NF_NOWRITE,ncid)
  if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
  iret=nf_inq_varid(ncid,'diss',id); if (iret /=0 ) then; print*,nf_strerror(iret) ; stop; endif
  iret= nf_get_vara_double(ncid,id,(/is_pe,js_pe,1/),(/i_blk,j_blk,12/), &
                           forc_iw_bottom(is_pe:ie_pe,js_pe:je_pe)  )
  iret = nf_close (ncid)

  do j=js_pe,je_pe
   do i=is_pe,ie_pe
      k=max(1,kbot(i,j))
      forc_iw_bottom(i,j) = forc_iw_bottom(i,j)*maskW(i,j,k)/rho_0
      forc_iw_surface(i,j) = forc_iw_surface(i,j)*maskW(i,j,nz)/rho_0*0.2
   enddo
  enddo

  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc_iw_bottom);
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc_iw_bottom)
  call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc_iw_surface);
  call setcyclic_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc_iw_surface)

  if (enable_idemix3) then
    do j=js_pe,je_pe
     do i=is_pe,ie_pe
      forc_iw_bottom_u(i,j) = 0.5*(forc_iw_bottom(i,j)+forc_iw_bottom(i+1,j))
      forc_iw_bottom_v(i,j) = 0.5*(forc_iw_bottom(i,j)+forc_iw_bottom(i,j+1))
      forc_iw_surface_u(i,j) = 0.5*(forc_iw_surface(i,j)+forc_iw_surface(i+1,j))
      forc_iw_surface_v(i,j) = 0.5*(forc_iw_surface(i,j)+forc_iw_surface(i,j+1))
     enddo
    enddo
   endif

 endif

!     Initialize penetration profile for solar radiation
!     and store divergence in divpen
!     note that pen(nz) is set 0.0 instead of 1.0 to compensate for the
!     shortwave part of the total surface flux 
  pen = 0.0
  do k=1,nz-1
     swarg1 = zw(k)/efold1_shortwave
     swarg2 = zw(k)/efold2_shortwave
     pen(k) = rpart_shortwave*exp(swarg1)+ (1.0-rpart_shortwave)*exp(swarg2)
  enddo
  do k=1,nz
     divpen_shortwave(k) = (pen(k) - pen(k-1))/dzt(k)
  enddo


end subroutine set_initial_conditions




subroutine get_periodic_interval(currentTime,cycleLength,recSpacing,nbrec,trec1,trec2,wght1,wght2)
 ! interpolation routine taken from mitgcm
 implicit none
 real*8, intent(in) :: currentTime,recSpacing,cycleLength
 integer, intent(in) :: nbrec
 real*8, intent(out) :: wght1,wght2
 integer, intent(out) :: tRec1,tRec2
 real*8 :: locTime,tmpTime
 locTime = currentTime - recSpacing*0.5 + cycleLength*( 2 - NINT(currentTime/cycleLength) )
 tmpTime = MOD( locTime, cycleLength )
 tRec1 = 1 + INT( tmpTime/recSpacing )
 tRec2 = 1 + MOD( tRec1, nbRec )
 wght2 = ( tmpTime - recSpacing*(tRec1 - 1) )/recSpacing
 wght1 = 1d0 - wght2
end subroutine



subroutine set_forcing
 use main_module 
 use tke_module   
 use config_module   
 implicit none
 integer :: i,j,k,n1,n2
 real*8 :: f1,f2,fxa,ice(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx), cp_0 = 3991.86795711963  ! J/kg /

 fxa = 365*86400d0
 call get_periodic_interval((itt-1)*dt_tracer,fxa,fxa/12.,12,n1,n2,f1,f2)

 do j=js_pe-onx,je_pe+onx
  do i=is_pe-onx,ie_pe+onx
    surface_taux(i,j)=(f1*taux(i,j,n1)+f2*taux(i,j,n2))*maskU(i,j,nz)
    surface_tauy(i,j)=(f1*tauy(i,j,n1)+f2*tauy(i,j,n2))*maskV(i,j,nz)
  enddo
 enddo

 !  q = qnet + dqdt*(SST-temp) = dqdt*(T_clim + qnet/dqdt - t(k=1) ) 
 ! heat flux = R*(T^* - t(k=1)), positive upward
 ice=1.0
 do j=js_pe-onx,je_pe +onx
  do i=is_pe-onx,ie_pe+onx
   forc_temp_surface(i,j)=(f1*t_rest(i,j,n1)+f2*t_rest(i,j,n2))* &
                          (f1*t_clim(i,j,n1)+f2*t_clim(i,j,n2)-temp(i,j,nz,tau))*maskT(i,j,nz)
   forc_salt_surface(i,j)=(f1*s_rest(i,j,n1)+f2*s_rest(i,j,n2))* &
                          (f1*s_clim(i,j,n1)+f2*s_clim(i,j,n2)-salt(i,j,nz,tau))*maskT(i,j,nz)
   if (temp(i,j,nz,tau)*maskT(i,j,nz) <= -1.8 .and. forc_temp_surface(i,j) <=0) then ! apply simple ice mask
       forc_temp_surface(i,j) = 0.0
       forc_salt_surface(i,j) = 0.0
       ice(i,j)               = 0.0
   endif
  enddo
 enddo

 if (enable_tke ) then
  do j=js_pe-onx+1,je_pe+onx
   do i=is_pe-onx+1,ie_pe+onx
     forc_tke_surface(i,j) = sqrt( (0.5*(surface_taux(i,j)+surface_taux(i-1,j)))**2  &
                                  +(0.5*(surface_tauy(i,j)+surface_tauy(i,j-1)))**2 )**(3./2.) 
   enddo
  enddo
 endif

 ! solar radiation
 do k=1,nz
  do j=js_pe,je_pe 
   do i=is_pe,ie_pe
    temp_source(i,j,k) =  (f1*qsol(i,j,n1)+f2*qsol(i,j,n2))*divpen_shortwave(k)*ice(i,j)*maskT(i,j,k)/cp_0/rho_0
   enddo
  enddo
 enddo

end subroutine set_forcing



subroutine set_diagnostics
end subroutine set_diagnostics


subroutine set_particles
end subroutine set_particles






