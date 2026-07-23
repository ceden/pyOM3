
! legacy fortran code for DIA is from WAM github repository
! mostly from
! https://github.com/myWAveModel/WAM/blob/WAM_Cycle7/src/mod/wam_source_module.f90
! and
! https://github.com/myWAveModel/WAM/blob/WAM_Cycle7/src/mod/wam_fre_dir_module.f90
! https://github.com/myWAveModel/WAM/blob/WAM_Cycle7/src/mod/wam_general_module.f90
! check for credits in original files, some go back to S. Hasselmann 

module Snonlin_module
  implicit none

  real(8), parameter :: g=9.81
  real(8), parameter :: rowater = 1000. 
  !REAL(8), PARAMETER :: CO = 1.1       !! FREQUENCY RATIO.
  REAL(8), PARAMETER :: PI = 3.1415927 
  REAL(8), PARAMETER :: DEG = 180./PI

  INTEGER            :: nx=-1, ny=-1   !! number of spatial grid points
  INTEGER            :: ML = -1        !! NUMBER OF FREQUENCIES
  INTEGER            :: KL = -1        !! NUMBER OF DIRECTIONS
  REAL(8)            :: DELTH = -1     !! ANGULAR INCREMENT OF SPECTRUM [RAD].

  REAL(8),ALLOCATABLE :: FR(:)        !! FREQUENCIES [HZ].
  REAL(8)             :: CO
  INTEGER, PARAMETER :: NINL = 5  !! SIZE OF INLCOEF
  INTEGER, PARAMETER :: NRNL = 25 !! SIZE OF RNLCOEF

  INTEGER :: KFRH     !! SIZE OF FRH
  INTEGER :: MFRSTLW  !! INDEX OF FIRST EXTRA LOW FREQUENCY FOR SNL
  INTEGER :: MLSTHG   !! INDEX OF LAST EXTRA HIGH FREQUENCY FOR SNL

  INTEGER,ALLOCATABLE, DIMENSION(:)   :: IKP    !! FREQUENCY INDEX FOR STORING
                                                !! ENERGY TRANSFER INCREMENTS
                                                !! INTO BINS, WAVE NO. 3.
  INTEGER,ALLOCATABLE, DIMENSION(:)   :: IKP1   !! IKP+1. 
  INTEGER,ALLOCATABLE, DIMENSION(:)   :: IKM    !! FREQUENCY INDEX FOR STORING
                                                !! ENERGY TRANSFER INCREMENTS
                                                !! INTO BINS, WAVE NO. 4. 
  INTEGER,ALLOCATABLE, DIMENSION(:)   :: IKM1   !! IKM+1
  INTEGER,ALLOCATABLE, DIMENSION(:,:) :: K1W    !! ANGULAR INDEX FOR STORING
                                                !! ENERGY TRANSFER INCREMENTS
                                                !! INTO BINS, WAVE NO. 3.
  INTEGER,ALLOCATABLE, DIMENSION(:,:) :: K2W    !! ANGULAR INDEX FOR STORING
                                                !! ENERGY TRANSFER INCREMENTS
                                                !! INTO BINS, WAVE NO. 4.
  INTEGER,ALLOCATABLE, DIMENSION(:,:) :: K11W   !! K1W(.,1)-1, K1W(.,2)+1.
  INTEGER,ALLOCATABLE, DIMENSION(:,:) :: K21W   !! K2W(.,1)+1, K2W(.,2)-1.
  INTEGER,ALLOCATABLE, DIMENSION(:,:) :: INLCOEF  !! STORES ALL FREQUENCY DEPENDENT
                                                  !! INDICES FOUND IN SNL

  REAL(8),   ALLOCATABLE, DIMENSION(:)   :: AF11   !! WEIGHTS FOR APPROXIMATION
                                                   !! OF NONL. TRANSFER (ONE 
                                                   !! TERM ONLY SET TO 3000). 
                                                   !! MULTIPLIED BY FREQ. **11.
  REAL(8),   ALLOCATABLE, DIMENSION(:)   :: FKLAP  !! WEIGHT IN FREQUENCY GRID  
                                                   !! FOR INTERPOLATION, WAVE  
                                                   !! NO. 3 ("1+LAMBDA" TERM)
  REAL(8),   ALLOCATABLE, DIMENSION(:)   :: FKLAP1 !! 1-FKLAP.
  REAL(8),   ALLOCATABLE, DIMENSION(:)   :: FKLAM  !! WEIGHT IN FREQUENCY GRID  
                                                   !! FOR INTERPOLATION, WAVE 
                                                   !! NO. 4 ("1-LAMBDA" TERM).
  REAL(8),   ALLOCATABLE, DIMENSION(:)   :: FKLAM1 !! 

  REAL(8), ALLOCATABLE, DIMENSION(:)     :: FRH    !! TAIL FREQUENCY RATION **5
  REAL(8), ALLOCATABLE, DIMENSION(:)     :: FTRF   !! FRONT TAIL REDUCTION FACTOR 
                                                   !! USED TO A SPECTRAL TAIL IN FRONT
                                                   !! OF THE FIRST DISCRETISED FREQUENCY
  REAL(8), ALLOCATABLE, DIMENSION(:,:)   :: RNLCOEF !! STORES ALL FREQUENCY DEPENDENT
                                                    !! COEFFICIENT FOUND IN SNL

  REAL(8)    :: ACL1        !! WEIGHT IN ANGULAR GRID FOR INTERPOLATION, WAVE NO. 3 ("1+LAMBDA" TERM).
  REAL(8)    :: ACL2        !! WEIGHT IN ANGULAR GRID FOR INTERPOLATION, WAVE NO. 4 ("1-LAMBDA" TERM).
  REAL(8)    :: CL11        !! 1.-ACL1.
  REAL(8)    :: CL21        !! 1.-ACL2.
  REAL(8)    :: DAL1        !! 1./ACL1.
  REAL(8)    :: DAL2        !! 1./ACL2.

end module

subroutine setup(nx_in,ny_in, KL_in, ml_in,fr_in)
  use Snonlin_module
  implicit none
  INTEGER , INTENT(IN) :: nx_in,ny_in !! number of grid points
  INTEGER , INTENT(IN) :: KL_in       !! NUMBER OF DIRECTIONS.
  INTEGER , INTENT(IN) :: ml_in       !! NUMBER OF FREQUENCIES.
!f2py depend(fr_in) :: ml_in = shape(fr_in,0),  
  REAL(8) , INTENT(IN) :: fr_in(ML_in) !! REFERENCE FREQUENCY [HZ].
  
  nx = nx_in
  ny = ny_in
  ML = ML_in
  KL = KL_in

  DELTH = 2.d0*PI/KL 
  
  if (allocated(fr)) deallocate(FR); ALLOCATE (FR(ML))
  ! WAM way to construct grid
  !FR(1) = CO**(-IREF+1)*FR1
  !DO M = 2,ML
  ! FR(M) = CO*FR(M-1)
  !END DO
  FR = FR_in
  CO = FR(2)/FR(1) ! = 1.1
  
  call nlweigt
  call init_snonlin
end subroutine setup



subroutine source(F, SL, FL, n1, n2, n3,n4)
  use Snonlin_module
  implicit none
  integer,  intent(in)  :: n1, n2, n3, n4
! n1 is nx, n2 is ny,  n3 is KL NUMBER OF DIRECTIONS, n4 is ML is NUMBER OF FREQUENCIES
!f2py intent(hide) :: n1, n2, n3,n4          ! f2py infers from array shape
!f2py depend(F) :: n1 = shape(F,0), n2 = shape(F,1), n3 = shape(F,2), n4 = shape(F,3)
  real(8),  intent(in)  :: F(n1, n2, n3,n4)
  real(8),  intent(out) :: SL(n1, n2, n3,n4)
  real(8),  intent(out) :: FL(n1, n2, n3,n4)

! ---------------------------------------------------------------------------- !
!                                                                              !
!     LOCAL VARIABLES.                                                         !
!     ----------------                                                         !

INTEGER :: MC, MP, MP1, MM, MM1, IC, IP, IP1, IM, IM1, KH, K, K1, K2, K11, K21
INTEGER :: ii,jj
INTEGER :: MFR1STFR, MFRLSTFR
REAL(8) :: FTAIL, FKLAMP, FKLAMP1, GW1, GW2, GW3, GW4
REAL(8) :: FKLAMPA, FKLAMPB, FKLAMP2, FKLAPA2, FKLAPB2, FKLAP12, FKLAP22
REAL(8) :: FKLAMM, FKLAMM1, GW5, GW6, GW7, GW8, FKLAMMA, FKLAMMB, FKLAMM2
REAL(8) :: FKLAMA2, FKLAMB2, FKLAM12, FKLAM22
REAL(8) :: SAP, SAM, FIJ, FAD1, FAD2, FCEN
REAL(8), DIMENSION(nx,ny) :: FTEMP, AD, DELAD, DELAP, DELAM


SL = 0. 
FL = 0.
! ---------------------------------------------------------------------------- !
!                                                                              !
!     2. FREQUENCY LOOP.                                                       !
!        ---------------                                                       !

MFR1STFR = -MFRSTLW+1
MFRLSTFR = ML-KFRH+MFR1STFR

FRE: DO MC = 1,MLSTHG
   MP  = IKP (MC)
   MP1 = IKP1(MC)
   MM  = IKM (MC)
   MM1 = IKM1(MC)
   IC  = INLCOEF(1,MC)
   IP  = INLCOEF(2,MC)
   IP1 = INLCOEF(3,MC)
   IM  = INLCOEF(4,MC)
   IM1 = INLCOEF(5,MC)

   FTAIL  = RNLCOEF(1,MC)

   FKLAMP  = FKLAP(MC)
   FKLAMP1 = FKLAP1(MC)
   GW1 = RNLCOEF(2,MC)
   GW2 = RNLCOEF(3,MC)
   GW3 = RNLCOEF(4,MC)
   GW4 = RNLCOEF(5,MC)
   FKLAMPA = RNLCOEF(6,MC)
   FKLAMPB = RNLCOEF(7,MC)
   FKLAMP2 = RNLCOEF(8,MC)
   FKLAMP1 = RNLCOEF(9,MC)
   FKLAPA2 = RNLCOEF(10,MC)
   FKLAPB2 = RNLCOEF(11,MC)
   FKLAP12 = RNLCOEF(12,MC)
   FKLAP22 = RNLCOEF(13,MC)

   FKLAMM  = FKLAM(MC)
   FKLAMM1 = FKLAM1(MC)
   GW5 = RNLCOEF(14,MC)
   GW6 = RNLCOEF(15,MC)
   GW7 = RNLCOEF(16,MC)
   GW8 = RNLCOEF(17,MC)
   FKLAMMA = RNLCOEF(18,MC)
   FKLAMMB = RNLCOEF(19,MC)
   FKLAMM2 = RNLCOEF(20,MC)
   FKLAMM1 = RNLCOEF(21,MC)
   FKLAMA2 = RNLCOEF(22,MC)
   FKLAMB2 = RNLCOEF(23,MC)
   FKLAM12 = RNLCOEF(24,MC)
   FKLAM22 = RNLCOEF(25,MC)


   FTEMP(:,:) = AF11(MC)


   IF (MC.GT.MFR1STFR .AND. MC.LT.MFRLSTFR ) THEN
!       MC is within the fully resolved spectral domain

      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

!*    2.1.1.1 LOOP OVER GRIDPOINTS.. NONLINEAR TRANSFER AND
!*            DIAGONAL MATRIX OF FUNCTIONAL DERIVATIVE.
!             ----------------------------------------------

            DO ii = 1,nx
             do jj = 1,ny          
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP )                     &
&                  + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM )                     &
&                  + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FAD1 = FAD1+FAD2
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN
               DELAD(ii,jj) = FAD1*FTEMP(ii,jj)
               DELAP(ii,jj) = (FIJ-2.*SAM)*DAL1*FCEN
               DELAM(ii,jj) = (FIJ-2.*SAP)*DAL2*FCEN
             ENDDO
            ENDDO

            SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)
            FL(:,:,K  ,MC ) = FL(:,:,K  ,MC ) - 2.*DELAD(:,:)
            SL(:,:,K2 ,MM ) = SL(:,:,K2 ,MM ) + AD(:,:)*FKLAMM1
            FL(:,:,K2 ,MM ) = FL(:,:,K2 ,MM ) + DELAM(:,:)*FKLAM12
            SL(:,:,K21,MM ) = SL(:,:,K21,MM ) + AD(:,:)*FKLAMM2
            FL(:,:,K21,MM ) = FL(:,:,K21,MM ) + DELAM(:,:)*FKLAM22
            SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
            FL(:,:,K2 ,MM1) = FL(:,:,K2 ,MM1) + DELAM(:,:)*FKLAMA2
            SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB
            FL(:,:,K21,MM1) = FL(:,:,K21,MM1) + DELAM(:,:)*FKLAMB2
            SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
            FL(:,:,K1 ,MP ) = FL(:,:,K1 ,MP ) + DELAP(:,:)*FKLAP12
            SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2
            FL(:,:,K11,MP ) = FL(:,:,K11,MP ) + DELAP(:,:)*FKLAP22
            SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
            FL(:,:,K1 ,MP1) = FL(:,:,K1 ,MP1) + DELAP(:,:)*FKLAPA2
            SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
            FL(:,:,K11,MP1) = FL(:,:,K11,MP1) + DELAP(:,:)*FKLAPB2
         ENDDO
      ENDDO

   ELSEIF (MC.GE.MFRLSTFR ) THEN
      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

            DO ii = 1,nx
             do jj = 1,ny
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP )                     &
&                  + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM )                     &
&                  + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )*FTAIL
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FAD1 = FAD1+FAD2
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN
               DELAD(ii,jj) = FAD1*FTEMP(ii,jj)
               DELAP(ii,jj) = (FIJ-2.*SAM)*DAL1*FCEN
               DELAM(ii,jj) = (FIJ-2.*SAP)*DAL2*FCEN
             ENDDO
            ENDDO
            SL(:,:,K2 ,MM ) = SL(:,:,K2 ,MM ) + AD(:,:)*FKLAMM1
            FL(:,:,K2 ,MM ) = FL(:,:,K2 ,MM ) + DELAM(:,:)*FKLAM12
            SL(:,:,K21,MM ) = SL(:,:,K21,MM ) + AD(:,:)*FKLAMM2
            FL(:,:,K21,MM ) = FL(:,:,K21,MM ) + DELAM(:,:)*FKLAM22

            IF (MM1.LE.ML) THEN
               SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
               FL(:,:,K2 ,MM1) = FL(:,:,K2 ,MM1) + DELAM(:,:)*FKLAMA2
               SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB
               FL(:,:,K21,MM1) = FL(:,:,K21,MM1) + DELAM(:,:)*FKLAMB2

               IF (MC .LE.ML) THEN
                  SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)
                  FL(:,:,K  ,MC ) = FL(:,:,K  ,MC ) - 2.*DELAD(:,:)

                  IF (MP .LE.ML) THEN
                     SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
                     FL(:,:,K1 ,MP ) = FL(:,:,K1 ,MP ) + DELAP(:,:)*FKLAP12
                     SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2
                     FL(:,:,K11,MP ) = FL(:,:,K11,MP ) + DELAP(:,:)*FKLAP22

                     IF (MP1.LE.ML) THEN
                        SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
                        FL(:,:,K1 ,MP1) = FL(:,:,K1 ,MP1) + DELAP(:,:)*FKLAPA2
                        SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
                        FL(:,:,K11,MP1) = FL(:,:,K11,MP1) + DELAP(:,:)*FKLAPB2
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDDO
      ENDDO

   ELSE

      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

            DO ii = 1,nx
             do jj = 1,ny
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP )     &
&                  + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM )     &
&                  + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )*FTAIL
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FAD1 = FAD1+FAD2
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN
               DELAD(ii,jj) = FAD1*FTEMP(ii,jj)
               DELAP(ii,jj) = (FIJ-2.*SAM)*DAL1*FCEN
               DELAM(ii,jj) = (FIJ-2.*SAP)*DAL2*FCEN
             ENDDO
            ENDDO 
            IF (MM1.GE.1) THEN
               SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
               FL(:,:,K2 ,MM1) = FL(:,:,K2 ,MM1) + DELAM(:,:)*FKLAMA2
               SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB
               FL(:,:,K21,MM1) = FL(:,:,K21,MM1) + DELAM(:,:)*FKLAMB2
            ENDIF

            SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)
            FL(:,:,K  ,MC ) = FL(:,:,K  ,MC ) - 2.*DELAD(:,:)
            SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
            FL(:,:,K1 ,MP ) = FL(:,:,K1 ,MP ) + DELAP(:,:)*FKLAP12
            SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2
            FL(:,:,K11,MP ) = FL(:,:,K11,MP ) + DELAP(:,:)*FKLAP22
            SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
            FL(:,:,K1 ,MP1) = FL(:,:,K1 ,MP1) + DELAP(:,:)*FKLAPA2
            SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
            FL(:,:,K11,MP1) = FL(:,:,K11,MP1) + DELAP(:,:)*FKLAPB2
         ENDDO
      ENDDO

   ENDIF
END DO FRE                  !! BRANCH BACK FOR NEXT FREQUENCY.


END SUBROUTINE source


 

subroutine source_nofl(F, SL, n1, n2, n3,n4)
  use Snonlin_module
  implicit none
  integer,  intent(in)  :: n1, n2, n3, n4
! n1 is nx, n2 is ny,  n3 is KL NUMBER OF DIRECTIONS, n4 is ML is NUMBER OF FREQUENCIES
!f2py intent(hide) :: n1, n2, n3,n4          ! f2py infers from array shape
!f2py depend(F) :: n1 = shape(F,0), n2 = shape(F,1), n3 = shape(F,2), n4 = shape(F,3)
  real(8),  intent(in)  :: F(n1, n2, n3,n4)
  real(8),  intent(out) :: SL(n1, n2, n3,n4)
  
  INTEGER :: MC, MP, MP1, MM, MM1, IC, IP, IP1, IM, IM1, KH, K, K1, K2, K11, K21
  INTEGER :: MFR1STFR, MFRLSTFR
  REAL(8)    :: FTAIL, FKLAMP, FKLAMP1, GW1, GW2, GW3, GW4
  REAL(8)    :: FKLAMPA, FKLAMPB, FKLAMP2, FKLAPA2, FKLAPB2, FKLAP12, FKLAP22
  REAL(8)    :: FKLAMM, FKLAMM1, GW5, GW6, GW7, GW8, FKLAMMA, FKLAMMB, FKLAMM2
  REAL(8)    :: FKLAMA2, FKLAMB2, FKLAM12, FKLAM22
  REAL(8)    :: SAP, SAM, FIJ, FAD1, FAD2, FCEN
  REAL(8), DIMENSION(nx,ny) :: FTEMP, AD
  integer :: ii,jj
  
  SL=0
  MFR1STFR = -MFRSTLW+1
  MFRLSTFR = ML-KFRH+MFR1STFR

  DO MC = 1,MLSTHG
   MP  = IKP (MC)
   MP1 = IKP1(MC)
   MM  = IKM (MC)
   MM1 = IKM1(MC)
   IC  = INLCOEF(1,MC)
   IP  = INLCOEF(2,MC)
   IP1 = INLCOEF(3,MC)
   IM  = INLCOEF(4,MC)
   IM1 = INLCOEF(5,MC)

   FTAIL  = RNLCOEF(1,MC)

   FKLAMP  = FKLAP(MC)
   FKLAMP1 = FKLAP1(MC)
   GW1 = RNLCOEF(2,MC)
   GW2 = RNLCOEF(3,MC)
   GW3 = RNLCOEF(4,MC)
   GW4 = RNLCOEF(5,MC)
   FKLAMPA = RNLCOEF(6,MC)
   FKLAMPB = RNLCOEF(7,MC)
   FKLAMP2 = RNLCOEF(8,MC)
   FKLAMP1 = RNLCOEF(9,MC)
   FKLAPA2 = RNLCOEF(10,MC)
   FKLAPB2 = RNLCOEF(11,MC)
   FKLAP12 = RNLCOEF(12,MC)
   FKLAP22 = RNLCOEF(13,MC)

   FKLAMM  = FKLAM(MC)
   FKLAMM1 = FKLAM1(MC)
   GW5 = RNLCOEF(14,MC)
   GW6 = RNLCOEF(15,MC)
   GW7 = RNLCOEF(16,MC)
   GW8 = RNLCOEF(17,MC)
   FKLAMMA = RNLCOEF(18,MC)
   FKLAMMB = RNLCOEF(19,MC)
   FKLAMM2 = RNLCOEF(20,MC)
   FKLAMM1 = RNLCOEF(21,MC)
   FKLAMA2 = RNLCOEF(22,MC)
   FKLAMB2 = RNLCOEF(23,MC)
   FKLAM12 = RNLCOEF(24,MC)
   FKLAM22 = RNLCOEF(25,MC)

   FTEMP(:,:) = AF11(MC)
   
   IF (MC.GT.MFR1STFR .AND. MC.LT.MFRLSTFR ) THEN
!       MC is within the fully resolved spectral domain

      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

!*    2.1.1.1 LOOP OVER GRIDPOINTS.. NONLINEAR TRANSFER AND
!*            DIAGONAL MATRIX OF FUNCTIONAL DERIVATIVE.
!             ----------------------------------------------

            DO ii = 1,nx
             do jj = 1,ny
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP ) &
                   + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM ) &
                   + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN               
             ENDDO  
            ENDDO
   
            SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)
            SL(:,:,K2 ,MM ) = SL(:,:,K2 ,MM ) + AD(:,:)*FKLAMM1
            SL(:,:,K21,MM ) = SL(:,:,K21,MM ) + AD(:,:)*FKLAMM2
            SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
            SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB
            SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
            SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2
            SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
            SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
         ENDDO
      ENDDO

   ELSEIF (MC.GE.MFRLSTFR ) THEN
      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

            DO ii = 1,nx
             do jj = 1,ny
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP )  &
                   + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM )  &
                   + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )*FTAIL
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN               
            ENDDO   
           ENDDO

            SL(:,:,K2 ,MM ) = SL(:,:,K2 ,MM ) + AD(:,:)*FKLAMM1
            SL(:,:,K21,MM ) = SL(:,:,K21,MM ) + AD(:,:)*FKLAMM2

            IF (MM1.LE.ML) THEN
               SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
               SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB

               IF (MC .LE.ML) THEN
                  SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)

                  IF (MP .LE.ML) THEN
                     SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
                     SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2

                     IF (MP1.LE.ML) THEN
                        SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
                        SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDDO
      ENDDO

   ELSE

      DO KH=1,2
         DO K=1,KL
            K1  = K1W (K,KH)
            K2  = K2W (K,KH)
            K11 = K11W(K,KH)
            K21 = K21W(K,KH)

            DO ii = 1,nx
             do jj = 1,ny            
               SAP = GW1*F(ii,jj,K1 ,IP ) + GW2*F(ii,jj,K11,IP )  &
                   + GW3*F(ii,jj,K1 ,IP1) + GW4*F(ii,jj,K11,IP1)
               SAM = GW5*F(ii,jj,K2 ,IM ) + GW6*F(ii,jj,K21,IM ) & 
                   + GW7*F(ii,jj,K2 ,IM1) + GW8*F(ii,jj,K21,IM1)
               FIJ = F(ii,jj,K  ,IC )*FTAIL
               FAD1 = FIJ*(SAP+SAM)
               FAD2 = FAD1-2.*SAP*SAM
               FCEN = FTEMP(ii,jj)*FIJ
               AD(ii,jj) = FAD2*FCEN               
             ENDDO
            ENDDO
            IF (MM1.GE.1) THEN
               SL(:,:,K2 ,MM1) = SL(:,:,K2 ,MM1) + AD(:,:)*FKLAMMA
               SL(:,:,K21,MM1) = SL(:,:,K21,MM1) + AD(:,:)*FKLAMMB
            ENDIF

            SL(:,:,K  ,MC ) = SL(:,:,K  ,MC ) - 2.*AD(:,:)
            SL(:,:,K1 ,MP ) = SL(:,:,K1 ,MP ) + AD(:,:)*FKLAMP1
            SL(:,:,K11,MP ) = SL(:,:,K11,MP ) + AD(:,:)*FKLAMP2
            SL(:,:,K1 ,MP1) = SL(:,:,K1 ,MP1) + AD(:,:)*FKLAMPA
            SL(:,:,K11,MP1) = SL(:,:,K11,MP1) + AD(:,:)*FKLAMPB
         ENDDO
      ENDDO

   ENDIF
  END DO   
end subroutine source_nofl






SUBROUTINE NLWEIGT
  use Snonlin_module     
  implicit none
                                                 !
  REAL(8), PARAMETER :: ALAMD   = 0.25     !! LAMBDA
  REAL(8), PARAMETER :: CON     = 3000.    !! WEIGHT FOR DISCRETE APPROXIMATION OF NONLINEAR TRANSFER
  INTEGER, EXTERNAL :: JAFU
  INTEGER :: KLP1, IC, KH, KLH, K, KS, ISG, K1, K11, K2, K21, M, IKN, I, ISP, ISM
  INTEGER, ALLOCATABLE :: JA1(:,:), JA2(:,:)
  REAL(8)    :: DELPHI1,DELPHI2,DELTHA, CL1, CL2, AL11, AL12, CH, CL1H, CL2H
  REAL(8)    :: F1P1, FRG, FLP, FLM, FKP, FKM, XF, COSTH3, COSTH4
  REAL(8),    ALLOCATABLE :: FRLON(:)

  F1P1 = LOG10(CO)
  ISP = INT(LOG10(1.+ALAMD)/F1P1+.000001)
  ISM = FLOOR(LOG10(1.-ALAMD)/F1P1+.0000001)
  MFRSTLW = 1+ISM
  MLSTHG = ML-ISM
  KFRH=-ISM+ISP+2

  if (allocated(JA1)) deallocate(JA1);     ALLOCATE(JA1(KL,2))
  if (allocated(JA2)) deallocate(JA2);     ALLOCATE(JA2(KL,2))
  if (allocated(FRLON)) deallocate(FRLON); ALLOCATE(FRLON(MFRSTLW:ML+KFRH))
  if (allocated(IKP)) deallocate(IKP);   ALLOCATE (IKP (MFRSTLW:MLSTHG))
  if (allocated(IKP1)) deallocate(IKP1); ALLOCATE (IKP1(MFRSTLW:MLSTHG))
  if (allocated(IKM)) deallocate(IKM);   ALLOCATE (IKM (MFRSTLW:MLSTHG))
  if (allocated(IKM1)) deallocate(IKM1); ALLOCATE (IKM1(MFRSTLW:MLSTHG))
  if (allocated(K1W)) deallocate(K1W);   ALLOCATE (K1W (KL,2))
  if (allocated(K2W)) deallocate(K2W);   ALLOCATE (K2W (KL,2))
  if (allocated(K11W)) deallocate(K11W); ALLOCATE (K11W(KL,2))
  if (allocated(K21W)) deallocate(K21W); ALLOCATE (K21W(KL,2))
  if (allocated(AF11)) deallocate(AF11); ALLOCATE (AF11(MFRSTLW:MLSTHG))
  if (allocated(FKLAP)) deallocate(FKLAP);   ALLOCATE (FKLAP(MFRSTLW:MLSTHG))
  if (allocated(FKLAP1)) deallocate(FKLAP1); ALLOCATE (FKLAP1(MFRSTLW:MLSTHG))
  if (allocated(FKLAM)) deallocate(FKLAM);   ALLOCATE (FKLAM(MFRSTLW:MLSTHG))
  if (allocated(FKLAM1)) deallocate(FKLAM1); ALLOCATE (FKLAM1(MFRSTLW:MLSTHG))
  if (allocated(FRH)) deallocate(FRH);       ALLOCATE(FRH(KFRH))

! ---------------------------------------------------------------------------- !
!                                                                              !
!     1. COMPUTATION FOR ANGULAR GRID.                                         !
!        -----------------------------                                         !

  XF      = ((1.+ALAMD)/(1.-ALAMD))**4
  COSTH3  = (1.+2.*ALAMD+2.*ALAMD**3)/(1.+ALAMD)**2
  DELPHI1 = -180./PI*ACOS(COSTH3)
  COSTH4  = SQRT(1.-XF+XF*COSTH3**2)
  DELPHI2 = 180./PI*ACOS(COSTH4)

  DELTHA = DELTH*DEG
  CL1 = DELPHI1/DELTHA
  CL2 = DELPHI2/DELTHA

!     1.1 COMPUTATION OF INDICES OF ANGULAR CELL.                              !
!         ---------------------------------------                              !

  KLP1 = KL+1
  IC = 1

  DO KH = 1,2
   KLH = KL
   IF (KH.EQ.2) KLH=KLP1
   DO K = 1,KLH
      KS = K
      IF (KH.GT.1) KS=KLP1-K+1
      IF (KS.GT.KL) CYCLE
      CH = IC*CL1
      JA1(KS,KH) = JAFU(CH,K,KL)
      CH = IC*CL2
      JA2(KS,KH) = JAFU(CH,K,KL)
   END DO
   IC = -1
  END DO

!     1.2 COMPUTATION OF ANGULAR WEIGHTS.                                      !
!         -------------------------------                                      !

  CL1  = CL1-INT(CL1)
  CL2  = CL2-INT(CL2)
  ACL1 = ABS(CL1)
  ACL2 = ABS(CL2)
  CL11 = 1.-ACL1
  CL21 = 1.-ACL2
  AL11 = (1.+ALAMD)**4
  AL12 = (1.-ALAMD)**4
  DAL1 = 1./AL11
  DAL2 = 1./AL12

!     1.3 COMPUTATION OF ANGULAR INDICES.                                      !
!         -------------------------------                                      !

  ISG = 1
  DO KH = 1,2
   CL1H = ISG*CL1
   CL2H = ISG*CL2
   DO K = 1,KL
      KS = K
      IF (KH.EQ.2) KS = KL-K+2
      IF(K.EQ.1) KS = 1
      K1 = JA1(K,KH)
      K1W(KS,KH) = K1
      IF (CL1H.LT.0.) THEN
         K11 = K1-1
         IF (K11.LT.1) K11 = KL
      ELSE
         K11 = K1+1
         IF (K11.GT.KL) K11 = 1
      END IF
      K11W(KS,KH) = K11
      K2 = JA2(K,KH)
      K2W(KS,KH) = K2
      IF (CL2H.LT.0) THEN
         K21 = K2-1
         IF(K21.LT.1) K21 = KL
      ELSE
         K21 = K2+1
         IF (K21.GT.KL) K21 = 1
      END IF
      K21W(KS,KH) = K21
   END DO
   ISG = -1
  END DO

! ---------------------------------------------------------------------------- !
!                                                                              !
!     2. COMPUTATION FOR FREQUENCY GRID.                                       !
!        -------------------------------                                       !

  FRLON(1:ML) = FR(1:ML)

  DO M=0,MFRSTLW,-1
   FRLON(M)=FRLON(M+1)/CO
  ENDDO
  DO M=ML+1,ML+KFRH
   FRLON(M) = CO*FRLON(M-1)
  ENDDO

  DO M = MFRSTLW,MLSTHG
   FRG = FRLON(M)
   AF11(M) = CON * FRG**11
   
   FLP = FRG*(1.+ALAMD)
   FLM = FRG*(1.-ALAMD)
   IKN = M+ISP
   IKP(M) = IKN
   FKP = FRLON(IKP(M))
   IKP1(M) = IKP(M)+1
   FKLAP(M) = (FLP-FKP)/(FRLON(IKP1(M))-FKP)
   FKLAP1(M) = 1.-FKLAP(M)
   IF (FRLON(MFRSTLW).GE.FLM) THEN
      IKM(M) = 1
      IKM1(M) = 1
      FKLAM(M) = 0.
      FKLAM1(M) = 0.
   ELSE
      IKN = M+ISM
      IKM(M) = IKN
      FKM = FRLON(IKM(M))
      IKM1(M) = IKM(M)+1
      FKLAM(M) = (FLM-FKM)/(FRLON(IKM1(M))-FKM)
      FKLAM1(M) = 1.-FKLAM(M)
      IF (IKN.LT.MFRSTLW) THEN
         IKM(M) = 1
         FKLAM1(M) = 0.
      ENDIF
   END IF
  END DO

! ---------------------------------------------------------------------------- !
!                                                                              !
!     3. COMPUTE TAIL FREQUENCY RATIOS.                                        !
!        ------------------------------                                        !

  DO I=1,KFRH
   M = ML+I-1
   FRH(I) = (FRLON(ML)/FRLON(M))**5
  END DO

END SUBROUTINE NLWEIGT






SUBROUTINE INIT_SNONLIN
  use Snonlin_module  
  implicit none

  INTEGER :: ICOUNT=1, IRCOUNT=1, MC, MP, MP1, MM, MM1, IC, IP, IP1, IM , IM1, ITEMP
  REAL(8) :: ALPH, FRR
  REAL(8) :: FFACP, FFACP1, FFACM, FFACM1, FTAIL, FKLAMP, FKLAMP1
  REAL(8) :: FKLAMPA, FKLAMPB, FKLAMP2, FKLAPA2, FKLAPB2
  REAL(8) :: FKLAP12, FKLAP22, FKLAMM, FKLAMM1, FKLAMMA, FKLAMMB
  REAL(8) :: FKLAMM2, FKLAMA2, FKLAMB2, FKLAM12, FKLAM22
  REAL(8) :: GW1, GW2, GW3, GW4, GW5, GW6, GW7, GW8 
  !     INLINE FUNCTION (PIERSON-MOSKOWITZ SMOOTH CUT-OFF)
  !     X == FR(1)/FREQUENCY
  REAL(8) :: EPMMA, X
  EPMMA(X) = EXP(-MIN(1.25*X**4,50.))*(X**5)

  if (allocated(FTRF)) deallocate(FTRF); ALLOCATE(FTRF(MFRSTLW:1))

  ALPH = 1d0/EPMMA(1d0)
  FRR = 1.d0
  DO MC=1,MFRSTLW,-1
   FTRF(MC)=ALPH*EPMMA(FRR)
   FRR=FRR*CO
  END DO

  if (allocated(INLCOEF)) deallocate(INLCOEF); ALLOCATE(INLCOEF(NINL,1:MLSTHG))
  if (allocated(RNLCOEF)) deallocate(RNLCOEF); ALLOCATE(RNLCOEF(NRNL,1:MLSTHG))

  DO MC=1,MLSTHG

   MP  = IKP (MC)
   MP1 = IKP1(MC)
   MM  = IKM (MC)
   MM1 = IKM1(MC)
   FFACP  = 1.
   FFACP1 = 1.
   FFACM  = 1.
   FFACM1 = 1.
   FTAIL  = 1.
   IC  = MC
   IP  = MP
   IP1 = MP1
   IM  = MM
   IM1 = MM1
!       LOW FREQUENCY FRONT TAIL
   IF (IM.LT.1) THEN
      FFACM = FTRF(IM)
      IM = 1
      IF (IM1.LT.1) THEN
         FFACM1 = FTRF(IM1)
         IM1 = 1
      ENDIF
   ENDIF
     
!       HIGH FREQUENCY TAIL
   IF (IP1.GT.ML) THEN
   
! Quick fix from Deborah
      ITEMP=IP1-ML+1
      IF(ITEMP .GT. SIZE(FRH))THEN
         ITEMP=SIZE(FRH)
      ENDIF
!         FFACP1 = FRH(IP1-ML+1)
      FFACP1 = FRH(ITEMP)

      IP1 = ML
      IF (IP .GT.ML) THEN
         FFACP  = FRH(IP -ML+1)
         IP  = ML
         IF (IC .GT.ML) THEN
            FTAIL  = FRH(IC -ML+1)
            IC  = ML
            IF (IM1.GT.ML) THEN
               FFACM1 = FRH(IM1-ML+1)
               IM1 = ML
            ENDIF
         ENDIF
      ENDIF
   ENDIF

   ICOUNT=1
   INLCOEF(ICOUNT,MC) = IC
   
   ICOUNT=ICOUNT+1
   INLCOEF(ICOUNT,MC) = IP
   ICOUNT=ICOUNT+1
   INLCOEF(ICOUNT,MC) = IP1
   ICOUNT=ICOUNT+1
   INLCOEF(ICOUNT,MC) = IM
   ICOUNT=ICOUNT+1
   INLCOEF(ICOUNT,MC) = IM1

   FKLAMP  = FKLAP(MC)
   FKLAMP1 = FKLAP1(MC)
   GW2 = FKLAMP1*FFACP*DAL1
   GW1 = GW2*CL11
   GW2 = GW2*ACL1
   GW4 = FKLAMP*FFACP1*DAL1
   GW3 = GW4*CL11
   GW4 = GW4*ACL1
   FKLAMPA = FKLAMP*CL11
   FKLAMPB = FKLAMP*ACL1
   FKLAMP2 = FKLAMP1*ACL1
   FKLAMP1 = FKLAMP1*CL11
   FKLAPA2 = FKLAMPA**2
   FKLAPB2 = FKLAMPB**2
   FKLAP12 = FKLAMP1**2
   FKLAP22 = FKLAMP2**2
   IRCOUNT=1
   RNLCOEF(IRCOUNT,MC) = FTAIL
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW1
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW3
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW4
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMPA
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMPB
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMP2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMP1
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAPA2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAPB2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAP12
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAP22

   FKLAMM  = FKLAM(MC)
   FKLAMM1 = FKLAM1(MC)
   GW6 = FKLAMM1*FFACM*DAL2
   GW5 = GW6*CL21
   GW6 = GW6*ACL2
   GW8 = FKLAMM*FFACM1*DAL2
   GW7 = GW8*CL21
   GW8 = GW8*ACL2
   FKLAMMA = FKLAMM*CL21
   FKLAMMB = FKLAMM*ACL2
   FKLAMM2 = FKLAMM1*ACL2
   FKLAMM1 = FKLAMM1*CL21
   FKLAMA2 = FKLAMMA**2
   FKLAMB2 = FKLAMMB**2
   FKLAM12 = FKLAMM1**2
   FKLAM22 = FKLAMM2**2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW5
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW6
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW7
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = GW8
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMMA
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMMB
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMM2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMM1
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMA2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAMB2
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAM12
   IRCOUNT=IRCOUNT+1
   RNLCOEF(IRCOUNT,MC) = FKLAM22

  ENDDO

  IF(ICOUNT.NE.NINL) THEN
   WRITE(6,*) '*************************************'
   WRITE(6,*) 'ERROR IN INISNONLIN : ICOUNT NE NINL'
   WRITE(6,*) 'ICOUNT= ',ICOUNT
   WRITE(6,*) 'NINL= ',NINL
   WRITE(6,*) '*************************************'
   !stop
  ENDIF
  IF(IRCOUNT.NE.NRNL) THEN
   WRITE(6,*) '*************************************'
   WRITE(6,*) 'ERROR IN INISNONLIN : IRCOUNT NE NRNL'
   WRITE(6,*) 'IRCOUNT= ',IRCOUNT
   WRITE(6,*) 'NRNL= ',NRNL
   WRITE(6,*) '*************************************'
   !stop
  ENDIF

END SUBROUTINE INIT_SNONLIN



INTEGER FUNCTION JAFU (CL, J, IAN)
   implicit none
   REAL(8), INTENT(IN) :: CL    !! WEIGHTS.
   INTEGER, INTENT(IN) :: J     !! INDEX IN ANGULAR ARRAY.
   INTEGER, INTENT(IN) :: IAN   !! NUMBER OF ANGLES IN ARRAY.

   JAFU = J + INT(CL)
   IF (JAFU.LE.0)   JAFU = JAFU+IAN
   IF (JAFU.GT.IAN) JAFU = JAFU-IAN

END FUNCTION JAFU


