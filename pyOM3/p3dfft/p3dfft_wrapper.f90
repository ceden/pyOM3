
!
! simple wrapper routines for P3DFFT library
!

module wrapper_local
 logical :: initialized = .false.
end module wrapper_local

subroutine setup(n_pes_x,n_pes_y,nx,ny,nz,mpi_comm)
 use p3dfft
 use wrapper_local
 implicit none
 integer, intent(in) :: n_pes_x,n_pes_y,nx,ny,nz,mpi_comm
 if (initialized) call clean()    
 call p3dfft_setup((/n_pes_x,n_pes_y/),nx,ny,nz,mpi_comm,overwrite=.false.)
 initialized = .true.
end subroutine setup


subroutine clean()
 use p3dfft
 use wrapper_local
 implicit none
 if (initialized) call p3dfft_clean()
 initialized = .false.
end subroutine clean


subroutine get_dims_physical(istart,iend,isize)
 use p3dfft
 use wrapper_local
 implicit none
 integer, intent(out) :: istart(3),iend(3),isize(3) 
! Get dimensions for the original array of real numbers, X-pencils
 if (initialized) call p3dfft_get_dims(istart,iend,isize,1)
end subroutine get_dims_physical

subroutine get_dims_spectral(fstart,fend,fsize)
 use p3dfft
 use wrapper_local 
 implicit none
 integer, intent(out) :: fstart(3),fend(3),fsize(3)
! Get dimensions for the R2C-forward-transformed array of complex numbers
!   Z-pencils (depending on how the library was compiled, the first
!   dimension could be either X or Z)
 if (initialized) call p3dfft_get_dims(fstart,fend,fsize,2)
end subroutine get_dims_spectral

subroutine ifft_2D(QS,QP)
 use p3dfft
 use wrapper_local 
 implicit none
 real(kind=8),intent(inout)    :: QP(:,:,:)
 complex(kind=8),intent(inout)  :: QS(:,:,:)
 if (initialized) call p3dfft_btran_c2r(QS,QP,'nff')
 !QP=QP /(Lx*Ly) 
end subroutine ifft_2D

subroutine fft_2D(QP,QS)
 use p3dfft
 use wrapper_local 
 implicit none
 real(kind=8),intent(inout)     :: QP(:,:,:)
 complex(kind=8),intent(inout)  :: QS(:,:,:)
 if (initialized) call p3dfft_ftran_r2c(QP,QS,'ffn')
 !QS = QS*dx*dy
end subroutine fft_2D

subroutine ifft_3D(QS,QP)
 use p3dfft
 use wrapper_local 
 implicit none
 real(kind=8) ,intent(inout)    :: QP(:,:,:)
 complex(kind=8),intent(inout)  :: QS(:,:,:)
 if (initialized) call p3dfft_btran_c2r(QS,QP,'fft')
 !QP=QP /(Lx*Ly*Lz) 
end subroutine ifft_3D

subroutine fft_3D(QP,QS)
 use p3dfft
 use wrapper_local 
 implicit none
 real(kind=8) ,intent(inout)    :: QP(:,:,:)
 complex(kind=8),intent(inout)  :: QS(:,:,:)
 if (initialized) call p3dfft_ftran_r2c(QP,QS,'fft')
 !QS = QS*dx*dy*dz 
end subroutine fft_3D






