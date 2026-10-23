#!/bin/tcsh
#SBATCH --job-name=icon1
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem-per-cpu=1920M
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --account=uo0780
#SBATCH --output=log-%j.out
#SBATCH --error=log-%j.err

limit stacksize 1g

module purge
module load intel-oneapi-compilers/2022.0.1-gcc-11.2.0
module load openmpi/4.1.2-intel-2021.5.0
module load intel-oneapi-mkl
module load netcdf-fortran/4.5.3-openmpi-4.1.2-intel-2021.5.0
module list


setenv OMPI_MCA_pml "ucx"
setenv OMPI_MCA_btl self
setenv OMPI_MCA_osc "pt2pt"
setenv UCX_IB_ADDR_TYPE ib_global
# for most runs one may or may not want to disable HCOLL
setenv OMPI_MCA_coll "^ml,hcoll"
setenv OMPI_MCA_coll_hcoll_enable "0"
setenv HCOLL_ENABLE_MCAST_ALL "0"
setenv HCOLL_MAIN_IB mlx5_0:1
setenv UCX_NET_DEVICES mlx5_0:1
setenv UCX_TLS mm,knem,cma,dc_mlx5,dc_x,self
setenv UCX_UNIFIED_MODE y
setenv HDF5_USE_FILE_LOCKING FALSE
setenv OMPI_MCA_io "romio321"
setenv UCX_HANDLE_ERRORS bt

# Use srun (not mpirun or mpiexec) command to launch
# programs compiled with any MPI library
rm -f ritt
srun -l --cpu_bind=verbose --hint=nomultithread  --distribution=block:cyclic ./setup1.x 64 16 > out.000

if (-r ritt) then
 set ritt = `cat ritt`
 mv out.000 out.000_${ritt}
 mv pyOM.cdf pyOM_${ritt}.cdf
 mv pyOM_idemix.cdf pyOM_idemix_${ritt}.cdf
 mv pyOM_tke.cdf pyOM_tke_${ritt}.cdf
 ##mv pyOM_biha_thk.cdf pyOM_biha_thk_${ritt}.cdf
 mv pyOM_energy.cdf pyOM_energy_${ritt}.cdf
 #if ( $ritt <  1460000 ) then # 10 Jahre
 if ( $ritt <  2190000 ) then # 15 Jahre
  sbatch run.sh
 endif
endif





