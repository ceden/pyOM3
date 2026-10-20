# how to use pyOM3 on levante at dkrz.de

## CPU jobs

```
#!/bin/tcsh
#SBATCH --job-name=pyOM3      # Specify job name
#SBATCH --partition=compute            # Specify partition name
#SBATCH --nodes=1               # Specify number of nodes
#SBATCH --ntasks-per-node=128        # Specify number of (MPI) tasks on each node
#SBATCH --time=01:30:00            # Set a limit on the total run time
#SBATCH --account=XXXXXX            # Charge resources on this project account
#SBATCH --output=log.cpu_%j        # File name for standard output

module purge
conda activate pyOM3_wave
limit stacksize 204800
setenv HDF5_USE_FILE_LOCKING FALSE
setenv OMPI_MCA_opal_common_ucx_opal_mem_hooks 1
setenv OMP_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1
srun --ntasks=$SLURM_NTASKS --cpu_bind=cores --distribution=block:cyclic  python3 -u <script>.py

```

## GPU jobs


```
#!/bin/tcsh 
#SBATCH --job-name=_gpu           # Specify job name
#SBATCH --partition=gpu           # Specify partition name
#SBATCH --nodes=1                 # Specify number of nodes
#SBATCH --ntasks-per-node=4       # Specify number of (MPI) tasks on each node
#SBATCH --gpus-per-node=4         # Specify number of GPUs per task
#SBATCH --exclusive
###SBATCH --constraint=a100_80
#SBATCH --time=02:40:00           # Set a limit on the total run time
#SBATCH --account=XXXXXX          # Charge resources on this project account
#SBATCH --output=log.out_%j       # File name for standard output

module purge
conda activate jax_sharding1
setenv HDF5_USE_FILE_LOCKING FALSE
setenv TF_CPP_MIN_LOG_LEVEL 2
setenv JAX_COMPILATION_CACHE_DIR /work/uo0780/u241155/jaxcache
setenv JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0.5

srun -l python3 -u <script>.py
```





