# Using pyOM3 on marin at cen.uni-hamburg.de

The installation is described in `README.md`, section "Installing on marin at cen.uni-hamburg.de".
This note describes the daily use on marin.
In the examples `<user>` is your user name and `marin05` stands for the node you work on.
There are the nodes marin01 to marin08.

## Logging in

Inside the institute you log in directly,

```
ssh <user>@marin05
```

From home marin cannot be reached directly, also not with the VPN of the university.
You log in through the login server of the institute,

```
ssh -J <user>@login.cen.uni-hamburg.de <user>@marin05
```

If `marin05` is not found, use the full name `marin05.cen.uni-hamburg.de`.
To avoid typing this each time, add to `~/.ssh/config` on your own computer

```
Host marin05
    HostName marin05
    User <user>
    ProxyJump <user>@login.cen.uni-hamburg.de
```

Then `ssh marin05` is enough, also for the tunnel below.

## Setting up the environment

The modules must be loaded in every session, for running the code and for compiling.
`echo $SHELL` shows your shell.
Only the last line differs between the shells.

For sh, bash or zsh type

```
module purge
module load miniforge3/25.11.0
module load meson/1.2.2-gcc-12.2.0 ninja/1.11.1-gcc-12.2.0 patchelf/0.18.0-gcc-12.2.0
export HWLOC_HIDE_ERRORS=2
```

For csh or tcsh type

```
module purge
module load miniforge3/25.11.0
module load meson/1.2.2-gcc-12.2.0 ninja/1.11.1-gcc-12.2.0 patchelf/0.18.0-gcc-12.2.0
setenv HWLOC_HIDE_ERRORS 2
```

To load them at every login, put the same four lines at the end of `~/.bashrc` (bash) or `~/.tcshrc` (tcsh).
The last line only hides a harmless warning of the MPI library about the processors of the node.

Do not install Python packages with pip or conda on marin.
Everything needed is in the module `miniforge3`.

## Compiling

The Fortran parts are compiled as described in `README.md`.
The Makefile of `p3dfft` recognises marin by the name of the node.
It uses the MPI library (MPICH) and FFTW of `miniforge3` and the system gfortran.
To check that `p3dfft` uses only the MPI library of `miniforge3`, type in the directory `pyOM3/p3dfft` where `make` was called

```
ldd fortran.cpython-*-linux-gnu.so | grep mpi
```

Only `libmpifort.so.12` and `libmpi.so.12` from `miniforge3` must appear.
`libmpi.so.40` is the Open MPI of the system and must not appear.

## Running a configuration

A configuration is started with `mpirun`, for instance

```
mpirun -n 4 python3 internal_wave1.py
```

marin is shared with other users.
`nproc` shows the number of processors of the node, and `top` shows how many are in use.
Keep the number of processes moderate.

## Jupyter notebooks

Jupyter runs on marin, but the browser runs on your own computer.
The two are connected with an ssh tunnel.

1. On marin, in the directory with the notebooks,

   ```
   jupyter lab --no-browser --port=8890
   ```

   Jupyter prints an address like `http://localhost:8890/lab?token=...`.
   If port 8890 is used by somebody else, Jupyter takes the next free port (8891, ...).
   The printed address shows which port is used.

2. On your own computer, in a second terminal, inside the institute

   ```
   ssh -N -L 8890:localhost:8890 <user>@marin05
   ```

   or from home

   ```
   ssh -N -J <user>@login.cen.uni-hamburg.de -L 8890:localhost:8890 <user>@marin05
   ```

   With the entry in `~/.ssh/config` above, `ssh -N -L 8890:localhost:8890 marin05` works in both cases.
   Use the port printed by Jupyter in both places of the command.
   The terminal shows nothing while the tunnel is open. Leave it open.

3. Paste the printed address into the browser on your own computer.

Stop Jupyter with Control-C on marin when you are done.

### Several processes from a notebook

A notebook runs as a single process.
A configuration with several processes is started from a notebook cell with

```
!mpirun -n 4 python3 internal_wave1.py
```

The results are then read and plotted in the notebook.
A notebook can be converted into such a script with

```
jupyter nbconvert --to script internal_wave1.ipynb
```

## When something goes wrong

**The kernel of a notebook dies again and again.**
The reason is not shown in the browser.
It is printed in the terminal where `jupyter lab` runs, but a crash of the kernel gives no message there.
Convert the notebook into a script and run it in the terminal, as above.
The last lines show the error.

**The message "The MPI_Comm_size() function was called before MPI_INIT was invoked".**
This message comes from Open MPI, but `miniforge3` uses MPICH.
A compiled part uses the wrong MPI library.
Check it with `ldd` as described under "Compiling" and compile it again with the modules loaded.

**`python: Command not found` or `No module named ...`.**
The modules are not loaded.
`module list` shows the loaded modules.

## Ask Claude

Claude (claude.ai) helps with installing, compiling and running pyOM3, and with errors.
Claude does not know pyOM3 by itself.
Start the chat with

```
pyOM3 is at https://github.com/ceden/pyOM3. Read README.md and howto_marin.md there first.
```

Then give it the full context, not only the last line of an error.

- Say that you work on marin at cen.uni-hamburg.de with the module `miniforge3`, and which shell you use.
- Paste the command you typed and its complete output.
- Attach the files involved, for instance the Makefile, the configuration script or the notebook.
- Say what you expected to happen.

Check what Claude suggests before you run it.
Do not install packages with pip or conda on marin, even if it is suggested.
