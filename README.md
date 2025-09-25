# Fix Rigid/Abrade

Personal repository for the development of Fix Rigid/Abrade. 

A sample input scripts, with the relevant mol files, are provided in ./examples

To compile:

Option 1 - Appending src files into your own LAMMPS Feature Release	(10Sep2025) repository:

  - Place the fix_rigid_abrade .cpp and .h file in your LAMMPS src/RIGID/
  - Place the compute_rigid_local_abrade .cpp and .h file in the same directory
  - Place the fix_wall_gran_region .cpp and .h file in your LAMMPS src/GRANULAR


in terminal at src:

- make yes-molecule
- make yes-granular
- make yes-rigid
- make mpi
- mv lmp_mpi to your <project_directory>

To run the input script:

- Place ./examples/in.example_script and ./examples/example_sphere.mol in your <project_directory>
- Open <project_directory> in terminal
- "mpirun --oversubscribe -np N lmp_mpi -in in.example_script" where N is the number of processors your wish to run
   Visualise dump.example_output in Ovito or other visualiser of your choice

