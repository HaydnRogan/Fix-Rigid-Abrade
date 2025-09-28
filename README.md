# Fix Rigid/Abrade

Personal repository for the development of Fix Rigid/Abrade. 

A sample input scripts, with the relevant mol files, are provided in ./Example_Scripts

To compile:

Option 1 - Appending src files into your own LAMMPS Feature Release	(10Sep2025) repository:

  - Place the fix_rigid_abrade .cpp and .h file in your LAMMPS src/RIGID/
  - Place the compute_rigid_local_abrade .cpp and .h file in the same directory
  - Place the fix_wall_gran_region .cpp and .h file in your LAMMPS src/GRANULAR


in terminal at src:

- make yes-granular
- make yes-rigid
- make yes-molecule
- make mpi
- mv lmp_mpi to your <project_directory>

To run the input script:

- Open the relevant ./Example_Scripts/ directory in terminal
- "mpirun --oversubscribe -np N lmp_mpi -in in.<example_script>" where N is the number of processors you wish to run
- Visualise dump.pos in Ovito or other visualiser of your choice

