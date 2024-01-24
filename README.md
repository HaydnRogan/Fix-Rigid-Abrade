# Fix_Abrasion

cpp and header files for the work in progress fix_abrasion. A sample input script, with the relevant mol files, is provided. 

To compile:

- Place the fix_rigid_abrade .cpp and .h file in your LAMMPS src/RIGID/
- Place the fix_wall_gran_region .cpp and .h file in your LAMMPS src/GRANULAR

in terminal at src:

- make yes-granular
- make yes-rigid
- make mpi
- mv lmp_mpi project_directory/lmp_mpi

To run the input script:
- in.test and sphere.mol in a project directory
- open directory in terminal
- mpirun --oversubscribe -np N lmp_mpi -in in.test 
