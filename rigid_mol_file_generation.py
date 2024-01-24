 # 05/02/2023 ~ Python script to generate a LAMMPS molecular template (.mol) from an inputted STL file

 

import numpy as np

from stl import mesh

from datetime import date

import trimesh

from math import *

mass_input = 50

central_atom = True;

 

# User Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

stl_file = 'sphere.stl'
mol_file_name = '/Users/haydn/Documents/_UNI_/PhD/Projects/Parallelisation_debug/sphere.mol'
# stl_file = input("STL: ")
scale = (0.1/1.5205830574035646)*0.5
atom_scale = 1

  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

print("Importing STL file... \n")

# Importing STL file

stl_data = mesh.Mesh.from_file(stl_file)

 

# Storing the facets and verticies into arrays

verticies =  np.unique(stl_data.points.reshape([-1, 3]), axis=0)

 

print("Assiging Vertex IDs... \n")



myobj = trimesh.load_mesh(stl_file, enable_post_processing=True, solid=True) # Import Objects
# Assigning each vertex a unique ID
xyz = []
for i,j in enumerate(myobj.vertices):
    xyz.append([i + 1, j[0]*scale, j[1]*scale, j[2]*scale])
xyz = np.array(xyz)

triangles = np.add(myobj.faces,1)

print("Generating Bonds... \n")

# Generating a list of the unique edges connecting verticies from the facets

edges = np.add(myobj.edges_unique,1)

# Assigning each edge a bond ID and type

bonds = []

for i,j in enumerate(edges):

    bonds.append([i+1, 1, j[0], j[1]])

bonds = np.array(bonds)


# Assigning each triangle to an angle with an ID and type

angles = []

angle_ID = 1

diameter = 0

for j in triangles:

    if (central_atom):
        angles.append([angle_ID, 1, j[0]+1, j[1]+1, j[2]+1])
    else:
        angles.append([angle_ID, 1, j[0], j[1], j[2]])

    angle_ID +=1

    temp_diameter = max([np.linalg.norm(xyz[int(j[0]-1)][1:] - xyz[int(j[1]-1)][1:]), np.linalg.norm(xyz[int(j[1]-1)][1:] - xyz[int(j[2]-1)][1:]), np.linalg.norm(xyz[int(j[2]-1)][1:] - xyz[int(j[0]-1)][1:])])

    if temp_diameter > diameter:

        diameter = temp_diameter 

angles = np.array(angles)

# Consituent atom properties

# Set atom diameter as 1.1* the largets gap between neighbouring atoms

diameter = 1.1 * diameter * atom_scale

# diameter = 0.05

mass =   (mass_input)/len(xyz)
# mass = pi * diameter * diameter * diameter * (1/6)

if (central_atom):
    print("Writing Files... \n")

    

    offset = max(xyz[:,1])*0.5  * 0 

    offset = max(xyz[:,2])*0.5 * 0

    offset = max(xyz[:,3])*0.5 * 0


    with open(

                mol_file_name,

                'w') as f:

            f.writelines(f"# Molecule template generated from python ({date.today()})" + '\n \n')

    
            # Printing Number of Atoms, Bonds, and Angles

            f.writelines(str(len(xyz)+1) + ' atoms\n')

            # f.writelines(str(len(bonds)) + ' bonds\n')

            f.writelines(str(len(angles)) + ' angles\n\n')


            # Printing XYZ Data

            f.writelines('Coords\n')

            f.writelines('# atom-ID, x, y, z\n')
            f.writelines(f'{int(1)} ' + f'{(0.0)} ' + f'{(0.0)} ' + f'{(0.0)}\n')
            for i in range(len(xyz)):

                f.writelines(f'{int(xyz[i,0]+1)} ' + f'{(xyz[i,1]-offset)} ' + f'{(xyz[i,2]-offset)} ' + f'{(xyz[i,3] - offset)}\n')

            f.writelines('\n')

    

            # Printing Types

            f.writelines('Types\n')

            f.writelines('#ID type\n')

            for i in np.arange(1, len(xyz) + 2):
                
                f.writelines(str((i)) + ' 1\n')
            
            f.writelines('\n')


            # Printing Molecule Types

            f.writelines('Molecules\n')

            f.writelines('#ID type\n')

            for i in np.arange(1, len(xyz) + 2):
                
                f.writelines(str((i)) + ' 1\n')
            
            f.writelines('\n')
        

            # Printing Diameters

            f.writelines('Diameters\n')

            f.writelines('#ID diameter\n')

            f.writelines(str((1)) + f' {diameter/10.0}\n')
            for i in range(len(xyz)):

                f.writelines(str((i+2)) + f' {diameter}\n')
                # f.writelines(str((i+1)) + f' {2*(8 - 7.977931022644043)}\n')
                
                # f.writelines(str((i+1)) + f' {0.2}\n')

            f.writelines('\n')
        
            # Printing Masses

            f.writelines('Masses\n')

            f.writelines('#ID mass\n')

            for i in range(len(xyz)+1):

                f.writelines(str((i+1)) + f' {mass}\n')
                # f.writelines(str((i+1)) + f' {0.5*(4/3)*(pi)*(0.2/2)*(0.2/2)*(0.2/2)}\n')

            f.writelines('\n')

            # # Printing Bonds Data

            # f.writelines('Bonds\n')

            # f.writelines('#ID type atom1 atom2\n')

            # np.savetxt(f, bonds, fmt='%d %d %d %d', delimiter=' ')

            # f.writelines('\n')

            # Printing Angles Data

            f.writelines('Angles\n')

            f.writelines('#ID type atom1 atom2 atom3\n')

            np.savetxt(f, angles, fmt='%d %d %d %d %d', delimiter=' ')

            f.writelines('\n')

    

    print("\nFinished")

    print(f"\nNo. of Atoms: {len(xyz)}\n")
    print(f"\nNo. of Angles: {len(angles)}\n")

    print(f"x: {max((xyz[:,1]-offset))}, {min((xyz[:,1]-offset))} -> {max((xyz[:,1]-offset)) - min((xyz[:,1]-offset))}")

    print(f"y: {max((xyz[:,2]-offset))}, {min((xyz[:,2]-offset))} -> {max((xyz[:,2]-offset)) - min((xyz[:,2]-offset))}")

    print(f"z: {max((xyz[:,3]-offset))}, {min((xyz[:,3]-offset))} -> {max((xyz[:,3]-offset)) - min((xyz[:,3]-offset))}")

    print(f"radius: {diameter/2}")
else:

    print("Writing Files... \n")

    

    offset = max(xyz[:,1])*0.5  * 0 

    offset = max(xyz[:,2])*0.5 * 0

    offset = max(xyz[:,3])*0.5 * 0


    with open(

                mol_file_name,

                'w') as f:

            f.writelines(f"# Molecule template generated from python ({date.today()})" + '\n \n')

    
            # Printing Number of Atoms, Bonds, and Angles

            f.writelines(str(len(xyz)) + ' atoms\n')

            # f.writelines(str(len(bonds)) + ' bonds\n')

            f.writelines(str(len(angles)) + ' angles\n\n')


            # Printing XYZ Data

            f.writelines('Coords\n')

            f.writelines('# atom-ID, atom-type, x, y, z\n')

            for i in range(len(xyz)):

                f.writelines(f'{int(xyz[i,0])} ' + f'{(xyz[i,1]-offset)} ' + f'{(xyz[i,2]-offset)} ' + f'{(xyz[i,3] - offset)}\n')

            f.writelines('\n')

    

            # Printing Types

            f.writelines('Types\n')

            f.writelines('#ID type\n')

            for i in np.arange(1, len(xyz) + 1):
                
                f.writelines(str((i)) + ' 1\n')
            
            f.writelines('\n')


            # Printing Molecule Types

            f.writelines('Molecules\n')

            f.writelines('#ID type\n')

            for i in np.arange(1, len(xyz) + 1):
                
                f.writelines(str((i)) + ' 1\n')
            
            f.writelines('\n')
        

            # Printing Diameters

            f.writelines('Diameters\n')

            f.writelines('#ID diameter\n')

            for i in range(len(xyz)):

                f.writelines(str((i+1)) + f' {diameter}\n')
                # f.writelines(str((i+1)) + f' {2*(8 - 7.977931022644043)}\n')
                
                # f.writelines(str((i+1)) + f' {0.2}\n')

            f.writelines('\n')
        
            # Printing Masses

            f.writelines('Masses\n')

            f.writelines('#ID mass\n')

            for i in range(len(xyz)):

                f.writelines(str((i+1)) + f' {mass}\n')
                # f.writelines(str((i+1)) + f' {0.5*(4/3)*(pi)*(0.2/2)*(0.2/2)*(0.2/2)}\n')

            f.writelines('\n')

            # # Printing Bonds Data

            # f.writelines('Bonds\n')

            # f.writelines('#ID type atom1 atom2\n')

            # np.savetxt(f, bonds, fmt='%d %d %d %d', delimiter=' ')

            # f.writelines('\n')

            # Printing Angles Data

            f.writelines('Angles\n')

            f.writelines('#ID type atom1 atom2 atom3\n')

            np.savetxt(f, angles, fmt='%d %d %d %d %d', delimiter=' ')

            f.writelines('\n')

    

    print("\nFinished")

    print(f"\nNo. of Atoms: {len(xyz)}\n")
    print(f"\nNo. of Angles: {len(angles)}\n")

    print(f"x: {max((xyz[:,1]-offset))}, {min((xyz[:,1]-offset))} -> {max((xyz[:,1]-offset)) - min((xyz[:,1]-offset))}")

    print(f"y: {max((xyz[:,2]-offset))}, {min((xyz[:,2]-offset))} -> {max((xyz[:,2]-offset)) - min((xyz[:,2]-offset))}")

    print(f"z: {max((xyz[:,3]-offset))}, {min((xyz[:,3]-offset))} -> {max((xyz[:,3]-offset)) - min((xyz[:,3]-offset))}")

    print(f"radius: {diameter/2}")
