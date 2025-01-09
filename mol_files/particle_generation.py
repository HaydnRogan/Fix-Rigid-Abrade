# 07/02/2023 Python script to generate an stl file for an arbitrarily sized cube

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from math import *


# User Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plot = False
create_STL = True
STL_file_name = "sphere.stl"

# STL_file_name = '/Users/haydn/Documents/Python Projects/Molecule Templates/cube.ST÷L'

single = False
l = 1 # Length of cube side [arbritary units]
n = 8 # Number of nodes per side
particle_name = "B-anm-00001"
N_max = 0
sphere = False
r_0 = 1
scale = 1
# N^3 -(N-2)^3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if n < 2:
    print("\n ERROR: inputted n must be greater than 1\n")
    quit()

r = np.linspace(0,l,n)
xyz = []
atom_id = 1

# Building the outside of the cube from top to bottom
print("Generating vertices... \n")
for i in range(n):
    #  Build 1st side
    for j in range(n):
        xyz.append([atom_id, r[j],r[0],r[i]])
        atom_id += 1
    # Build 2nd side
    for j in range(n-1):
        xyz.append([atom_id, r[(n-1)], r[j+1], r[i]])
        atom_id += 1
    # Build 3rd side
    for j in range(n-1):
        xyz.append([atom_id, r[n-j-2], r[n-1], r[i]])
        atom_id += 1
    # Close 4th side
    for j in range(n-2):
        xyz.append([atom_id, r[0], r[n-j-2], r[i]])
        atom_id += 1

#  # Sealing the top and bottom of cube

for z in [0,n-1]: 
    for i in np.arange(1,n-1):
        for j in  np.arange(1,n-1):
            xyz.append([atom_id, r[j], r[i], r[z]])
            atom_id += 1
xyz=np.array(xyz)

# Return an error if there are duplicate atoms placed
if len(xyz) != len(np.unique(xyz[:,1:],axis=0)):
    print("Error: Duplicate atoms placed")
    quit()

# Generating Facets
print("Generating facets... \n")
P = 4*(n-1)
Pn = 4*n*(n-1)
angles = []

# For the outside from bottom to top
for i in range (n-1):
    for j in np.arange(1,P):
        # print(f"{j + i*P} -> {j + 1 + i*P} -> {j + (i+1)*P}")
        angles.append([j + i*P, j + 1 + i*P , j + (i+1)*P])
        # print(f"{j + 1 + i*P} -> {j + 1 + (i+1)*P} -> {j + (i+1)*P}\n")
        angles.append([j + 1 + i*P , j + 1 + (i+1)*P , j + (i+1)*P])
    # print(f"{(i+1)*P} -> {1 + i*P} -> {(i+2)*P}")
    angles.append([(i+1)*P, 1 + i*P , (i+2)*P])
    # print(f"{1 + i*P} -> {1 + (i+1)*P} -> {(i+2) * P}\n")
    angles.append([1 + i*P, 1 + (i+1)*P, (i+2) * P])

if n > 3:

    # # For the bottom
    m = n-2
    for i in range(m-1):
        for j in np.arange(1,m):
            # print(f"{j+(i*m)} -> {j+1+(i*m)} -> {j+m+(i*m)}")
            angles.append(list(reversed([j+(i*m) + Pn, j+1+(i*m)+ Pn, j+m+(i*m)+ Pn])))
            # print(f"{j+1+(i*m)} -> {m+j+1+(i*m)} -> {m+j+(i*m)} \n")
            angles.append(list(reversed([j+1+(i*m)+ Pn , m+j+1+(i*m)+ Pn, m+j+(i*m)+ Pn])))

    # Stitch bottom to sides
    # 1st side:
    # print("Side 1")
    i = 1
    # print(f"{i} -> {i+1} -> {P}")
    angles.append(list(reversed([i, i+1, P])))
    # print(f"{i+1} -> _{i+Pn}_ -> {P} \n")
    angles.append(list(reversed([i+1, i+Pn, P])))
    for i in np.arange(2,n-1):
        # print(f"{i} -> {i+1} -> _{i-1+Pn}_")
        angles.append(list(reversed([i, i+1, i-1+Pn])))
        # print(f"{i+1} -> _{i+Pn}_ -> _{i-1+Pn}_ \n")
        angles.append(list(reversed([i+1, i+Pn, i-1+Pn])))
    i += 1
    # print(f"{i} -> {i+1} -> _{i-1+Pn}_")
    angles.append(list(reversed([i, i+1, i-1+Pn])))
    # print(f"{i+1} -> {i+2} -> _{i-1+Pn}_ \n")
    angles.append(list(reversed([i+1, i+2, i-1+Pn])))

    # Second side:
    # print("Side 2")
    for i in np.arange(n+1,2*n-2):
        # print(f"{i} -> {i+1} -> _{(n-2)*(1+(i-n-1))+Pn}_")
        angles.append(list(reversed([i, i+1, (n-2)*(1+(i-n-1))+Pn])))
        # print(f"{i+1} -> _{(n-2)*(1+(i-n))+Pn}_ -> _{(n-2)*(1+(i-n-1))+Pn}_ \n")
        angles.append(list(reversed([i+1, (n-2)*(1+(i-n))+Pn, (n-2)*(1+(i-n-1))+Pn])))
    i += 1
    # print(f"{i} -> {i+1} -> _{(n-2)*(1+(i-n-1))+Pn}_")
    angles.append(list(reversed([i, i+1, (n-2)*(1+(i-n-1))+Pn])))
    # print(f"{i+1} -> {i+2} -> _{(n-2)*(1+(i-n-1))+Pn}_ \n")
    angles.append(list(reversed([i+1, i+2, (n-2)*(1+(i-n-1))+Pn])))

    # Third side:
    # print("Side 3")
    for i in np.arange(2*n,3*n-3):
        # print(f"{i} -> {i+1} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_")
        angles.append(list(reversed([i, i+1, (n-2)*(n-2)-(i-2*n)+Pn])))
        # print(f"{i+1} -> _{(n-2)*(n-2)-(i-2*n+1)+Pn}_ -> _{(n-2)*(n-2)-(i-2*n)+Pn}_ \n")
        angles.append(list(reversed([i+1, (n-2)*(n-2)-(i-2*n+1)+Pn, (n-2)*(n-2)-(i-2*n)+Pn])))
    i +=1
    # print(f"{i} -> {i+1} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_")
    angles.append(list(reversed([i, i+1, (n-2)*(n-2)-(i-2*n)+Pn])))
    # print(f"{i+1} -> {i+2} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_ \n")
    angles.append(list(reversed([i+1, i+2, (n-2)*(n-2)-(i-2*n)+Pn])))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Fourth side:
    # print("Side 4")
    for i in np.arange(3*n-1,4*n-4):
        # print(f"{i} -> {i+1} -> _{(n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn}_")
        angles.append(list(reversed([i, i+1, (n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn])))
        # print(f"{i+1} -> _{(n-2)*(n-2)-(n-3) -(n-2)*(i-(3*n-1)+1)+Pn}_ -> _{(n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn}_ \n")
        angles.append(list(reversed([i+1, (n-2)*(n-2)-(n-3) -(n-2)*(i-(3*n-1)+1)+Pn, (n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn])))


    Pn = 4*n*(n-1) + (n-2)*(n-2)
    P_offset = 4*(n-1)*(n-1)

    # For the Top
    m = n-2
    for i in range(m-1):
        for j in np.arange(1,m):
            # print(f"{j+(i*m)+Pn} -> {j+1+(i*m)+Pn} -> {j+m+(i*m)+Pn}")
            angles.append([j+(i*m)+Pn, j+1+(i*m)+Pn, j+m+(i*m)+Pn])
            # print(f"{j+1+(i*m)+Pn} -> {m+j+1+(i*m)+Pn} -> {m+j+(i*m)+Pn} \n")
            angles.append([j+1+(i*m)+Pn, m+j+1+(i*m)+Pn, m+j+(i*m)+Pn])

    # Stitch Top to sides
    # 1st side:
    # print("Side 1")
    i = 1
    # print(f"{i+P_offset} -> {i+1+P_offset} -> {P+P_offset}")
    angles.append([i+P_offset, i+1+P_offset, P+P_offset])
    # print(f"{i+1+P_offset} -> _{i+Pn}_ -> {P+P_offset} \n")
    angles.append([i+1+P_offset, i+Pn, P+P_offset])


    for i in np.arange(2,n-1):
        # print(f"{i+P_offset} -> {i+1+P_offset} -> _{i-1+Pn}_")
        angles.append([i+P_offset, i+1+P_offset, i-1+Pn])
        # print(f"{i+1+P_offset} -> _{i+Pn}_ -> _{i-1+Pn}_ \n")
        angles.append([i+1+P_offset, i+Pn, i-1+Pn])

    i += 1
    # print(f"{i+P_offset} -> {i+1+P_offset} -> _{i-1+Pn}_")
    angles.append([i+P_offset, i+1+P_offset, i-1+Pn])
    # print(f"{i+1+P_offset} -> {i+2+P_offset} -> _{i-1+Pn}_ \n")
    angles.append([i+1+P_offset, i+2+P_offset, i-1+Pn])

    # Second side:
    # print("Side 2")
    for i in np.arange(n+1,2*n-2):
        # print(f"{i+P_offset} -> {i+1+P_offset} -> _{(n-2)*(1+(i-n-1))+Pn}_")
        angles.append([i+P_offset, i+1+P_offset, (n-2)*(1+(i-n-1))+Pn])
        # print(f"{i+1+P_offset} -> _{(n-2)*(1+(i-n))+Pn}_ -> _{(n-2)*(1+(i-n-1))+Pn}_ \n")
        angles.append([i+1+P_offset, (n-2)*(1+(i-n))+Pn, (n-2)*(1+(i-n-1))+Pn])
    i += 1
    # print(f"{i+P_offset} -> {i+1+P_offset} -> _{(n-2)*(1+(i-n-1))+Pn}_")
    angles.append([i+P_offset, i+1+P_offset, (n-2)*(1+(i-n-1))+Pn])
    # print(f"{i+1+P_offset} -> {i+2+P_offset} -> _{(n-2)*(1+(i-n-1))+Pn}_ \n")
    angles.append([i+1+P_offset, i+2+P_offset, (n-2)*(1+(i-n-1))+Pn])

    # Third side:
    # print("Side 3")
    for i in np.arange(2*n,3*n-3):
        # print(f"{i+P_offset} -> {i+1+P_offset} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_")
        angles.append([i+P_offset, i+1+P_offset, (n-2)*(n-2)-(i-2*n)+Pn])
        # print(f"{i+1+P_offset} -> _{(n-2)*(n-2)-(i-2*n+1)+Pn}_ -> _{(n-2)*(n-2)-(i-2*n)+Pn}_ \n")
        angles.append([i+1+P_offset, (n-2)*(n-2)-(i-2*n+1)+Pn, (n-2)*(n-2)-(i-2*n)+Pn])

    i +=1
    # print(f"{i+P_offset} -> {i+1+P_offset} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_")
    angles.append([i+P_offset, i+1+P_offset, (n-2)*(n-2)-(i-2*n)+Pn])
    # print(f"{i+1+P_offset} -> {i+2+P_offset} -> _{(n-2)*(n-2)-(i-2*n)+Pn}_ \n")
    angles.append([i+1+P_offset, i+2+P_offset, (n-2)*(n-2)-(i-2*n)+Pn])

    # Fourth side:
    # print("Side 4")
    for i in np.arange(3*n-1,4*n-4):
        # print(f"{i+P_offset} -> {i+1+P_offset} -> _{(n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn}_")
        angles.append([i+P_offset, i+1+P_offset, (n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn])
        # print(f"{i+1+P_offset} -> _{(n-2)*(n-2)-(n-3) -(n-2)*(i-(3*n-1)+1)+Pn}_ -> _{(n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn}_ \n")
        angles.append([i+1+P_offset, (n-2)*(n-2)-(n-3) -(n-2)*(i-(3*n-1)+1)+Pn, (n-2)*(n-2)-(n-3)-(n-2)*(i-(3*n-1))+Pn])

# Edge case for n =2
if n == 2: 
    
    angles.append([4, 2, 1])
    angles.append([4, 3 , 2])
    angles.append([5, 6, 8])
    angles.append([6, 7, 8])

#  Edge case for n = 3
if n == 3: 
    for i in range(2):
        for j in range(8):
            if i:
                angles.append([j + 1 + i * 16, ((j+1) % 8) + 1 + i * 16, 24 + 1 + i])
            else:
                angles.append(list(reversed([j + 1 + i * 16, ((j+1) % 8) + 1 + i * 16, 24 + 1 + i])))


angles = np.array(angles)
faces = np.subtract(angles, 1)
xyz[:,1:] = np.subtract(xyz[:,1:], (l/2))
# Storing the spherical coords of each vertex from  [l/2, l/2, l/2]


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

xyz[:,1:] = appendSpherical_np(xyz[:,1:])[:,3:]
print(xyz)
    # 05/03/2023

# Python script to generate STL files from SH coefficients for a given resolution and degree of expansion

print("\n Running \n")

from math import *
import numpy as np
from scipy import special as sp
from stl import mesh
import time



# Define import and export directories
export_folder = f"/Users/haydn/Documents/Python Projects/Normal_Investigation/Closed_Mesh_Tests"
shape_coeff_file = f"/Users/haydn/Documents/Python Projects/3D Inertia/Coeffs/{particle_name}.dat.txt"

# Importing shape coefficents 
shape_coeffs = np.array(np.loadtxt(shape_coeff_file, delimiter=None, usecols=None))

# Defining particle density
density = 1

# Define function to reurn legendre polynomial of order n evaulated at x. 
# Note, the corresponding degree of m -> 0,n can be accessed by legendre(n,x)[m]
def legendre(n, x):
    P_mn = sp.lpmn(n,n,x)[0][:,-1]
    return P_mn

# Define function to evaluate Ymn
def Ymn(n, m, theta, phi):
    Y_mn = sqrt(((2*n+1)*factorial(n-abs(m)))/((4*pi)*factorial(n+abs(m)))) * (legendre(n,np.cos(theta))[abs(m)]) * np.exp(1j * m * phi) 
    return Y_mn

# Specifying the maxium SH degree of expansion
# N_max = int(max(shape_coeffs[:,0]))


# Setting up dictionary to store the volume and mass of each particle for each resolution N =1 -> N_max
results = {}
for N in range(0, N_max+1):
    key = f"N = {N}"
    results[key] = []   

continuous_volume = {}
for N in range(0, N_max+1):
    key = f"N = {N}"
    continuous_volume[key] = []   

continuous_mass = {}
for N in range(0, N_max+1):
    key = f"N = {N}"
    continuous_mass[key] = []    
    

for i,[theta,phi] in enumerate(xyz[:,2:]):
    # print(f"i:{i}, Theta: {theta}, Phi: {phi}")
    sum1 = 0
    index = 0
    for n in range(0, N_max + 1):
        sum2 = 0
        for m in reversed(range(-n,n+1)):
            amn = (shape_coeffs[index][2] + shape_coeffs[index][3]*1j)
            index += 1
            sum2 += ((Ymn(n,m,(theta), (phi))) * amn)
        sum1 += sum2

        if sphere:
            xyz[i,1] = (r_0) * scale
        else:
            xyz[i,1] = (np.absolute(sum1) + r_0) * scale


vertices = []
for i in xyz:
    x = i[1]*np.sin(i[2])*np.cos(i[3])
    y = i[1]*np.sin(i[2])*np.sin(i[3])
    z = i[1]*np.cos(i[2])
    vertices.append([x,y,z])

vertices = np.array(vertices)
print(vertices)
# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]


if create_STL:
    print("Generating STL... \n")
    # Write the mesh to file "cube.stl"
    cube.save(STL_file_name)
    print(f"{len(xyz)} atoms")
print("Finished.")
