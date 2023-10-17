import numpy as np
import matplotlib.pyplot as plt
from stl import mesh

plot = False
plot_angles = True
create_STL = True
STL_file_name = 'cube.stl'
l = 1 # Length of cube side
n = 8 # Number of nodes per side

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if n < 2:
    print("\n ERROR: inputted n must be greater than 1\n")
    quit()

r = np.linspace(0,l,n)
# print(r)
atom_id = 1
xyz = []

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
vertices = xyz[:,1:]

# vertices[-1][2] *= 1.5
for i in range(0, 1):
    # vertices[2*(i+1)][2] *= 1.2 * 1.2
    # vertices[4*(i+1)][2] *= 1.2
    pass
# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]


if create_STL:
    print("Generating STL... \n")
    # Write the mesh to file "cube.stl"
    cube.save(STL_file_name)

if plot:
    m=xyz # m is an array of (x,y,z) coordinate triplets
    print("Plotting vertices... \n")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # fig.suptitle(f'Cube with {len(xyz)} atoms and {len(faces)} facets (n={n})', fontsize=12)
    for i in range(len(m)): #plot each point + it's index as text above
            ax.scatter(m[i,0+1],m[i,1+1],m[i,2+1], color = "cadetblue", alpha=0.5) 
            # ax.text(m[i,0+1],m[i,1+1],m[i,2+1],  '%s' % (str(round(m[i,0],0))), size= 7.5, zorder=1,  
            # color='k') 
    if plot_angles:
        for i in angles:
            x = []
            y = []
            z = []
            for j in i:
                # print(j)
                # print(xyz[j-1,1:])
                x.append(xyz[j-1,1])
                y.append(xyz[j-1,2])
                z.append(xyz[j-1,3])
            plt.plot(x, y, z, color = "black", alpha = 0.5)
            plt.pause(0.0000001)


    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.xlim([0, l])
    plt.ylim([0, l])
    plt.show()


# # Checking for duplicate facets

# for i,j in enumerate(faces):
#     faces[i] = np.sort(j)

# unq, count = np.unique(faces, axis=0, return_counts=True)

# print(unq[count>1])


# Printing summary
print(f"\nn = {n}")
# print(f"expected nodes: {4*n*(n-1)+ 2*(n-2)*(n-2)}")
print(f"Number of nodes: {len(xyz)}")
print(f"Number of facets: {len(faces)}")

print("Finished...")
