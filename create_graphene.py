import numpy as np


def main():
    # Lattice constants for graphene
    lattice_a = 1.42 * 2  # Angstrom, graphene lattice constant
    carbon_distance = 1.42

    # Size of graphene sheet
    n_atoms_x = 80
    n_atoms_y = 80

    printBonds = False

    box_tol = 0.0
    z_tol = 40.0

    dims = np.zeros((6))

    # Box dimensions
    dims[0] = (n_atoms_x / 2 - 0.5) * (carbon_distance + lattice_a) + box_tol
    dims[1] = (n_atoms_y / 2) * (carbon_distance * np.sqrt(3)) + box_tol
    dims[2] = carbon_distance + z_tol
    dims[3] = -box_tol - lattice_a - (carbon_distance / 2)
    dims[4] = -box_tol - carbon_distance
    dims[5] = -z_tol - carbon_distance

    print("Input name for data file (will be saved as data.<name>)\n")
    filename = input()
    filepath = 'data_files/data.' + str(filename)
    
    atom_id = 1
    bond_id = 1

    x_odd = n_atoms_x % 2
    if x_odd == 1:
        x_num = n_atoms_x - 1
    else:
        x_num = n_atoms_x + 0
    
    atoms = []
    bonds = []

    for i in range(x_num):
        for j in range(int(np.trunc(n_atoms_y / 2))):

            if j != 0:
                bonds.append([bond_id, atom_id - 1, atom_id])  # vertical bond
                bond_id += 1

            # get the positions for two atoms, going vertically up the zigzag
            x1, y1, x2, y2, z = calc_pos(i, j, carbon_distance, lattice_a)

            atoms.append([atom_id, x1, y1, z])
            atoms.append([atom_id + 1, x2, y2, z])

            bonds.append([bond_id, atom_id, atom_id + 1])  # vertical bond

            bond_id += 1
            atom_id += 2

        # for the case that n_atom_y is odd, we need to add the atom at the top
        if n_atoms_y % 2 != 0:
            x, y, _, _, z = calc_pos(i, int((n_atoms_y-1)/2), carbon_distance, lattice_a)
            atoms.append([atom_id, x, y, z])

            bonds.append([bond_id, atom_id - 1, atom_id])  # vertical bond

            bond_id += 1
            atom_id += 1
    
    if x_odd == 1:
        for j in range(int(np.trunc(n_atoms_y / 2))):
            x1, y1, x2, y2, z = calc_pos(int(n_atoms_x), j, carbon_distance, lattice_a)
            atoms.append([atom_id, x1, y1, z])
            atoms.append([atom_id + 1, x2, y2, z])
            atom_id += 2
        if n_atoms_y % 2 != 0:
            x, y, _, _, z = calc_pos(int(n_atoms_x), int((n_atoms_y-1)/2), carbon_distance, lattice_a)
            atoms.append([atom_id, x, y, z])
            atom_id += 1
    
    # get the horizontal bonds
    for pairs in range(int(np.trunc(n_atoms_x / 2))):
        for id in range(1, n_atoms_y + 1):

            global_id = (2 * pairs * n_atoms_y) + id

            if id % 2 == 1:
                # this gets the bonds from the right particle to the left particle. 
                bonds.append([bond_id, global_id, global_id + n_atoms_y])
                bond_id += 1

            # if we want to bond to the right
            elif (id % 2 == 0):
                # if x_atoms is odd and we are on the last pair, the id's only go up by 2 * n_atoms
                if (n_atoms_x % 2 != 0) and (pairs + 1 == np.trunc(n_atoms_x / 2)):
                    bonds.append([bond_id, global_id, global_id + (2 * n_atoms_y)])
                    bond_id += 1
                
                # this is needed because we don't want to bond to the right on the last one if we're not even
                elif (pairs + 1 != np.round(n_atoms_x / 2)):
                    bonds.append([bond_id, global_id, global_id + (3 * n_atoms_y)])
                    bond_id += 1
        
    writefile(filepath, dims, atoms, bonds, printBonds)

    print("LAMMPS data file", str(filename), "created successfully.")


def calc_pos(i, j, dist, lat):
    # if we are on an i column where the second atom is up and to the right
    if i % 2 == 0:
        x1 = (i / 2) * (dist + lat)
        y1 = j * dist * np.sqrt(3)

        x2 = x1 + dist / 2
        y2 = y1 + ((dist * np.sqrt(3)) / 2)

        z = 0.0
    else:
        x1 = ((i-1)/2) * (dist + lat) - dist
        y1 = j * dist * np.sqrt(3)

        x2 = x1 - dist / 2
        y2 = y1 + ((dist * np.sqrt(3)) / 2)

        z = 0.0
    return x1, y1, x2, y2, z


def writefile(filepath, dims, atoms, bonds, printBonds):
    # Open file to write LAMMPS data
    with open(filepath, "w") as f:
        # Header for LAMMPS data file
        f.write("LAMMPS data file for graphene sheet\n\n")
        
        # Total number of atoms and atom types
        f.write(f"{len(atoms)} atoms\n")
        f.write("1 atom types\n")

        if printBonds:
            f.write(f"{len(bonds)} bonds\n")
            f.write("1 bond types\n")
        
        f.write("\n")

        # Box dimensions: assuming z=0 for a 2D graphene sheet
        f.write(f"{dims[3]:.16e} {dims[0]:.16e} xlo xhi\n")
        f.write(f"{dims[4]:.16e} {dims[1]:.16e} ylo yhi\n")
        f.write(f"{dims[5]:.16e} {dims[2]:.16e} zlo zhi\n\n")
        f.write(f"0.0 0.0 0.0 xy xz yz\n")  # added for shear functionality
        
        # Masses section
        f.write("Masses\n\n")
        f.write("1 12.01  # carbon\n\n")
        
        # Atoms section
        f.write("Atoms\n\n")

        if printBonds:
            for i in range(len(atoms)):
                f.write(f"{atoms[i][0]} 1 1 {atoms[i][1]:.16e} {atoms[i][2]:.16e} {atoms[i][3]:.16e}\n")

            f.write("\nBonds\n\n")
            for i in range(len(bonds)):
                f.write(f"{bonds[i][0]} 1 {bonds[i][1]} {bonds[i][2]}\n")
        
        else:
            for i in range(len(atoms)):
                f.write(f"{atoms[i][0]} 1 {atoms[i][1]:.16e} {atoms[i][2]:.16e} {atoms[i][3]:.16e}\n")
        

if __name__ == "__main__":
    main()
