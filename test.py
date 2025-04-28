from mpi4py import MPI
from lammps import lammps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} of {size} starting.")

# All ranks must create the LAMMPS object
lmp = lammps(comm=comm)

print(f"Rank {rank}: LAMMPS instance created.")

# Optionally, do some minimal LAMMPS commands here
# For example:
# lmp.command("units metal")
# lmp.command("atom_style atomic")

lmp.close()

print(f"Rank {rank}: LAMMPS closed successfully.")
