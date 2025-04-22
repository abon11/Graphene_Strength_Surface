from mpi4py import MPI
from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np

# initialize core usage etc.
def initialize_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank


comm, rank = initialize_rank()

sheet = GrapheneSheet("data_files/data.60_60_rel1", 60, 60)

test = Simulation(comm, rank, sheet, x_erate=1e-3, y_erate=1e-3, thermo=1000, sim_length=1000)


print(test.strength)