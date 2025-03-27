from mpi4py import MPI
from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np
import matplotlib.pyplot as plt


def main():
    comm, rank = initialize_rank()
    # s25x25 = GrapheneSheet("data_files/data.25_25_rel", 25, 25)
    # s50x50 = GrapheneSheet("data_files/data.50_50_rel", 50, 50)

    sheet1 = GrapheneSheet("data_files/data.60_60_rel1", 60, 60)

    sheets = [sheet1]

    # run x-dominant tests 
    x_rates = []
    y_rates = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]

    for i in range(len(y_rates)):
        x_rates.append(1e-3)  # negative because we are trying compression (going much slower than 1e-3)

    for sheet in sheets:
        strengths = param_test(comm, rank, sheet, x_rates, y_rates)
        print(f'{sheet.x_atoms}_strengths: {strengths}')

    if rank == 0:
        print('DONE')


def param_test(comm, rank, sheet, x_rates, y_rates):
    strengths = []
    for i in range(len(x_rates)):
        test = Simulation(comm, rank, sheet, x_erate=x_rates[i], y_erate=y_rates[i], thermo=1000, add_defects=True, sim_length=10000000)
        if test.strength[0] is None:
            strengths.append(test.strength[0])
        else:
            strengths.append(np.max(test.strength))

    return np.array(strengths)


# initialize core usage etc.
def initialize_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank

if __name__ == '__main__':
    main()