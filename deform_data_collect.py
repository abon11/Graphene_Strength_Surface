from mpi4py import MPI
from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np
import matplotlib.pyplot as plt


def main():
    comm, rank = initialize_rank()
    # s25x25 = GrapheneSheet("data_files/data.25_25_rel", 25, 25)
    # s50x50 = GrapheneSheet("data_files/data.50_50_rel", 50, 50)

    sheet1 = GrapheneSheet("/data1/avb25/graphene_sim_data/data_files/data.60_60_rel1", 60, 60)

    sheets = [sheet1]

    # run x-dominant tests 
    x_rates = []
    y_rates = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]
    # y_rates = [0]

    for i in range(len(y_rates)):
        x_rates.append(1e-3)
    
    defect_rates = [0.005, 0.01, 0.02]

    # seeds = [42, 1, 2, 3, 4, 5, 6]
    seeds = [7, 8, 9]
    for drate in defect_rates:
        for random_seed in seeds:
            # x-dominant
            for sheet in sheets:
                strengths = param_test(comm, rank, sheet, x_rates, y_rates, random_seed, drate)
                print(f'{sheet.x_atoms}_strengths: {strengths}')

            # y-dominant
            for sheet in sheets:
                strengths = param_test(comm, rank, sheet, y_rates, x_rates, random_seed, drate)  # SWITCHEDDDDD
                print(f'{sheet.x_atoms}_strengths: {strengths}')


    sheet1 = GrapheneSheet("data_files/data.100_100_rel", 100, 100)

    sheets = [sheet1]

    
    for drate in defect_rates:
        for random_seed in seeds:
            # x-dominant
            for sheet in sheets:
                strengths = param_test(comm, rank, sheet, x_rates, y_rates, random_seed, drate)
                print(f'{sheet.x_atoms}_strengths: {strengths}')

            # y-dominant
            for sheet in sheets:
                strengths = param_test(comm, rank, sheet, y_rates, x_rates, random_seed, drate)  # SWITCHEDDDDD
                print(f'{sheet.x_atoms}_strengths: {strengths}')

    if rank == 0:
        print('DONE')


def param_test(comm, rank, sheet, x_rates, y_rates, random_seed, defect_frac):
    strengths = []
    for i in range(len(x_rates)):
        test = Simulation(comm, rank, sheet, x_erate=x_rates[i], y_erate=y_rates[i], thermo=1000, defect_frac=defect_frac, defect_random_seed=random_seed, sim_length=10000000)
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