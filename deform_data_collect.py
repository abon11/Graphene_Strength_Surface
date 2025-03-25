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
    sheet2 = GrapheneSheet("data_files/data.60_60_rel2", 60, 60)
    sheet3 = GrapheneSheet("data_files/data.60_60_rel3", 60, 60)
    sheet4 = GrapheneSheet("data_files/data.60_60_rel4", 60, 60)
    sheet5 = GrapheneSheet("data_files/data.60_60_rel5", 60, 60)
    sheet6 = GrapheneSheet("data_files/data.60_60_rel6", 60, 60)
    sheet7 = GrapheneSheet("data_files/data.60_60_rel7", 60, 60)


    sheets = [sheet1, sheet2, sheet3, sheet4, sheet5, sheet6, sheet7]

    # test1 = Simulation(comm, rank, s25x25, target_x=-35000.0, tau=0.0028571486, makeplots=True)  # Px, Py, Pz, max sim length, output timesteps
    # run x-dominant tests 
    x_rates = []
    y_rates = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]
    for i in range(len(y_rates)):
        x_rates.append(1e-3)

    for sheet in sheets:
        strengths = param_test(comm, rank, sheet, x_rates, y_rates)
        print(f'{sheet.x_atoms}_strengths: {strengths}')

    # run y-dominant tests
    x_rates = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]
    y_rates = []
    for i in range(len(x_rates)):
        y_rates.append(1e-3)

    for sheet in sheets:
        strengths = param_test(comm, rank, sheet, x_rates, y_rates)
        print(f'{sheet.x_atoms}_strengths: {strengths}')
    

    if rank == 0:
        print('DONE')
        # s25x25.finish_surface_plot()

def param_test(comm, rank, sheet, x_rates, y_rates):
    strengths = []
    for i in range(len(x_rates)):
        test = Simulation(comm, rank, sheet, x_erate=x_rates[i], y_erate=y_rates[i], thermo=1000, sim_length=100000000, makeplots=True)
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