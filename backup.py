from mpi4py import MPI
from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np
import matplotlib.pyplot as plt


def main():
    comm, rank = initialize_rank()
    s25x25 = GrapheneSheet("data_files/data.25_25_rel", 25, 25, makeStrengthSurface=False)
    # s50x50 = GrapheneSheet("data_files/data.50_50_rel", 50, 50, makeStrengthSurface=False)


    # test1 = Simulation(comm, rank, s25x25, target_x=-35000.0, tau=0.0028571486, makeplots=True)  # Px, Py, Pz, max sim length, output timesteps
    x_rates = [0, 1.5e-5, 2.5e-5, 3.5e-5, 4.5e-5, 5.5e-5, 6.5e-5, 7.5e-5, 8.5e-5, 9.5e-5]
    y_rates = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    strengths25 = param_test(comm, rank, s25x25, x_rates, y_rates)

    print('25x25 strengths:', strengths25)

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