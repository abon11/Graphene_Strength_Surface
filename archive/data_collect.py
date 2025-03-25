from mpi4py import MPI
from graphene_classes import GrapheneSheet
from graphene_classes import Simulation
import numpy as np
import matplotlib.pyplot as plt


def main():
    comm, rank = initialize_rank()
    s25x25 = GrapheneSheet("data_files/data.25_25_rel", 25, 25, makeStrengthSurface=False)
    s50x50 = GrapheneSheet("data_files/data.50_50_rel", 50, 50, makeStrengthSurface=False)


    # test1 = Simulation(comm, rank, s25x25, target_x=-35000.0, tau=0.0028571486, makeplots=True)  # Px, Py, Pz, max sim length, output timesteps
    target = [-70000, -80000, -90000, -100000, -110000, -120000, -130000, -150000, -175000, -200000]
    strengths25 = param_test(comm, rank, s25x25, target, 0.001)
    # strengths50 = damping_param_test(comm, rank, s50x50, -50000, taus)

    print('25x25 strengths:', strengths25)

    if rank == 0:
        print('DONE')
        # s25x25.finish_surface_plot()


# explores differences in speed of loading (NOTE: TAU IS NOT THE DAMPING PARAM IN LAMMPS, IT IS NORMALIZED BY P_TARGET)
def param_test(comm, rank, sheet, target, tau):
    strengths = []
    for i in range(len(target)):
        test = Simulation(comm, rank, sheet, target_x=target[i], tau=tau, sim_length=10000000, makeplots=True)
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