from mpi4py import MPI
from graphene_classes import GrapheneSheet
from graphene_classes import Simulation


def main():
    comm, rank = initialize_rank()

    strength_list = []

    # pressures = [[-50000.0, 0.0, 0.0], [0.0, -75000.0, 0.0]]
    pressures = [[-50000.0, 0.0, 0.0]]

    # graphene_sheets = [
    #     GrapheneSheet("data_files/data.25_25_rel", 25, 25, 5195.5),
    #     GrapheneSheet("data_files/data.50_50_rel", 50, 50, 22766.4)
    #     ]

    s25x25 = GrapheneSheet("data_files/data.25_25_rel", 25, 25, 5195.5)

    # for press in pressures:
    #     strength = execute_test(comm, rank, s25x25, press[0], press[1], press[2], showplots=True)
    #     strength_list.append(strength)

    test1 = Simulation(comm, rank, s25x25, -50000.0, 0.0, 0.0, 40000, 1000, makeplots=True)

    if rank == 0:
        s25x25.finish_surface_plot()
        print(test1.sheet.x_atoms, test1.sheet.y_atoms, test1.strength)


# initialize core usage etc.
def initialize_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank

if __name__ == '__main__':
    main()