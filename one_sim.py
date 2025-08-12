""" This is what actually runs one simulation of a deformation of a graphene sheet """

from mpi4py import MPI
from deform_graphene import GrapheneSheet
from deform_graphene import Simulation
import socket, os
import argparse


def main():
    comm, rank = initialize_rank()

    print(f"Running on {socket.gethostname()}, PID={os.getpid()}, MPI size={os.environ.get('OMPI_COMM_WORLD_SIZE', '?')}")


    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sheet_path", type=str, required=True)
    parser.add_argument("--x_atoms", type=int, required=True)
    parser.add_argument("--y_atoms", type=int, required=True)

    parser.add_argument("--defects", type=str, required=True)
    parser.add_argument("--defect_random_seed", type=int, required=True)
    parser.add_argument("--sim_length", type=int, required=True)
    parser.add_argument("--timestep", type=float, required=True)
    parser.add_argument("--thermo", type=int, required=True)
    parser.add_argument("--makeplots", type=str, required=True)  # bool
    parser.add_argument("--detailed_data", type=str, required=True)  # bool
    parser.add_argument("--fracture_window", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)

    parser.add_argument("--x_erate", type=float, required=True)
    parser.add_argument("--y_erate", type=float, required=True)
    parser.add_argument("--z_erate", type=float, required=True)
    parser.add_argument("--xy_erate", type=float, required=True)
    parser.add_argument("--xz_erate", type=float, required=True)
    parser.add_argument("--yz_erate", type=float, required=True)

    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--accept_dupes", type=str, required=True)  # bool
    parser.add_argument("--angle_testing", type=str, required=True)  # bool
    parser.add_argument("--num_procs", type=int, required=True)

    parser.add_argument("--repeat_sim", type=int, default=None)


    args = parser.parse_args()

    sheet = GrapheneSheet(args.sheet_path, args.x_atoms, args.y_atoms)

    test = Simulation(comm=comm, rank=rank, sheet=sheet, num_procs=args.num_procs,
                 x_erate=args.x_erate, y_erate=args.y_erate, z_erate=args.z_erate, 
                 xy_erate=args.xy_erate, xz_erate=args.xz_erate, yz_erate=args.yz_erate, 
                 sim_length=args.sim_length, timestep=args.timestep, thermo=args.thermo, 
                 defects=args.defects, defect_random_seed=args.defect_random_seed,
                 makeplots=str2bool(args.makeplots), detailed_data=str2bool(args.detailed_data), fracture_window=args.fracture_window, 
                 theta=args.theta, storage_path=args.storage_path, accept_dupes=str2bool(args.accept_dupes), angle_testing=str2bool(args.angle_testing),
                 repeat_sim=args.repeat_sim)


def str2bool(s):
    if s.lower() == "true" or s.lower() == "yes" or s.lower() == "y" or s.lower() == "t":
        return True
    else:
        return False



# initialize core usage etc.
def initialize_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank

if __name__ == '__main__':
    main()