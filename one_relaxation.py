from mpi4py import MPI
from deform_graphene import Relaxation
from one_sim import initialize_rank, str2bool
import socket, os
import argparse


def main():
    comm, rank = initialize_rank()

    print(f"Running on {socket.gethostname()}, PID={os.getpid()}, MPI size={os.environ.get('OMPI_COMM_WORLD_SIZE', '?')}")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sheet_path", type=str, required=True)
    parser.add_argument("--x_atoms", type=int, required=True)
    parser.add_argument("--y_atoms", type=int, required=True)
    parser.add_argument("--sim_length", type=int, required=True)
    parser.add_argument("--timestep", type=float, required=True)
    parser.add_argument("--thermo", type=int, required=True)
    parser.add_argument("--nvt_percentage", type=float, required=True)
    parser.add_argument("--detailed_data", type=str, required=True)  # bool

    args = parser.parse_args()
    Relaxation(comm, rank, sheet_path=args.sheet_path, x_atoms=args.x_atoms, y_atoms=args.y_atoms, 
               sim_length=args.sim_length, timestep=args.timestep, thermo=args.thermo, 
               nvt_percentage=args.nvt_percentage, detailed_data=str2bool(args.detailed_data))
    

if __name__ == "__main__":
    main()