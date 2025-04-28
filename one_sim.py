from mpi4py import MPI
from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np
import argparse
from make_surface import str2bool


def main():
    comm, rank = initialize_rank()

    parser = argparse.ArgumentParser()
    
    # Default parameters
    # parser.add_argument("--sheet_path", type=str, default="/data1/avb25/graphene_sim_data/data_files/data.60_60_rel1")
    # parser.add_argument("--x_atoms", type=int, default=60)
    # parser.add_argument("--y_atoms", type=int, default=60)

    # parser.add_argument("--defect_type", type=str, default="SV")
    # parser.add_argument("--defect_frac", type=float, default=0)
    # parser.add_argument("--defect_random_seed", type=int, default=42)
    # parser.add_argument("--sim_length", type=int, default=100000)
    # parser.add_argument("--timestep", type=float, default=0.0005)
    # parser.add_argument("--thermo", type=int, default=1000)
    # parser.add_argument("--makeplots", type=bool, default=False)
    # parser.add_argument("--detailed_data", type=bool, default=False)
    
    parser.add_argument("--sheet_path", type=str, required=True)
    parser.add_argument("--x_atoms", type=int, required=True)
    parser.add_argument("--y_atoms", type=int, required=True)

    parser.add_argument("--defect_type", type=str, required=True)
    parser.add_argument("--defect_perc", type=float, required=True)
    parser.add_argument("--defect_random_seed", type=int, required=True)
    parser.add_argument("--sim_length", type=int, required=True)
    parser.add_argument("--timestep", type=float, required=True)
    parser.add_argument("--thermo", type=int, required=True)
    parser.add_argument("--makeplots", type=str, required=True)  # bool
    parser.add_argument("--detailed_data", type=str, required=True)  # bool
    parser.add_argument("--fracture_window", type=int, required=True)

    parser.add_argument("--x_erate", type=float, required=True)
    parser.add_argument("--y_erate", type=float, required=True)
    parser.add_argument("--z_erate", type=float, required=True)
    parser.add_argument("--xy_erate", type=float, required=True)
    parser.add_argument("--xz_erate", type=float, required=True)
    parser.add_argument("--yz_erate", type=float, required=True)

    parser.add_argument("--storage_path", type=str, required=True)


    args = parser.parse_args()

    sheet = GrapheneSheet(args.sheet_path, args.x_atoms, args.y_atoms)

    test = Simulation(comm=comm, rank=rank, sheet=sheet,
                 x_erate=args.x_erate, y_erate=args.y_erate, z_erate=args.z_erate, 
                 xy_erate=args.xy_erate, xz_erate=args.xz_erate, yz_erate=args.yz_erate, 
                 sim_length=args.sim_length, timestep=args.timestep, thermo=args.thermo, 
                 defect_type=args.defect_type, defect_perc=args.defect_perc, defect_random_seed=args.defect_random_seed,
                 makeplots=str2bool(args.makeplots), detailed_data=str2bool(args.detailed_data), fracture_window=args.fracture_window, 
                 storage_path=args.storage_path)
    

# initialize core usage etc.
def initialize_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank

if __name__ == '__main__':
    main()