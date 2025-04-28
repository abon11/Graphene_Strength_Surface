from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np
import argparse
import subprocess
import time


def main():

    parser = argparse.ArgumentParser()
    
    # Default parameters
    parser.add_argument("--sheet_path", type=str, default="/data1/avb25/graphene_sim_data/data_files/data.60_60_rel1")
    parser.add_argument("--x_atoms", type=int, default=60)
    parser.add_argument("--y_atoms", type=int, default=60)

    parser.add_argument("--defect_type", type=str, default="SV")
    parser.add_argument("--defect_perc", type=float, default=0)
    parser.add_argument("--defect_random_seed", type=int, default=42)
    parser.add_argument("--sim_length", type=int, default=10000000)
    parser.add_argument("--timestep", type=float, default=0.0005)
    parser.add_argument("--thermo", type=int, default=1000)
    parser.add_argument("--makeplots", type=str, default="false")  # these are strings, then we do str2bool to make them bools
    parser.add_argument("--detailed_data", type=str, default="false")
    parser.add_argument("--fracture_window", type=int, default=10)

    parser.add_argument("--both_directions", type=str, default="true")
    parser.add_argument("--storage_path", type=str, default='/data1/avb25/graphene_sim_data/defected_data')

    parser.add_argument("--nproc", type=int, required=True)  # user specified number of processors to use

    args = parser.parse_args()

    min_cores = 12
    if args.nproc < min_cores:
        raise ValueError(f"Must use at least {min_cores} cores")

    max_jobs = args.nproc // min_cores

    # these are the strain rates that correspond to creating a surface
    x_rates = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    y_rates = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]

    job_args = vars(args).copy()
    cli_args = ' '.join(f"--{key} {value}" for key, value in job_args.items() if key not in ['nproc', 'both_directions'])

    jobs = []
    for i in range(len(x_rates)):
        job_cmd = (
            f"mpiexec -n {min_cores} python3 one_sim.py {cli_args} "
            f"--x_erate {x_rates[i]} --y_erate {y_rates[i]} --z_erate {0} "
            f"--xy_erate {0} --xz_erate {0} --yz_erate {0}"
        )

        jobs.append(job_cmd)

        # flip x and y rates if necessary to get both sides
        if str2bool(args.both_directions):
            if x_rates[i] != y_rates[i]:
                job_cmd = (
                    f"mpiexec -n {min_cores} python3 one_sim.py {cli_args} "
                    f"--x_erate {y_rates[i]} --y_erate {x_rates[i]} --z_erate {0} "
                    f"--xy_erate {0} --xz_erate {0} --yz_erate {0}"
                )
                jobs.append(job_cmd)

    
    active = []
    while jobs or active:
        while len(active) < max_jobs and jobs:
            job_cmd = jobs.pop(0)
            print(f"Launching: {job_cmd}")
            proc = subprocess.Popen(job_cmd, shell=True)
            active.append(proc)

        for proc in active[:]:
            if proc.poll() is not None:
                active.remove(proc)

        time.sleep(1)

    print("All surfaces generated successfully!")


def str2bool(string):
        if isinstance(string, bool):
            return string
        if string.lower() in ('yes', 'true', 't', '1', 'y'):
            return True
        elif string.lower() in ('no', 'false', 'f', '0', 'n'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    main()