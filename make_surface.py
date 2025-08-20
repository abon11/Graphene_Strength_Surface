""" This is used if you want to make a strength surface not using SLURM """

import argparse
import subprocess
import time
import local_config
import sys
import numpy as np

MIN_CORES_PER_JOB = 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=True, help="Total cores allocated to this SLURM job")
    parser.add_argument("--sheet_path", type=str, default=f"{local_config.DATA_DIR}/data_files/data.60_60_rel1")
    parser.add_argument("--x_atoms", type=int, default=60)
    parser.add_argument("--y_atoms", type=int, default=60)
    parser.add_argument("--defects", type=str, default="None")
    parser.add_argument("--defect_random_seed", type=int, default=42)
    parser.add_argument("--sim_length", type=int, default=10000000)
    parser.add_argument("--timestep", type=float, default=0.0005)
    parser.add_argument("--thermo", type=int, default=1000)
    parser.add_argument("--detailed_data", type=str, default="false")
    parser.add_argument("--fracture_window", type=int, default=10)
    parser.add_argument("--storage_path", type=str, default=f"{local_config.DATA_DIR}/defected_data")
    parser.add_argument("--both_directions", type=str, default="true")
    parser.add_argument("--theta", type=float, default=0.0)
    return parser.parse_args()


def str2bool(s):
    return s.lower() in ("true", "1", "yes", "y")


def build_job_command(base_args, x_erate, y_erate, xy_erate, cores):
    return [
        "mpiexec", "-n", str(cores), 
        "python3", "one_sim.py",
        "--sheet_path", base_args.sheet_path,
        "--x_atoms", str(base_args.x_atoms),
        "--y_atoms", str(base_args.y_atoms),
        "--defects", base_args.defects,
        "--defect_random_seed", str(base_args.defect_random_seed),
        "--sim_length", str(base_args.sim_length),
        "--timestep", str(base_args.timestep),
        "--thermo", str(base_args.thermo),
        "--detailed_data", base_args.detailed_data,
        "--fracture_window", str(base_args.fracture_window),
        "--theta", base_args.theta,
        "--storage_path", base_args.storage_path,
        "--num_procs", str(cores),
        "--x_erate", str(x_erate),
        "--y_erate", str(y_erate),
        "--z_erate", "0",
        "--xy_erate", str(xy_erate),
        "--xz_erate", "0",
        "--yz_erate", "0"
    ]

# given strain rates in principal directions and theta, it returns the x, y, and shear strain rates necessary to recreate it with sigma_1 being aligned with theta
def rotate_load(erate_1, erate_2, theta_deg):
    theta = np.deg2rad(theta_deg)
    
    # convert lists to numpy arrays
    erate_1 = np.array(erate_1)
    erate_2 = np.array(erate_2)
    
    cos2 = np.cos(theta)**2
    sin2 = np.sin(theta)**2
    sincos = np.sin(theta) * np.cos(theta)
    
    x_rates = erate_1 * cos2 + erate_2 * sin2
    y_rates = erate_2 * cos2 + erate_1 * sin2
    xy_rates = (erate_1 - erate_2) * sincos

    return x_rates, y_rates, xy_rates


def main():
    args = parse_args()
    max_jobs = args.nproc // MIN_CORES_PER_JOB

    erate_1 = [1e-3] * 11
    erate_2 = [0, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]

    x_rates, y_rates, xy_rates = rotate_load(erate_1, erate_2, args.theta)

    job_queue = []
    for i in range(len(x_rates)):
        job_queue.append(build_job_command(args, x_rates[i], y_rates[i], xy_rates[i], MIN_CORES_PER_JOB))
        if str2bool(args.both_directions) and x_rates[i] != y_rates[i]:
            job_queue.append(build_job_command(args, y_rates[i], x_rates[i], xy_rates[i], MIN_CORES_PER_JOB))

    active = []
    while job_queue or active:
        while len(active) < max_jobs and job_queue:
            cmd = job_queue.pop(0)
            print("Launching:", " ".join(cmd), flush=True)
            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            active.append((proc, cmd))

        for proc, cmd in active[:]:
            if proc.poll() is not None:
                print(f"Finished: {' '.join(cmd)}\nReturn code: {proc.returncode}", flush=True)
                active.remove((proc, cmd))

        time.sleep(2)

    print("All simulations completed successfully.", flush=True)


if __name__ == '__main__':
    main()
