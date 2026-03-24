#!/bin/bash
# The most simple run script for setting up with slurm

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=solo_test
#SBATCH --partition=scavenger
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avb25@duke.edu
#SBATCH -o solo_test%j.out
# SBATCH -d afterany:20548701 (for dependent launch)

module purge

echo "Start: $(date)"
echo "cwd: $(pwd)"

# mpiexec -n 16 lmp -in in.rotate
# python3 make_surface_slurm.py --nproc 16 --detailed_data true --defect_random_seed 13
# mpiexec -n 16 python3 one_sim.py --sheet_path /hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1 --x_atoms 60 --y_atoms 60 --defect_type None --defect_perc 0 --defect_random_seed 13 --sim_length 10000000 --timestep 0.0005 --thermo 200 --makeplots false --theta 90 --detailed_data true --fracture_window 10 --accept_dupes true --storage_path /hpc/home/avb25/Graphene_Strength_Surface/simulation_data/defected_data --num_procs 16 --x_erate 3.749399456654644e-36 --y_erate 0.001 --z_erate 0 --xy_erate 6.123233995736766e-20 --xz_erate 0 --yz_erate 0
mpiexec -n 16 python3 one_sim.py --sheet_path /hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1 --x_atoms 60 --y_atoms 60 --defect_type None --defect_perc 0 --defect_random_seed 1 --sim_length 25000 --accept_dupes true --timestep 0.0005 --thermo 1000 --makeplots false --detailed_data true --angle_testing true --theta 0 --fracture_window 10 --storage_path /hpc/home/avb25/Graphene_Strength_Surface/simulation_data/angle_testing --num_procs 16 --x_erate 0.000747335 --y_erate 0 --z_erate 0 --xy_erate 0.000648058 --xz_erate 0 --yz_erate 0

echo "End: $(date)"

# maximum allocation for single user on common: 400cpu + 1.5TB

## Some useful command
##  List all my jobs' info
# squeue -u NetID -o "%.8i %.9P %.40j %.5u %.8T %.8M  %.16R %.4C %.5m"

## Storage address
# /cwork/NetID/

##  To learn about partition common and scavanger
## learn about common in general
# scontrol show partition common
## learn abt node 01
# scontrol show node=dcc-core-01
##  abt status of each core
# sinfo -p common -N -o "%C %N %T %m" 
# sinfo -p scavenger -N -o "%C %N %T %m" 
## how many jobs in front of me
# squeue -p common -t PENDING --sort=+p -o "%.18i" | awk '$1==JOBID  {exit} {print NR}'
# squeue -p common -t PENDING --sort=+p -o "%.18i" | awk -v job_id=JOBID '$1==job_id {print NR-1; exit}'