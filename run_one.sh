#!/bin/bash
# This is called by run_manual.sh, run_surface.sh, run_specific.sh, etc.
# It is the actual script that puts together the command and runs python3 one_sim.py
# Works well with the SLURM system

#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --job-name=one_sim_%j
#SBATCH --partition=scavenger
#SBATCH --mem-per-cpu=2GB
#SBATCH --output=/dev/null
# SBATCH -o one_sim_%j.out

# Parse inputs
nprocs=$1
sheet_path=$2
x_atoms=$3
y_atoms=$4
defects=$5
defect_seed=$6
sim_length=$7
timestep=$8
thermo=$9
detailed_data=${10}
theta=${11}
fracture_window=${12}
storage_path=${13}
accept_dupes=${14}
angle_testing=${15}
x_erate=${16}
y_erate=${17}
xy_erate=${18}
repeat_sim=${19}


echo "Running one_sim with: x="$x_erate" y="$y_erate" xy="$xy_erate" on $nprocs procs"

if [ -n "$repeat_sim" ]; then
    mpiexec -n "$nprocs" python3 one_sim.py \
        --sheet_path "$sheet_path" \
        --x_atoms "$x_atoms" \
        --y_atoms "$y_atoms" \
        --defects "$defects" \
        --defect_random_seed "$defect_seed" \
        --sim_length "$sim_length" \
        --timestep "$timestep" \
        --thermo "$thermo" \
        --detailed_data "$detailed_data" \
        --fracture_window "$fracture_window" \
        --storage_path "$storage_path" \
        --accept_dupes "$accept_dupes" \
        --angle_testing "$angle_testing" \
        --num_procs "$nprocs" \
        --theta "$theta" \
        --x_erate "$x_erate" \
        --y_erate "$y_erate" \
        --z_erate 0 \
        --xy_erate "$xy_erate" \
        --xz_erate 0 \
        --yz_erate 0 \
        --repeat_sim "$repeat_sim"
else
    mpiexec -n "$nprocs" python3 one_sim.py \
        --sheet_path "$sheet_path" \
        --x_atoms "$x_atoms" \
        --y_atoms "$y_atoms" \
        --defects "$defects" \
        --defect_random_seed "$defect_seed" \
        --sim_length "$sim_length" \
        --timestep "$timestep" \
        --thermo "$thermo" \
        --detailed_data "$detailed_data" \
        --fracture_window "$fracture_window" \
        --storage_path "$storage_path" \
        --accept_dupes "$accept_dupes" \
        --angle_testing "$angle_testing" \
        --num_procs "$nprocs" \
        --theta "$theta" \
        --x_erate "$x_erate" \
        --y_erate "$y_erate" \
        --z_erate 0 \
        --xy_erate "$xy_erate" \
        --xz_erate 0 \
        --yz_erate 0
fi