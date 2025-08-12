#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --job-name=one_sim_%j
#SBATCH --partition=scavenger
#SBATCH --mem-per-cpu=2GB
#SBATCH --output=/dev/null
#SBATCH -o one_sim_%j.out

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
makeplots=${10}
detailed_data=${11}
theta=${12}
fracture_window=${13}
storage_path=${14}
accept_dupes=${15}
angle_testing=${16}
x_erate=${17}
y_erate=${18}
xy_erate=${19}
repeat_sim=${20}


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
        --makeplots "$makeplots" \
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
        --makeplots "$makeplots" \
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