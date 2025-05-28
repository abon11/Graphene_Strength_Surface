#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
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
defect_type=$5
defect_perc=$6
defect_seed=$7
sim_length=$8
timestep=$9
thermo=${10}
makeplots=${11}
detailed_data=${12}
theta=${13}
fracture_window=${14}
storage_path=${15}
accept_dupes=${16}
angle_testing=${17}
x_erate=${18}
y_erate=${19}
xy_erate=${20}


echo "Running one_sim with: x="$x_erate" y="$y_erate" xy="$xy_erate" on $nprocs procs"

mpiexec -n "$nprocs" python3 one_sim.py \
    --sheet_path "$sheet_path" \
    --x_atoms "$x_atoms" \
    --y_atoms "$y_atoms" \
    --defect_type "$defect_type" \
    --defect_perc "$defect_perc" \
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
