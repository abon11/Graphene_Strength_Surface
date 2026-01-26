#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --job-name=one_sim_%j
#SBATCH --partition=scavenger
#SBATCH --mem-per-cpu=2GB
#SBATCH --output=/dev/null
# SBATCH -o one_relax_%j.out

# Parse inputs
nprocs=$1
sheet_path=$2
x_atoms=$3
y_atoms=$4
sim_length=$5
timestep=$6
thermo=$7
nvt_percentage=$8
detailed_data=$9

mpiexec -n "$nprocs" python3 one_relaxation.py \
    --sheet_path "$sheet_path" \
    --x_atoms "$x_atoms" \
    --y_atoms "$y_atoms" \
    --sim_length "$sim_length" \
    --timestep "$timestep" \
    --thermo "$thermo" \
    --nvt_percentage "$nvt_percentage" \
    --detailed_data "$detailed_data"
