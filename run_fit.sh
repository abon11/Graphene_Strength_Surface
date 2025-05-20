#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=model_train
#SBATCH --partition=scavenger
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avb25@duke.edu
#SBATCH -o model_train%j.out
# SBATCH -d afterany:20548701 (for dependent launch)

module purge

echo "Start: $(date)"
echo "cwd: $(pwd)"

python3 stress_strain_map.py

echo "End: $(date)"