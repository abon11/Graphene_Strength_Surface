#!/bin/bash
#SBATCH --job-name=dcc_TestRun
#SBATCH --output=TestRun_%j.out
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000MB
#SBATCH --time=00:05:00

module purge

echo "Running basic test job"
hostname
date
sleep 30
date
echo "Finished basic test job"
