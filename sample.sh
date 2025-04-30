#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=76
#SBATCH --job-name=dcc_YourJobName
#SBATCH --partition=common
#SBATCH --mem-per-cpu=1234MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOURS@duke.edu
#SBATCH -o YourOutDoc_%j.out
# SBATCH -d afterany:20548701 (for dependent launch)

module purge

echo "Start: $(date)"
echo "cwd: $(pwd)"

mpiexec YourCommand

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