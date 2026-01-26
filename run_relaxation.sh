#!/bin/bash

EMAIL="avb25@duke.edu"
SLURM_SCRIPT="./one_relaxation.sh"   # must exist
MAX_JOBS=25                          # throttle level
CORES_PER_JOB=14                     # unused but leaving for clarity

SIM_LENGTH="${SIM_LENGTH:-120000}"
TIMESTEP="${TIMESTEP:-0.0005}"
THERMO="${THERMO:-1000}"
NVT_PERCENTAGE="${NVT_PERCENTAGE:-0.2}"
DETAILED_DATA="${DETAILED_DATA:-false}"

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

submit_simulation() {
    local X=$1
    local Y=$2
    local SHEET_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.${X}_${Y}"

    echo "Submitting X=${X}, Y=${Y}"
    sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X" "$Y" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$NVT_PERCENTAGE" "$DETAILED_DATA"
}

send_email_notification() {
    local nums=$1
    echo "Sent $nums x $nums" | mail -s "HPC Job Notification" "$EMAIL"
}

# pairs=("140,90" "130,110" "140,110" "130,50" "150,110" "140,150" "150,140" "150,130" "20,120" "110,20" "110,30"
# "120,20" "120,40" "130,20" "140,20" "150,20" "130,70" "110,100" "150,70" "140,30" "150,120" "140,70"
# "150,40" "150,30" "140,50" "140,80" "130,150" "110,60" "140,140" "150,100" "150,50" "150,80" "150,150")
# pairs=("110,20")  # add pairs here
pairs=("130,150")

for p in "${pairs[@]}"; do
  IFS=, read -r X_ATOMS Y_ATOMS <<< "$p"

  # Throttle jobs (same logic as your original)
  while true; do
    current_jobs=$(count_jobs)
    if (( current_jobs < MAX_JOBS )); then
      submit_simulation "$X_ATOMS" "$Y_ATOMS"
      if [[ "$X_ATOMS" -eq "$Y_ATOMS" ]]; then
        send_email_notification "$X_ATOMS"
      fi
      break
    else
      sleep 20
    fi
  done
done


#!/usr/bin/env bash

# Assumes these functions/vars are defined elsewhere:
#   count_jobs   -> prints number of currently running jobs
#   submit_simulation X Y
#   send_email_notification X
#   MAX_JOBS     -> integer limit

# Loop X and Y from 20 to 150 in steps of 10
# for (( X=20; X<=150; X+=10 )); do
#   for (( Y=20; Y<=150; Y+=10 )); do

#     # Skip when both X and Y are 100 or less
#     if (( X <= 100 && Y <= 100 )); then
#       continue
#     fi

#     # Throttle jobs (your original logic)
#     while true; do
#       current_jobs=$(count_jobs)
#       if (( current_jobs < MAX_JOBS )); then
#         submit_simulation "$X" "$Y"
#         if (( X == Y )); then
#           send_email_notification "$X"
#         fi
#         break
#       else
#         sleep 20
#       fi
#     done

#   done
# done