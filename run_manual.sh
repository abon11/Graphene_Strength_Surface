#!/bin/bash
# Run one specific simulation manually, specifically for fracture events
# As you can see this has turned into a "i need to run this specific batch of sims" script
# set -euo pipefail
EMAIL="avb25@duke.edu"

# Configuration
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=14

x_erate=-0.0001
y_erate=0.001
xy_erate=0.0

# SHEET_PATH="${SHEET_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1}"
# X_ATOMS="${X_ATOMS:-60}"
# Y_ATOMS="${Y_ATOMS:-60}"
# DEFECTS="${DEFECTS:-"None"}"
# DEFECTS="${DEFECTS:-"{\"SV\": 0.5}"}"
DEFECTS="${DEFECTS:-"{\"DV\": 0.25, \"SV\": 0.25}"}"

DEFECT_RANDOM_SEED="${DEFECT_RANDOM_SEED:-2}"
SIM_LENGTH="${SIM_LENGTH:-3000000}"
ACCEPT_DUPES="${ACCEPT_DUPES:-false}"
TIMESTEP="${TIMESTEP:-0.0005}"
THERMO="${THERMO:-1000}"
DETAILED_DATA="${DETAILED_DATA:-false}"
ANGLE_TESTING="${ANGLE_TESTING:-false}"
THETA="${THETA:-0}"
FRACTURE_WINDOW="${FRACTURE_WINDOW:-10}"
STORAGE_PATH="${STORAGE_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/size_tests}"

submit_simulation() {
    local repeat_sim=$1

    if [ -n "$repeat_sim" ]; then
        sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X_ATOMS" "$Y_ATOMS" "$DEFECTS" "$DEFECT_RANDOM_SEED" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$DETAILED_DATA" "$THETA" "$FRACTURE_WINDOW" "$STORAGE_PATH" "$ACCEPT_DUPES" "$ANGLE_TESTING" "$x_erate" "$y_erate" "$xy_erate" "$repeat_sim"
        echo "REPEATING SIM $repeat_sim."
    else
        sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X_ATOMS" "$Y_ATOMS" "$DEFECTS" "$DEFECT_RANDOM_SEED" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$DETAILED_DATA" "$THETA" "$FRACTURE_WINDOW" "$STORAGE_PATH" "$ACCEPT_DUPES" "$ANGLE_TESTING" "$x_erate" "$y_erate" "$xy_erate"
        echo "Starting sheet with: x_erate=$x_erate, y_erate=$y_erate, xy_erate=$xy_erate, SHEET_PATH=$SHEET_PATH, X_ATOMS=$X_ATOMS, Y_ATOMS=$Y_ATOMS, DEFECTS=$DEFECTS, DEFECT_RANDOM_SEED=$DEFECT_RANDOM_SEED, ACCEPT_DUPES=$ACCEPT_DUPES, SIM_LENGTH=$SIM_LENGTH, TIMESTEP=$TIMESTEP, THERMO=$THERMO, DETAILED_DATA=$DETAILED_DATA, ANGLE_TESTING=$ANGLE_TESTING, THETA=$THETA, FRACTURE_WINDOW=$FRACTURE_WINDOW, STORAGE_PATH=$STORAGE_PATH"

    fi
}

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

send_email_notification() {
    local nums=$1
    echo "Sent $nums x $nums" | mail -s "HPC Job Notification" "$EMAIL"
}

# X_ATOMS="${X_ATOMS:-60}"
# Y_ATOMS="${Y_ATOMS:-20}"
# SHEET_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.${X_ATOMS}_${Y_ATOMS}"
# submit_simulation

# submit_simulation 2935 # put sim_id number here if you want it to repeat a sim

# submit_simulation 2926
# submit_simulation 36907







x_erate=-0.0001
y_erate=0.001

for (( X_ATOMS=40; X_ATOMS<=150; X_ATOMS+=10 )); do
  for (( Y_ATOMS=20; Y_ATOMS<=20; Y_ATOMS+=10 )); do

    # Skip when both X and Y are 100 or less
    # if (( X_ATOMS <= 100 && Y_ATOMS <= 100 )); then
    #   continue
    # fi
    # if (( X_ATOMS == 90 && Y_ATOMS <= 130 )); then
    #   continue
    # fi
  
    # # skip forbidden combinations
    # if [[ ( "$X_ATOMS" -eq 20 && "$Y_ATOMS" -eq 20 ) \
    #    || ( "$X_ATOMS" -eq 20 && "$Y_ATOMS" -eq 30 ) \
    #    || ( "$X_ATOMS" -eq 30 && "$Y_ATOMS" -eq 20 ) ]]; then
    #   continue
    # fi


    count=$(python3 - <<PY
import pandas as pd
from filter_csv import filter_data

df = pd.read_csv("${STORAGE_PATH}/all_simulations.csv")
filtered_df = filter_data(
    df,
    exact_filters={
        "Num Atoms x": ${X_ATOMS},
        "Num Atoms y": ${Y_ATOMS},
        "Strain Rate x": ${x_erate},
        "Strain Rate y": ${y_erate},
    },
    suppress_message=True,
    remove_nones=True,
    remove_dupes=False,
)

print(len(filtered_df))
PY
)

    if (( count >= 20 )); then
      echo "Skipping X=$X_ATOMS Y=$Y_ATOMS (already have $count runs)"
      continue
    fi

    SHEET_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.${X_ATOMS}_${Y_ATOMS}"

    for DEFECT_RANDOM_SEED in $(seq 1 1 20); do

      echo "Queued params: X=$X_ATOMS Y=$Y_ATOMS seed=$DEFECT_RANDOM_SEED SHEET_PATH=$SHEET_PATH"

      while true; do
        if (( $(count_jobs) < 28 )); then
          submit_simulation
            if [[ "$X_ATOMS" -eq "$Y_ATOMS" && "$DEFECT_RANDOM_SEED" -eq 1 ]]; then
              send_email_notification "$X_ATOMS"
            fi
          break
        fi
        sleep 20
      done
    done
  done
done







# for (( X_ATOMS=20; X_ATOMS<=150; X_ATOMS+=10 )); do
#   for (( Y_ATOMS=20; Y_ATOMS<=150; Y_ATOMS+=10 )); do

#     # Skip when both X and Y are 100 or less
#     if (( X_ATOMS <= 100 && Y_ATOMS <= 100 )); then
#       continue
#     fi
  
#     # # skip forbidden combinations
#     # if [[ ( "$X_ATOMS" -eq 20 && "$Y_ATOMS" -eq 20 ) \
#     #    || ( "$X_ATOMS" -eq 20 && "$Y_ATOMS" -eq 30 ) \
#     #    || ( "$X_ATOMS" -eq 30 && "$Y_ATOMS" -eq 20 ) ]]; then
#     #   continue
#     # fi

#     SHEET_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.${X_ATOMS}_${Y_ATOMS}"

#     for DEFECT_RANDOM_SEED in $(seq 1 1 20); do

#       echo "Queued params: X=$X_ATOMS Y=$Y_ATOMS seed=$DEFECT_RANDOM_SEED SHEET_PATH=$SHEET_PATH"

#       while true; do
#         if (( $(count_jobs) < 28 )); then
#           submit_simulation
#             if [[ "$X_ATOMS" -eq "$Y_ATOMS" && "$DEFECT_RANDOM_SEED" -eq 1 ]]; then
#               send_email_notification "$X_ATOMS"
#             fi
#           break
#         fi
#         sleep 20
#       done
#     done
#   done
# done
