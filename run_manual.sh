# Run one specific simulation manually
#!/bin/bash

# Configuration
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=14

x_erate=0.001
y_erate=0.001
xy_erate=0

SHEET_PATH="${SHEET_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1}"
X_ATOMS="${X_ATOMS:-60}"
Y_ATOMS="${Y_ATOMS:-60}"
# DEFECTS="${DEFECTS:-"{\"SV\": 0.1, \"DV\": 0.2}"}"
DEFECTS="${DEFECTS:-"{\"DV\": 0.5}"}"
# DEFECTS="${DEFECTS:-"{\"DV\": 0.25, \"SV\": 0.25}"}"

DEFECT_RANDOM_SEED="${DEFECT_RANDOM_SEED:-54}"
SIM_LENGTH="${SIM_LENGTH:-10000000}"
ACCEPT_DUPES="${ACCEPT_DUPES:-false}"
TIMESTEP="${TIMESTEP:-0.0005}"
THERMO="${THERMO:-1000}"
DETAILED_DATA="${DETAILED_DATA:-true}"
ANGLE_TESTING="${ANGLE_TESTING:-false}"
THETA="${THETA:-30}"
FRACTURE_WINDOW="${FRACTURE_WINDOW:-10}"
STORAGE_PATH="${STORAGE_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/rotation_tests}"

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

# submit_simulation 2935 # put sim_id number here if you want it to repeat a sim

submit_simulation

# export DEFECTS="{\"SV\": 0.5}"

# for seed in $(seq 0 1 30); do
#     export DEFECT_RANDOM_SEED="$seed"
#     while true; do
#         if (( $seed < 25 )); then
#             submit_simulation
#             break
#         fi

#         if (( $(count_jobs) < 25 )); then
#             sleep 20
#             if (( $(count_jobs) < 25 )); then
#                 submit_simulation
#                 sleep 30
#                 break
#             fi
#         fi
#         sleep 60
#     done
# done
