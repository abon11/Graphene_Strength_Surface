#!/bin/bash

# Configuration
MAX_JOBS_IN_FLIGHT="${MAX_JOBS_IN_FLIGHT:-10}"
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=12

SHEET_PATH="${SHEET_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1}"
X_ATOMS="${X_ATOMS:-60}"
Y_ATOMS="${Y_ATOMS:-60}"
DEFECT_TYPE="${DEFECT_TYPE:-SV}"
DEFECT_PERC="${DEFECT_PERC:-0.5}"
DEFECT_RANDOM_SEED="${DEFECT_RANDOM_SEED:-12}"
SIM_LENGTH="${SIM_LENGTH:-10000000}"
ACCEPT_DUPES="${ACCEPT_DUPES:-false}"
TIMESTEP="${TIMESTEP:-0.0005}"
THERMO="${THERMO:-1000}"
MAKEPLOTS="${MAKEPLOTS:-false}"
DETAILED_DATA="${DETAILED_DATA:-false}"
ANGLE_TESTING="${ANGLE_TESTING:-false}"
THETA="${THETA:-0}"
FRACTURE_WINDOW="${FRACTURE_WINDOW:-10}"
STORAGE_PATH="${STORAGE_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/defected_data}"

echo "Starting sheet with: SHEET_PATH=$SHEET_PATH, X_ATOMS=$X_ATOMS, Y_ATOMS=$Y_ATOMS, DEFECT_TYPE=$DEFECT_TYPE, DEFECT_PERC=$DEFECT_PERC, DEFECT_RANDOM_SEED=$DEFECT_RANDOM_SEED, ACCEPT_DUPES=$ACCEPT_DUPES, SIM_LENGTH=$SIM_LENGTH, TIMESTEP=$TIMESTEP, THERMO=$THERMO, MAKEPLOTS=$MAKEPLOTS, DETAILED_DATA=$DETAILED_DATA, ANGLE_TESTING=$ANGLE_TESTING THETA=$THETA, FRACTURE_WINDOW=$FRACTURE_WINDOW, STORAGE_PATH=$STORAGE_PATH"


rotate_rates() {
    RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    local theta_int=$(printf "%.0f" "$THETA")

    for ratio in "${RATIOS[@]}"; do
        if [[ "$theta_int" == "0" ]]; then
            x=0.001
            y=$(awk -v r="$ratio" 'BEGIN { printf "%.4f\n", r * 0.001 }')
            xy=0.0
            echo "$x $y $xy"
        elif [[ "$theta_int" == "90" ]]; then
            y=0.001
            x=$(awk -v r="$ratio" 'BEGIN { printf "%.4f\n", r * 0.001 }')
            xy=0.0
            echo "$x $y $xy"
        else
            # Call the Python inverse model script for general theta
            strain=$(python3 inverse_strain.py --ratio "$ratio" --theta "$THETA")
            echo "$strain"
        fi
    done
}

# Read rotated strain rates
mapfile -t STRAIN_LIST < <(rotate_rates)

submit_job() {
    local x_erate=$1
    local y_erate=$2
    local xy_erate=$3
    sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X_ATOMS" "$Y_ATOMS" "$DEFECT_TYPE" "$DEFECT_PERC" "$DEFECT_RANDOM_SEED" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$MAKEPLOTS" "$DETAILED_DATA" "$THETA" "$FRACTURE_WINDOW" "$STORAGE_PATH" "$ACCEPT_DUPES" "$ANGLE_TESTING" "$x_erate" "$y_erate" "$xy_erate"
}

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

# Main loop
for strain in "${STRAIN_LIST[@]}"; do
    read -r x_erate y_erate xy_erate <<< "$strain"

    # Throttle to MAX_JOBS_IN_FLIGHT
    while (( $(count_jobs) >= MAX_JOBS_IN_FLIGHT )); do
        sleep 29
    done

    echo "Submitting job: Seed=$DEFECT_RANDOM_SEED x=$x_erate y=$y_erate xy=$xy_erate"
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done
