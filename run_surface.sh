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
    python3 - <<END
import numpy as np
theta = np.deg2rad($THETA)
cos2, sin2, sincos = np.cos(theta)**2, np.sin(theta)**2, np.sin(theta)*np.cos(theta)

erate_1 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
erate_2 = [0, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

for e1, e2 in zip(erate_1, erate_2):
    x  = e1 * cos2 + e2 * sin2
    y  = e2 * cos2 + e1 * sin2
    xy = (e1 - e2) * sincos
    print(f"{x} {y} {xy}")
END
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

    echo "Submitting job: x=$x_erate y=$y_erate xy=$xy_erate"
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done
