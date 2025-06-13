#!/bin/bash

# Configuration
MAX_JOBS_IN_FLIGHT=25
TOTAL_SIMS=5000
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=12


SHEET_PATH="${SHEET_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1}"
X_ATOMS="${X_ATOMS:-60}"
Y_ATOMS="${Y_ATOMS:-60}"
DEFECT_TYPE="${DEFECT_TYPE:-None}"
DEFECT_PERC="${DEFECT_PERC:-0}"
DEFECT_RANDOM_SEED="${DEFECT_RANDOM_SEED:-1}"
SIM_LENGTH="${SIM_LENGTH:-25000}"
ACCEPT_DUPES="${ACCEPT_DUPES:-false}"
TIMESTEP="${TIMESTEP:-0.0005}"
THERMO="${THERMO:-1000}"
MAKEPLOTS="${MAKEPLOTS:-false}"
DETAILED_DATA="${DETAILED_DATA:-false}"
ANGLE_TESTING="${ANGLE_TESTING:-true}"
THETA="${THETA:-0}"
FRACTURE_WINDOW="${FRACTURE_WINDOW:-10}"
STORAGE_PATH="${STORAGE_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/angle_testing}"

echo "Starting sheet with: SHEET_PATH=$SHEET_PATH, X_ATOMS=$X_ATOMS, Y_ATOMS=$Y_ATOMS, DEFECT_TYPE=$DEFECT_TYPE, DEFECT_PERC=$DEFECT_PERC, DEFECT_RANDOM_SEED=$DEFECT_RANDOM_SEED, ACCEPT_DUPES=$ACCEPT_DUPES, SIM_LENGTH=$SIM_LENGTH, TIMESTEP=$TIMESTEP, THERMO=$THERMO, MAKEPLOTS=$MAKEPLOTS, DETAILED_DATA=$DETAILED_DATA, ANGLE_TESTING=$ANGLE_TESTING THETA=$THETA, FRACTURE_WINDOW=$FRACTURE_WINDOW, STORAGE_PATH=$STORAGE_PATH"

# Utility: Generate random float between min and max
rand_float() {
    awk -v min="$1" -v max="$2" 'BEGIN {srand(); print min + rand() * (max - min)}'
}

rand_erates() {
    for _ in {1..3}; do
        seed=$(od -An -N2 -i /dev/random | tr -d ' ')
        awk -v seed=$seed 'BEGIN {
            srand(seed)
            r = rand()
            if (r < 0.02) {
                printf "0.00000000 "
            } else if (r > 0.95) {
                printf "0.00100000 "
            } else {
                printf "%.8f ", 0.0001 + rand() * (0.001 - 0.0001)
            }
        }'
    done
    echo
}


send_email_notification() {
    echo "Angle sim $1 has been submitted" | mail -s "HPC Job Notification" avb25@duke.edu
}

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
for ((i = 1; i <= TOTAL_SIMS; i++)); do
    # Generate random strain rates
    read -r x_erate y_erate xy_erate <<< "$(rand_erates)"

    # Throttle job submissions
    while (( $(count_jobs) >= MAX_JOBS_IN_FLIGHT )); do
        sleep 30
    done

    echo "Submitting job #$i: x=$x_erate y=$y_erate xy=$xy_erate"

    if (( i % 1000 == 0 )); then
        send_email_notification "$i"
    fi
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done


