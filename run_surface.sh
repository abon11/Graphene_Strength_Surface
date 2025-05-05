#!/bin/bash

# Configuration
MAX_JOBS_IN_FLIGHT=4
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=16

SHEET_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1"
X_ATOMS=60
Y_ATOMS=60
DEFECT_TYPE="SV"
DEFECT_PERC=0.5
DEFECT_RANDOM_SEED=12
SIM_LENGTH=10000000
TIMESTEP=0.0005
THERMO=1000
MAKEPLOTS="false"
DETAILED_DATA="true"
THETA=0  # degrees
FRACTURE_WINDOW=10
STORAGE_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/defected_data"


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
    sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X_ATOMS" "$Y_ATOMS" "$DEFECT_TYPE" "$DEFECT_PERC" "$DEFECT_RANDOM_SEED" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$MAKEPLOTS" "$DETAILED_DATA" "$THETA" "$FRACTURE_WINDOW" "$STORAGE_PATH" "$x_erate" "$y_erate" "$xy_erate"
}

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

# Main loop
for strain in "${STRAIN_LIST[@]}"; do
    read -r x_erate y_erate xy_erate <<< "$strain"

    # Throttle to MAX_JOBS_IN_FLIGHT
    while (( $(count_jobs) >= MAX_JOBS_IN_FLIGHT )); do
        sleep 10
    done

    echo "Submitting job: x=$x_erate y=$y_erate xy=$xy_erate"
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done
