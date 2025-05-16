#!/bin/bash

# Configuration
MAX_JOBS_IN_FLIGHT=10
TOTAL_SIMS=10000
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
    local min_mag=1e-4
    local max_mag=1e-3
    local p_zero=0.05

    # Seed once for this function call
    seed=$(od -An -N2 -i /dev/random | tr -d ' ')
    
    read x_raw y_raw xy_raw <<< $(awk -v seed=$seed 'BEGIN {
        srand(seed)
        print rand(), rand(), rand()
    }')

    # Randomly zero out components (note: must also fix srand in these)
    seed=$(od -An -N2 -i /dev/random | tr -d ' ')
    read zero_x zero_y zero_xy <<< $(awk -v seed=$seed -v p=$p_zero 'BEGIN {
        srand(seed)
        print (rand()<p)?0:1, (rand()<p)?0:1, (rand()<p)?0:1
    }')

    x_raw=$(awk -v x=$x_raw -v f=$zero_x 'BEGIN {print x * f}')
    y_raw=$(awk -v y=$y_raw -v f=$zero_y 'BEGIN {print y * f}')
    xy_raw=$(awk -v z=$xy_raw -v f=$zero_xy 'BEGIN {print z * f}')

    # Ensure at least one component is non-zero
    total=$(awk -v x=$x_raw -v y=$y_raw -v z=$xy_raw 'BEGIN {print x + y + z}')
    if (( $(awk -v t=$total 'BEGIN {print (t == 0)}') )); then
        x_raw=1
        y_raw=0
        xy_raw=0
    fi

    norm=$(awk -v x=$x_raw -v y=$y_raw -v z=$xy_raw 'BEGIN {print sqrt(x*x + y*y + z*z)}')
    x_unit=$(awk -v x=$x_raw -v n=$norm 'BEGIN {print x/n}')
    y_unit=$(awk -v y=$y_raw -v n=$norm 'BEGIN {print y/n}')
    xy_unit=$(awk -v z=$xy_raw -v n=$norm 'BEGIN {print z/n}')

    # Magnitude
    seed=$(od -An -N2 -i /dev/random | tr -d ' ')
    mag=$(awk -v seed=$seed -v min=$min_mag -v max=$max_mag 'BEGIN {
        srand(seed)
        print min + rand() * (max - min)
    }')

    x_erate=$(awk -v u=$x_unit -v m=$mag 'BEGIN {print u * m}')
    y_erate=$(awk -v u=$y_unit -v m=$mag 'BEGIN {print u * m}')
    xy_erate=$(awk -v u=$xy_unit -v m=$mag 'BEGIN {print u * m}')

    echo "$x_erate $y_erate $xy_erate"
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
        sleep_time=$((30 + RANDOM % 31))  # Sleep 30–60 sec
        sleep $sleep_time
    done

    echo "Submitting job #$i: x=$x_erate y=$y_erate xy=$xy_erate"
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done
