# Configuration
MAX_JOBS_IN_FLIGHT="${MAX_JOBS_IN_FLIGHT:-10}"
SLURM_SCRIPT="./run_one.sh"  # must exist
CORES_PER_JOB=14

SHEET_PATH="${SHEET_PATH:-/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/data_files/data.60_60_rel1}"
X_ATOMS="${X_ATOMS:-60}"
Y_ATOMS="${Y_ATOMS:-60}"
DEFECTS="${DEFECTS:-"{\"SV\": 0.5}"}"
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


export MAX_JOBS_IN_FLIGHT=25
export STORAGE_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/rotation_tests"
export THETA=0
export DETAILED_DATA="True"
export ACCEPT_DUPES="True"

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

submit_job() {
    local x_erate=$1
    local y_erate=$2
    local xy_erate=$3
    sbatch "$SLURM_SCRIPT" "$CORES_PER_JOB" "$SHEET_PATH" "$X_ATOMS" "$Y_ATOMS" "$DEFECTS" "$DEFECT_RANDOM_SEED" "$SIM_LENGTH" "$TIMESTEP" "$THERMO" "$MAKEPLOTS" "$DETAILED_DATA" "$THETA" "$FRACTURE_WINDOW" "$STORAGE_PATH" "$ACCEPT_DUPES" "$ANGLE_TESTING" "$x_erate" "$y_erate" "$xy_erate"
}

for j in $(seq 0 1 25); do
    export DEFECT_RANDOM_SEED="$j"
    # Throttle to MAX_JOBS_IN_FLIGHT
    while (( $(count_jobs) >= MAX_JOBS_IN_FLIGHT )); do
        sleep 29
    done

    x_erate=0.001
    y_erate=0.0
    xy_erate=0.0

    echo "Submitting specific job: Seed=$DEFECT_RANDOM_SEED, x=$x_erate y=$y_erate xy=$xy_erate"
    submit_job "$x_erate" "$y_erate" "$xy_erate"
done