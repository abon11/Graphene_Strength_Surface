#!/bin/bash
set -euo pipefail

# ======== Config (used unless overridden below) ========
export MAX_JOBS_IN_FLIGHT=25
DEFAULT_STORAGE_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/rotation_tests"
DEFAULT_DETAILED_DATA="False"
DEFAULT_STRAIN_TABLE="strain_table.csv"
RUN_SCRIPT="./run_surface.sh"
EMAIL="avb25@duke.edu"

# ======== Functions ========

count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

send_email_notification() {
    echo "Seed $1 has been submitted" | mail -s "HPC Job Notification" "$EMAIL"
}

submit_job() {
    local seed="$1"
    local theta="$2"
    local defects="$3"

    while true; do
        if (( $(count_jobs) < MAX_JOBS_IN_FLIGHT )); then
            sleep 60
            if (( $(count_jobs) < MAX_JOBS_IN_FLIGHT )); then
                echo "SUBMITTING: SEED=$seed, THETA=$theta, DEFECTS=$defects"
                DEFECT_RANDOM_SEED="$seed" \
                THETA="$theta" \
                DEFECTS="$defects" \
                STORAGE_PATH="$DEFAULT_STORAGE_PATH" \
                DETAILED_DATA="$DEFAULT_DETAILED_DATA" \
                STRAIN_TABLE="$DEFAULT_STRAIN_TABLE" \
                bash "$RUN_SCRIPT" &
                sleep 60
                break
            fi
        fi
        sleep 100
    done
}

launch_block() {
    local defect_str="$1"
    local start_seed="$2"
    local end_seed="$3"
    local start_theta="$4"
    local notify_every="$5"

    echo "Launching block: $defect_str"

    for seed in $(seq "$start_seed" 1 "$end_seed"); do
        (( seed % notify_every == 0 )) && send_email_notification "$seed"

        for theta in $(seq "$start_theta" 10 90); do
            submit_job "$seed" "$theta" "$defect_str"
        done
    done
}

# # ======== Optional: Initial Job with Different Defaults ========
# DEFECT_RANDOM_SEED=97 THETA=60 DEFECTS="{\"DV\": 0.25, \"SV\": 0.25}" \
# STORAGE_PATH="$DEFAULT_STORAGE_PATH" \
# DETAILED_DATA="$DEFAULT_DETAILED_DATA" \
# STRAIN_TABLE="$DEFAULT_STRAIN_TABLE" \
# bash "$RUN_SCRIPT" &

# ======== Launch All Blocks ========

# Mixed defects block
launch_block "{\"DV\": 0.25, \"SV\": 0.25}" 97 97 60 1
launch_block "{\"DV\": 0.25, \"SV\": 0.25}" 98 100 0 1


# DV-only block
# launch_block "{\"DV\": 0.5}" 0 100 20
