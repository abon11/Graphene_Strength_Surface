#!/bin/bash

# First batch
export DEFECT_TYPE="SV"
export DEFECT_PERC=0.5
export THETA=0

echo "SUBMITTING SEED 101"
export DEFECT_RANDOM_SEED=101
bash ./run_surface.sh &


count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}


for i in {102..200}; do
    while true; do
        current_jobs=$(count_jobs)

        if (( current_jobs < 10 )); then
            # Wait a little, then re-check
            sleep 10
            current_jobs_post_wait=$(count_jobs)

            if (( current_jobs_post_wait < 10 )); then
                echo "SUBMITTING SEED $i"
                export DEFECT_RANDOM_SEED="$i"
                bash ./run_surface.sh &
                break  # move to next i
            fi
        fi

        sleep 60  # Wait before checking again
    done
done