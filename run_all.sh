#!/bin/bash
export MAX_JOBS_IN_FLIGHT=30

export START_SEED=101

# First batch
export DEFECT_TYPE="SV"
export DEFECT_PERC=2.0
export THETA=0

echo "SUBMITTING SEED $START_SEED"
export DEFECT_RANDOM_SEED="$START_SEED"
bash ./run_surface.sh &


count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

send_email_notification() {
    echo "Seed $1 has been submitted" | mail -s "HPC Job Notification" avb25@duke.edu
}

send_email_notification "$START_SEED"



for i in $(seq $((START_SEED + 1)) 300); do
    while true; do
        current_jobs=$(count_jobs)

        if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
            # Wait a little, then re-check
            sleep 20
            current_jobs_post_wait=$(count_jobs)

            if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
                echo "SUBMITTING SEED $i"
                export DEFECT_RANDOM_SEED="$i"
                bash ./run_surface.sh &
                if (( i % 25 == 0 )); then
                    send_email_notification "$i"
                fi
                break  # move to next i
            fi
        fi

        sleep 60  # Wait before checking again
    done
done