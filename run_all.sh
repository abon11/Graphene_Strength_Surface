#!/bin/bash
export MAX_JOBS_IN_FLIGHT=25

# First batch
export DEFECT_TYPE="SV"
export DEFECT_PERC=0.5
export THETA=0

echo "SUBMITTING SEED 751"
export DEFECT_RANDOM_SEED=751
bash ./run_surface.sh &


count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

send_email_notification() {
    echo "Seed $1 has been submitted" | mail -s "HPC Job Notification" avb25@duke.edu
}

send_email_notification "751"



for i in {752..1000}; do
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