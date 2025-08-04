#!/bin/bash
export MAX_JOBS_IN_FLIGHT=25

# export START_SEED=101

# # First batch
# export THETA=0

# echo "SUBMITTING SEED $START_SEED"
# export DEFECT_RANDOM_SEED="$START_SEED"
# bash ./run_surface.sh &


count_jobs() {
    squeue -u "$USER" --noheader | awk '$3 ~ /^one_sim_/ {count++} END {print count+0}'
}

send_email_notification() {
    echo "Seed $1 has been submitted" | mail -s "HPC Job Notification" avb25@duke.edu
}


export STORAGE_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/rotation_tests"
export DEFECT_RANDOM_SEED=0
# export DEFECTS="{\"DV\": 0.5}"
export THETA=0
export DETAILED_DATA="False"

bash ./run_surface.sh &

export DEFECTS="{\"DV\": 0.25, \"SV\": 0.25}"
for j in $(seq 6 1 100); do
    # export DEFECT_RANDOM_SEED="$j"
    if (( j % 10 == 0 )); then
        send_email_notification "$j"
    fi
    for i in $(seq 0 10 90); do
        while true; do
            current_jobs=$(count_jobs)

            if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
                # Wait a little, then re-check
                sleep 60
                current_jobs_post_wait=$(count_jobs)

                if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
                    echo "SUBMITTING SEED $j THETA $i"
                    # export THETA="$i"
                    THETA="$i" DEFECT_RANDOM_SEED="$j" bash ./run_surface.sh &
                    sleep 100
                    break  # move to next i
                fi
            fi

            sleep 100  # Wait before checking again
        done
    done
done

export DEFECTS="{\"DV\": 0.5}"
for j in $(seq 0 1 100); do
    # export DEFECT_RANDOM_SEED="$j"
    if (( j % 20 == 0 )); then
        send_email_notification "$j"
    fi
    for i in $(seq 0 10 90); do
        while true; do
            current_jobs=$(count_jobs)

            if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
                # Wait a little, then re-check
                sleep 60
                current_jobs_post_wait=$(count_jobs)

                if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
                    echo "SUBMITTING SEED $j THETA $i"
                    # export THETA="$i"
                    THETA="$i" DEFECT_RANDOM_SEED="$j" bash ./run_surface.sh &
                    sleep 100
                    break  # move to next i
                fi
            fi

            sleep 100  # Wait before checking again
        done
    done
done