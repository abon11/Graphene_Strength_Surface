#!/bin/bash
export MAX_JOBS_IN_FLIGHT=25

# export START_SEED=101

# # First batch
# export DEFECT_TYPE="SV"
# export DEFECT_PERC=2.0
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

# send_email_notification "$START_SEED"

# for i in $(seq $((START_SEED + 1)) 300); do
#     while true; do
#         current_jobs=$(count_jobs)

#         if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
#             # Wait a little, then re-check
#             sleep 20
#             current_jobs_post_wait=$(count_jobs)

#             if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
#                 echo "SUBMITTING SEED $i"
#                 export DEFECT_RANDOM_SEED="$i"
#                 bash ./run_surface.sh &
#                 if (( i % 25 == 0 )); then
#                     send_email_notification "$i"
#                 fi
#                 break  # move to next i
#             fi
#         fi

#         sleep 60  # Wait before checking again
#     done
# done


export STORAGE_PATH="/hpc/home/avb25/Graphene_Strength_Surface/simulation_data/rotation_tests"
export DEFECT_RANDOM_SEED=0
export DEFECTS="{\"DV\": 0.5}"
export THETA=0
export DETAILED_DATA="False"

bash ./run_surface.sh &


# # do 5 seeds up to 90 degrees for a sanity check
# for j in $(seq 0 1 0); do
#     export DEFECT_RANDOM_SEED="$j"
#     send_email_notification "$DEFECT_RANDOM_SEED"
#     for i in $(seq 0 10 90); do
#         while true; do
#             current_jobs=$(count_jobs)

#             if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
#                 # Wait a little, then re-check
#                 sleep 60
#                 current_jobs_post_wait=$(count_jobs)

#                 if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
#                     echo "SUBMITTING SEED $j THETA $i"
#                     export THETA="$i"
#                     bash ./run_surface.sh &
#                     sleep 240
#                     break  # move to next i
#                 fi
#             fi

#             sleep 100  # Wait before checking again
#         done
#     done
# done

export DEFECTS="{\"DV\": 0.25, \"SV\": 0.25}"
for j in $(seq 0 1 0); do
    export DEFECT_RANDOM_SEED="$j"
    send_email_notification "$DEFECT_RANDOM_SEED"
    for i in $(seq 0 10 90); do
        while true; do
            current_jobs=$(count_jobs)

            if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
                # Wait a little, then re-check
                sleep 60
                current_jobs_post_wait=$(count_jobs)

                if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
                    echo "SUBMITTING SEED $j THETA $i"
                    export THETA="$i"
                    bash ./run_surface.sh &
                    sleep 240
                    break  # move to next i
                fi
            fi

            sleep 100  # Wait before checking again
        done
    done
done

# # then for the rest of the seeds keep it at 30 deg and do every 5
# for j in $(seq 6 1 30); do
#     export DEFECT_RANDOM_SEED="$j"
#     send_email_notification "$DEFECT_RANDOM_SEED"
#     for i in $(seq 0 5 30); do
#         while true; do
#             current_jobs=$(count_jobs)

#             if (( current_jobs < MAX_JOBS_IN_FLIGHT )); then
#                 # Wait a little, then re-check
#                 sleep 60
#                 current_jobs_post_wait=$(count_jobs)

#                 if (( current_jobs_post_wait < MAX_JOBS_IN_FLIGHT )); then
#                     echo "SUBMITTING SEED $j THETA $i"
#                     export THETA="$i"
#                     bash ./run_surface.sh &
#                     sleep 240
#                     break  # move to next i
#                 fi
#             fi

#             sleep 100  # Wait before checking again
#         done
#     done
# done