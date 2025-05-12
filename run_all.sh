#!/bin/bash

# First batch
export DEFECT_TYPE="None"
export DEFECT_PERC=0
export THETA=0
export DETAILED_DATA="true"
export THERMO=500
export ACCEPT_DUPES="true"
bash ./run_surface.sh &

sleep 60

export THETA=30
bash ./run_surface.sh &

export THETA=60
bash ./run_surface.sh &

export THETA=90
bash ./run_surface.sh &

# export DETAILED_DATA="false"
# export DEFECT_TYPE="SV"
# export DEFECT_PERC=0.5
# export DEFECT_RANDOM_SEED=14
# export THETA=0
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=15
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=16
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=17
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=18
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=19
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=20
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=21
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=22
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=23
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=24
# bash ./run_surface.sh &


# export DEFECT_RANDOM_SEED=14
# export THETA=15
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=15
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=16
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=17
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=18
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=19
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=20
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=21
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=22
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=23
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=24
# bash ./run_surface.sh &


# export DEFECT_RANDOM_SEED=14
# export THETA=30
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=15
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=16
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=17
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=18
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=19
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=20
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=21
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=22
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=23
# bash ./run_surface.sh &

# export DEFECT_RANDOM_SEED=24
# bash ./run_surface.sh &

echo "DONE!!!"