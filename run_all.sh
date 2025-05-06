#!/bin/bash

# First batch
export DEFECT_TYPE="SV"
export DEFECT_PERC=0.5
export DEFECT_RANDOM_SEED=42
export THETA=15
bash ./run_surface.sh &

sleep 60

export DEFECT_RANDOM_SEED=1
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=2
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=3
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=4
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=5
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=6
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=7
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=8
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=9
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=10
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=11
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=12
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=13
bash ./run_surface.sh &

echo "DONE!!!"