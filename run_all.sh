#!/bin/bash

# First batch
export DEFECT_PERC=0.5
export DEFECT_RANDOM_SEED=12
export THETA=30
bash ./run_surface.sh &

sleep 60

# Second batch
export DEFECT_RANDOM_SEED=13
bash ./run_surface.sh &

# Third batch...
# Add more batches as needed

export DEFECT_RANDOM_SEED=12
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=11
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=10
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=9
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=8
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=7
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=6
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=5
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=4
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=3
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=2
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=1
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

export DEFECT_RANDOM_SEED=42
export THETA=10
bash ./run_surface.sh &

export THETA=20
bash ./run_surface.sh &

echo "DONE!!!"