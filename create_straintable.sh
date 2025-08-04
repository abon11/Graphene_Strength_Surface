#!/bin/bash
# calls inverse_strain.py to populate table with all strain tensors for each ratio and theta (so we don't have to call nn each time)

# Output CSV file
output_file="strain_table.csv"
echo "ratio,theta,erate_xx,erate_yy,erate_xy" > "$output_file"

# Loop over ratios and thetas
for ratio in $(seq 0.0 0.1 0.9); do
    for theta in $(seq 5 5 85); do
        echo "Computing for ratio=$ratio, theta=$theta"
        result=$(python3 inverse_strain.py --ratio "$ratio" --theta "$theta")
        echo "$result" >> "$output_file"
    done
done