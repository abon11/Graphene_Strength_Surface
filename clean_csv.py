import pandas as pd

# Load the CSV file
input_file = "simulation_data/deform_data/sim00009/sim00009.csv"
output_file = "sim00009_reduced.csv"

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Filter the DataFrame to keep only rows where Timestep % 1000 == 0
df_filtered = df[df["Timestep"] % 5000 == 0]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
