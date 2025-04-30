import pandas as pd
import local_config

df = pd.read_csv(f'{local_config.DATA_DIR}/defected_data/all_simulations.csv')

# # Compute Theta
# df["Theta"] = df.apply(
#     lambda row: 0.0 if row["Strain Rate x"] >= row["Strain Rate y"] else 30.0,
#     axis=1
# )

# # Reorder columns to place Theta between 'Fracture Window' and 'Defect Type'
# cols = list(df.columns)
# fw_index = cols.index("Fracture Window")
# df = df[cols[:fw_index+1] + ["Theta"] + cols[fw_index+1:-1] + [cols[-1]]]
df = df.drop(columns=["Theta1"])

# Save it back
df.to_csv(f'{local_config.DATA_DIR}/defected_data/all_simulations.csv', index=False)


