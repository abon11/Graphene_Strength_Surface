import pandas as pd

df = pd.read_csv('/data1/avb25/graphene_sim_data/defected_data/all_simulations.csv')

# Force 'Simulation ID' to be string and zero-padded
df['Simulation ID'] = df['Simulation ID'].astype(str).str.zfill(5)

# Save it back
df.to_csv('/data1/avb25/graphene_sim_data/defected_data/all_simulations.csv', index=False)

print("Simulation IDs successfully reformatted to 5-digit strings!")

