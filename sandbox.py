import pandas as pd

df = pd.read_csv('/data1/avb25/graphene_sim_data/defected_data/all_simulations.csv')

# Force 'Simulation ID' to be string and zero-padded
df["Threads"] = 16

# Save it back
df.to_csv('/data1/avb25/graphene_sim_data/defected_data/all_simulations.csv', index=False)


