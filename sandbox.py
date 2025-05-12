# import pandas as pd
# import local_config

# df = pd.read_csv(f'{local_config.DATA_DIR}/defected_data/all_simulations.csv')


# # Convert Simulation ID to integer (in case it's stored as string)
# df["Simulation ID"] = df["Simulation ID"].astype(str).str.zfill(5)
# mask = (df["Simulation ID"] < "01438") & (df["Theta"] == 30)

# # Update Theta values
# df.loc[mask, "Theta"] = 90

# # Save it back
# df.to_csv(f'{local_config.DATA_DIR}/defected_data/all_simulations.csv', index=False)


import numpy as np
theta = np.deg2rad(60)
cos2, sin2, sincos = np.cos(theta)**2, np.sin(theta)**2, np.sin(theta)*np.cos(theta)

erate_1 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
erate_2 = [0, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

for e1, e2 in zip(erate_1, erate_2):
    x  = e1 * cos2 + e2 * sin2
    y  = e2 * cos2 + e1 * sin2
    xy = (e1 - e2) * sincos
    print(f"{x} {y} {xy}")