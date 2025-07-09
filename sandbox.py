import pandas as pd
import local_config
import matplotlib.pyplot as plt
import numpy as np

for i in range(1, 102):
    id = str(i).zfill(5)
    df = pd.read_csv(f'{local_config.DATA_DIR}/rotation_tests/sim{id}/sim{id}.csv')
    df['Theta'] = df['Theta'].apply(lambda x: min(x, 180 - x))  # keep thetas in bounds

    plt.plot(df["Timestep"], df["Theta"], color='black', alpha=0.2)
    plt.title("Theta vs time - Fracture simulation")
    plt.xlabel("Timesteps")
    plt.ylabel("Theta")

plt.plot([20000, 20000], [0, 90], '--', color='red', label="Dataset Stopping Point")
plt.legend()
plt.savefig("TEST.png")