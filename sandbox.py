import pandas as pd
import local_config
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv')

# df["Sigma_Ratio"] = np.abs(df["Sigma_2"] / df["Sigma_1"])

# df['Theta'] = df['Theta'].apply(lambda x: min(x, 180 - x))

# # Save it back
# df.to_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv', index=False)

plt.scatter(df["Sigma_Ratio"], df["Theta"], s=0.5)
plt.xlabel("Sigma Ratio")
plt.ylabel("Theta")
plt.save("ratio_theta_scatter")

plt.hist(df["Strain Rate x"], bins=60, color='skyblue', alpha=0.5, label='x')
plt.hist(df["Strain Rate y"], bins=60, color='red', alpha=0.5, label='y')
plt.hist(df["Strain Rate xy"], bins=60, color='grey', alpha=0.5, label='xy')

plt.xlabel("Strain Rate")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Applied Strain Rates")
plt.savefig("erates.png")
plt.close()


plt.hist(df["Theta"], bins=60, color='skyblue', alpha=0.5, label='theta')

plt.xlabel("Theta")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Theta")
plt.savefig("thetas.png")
plt.close()



plt.hist(df["Sigma_Ratio"], bins=60, color='skyblue', alpha=0.5, label='ratio')

plt.xlabel("Ratio")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Sigma Ratio")
plt.savefig("ratios.png")



# import numpy as np
# theta = np.deg2rad(60)
# cos2, sin2, sincos = np.cos(theta)**2, np.sin(theta)**2, np.sin(theta)*np.cos(theta)

# erate_1 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# erate_2 = [0, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

# for e1, e2 in zip(erate_1, erate_2):
#     x  = e1 * cos2 + e2 * sin2
#     y  = e2 * cos2 + e1 * sin2
#     xy = (e1 - e2) * sincos
#     print(f"{x} {y} {xy}")

# def f(x0, x1, x2):
#     return (((((x1 * 1.6953) - (np.sqrt(x1) * 0.0072833)) + ((x0 - x1) * 31.825) ** 2) + x2) - (x0 * ((np.sqrt(np.sqrt(x2)) * 2.5234) + -2.0038))) * 4222

# print(f(0.000523804, 7.21375e-05, 3.98339e-05))  # 5.981216118903219
# print(f(0.000699599, 0.000666074, 0.000168195))  # 9.4895822170042
