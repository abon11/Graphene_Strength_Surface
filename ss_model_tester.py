"""
This tests the performance of a stress-strain mapping model.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import local_config
import os
import pickle
import matplotlib.pyplot as plt
import joblib
import numpy as np

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']

mpl.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
"""

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 10,      # match normalsize
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8, # slightly smaller like LaTeX
    'text.usetex': True,
})


target = 'sigmas'  # can be 'theta', 'ratio', or 'both'
mod = 'nn'  # can be 'nn' or 'sr'

# if target != 'both' and target != 'theta' and target != 'ratio':
#     print("Target must be 'both', 'theta', or 'ratio'!")
#     exit()


os.environ["NUM_THREADS"] = "8"

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations_filtered.csv')
# df = pd.read_csv("filtered.csv")

# could do try/except here for more robustness in the future
model = joblib.load(f"outputs/{mod}_{target}.pkl")

# with open(f"outputs/symbreg_ratio/checkpoint.pkl", "rb") as f:
#     model = pickle.load(f)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
y = df[["Sigma_x", "Sigma_y", "Sigma_xy"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = joblib.load("outputs/x_scaler.pkl")
y_scaler = joblib.load(f"outputs/y_scaler_{target}.pkl")

X_test_scaled = x_scaler.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data

y_pred = y_scaler.inverse_transform(y_pred_scaled)  # unscale the predicted y, giving us actual predictions

labels = ['sigma_x', 'sigma_y', 'sigma_xy']

fig1, axs1 = plt.subplots(1, 3, figsize=(5.83, 2.3))
fig2, axs2 = plt.subplots(1, 3, figsize=(5.83, 2.3))


# compute global residual range so all subplots use same y-limits
residuals_all = []
for i in range(len(labels)):
    residuals_all.append(y_test[:, i] - y_pred[:, i])
residuals_all = np.hstack([r.reshape(-1, 1) for r in residuals_all])
res_min = np.min(residuals_all)
res_max = np.max(residuals_all)
# add small margin
res_margin = 0.05 * (res_max - res_min) if (res_max - res_min) != 0 else 1.0
ylim = (res_min - res_margin, res_max + res_margin)


for i, label in enumerate(labels):
    axs1[i].grid(alpha=0.3)
    axs2[i].grid(alpha=0.3)

    ultimate_max = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
    y_test_normed = y_test[:, i] / ultimate_max
    y_pred_normed = y_pred[:, i] / ultimate_max

    axs1[i].scatter(y_test_normed, y_pred_normed, alpha=0.5, s=10)
    axs1[i].plot([0.0, 1.0], [0.0, 1.0], '--r', linewidth=1)

    axs1[i].set_xlim(0.0, 1.0)
    axs1[i].set_ylim(0.0, 1.0)

    if i == 0:
        axs1[i].set_ylabel("Predicted (normalized)")
        axs1[i].tick_params(axis='y', which='both', labelleft=True)
    else:
        # hide y tick labels, but keep ticks (visually neat)
        axs1[i].tick_params(axis='y', which='both', labelleft=False)
    if i == 1:
        axs1[i].set_xlabel("True (normalized)")


    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.title(f"Predicted vs. Actual")
    # plt.tight_layout()
    # plt.savefig(f"outputs/{mod}_pva_{label}.pdf")
    # plt.close()

    residuals = y_test[:, i] - y_pred[:, i]
    axs2[i].scatter(y_pred_normed, residuals, alpha=0.5, s=10)
    axs2[i].axhline(0, color='r', linestyle='--')
    axs2[i].set_xlim(0, 1)
    axs2[i].set_ylim(ylim)
    # Only show y-label and ticks on the left-most axis
    if i == 0:
        axs2[i].set_ylabel("Residual")  # (True - Pred)
        axs2[i].tick_params(axis='y', which='both', labelleft=True)
    else:
        axs2[i].tick_params(axis='y', which='both', labelleft=False)

    if i == 1:
        axs2[i].set_xlabel("Predicted (normalized)")


    # plt.xlabel("Predicted")
    # plt.ylabel("Residual")
    # plt.title(f"Residuals vs. Predicted - {label}, {mod}")
    # plt.tight_layout()
    # plt.savefig(f"outputs/{mod}_resid_{label}.pdf")
    # plt.close()

# fig1.suptitle(f"Predicted vs Actual (normalized)", y=1.03)
fig1.tight_layout()
fig1.savefig(f"outputs/{mod}_{target}_pred_vs_actual_norm.pdf", dpi=200, bbox_inches='tight')
plt.close(fig1)
# fig2.suptitle(f"Residuals vs Predicted", y=1.03)
fig2.tight_layout()
fig2.savefig(f"outputs/{mod}_{target}_residuals_sharedy.pdf", dpi=200, bbox_inches='tight')
plt.close(fig2)

# Combined metrics
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(y_test)
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

loss_curve = np.asarray(model.loss_curve_)
plt.figure(figsize=(5.18, 3.5))
plt.plot(np.arange(1, len(loss_curve)+1), loss_curve, markersize=3, linewidth=1, label='Train Loss', c='b')
test_mse = mean_squared_error(y_test, y_pred)
print("TEST ", test_mse)
train_mse = mean_squared_error(y_train, y_scaler.inverse_transform(model.predict(x_scaler.transform(X_train))))
print("TRAIN ", train_mse)

plt.axhline(test_mse, color='r', linestyle='--', linewidth=1.5, label=f"Final Test MSE")
# plt.yscale('log')                # optional but often helpful
plt.xlabel("Iteration")
plt.ylabel("Loss")
# plt.title(f"Training and Final Test Loss")
plt.xticks([0, 3, 6, 9, 12, 15, 18])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.savefig(f"outputs/{mod}_{target}_loss_curve.pdf", dpi=200)
plt.close()
print(f"Saved loss curve to {f'outputs/{mod}_{target}_loss_curve.pdf'}")

# print(np.sqrt(mse[0]) / (np.max(y_test[:, 0])), (np.max(y_test[:, 1])**2), (np.max(y_test[:, 2])**2))
print(f"MSE (Sigma x):  {(mse[0]/(np.max(y_test[:, 0])**2)):.6f}")
print(f"MSE (Sigma y):  {(mse[1]/(np.max(y_test[:, 1])**2)):.6f}")
print(f"MSE (Sigma xy): {(mse[2]/(np.max(y_test[:, 2])**2)):.6f}")
print(f"R^2 (Sigma x):  {r2[0]:.4f}")
print(f"R^2 (Sigma y):  {r2[1]:.4f}")
print(f"R^2 (Sigma xy): {r2[2]:.4f}")

print(model)
