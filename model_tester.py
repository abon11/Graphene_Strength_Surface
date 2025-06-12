from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import local_config
import os
import pickle
import matplotlib.pyplot as plt
import joblib

target = 'both'  # can be 'theta', 'ratio', or 'both'

if target != 'both' and target != 'theta' and target != 'ratio':
    print("Target must be 'both', 'theta', or 'ratio'!")
    exit()


os.environ["NUM_THREADS"] = "8"

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv')

# could do try/except here for more robustness in the future
model = joblib.load(f"outputs/nn_{target}.pkl")

# with open(f"outputs/symbreg_ratio/checkpoint.pkl", "rb") as f:
#     model = pickle.load(f)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
if target == 'theta':
    y = df["Theta"].values
elif target == 'ratio':
    y = df["Sigma_Ratio"].values
elif target == 'both':
    y = df[["Sigma_Ratio", "Theta"]].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = joblib.load("outputs/x_scaler.pkl")
y_scaler = joblib.load(f"outputs/y_scaler_{target}.pkl")

X_test_scaled = x_scaler.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data

if target != 'both':
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # unscale the predicted y, giving us actual predictions
else:
    y_pred = y_scaler.inverse_transform(y_pred_scaled)  # unscale the predicted y, giving us actual predictions


if target != 'both':
    # Single-output plot
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs. Actual - {target}")
    plt.savefig(f"outputs/pva_{target}.png")
    plt.close()

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs. Predicted - {target}")
    plt.tight_layout()
    plt.savefig(f"outputs/resid_{target}.png")
    plt.close()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.4f}")

else:
    # Multi-output: Sigma Ratio (index 0) and Theta (index 1)
    labels = ["both_ratio", "both_theta"]
    for i, label in enumerate(labels):
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
        plt.plot(
            [min(y_test[:, i]), max(y_test[:, i])],
            [min(y_test[:, i]), max(y_test[:, i])],
            '--r'
        )
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs. Actual - {label}")
        plt.tight_layout()
        plt.savefig(f"outputs/pva_{label}.png")
        plt.close()

        residuals = y_test[:, i] - y_pred[:, i]
        plt.scatter(y_pred[:, i], residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residuals vs. Predicted - {label}")
        plt.tight_layout()
        plt.savefig(f"outputs/resid_{label}.png")
        plt.close()

    # Combined metrics
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    print(f"MSE (Sigma Ratio): {mse[0]:.6f}")
    print(f"MSE (Theta):       {mse[1]:.6f}")
    print(f"R²  (Sigma Ratio): {r2[0]:.4f}")
    print(f"R²  (Theta):       {r2[1]:.4f}")


print(model)
