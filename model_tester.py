from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import local_config
import os
import pickle
import matplotlib.pyplot as plt
import joblib

target_theta = False


os.environ["NUM_THREADS"] = "8"

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv')

# could do try/except here for more robustness in the future
if target_theta:
    model = joblib.load("outputs/nn_theta.pkl")
else:
    model = joblib.load("outputs/nn_ratio.pkl")

# with open(f"outputs/symbreg_ratio/checkpoint.pkl", "rb") as f:
#     model = pickle.load(f)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
if target_theta:
    y = df["Theta"].values
else:
    y = df["Sigma_Ratio"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = joblib.load("outputs/x_scaler.pkl")

# import the scales used when training
if target_theta:
    y_scaler = joblib.load("outputs/y_scaler_theta.pkl")
else:
    y_scaler = joblib.load("outputs/y_scaler_ratio.pkl")

X_test_scaled = x_scaler.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # unscale the predicted y, giving us actual predictions

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  # ideal line
plt.xlabel("True")
plt.ylabel("Predicted")
if target_theta:
    plt.title("Predicted vs. Actual - Theta")
    plt.savefig("outputs/pva_theta.png")
else:
    plt.title("Predicted vs. Actual - Sigma Ratio")
    plt.savefig("outputs/pva_ratio.png")
plt.close()

residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()

if target_theta:
    plt.title("Residuals vs. Predicted - Theta")
    plt.savefig("outputs/resid_theta.png")
else:
    plt.title("Residuals vs. Predicted - Sigma Ratio")
    plt.savefig("outputs/resid_ratio.png")


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.4f}")

print(model)
