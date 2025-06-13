import numpy as np
import joblib
from scipy.optimize import minimize

# === Load trained models ===
model_theta = joblib.load("outputs/nn_theta.pkl")
model_ratio = joblib.load("outputs/nn_ratio.pkl")

# === Load scalers ===
x_scaler = joblib.load("outputs/x_scaler.pkl")
y_scaler_theta = joblib.load("outputs/y_scaler_theta.pkl")
y_scaler_ratio = joblib.load("outputs/y_scaler_ratio.pkl")

# === Define target outputs ===
target_theta = 0       # degrees
target_ratio = 0       # sigma_2 / sigma_1

# === Scale target outputs ===
target_theta_scaled = y_scaler_theta.transform([[target_theta]])[0][0]
target_ratio_scaled = y_scaler_ratio.transform([[target_ratio]])[0][0]

# === Objective function ===
def inverse_objective(strain_scaled):
    strain_scaled = np.array(strain_scaled).reshape(1, -1)
    
    pred_theta_scaled = model_theta.predict(strain_scaled)[0]
    pred_ratio_scaled = model_ratio.predict(strain_scaled)[0]
    
    theta_error = (pred_theta_scaled - target_theta_scaled) ** 2
    ratio_error = (pred_ratio_scaled - target_ratio_scaled) ** 2
    
    return theta_error + ratio_error

# === Initial guess: midpoint of physical space ===
guess_physical = np.array([0.0005, 0.0005, 0.0005]).reshape(1, -1)
guess_scaled = x_scaler.transform(guess_physical)[0]

# === Bounds in scaled space ===
lower = np.array([0.0001, 0.0001, 0.0001]).reshape(1, -1)
upper = np.array([0.0010, 0.0010, 0.0010]).reshape(1, -1)
scaled_bounds_array = x_scaler.transform(np.vstack([lower, upper]))
bounds_scaled = [(scaled_bounds_array[0, i], scaled_bounds_array[1, i]) for i in range(3)]


# === Solve optimization problem ===
result = minimize(
    inverse_objective,
    guess_scaled,
    bounds=bounds_scaled,
    method='L-BFGS-B'
)

# === Recover optimized strain rates in physical space ===
strain_opt_scaled = result.x.reshape(1, -1)
strain_opt_physical = x_scaler.inverse_transform(strain_opt_scaled)[0]

# === Forward check: verify predicted output ===
theta_pred_scaled = model_theta.predict(strain_opt_scaled)[0]
ratio_pred_scaled = model_ratio.predict(strain_opt_scaled)[0]

theta_pred = y_scaler_theta.inverse_transform([[theta_pred_scaled]])[0][0]
ratio_pred = y_scaler_ratio.inverse_transform([[ratio_pred_scaled]])[0][0]

# === Print results ===
print("\nTarget:")
print(f"  Theta:        {target_theta} degrees")
print(f"  Sigma Ratio:  {target_ratio}")

print("\nOptimized Strain Rates:")
print(f"  erate_x:   {strain_opt_physical[0]:.6e}")
print(f"  erate_y:   {strain_opt_physical[1]:.6e}")
print(f"  erate_xy:  {strain_opt_physical[2]:.6e}")

print("\nModel Prediction from Optimized Strains:")
print(f"  Theta:        {theta_pred:.2f} degrees")
print(f"  Sigma Ratio:  {ratio_pred:.4f}")

print(f"\nOptimization Success: {result.success}")
