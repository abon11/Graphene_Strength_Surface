"""
Trains a model to learn the stress-strain mapping for our graphene sheets, from the angle_testing data
"""

from pysr import PySRRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import local_config
import os
import pickle
import joblib

target = 'sigmas'  # can be 'theta', 'ratio', 'sigmas', or 'both'
mod = 'nn'  # can be 'nn', 'gb, or 'sr'

# if target != 'both' and target != 'theta' and target != 'ratio':
#     print("Target must be 'both', 'theta', or 'ratio'!")
#     exit()

if mod != 'nn' and mod != 'sr' and mod != 'gb':
    print("Model must be either 'nn', 'gb', or 'sr'!")
    exit()

os.environ["NUM_THREADS"] = "8"

# df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv')
df = pd.read_csv("filtered.csv")

# {'estimator__activation': 'relu', 'estimator__alpha': 0.001, 'estimator__hidden_layer_sizes': (256, 128, 64), 'estimator__learning_rate_init': 0.001, 'estimator__solver': 'adam'}

if mod == 'sr':
    model = PySRRegressor(
        model_selection="best",
        niterations=1000,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", "sqrt", "log"],
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        procs=0,  # auto-threaded
    )
elif mod == 'nn':
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',             # ReLU nonlinearity
        solver='adam',                 # good for most problems
        max_iter=2000,                 # increase if convergence is slow
        alpha=0.001,
        learning_rate_init=0.001,
        random_state=42,
        verbose=True
    )
elif mod == 'gb':
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
    if target == 'both':
        model = MultiOutputRegressor(model)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
if target == 'theta':
    y = df["Theta"].values
elif target == 'ratio':
    y = df["Sigma_Ratio"].values
elif target == 'both':
    y = df[["Sigma_Ratio", "Theta"]].values 
elif target == 'sigmas':
    y = df[["Sigma_x", "Sigma_y", "Sigma_xy"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

# scale x train, y train, x test for best performance
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

if target == 'both' or target == 'sigmas':
    y_train_scaled = y_scaler.fit_transform(y_train)

    model.fit(X_train_scaled, y_train_scaled)  # fit the model using scaled training data
    y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
    y_pred = y_scaler.inverse_transform(y_pred_scaled)  # unscale the predicted y, giving us actual predictions

else:
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model.fit(X_train_scaled, y_train_scaled)  # fit the model using scaled training data
    y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # unscale the predicted y, giving us actual predictions


joblib.dump(x_scaler, "outputs/x_scaler.pkl")
joblib.dump(model, f"outputs/{mod}_{target}.pkl")
joblib.dump(y_scaler, f"outputs/y_scaler_{target}.pkl")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.4f}")

print(model)


# ===== GRID SEARCH: =====
# import pandas as pd
# import joblib
# import os
# from sklearn.neural_network import MLPRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score

# # Load and prepare data
# df = pd.read_csv("filtered.csv")

# X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
# y = df[["Sigma_x", "Sigma_y", "Sigma_xy"]].values

# # Split and scale
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
# X_train_scaled = x_scaler.fit_transform(X_train)
# X_test_scaled = x_scaler.transform(X_test)
# y_train_scaled = y_scaler.fit_transform(y_train)

# # Define base model and search space
# mlp = MLPRegressor(max_iter=2000, random_state=42)
# param_grid = {
#     'estimator__hidden_layer_sizes': [(64, 64), (128, 64), (128, 128), (256, 128, 64)],
#     'estimator__activation': ['relu', 'tanh'],
#     'estimator__alpha': [0.0001, 0.001, 0.01],
#     'estimator__learning_rate_init': [0.001, 0.01],
#     'estimator__solver': ['adam']
# }
# # {'estimator__activation': 'relu', 'estimator__alpha': 0.001, 'estimator__hidden_layer_sizes': (256, 128, 64), 'estimator__learning_rate_init': 0.001, 'estimator__solver': 'adam'}

# # Multi-output wrapper
# wrapped = MultiOutputRegressor(mlp)

# # Grid search
# grid = GridSearchCV(wrapped, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
# grid.fit(X_train_scaled, y_train_scaled)

# # Evaluate best model
# best_model = grid.best_estimator_
# y_pred_scaled = best_model.predict(X_test_scaled)
# y_pred = y_scaler.inverse_transform(y_pred_scaled)

# # Score
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Print and save
# print("Best Parameters:")
# print(grid.best_params_)
# print(f"\nMSE: {mse:.6f}")
# print(f"R² Score: {r2:.4f}")

# os.makedirs("outputs", exist_ok=True)
# joblib.dump(x_scaler, "outputs/x_scaler.pkl")
# joblib.dump(y_scaler, "outputs/y_scaler_sigmas.pkl")
# joblib.dump(best_model, "outputs/nn_sigmas.pkl")
