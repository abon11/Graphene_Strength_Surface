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
import matplotlib.pyplot as plt
import numpy as np

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
df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations_filtered.csv')  # filter such that Sigma_1 is between 4 and 20 (removing experiments where magnitude of strain was too small)

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
        max_iter=2000,                 # increase if convergence is slow (2000)
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

y_train_scaled = y_scaler.fit_transform(y_train)

y_test_scaled = y_scaler.transform(y_test)

train_mses = []
val_mses   = []

def sklearn_regression_loss_exact(model, X, y):
    """
    Computes the exact loss used internally by sklearn MLPRegressor 
    for squared error regression (matching what appears in the training log).
    
    This matches:
      - 0.5 * mean squared error
      - + (0.5 * alpha) * sum(weights^2) / n_samples

    Important: this must be computed in the *scaled* space used in training.
    """

    # Forward pass like sklearn does internally
    # (same architecture and activations)
    A = X
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        Z = A.dot(W) + b
        if i < len(model.coefs_) - 1:
            A = np.maximum(Z, 0.0)  # ReLU
        else:
            A = Z  # identity for output
    
    y_pred = A

    n_samples = X.shape[0]

    # 1) Compute the *internal* squared loss term
    # sklearn accumulates squared errors and multiplies by 0.5 / n_samples
    se = 0.5 * np.sum((y - y_pred) ** 2) / n_samples

    # 2) Regularization on weights only
    sq_norm = 0.0
    for W in model.coefs_:
        v = W.ravel()
        sq_norm += np.dot(v, v)

    reg = (0.5 * model.alpha) * sq_norm / n_samples

    return se + reg

# model.fit(X_train_scaled, y_train_scaled)  # fit the model using scaled training data
# y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
# y_pred = y_scaler.inverse_transform(y_pred_scaled)  # unscale the predicted y, giving us actual predictions


num_epochs = 18  # or however many you want to visualize
for epoch in range(num_epochs):
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',             # ReLU nonlinearity
        solver='adam',                 # good for most problems
        max_iter=epoch+1,                 # increase if convergence is slow (2000)
        alpha=0.001,
        learning_rate_init=0.001,
        random_state=42,
        verbose=True
    )

    model.fit(X_train_scaled, y_train_scaled)  # does *one more* batch
    y_train_pred = y_scaler.inverse_transform(model.predict(X_train_scaled))
    y_val_pred   = y_scaler.inverse_transform(model.predict(X_test_scaled))
    loss = sklearn_regression_loss_exact(model, X_train_scaled, y_train_scaled)
    print('Calculated loss (iter 1): ', loss)
    print('Loss ratio = ', loss / model.loss_)

    # train_mses.append(mean_squared_error(y_train,  y_train_pred))
    # val_mses.append(mean_squared_error(y_test,    y_val_pred))

# # --- plot ---
# plt.plot(train_mses, label="Train MSE")
# plt.plot(val_mses,   label="Val MSE")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend()
# plt.tight_layout()
# plt.savefig("train_vs_val_mse.png", dpi=200)
# plt.show()


# model.fit(X_train_scaled, y_train_scaled)  # fit the model using scaled training data
# y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
# y_pred = y_scaler.inverse_transform(y_pred_scaled)  # unscale the predicted y, giving us actual predictions

# # joblib.dump(x_scaler, "outputs/x_scaler.pkl")
# joblib.dump(model, f"outputs/NEWMOD.pkl")
# # joblib.dump(y_scaler, f"outputs/y_scaler_{target}.pkl")

# def plot_train_test_loss(model, outpath):
#     """
#     Plots training loss and validation loss vs iteration
#     for an MLPRegressor trained with early_stopping=True.
#     """
#     train_loss = model.loss_curve_
#     val_scores = model.validation_scores_   # R^2 per epoch

#     # Convert validation R^2 to a loss-like quantity
#     val_loss = [1.0 - s for s in val_scores]

#     import matplotlib.pyplot as plt
#     import numpy as np

#     it_train = np.arange(1, len(train_loss) + 1)
#     it_val = np.arange(1, len(val_loss) + 1)

#     plt.figure(figsize=(6, 3.5))
#     plt.plot(it_train, train_loss, label="Training loss")
#     plt.plot(it_val, val_loss, '--', label="Validation loss (1 − R²)")
#     plt.yscale("log")
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.title("Training and validation loss (MLP)")
#     plt.legend()
#     plt.grid(alpha=0.3, linestyle='--')
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()

# plot_train_test_loss(model, outpath=f"outputs/{mod}_{target}_train_val_loss____.pdf")


# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"MSE: {mse:.6f}")
# print(f"R² Score: {r2:.4f}")

# print(model)


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
