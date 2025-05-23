from pysr import PySRRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import local_config
import os
import pickle
import joblib


os.environ["NUM_THREADS"] = "8"

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/all_simulations.csv')

# model = PySRRegressor(
#     model_selection="best",
#     niterations=1000,
#     binary_operators=["+", "*", "-", "/"],
#     unary_operators=["square", "sqrt", "log"],
#     loss="loss(x, y) = (x - y)^2",
#     verbosity=1,
#     procs=0,  # auto-threaded
# )

model = MLPRegressor(
    hidden_layer_sizes=(64, 64),   # two hidden layers with 64 neurons each
    activation='relu',             # ReLU nonlinearity
    solver='adam',                 # good for most problems
    max_iter=2000,                 # increase if convergence is slow
    random_state=42,
    verbose=True
)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
y = df["Sigma_Ratio"].values  # repeat for Sigma_1, Sigma_2 and Theta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

# scale x train, y train, x test for best performance
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

model.fit(X_train_scaled, y_train_scaled)  # fit the model using scaled training data
y_pred_scaled = model.predict(X_test_scaled)  # predict from the scaled x test data, getting scaled y data
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # unscale the predicted y, giving us actual predictions

joblib.dump(model, "outputs/model.pkl")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.4f}")

print(model)
