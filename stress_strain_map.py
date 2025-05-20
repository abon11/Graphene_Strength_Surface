from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import local_config
import os


os.environ["NUM_THREADS"] = "8"

df = pd.read_csv(f'{local_config.DATA_DIR}/angle_testing/garbage.csv')
model = PySRRegressor(
    model_selection="best",
    niterations=1000,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["square", "sqrt", "log"],
    loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    procs=0,  # auto-threaded
)

X = df[["Strain Rate x", "Strain Rate y", "Strain Rate xy"]].values
y = df["Sigma_1"].values  # repeat for Sigma_2 and Theta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X, y)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.4f}")

print(model)
