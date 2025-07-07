import numpy as np
import joblib
from scipy.optimize import differential_evolution


def main():
    # === Load trained model and scalers ===
    mod = 'nn'
    targ = 'sigmas'
    model_info = load_model(mod, targ)  # holds [model, x_scaler, y_scaler]

    # === Target stress ===
    target_sigma_x = 10
    target_sigma_y = 1
    target_sigma_xy = 4
    target = [target_sigma_x, target_sigma_y, target_sigma_xy]

    pred = StrainPrediction(model_info, target)
    pred.print_results()


def load_model(mod, targ):
    model = joblib.load(f"outputs/{mod}_{targ}.pkl")
    x_scaler = joblib.load("outputs/x_scaler.pkl")
    y_scaler = joblib.load(f"outputs/y_scaler_{targ}.pkl")
    return [model, x_scaler, y_scaler]



class StrainPrediction:
    def __init__(self, model_info, target):
        '''
        - model_info (list): list of .pkl files [model, x_scaler, y_scaler]
        - target (list): list of floats of the actual target stress we want
        '''
        self.model = model_info[0]
        self.x_scaler = model_info[1]
        self.y_scaler = model_info[2]
        self.target = target
        self.get_strain()

    def get_strain(self):
        target_scaled = self.y_scaler.transform([self.target])[0]  # must scale target using y_scaler
        # === Objective function ===
        bounds_physical = [(0.0, 0.01), (0.0, 0.01), (0.0, 0.01)]

        def inverse_objective_physical(strain_physical):
            strain_physical = np.array(strain_physical).reshape(1, -1)
            strain_scaled = self.x_scaler.transform(strain_physical)  # scale the x
            pred_scaled = self.model.predict(strain_scaled)[0]  # use the model to predict
            x_error = (pred_scaled[0] - target_scaled[0]) ** 2
            y_error = (pred_scaled[1] - target_scaled[1]) ** 2
            xy_error = (pred_scaled[2] - target_scaled[2]) ** 2
            return x_error + y_error + xy_error

        # === Run optimizer ===
        result = differential_evolution(inverse_objective_physical, bounds=bounds_physical, seed=42, polish=True, disp=True)

        # === Extract result ===
        self.strain_output = result.x  # get the outputted strain rate
        # scale and predict on these strain rates to see how close we got to the requested number
        strain_scaled = self.x_scaler.transform([self.strain_output])
        pred_scaled = self.model.predict(strain_scaled)[0]
        self.pred_check = self.y_scaler.inverse_transform([pred_scaled])[0]
        print(f"\nOptimization Success: {result.success}")


    def print_results(self):
        # === Print ===
        print("\nTarget:")
        print(f"  Sigma_x:  {self.target[0]}")
        print(f"  Sigma_y:  {self.target[1]}")
        print(f"  Sigma_xy: {self.target[2]}")

        print("\nOptimized Strain Rates:")
        print(f"  erate_x:   {self.strain_output[0]:.6e}")
        print(f"  erate_y:   {self.strain_output[1]:.6e}")
        print(f"  erate_xy:  {self.strain_output[2]:.6e}")

        print("\nModel Prediction from Optimized Strains:")
        print(f"  Sigma_x:  {self.pred_check[0]:.4f}")
        print(f"  Sigma_y:  {self.pred_check[1]:.4f}")
        print(f"  Sigma_xy: {self.pred_check[2]:.4f}")


if __name__ == "__main__":
    main()