import numpy as np
import joblib
from scipy.optimize import differential_evolution
from scipy.optimize import brentq
import argparse


def main():
    # === Load trained model and scalers ===
    mod = 'nn'
    targ = 'sigmas'
    model_info = load_model(mod, targ)  # holds [model, x_scaler, y_scaler]

    args = parse_args()  # get input

    target_ratio = args.ratio
    target_theta = args.theta
    target = [target_ratio, target_theta]

    # Optimization loop such that max strain rate is approx 0.001:
    pred = find_optimal_strain(model_info, target)

    # pred.print_results()
    pred.output_results(csv=True)  # prints for computer to read easier (showinputs for csv storage)


def load_model(mod, targ):
    model = joblib.load(f"outputs/{mod}_{targ}.pkl")
    x_scaler = joblib.load("outputs/x_scaler.pkl")
    y_scaler = joblib.load(f"outputs/y_scaler_{targ}.pkl")
    return [model, x_scaler, y_scaler]


def find_optimal_strain(model_info, target, target_max_strain=0.001):
    def strain_error(sigma1_guess):
        prediction = StrainPrediction(model_info, target, sigma1=sigma1_guess)
        strain = prediction.strain_output
        return np.max(np.abs(strain)) - target_max_strain

    sigma1_opt = brentq(strain_error, 0.01, 100.0, xtol=1e-4)

    # Final run to extract strain tensor
    final_pred = StrainPrediction(model_info, target, sigma1=sigma1_opt)
    return final_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True, default=0)
    parser.add_argument("--theta", type=float, default=45)
    return parser.parse_args()


class StrainPrediction:
    def __init__(self, model_info, targets, sigma1=10):
        '''
        - model_info (list): list of .pkl files [model, x_scaler, y_scaler]
        - target (list): list of floats of the target ratio then target theta
        '''
        self.model = model_info[0]
        self.x_scaler = model_info[1]
        self.y_scaler = model_info[2]
        self.ratio = targets[0]
        self.theta = targets[1]
        self.target = self.stress_from_ratio_theta(sigma1=sigma1)
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
        result = differential_evolution(inverse_objective_physical, bounds=bounds_physical, seed=42, polish=True)

        # === Extract result ===
        self.strain_output = result.x  # get the outputted strain rate
        # scale and predict on these strain rates to see how close we got to the requested number
        strain_scaled = self.x_scaler.transform([self.strain_output])
        pred_scaled = self.model.predict(strain_scaled)[0]
        self.pred_check = self.y_scaler.inverse_transform([pred_scaled])[0]
        # print(f"Optimization Success: {result.success}")

    def stress_from_ratio_theta(self, sigma1=8.0):
        theta_rad = np.deg2rad(self.theta)
        sigma2 = self.ratio * sigma1

        # Principal stress matrix
        sigma_p = np.array([[sigma1, 0],
                            [0, sigma2]])

        # Rotation matrix
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        Q = np.array([[c, s],
                    [-s, c]])

        # Rotate principal stresses back to x-y
        sigma = Q @ sigma_p @ Q.T

        sigma_x = sigma[0, 0]
        sigma_y = sigma[1, 1]
        sigma_xy = sigma[0, 1]

        # we return the abs of sigma_xy because we rotate everything into that first quadrant so its all positive.
        return [sigma_x, sigma_y, abs(sigma_xy)]

    # prints results in user-friendly, readable way
    def print_results(self):
        # === Print ===
        print("\nTarget:")
        print(f"  Ratio: {self.ratio:.4f}\n  Theta: {self.theta:.4f}")

        print(f"  Sigma_x:  {self.target[0]:.4f}")
        print(f"  Sigma_y:  {self.target[1]:.4f}")
        print(f"  Sigma_xy: {self.target[2]:.4f}")

        print("\nOptimized Strain Rates:")
        print(f"  erate_x:   {self.strain_output[0]:.2e}")
        print(f"  erate_y:   {self.strain_output[1]:.2e}")
        print(f"  erate_xy:  {self.strain_output[2]:.2e}")

        print("\nModel Prediction from Optimized Strains:")
        print(f"  Sigma_x:  {self.pred_check[0]:.4f}")
        print(f"  Sigma_y:  {self.pred_check[1]:.4f}")
        print(f"  Sigma_xy: {self.pred_check[2]:.4f}")

        # get principal stresses and principal directions
        vals, vecs = np.linalg.eigh(np.array([[self.pred_check[0], self.pred_check[2]],
                                               [self.pred_check[2], self.pred_check[1]]]))
        # eigh returns from lowest eigval to highest eigval, so the last one is sigma_1
        ratio = vals[-2] / vals[-1]

        eigvecs = vecs[:, ::-1]  # reverse the order so the first one is dominant
        theta_deg = np.degrees(np.arctan2(eigvecs[:, 0][1], eigvecs[:, 0][0])) % 180
        theta = min(theta_deg, 180 - theta_deg)  # normalize it to [0, 90]

        print(f"  Ratio: {ratio:.4f}")
        print(f"  Theta: {theta:.4f}")

    # prints results to make integration easy with bash scripts etc.
    def output_results(self, csv=False):
        if csv:
            print(f"{self.ratio:.1f},{self.theta:.0f},{self.strain_output[0]:.4e},{self.strain_output[1]:.4e},{self.strain_output[2]:.4e}")
        
        else:
            print(f"{self.strain_output[0]:.4e} {self.strain_output[1]:.4e} {self.strain_output[2]:.4e}")


if __name__ == "__main__":
    main()