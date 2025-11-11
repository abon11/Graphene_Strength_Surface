"""
This fits the 2-parameter Drucker-Prager model to a set of strength surface data, where alpha and k are 
functions of theta. It fits alpha and k as 4th order fourier series, and saves all of the coefficients in the csv
"""

import pandas as pd
from filter_csv import filter_data, parse_defects_json
import local_config
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def main():
    # ========== USER INTERFACE ==========
    folder = f'{local_config.DATA_DIR}/rotation_tests'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defects": 'None',
        # "Defects": '{"SV": 0.25, "DV": 0.25}',
        # "Defects": None,
        # "Theta Requested": 0,
        # "Defect Random Seed": 54
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6),
        # "Defect Random Seed": (1, 1000)
        # "Theta Requested": (0, 30),
    }

    or_filters = {
        # "Theta Requested": [0, 90]
        # "Defects": ['{"DV": 0.25, "SV": 0.25}', '{"SV": 0.5}', '{"DV": 0.5}']
        "Theta Requested": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    }

    print("Doing None:")

    # save_fits_to, plot_title = make_filename(exact_filters, return_title=True)
    save_fits_to = 'zs_np_none.csv'
    # ====================================
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, flip_strengths=True, duplic_freq=(0, 91, 90))
    interest_value = 'Defect Random Seed'

    # Group by defect seed
    grouped = filtered_df.groupby(interest_value)

    surfaces = []
    zs = []

    rows = []
    rmse = []
    loss = []

    for instance, group_df in grouped:

        # Create Surface and fit Drucker-Prager
        surface = Surface(group_df, interest_value, 2)  # changed from just surface
        surface.fit_drucker_prager()

        print(f"Fit surface for {interest_value} {int(instance)}.")

        stats = surface.compute_loss_statistics(print_outputs=True)
        if stats["rmse"] is not np.nan:
            rmse.append(stats["rmse"])
            loss.append(stats["total_loss"])

        surfaces.append(surface)
        zs.append(surface.zs)

        try:
            rows.append({
                f"{interest_value}": surface.instance,
                **{f"z{i}": surface.zs[i] for i in range(len(surface.zs))},
                "Total Loss": float(stats["total_loss"]), "RMSE": float(stats["rmse"])
            })
        except TypeError:
            print("Not adding this seed to the list ##############")

    if len(rmse) > 0:
        print(f"Final average RMSE over {len(rmse)} samples: {np.sum(rmse) / len(rmse)}")
        print(f"Final average total loss over {len(loss)} samples: {np.sum(loss) / len(loss)}")

    df_params = pd.DataFrame(rows)
    df_params.to_csv(f"{save_fits_to}", index=False)

class Surface():
    def __init__(self, points, interest_value, fourier_order):
        """
        This class defines a single strength surface and can fit a Drucker-Prager model to it
        
        points (pd.df): df of datapoints for the surface
        """

        self.points = points
        self.interest_value = interest_value
        self.instance = self.check_instance()
        self.zs = None
        self.fit_result = None
        self.fourier_order = fourier_order

    def check_instance(self):
        """
        Check to make sure all of our random seeds match up, returns the seed if it does match
        -- expanded this to cover all "interest values", such as thetas, etc
        """
        unique_vals = self.points[self.interest_value].unique()

        if len(unique_vals) == 1:
            return unique_vals[0]
        else:
            raise ValueError(f"{self.interest_value}'s do not match up in this surface!")
        
    @staticmethod
    def get_alpha_k(params, theta, N, return_k=True):
        omega = 2 * np.pi * theta / 180

        z_alpha = params[0]
        z_k = params[2*N+1]

        for m in range(1, N + 1):
            cos_coeff_a = params[2 * m - 1]
            sin_coeff_a = params[2 * m]
            z_alpha += cos_coeff_a * np.cos(m * omega) + sin_coeff_a * np.sin(m * omega)
            if return_k:
                cos_coeff_k = params[(2*N+1)+(2 * m - 1)]
                sin_coeff_k = params[(2*N+1)+(2 * m)]
                z_k += cos_coeff_k * np.cos(m * omega) + sin_coeff_k * np.sin(m * omega)
        
        # once we have the value of z_alpha and z_k, we must transform back to alpha and k:
        def softplus(z):
            return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

        alpha = -np.sqrt(3) / 6 + softplus(z_alpha)
        if return_k:
            k = softplus(z_k)
            return alpha, k
        else:
            return alpha


    def dp(self, s1, s2, s3, theta, params, N):
        """
        Vectorized Drucker-Prager residual function F(σ, θ) for all data points.
        Returns an array of residuals F = √J₂ + α(θ) I₁ - k(θ).
        Now we are defining the params as z, and we must transform them to get alpha and k
        """

        alpha, k = self.get_alpha_k(params, theta, N)

        i1 = s1 + s2 + s3
        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0
        return np.sqrt(j2) + alpha * i1 - k

    def loss(self, params, return_resid=False, N=4):
        """
        Vectorized loss: sum of squared Drucker-Prager residuals normalized by ‖∇F‖.
        """
        s1 = np.asarray(self.points["Strength_1"].to_numpy(), dtype=float)
        s2 = np.asarray(self.points["Strength_2"].to_numpy(), dtype=float)
        s3 = np.asarray(self.points["Strength_3"].to_numpy(), dtype=float)
        theta = np.asarray(self.points["Theta"].to_numpy(), dtype=float)

        # Evaluate F for all points
        F = self.dp(s1, s2, s3, theta, params, N=N)

        alpha = self.get_alpha_k(params, theta, N, return_k=False)
        # Compute invariants
        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0
        sqrt_j2 = np.sqrt(j2) + 1e-24

        # Gradient norm ||dF/dσ||
        dF_dsig1 = (2*s1 - s2 - s3) / (6*sqrt_j2) + alpha
        dF_dsig2 = (2*s2 - s3 - s1) / (6*sqrt_j2) + alpha
        dF_dsig3 = (2*s3 - s1 - s2) / (6*sqrt_j2) + alpha
        grad_norm = np.sqrt(dF_dsig1**2 + dF_dsig2**2 + dF_dsig3**2)

        # Residuals
        residuals = F / (grad_norm + 1e-18)  # may want to scale this to tame the gradients

        if return_resid:
            return residuals
        else:
            return np.sum(residuals**2)
        

    def fit_drucker_prager(self):
        """
        Fit Drucker-Prager parameters (alpha, k) to the current surface by minimizing the least squares loss.
        Stores the optimized values in self.zs
        """
        try:
            # def dp_physical_bounds_constraint(params, N, n_points=100):
            #     """
            #     Enforce that:
            #     alpha(theta) >= -sqrt(3)/6
            #     k(theta) >= 0
            #     for all theta in [0, 90].
            #     Returns a concatenated vector of both inequalities.
            #     """
            #     theta = np.linspace(0, 90, n_points)

            #     # Split params into alpha and k parts
            #     alpha_params = params[:2*N+1]
            #     k_params = params[2*N+1:]

            #     alpha_vals = self.fourier_eval(alpha_params, N, theta)
            #     k_vals = self.fourier_eval(k_params, N, theta)

            #     lower_alpha = -np.sqrt(3)/6
            #     lower_k = 0.0

            #     # Return both sets stacked into one vector
            #     return np.concatenate([
            #         alpha_vals - lower_alpha,
            #         k_vals - lower_k
            #     ])

            # result = minimize(self.loss, x0=[0, 0, 0, 1, 0, 0], method="BFGS")
            # 4th order fourier for alpha and k
            N = self.fourier_order

            # start with a good initial guess with the likely shape we are looking for
            if N == 2:
                initial_guess = [-0.88, -0.33, -0.07, 0.08, 0.06, 39.45, -10.38, -1.19, 4.15, 2.09]
            else:
                initial_guess = np.zeros(4*N+2)
                initial_guess[0] = np.log(np.exp(0.2 + np.sqrt(3)/6) - 1)
                initial_guess[2*N+1] = np.log(np.exp(40) - 1) # set initial guess in z-space
            # dp_constraint = NonlinearConstraint(lambda p: dp_physical_bounds_constraint(p, N=N), 0, np.inf)

            result = minimize(
                lambda p: self.loss(p, N=N),
                x0=initial_guess,
                method='trust-constr',
                options={'verbose': 1, 'maxiter': 2000}
            )

            self.fit_result = result 
            if result.success or "precision loss" in result.message:
                self.zs = result.x
            else:
                raise RuntimeError(f"Minimization failed: {result.message}")

        except RuntimeError as e:
            print(f"Warning: {self.interest_value} {self.instance} fit failed. {e}")
            self.zs = np.nan

    def compute_loss_statistics(self, print_outputs=False):
        if self.zs is np.nan:
            return {
                "total_loss": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "max_residual": np.nan
            }

        residuals = self.loss(self.zs, return_resid=True, N=self.fourier_order)  # combine two lists

        residuals = np.array(residuals)
        n = len(residuals)
        total_loss = np.sum(residuals**2)
        mse = total_loss / n
        rmse = np.sqrt(mse)
        max_residual = np.max(np.abs(residuals))

        if print_outputs:
            print(f"{self.interest_value} {self.instance}:")
            np.set_printoptions(precision=4, floatmode='fixed')
            print(f"zs = {self.zs}")
            print(f"RMSE: {rmse:.4f}, Max Residual: {max_residual:.4f}, Total Loss: {total_loss:.4f}.")

        return {
            "total_loss": total_loss,
            "mse": mse,
            "rmse": rmse,
            "max_residual": max_residual
        }
    

if __name__ == "__main__":
    main()