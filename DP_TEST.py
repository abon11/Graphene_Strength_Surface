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
        "Defects": '{"SV": 0.5}',
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
    }

    print("Doing SV:")

    save_fits_to = 'DPparams_3D_SV2_quad.csv'
    # ====================================
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, flip_strengths=True, duplic_freq=(0, 91, 90))
    interest_value = 'Defect Random Seed'

    # Group by defect seed
    grouped = filtered_df.groupby(interest_value)

    surfaces = []
    alphas = []
    ks = []

    rows = []
    rmse = []
    loss = []

    for instance, group_df in grouped:
        # Create a list of DataPoints for this seed
        datapoints = [DataPoint(row) for _, row in group_df.iterrows()]

        # Create Surface and fit Drucker-Prager
        surface = Surface(datapoints, interest_value)  # changed from just surface
        surface.fit_drucker_prager()

        print(f"Fit surface for {interest_value} {int(instance)}.")

        stats = surface.compute_loss_statistics(print_outputs=True)
        if stats["rmse"] is not np.nan:
            rmse.append(stats["rmse"])
            loss.append(stats["total_loss"])

        surfaces.append(surface)
        alphas.append(surface.alpha)
        ks.append(surface.k)

        try:
            rows.append({
                f"{interest_value}": surface.instance,
                **{f"a{i}": surface.alpha[i] for i in range(len(surface.alpha))},
                **{f"k{i}": surface.k[i] for i in range(len(surface.k))}
            })
        except TypeError:
            print("Not adding this seed to the list ##############")

    if len(rmse) > 0:
        print(f"Final average RMSE over {len(rmse)} samples: {np.sum(rmse) / len(rmse)}")
        print(f"Final average total loss over {len(loss)} samples: {np.sum(loss) / len(loss)}")

    df_params = pd.DataFrame(rows)
    df_params.to_csv(f"{save_fits_to}", index=False)


class DataPoint:
    def __init__(self, df):
        """
        This class defines a single point in the strength surface
        
        df (pandas dataframe): dataframe for one single datapoint in the big dataframe (one row long)
        """
        self.df = df

    def calculate_invariants(self):
        # we only store the principal stresses, so the stress tensor is diagonal by default
        sigma = np.diagflat([self.df["Strength_1"], self.df["Strength_2"], self.df["Strength_3"]])
        i1 = np.trace(sigma)  # tr(sigma)
        dev_stress = sigma - (1 / 3) * i1 * np.identity(3)
        dev2 = dev_stress @ dev_stress
        j2 = 0.5 * np.trace(dev2)
        return i1, j2
    

class Surface():
    def __init__(self, points, interest_value):
        """
        This class defines a single strength surface and can fit a Drucker-Prager model to it
        
        points (list[DataPoint]): list of datapoints for the surface
        """

        self.points = points
        self.interest_value = interest_value
        self.instance = self.check_instance()
        self.alpha = None
        self.k = None
        self.fit_result = None

    def check_instance(self):
        """
        Check to make sure all of our random seeds match up, returns the seed if it does match
        -- expanded this to cover all "interest values", such as thetas, etc
        """
        df_list = []

        val = self.points[0].df[self.interest_value]
        for p in self.points:
            if p.df[self.interest_value] != val:
                raise ValueError(f"{self.interest_value}'s do not match up in this surface!")
            df_list.append(p.df.to_frame().T)

        # Combine into a single DataFrame
        self.full_df = pd.concat(df_list, ignore_index=False)
        return val

    @staticmethod
    def wrapped_distance(theta, center):
        """Compute signed angular distance (deg) with wrap-around handling."""
        return (theta - center + 180) % 360 - 180

    def spliced_valley_quad(self, theta_deg, a, c, centers=(0, 60), halfwidth=30.0):
        """
        Piecewise quadratic valleys centered at given centers with given halfwidth.
        Always returns a float NumPy array (even for scalar input).
        """
        theta_deg = np.atleast_1d(theta_deg).astype(float)  # ensures array
        output = np.zeros(theta_deg.shape, dtype=float)

        for mu in centers:
            d = self.wrapped_distance(theta_deg, mu)
            mask = np.abs(d) <= halfwidth
            output[mask] = a * (d[mask] / halfwidth)**2 + c

        # If input was scalar, return scalar float instead of 1-element array
        return output[0] if output.size == 1 else output

    def dp(self, s1, s2, s3, theta, params):
        s1 = np.asarray(s1, dtype=float)
        s2 = np.asarray(s2, dtype=float)
        s3 = np.asarray(s3, dtype=float)
        theta = np.asarray(theta, dtype=float)

        a_alpha, c_alpha, a_k, c_k = map(float, params)

        alpha = np.asarray(self.spliced_valley_quad(theta, a_alpha, c_alpha), dtype=float)
        k = np.asarray(self.spliced_valley_quad(theta, a_k, c_k), dtype=float)

        i1 = s1 + s2 + s3
        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0

        return np.sqrt(np.maximum(j2, 0.0)) + alpha * i1 - k


    def loss(self, params, return_resid=False):
        """
        Sum of squared residuals of normalized Drucker–Prager equation.
        """
        s1 = np.asarray(self.full_df["Strength_1"].to_numpy(), dtype=float)
        s2 = np.asarray(self.full_df["Strength_2"].to_numpy(), dtype=float)
        s3 = np.asarray(self.full_df["Strength_3"].to_numpy(), dtype=float)
        theta = np.asarray(self.full_df["Theta"].to_numpy(), dtype=float)

        F = self.dp(s1, s2, s3, theta, params)
        a_alpha, c_alpha, a_k, c_k = params
        alpha = self.spliced_valley_quad(theta, a_alpha, c_alpha)

        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0
        sqrt_j2 = np.sqrt(j2) + 1e-24

        dF_dsig1 = (2*s1 - s2 - s3)/(6*sqrt_j2) + alpha
        dF_dsig2 = (2*s2 - s3 - s1)/(6*sqrt_j2) + alpha
        dF_dsig3 = (2*s3 - s1 - s2)/(6*sqrt_j2) + alpha
        grad_norm = np.sqrt(dF_dsig1**2 + dF_dsig2**2 + dF_dsig3**2)

        residuals = F / (grad_norm + 1e-18)
        return residuals if return_resid else np.sum(residuals**2)

    def fit_drucker_prager(self):
        """
        Fit the 4-parameter spliced-quadratic DP model:
        params = [a_alpha, c_alpha, a_k, c_k]
        Enforces α(θ) ≥ −√3/6 and k(θ) ≥ 0 for all θ∈[0,90].
        """
        try:
            theta_grid = np.linspace(0, 90, 181)

            def dp_physical_bounds_constraint(params):
                a_alpha, c_alpha, a_k, c_k = params
                alpha_vals = self.spliced_valley_quad(theta_grid, a_alpha, c_alpha)
                k_vals = self.spliced_valley_quad(theta_grid, a_k, c_k)
                lower_alpha = -np.sqrt(3)/6
                lower_k = 0.0
                return np.concatenate([alpha_vals - lower_alpha, k_vals - lower_k])

            dp_constraint = NonlinearConstraint(dp_physical_bounds_constraint, 0, np.inf)

            # Initial guess
            initial_guess = np.array([1.0, 0.2, 10.0, 60.0])

            result = minimize(
                lambda p: self.loss(p),
                x0=initial_guess,
                method='trust-constr',
                constraints=[dp_constraint],
                options={'verbose': 1, 'maxiter': 2000}
            )

            self.fit_result = result
            if result.success or "precision loss" in str(result.message).lower():
                self.alpha = np.array([result.x[0], result.x[1]])
                self.k = np.array([result.x[2], result.x[3]])
                print(f"✅ Fit succeeded: α = {self.alpha}, k = {self.k}")
            else:
                raise RuntimeError(result.message)

        except RuntimeError as e:
            print(f"⚠️ Fit failed for {self.interest_value} {self.instance}: {e}")
            self.alpha, self.k = np.nan, np.nan

    def compute_loss_statistics(self, print_outputs=False):
        if self.alpha is np.nan or self.k is np.nan:
            return {
                "total_loss": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "max_residual": np.nan
            }

        residuals = self.loss(np.concatenate((self.alpha, self.k)), return_resid=True)  # combine two lists

        residuals = np.array(residuals)
        n = len(residuals)
        total_loss = np.sum(residuals**2)
        mse = total_loss / n
        rmse = np.sqrt(mse)
        max_residual = np.max(np.abs(residuals))

        if print_outputs:
            print(f"{self.interest_value} {self.instance}:")
            np.set_printoptions(precision=4, floatmode='fixed')
            print(f"alpha = {self.alpha}")
            print(f"k = [{self.k}]")
            print(f"RMSE: {rmse:.4f}, Max Residual: {max_residual:.4f}, Total Loss: {total_loss:.4f}.")

        return {
            "total_loss": total_loss,
            "mse": mse,
            "rmse": rmse,
            "max_residual": max_residual
        }
    

if __name__ == "__main__":
    main()