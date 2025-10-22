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
        "Defects": '{"DV": 0.25, "SV": 0.25}',
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

    print("Doing MX:")

    # save_fits_to, plot_title = make_filename(exact_filters, return_title=True)
    save_fits_to = 'DPparams_3D_MX2_NP.csv'
    plot_title = None
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
        surface = Surface(datapoints, interest_value, 2)  # changed from just surface
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


def make_filename(exact_filters, return_title=False):
    defects = parse_defects_json(exact_filters["Defects"]).keys()
    if len(defects) == 1:
        defect = list(defects)[0]
    else:
        defect = 'MX'

    theta = exact_filters["Theta Requested"]
    if theta == 0:
        orientation = 'AC'
    elif theta == 90:
        orientation = 'ZZ'
    else:
        orientation = ''
    
    if return_title:
        title = ''
        for defct, size in parse_defects_json(exact_filters["Defects"]).items():
            title += f'{defct} {size}%, '
        if orientation == 'AC':
            title += 'Armchair'
        elif orientation == 'ZZ':
            title += 'Zigzag'
        else:
            title = title[:-2]

        return f'DPparams_{orientation}_{defect}.csv', title

    return f'DPparams_{orientation}_{defect}.csv'


class MadeSurface():
    def __init__(self, alpha, k, interest_value=None, instance=None):
        self.alpha = alpha
        self.k = k
        self.interest_value = interest_value
        self.instance = instance


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
    def __init__(self, points, interest_value, fourier_order):
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
        self.fourier_order = fourier_order

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

    def dp(self, s1, s2, s3, theta, params, N):
        """
        Vectorized Drucker-Prager residual function F(σ, θ) for all data points.
        Returns an array of residuals F = √J₂ + α(θ) I₁ - k(θ).
        """

        omega = 2 * np.pi * theta / 180

        alpha = params[0]
        k = params[2*N+1]

        for m in range(1, N + 1):
            cos_coeff_a = params[2 * m - 1]
            sin_coeff_a = params[2 * m]
            alpha += cos_coeff_a * np.cos(m * omega) + sin_coeff_a * np.sin(m * omega)
            cos_coeff_k = params[(2*N+1)+(2 * m - 1)]
            sin_coeff_k = params[(2*N+1)+(2 * m)]
            k += cos_coeff_k * np.cos(m * omega) + sin_coeff_k * np.sin(m * omega)

        i1 = s1 + s2 + s3
        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0
        return np.sqrt(j2) + alpha * i1 - k

    def loss(self, params, return_resid=False, N=4):
        """
        Vectorized loss: sum of squared Drucker-Prager residuals normalized by ‖∇F‖.
        """
        s1 = np.asarray(self.full_df["Strength_1"].to_numpy(), dtype=float)
        s2 = np.asarray(self.full_df["Strength_2"].to_numpy(), dtype=float)
        s3 = np.asarray(self.full_df["Strength_3"].to_numpy(), dtype=float)
        theta = np.asarray(self.full_df["Theta"].to_numpy(), dtype=float)

        # Evaluate F for all points
        F = self.dp(s1, s2, s3, theta, params, N=N)

        # alpha = self.fourier_eval(params[:2*N+1], N, theta)

        omega = 2 * np.pi * theta / 180
        alpha = params[0]
        for m in range(1, N + 1):
            alpha += params[2*m - 1] * np.cos(m * omega) + params[2*m] * np.sin(m * omega)
        # Compute invariants
        j2 = (s1**2 + s2**2 + s3**2 - s1*s2 - s2*s3 - s3*s1) / 3.0
        sqrt_j2 = np.sqrt(j2) + 1e-24

        # Gradient norm ||dF/dσ||
        dF_dsig1 = (2*s1 - s2 - s3) / (6*sqrt_j2) + alpha
        dF_dsig2 = (2*s2 - s3 - s1) / (6*sqrt_j2) + alpha
        dF_dsig3 = (2*s3 - s1 - s2) / (6*sqrt_j2) + alpha
        grad_norm = np.sqrt(dF_dsig1**2 + dF_dsig2**2 + dF_dsig3**2)

        # Residuals
        residuals = F / (grad_norm + 1e-18)

        if return_resid:
            return residuals
        else:
            return np.sum(residuals**2)
        
    def fourier_eval(self, params, N, theta_deg):
        """Compute Fourier series f(theta) = a0 + Σ[a_cos cos + a_sin sin]."""
        omega = 2 * np.pi * theta_deg / 180   # periodic over [0, 60]
        f = params[0]
        for m in range(1, N + 1):
            cos_coeff = params[2*m - 1]
            sin_coeff = params[2*m]
            f += cos_coeff * np.cos(m * omega) + sin_coeff * np.sin(m * omega)
        return f

    def fit_drucker_prager(self):
        """
        Fit Drucker-Prager parameters (alpha, k) to the current surface by minimizing the least squares loss.
        Stores the optimized values in self.alpha and self.k.
        """
        try:
            def dp_physical_bounds_constraint(params, N, n_points=100):
                """
                Enforce that:
                alpha(theta) >= -sqrt(3)/6
                k(theta) >= 0
                for all theta in [0, 90].
                Returns a concatenated vector of both inequalities.
                """
                theta = np.linspace(0, 90, n_points)

                # Split params into alpha and k parts
                alpha_params = params[:2*N+1]
                k_params = params[2*N+1:]

                alpha_vals = self.fourier_eval(alpha_params, N, theta)
                k_vals = self.fourier_eval(k_params, N, theta)

                lower_alpha = -np.sqrt(3)/6
                lower_k = 0.0

                # Return both sets stacked into one vector
                return np.concatenate([
                    alpha_vals - lower_alpha,
                    k_vals - lower_k
                ])

            # result = minimize(self.loss, x0=[0, 0, 0, 1, 0, 0], method="BFGS")
            # 4th order fourier for alpha and k
            N = self.fourier_order

            initial_guess = np.zeros(4*N+2)
            initial_guess[0] = 0.2
            initial_guess[2*N+1] = 60  # set initial guess
            dp_constraint = NonlinearConstraint(lambda p: dp_physical_bounds_constraint(p, N=N), 0, np.inf)

            result = minimize(
                lambda p: self.loss(p, N=N),
                x0=initial_guess,
                method='trust-constr',
                constraints=[dp_constraint],
                options={'verbose': 1, 'maxiter': 2000}
            )

            self.fit_result = result 
            if result.success or "precision loss" in result.message:
                self.alpha = result.x[:2*N+1]
                self.k = result.x[2*N+1:]
            else:
                raise RuntimeError(f"Minimization failed: {result.message}")

        except RuntimeError as e:
            print(f"Warning: {self.interest_value} {self.instance} fit failed. {e}")
            self.alpha, self.k = np.nan, np.nan

    def compute_loss_statistics(self, print_outputs=False):
        if self.alpha is np.nan or self.k is np.nan:
            return {
                "total_loss": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "max_residual": np.nan
            }

        residuals = self.loss(np.concatenate((self.alpha, self.k)), return_resid=True, N=self.fourier_order)  # combine two lists

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