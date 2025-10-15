"""
This fits the 2-parameter Drucker-Prager model to a set of strength surface data and stores the alpha, k, and seed in the csv
It also is now expanded to fit alpha and k as functions of theta
"""

import pandas as pd
from filter_csv import filter_data, parse_defects_json
import local_config
import numpy as np
from scipy.optimize import minimize, least_squares, NonlinearConstraint
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


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
    DP_3D = True
    # save_fits_to, plot_title = make_filename(exact_filters, return_title=True)
    save_fits_to = 'DPparams_3D_MX.csv'
    plot_title = None
    # ====================================
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, flip_strengths=True, duplic_freq=(0, 91, 90))
    interest_value = 'Defect Random Seed'
    # interest_value = "Theta Requested"

    # Group by defect seed
    grouped = filtered_df.groupby(interest_value)

    surfaces = []
    alphas = []
    ks = []

    individual_plots = False

    if ((len(grouped) >= 10) and (individual_plots == True)):
        inp = input(f"Warning! Set to save {len(grouped)} plots. Was this intentional? Type 'n' to quit. ")
        if inp == 'n':
            exit()

    rows = []
    rmse = []
    loss = []

    # fig = go.Figure()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Fit Strength Surfaces, {plot_title}", fontsize=20)

    for instance, group_df in grouped:
        # Create a list of DataPoints for this seed
        datapoints = [DataPoint(row) for _, row in group_df.iterrows()]

        # Create Surface and fit Drucker-Prager
        surface = Surface(datapoints, interest_value, fit_full3D=DP_3D)  # changed from just surface
        surface.fit_drucker_prager()

        print(f"Fit surface for {interest_value} {int(instance)}.")

        stats = surface.compute_loss_statistics(print_outputs=True)
        if stats["rmse"] is not np.nan:
            rmse.append(stats["rmse"])
            loss.append(stats["total_loss"])

        surfaces.append(surface)
        # surface.plot_3d_fit(fig=fig) ##############
        alphas.append(surface.alpha)
        ks.append(surface.k)

        # if instance == 0:
        #     surface.plot_onto_ax(ax, 'red', 'Armchair')
        # else:
        #     surface.plot_onto_ax(ax, 'blue', 'Zigzag')

        if DP_3D:
            try:
                rows.append({f"{interest_value}": surface.instance, "a0": surface.alpha[0], "a1": surface.alpha[1], 
                            "a2": surface.alpha[2], "a3": surface.alpha[3], "a4": surface.alpha[4], 
                            "a5": surface.alpha[5], "a6": surface.alpha[6], "a7": surface.alpha[7], 
                            "a8": surface.alpha[8], "k0": surface.k[0],
                            "k1": surface.k[1],  "k2": surface.k[2],  "k3": surface.k[3], "k4": surface.k[4],
                            "k5": surface.k[5],  "k6": surface.k[6],  "k7": surface.k[7], "k8": surface.k[8],})
            except TypeError:
                print("Not adding this seed to the list")
            if individual_plots:
                surface.plot_3d_fit()
        else:
            rows.append({f"{interest_value}": surface.instance, "alpha": surface.alpha, "k": surface.k})
            if individual_plots:
                surface.plot_surface_fit()

    if len(rmse) > 0:
        print(f"Final average RMSE over {len(rmse)} samples: {np.sum(rmse) / len(rmse)}")
        print(f"Final average total loss over {len(loss)} samples: {np.sum(loss) / len(loss)}")

    df_params = pd.DataFrame(rows)
    df_params.to_csv(f"{save_fits_to}", index=False)


    # folder = f"{local_config.DATA_DIR}/rotation_tests"
    # fullpath = f"{folder}/plots/{save_fits_to[:-4]}.png"
    # fig.tight_layout()
    # fig.savefig(fullpath)
    # print(f"Figure saved to {fullpath}")
    # html_path = f"{folder}/plots/3D_SS_FULL_test.html"
    # fig.write_html(html_path, include_plotlyjs="cdn")
    # print(f"Interactive 3D plot saved to {html_path}")


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
    def __init__(self, points, interest_value, fit_full3D=False):
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
        self.fit_full3D = fit_full3D

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

    def dp(self, point, params, N=4):
        # N is harmonic order for fourier series
        if self.fit_full3D:
            theta = point.df["Theta"]
            omega = 2 * np.pi * theta / 60

            alpha = params[0]
            k = params[2*N+1]

            for m in range(1, N + 1):
                cos_coeff_a = params[2 * m - 1]
                sin_coeff_a = params[2 * m]
                alpha += cos_coeff_a * np.cos(m * omega) + sin_coeff_a * np.sin(m * omega)
                cos_coeff_k = params[(2*N+1)+(2 * m - 1)]
                sin_coeff_k = params[(2*N+1)+(2 * m)]
                k += cos_coeff_k * np.cos(m * omega) + sin_coeff_k * np.sin(m * omega)

        else:
            alpha = params[0]
            k = params[1]

        i1, j2 = point.calculate_invariants()
        residual = np.sqrt(j2) + alpha * i1 - k
        return residual

    def loss(self, params, return_resid=False, N=4):
        """
        Loss function to minimize: sum of squared Drucker-Prager residuals.

        params: [alpha, k]
        """
        residuals = []
        for point in self.points:
            s1 = point.df["Strength_1"]
            s2 = point.df["Strength_2"]
            s3 = point.df["Strength_3"]

            # do F / gradnorm F
            F = self.dp(point, params, N=N)
            J2 = (s1*s1 + s2*s2 + s3*s3 - s1*s2 - s2*s3 - s3*s1) / 3.0
            q = np.sqrt(J2) + 1e-24  # protect against J2 = 0
            if self.fit_full3D:
                alpha = self.fourier_eval(params[:2*N+1], N, point.df["Theta"])
            else:
                alpha = params[0]
            dF_dsig1 = (2.0*s1 - s2 - s3) / (6.0*q) + alpha
            dF_dsig2 = (2.0*s2 - s3 - s1) / (6.0*q) + alpha
            dF_dsig3 = (2.0*s3 - s1 - s2) / (6.0*q) + alpha
            grad_norm = np.sqrt(dF_dsig1*dF_dsig1 + dF_dsig2*dF_dsig2 + dF_dsig3*dF_dsig3)
            
            residuals.append(F / (grad_norm + 1e-18))

        if return_resid:
            return residuals
        else:
            return sum(np.array(residuals)**2)
        
    def fourier_eval(self, params, N, theta_deg):
        """Compute Fourier series f(theta) = a0 + Σ[a_cos cos + a_sin sin]."""
        omega = 2 * np.pi * theta_deg / 60.0   # periodic over [0, 60]
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
            def residual_vec(p):
                return np.asarray(self.loss(p, return_resid=True))
            
            def dp_physical_bounds_constraint(params, N=4, n_points=100):
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

            if self.fit_full3D:
                # result = minimize(self.loss, x0=[0, 0, 0, 1, 0, 0], method="BFGS")
                # 4th order fourier for alpha and k
                N = 4
                initial_guess = np.zeros(4*N+2)
                initial_guess[0] = 0.2
                initial_guess[2*N+1] = 60  # set initial guess
                dp_constraint = NonlinearConstraint(
                lambda p: dp_physical_bounds_constraint(p, N=N), 0, np.inf)

                result = minimize(
                    lambda p: self.loss(p, N=N),
                    x0=initial_guess,
                    method='trust-constr',
                    constraints=[dp_constraint],
                    options={'verbose': 1, 'maxiter': 2000}
                )

            else:
                # result = minimize(self.loss, x0=[0.2, 60.0], bounds=[(-np.sqrt(3)/6, np.inf), (0.0, np.inf)], method="L-BFGS-B")
                x0 = np.array([0.2, 60.0])
                lb = np.array([-np.sqrt(3)/6, 0.0])
                ub = np.array([ np.inf, np.inf])
                result = least_squares(
                    residual_vec, x0,
                    bounds=(lb, ub),
                    method="trf",
                    loss="soft_l1",  # robust to outliers or noisy points
                    f_scale=1.0,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
            self.fit_result = result 
            if result.success or "precision loss" in result.message:
                if self.fit_full3D:
                    # a0, a1, a2, b0, b1, b2 = result.x
                    # self.alpha = [a0, a1, a2]
                    # self.k = [b0, b1, b2]
                    # a0, a1, a2, a3, a4, b0, b1, b2, b3, b4 = result.x
                    self.alpha = result.x[:2*N+1]
                    self.k = result.x[2*N+1:]
                else:
                    self.alpha, self.k = result.x
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
        if self.fit_full3D:
            residuals = self.loss(np.concatenate((self.alpha, self.k)), return_resid=True)  # combine two lists
        else:
            residuals=self.loss([self.alpha, self.k], return_resid=True)

        residuals = np.array(residuals)
        n = len(residuals)
        total_loss = np.sum(residuals**2)
        mse = total_loss / n
        rmse = np.sqrt(mse)
        max_residual = np.max(np.abs(residuals))

        if print_outputs:
            if self.fit_full3D:
                print(f"{self.interest_value} {self.instance}:")
                np.set_printoptions(precision=4, floatmode='fixed')
                print(f"alpha = {self.alpha}")
                print(f"k = [{self.k}]")
                print(f"RMSE: {rmse:.4f}, Max Residual: {max_residual:.4f}, Total Loss: {total_loss:.4f}.")
            else:
                print(f"{self.interest_value} {int(self.instance)}: alpha = {self.alpha:.4f}, k = {self.k:.4f}... RMSE: {rmse:.4f}, Max Residual: {max_residual:.4f}, Total Loss: {total_loss:.4f}.")

        return {
            "total_loss": total_loss,
            "mse": mse,
            "rmse": rmse,
            "max_residual": max_residual
        }
    
    def get_vals_to_plot(self, resolution):
        sig1_vals = [dp.df["Strength_1"] for dp in self.points]
        sig2_vals = [dp.df["Strength_2"] for dp in self.points]
        min_sig, max_sig = min(sig1_vals + sig2_vals), max(sig1_vals + sig2_vals)
        grid = np.linspace(min_sig * 1.1, max_sig * 1.1, resolution)

        # Create sigma1, sigma2 grid
        sig1, sig2 = np.meshgrid(grid, grid)
        sig3 = np.zeros_like(sig1)  

        # Compute I1 and sqrt(J2)
        i1 = sig1 + sig2 + sig3
        mean_stress = i1 / 3
        dev_xx = sig1 - mean_stress
        dev_yy = sig2 - mean_stress
        dev_zz = sig3 - mean_stress

        j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2)

        # Evaluate DP function
        F = np.sqrt(j2) + self.alpha * i1 - self.k
        return sig1_vals, sig2_vals, sig1, sig2, F

    def plot_onto_ax(self, ax, color, lab, resolution=1000):
        sig1_vals, sig2_vals, sig1, sig2, F = self.get_vals_to_plot(resolution)
        # Plot contour where f = 0 (the strength boundary)
        ax.contour(sig1, sig2, F, levels=[0], colors='k', linewidths=2, alpha=0.2)
        ax.plot([], [], color=color, label=f"DP surface - {lab}")  # for legend

        # Plot data points
        # ax.scatter(sig1_vals, sig2_vals, c=color, label=f"MD failure points - {lab}", alpha=0.8)
        ax.scatter(sig2_vals, sig1_vals, c='blue', alpha=0.2)

        ax.plot([-50, 130], [0, 0], color='black')
        ax.plot([0, 0], [-50, 130], color='black')

        ax.set_xlabel(r"$\sigma_1$ (GPa)", fontsize=18)
        ax.set_ylabel(r"$\sigma_2$ (GPa)", fontsize=18)

        ax.set_xlim(-15, 130)
        ax.set_ylim(-15, 130)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # ax.set_title(f"Fit Strength Surfaces, Pristine", fontsize=20)

        # ax.legend(fontsize=15)
    

    def plot_surface_fit(self, resolution=1000):
        # # Set plot range around your data

        # theta_vals = [dp.df["Theta"] for dp in self.points]
        # thetareq_vals = [dp.df["Theta Requested"] for dp in self.points]
        # # dist = np.abs(np.array(theta_vals) - np.array(thetareq_vals))
        # dist = [
        #     0 if dp.df["Strain Rate x"] == dp.df["Strain Rate y"]
        #     else abs(dp.df["Theta"] - dp.df["Theta Requested"])
        #     for dp in self.points
        # ]

        sig1_vals, sig2_vals, sig1, sig2, F = self.get_vals_to_plot(resolution)

        plt.figure(figsize=(8, 8))

        # Plot contour where f = 0 (the strength boundary)
        plt.contour(sig1, sig2, F, levels=[0], colors="red", linewidths=2)
        plt.plot([], [], color="red", label="DP surface")  # for legend (cs.collections is not working)

        # Plot data points
        # plt.scatter(sig1_vals, sig2_vals, color="blue", label="MD failure points")
        # plt.scatter(sig2_vals, sig1_vals, color="blue")
        # Plot both sets, but assign the first one to a handle
        # sc = plt.scatter(sig1_vals, sig2_vals, c=dist, label="MD failure points", cmap='cool', vmin=0, vmax=13)
        # plt.scatter(sig2_vals, sig1_vals, c=dist, cmap='cool', vmin=0, vmax=13)

        plt.scatter(sig1_vals, sig2_vals, c='blue', label="MD failure points")
        plt.scatter(sig2_vals, sig1_vals, c='blue')

        # Attach colorbar to the first scatter
        # plt.colorbar(sc, label="Theta Error")
        
        plt.plot([-50, 130], [0, 0], color='black')
        plt.plot([0, 0], [-50, 130], color='black')

        plt.xlabel(r"$\sigma_1$ (GPa)", fontsize=18)
        plt.ylabel(r"$\sigma_2$ (GPa)", fontsize=18)

        # plt.xlim(-15, 100)
        # plt.ylim(-15, 100)
        plt.xlim(-15, 130)
        plt.ylim(-15, 130)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        # plt.title(f"Fit Drucker-Prager Surface, theta={self.instance}", fontsize=20)
        plt.title(f"Fit Strength Surface, DV 0.5% Zigzag (Seed 54)", fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()

        plt.savefig(f'{local_config.DATA_DIR}/rotation_tests/plots/DP_fitted_{int(self.instance)}.png')
        plt.close()

    def plot_3d_fit(self, fig=None, resolution=110):
        wasfig = True
        if fig is None:
            wasfig = False
        
        sig1_vals = [dp.df["Strength_1"] for dp in self.points] + [dp.df["Strength_2"] for dp in self.points]
        theta_vals = [dp.df["Theta"] for dp in self.points] + [dp.df["Theta"] for dp in self.points]

        min_sig, max_sig  = min(sig1_vals), max(sig1_vals)
        sig_grid = np.linspace(min_sig, max_sig, resolution)
        theta_vals = np.linspace(0, 30, resolution)

        # Create 3D meshgrid (σ1, σ2, θ)
        sig1, sig2, theta = np.meshgrid(sig_grid, sig_grid, theta_vals, indexing='ij')
        sig3 = np.zeros_like(sig1)

        # Invariants
        i1 = sig1 + sig2 + sig3
        mean_stress = i1 / 3
        dev_xx = sig1 - mean_stress
        dev_yy = sig2 - mean_stress
        dev_zz = sig3 - mean_stress
        j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2)

        # alpha_theta = self.alpha[0] + self.alpha[1]*np.cos(6*theta_rad) + self.alpha[2]*np.sin(6*theta_rad)
        # k_theta = self.k[0] + self.k[1]*np.cos(6*theta_rad) + self.k[2]*np.sin(6*theta_rad)
        omega = 2 * np.pi * theta / 60
        alpha_theta = (
            self.alpha[0]
            + self.alpha[1] * np.cos(omega)
            + self.alpha[2] * np.sin(omega)
            + self.alpha[3] * np.cos(2 * omega)
            + self.alpha[4] * np.sin(2 * omega)
        )
        k_theta = (
            self.k[0]
            + self.k[1] * np.cos(omega)
            + self.k[2] * np.sin(omega)
            + self.k[3] * np.cos(2 * omega)
            + self.k[4] * np.sin(2 * omega)
        )

        # Evaluate Drucker-Prager F = √J₂ + α(θ)·I₁ - k(θ)
        F = np.sqrt(j2) + alpha_theta * i1 - k_theta

        # Threshold for where to plot the surface (surface band)
        f_tol = 0.5  # change if surface too thick or thin

        # Mask for F ≈ 0
        mask = np.abs(F) < f_tol

        # Extract matching surface points
        surface_x = sig1[mask].flatten()
        surface_y = sig2[mask].flatten()
        surface_theta = theta[mask].flatten()

        color_vals = np.maximum(surface_x, surface_y)

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=self.full_df["Strength_1"],
                y=self.full_df["Strength_2"],
                z=self.full_df["Theta"],
                mode="markers",
                marker=dict(color="black", size=3),
                name=f"Data Points - {self.instance}"
            )
        )

        # Add the Drucker–Prager surface (where F ≈ 0)
        fig.add_trace(
            go.Scatter3d(
                x=surface_x,
                y=surface_y,
                z=surface_theta,
                mode="markers",
                marker=dict(
                    size=2,
                    # color="red",
                    color=color_vals,
                    colorscale="ice",
                    opacity=0.2,
                    # colorbar=dict(title="max(σ₁, σ₂)", x=-0.2)
                ),
                name=f"DP Surface {self.instance}"
            )
        )

        fig.update_layout(
            title=f"Strength Surface at Different Angles", 
            coloraxis_colorbar=dict(x=-5),
            scene=dict(xaxis_title="σ₁", yaxis_title="σ₂", zaxis_title="Angle (deg)", 
                       xaxis_title_font=dict(size=35), yaxis_title_font=dict(size=35), zaxis_title_font=dict(size=35), 
                       xaxis=dict(tickfont=dict(size=18)), yaxis=dict(tickfont=dict(size=18)), zaxis=dict(tickfont=dict(size=18))),

            scene_camera=dict(eye=dict(x=2, y=0.5, z=1.5))
        )

        if not wasfig:
            # Save plot
            folder = f"{local_config.DATA_DIR}/rotation_tests"
            html_path = f"{folder}/plots/3D_SS_FULL.html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Interactive 3D plot saved to {html_path}")



if __name__ == "__main__":
    main()