"""
This fits the 2-parameter Drucker-Prager model to a set of strength surface data and stores the alpha, k, and seed in the csv
"""

import pandas as pd
from plot_StrengthSurface import filter_data
import local_config
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def main():
    folder = f'{local_config.DATA_DIR}/defected_data'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defect Type": "SV",
        "Defect Percentage": 0.5,
        "Theta": 0,
        # "Defect Random Seed": 1
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6),
        "Defect Random Seed": (1, 100)
        # "Theta": (0, 90),
    }

    or_filters = {
        # "Defect Type": ["SV", "DV"],
        # "Theta": [0, 30, 60, 90]
    }

    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, dupe_thetas=False)
    
    # Group by defect seed
    grouped = filtered_df.groupby("Defect Random Seed")

    surfaces = []
    alphas = []
    ks = []
    ps = []

    plot_together = True

    if ((len(grouped) >= 10) and (plot_together == False)):
        inp = input(f"Warning! Set to save {len(grouped)} plots. Was this intentional? Type 'n' to quit. ")
        if inp == 'n':
            exit()

    rows = []
    rmse = []
    loss = []

    for seed, group_df in grouped:
        # Create a list of DataPoints for this seed
        datapoints = [DataPoint(row) for _, row in group_df.iterrows()]

        # Create Surface and fit Drucker-Prager
        surface = Surface3P(datapoints)  # changed from just surface
        surface.fit_drucker_prager()

        # Optionally store the seed for tracking
        surface.seed = seed
        print(f"Fit surface for seed {int(seed)}.")

        stats = surface.compute_loss_statistics()
        print(f"Seed {int(surface.seed)}: alpha = {surface.alpha:.4f}, k = {surface.k:.4f}, p = {surface.p:.4f}... RMSE: {stats["rmse"]:.4f}, Max Residual: {stats["max_residual"]:.4f}, Total Loss: {stats["total_loss"]:.4f}.")
        if stats["rmse"] is not np.nan:
            rmse.append(stats["rmse"])
            loss.append(stats["total_loss"])

        surfaces.append(surface)
        alphas.append(surface.alpha)
        ks.append(surface.k)
        ps.append(surface.p)

        rows.append({"Seed": surface.seed, "alpha": surface.alpha, "k": surface.k, "p": surface.p})

        surface.plot_surface_fit()

    if len(rmse) > 0:
        print(f"Final average RMSE over {len(rmse)} samples: {np.sum(rmse) / len(rmse)}")
        print(f"Final average total loss over {len(loss)} samples: {np.sum(loss) / len(loss)}")

    df_params = pd.DataFrame(rows)
    df_params.to_csv("drucker_prager3_params.csv", index=False)


class BaseSurface:    
    def plot_loss_landscape(self, alpha_range=(-0.1, 0.1), k_range=(25, 50), res=100):

        alphas = np.linspace(*alpha_range, res)
        ks = np.linspace(*k_range, res)
        A, K = np.meshgrid(alphas, ks)
        Loss = np.zeros_like(A)

        for i in range(res):
            for j in range(res):
                Loss[i, j] = self.loss([A[i, j], K[i, j]])

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(A, K, Loss, levels=50, cmap='viridis')
        plt.colorbar(cp, label='Loss')
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$k$")
        plt.title(f"Loss Landscape for Seed {self.seed}")

        # Plot optimizer's best guess
        if self.fit_result is not None:
            a, k = self.fit_result.x
            plt.plot(a, k, 'ro', label="Optimizer guess")
            plt.legend()

        plt.savefig(f"Loss_Landscape_{self.seed}.png")


class MadeSurface(BaseSurface):
    def __init__(self, alpha, k, seed=None):
        self.alpha = alpha
        self.k = k
        self.seed = seed


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
    

class Surface(BaseSurface):
    def __init__(self, points):
        """
        This class defines a single strength surface and can fit a Drucker-Prager model to it
        
        points (list[DataPoint]): list of datapoints for the surface
        """

        self.points = points
        self.seed = self.check_seed()
        self.alpha = None
        self.k = None
        self.fit_result = None

    def check_seed(self):
        """
        Check to make sure all of our random seeds match up, returns the seed if it does match
        """

        seed = self.points[0].df["Defect Random Seed"]
        for p in self.points:
            if p.df["Defect Random Seed"] != seed:
                raise ValueError("Random Seeds do not match up in this surface!")
        return seed

    def loss(self, params):
        """
        Loss function to minimize: sum of squared Drucker-Prager residuals.

        params: [alpha, k]
        """
        alpha, k = params
        residuals = []
        for point in self.points:
            i1, j2 = point.calculate_invariants()
            residual = np.sqrt(j2) + alpha * i1 - k
            residuals.append(residual ** 2)
        return sum(residuals)

    def fit_drucker_prager(self):
        """
        Fit Drucker-Prager parameters (alpha, k) to the current surface by minimizing the least squares loss.
        Stores the optimized values in self.alpha and self.k.
        """
        try:
            result = minimize(self.loss, x0=[0.0, 1.0])
            self.fit_result = result 
            if result.success or "precision loss" in result.message:
                self.alpha, self.k = result.x
            else:
                raise RuntimeError(f"Minimization failed: {result.message}")

        except RuntimeError as e:
            print(f"Warning: Seed {int(self.seed)} fit failed. {e}")
            self.alpha, self.k = np.nan, np.nan

    def compute_loss_statistics(self):
        if self.alpha is np.nan or self.k is np.nan:
            return {
                "total_loss": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "max_residual": np.nan
            }
        
        residuals = []
        for point in self.points:
            i1, j2 = point.calculate_invariants()
            r = np.sqrt(j2) + self.alpha * i1 - self.k
            residuals.append(r)

        residuals = np.array(residuals)
        n = len(residuals)
        total_loss = np.sum(residuals**2)
        mse = total_loss / n
        rmse = np.sqrt(mse)
        max_residual = np.max(np.abs(residuals))

        return {
            "total_loss": total_loss,
            "mse": mse,
            "rmse": rmse,
            "max_residual": max_residual
        }


class Surface3P(Surface):
    def __init__(self, points):
        """
        This class defines a single strength surface and can fit a Drucker-Prager model to it
        
        points (list[DataPoint]): list of datapoints for the surface
        """

        self.points = points
        self.seed = self.check_seed()
        self.alpha = None
        self.k = None
        self.p = None
        self.fit_result = None

    def loss(self, params):
        """
        Loss function to minimize: sum of squared Drucker-Prager residuals.

        params: [alpha, k, p]
        """
        alpha, k, p = params
        residuals = []
        for point in self.points:
            i1, j2 = point.calculate_invariants()
            residual = np.sqrt(j2) + alpha * i1 + p * i1 ** 2 - k
            residuals.append(residual ** 2)
        return sum(residuals)

    def fit_drucker_prager(self):
        """
        Fit Drucker-Prager3 parameters (alpha, k, p) to the current surface by minimizing the least squares loss.
        Stores the optimized values in self.alpha, self.k, and self.p.
        """
        try:
            result = minimize(self.loss, x0=[0.0, 1.0, 0.5])
            self.fit_result = result 
            if result.success or "precision loss" in result.message:
                self.alpha, self.k, self.p = result.x
            else:
                raise RuntimeError(f"Minimization failed: {result.message}")

        except RuntimeError as e:
            print(f"Warning: Seed {int(self.seed)} fit failed. {e}")
            self.alpha, self.k, self.p = np.nan, np.nan, np.nan

    def compute_loss_statistics(self):
        if self.alpha is np.nan or self.k is np.nan or self.p is np.nan:
            return {
                "total_loss": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "max_residual": np.nan
            }
        
        residuals = []
        for point in self.points:
            i1, j2 = point.calculate_invariants()
            r = np.sqrt(j2) + self.alpha * i1 + self.p * i1 ** 2 - self.k
            residuals.append(r)

        residuals = np.array(residuals)
        n = len(residuals)
        total_loss = np.sum(residuals**2)
        mse = total_loss / n
        rmse = np.sqrt(mse)
        max_residual = np.max(np.abs(residuals))

        return {
            "total_loss": total_loss,
            "mse": mse,
            "rmse": rmse,
            "max_residual": max_residual
        }
    
    def plot_surface_fit(self, resolution=1000):

        # Set plot range around your data
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
        F = np.sqrt(j2) + self.alpha * i1 + self.p * i1 ** 2 - self.k

        plt.figure(figsize=(8, 8))

        # Plot contour where f = 0 (the strength boundary)
        plt.contour(sig1, sig2, F, levels=[0], colors="red", linewidths=2)
        plt.plot([], [], color="red", label="DP surface")  # for legend (cs.collections is not working)

        # Plot data points
        plt.scatter(sig1_vals, sig2_vals, color="blue", label="MD failure points")
        plt.scatter(sig2_vals, sig1_vals, color="blue")

        plt.plot([-50, 130], [0, 0], color='black')
        plt.plot([0, 0], [-50, 130], color='black')

        plt.xlabel(r"$\sigma_1$")
        plt.ylabel(r"$\sigma_2$")

        plt.xlim(-15, 100)
        plt.ylim(-15, 100)

        plt.title(f"Fitted Drucker-Prager Surface (Seed {int(self.seed)})")
        plt.legend()

        plt.savefig(f'{local_config.DATA_DIR}/defected_data/plots/DP3_fitted_{int(self.seed)}.png')
        plt.close()

if __name__ == "__main__":
    main()