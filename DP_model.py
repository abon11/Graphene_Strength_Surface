import pandas as pd
from plot_StrengthSurface import filter_data
import local_config
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    folder = f'{local_config.DATA_DIR}/defected_data'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defect Type": "SV",
        "Defect Percentage": 0.5,
        "Theta": 0
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6),
        # "Defect Random Seed": (1, 100)
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

    plot_together = True

    for seed, group_df in grouped:
        # Create a list of DataPoints for this seed
        datapoints = [DataPoint(row) for _, row in group_df.iterrows()]

        # Create Surface and fit Drucker-Prager
        surface = Surface(datapoints)
        surface.fit_drucker_prager()

        # Optionally store the seed for tracking
        surface.seed = seed
        print(f"Fit surface for seed {int(seed)}.")

        surfaces.append(surface)

    if plot_together:
        plot_all_surfaces(surfaces)

    for surface in surfaces:
        print(f"Seed {int(surface.seed)}: alpha = {surface.alpha:.4f}, k = {surface.k:.4f}")
        if not plot_together:
            plot_surface_fit(surface)


def plot_surface_fit(surface, resolution=1000):
    alpha = surface.alpha
    k = surface.k

    # Set plot range around your data
    sig1_vals = [dp.df["Strength_1"] for dp in surface.points]
    sig2_vals = [dp.df["Strength_2"] for dp in surface.points]
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
    sqrtJ2 = np.sqrt(j2)

    # Evaluate DP function
    F = sqrtJ2 + alpha * i1 - k

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

    plt.title(f"Fitted Drucker-Prager Surface (Seed {int(surface.seed)})")
    plt.legend()

    plt.savefig(f'{local_config.DATA_DIR}/defected_data/plots/DP_fitted_{int(surface.seed)}.png')
    plt.close()


def plot_all_surfaces(surfaces, resolution=1000, showlabels=False):
    # Set global grid range
    all_sig_vals = [dp.df["Strength_1"] for s in surfaces for dp in s.points] + \
                   [dp.df["Strength_2"] for s in surfaces for dp in s.points]
    min_sig, max_sig = min(all_sig_vals), max(all_sig_vals)
    padding = 0.2 * (max_sig - min_sig)
    grid = np.linspace(min_sig - padding, max_sig + padding, resolution)

    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cycle through colors for each seed
    colors = cm.hsv(np.linspace(0, 1, len(surfaces)))

    for surface, color in zip(surfaces, colors):
        alpha = surface.alpha
        k = surface.k

        i1 = sig1 + sig2 + sig3
        mean_stress = i1 / 3
        dev_xx = sig1 - mean_stress
        dev_yy = sig2 - mean_stress
        dev_zz = sig3 - mean_stress
        j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2)
        sqrtJ2 = np.sqrt(j2)
        F = sqrtJ2 + alpha * i1 - k

        # Plot contour
        ax.contour(sig1, sig2, F, levels=[0], colors=[color], linewidths=1.5)
        if showlabels:
            ax.plot([], [], color=color, label=f"Seed {int(surface.seed)}")

    ax.plot([-50, 130], [0, 0], color='black')
    ax.plot([0, 0], [-50, 130], color='black')

    ax.set_xlim(-15, 100)
    ax.set_ylim(-15, 100)
    ax.set_xlabel(r"$\sigma_1$")
    ax.set_ylabel(r"$\sigma_2$")
    ax.set_title("DP Surfaces Overlayed by Random Seed")
    if showlabels:
        ax.legend()

    plt.savefig(f'{local_config.DATA_DIR}/defected_data/plots/DP_overlay_all_seeds.png')
    plt.close()


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
    

class Surface:
    def __init__(self, points):
        """
        This class defines a single strength surface and can fit a Drucker-Prager model to it
        
        df (list[DataPoint]): list of datapoints for the surface
        """

        self.points = points
        self.seed = self.check_seed()
        self.alpha = None
        self.k = None

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
        initial_guess = [0.0, 1.0]
        result = minimize(self.loss, x0=initial_guess)  # pass in loss function and initial guess
        if result.success:
            self.alpha, self.k = result.x
        else:
            raise RuntimeError(f"Minimization failed: {result.message}")


if __name__ == "__main__":
    main()