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
        # "Defect Random Seed": 65
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

    plot_together = True

    if ((len(grouped) >= 10) and (plot_together == False)):
        inp = input(f"Warning! Set to save {len(grouped)} plots. Was this intentional? Type 'n' to quit. ")
        if inp == 'n':
            exit()

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
        alphas.append(surface.alpha)
        ks.append(surface.k)

    # if plot_together:
    #     plot_all_surfaces(surfaces)

    rmse = []
    loss = []

    for surface in surfaces:
        stats = surface.compute_loss_statistics()
        print(f"Seed {int(surface.seed)}: alpha = {surface.alpha:.4f}, k = {surface.k:.4f}... RMSE: {stats["rmse"]:.4f}, Max Residual: {stats["max_residual"]:.4f}, Total Loss: {stats["total_loss"]:.4f}.")
        if stats["rmse"] is not np.nan:
            rmse.append(stats["rmse"])
            loss.append(stats["total_loss"])
        if not plot_together:
            plot_surface_fit(surface)

        # surface.plot_loss_landscape()
    
    if len(rmse) > 0:
        print(f"Final average RMSE over {len(rmse)} samples: {np.sum(rmse) / len(rmse)}")
        print(f"Final average total loss over {len(loss)} samples: {np.sum(loss) / len(loss)}")

    pdf, mean, cov = mv_normal_approx(alphas, ks)
    plot_alpha_k(surfaces, pdf=pdf, ci=[mean, cov])
    ci_stress_surfaces = sample_from_ellipse(mean, cov)
    plot_all_surfaces(ci_stress_surfaces, mean=mean, title='90% CI Sampling')



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


def plot_alpha_k(surfaces, pdf=None, ci=None, resolution=100):
    alphas = []
    ks = []
    rmses = []
    for surf in surfaces:
        alphas.append(surf.alpha)
        ks.append(surf.k)
        rmses.append(surf.compute_loss_statistics()["rmse"])

    corr, pval = pearsonr(alphas, ks)
    print(f"Correlation between alpha and k: r = {corr}, p = {pval}")

    fig, ax = plt.subplots()

    if pdf is not None:
        alpha_min, alpha_max = np.min(alphas), np.max(alphas)
        k_min, k_max = np.min(ks), np.max(ks)

        # Pad a little to give breathing room
        alpha_pad = 0.1 * (alpha_max - alpha_min)
        k_pad = 0.1 * (k_max - k_min)

        alpha_range = np.linspace(alpha_min - alpha_pad, alpha_max + alpha_pad, resolution)
        k_range = np.linspace(k_min - k_pad, k_max + k_pad, resolution)

        A, K = np.meshgrid(alpha_range, k_range)
        pos = np.dstack((A, K))

        # Evaluate PDF at each grid point
        Z = pdf.pdf(pos)

        # Plot
        
        contour = ax.contourf(A, K, Z, levels=50, cmap='viridis')
        fig.colorbar(contour, label="PDF")
        ax.scatter(alphas, ks, color='white', s=10, label='Samples')

        if ci:
            vals, vecs = np.linalg.eigh(ci[1])  # eigenvalues and eigenvectors
            vals = vals[::-1]                 # sort descending
            vecs = vecs[:, ::-1]
            chi2_val = chi2.ppf(0.90, df=2)
            width, height = 2 * np.sqrt(vals * chi2_val)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            ellipse = Ellipse(xy=ci[0], width=width, height=height, angle=angle, edgecolor='red', facecolor='none', label="90% CI")
            ax.add_patch(ellipse)

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel("k")
        ax.set_title(r'Multivariate Gaussian PDF in ($\alpha$, k) Space')
        ax.legend()
        fig.savefig("alpha_vs_k_MAP.png")
    else:
        ax.scatter(alphas, ks, c=rmses, cmap="viridis")
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel("k")
        ax.set_title("Drucker-Prager Fit Parameters")
        fig.colorbar(label="RMSE")
        fig.savefig("alpha_vs_k.png")


def plot_all_surfaces(surfaces, resolution=1000, mean=None, showlabels=False, title=None):
    # Set global grid range
    grid = np.linspace(-10, 100, resolution)

    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cycle through colors for each seed
    # colors = cm.binary(np.linspace(0, 1, len(surfaces)))  # hsv for rainbow
    colors = ['gray'] * len(surfaces)
    if mean is not None:
        surfaces = list(surfaces)
        surfaces.append(MadeSurface(mean[0], mean[1]))  # plot the mean if we want
        colors.append('red')

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
    if title:
        ax.set_title(title)
    else:
        ax.set_title("DP Surfaces Overlayed by Random Seed")
    if showlabels:
        ax.legend()
    if title:
        plt.savefig(title)
    else:
        plt.savefig(f'{local_config.DATA_DIR}/defected_data/plots/DP_overlay_all_seeds.png')
    plt.close()


# takes the multivariate normal approximation to generate PDF in terms of alpha and k
def mv_normal_approx(alphas, ks):
    X = np.column_stack([alphas, ks])
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    pdf = multivariate_normal(mean=mean, cov=cov)
    return pdf, mean, cov


def sample_from_ellipse(mean, cov, n_samples=100, confidence=0.90):
    mvn = multivariate_normal(mean=mean, cov=cov)
    samples = []
    threshold = chi2.ppf(confidence, df=2)

    while len(samples) < n_samples:
        sample = mvn.rvs()
        # Mahalanobis distance squared
        d2 = (sample - mean).T @ np.linalg.inv(cov) @ (sample - mean)
        # if abs(d2 - threshold) <= 0.01:
        if d2 <= threshold:
            samples.append(MadeSurface(sample[0], sample[1]))

    return np.array(samples)


class MadeSurface:
    def __init__(self, alpha, k):
        self.alpha = alpha
        self.k = k


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


if __name__ == "__main__":
    main()