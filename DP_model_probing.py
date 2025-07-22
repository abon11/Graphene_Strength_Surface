import pandas as pd
from plot_StrengthSurface import filter_data
import local_config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from DP_model import MadeSurface


def main():
    plot_together = True

    # read the DP models from the csv
    # df_params = pd.read_csv("drucker_prager_params_thetas.csv")
    df_params = pd.read_csv("drucker_prager_params.csv")

    alphas = df_params["alpha"].values
    ks = df_params["k"].values
    instances = df_params["Seed"].values

    # turn them into Surface objects for convenience (and plot if we want)
    surfaces = []
    for a, k, instance in zip(alphas, ks, instances):
        surface = MadeSurface(a, k, interest_value="Seed", instance=instance)
        if not plot_together:
            plot_surface_fit(surface)

        surfaces.append(surface)

    if plot_together:
        plot_all_surfaces(surfaces)

    # plot_all_surfaces([surfaces[0], surfaces[3]], showlabels=True, title="Anisotropy of Graphene Strength")


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


def plot_all_surfaces(surfaces, resolution=1000, mean=None, showlabels=False, title=None):
    # Set global grid range
    grid = np.linspace(-150, 130, resolution)

    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cycle through colors for each seed
    # colors = cm.Set1(np.linspace(0, 1, len(surfaces)))  # hsv for rainbow
    # colors = ['red', 'blue']
    colors = ['black'] * len(surfaces)
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
        ax.contour(sig1, sig2, F, levels=[0], colors=[color], linewidths=2, alpha=0.05)
        if showlabels:
            # if i == 0:
            #     ax.plot([], [], color=color, label=f"Armchair")
            # else:
            #     ax.plot([], [], color=color, label=f"Zigzag")
            ax.plot([], [], color=color, label=f"{surface.interest_value} {int(surface.instance)}")

    ax.plot([-150, 130], [0, 0], color='black')
    ax.plot([0, 0], [-150, 130], color='black')

    ax.tick_params(axis='both', labelsize=15)

    ax.set_xlim(-15, 100)
    ax.set_ylim(-15, 100)
    # ax.set_xlim(-150, 100)
    # ax.set_ylim(-150, 100)

    ax.set_xlabel(r"$\sigma_1$ (GPa)", fontsize=18)
    ax.set_ylabel(r"$\sigma_2$ (GPa)", fontsize=18)
    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title("DP Surfaces Overlayed by Defect Configuration", fontsize=20)
    if showlabels:
        ax.legend(fontsize=15)

    fig.tight_layout()
    if title:
        plt.savefig(title)
    else:
        plt.savefig(f'{local_config.DATA_DIR}/rotation_tests/plots/DP_overlay_all_seeds.png')
    plt.close()


# maps alphas and ks to tensile and compressive strength
def map_to_strength(alphas, ks):
    cs = (3 * ks) / (np.sqrt(3) - 3 * alphas)
    ts = (3 * ks) / (np.sqrt(3) + 3 * alphas)
    return [cs, ts]

if __name__ == "__main__":
    main()