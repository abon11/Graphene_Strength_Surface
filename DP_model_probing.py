"""
Given a csv with drucker prager parameters (for 2D or 3D), this generates plots and calculates the errors
without having to refit the least squares model. This is a bit hard-coded from testing, but the logic
is there if we care to improve it in the future.
"""

import pandas as pd
import local_config
import numpy as np
import matplotlib.pyplot as plt
from DP_model import MadeSurface
import plotly.graph_objects as go


def main():
    plot_together = False
    DP_3D = True

    # read the DP models from the csv
    # df_params = pd.read_csv("drucker_prager_params_thetas.csv")
    df_params = pd.read_csv("DP_params_3D_SV.csv")

    if DP_3D:
        alphas = df_params[["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]].to_numpy()
        ks = df_params[["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]].to_numpy()
    else: 
        alphas = df_params["alpha"].values
        ks = df_params["k"].values
    
    instances = df_params["Defect Random Seed"].values

    # turn them into Surface objects for convenience (and plot if we want)
    surfaces = []
    for a, k, instance in zip(alphas, ks, instances):
        surface = MadeSurface(a, k, interest_value="Theta Requested", instance=instance)
        if not plot_together:
            plot_surface_fit(surface)

        surfaces.append(surface)

    if plot_together:
        if DP_3D:
            plot_3d_surface(surfaces)
        else:
            plot_all_surfaces(surfaces)
        
    plot_all_surfaces([surfaces[0], surfaces[3]], showlabels=True, title="Anisotropy of Graphene Strength")


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
    colors = ['red', 'blue']
    # colors = ['black'] * len(surfaces)
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


def plot_3d_surface(surfaces, resolution=150):
    sig_grid = np.linspace(-15, 130, resolution)
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

    fig = go.Figure()

    for surf in surfaces:
        omega = 2 * np.pi * theta / 60
        alpha_theta = (
            surf.alpha[0]
            + surf.alpha[1] * np.cos(omega)
            + surf.alpha[2] * np.sin(omega)
            + surf.alpha[3] * np.cos(2 * omega)
            + surf.alpha[4] * np.sin(2 * omega)
        )
        k_theta = (
            surf.k[0]
            + surf.k[1] * np.cos(omega)
            + surf.k[2] * np.sin(omega)
            + surf.k[3] * np.cos(2 * omega)
            + surf.k[4] * np.sin(2 * omega)
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

        # color_vals = np.maximum(surface_x, surface_y)
        color_vals = surf.instance

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=surf.full_df["Strength_1"],
        #         y=self.full_df["Strength_2"],
        #         z=self.full_df["Theta"],
        #         mode="markers",
        #         marker=dict(color="black", size=3),
        #         name="Data Points"
        #     )
        # )

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
                    # showscale=((surf.instance == 0)),
                    colorbar=dict(title="max(σ₁, σ₂)", x=-0.2) if surf.instance == 0 else None
                ),
                name=f"DP Surface {surf.instance}"
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
    # Save plot
    folder = f"{local_config.DATA_DIR}/rotation_tests"
    html_path = f"{folder}/plots/3D_SS_FULL_SV.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive 3D plot saved to {html_path}")


# maps alphas and ks to tensile and compressive strength
def map_to_strength(alphas, ks):
    cs = (3 * ks) / (np.sqrt(3) - 3 * alphas)
    ts = (3 * ks) / (np.sqrt(3) + 3 * alphas)
    return [cs, ts]

if __name__ == "__main__":
    main()