import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import random


def main():
    defs = "sv"
    fo = 4
    df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")
    full_workflow(df, show_pca=True)
    # full_workflow(df, n_components=6, show_ci=True, save_ci_csv='stats_dv.csv', periodic=False)
    # full_workflow(df, pca_dims=6, ss_theta=0, periodic=False)
    # overlay_cis()



def full_workflow(df, pca_dims=2, gaussian_modes=1, show_pca=False, show_functions=False, 
                  periodic=False, show_ci=False, show_latent=False, show_loss=False,
                  n_samples=100000, save_ci_csv=None, ss_theta=None):
    """
    Given the original dataset, this applies the full workflow
    OPTIONS:
        - show_pca (bool): Shows the PCA reduction of the zs... this is good to determine how many components to keep
        - n_components (int): Number of components to keep, set this after analyzing the PCA
        - show_functions (bool): Shows the true alpha and k functions as well as the sampled ones
        - show_ci (bool): Shows the nice confidence interval of the alpha-k functions with the true overlayed
        - save_ci_csv (str): If not None, will save the values of the mean, lower and upper 95%'s to a csv
        - show_latent (bool): Shows the distributions of the latent space and the samples overlaying
        - show_loss (bool): This simply plots a scatter of the samples and the loss accrued fitting them
        - ss_theta (int): If not None, this generates a strength surface at the specified angle from the sampled alpha-k functions
    """

    # to get a gague of how good the minimization was for this dataset (and maybe remove some outliers?)
    if show_loss:
        plot_loss_data(df, "Total Loss")

    if "Defect Type" in df.columns:
        zs = df.drop(columns=["Defect Type", "Defect Random Seed", "Total Loss", "RMSE", "norm_z"]).to_numpy()
    else:
        zs = df.drop(columns=["Defect Random Seed", "Total Loss", "RMSE", "norm_z"]).to_numpy()
    
    if show_pca:
        display_pca(zs)
        return
    
    # apply pca, fit gaussian, sample, apply inverse pca
    samples_z = fit_latent_density(zs, pca_dims, gaussian_modes, n_samples, show_latent_space=show_latent)

    if show_functions:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_alpha_k(zs, axs[:, 0], label='true', periodic=periodic)
        plot_alpha_k(samples_z, axs[:, 1], n_samples=1000, label='generated', periodic=periodic)
        plt.show()

    if show_ci:
        plot_ci(zs, samples_z, periodic, title="Double Vacancies", save_csv=save_ci_csv)
    
    if ss_theta is not None:
        samples = samples_z.to_numpy() if isinstance(samples_z, pd.DataFrame) else np.asarray(samples_z)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        for i in range(min(len(samples), 1000)):
            a, k = get_alpha_k(samples[i], ss_theta, periodic=False)
            plot_strength_surface(ax, a, k)
        ax.set_title(f"MX Artificial Strength Surfaces: θ={ss_theta}")
        ax.set_xlabel(r"$\sigma_1$")
        ax.set_ylabel(r"$\sigma_2$")
        plt.show()

def overlay_cis():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plot_given_ci(axs, pd.read_csv("stats_sv.csv"), 'red', 'SV')
    plot_given_ci(axs, pd.read_csv("stats_dv.csv"), 'blue', 'DV')
    plot_given_ci(axs, pd.read_csv("stats_mx.csv"), 'green', 'SV+DV')

    axs[0].set_title("α(θ) Statistical Analysis")
    axs[0].set_xlabel("θ (deg)")
    axs[0].set_ylabel("α")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title("k(θ) Statistical Analysis")
    axs[1].set_xlabel("θ (deg)")
    axs[1].set_ylabel("k")
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


def plot_strength_surface(ax, a, k, min_strength=-20, max_strength=130):
    grid = np.linspace(min_strength, max_strength, 600)
    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)
    i1 = sig1 + sig2 + sig3
    j2 = (sig1**2 + sig2**2 + sig3**2 - sig1*sig2 - sig2*sig3 - sig3*sig1) / 3.0
    F = np.sqrt(j2) + a * i1 - k
    ax.contour(sig1, sig2, F, levels=[0], linewidths=2, colors='black', alpha=0.07)  # F=0 curve



def display_pca(z, threshold=0.999):
    """This applies pca to the scaled zs dataset and shows the eigenvalue decay (so you can choose how many components you want to keep)"""
    pca = PCA()
    z_pca = pca.fit_transform(z)

    np.set_printoptions(precision=5, suppress=True)

    y = (1 - np.cumsum(pca.explained_variance_ratio_)) / np.cumsum(pca.explained_variance_ratio_)
    print('Fraction of total variance explained when using k components:\n', np.cumsum(pca.explained_variance_ratio_))
    print('Ratio of unexplained to explained variance after keeping k components:\n', y)
    print(f'Eigenvalues: {pca.explained_variance_}')
    fig, ax = plt.subplots(figsize=(8, 6))
    if isinstance(threshold, int):
        threshold = [threshold]
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [item['color'] for item in list(color_cycle) if 'color' in item]
    for i, thresh in enumerate(threshold):
        suggested_idx = [idx for idx, value in enumerate(np.cumsum(pca.explained_variance_ratio_)) if value > thresh]
        suggested_dims = suggested_idx[0] + 1
        print(f"For a threshold of {thresh*100:.8g}%, you should choose {suggested_dims} dimensions.")
        ax.axvline(x=suggested_dims, ymin=0, ymax=1, color=colors[i], lw=1.5, linestyle='dashed', label=f'threshold={thresh*100:.8g}%')

    ax.plot(range(1, len(pca.explained_variance_)+1), y, lw=2, marker='o', c='k')
    ax.set_xlabel('Number of Principal Components')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_yscale('log')
    ax.set_ylabel('Cumulative Explained Variance (logscale)')
    ax.set_title('PCA Explained Variance')
    ax.grid(True)
    ax.legend()
    plt.show()


def fit_latent_density(z, pca_dims, gaussian_modes, n_samples, print_eigs=False, show_latent_space=False):
    """This applies pca on the dataset (should be scaled) and keeps specified number of components. Then it fits a 
    multivariate gaussian to the latent dataset, samples from it however many times you want, then applies inverse 
    pca and returns the samples (still scaled)"""
    pca = PCA(n_components=pca_dims)
    z_pca = pca.fit_transform(z)

    # print eigenvalues and explained variance if we want
    if print_eigs:
        print("Eigenvalues:", pca.explained_variance_)
        print("Explained variance ratio:", pca.explained_variance_ratio_)

    # now fit Gaussian Mixture in latent space:
    gmm = GaussianMixture(n_components=gaussian_modes, covariance_type="full", reg_covar=1e-4, random_state=42)
    gmm.fit(z_pca)
    samples_pca, _ = gmm.sample(n_samples=n_samples)
    # inverse PCA transform (back to standardized space)
    samples = pca.inverse_transform(samples_pca)

    if show_latent_space:
        max_samples = 1000
        plot_latent_space(z_pca, samples_pca[:min(len(samples_pca), max_samples)])

    return samples


def plot_alpha_k(samples, axs, periodic, n_samples=None, label='', return_params=False):
    """Plots the alphas and ks as functions of theta for all of the samples given. Must give axs which is len = 2 (one for alpha and one for k)"""
    # handle if samples is a pandas df
    samples = samples.to_numpy() if isinstance(samples, pd.DataFrame) else np.asarray(samples)

    def get_color(value, normalization, black=True):
        if black:
            return 'k'
        cmap = plt.get_cmap('rainbow')   # or 'jet', 'turbo', 'plasma', etc.
        return cmap(value / normalization)

    theta = np.linspace(0, 90, 200)
    if n_samples is None:
        n_samples = len(samples)

    if return_params:
        all_alphas = np.empty((n_samples, len(theta)))
        all_ks = np.empty((n_samples, len(theta)))

    for i in range(n_samples):  # plot samples
        alpha, k = get_alpha_k(samples[i, :], theta, 2, periodic=periodic)

        axs[0].plot(theta, alpha, c=get_color(i, n_samples, black=True), alpha=0.1)
        axs[1].plot(theta, k, c=get_color(i, n_samples, black=True), alpha=0.1)
        if return_params:
            all_alphas[i] = alpha
            all_ks[i] = k

    axs[0].set_xlabel("θ")
    axs[0].set_ylabel("α")
    axs[0].set_title(f"{label} α(θ)")
    axs[0].set_ylim(-0.2, 0.3)
    axs[1].set_xlabel("θ")
    axs[1].set_ylabel("k")
    axs[1].set_title(f"{label} k(θ)")
    axs[1].set_ylim(20, 70)

    if return_params:
        return all_alphas, all_ks


def get_alpha_k(params, theta, periodic=False):
    if periodic:
        omega = 2 * np.pi * theta / 60
    else:
        omega = 2 * np.pi * theta / 180
    
    # infer N from length of data
    N = int((len(params) - 2) / 4)

    z_alpha = params[0]
    z_k = params[2*N+1]

    for m in range(1, N + 1):
        cos_coeff_a = params[2 * m - 1]
        sin_coeff_a = params[2 * m]
        z_alpha += cos_coeff_a * np.cos(m * omega) + sin_coeff_a * np.sin(m * omega)
        cos_coeff_k = params[(2*N+1)+(2 * m - 1)]
        sin_coeff_k = params[(2*N+1)+(2 * m)]
        z_k += cos_coeff_k * np.cos(m * omega) + sin_coeff_k * np.sin(m * omega)

    # once we have the value of z_alpha and z_k, we must transform back to alpha and k:
    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

    alpha = -np.sqrt(3) / 6 + softplus(z_alpha)
    k = softplus(z_k)
    return alpha, k


def plot_ci(true_z, samples_z, periodic, title=None, save_csv=None):
    theta = np.linspace(0, 90, 200)
    alphas = []
    ks = []

    n_samples = samples_z.shape[0]

    # preallocate arrays
    alphas = np.empty((n_samples, len(theta)))
    ks = np.empty((n_samples, len(theta)))

    # loop, but fill directly into arrays
    for i in range(n_samples):
        alpha, k = get_alpha_k(samples_z[i, :], theta, periodic=periodic)
        alphas[i, :] = alpha
        ks[i, :] = k

    alpha_mean = np.mean(alphas, axis=0)
    alpha_lower = np.percentile(alphas, 2.5, axis=0)
    alpha_upper = np.percentile(alphas, 97.5, axis=0)

    k_mean = np.mean(ks, axis=0)
    k_lower = np.percentile(ks, 2.5, axis=0)
    k_upper = np.percentile(ks, 97.5, axis=0)

    print(f"K mean at theta=0: {k_mean[0]}. at theta=90: {k_mean[-1]}")
    print(f"K lower 95% bound at theta=0: {k_lower[0]} at theta=90: {k_lower[-1]}")
    print(f"K upper 95% bound at theta=0: {k_upper[0]} at theta=90: {k_upper[-1]}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    all_real_alphas, all_real_ks = plot_alpha_k(true_z, axs, label='true', periodic=periodic, return_params=True)  # plot the true z's on the ax
    # α(θ)
    axs[0].fill_between(theta, alpha_lower, alpha_upper, color='lightblue', alpha=0.6, label='95% CI')
    axs[0].plot(theta, np.mean(all_real_alphas, axis=0), color='black', linewidth=3, label='True Mean')
    axs[0].plot(theta, alpha_mean, color='gold', linewidth=2, label='Sample Mean', linestyle='--')
    axs[0].plot([], [], color='k', alpha=0.1, label='True Data')  # for the label
    axs[0].set_title("α(θ): True vs Sampled 95% CI")
    axs[0].set_xlabel("θ (deg)")
    axs[0].set_ylabel("α")
    axs[0].legend()
    axs[0].grid(True)

    # k(θ)
    axs[1].fill_between(theta, k_lower, k_upper, color='lightblue', alpha=0.8, label='95% CI')
    axs[1].plot(theta, np.mean(all_real_ks, axis=0), color='black', linewidth=3, label='True Mean')
    axs[1].plot(theta, k_mean, color='gold', linewidth=2, label='Sample Mean', linestyle='--')
    axs[1].plot([], [], color='k', alpha=0.1, label='True Data')
    axs[1].set_title("k(θ): True vs Sampled 95% CI")
    axs[1].set_xlabel("θ (deg)")
    axs[1].set_ylabel("k")
    axs[1].legend()
    axs[1].grid(True)

    if title is not None:
        fig.suptitle(title)
    
    if save_csv is not None:
        save_df = pd.DataFrame(columns=['theta', 'a mean', 'a lower 95', 'a upper 95', 'k mean', 'k lower 95', 'k upper 95'])
        save_df['theta'] = theta
        save_df['a mean'] = alpha_mean
        save_df['a lower 95'] = alpha_lower
        save_df['a upper 95'] = alpha_upper
        save_df['k mean'] = k_mean
        save_df['k lower 95'] = k_lower
        save_df['k upper 95'] = k_upper
        save_df.to_csv(f'{save_csv}', index=False)

    plt.tight_layout()
    plt.show()


def plot_latent_space(latent_real, latent_sampled, bandwidth='scott', resolution=200, scatter_alpha=0.1, scatter_size=2):
    n_latent = latent_real.shape[1]
    fig, axs = plt.subplots(n_latent, n_latent, figsize=(2.5 * n_latent, 2.5 * n_latent))

    max_val = max(np.max(latent_sampled), np.max(latent_real))
    max_val = 8

    for i in range(n_latent):
        for j in range(n_latent):
            ax = axs[i, j]

            if i == j:
                # --- 1D KDE on diagonal ---
                x = latent_real[:, i]
                kde = gaussian_kde(x, bw_method=bandwidth)
                xs = np.linspace(np.min(x), np.max(x), resolution)
                ys = kde(xs)
                ax.plot(xs, ys, color='blue', lw=2)
                ax.fill_between(xs, ys, color='blue', alpha=0.3)

                # overlay histogram of sampled data
                x_samp = latent_sampled[:, i]
                kde_samp = gaussian_kde(x_samp, bw_method=bandwidth)
                ax.plot(xs, kde_samp(xs), color='k', lw=2)
                ax.fill_between(xs, kde_samp(xs), color='k', alpha=0.3)
            else:
                # --- 2D KDE on off-diagonals ---
                x = latent_real[:, j]
                y = latent_real[:, i]

                max_val = max(np.max(latent_real[:, j]), np.max(latent_real[:, i]))
                min_val = min(np.min(latent_real[:, j]), np.min(latent_real[:, i]))
                # print(round(np.mean(x), 4), round(np.mean(y), 4))
                # print(round(np.std(x), 4), round(np.std(y), 4))
                kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

                X, Y = np.meshgrid(np.linspace(min_val, max_val, resolution), np.linspace(min_val, max_val, resolution))
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                # plot KDE heatmap
                ax.imshow(Z, extent=[min_val, max_val, min_val, max_val], origin='lower', cmap="plasma", aspect='auto', alpha=0.9)
                # overlay sampled scatter points
                # ax.scatter(latent_sampled[:, j], latent_sampled[:, i], color='k', s=scatter_size, alpha=scatter_alpha)
                # ax.axvline(0, lw=0.8, color='white', alpha=0.6)
                # ax.axhline(0, lw=0.8, color='white', alpha=0.6)
                ax.scatter(x[7], y[7], color='g', s=2, alpha=0.5)
                ax.scatter(latent_sampled[22, j], latent_sampled[22, i], color='r', s=2, alpha=0.5)

            # axis labels
            if i == n_latent - 1:
                ax.set_xlabel(f"Latent {j+1}")
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f"Latent {i+1}")
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle("Joint KDE (real) + Sample Scatter (latent space)", y=1.02, fontsize=14)
    plt.show()


def plot_loss_data(df_params, loss_measure):
    fig, ax = plt.subplots(figsize=(8, 8))

    if "Defect Type" in df_params.columns:
        unique_types = df_params["Defect Type"].unique()
        color_map = {t: c for t, c in zip(unique_types, plt.cm.tab10.colors)}
        # Map to colors
        colors = df_params["Defect Type"].map(color_map)
    else:
        colors='k'

    ax.set_title("Error from minimization")

    ax.scatter(df_params["Defect Random Seed"], df_params[loss_measure], c=colors)
    ax.set_xlabel("Defect Random Seed")
    ax.set_ylabel(loss_measure)

    # Add legend if there are multiple different defect types
    if "Defect Type" in df_params.columns:
        for defect_type, color in color_map.items():
            ax.scatter([], [], color=color, label=defect_type)
        ax.legend(title="Defect Type")

    ax.grid()
    fig.tight_layout()
    plt.show()


def plot_given_ci(axs, df, col, lab):
    axs[0].plot(df["theta"], df["a mean"], lw=3, color=col, label=f'{lab} Mean')
    axs[0].plot(df["theta"], df["a lower 95"], lw=2, color=col, linestyle='dashed', label=f'{lab} CI')
    axs[0].plot(df["theta"], df["a upper 95"], lw=2, color=col, linestyle='dashed')

    axs[1].plot(df["theta"], df["k mean"], lw=3, color=col, label=f'{lab} Mean')
    axs[1].plot(df["theta"], df["k lower 95"], lw=2, color=col, linestyle='dashed', label=f'{lab} CI')
    axs[1].plot(df["theta"], df["k upper 95"], lw=2, color=col, linestyle='dashed')


if __name__ == "__main__":
    main()