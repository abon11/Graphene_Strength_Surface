import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.random import multivariate_normal
from scipy.stats import gaussian_kde


def main():
    df = pd.read_csv("z_np_sv.csv")
    # full_workflow(df, show_pca=True)
    # full_workflow(df, n_components=6, show_ci=True, save_ci_csv='stats_dv.csv', periodic=False)
    full_workflow(df, n_components=6, ss_theta=0, periodic=False)
    # overlay_cis()



def full_workflow(df, n_components=2, show_pca=False, show_functions=False, 
                  periodic=True, show_ci=False, show_latent=False, show_loss=False,
                  n_samples=100000, save_ci_csv=None, ss_theta=None):
    """Given the original dataset, this applies the full workflow"""

    # to get a gague of how good the minimization was for this dataset (and maybe remove some outliers?)
    if show_loss:
        plot_loss_data(df, "Total Loss")

    if "Defect Type" in df.columns:
        zs = df.drop(columns=["Defect Type", "Defect Random Seed", "Total Loss", "RMSE"])
    else:
        zs = df.drop(columns=["Defect Random Seed", "Total Loss", "RMSE"])
    scaler = StandardScaler()  # define scaler
    z_scaled = scaler.fit_transform(zs)  # scale z data
    if show_pca:
        display_pca(z_scaled)
        return
    
    # apply pca, fit gaussian, sample, apply inverse pca
    z_sampled_scaled = fit_latent_density(z_scaled, n_components, n_samples, show_latent_space=show_latent)

    # inverse standardization (back to physical coefficient scale)
    samples_z = scaler.inverse_transform(z_sampled_scaled)

    if show_functions:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_alpha_k(zs, axs[0], label='true', periodic=periodic)
        plot_alpha_k(samples_z, axs[1], n_samples=1000, label='generated', periodic=periodic)
        plt.show()

    if show_ci:
        plot_ci(zs, samples_z, periodic, title="Single Vacancies", save_csv=save_ci_csv)
    
    if ss_theta is not None:
        samples = samples_z.to_numpy() if isinstance(samples_z, pd.DataFrame) else np.asarray(samples_z)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        for i in range(min(len(samples), 100)):
            a, k = get_alpha_k(samples[i], ss_theta, periodic=False)
            plot_strength_surface(ax, a, k)
        
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
    ax.contour(sig1, sig2, F, levels=[0], linewidths=2, colors='blue', alpha=0.2)  # F=0 curve


def display_pca(z_scaled):
    """This applies pca to the scaled zs dataset and shows the eigenvalue decay (so you can choose how many components you want to keep)"""
    pca = PCA()
    z_pca = pca.fit_transform(z_scaled)

    np.set_printoptions(precision=5, suppress=True)

    y = (1 - np.cumsum(pca.explained_variance_ratio_)) / np.cumsum(pca.explained_variance_ratio_)
    print('Fraction of total variance explained when using k components:\n', np.cumsum(pca.explained_variance_ratio_))
    print('Ratio of unexplained to explained variance after keeping k components:\n', y)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(pca.explained_variance_)+1), y, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()


def fit_latent_density(z_scaled, n_components, n_samples, print_eigs=False, show_latent_space=False):
    """This applies pca on the dataset (should be scaled) and keeps specified number of components. Then it fits a 
    multivariate gaussian to the latent dataset, samples from it however many times you want, then applies inverse 
    pca and returns the samples (still scaled)"""
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z_scaled)

    # print eigenvalues and explained variance if we want
    if print_eigs:
        print("Eigenvalues:", pca.explained_variance_)
        print("Explained variance ratio:", pca.explained_variance_ratio_)

    # now fit MV normal in latent space:
    mu = z_pca.mean(axis=0)
    Sigma = np.cov(z_pca, rowvar=False)
    samples_pca = multivariate_normal(mu, Sigma, size=n_samples)
    # inverse PCA transform (back to standardized space)
    samples_scaled = pca.inverse_transform(samples_pca)

    if show_latent_space:
        plot_latent_space(z_pca, samples_pca)

    return samples_scaled


def plot_alpha_k(samples, axs, periodic, n_samples=None, label='', return_params=False):
    """Plots the alphas and ks as functions of theta for all of the samples given. Must give axs which is len = 2 (one for alpha and one for k)"""
    # handle if samples is a pandas df
    samples = samples.to_numpy() if isinstance(samples, pd.DataFrame) else np.asarray(samples)

    def get_color(value, normalization, black='true'):
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
    axs[0].set_ylabel("α value")
    axs[0].set_title(f"{label.capitalize()} α(θ)")
    axs[0].set_ylim(-0.2, 0.3)
    axs[1].set_xlabel("θ")
    axs[1].set_ylabel("k value")
    axs[1].set_title(f"{label.capitalize()} k(θ)")
    axs[1].set_ylim(20, 70)

    if return_params:
        return all_alphas, all_ks


def get_alpha_k(params, theta, return_k=True, periodic=True):
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


def plot_latent_space(latent_real, latent_sampled, bandwidth='scott', resolution=150, scatter_alpha=0.15, scatter_size=5):
    n_latent = latent_real.shape[1]
    fig, axs = plt.subplots(n_latent, n_latent, figsize=(2.5 * n_latent, 2.5 * n_latent))

    for i in range(n_latent):
        for j in range(n_latent):
            ax = axs[i, j]

            if i == j:
                # --- 1D KDE on diagonal ---
                x = latent_real[:, i]
                kde = gaussian_kde(x, bw_method=bandwidth)
                xs = np.linspace(-8, 8, resolution)
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
                kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

                X, Y = np.meshgrid(np.linspace(-8, 8, resolution), np.linspace(-8, 8, resolution))
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                # plot KDE heatmap
                ax.imshow(Z, extent=[-8, 8, -8, 8], cmap="plasma", aspect='auto', alpha=0.8)
                # overlay sampled scatter points
                ax.scatter( latent_sampled[:, j], latent_sampled[:, i], color='k', s=scatter_size, alpha=scatter_alpha)

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