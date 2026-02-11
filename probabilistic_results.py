"""
This runs the full workflow of generating the probabilistic results -- it takes the 3D DP fits
as inputs, and the user can specify how they go down to the latent space, sample, project, etc.
"""


import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import random
from matplotlib.patches import Polygon
from filter_csv import filter_data
import local_config
import matplotlib as mpl


def main():
    # these are here for plotting for the paper
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']

    mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{xcolor}
    """

    mpl.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 10,      # match normalsize
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8, # slightly smaller like LaTeX
        'text.usetex': True,
    })

    # examples of how to use this
    defs = "sv"
    fo = 4
    # generate_full_pca_plot(fo)
    # df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")
    # md_df = pd.read_csv(f'{local_config.DATA_DIR}/rotation_tests/all_simulations.csv')
    # exact_filters, or_filters = get_filters(defs)
    # raw_md_data = filter_data(md_df, exact_filters=exact_filters, or_filters=or_filters, remove_nones=True, remove_dupes=True, duplic_freq=(0, 91, 10))
    # full_workflow(df, show_pca=True)
    # full_workflow(df, defs=defs, pca_dims=13, gaussian_modes=1, show_functions=True)

    # full_workflow(df, defs=defs, pca_dims=13, gaussian_modes=1, show_ci=True, save_ci_csv=f"stats_{defs}.csv")
    # full_workflow(df, defs=defs, pca_dims=14, gaussian_modes=1, ci_theta=30, raw_md_data=raw_md_data)
    # full_workflow(df, pca_dims=6, ss_theta=0, periodic=False)
    # overlay_cis()
    # generate_full_param_cis()
    generate_full_strength_cis()



def full_workflow(df, pca_dims=2, gaussian_modes=1, show_pca=False, show_functions=False, 
                  periodic=False, show_ci=False, show_latent=False, show_loss=False,
                  n_samples=100000, save_ci_csv=None, ss_theta=None, defs="", ci_theta=None, 
                  raw_md_data=None):
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

    zs = drop_non_z_columns(df)
    
    if show_pca:
        display_pca(zs)
        return
    
    # apply pca, fit gaussian, sample, apply inverse pca
    samples_z = fit_latent_density(zs, pca_dims, gaussian_modes, n_samples, show_latent_space=show_latent)

    if show_functions:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.83, 2.45))
        plot_alpha_k(zs, axs, label='true', periodic=periodic)
        # plot_alpha_k(samples_z, axs[:, 1], n_samples=1000, label='generated', periodic=periodic)
        fig.savefig(f'ak_fo4_{defs}.pdf', bbox_inches='tight')

    if show_ci:
        char = defs[0].lower()
        plot_title = "Single" if char == 's' else "Double" if char == 'd' else "Mixed" if char == 'm' else "No"
        plot_ci(zs, samples_z, periodic, title=f"{plot_title} Vacancies", save_csv=save_ci_csv)
    
    if ss_theta is not None:
        samples = samples_z.to_numpy() if isinstance(samples_z, pd.DataFrame) else np.asarray(samples_z)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        for i in range(min(len(samples), 1000)):
            a, k = get_alpha_k(samples[i], ss_theta, periodic=periodic)
            plot_strength_surface(ax, a, k)
        ax.set_title(f"{defs.upper()} Artificial Strength Surfaces: θ={ss_theta}")
        ax.set_xlabel(r"$\sigma_1$")
        ax.set_ylabel(r"$\sigma_2$")
        plt.show()

    if ci_theta is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        if raw_md_data is not None:
            plot_raw_data(ax, raw_md_data, ci_theta, color='red')
        for i in range(min(len(zs), 1000)):
            a, k = get_alpha_k(zs[i], ci_theta, periodic=False)
            plot_strength_surface(ax, a, k)
        ax.set_title(f"{defs.upper()} Strength Surface CI: θ={ci_theta}")
        ax.set_xlabel(r"$\sigma_1$")
        ax.set_ylabel(r"$\sigma_2$")
        ax.plot([], [], color='k', alpha=0.1, label='True Data')
        plot_strength_ci(zs, ci_theta, ax=ax, show_ci=False, mean_color='k', mean_lw=3, label="True Mean")
        plot_strength_ci(samples_z, ci_theta, ax=ax, mean_linestyle='--')
        plt.show()

def set_lower_legend(fig, axs, y_loc=-0.02):
    handles = []
    labels = []
    for ax in axs.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))
    
    leg = fig.legend(
        unique.values(),
        unique.keys(),
        loc='lower center',
        bbox_to_anchor=(0.5, y_loc),
        ncol=len(unique)     # all labels on one line
    )
    return leg

def generate_full_pca_plot(fo):
    defs = ["sv", "dv", "mx"]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5.83, 2.43))
    for i, defect in enumerate(defs):
        if i == 0:
            show_y=True
        else:
            show_y=False
        df = pd.read_csv(f"z_{defect}{fo}_reg1e-1.csv")
        zs = drop_non_z_columns(df)
        display_pca(zs, ax=axs[i], show_y=show_y, title=f"{defect.upper()}")
    set_lower_legend(fig, axs, y_loc=0)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.28, top=0.90)
    fig.savefig("PCA_all.pdf")


def generate_full_strength_cis():
    defs = ["sv", "dv", "mx"]
    fo = 4
    fig, axs = plt.subplots(3, 3, figsize=(5.83, 5.83)) 
    for i, defect in enumerate(defs):
        df = pd.read_csv(f"z_{defect}{fo}_reg1e-1.csv")
        md_df = pd.read_csv(f'{local_config.DATA_DIR}/rotation_tests/all_simulations.csv')
        exact_filters, or_filters = get_filters(defect)
        raw_md_data = filter_data(md_df, exact_filters=exact_filters, or_filters=or_filters, remove_nones=True, remove_dupes=True, duplic_freq=(0, 91, 10))

        zs = drop_non_z_columns(df)
        dims, modes = decipher_model_params(defect)
        samples_z = fit_latent_density(zs, dims, modes, 100000)

        thetas = [0, 30, 90]
        for j, theta in enumerate(thetas):
            plot_raw_data(axs[i, j], raw_md_data, theta, color='k', s=3, lab=None)

            # for sample in range(min(len(zs), 1000)):
            #     a, k = get_alpha_k(zs[sample], theta, periodic=False)
            #     plot_strength_surface(axs[i, j], a, k)

            if j == 0:
                show_y = True
            else:
                show_y = False
            if i == 2:
                show_x = True
            else:
                show_x = False

            plot_strength_ci(zs, theta, ax=axs[i, j], label='True Surface Mean', mean_color='red', mean_linestyle='solid', mean_lw=2.5, show_x_axis=show_x, show_y_axis=show_y)
            plot_strength_ci(samples_z, theta, ax=axs[i, j], label='Sample Mean', show_ci=True, mean_color='b', mean_linestyle='dashed', mean_lw=1.5, show_x_axis=show_x, show_y_axis=show_y)
            # axs[i, j].plot([], [], color='k', alpha=0.8, label='True Data')
            axs[i, j].grid()
            axs[i, j].scatter([], [], s=3, c='k', label="Raw MD Data")
            if i == 0:
                axs[i, j].set_title(fr"$\theta = {theta}$\textdegree")

            print(f"Plotted {defect} {theta}")

        leg = set_lower_legend(fig, axs)
        leg.get_frame().set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(f"strength_ci_all.pdf", bbox_inches='tight')


def generate_full_param_cis():
    defs = ["sv", "dv", "mx"]
    fo = 4
    fig, axs = plt.subplots(3, 2, figsize=(5.83, 6.8)) 
    axs[0, 0].set_title(r"$\alpha(\theta)$")
    axs[0, 1].set_title(r"$k(\theta)$")
    for i, defect in enumerate(defs):
        df = pd.read_csv(f"z_{defect}{fo}_reg1e-1.csv")
        zs = drop_non_z_columns(df)
        dims, modes = decipher_model_params(defect)
        samples_z = fit_latent_density(zs, dims, modes, 100000)
        plot_ci(zs, samples_z, False, axs=axs[i], show_x_axis=(i==2), show_legend=False)
        print(f"Plotted {defect}")
        leg = set_lower_legend(fig, axs)
        leg.get_frame().set_alpha(1.0)

def decipher_model_params(defect):
        if defect == "sv":
            return 14, 1
        elif defect == "dv":
            return 13, 3
        else:
            return 13, 1

def drop_non_z_columns(df):
    keep = [col for col in df.columns if col.startswith("z")]
    return df[keep].copy().to_numpy()

def get_filters(defs):
    if defs == "mx":
        exact_filters = {"Defects": '{"DV": 0.25, "SV": 0.25}'}
        or_filters = {}
    elif defs == "none":
        exact_filters = {"Defects": "None"}
        or_filters = {"Theta Requested": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}  # filter out all of the uniaxials
    else:
        exact_filters = {"Defects": f'{{"{defs.upper()}": 0.5}}'}
        or_filters = {}
    return exact_filters, or_filters

def overlay_cis():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.83, 2.7))
    plot_given_ci(axs, pd.read_csv("stats_sv.csv"), 'red', 'SV')
    plot_given_ci(axs, pd.read_csv("stats_dv.csv"), 'blue', 'DV')
    plot_given_ci(axs, pd.read_csv("stats_mx.csv"), 'green', 'MX')

    ticks = np.arange(0, 91, 30)

    # axs[0].set_title("\alpha(\theta) Statistical Analysis")
    axs[0].set_xlabel(r"$\theta$ (\textdegree)")
    axs[0].set_ylabel(r"$\alpha$")
    axs[0].grid(True)
    axs[0].set_xticks(ticks)
    axs[0].set_ylim(-0.1, 0.4)

    # axs[1].set_title("k(\theta) Statistical Analysis")
    axs[1].set_xlabel(r"$\theta$ (\textdegree)")
    axs[1].set_ylabel(r"$k$")
    axs[1].grid(True)
    axs[1].set_xticks(ticks)
    axs[1].set_ylim(25, 80)

    axs[0].set_xlim([0, 90])
    axs[1].set_xlim([0, 90])
    
    set_lower_legend(fig, axs, y_loc=-0.02)
    
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.22, top=0.98)
    plt.savefig(f"param_ci_overlay.pdf")


def plot_strength_surface(ax, a, k, min_strength=-20, max_strength=130, color='k', alpha=0.1, label=None):
    grid = np.linspace(min_strength, max_strength, 600)
    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)
    i1 = sig1 + sig2 + sig3
    j2 = (sig1**2 + sig2**2 + sig3**2 - sig1*sig2 - sig2*sig3 - sig3*sig1) / 3.0
    F = np.sqrt(j2) + a * i1 - k
    ax.contour(sig1, sig2, F, levels=[0], linewidths=2, colors=color, alpha=alpha)  # F=0 curve
    if label is not None:
        ax.plot([], [], c=color, lw=2, alpha=0.8, label=label)



def display_pca(z, ax=None, threshold=[1e-3, 1e-4, 1e-5], show_x=True, show_y=True, title=''):
    """This applies pca to the scaled zs dataset and shows the eigenvalue decay (so you can choose how many components you want to keep)"""
    def threshold_to_power(x):
        return f"10^{{-{int(round(-np.log10(x)))}}}"
    
    pca = PCA()
    z_pca = pca.fit_transform(z)

    np.set_printoptions(precision=5, suppress=True)

    # y = (1 - np.cumsum(pca.explained_variance_ratio_)) / np.cumsum(pca.explained_variance_ratio_)
    # y = pca.explained_variance_
    y = 1 - np.cumsum(pca.explained_variance_ratio_)
    print('Fraction of total variance explained when using k components:\n', np.cumsum(pca.explained_variance_ratio_))
    print(f'Eigenvalues: {pca.explained_variance_}')
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.56, 3))
    if isinstance(threshold, float):
        threshold = [threshold]
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [item['color'] for item in list(color_cycle) if 'color' in item]
    for i, thresh in enumerate(threshold):
        suggested_idx = [idx for idx, value in enumerate(y) if value < thresh]
        suggested_dims = suggested_idx[0] + 1
        print(f"For a threshold of {threshold_to_power(thresh)}, you should choose {suggested_dims} dimensions.")
        ax.axvline(x=suggested_dims, ymin=0, ymax=1, color=colors[i], lw=1, linestyle='dashed', label=rf'$E_{{\text{{thresh}}}}={threshold_to_power(thresh)}$')

    ax.plot(range(1, len(pca.explained_variance_)+1), y, lw=1, marker='o', c='k', markersize=3, label=r'$E$')
    ax.set_xlabel(r'$d$')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_ylabel(r'Recovery Error $E$')
    ax.set_title(title)
    ax.grid(True)
    # ax.legend()

    if not show_x:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel("")

    if not show_y:
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.set_ylabel("")

    if ax is None:
        fig.savefig('PCA_eigs.pdf', bbox_inches='tight')


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
        alpha, k = get_alpha_k(samples[i, :], theta, periodic=periodic)

        axs[0].plot(theta, alpha, c=get_color(i, n_samples, black=True), alpha=0.1, lw=0.9)
        axs[1].plot(theta, k, c=get_color(i, n_samples, black=True), alpha=0.1, lw=0.9)
        if return_params:
            all_alphas[i] = alpha
            all_ks[i] = k
    ticks = np.arange(0, 91, 30)
    axs[0].set_xticks(ticks)
    axs[1].set_xticks(ticks)
    axs[0].grid()
    axs[1].grid()

    axs[0].set_xlabel(r"$\theta$")
    axs[0].set_ylabel(r"$\alpha$")
    axs[0].set_title(rf"{label} $\alpha(\theta)$")
    axs[0].set_ylim(-0.1, 0.4)
    axs[1].set_xlabel(r"$\theta$")
    axs[1].set_ylabel(r"$k$")
    axs[1].set_title(rf"{label} $k(\theta)$")
    axs[1].set_ylim(25, 80)

    axs[0].set_xlim(0, 90)
    axs[1].set_xlim(0, 90)

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


def plot_ci(true_z, samples_z, periodic, title=None, save_csv=None, axs=None, show_legend=True, show_x_axis=True):
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
    # 2.92 perfect square, 2.505 good for labels, 2.22 good for no labels, 2.23 if you have legend
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(5.83, 2.505))

    all_real_alphas, all_real_ks = plot_alpha_k(true_z, axs, label='true', periodic=periodic, return_params=True)  # plot the true z's on the ax
    # α(θ)
    axs[0].plot(theta, np.mean(all_real_alphas, axis=0), color='r', linewidth=2.5, label='True Mean')
    axs[0].fill_between(theta, alpha_lower, alpha_upper, color='lightblue', alpha=0.6, label=r'95\% CI')
    axs[0].plot(theta, alpha_mean, color='b', linewidth=1.5, label='Sample Mean', linestyle='--')
    axs[0].plot([], [], color='k', alpha=0.8, label='True Data')  # for the label
    # axs[0].set_title("\alpha(\theta): True vs Sampled 95% CI")
    axs[0].set_xlabel(r"$\theta$ (\textdegree)")
    axs[0].set_ylabel(r"$\alpha$")
    ticks = np.arange(0, 91, 30)
    axs[0].set_xticks(ticks)
    # axs[0].legend()
    axs[0].grid(True)

    # k(θ)
    axs[1].plot(theta, np.mean(all_real_ks, axis=0), color='r', linewidth=2.5, label='True Mean')
    axs[1].fill_between(theta, k_lower, k_upper, color='lightblue', alpha=0.8, label=r'95\% CI')
    axs[1].plot(theta, k_mean, color='b', linewidth=1.5, label='Sample Mean', linestyle='--')
    axs[1].plot([], [], color='k', alpha=0.8, label='True Data')
    # axs[1].fill_between(theta, k_lower, k_upper, color='lightblue', alpha=0.8)
    # axs[1].plot(theta, np.mean(all_real_ks, axis=0), color='r', linewidth=2.5)
    # axs[1].plot(theta, k_mean, color='b', linewidth=1.5, linestyle='--')

    # axs[1].set_title("k(\theta): True vs Sampled 95% CI")
    axs[1].set_xlabel(r"$\theta$ (\textdegree)")
    axs[1].set_ylabel("$k$")
    axs[1].set_xticks(ticks)

    if show_legend:
        handles, labels = fig.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys())

    axs[1].grid(True)

    if not show_x_axis:
        axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[0].set_xlabel("")
        axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[1].set_xlabel("")

    # if title is not None:
    #     fig.suptitle(title)
    
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
    plt.savefig(f"param_ci_{title}.pdf", bbox_inches='tight')


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
    axs[0].plot(df["theta"], df["a mean"], lw=1.5, color=col, label=f'{lab}')
    axs[0].plot(df["theta"], df["a lower 95"], lw=0.9, color=col, linestyle='dashed')
    axs[0].plot(df["theta"], df["a upper 95"], lw=0.9, color=col, linestyle='dashed')

    axs[1].plot(df["theta"], df["k mean"], lw=1.5, color=col, label=f'{lab}')
    axs[1].plot(df["theta"], df["k lower 95"], lw=0.9, color=col, linestyle='dashed')
    axs[1].plot(df["theta"], df["k upper 95"], lw=0.9, color=col, linestyle='dashed')


def strength_ci(z_samples, theta_deg, r_grid=None):
    """
    Given:
      z_samples : array shape (n_samples, n_params)
      theta_deg : scalar angle in degrees
      r_grid    : 1D array of stress ratios r = sigma2 / sigma1

    Returns:
      dict with mean, lower, upper curves in strength space:
        {
          "r": r_grid,
          "sigma1_mean": ...,
          "sigma1_low" : ...,
          "sigma1_high": ...,
          "sigma2_mean": ...,
          "sigma2_low" : ...,
          "sigma2_high": ...
        }
    """
    z_samples = np.asarray(z_samples)
    n_samples, _ = z_samples.shape

    if r_grid is None:
        r_grid = np.linspace(0.0, 1.0, 51)
    else:
        r_grid = np.asarray(r_grid)
    n_r = len(r_grid)

    # storage: sigma1 for each sample and each ratio
    sigma1_samples = np.zeros((n_samples, n_r))

    # precompute the r-dependent piece that does not depend on z
    base_term = np.sqrt(1.0 + r_grid**2 - r_grid) / np.sqrt(3.0)

    for s in range(n_samples):
        alpha_s, k_s = get_alpha_k(z_samples[s, :], theta_deg)

        # denominator: base_term + alpha(θ) * (1 + r)
        denom = base_term + alpha_s * (1.0 + r_grid)
        # denom should never be zero because of constraints from softplus fn
        sigma1_samples[s, :] = k_s / denom

    # compute stats along the sample axis
    sigma1_mean = np.mean(sigma1_samples, axis=0)
    sigma1_low  = np.percentile(sigma1_samples,  2.5, axis=0)
    sigma1_high = np.percentile(sigma1_samples, 97.5, axis=0)

    sigma2_mean = r_grid * sigma1_mean
    sigma2_low  = r_grid * sigma1_low
    sigma2_high = r_grid * sigma1_high

    return {
        "r": r_grid,
        "sigma1_mean": sigma1_mean,
        "sigma1_low" : sigma1_low,
        "sigma1_high": sigma1_high,
        "sigma2_mean": sigma2_mean,
        "sigma2_low" : sigma2_low,
        "sigma2_high": sigma2_high,
    }

def plot_strength_ci(z_samples, theta_deg, r_grid=None, ax=None, label=None, show_ci=True, mean_color='gold', mean_linestyle='solid', mean_lw=2,
                    show_x_axis=True, show_y_axis=True):
    """
    Convenience wrapper that calls strength_curve_CI_for_theta and plots the mean and CI band.
    """
    stats = strength_ci(z_samples, theta_deg, r_grid=r_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    s1m = np.concatenate([stats["sigma1_mean"], stats["sigma2_mean"][::-1]])
    s1L = np.concatenate([stats["sigma1_low"], stats["sigma2_low"][::-1]])
    s1U = np.concatenate([stats["sigma1_high"], stats["sigma2_high"][::-1]])
    s2m = np.concatenate([stats["sigma2_mean"], stats["sigma1_mean"][::-1]])
    s2L = np.concatenate([stats["sigma2_low"], stats["sigma1_low"][::-1]])
    s2U = np.concatenate([stats["sigma2_high"], stats["sigma1_high"][::-1]])

    # mean curve
    ax.plot(s1m, s2m, color=mean_color, lw=mean_lw, linestyle=mean_linestyle, label=f"Sample Mean" if label is None else label)

    if show_ci:
        fill_between_parametric(ax, s1L, s2L, s1U, s2U, facecolor='lightblue', alpha=0.6, edgecolor=None)
        ax.plot([], [], marker='s', markersize=12, markerfacecolor='lightblue', markeredgecolor='lightblue', alpha=0.6, linestyle='None', label=r'95\% CI')

    ax.set_xlabel(r"$\sigma_1$ (GPa)")
    ax.set_ylabel(r"$\sigma_2$ (GPa)")
    ax.set_xlim(-5, 120)
    ax.set_ylim(-5, 120)
    ax.axvline(0, c='k')
    ax.axhline(0, c='k')
    # ax.set_aspect("equal", adjustable="box")
    # ax.legend()

    ticks = [0, 30, 60, 90, 120]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if not show_x_axis:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel("")

    if not show_y_axis:
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.set_ylabel("")


    return ax

def fill_between_parametric(ax, x1, y1, x2, y2, **kwargs):
    """
    Fill region between two parametric curves:
        (x1[i], y1[i]) = lower branch
        (x2[i], y2[i]) = upper branch
    """
    # Build polygon: lower curve forward, upper curve backward
    xs = np.concatenate([x1, x2[::-1]])
    ys = np.concatenate([y1, y2[::-1]])

    poly = Polygon(np.column_stack([xs, ys]), closed=True, **kwargs)
    ax.add_patch(poly)

def plot_raw_data(ax, md, theta, color='k', base_alpha=0.1, s=5, theta_buffer=4, lab="Raw MD Data"):
    filtered_df = filter_data(md, range_filters={"Theta": (theta-theta_buffer, theta+theta_buffer)}, shift_theta=False, flip_strengths=True)
    alpha = ((abs(filtered_df["Theta"]-theta) - theta_buffer) / -theta_buffer) * base_alpha
    ax.scatter(filtered_df["Strength_1"], filtered_df["Strength_2"], color=color, alpha=alpha, s=s, label=lab)

if __name__ == "__main__":
    main()