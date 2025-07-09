import pandas as pd
from plot_StrengthSurface import filter_data
import local_config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from DP_model import MadeSurface
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.stats import jarque_bera
from scipy.stats import norm, rankdata
from scipy.interpolate import interp1d


def main():
    plot_together = True

    # read the DP models from the csv
    df_params = pd.read_csv("drucker_prager_params_thetas.csv")
    alphas = df_params["alpha"].values
    ks = df_params["k"].values
    seeds = df_params["Theta Requested"].values

    # turn them into Surface objects for convenience (and plot if we want)
    surfaces = []
    for a, k, seed in zip(alphas, ks, seeds):
        surface = MadeSurface(a, k, interest_value="Theta", instance=seed)
        if not plot_together:
            plot_surface_fit(surface)

        surfaces.append(surface)
    if plot_together:
        plot_all_surfaces(surfaces)

    plot_all_surfaces([surfaces[3], surfaces[9]], showlabels=True, title="30 and 90 degrees")

    

    # sns.histplot(alphas, kde=True, stat="density")
    # # sns.kdeplot(alphas, bw_adjust=0.5, label="KDE")
    # plt.title("Distribution of Alpha")
    # plt.savefig("testalpha.png")

    # sns.histplot(ks, kde=True, stat="density")
    # plt.title("Distribution of k")
    # plt.savefig("testk.png")

    # df = pd.DataFrame({'alpha': alphas, 'k': ks})
    # kde = fit_gaussian_kde(alphas, ks)
    # plot_joint_kde(kde, alphas, ks)
    # jb_a, p_a = jarque_bera(alphas)
    # jb_k, p_k = jarque_bera(ks)

    # print(f"Alpha: JB = {jb_a:.3f}, p = {p_a:.4g}")
    # print(f"k:     JB = {jb_k:.3f}, p = {p_k:.4g}")
    # # now we can do stats on the models
    # pdf, mean, cov = mv_normal_approx(alphas, ks)
    # plot_alpha_k(surfaces, pdf=pdf, ci=[mean, cov])
    # ci_stress_surfaces = sample_from_ellipse(mean, cov)
    # plot_all_surfaces(ci_stress_surfaces, mean=mean, title='90% CI Sampling')

    # Fit marginal distributions, converting to uniform distribution using empirical CDF
    # u_alpha = rankdata(alphas) / (len(alphas) + 1)
    # u_k = rankdata(ks) / (len(ks) + 1)

    # # Map uniforms to standard normals
    # z_alpha = norm.ppf(u_alpha)
    # z_k = norm.ppf(u_k)

    # # Model the joint distribution in standard normal space
    # Z = np.column_stack([z_alpha, z_k])
    # mean = np.mean(Z, axis=0)
    # cov = np.cov(Z.T)
    # copula_model = multivariate_normal(mean=mean, cov=cov)

    # # Sample new joint gaussian points
    # samples = copula_model.rvs(size=1000)  # shape (1000, 2)
    # z_alpha_new, z_k_new = samples[:, 0], samples[:, 1]

    # # Convert Gaussian samples to uniform
    # u_alpha_new = norm.cdf(z_alpha_new)
    # u_k_new = norm.cdf(z_k_new)

    # # Map back
    # alpha_new = inverse_empirical_cdf(alphas, u_alpha_new)
    # k_new = inverse_empirical_cdf(ks, u_k_new)
    # plot_alpha_k(alphas, ks, newdata=[alpha_new, k_new])

    # plot_pdf_in_gaussian_space(z_alpha, z_k, mean, cov)
    # estimate_pdf_in_alpha_k_space(alphas, ks, mean, cov)
    # plot_marginal_histogram(alphas, data2=alpha_new, label="alpha")
    # plot_marginal_histogram(ks, data2=k_new, label="k")
    # # plot_marginal_histogram(alpha_new, label="new_alpha_samples")
    # # plot_marginal_histogram(k_new, label="new_k_samples")
    # qq_plot(alphas, alpha_new, "alpha")
    # qq_plot(ks, k_new, "k")
    # plot_alpha_k(alpha_new, k_new, pdf=copula_model)

    # strengths = map_to_strength(alphas, ks)
    # # generic_scatter(strengths[0], strengths[1], xlab=r'$\sigma_{cs}$', ylab=r'$\sigma_{ts}$', title='data_in_stress')
    # new_strengths = map_to_strength(alpha_new, k_new)
    # generic_scatter(strengths[0], strengths[1], newdata=new_strengths, xlab=r'$\sigma_{cs}$', ylab=r'$\sigma_{ts}$', title='Uniaxial Strengths')


def inverse_empirical_cdf(original_data, u_vals):
    sorted_data = np.sort(original_data)
    n = len(sorted_data)
    quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    return np.interp(u_vals, quantiles, sorted_data)


def fit_gaussian_kde(alpha_vals, k_vals):
    data = np.vstack([alpha_vals, k_vals])  # shape (2, N)
    kde = gaussian_kde(data)
    return kde


# takes the multivariate normal approximation to generate PDF in terms of alpha and k
def mv_normal_approx(alphas, ks):
    X = np.column_stack([alphas, ks])
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    pdf = multivariate_normal(mean=mean, cov=cov)
    return pdf, mean, cov


def plot_pdf_in_gaussian_space(z_alpha, z_k, mean, cov, resolution=200):
    # Set up Gaussian grid
    z1 = np.linspace(min(z_alpha) - 3, max(z_alpha) + 3, resolution)
    z2 = np.linspace(min(z_k) - 3, max(z_k) + 3, resolution)
    Z1, Z2 = np.meshgrid(z1, z2)
    
    grid_points = np.column_stack([Z1.ravel(), Z2.ravel()])
    
    # Evaluate multivariate Gaussian PDF
    pdf_vals = multivariate_normal(mean=mean, cov=cov).pdf(grid_points)
    pdf_grid = pdf_vals.reshape(Z1.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Z1, Z2, pdf_grid, levels=30, cmap="viridis")
    plt.colorbar(contour, label="PDF")
    plt.scatter(z_alpha, z_k, color='red', s=10, label="Transformed Data")
    plt.title("PDF in Gaussian Space (z_alpha, z_k)")
    plt.xlabel("z_alpha")
    plt.ylabel("z_k")
    plt.legend()
    plt.savefig("gaussian_pdf.png")


def estimate_pdf_in_alpha_k_space(alphas, ks, mean, cov, resolution=50):

    # Step 1: Set up grid in (alpha, k) space
    alpha_grid = np.linspace(min(alphas), max(alphas), resolution)
    k_grid = np.linspace(min(ks), max(ks), resolution)
    A, K = np.meshgrid(alpha_grid, k_grid)
    
    # Step 2: Build inverse empirical CDF functions
    def empirical_cdf(x, samples):
        ranks = rankdata(samples)
        u = ranks / (len(samples) + 1)
        sorted_x = np.sort(samples)
        return interp1d(sorted_x, u, bounds_error=False, fill_value=(0,1))(x)

    u_alpha_vals = empirical_cdf(A.ravel(), alphas)
    u_k_vals = empirical_cdf(K.ravel(), ks)

    z_alpha_vals = norm.ppf(u_alpha_vals)
    z_k_vals = norm.ppf(u_k_vals)

    Z = np.column_stack([z_alpha_vals, z_k_vals])
    pdf_vals = multivariate_normal.pdf(Z, mean=mean, cov=cov)
    pdf_grid = pdf_vals.reshape(A.shape)

    # Step 3: Plot
    plt.figure(figsize=(8,6))
    cp = plt.contourf(A, K, pdf_grid, levels=20, cmap='viridis')
    plt.colorbar(cp, label="PDF")
    plt.xlabel("alpha")
    plt.ylabel("k")
    plt.title("Estimated PDF in (alpha, k) space")
    plt.savefig("PDF.png")


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


def plot_marginal_histogram(data, data2=None, label=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=30, color='blue', alpha=0.5, ls='dashed', lw=3, label="Original Data", density=True, edgecolor='black')
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {label}")
    if data2 is not None:
        ax.hist(data2, bins=30, color='green', alpha=0.5, ls='dotted', lw=3, label="Sampled Data", density=True, edgecolor='black')
        fig.legend()
    fig.savefig(f"{label}_dist.png")


def qq_plot(original, sampled, label):
    n = min(len(original), len(sampled))
    original = original[:n]
    sampled = sampled[:n]
    original_sorted = np.sort(original)
    sampled_sorted = np.sort(sampled)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.plot(original_sorted, sampled_sorted, 'o', alpha=0.6, label='Quantile Comparison')

    # Identity line for reference
    min_val = min(original_sorted[0], sampled_sorted[0])
    max_val = max(original_sorted[-1], sampled_sorted[-1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit')

    # Labels and layout
    plt.xlabel(f'Quantiles of Original {label}')
    plt.ylabel(f'Quantiles of Sampled {label}')
    plt.title(f'Q-Q Plot for {label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"qq_{label}.png")
    plt.close()


def plot_joint_kde(kde, alpha_vals, k_vals, resolution=100):
    alpha_grid = np.linspace(min(alpha_vals) - 0.01, max(alpha_vals) + 0.01, resolution)
    k_grid = np.linspace(0, max(k_vals) + 5, resolution)  # ensure k > 0
    A, K = np.meshgrid(alpha_grid, k_grid)
    
    # Flatten and stack grid for KDE evaluation
    coords = np.vstack([A.ravel(), K.ravel()])
    Z = kde(coords).reshape(A.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(A, K, Z, levels=30, cmap='viridis')
    plt.scatter(alpha_vals, k_vals, s=10, c='white', alpha=0.6)
    plt.colorbar(label='Density')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$k$')
    plt.title(r'Joint Density of $(\alpha, k)$ using Gaussian KDE')
    plt.savefig("KDE.png")


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


# changed this from inputing surfaces to alphas, ks
def plot_alpha_k(alphas, ks, newdata=None, pdf=None, ci=None, resolution=100):

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
        fig.savefig("alpha_vs_k_MAP_new.png")
    else:
        if newdata is not None:
            ax.scatter(newdata[0], newdata[1], label='Sampled Data', alpha=0.4, color='green', s=15)
        ax.scatter(alphas, ks, label='Original Data', alpha=0.4, color='k', s=15)
        ax.plot([0, 0], [25, 50], '--', color='red', label=r'$\sigma_{ts} = \sigma_{cs}$')
        if newdata is not None:
            fig.legend()
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel("k")
        ax.set_title("Drucker-Prager Fit Parameters")
        fig.savefig("alpha_vs_k_all.png")


def generic_scatter(x1, x2, newdata=None, xlab='', ylab='', title=''):
    fig, ax = plt.subplots()
    if newdata is not None:
        ax.scatter(newdata[0], newdata[1], label="Sampled Data", alpha=0.4, color='green', s=15)
    ax.scatter(x1, x2, label="Original Data", alpha=0.4, color='k', s=15)
    ax.plot([25, 110], [25, 110], '--', color='red', label=r'$\sigma_{ts} = \sigma_{cs}$')
    if newdata is not None:
        plt.legend()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xlim(25, 110)
    ax.set_ylim(25, 110)
    fig.savefig(f'{title}.png')


def plot_all_surfaces(surfaces, resolution=1000, mean=None, showlabels=False, title=None):
    # Set global grid range
    grid = np.linspace(-10, 130, resolution)

    sig1, sig2 = np.meshgrid(grid, grid)
    sig3 = np.zeros_like(sig1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cycle through colors for each seed
    colors = cm.Set1(np.linspace(0, 1, len(surfaces)))  # hsv for rainbow
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
        ax.contour(sig1, sig2, F, levels=[0], colors=[color], linewidths=2, alpha=1)
        if showlabels:
            ax.plot([], [], color=color, label=f"{surface.interest_value} {int(surface.instance)}")

    ax.plot([-50, 130], [0, 0], color='black')
    ax.plot([0, 0], [-50, 130], color='black')

    ax.tick_params(axis='both', labelsize=15)

    # ax.set_xlim(-15, 100)
    # ax.set_ylim(-15, 100)
    ax.set_xlim(-15, 130)
    ax.set_ylim(-15, 130)

    ax.set_xlabel(r"$\sigma_1$ (GPa)", fontsize=18)
    ax.set_ylabel(r"$\sigma_2$ (GPa)", fontsize=18)
    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title("DP Surfaces Overlayed by Random Seed", fontsize=20)
    if showlabels:
        ax.legend()

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