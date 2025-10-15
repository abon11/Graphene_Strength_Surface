import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, entropy, Mixture
import numpy as np
from scipy.stats import gamma
import pandas as pd
import seaborn as sns

def main():
    # load your fitted parameters CSV
    df = pd.read_csv("fitted_parameters.csv")
    params = df.sort_values(by='dataset', ignore_index=True)
    # visualize_all_marginals(params, "marginal_densities.png")
    # visualize_all_joints(params, "joint_densities.png")

    a_div, k_div, j_div = compare_distributions(params)
    divergence_visualization([a_div, k_div, j_div], ["Alpha Marginal", "K Marginal", "Joint Distribution"], "js_divergence.png")


def divergence_visualization(dfs, df_labels, filename):
    fig, axs = plt.subplots(nrows=1, ncols=len(dfs), figsize=(len(dfs) * 5, 5))
    for i in range(len(dfs)):
        print(f"\n{df_labels[i]}:")
        print(dfs[i])
        sns.heatmap(dfs[i], annot=True, fmt=".3f", cmap="cividis", square=True,
                    cbar_kws={"label": "JS Divergence"}, ax=axs[i])
        plt.title(f"{df_labels[i]}", fontsize=14)
        axs[i].set_xlabel("Dataset")
        axs[i].set_ylabel("Dataset")
        axs[i].set_title(f"{df_labels[i]}")
    plt.tight_layout()
    fig.suptitle("Pairwise Jensen-Shannon Divergences")
    print(f"JS-Divergence plot saved to {filename}")
    plt.savefig(filename)
    
def compare_distributions(params):
    a_dists = []
    k_dists = []
    names = []  # keep track of the names in this order so everything stays consistent
    joints = []
    for idx, row in params.iterrows():
        alpha_dist, k_dist = construct_marginals(row)
        joint_dist = reconstruct_copula(alpha_dist, k_dist, row["rho"])
        a_dists.append(alpha_dist)
        k_dists.append(k_dist)
        names.append(row["dataset"])
        joints.append(joint_dist)

    a_comparison = np.zeros((len(a_dists), len(a_dists)))
    k_comparison = np.zeros((len(a_dists), len(a_dists)))
    joint_comparison = np.zeros((len(a_dists), len(a_dists)))
    alph_vals = np.linspace(-0.3, 0.5, 200)
    k_vals = np.linspace(20, 100, 200)
    A, K = np.meshgrid(alph_vals, k_vals)
    for i in range(len(a_dists)):
        for j in range(len(a_dists)):
            a_comparison[i, j] = js_divergence_marginals(a_dists[i], a_dists[j], alph_vals)
            k_comparison[i, j] = js_divergence_marginals(k_dists[i], k_dists[j], k_vals)
            joint_comparison[i, j] = js_divergence_joint(joints[i], joints[j], A, K)
    
    alpha_js = pd.DataFrame(a_comparison, index=names, columns=names)
    k_js = pd.DataFrame(k_comparison, index=names, columns=names)
    joint_js = pd.DataFrame(joint_comparison, index=names, columns=names)

    return alpha_js, k_js, joint_js


def js_divergence_joint(f_joint1, f_joint2, A, K, base=np.e):
    """
    Jensen–Shannon divergence between two joint PDF functions f1(α,k) and f2(α,k).

    Parameters
    ----------
    f_joint1, f_joint2 : callable
        Functions that accept meshgrid arrays (A, K) and return joint PDF values.
    A, K : 2D arrays
        Meshgrids of α and k values.
    base : float, optional
        Logarithm base (default e → nats; base=2 → bits).

    Returns
    -------
    js : float
        Jensen–Shannon divergence between the two joint PDFs.
    """
    # Evaluate both joint PDFs on the same grid
    p = f_joint1(A, K).astype(float)
    q = f_joint2(A, K).astype(float)

    # Clip very small values for numerical stability
    eps = np.finfo(float).eps
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    # Normalize to integrate to 1
    norm_p = np.trapz(np.trapz(p, A[0, :]), K[:, 0])
    norm_q = np.trapz(np.trapz(q, A[0, :]), K[:, 0])
    p /= norm_p
    q /= norm_q

    # Mixture distribution
    m = 0.5 * (p + q)

    # Compute Kullback–Leibler terms (mask avoids log(0))
    kl_pm = np.trapz(
        np.trapz(p * np.log(p / m), A[0, :]),
        K[:, 0]
    )
    kl_qm = np.trapz(
        np.trapz(q * np.log(q / m), A[0, :]),
        K[:, 0]
    )

    js = 0.5 * (kl_pm + kl_qm)
    if base != np.e:
        js /= np.log(base)

    return js

def js_divergence_marginals(p_dist, q_dist, grid, base=np.e):
    """
    Compute the Jensen-Shannon divergence between two scipy.stats distributions.

    Parameters
    ----------
    p_dist, q_dist : scipy.stats.rv_continuous_frozen
        Distribution objects with a .pdf() method.
    grid : array-like
        Points over which to evaluate the PDFs.

    Returns
    -------
    js : float
        Jensen-Shannon divergence (in nats).
    """

    # Evaluate the PDFs
    p = p_dist.pdf(grid)
    q = q_dist.pdf(grid)

    # Normalize (safety)
    p /= np.trapz(p, grid)
    q /= np.trapz(q, grid)

    # Mixture distribution
    m = 0.5 * (p + q)

    # Avoid division by zero or log(0)
    mask_p = p > 0
    mask_q = q > 0
    mask_m = m > 0

    kl_pm = np.trapz(p[mask_p] * np.log(p[mask_p] / m[mask_m]), grid[mask_p])
    kl_qm = np.trapz(q[mask_q] * np.log(q[mask_q] / m[mask_m]), grid[mask_q])

    js = 0.5 * (kl_pm + kl_qm)

    if base != np.e:
        js /= np.log(base)

    return js


def construct_marginals(row):
    """
    Given a row of the fitted parameters, it returns the fitted gamma distributions
    """
    alpha_dist = gamma(a=row["alpha_shape"], scale=(1 / row["alpha_rate"]), loc=row.a0)
    k_dist = gamma(a=row["k_shape"], scale=(1 / row["k_rate"]))
    return alpha_dist, k_dist

def visualize_all_joints(params, filename):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), constrained_layout=True)
    meshes = []   # store the mappables for colorbar scaling
    for idx, row in params.iterrows():
        alpha_dist, k_dist = construct_marginals(row)
        joint_pdf = reconstruct_copula(alpha_dist, k_dist, row["rho"])

        i, j = divmod(idx, 3)  # this does [int(np.trunc(idx/3)), idx % 3]
        mesh = plot_joint_dist(ax[i, j], joint_pdf, title=row["dataset"])
        meshes.append(mesh)
        ax[i, j].set_xlabel(r"$\alpha$")
        ax[i, j].set_ylabel("k")

    # compute global min and max across all subplots (for them to share the same cmap)
    vmin = min(m.get_array().min() for m in meshes)
    vmax = max(m.get_array().max() for m in meshes)

    # apply same color scale to all meshes
    for m in meshes:
        m.set_clim(vmin, vmax)
    cbar = fig.colorbar(meshes[0], ax=ax, label=r"$f_{\alpha,k}(\alpha,k)$",
                        fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    print(f"Joint plots saved to {filename}")
    plt.savefig(filename)


def plot_joint_dist(ax, f_joint, title="Copula"):
    alpha_vals = np.linspace(-0.3, 0.5, 200)
    k_vals = np.linspace(20, 100, 200)
    A, K = np.meshgrid(alpha_vals, k_vals)

    Z = f_joint(A, K)
    mesh = ax.pcolormesh(A, K, Z, shading="auto", cmap="inferno")
    cs = ax.contour(A, K, Z, levels=[1e-12], colors="red", linewidths=1.5)
    ax.set_title(title)
    return mesh

def reconstruct_copula(alpha_marginal, k_marginal, rho):
    """
    Construct the full joint density function f_{alpha,k}(alpha, k)
    from the given marginal distributions and copula correlation rho.

    Parameters
    ----------
    alpha_marginal (scipy.stats.rv_continuous): Fitted marginal distribution for alpha.
    k_marginal (scipy.stats.rv_continuous): Fitted marginal distribution for k.
    rho (float): Correlation parameter (-1 < rho < 1).

    Returns
    -------
    joint_pdf (callable): A function f(alpha, k) that returns the joint PDF at any coordinates.
    """
    def joint_pdf(alpha, k):
        # Marginal densities
        f_alpha = alpha_marginal.pdf(alpha)
        f_k = k_marginal.pdf(k)

        # Marginal CDFs (uniformized)
        u1 = alpha_marginal.cdf(alpha)
        u2 = k_marginal.cdf(k)

        # Numerical stability: keep within (0,1)
        eps = np.finfo(float).eps
        u1 = np.clip(u1, eps, 1 - eps)
        u2 = np.clip(u2, eps, 1 - eps)

        # Transform to Gaussian space
        z1 = norm.ppf(u1)
        z2 = norm.ppf(u2)

        # Gaussian copula density
        denom = np.sqrt(1 - rho**2)
        exponent = (2 * rho * z1 * z2 - rho**2 * (z1**2 + z2**2)) / (2 * (1 - rho**2))
        c = (1.0 / denom) * np.exp(exponent)

        # Combine to get joint PDF
        return c * f_alpha * f_k

    return joint_pdf


def visualize_all_marginals(params, filename):
    """
    This plots all of the marginals on top of each other so you can see that they are correct (for alpha and k)
    """
        # make plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    # fig.tight_layout()
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel("Density")
    ax[0].set_title(r"$\alpha$ Marginal")
    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel("Density")
    ax[1].set_title(r"$k$ Marginal")

    colors = ['orangered', 'gold', 'limegreen', 'darkturquoise', 'steelblue', 'magenta']

    # pick a dataset
    for idx, row in params.iterrows():
        # reconstruct the marginals
        alpha_dist, k_dist = construct_marginals(row)

        plot_gamma(ax[0], alpha_dist, label=row["dataset"], color=colors[idx % len(colors)])
        plot_gamma(ax[1], k_dist, label=row["dataset"], color=colors[idx % len(colors)])

    fig.suptitle("Reconstructed Gamma Marginals")
    print(f"Marginal plots saved to {filename}")
    plt.savefig(filename)

def plot_gamma(ax, dist, label=None, color=None, n=600):
    """
    Plot a gamma PDF (and optionally a KDE of sample data) on given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    dist : scipy.stats.gamma
        Fitted gamma distribution (from reconstruct_gamma()).
    data : array-like, optional
        Raw data to overlay as KDE.
    label : str, optional
        Label for the fitted curve.
    color : str, optional
        Matplotlib color for the curve.
    n : int, optional
        Number of points for the smooth PDF curve.
    kde : bool, optional
        Whether to plot KDE for provided data.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Updated axis with plot.
    """
    # PDF curve
    x = np.linspace(dist.ppf(1e-5), dist.ppf(1 - 1e-5), n)
    y = dist.pdf(x)
    ax.plot(x, y, label=label or "Fit Gamma", color=color, linewidth=2)

    ax.legend()
    ax.grid(alpha=0.3)
    return ax


if __name__ == "__main__":
    main()