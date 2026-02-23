"""
This was the sandbox for me to test out any ideas quickly. It turned into the 
conglomerate of plotting scripts necessary for the paper.
"""

# import os
# os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
# from filter_csv import filter_data
# import local_config
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Computer Modern Roman']

# mpl.rcParams['text.latex.preamble'] = r"""
# \usepackage{amsmath}
# \usepackage{amssymb}
# \usepackage{xcolor}
# """

# mpl.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({
#     'font.size': 10,      # match normalsize
#     'axes.labelsize': 10,
#     'axes.titlesize': 10,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'legend.fontsize': 8, # slightly smaller like LaTeX
#     'text.usetex': True,
# })

##################################################################### Uniaxial tension as fn of theta

# # ========== USER INTERFACE ==========
# folder = f'{local_config.DATA_DIR}/rotation_tests'
# csv_file = f"{folder}/all_simulations.csv"

# exact_filters = {
#     "Num Atoms x": 60,
#     "Num Atoms y": 60,
#     "Defects": "None",  # will match NaN or "None"
#     # "Defect Percentage": 0.5,
#     # "Defect Random Seed": 3,
#     # "Theta Requested": 0
#     # "Strain Rate x": 0.001,
#     # "Strain Rate y": 0.001
# }

# range_filters = {
#     # "Defect Percentage": (0.4, 0.6),
#     # "Defect Random Seed": (0, 10)
#     # "Theta Requested": (90, 90),
#     # "Sigma_1": (4, 20)
# }

# or_filters = {
#     # "Defect Type": ["SV", "DV"],
#     # "Theta Requested": [0, 60]
# }
# # ====================================
# df = pd.read_csv(csv_file)
# filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, flip_strengths=False, shift_theta=False)

# optimal_rows = []
# for angle in range(0, 91, 1):
#     this_df = filter_data(filtered_df, exact_filters={"Theta Requested": angle}, shift_theta=False)
    
#     if this_df.empty:
#         continue  # skip if no data for this angle

#     this_df = this_df.copy()
#     this_df["ratio"] = this_df["Strength_2"] / this_df["Strength_1"]

#     # Find the row with the minimum ratio
#     min_row = this_df.loc[this_df["ratio"].idxmin()]

#     # Append the row to the list
#     optimal_rows.append(min_row)

# # Combine all optimal rows into a single DataFrame
# unscaled_df = pd.DataFrame(optimal_rows)

# final_df = unscaled_df.copy()
# final_df["Theta"] = final_df["Theta"] - final_df["Rotation Angle"]  # apply deformation rotation fix

# fig, ax = plt.subplots(figsize=(3.56, 3))

# # plt.scatter(result["Theta Requested"], result["Strength_1"], color='blue', alpha=0.5)
# ax.scatter(final_df["Theta"], final_df["Strength_1"], color='k', alpha=0.8, label="scaled", s=10)
# # plt.scatter(unscaled_df["Theta"], final_df["Strength_1"], color='grey', alpha=0.5, label="unscaled")
# # plt.plot([25, 25], [100, 125], lw=2, label="25°")
# # plt.plot([30, 30], [100, 125], lw=2, label="30°")
# ticks = np.arange(0, 91, 30)
# ax.set_xticks(ticks)
# ax.set_xlabel("Loading Angle (°)")
# # plt.legend()
# ax.set_ylabel("Uniaxial Tensile Strength (GPa)")
# # plt.title("Loading angle with/without lattice rotation")
# ax.grid()
# fig.savefig('aniso_uniaxial.pdf', bbox_inches='tight')

###########################################################################################################################

##################################################################### Regularization Parameter
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# def to_power_of_ten_str(x):
#     mapping = {
#         "0": 0,
#         "1e-2": 0.01,
#         "1e-1": 0.1,
#         "1e0": 1,
#         "1e1": 10,
#         "1e2": 100
#     }
#     return mapping.get(x, None)

# def plot_datapoint(df, ax, lab=None, marker='o', color='k'):
#     x = np.mean(df["norm_z"])
#     y = np.mean(df["NRMSE"])
#     lab = rf"$\lambda = {to_power_of_ten_str(lab)}$"
#     ax.scatter(x, y, label=lab, marker=marker, s=30, c=color)
#     return x, y

# def create_large_df(fo, lam, strict_zero=False):
#     def get_csv_format(defs):
#         if strict_zero is True:
#             return f"z_{defs}{fo}_reg{lam}.csv"
#         else:
#             return f"z_{defs}{fo}_reg1e{lam}.csv"
    
#     sv = pd.read_csv(get_csv_format("sv"))
#     sv.insert(loc=0, column='Defect Type', value=["SV"]*len(sv))
#     dv = pd.read_csv(get_csv_format("dv"))
#     dv.insert(loc=0, column='Defect Type', value=["DV"]*len(dv))
#     mx = pd.read_csv(get_csv_format("mx"))
#     mx.insert(loc=0, column='Defect Type', value=["MX"]*len(mx))
#     large_df = pd.concat([sv, dv, mx])
#     return large_df

# # fourier_order = 3
# all_rows = []
# fig, ax = plt.subplots(figsize=(3.11, 2.62))
# plot_datapoint(pd.read_csv(f"z_dv4_reg0.csv"), ax, lab=f"0", marker='s')
# markers = ['+', 'x', '*', 'o', 'v']
# for lam in range(-2, 3):
#     large_df = create_large_df(4, lam)
#     norm_z, nrmse = plot_datapoint(large_df, ax, lab=f"1e{lam}", marker=markers[lam+2], color='k')
#     all_rows.append({"Order": 4, "lambda": f"{(10**lam):g}", "NRMSE": nrmse, "norm_z": norm_z})
# summary_df = pd.DataFrame(all_rows).sort_values(["Order", "lambda"])
# pd.set_option("display.float_format", lambda x: f"{x:0.5f}")
# # print(summary_df)
        

# # ax.legend()
# ax.set_xlabel(r"Mean $||\mathbf{z}||$")
# ax.set_ylabel("Mean NRMSE")
# ax.legend()
# ax.grid()
# plt.savefig('lambda_conv.pdf', bbox_inches='tight')

###########################################################################################################################

##################################################################### Fourier Order

# import pandas as pd
# import numpy as np
# from scipy.stats import gaussian_kde
# import local_config
# from filter_csv import filter_data
# import matplotlib.pyplot as plt

# def get_mean_val():
#     df = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
#     or_filters = {"Defects": ['{"SV": 0.5}', '{"DV": 0.5}', '{"SV": 0.25, "DV": 0.25}']}
#     filtered_df = filter_data(df, or_filters=or_filters, flip_strengths=False, duplic_freq=(0, 91, 10), remove_dupes=True)
#     return np.mean(filtered_df["Strength_1"].to_numpy())

# def create_large_df(n):
#     def get_csv_format(defs, num):
#         return f"z_{defs}{num}_reg1e-1.csv"
    
#     sv = pd.read_csv(get_csv_format("sv", n))
#     sv.insert(loc=0, column='Defect Type', value=["SV"]*len(sv))
#     dv = pd.read_csv(get_csv_format("dv", n))
#     dv.insert(loc=0, column='Defect Type', value=["DV"]*len(dv))
#     mx = pd.read_csv(get_csv_format("mx", n))
#     mx.insert(loc=0, column='Defect Type', value=["MX"]*len(mx))
#     large_df = pd.concat([sv, dv, mx])
#     return large_df

# def plot_rmse_kde(ax, large_df, n):
#     nrmses = large_df["NRMSE"]
#     kde = gaussian_kde(nrmses)
#     x = np.linspace(min(nrmses) - np.mean(nrmses)/5, max(nrmses) + np.mean(nrmses)/8, 500)
#     density = kde(x)
#     N = r"$n_{\text{FO}}$"
#     if n == 4:
#         ax.plot(x, density, label=f'{N}={n}', lw=0.75, c='k')
#     else:
#         ax.plot(x, density, label=f'{N}={n}', lw=0.75)
#     mean_nrmse = np.mean(nrmses)
#     upper_nrmse = np.percentile(nrmses, 97.5)
#     variation = np.std(nrmses)/mean_nrmse
#     print(f"N = {n}: mean={mean_nrmse}, upper 95th={upper_nrmse}")
#     return mean_nrmse, upper_nrmse, variation


# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.11, 2.62))

# mean_strengths = get_mean_val()
# for i in range(6):
#     df = create_large_df(i+1)
#     # df = pd.read_csv(f"z_dv{i+1}_reg1e-1.csv")
#     mean, upper, variation = plot_rmse_kde(ax, df, i+1)

# ax.set_xlabel('NRMSE')
# ax.set_ylabel('Density')
# ax.legend(loc='upper right')
# ax.set_xlim([0.02, 0.08])

# plt.tight_layout()
# ax.grid()
# plt.savefig('fo_conv.pdf', bbox_inches='tight')

###########################################################################################################

##################################################################### parametric and strength functions (for big plot)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# import local_config
# from filter_csv import filter_data


# def set_lower_legend(fig, axs, y_loc=-0.02, m_first=False):
#     handles = []
#     labels = []
#     for ax in axs.flatten():
#         h, l = ax.get_legend_handles_labels()
#         handles.extend(h)
#         labels.extend(l)
#     unique = dict(zip(labels, handles))

#     if m_first:
#         # Find the index of the label that starts with 'M'
#         m_idx = next(i for i, lbl in enumerate(labels) if lbl.startswith("M"))

#         # Reorder so that entry comes first
#         new_order = [m_idx] + [i for i in range(len(labels)) if i != m_idx]

#         handles = [handles[i] for i in new_order]
#         labels  = [labels[i]  for i in new_order]
#         unique = dict(zip(labels, handles))
#     leg = fig.legend(
#         unique.values(),
#         unique.keys(),
#         loc='lower center',
#         bbox_to_anchor=(0.5, y_loc),
#         ncol=len(unique)     # all labels on one line
#     )
#     return leg

# def get_alpha_k(params, theta, return_k=True, periodic=False):
#     if periodic:
#         omega = 2 * np.pi * theta / 60
#     else:
#         omega = 2 * np.pi * theta / 180
    
#     # infer N from length of data
#     N = int((len(params) - 2) / 4)

#     z_alpha = params[0]
#     z_k = params[2*N+1]

#     for m in range(1, N + 1):
#         cos_coeff_a = params[2 * m - 1]
#         sin_coeff_a = params[2 * m]
#         z_alpha += cos_coeff_a * np.cos(m * omega) + sin_coeff_a * np.sin(m * omega)
#         if return_k:
#             cos_coeff_k = params[(2*N+1)+(2 * m - 1)]
#             sin_coeff_k = params[(2*N+1)+(2 * m)]
#             z_k += cos_coeff_k * np.cos(m * omega) + sin_coeff_k * np.sin(m * omega)
    
#     # once we have the value of z_alpha and z_k, we must transform back to alpha and k:
#     def softplus(z):
#         return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

#     alpha = -np.sqrt(3) / 6 + softplus(z_alpha)
#     if return_k:
#         k = softplus(z_k)
#         return alpha, k
#     else:
#         return alpha

# def plot_dp(params, index, real_data=None, current_theta=0, ax=None, color='b'):
#     z_coeffs = params[index]

#     theta_buffer = 4

#     current_alpha, current_k = get_alpha_k(z_coeffs, current_theta)

#     min_strength = -5
#     max_strength = 120

#     grid = np.linspace(min_strength, max_strength, 600)
#     sig1, sig2 = np.meshgrid(grid, grid)
#     sig3 = np.zeros_like(sig1)
#     i1 = sig1 + sig2 + sig3
#     j2 = (sig1**2 + sig2**2 + sig3**2 - sig1*sig2 - sig2*sig3 - sig3*sig1) / 3.0

#     F = np.sqrt(j2) + current_alpha * i1 - current_k
#     if ax is None:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.83, 5.83))


#     ax.contour(sig1, sig2, F, levels=[0], linewidths=2, colors=color)  # F=0 curve
#     ax.plot([], [], color=color, label=rf'$\theta={current_theta}$\textdegree')  # for legend
#     ax.axhline(0, color='black', linewidth=1)
#     ax.axvline(0, color='black', linewidth=1)
#     ax.set_xlim(min_strength, max_strength); ax.set_ylim(min_strength, max_strength)
#     ax.set_xlabel(r"$\sigma_1$ (GPa)")
#     ax.set_ylabel(r"$\sigma_2$ (GPa)")
#     ax.set_title(rf"$\theta={current_theta:.0f}$\textdegree{{}} $(\alpha={current_alpha:.2f}, k={current_k:.1f})$")
#     ticks = [0, 30, 60, 90, 120]
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)
#     if real_data is not None:
#         df = filter_data(real_data, range_filters={"Theta": (current_theta-theta_buffer, current_theta+theta_buffer)}, shift_theta=False)
#         try:
#             ax.scatter(df["Strength_1"], df["Strength_2"], color='grey', alpha=(abs(df["Theta"]-current_theta) - theta_buffer) / -theta_buffer, label=rf'MD Data +/- {theta_buffer}\textdegree', s=15)
#         except ValueError:
#             ax.scatter([], [], color="grey", label=rf'MD Data +/- {theta_buffer}\textdegree')

#     # ax.legend()
#     ax.grid()
#     if ax is None:
#         fig.tight_layout(pad=0.4)
#         # fig.subplots_adjust(left=0.1, right=0.98, bottom=0.22, top=0.90)
#         fig.savefig('sample_dps.pdf')

# def plot_ak(params, index, real_data=None):
#     z_coeffs = params[index]

#     ticks = np.arange(0, 91, 30)

#     thetas = np.linspace(0, 90, 200)

#     fig, axs = plt.subplots(1, 2, figsize=(5.83, 2.6))

#     alpha_fn, k_fn = get_alpha_k(z_coeffs, thetas)
#     axs[0].plot(thetas, alpha_fn, color='k', alpha=0.8, label='Parameter')
#     axs[0].axvline(0, lw=0.9, linestyle='dashed', c='r', label=r"$\theta=0$\textdegree")
#     axs[0].axvline(15, lw=0.9, linestyle='dashed', c='orange', label=r"$\theta=15$\textdegree")
#     axs[0].axvline(30, lw=0.9, linestyle='dashed', c='green', label=r"$\theta=30$\textdegree")
#     axs[0].axvline(90, lw=0.9, linestyle='dashed', c='blue', label=r"$\theta=90$\textdegree")
#     axs[0].set_xticks(ticks)

#     axs[1].plot(thetas, k_fn, color='k', alpha=0.8, label='Parameter')
#     axs[1].axvline(0, lw=0.9, linestyle='dashed', c='r', label=r"$\theta=0$\textdegree")
#     axs[1].axvline(15, lw=0.9, linestyle='dashed', c='orange', label=r"$\theta=15$\textdegree")
#     axs[1].axvline(30, lw=0.9, linestyle='dashed', c='green', label=r"$\theta=30$\textdegree")
#     axs[1].axvline(90, lw=0.9, linestyle='dashed', c='blue', label=r"$\theta=90$\textdegree")
#     axs[1].set_xticks(ticks)

#     axs[0].set_xlabel(r"$\theta$ (\textdegree)")
#     axs[0].set_ylabel(r"$\alpha$")
#     axs[0].grid(alpha=0.3)
#     axs[0].set_ylim(-0.2, 0.4)
    
#     axs[1].set_xlabel(r"$\theta$ (\textdegree)")
#     axs[1].set_ylabel(r"$k$")
#     axs[1].grid(alpha=0.3)
#     axs[1].set_ylim(20, 80)

#     plot_theta_kde(axs[0], real_data)
#     plot_theta_kde(axs[1], real_data)

#     set_lower_legend(fig, axs, y_loc=0)
#     fig.tight_layout(pad=0.4)
#     fig.subplots_adjust(left=0.1, right=0.98, bottom=0.26, top=0.98)
#     fig.savefig('ak_functions.pdf')

# def plot_theta_kde(ax, df):
#     # compute KDE over 0–90 range
#     theta_vals = np.linspace(0, 90, 200)
#     theta_data = df["Theta"].dropna()
#     kde = gaussian_kde(theta_data)
#     kde.set_bandwidth(bw_method=kde.factor / 25)
#     density = kde(theta_vals)

#     # normalize the density so it fits nicely at the bottom (say bottom 20%)
#     ymin, ymax = ax.get_ylim()
#     scale = 0.4 * (ymax - ymin)
#     density_scaled = ymin + density / np.max(density) * scale

#     # plot the KDE line at the bottom of the alpha plot
#     ax.fill_between(theta_vals, ymin, density_scaled, color="grey", alpha=0.6)
#     ax.plot(theta_vals, density_scaled, color="grey", linewidth=1, alpha=0.8, label=r"MD Density")

# df = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")

# N = 4
# defs = "SV"

# exact_filters = {"Defects": f'{{"{defs}": 0.5}}', "Defect Random Seed": 94}


# filtered_df = filter_data(df, exact_filters=exact_filters, flip_strengths=True, duplic_freq=(0, 91, 10), remove_dupes=True)

# df_params = pd.read_csv(f"z_{defs.lower()}{N}_reg1e-1.csv")
# keep = [col for col in df_params.columns if col.startswith("z")]
# zs = df_params[keep].copy().to_numpy()


# plot_ak(params=zs, real_data=filtered_df, index=94)

# colors = ['r', 'orange', 'green', 'b']
# thetas = [0, 15, 30, 90]
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.18, 5.18))
# for ax, color, theta in zip(axs.flatten(), colors, thetas):
#     plot_dp(params=zs, real_data=filtered_df, index=94, current_theta=theta, ax=ax, color=color)


# leg = set_lower_legend(fig, axs, y_loc=0, m_first=True)

# fig.tight_layout(pad=0.4)
# fig.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)
# fig.savefig('sample_dps.pdf')

#####################################################################################################################################

##################################################################### PCA reconstruction

# import local_config
# from filter_csv import filter_data
# from sklearn.decomposition import PCA
# from probabilistic_results import get_alpha_k, plot_strength_surface, set_lower_legend

# fo = 4
# defs = "sv"

# df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")

# zs = df.drop(columns=["Defect Random Seed", "Total Loss", "NRMSE", "norm_z"]).to_numpy()

# mddf = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
# filtered_df = filter_data(mddf, exact_filters={"Defects": '{"SV": 0.5}', "Theta Requested": 90}, flip_strengths=True, duplic_freq=(0, 91, 10), remove_dupes=True)

# test_components = [10, 11, 14, 18]
# thresholds = ["10^{-3}", "10^{-4}", "10^{-5}", "0"]

# fig, axs = plt.subplots(2, 2, figsize=(5.18, 5.18))
# for ax, comp, thresh in zip(axs.flatten(), test_components, thresholds):
#     pca = PCA(n_components=comp)
#     z_pca = pca.fit_transform(zs)

#     z_back = pca.inverse_transform(z_pca)
#     ax.axhline(0, color='black', linewidth=1)
#     ax.axvline(0, color='black', linewidth=1)
#     for i in range(len(z_back)):
#         a, k = get_alpha_k(z_back[i], 90, periodic=False)
#         plot_strength_surface(ax, a, k, color='k', min_strength=-5, max_strength=120)
#         # a, k = get_alpha_k(zs[i], 90, periodic=False)
#         # plot_strength_surface(ax, a, k, color='k')
#     ax.scatter(filtered_df["Strength_1"], filtered_df["Strength_2"], c='r', alpha=0.1, s=5, label="MD Data")
#     ax.set_title(rf"$E_{{\text{{thresh}}}}={thresh} \;\; (d={comp})$")
#     ax.grid()
#     ax.set_xlabel(r"$\sigma_1$ (GPa)")
#     ax.set_ylabel(r"$\sigma_2$ (GPa)")
#     ax.plot([], [], c='k', label="Recovered Surfaces")
#     ticks = [0, 30, 60, 90, 120]
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)
#     if comp == 10 or comp == 11:
#         ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#         ax.set_xlabel("")

#     if comp == 11 or comp == 18:
#         ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#         ax.set_ylabel("")

# leg = set_lower_legend(fig, axs, y_loc=0)
# for handle in leg.legend_handles:
#     handle.set_alpha(0.8)
# fig.tight_layout(pad=0.4)
# fig.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)
# fig.savefig('pca_reconstruction.pdf')

#####################################################################################################################################

##################################################################### GMM

# import local_config
# from filter_csv import filter_data
# from sklearn.decomposition import PCA
# from probabilistic_results import get_alpha_k, plot_strength_surface, set_lower_legend
# from sklearn.mixture import GaussianMixture

# fo = 4
# defs = "dv"

# df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")

# zs = df.drop(columns=["Defect Random Seed", "Total Loss", "NRMSE", "norm_z"]).to_numpy()

# mddf = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
# filtered_df = filter_data(mddf, exact_filters={"Defects": '{"DV": 0.5}', "Theta Requested": 90}, flip_strengths=True, duplic_freq=(0, 91, 10), remove_dupes=True)

# n_components = 14
# pca = PCA(n_components=n_components)
# z_pca = pca.fit_transform(zs)

# n_modes = [1, 2, 3, 4]

# fig, axs = plt.subplots(2, 2, figsize=(5.18, 5.18))
# for mode in n_modes:
#     gmm = GaussianMixture(n_components=mode, covariance_type="full", reg_covar=1e-4, random_state=42)
#     gmm.fit(z_pca)
#     samples_pca, _ = gmm.sample(n_samples=200)
#     # inverse PCA transform (back to standardized space)
#     samples = pca.inverse_transform(samples_pca)

#     axs.flatten()[mode-1].axhline(0, color='black', linewidth=1)
#     axs.flatten()[mode-1].axvline(0, color='black', linewidth=1)
#     color = 'k'
#     if mode == 4:
#         samples = zs
#         color = 'b'
#     for i in range(len(samples)):
#         a, k = get_alpha_k(samples[i], 90, periodic=False)
#         plot_strength_surface(axs.flatten()[mode-1], a, k, color=color, min_strength=-5, max_strength=120)
#     axs.flatten()[mode-1].scatter(filtered_df["Strength_1"], filtered_df["Strength_2"], c='r', alpha=0.1, s=5, label="MD Data")
#     if mode == 4:
#         axs.flatten()[mode-1].set_title(rf"Original Surface Fits")
#     else:
#         axs.flatten()[mode-1].set_title(rf"$M={mode}$")
#     axs.flatten()[mode-1].grid()
#     axs.flatten()[mode-1].set_xlabel(r"$\sigma_1$ (GPa)")
#     axs.flatten()[mode-1].set_ylabel(r"$\sigma_2$ (GPa)")
#     axs.flatten()[mode-1].plot([], [], c='k', label="Sampled Surfaces")
#     axs.flatten()[mode-1].plot([], [], c='b', label="Original Surfaces")
#     ticks = [0, 30, 60, 90, 120]
#     axs.flatten()[mode-1].set_xticks(ticks)
#     axs.flatten()[mode-1].set_yticks(ticks)
#     if mode == 1 or mode == 2:
#         axs.flatten()[mode-1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#         axs.flatten()[mode-1].set_xlabel("")

#     if mode == 2 or mode == 4:
#         axs.flatten()[mode-1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#         axs.flatten()[mode-1].set_ylabel("")

# leg = set_lower_legend(fig, axs, y_loc=0)
# for handle in leg.legend_handles:
#     handle.set_alpha(0.8)
# fig.tight_layout(pad=0.4)
# fig.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)
# fig.savefig('gmm.pdf')

#####################################################################################################################################

##################################################################### Outliers


# from filter_csv import filter_data
# import local_config
# from probabilistic_results import get_alpha_k, plot_strength_surface, set_lower_legend

# fo = 4
# defs = "dv"
# df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")
# zs = df.drop(columns=["Defect Random Seed", "Total Loss", "NRMSE", "norm_z"]).to_numpy()

# mddf = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
# filtered_df = filter_data(mddf, exact_filters={"Defects": '{"DV": 0.5}', "Theta Requested": 90}, 
#                           or_filters={"Defect Random Seed": [54, 74]}, flip_strengths=True, duplic_freq=(0, 91, 10), remove_dupes=True)

# fig, ax = plt.subplots(figsize=(3.11, 3.11))
# ax.axhline(0, color='black', linewidth=1)
# ax.axvline(0, color='black', linewidth=1)
# a, k = get_alpha_k(zs[54], 90, periodic=False)
# plot_strength_surface(ax, a, k, color='k', alpha=1, min_strength=-5, max_strength=120, label='Surface Fit')
# # a, k = get_alpha_k(zs[74], 90, periodic=False)
# # plot_strength_surface(ax, a, k, color='g', alpha=1, min_strength=-5, max_strength=120)
# # ax.scatter(filtered_df["Strength_1"], filtered_df["Strength_2"], c='r', alpha=0.1)
# outliers54 = filtered_df[(filtered_df["Defect Random Seed"] == 54)]
# # outliers74 = filtered_df[(filtered_df["Defect Random Seed"] == 74)]
# ax.scatter(outliers54["Strength_1"], outliers54["Strength_2"], c='k', alpha=0.8, s=15, label='MD Data')
# # ax.scatter(outliers74["Strength_1"], outliers74["Strength_2"], c='g', alpha=0.7)

# # ax.set_title(f"")
# ax.grid()
# ax.set_xlabel(r"$\sigma_1$ (GPa)")
# ax.set_ylabel(r"$\sigma_2$ (GPa)")
# ax.legend()
# fig.tight_layout()

# fig.savefig('outliers.pdf')


#####################################################################################################################################

##################################################################### 3D Strength Surface

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# from probabilistic_results import get_alpha_k

# def plot_3d_strength_surface(zs, data_df, resolution=100, theta_max=90, f_tol=0.5, ax=None):

#     # Create axes if none provided
#     if ax is None:
#         fig = plt.figure(figsize=(3.11, 3.11))
#         ax = fig.add_subplot(111, projection="3d")
#     else:
#         fig = ax.get_figure()

#     # ----------------------------
#     # 1. Build grid in σ1–σ2–θ
#     # ----------------------------
#     sig_vals = np.linspace(0, data_df["Strength_1"].max(), resolution)
#     theta_vals = np.linspace(0, theta_max, resolution)

#     sig1, sig2, theta = np.meshgrid(sig_vals, sig_vals, theta_vals, indexing="ij")
#     sig3 = np.zeros_like(sig1)

#     # Invariants
#     I1 = sig1 + sig2 + sig3
#     mean = I1 / 3
#     dev_x = sig1 - mean
#     dev_y = sig2 - mean
#     dev_z = sig3 - mean
#     J2 = 0.5 * (dev_x**2 + dev_y**2 + dev_z**2)

#     # ----------------------------
#     # 2. Compute α(θ) and k(θ)
#     # ----------------------------
#     alpha, k = get_alpha_k(zs[0], theta)

#     # Full DP function
#     F = np.sqrt(J2) + alpha * I1 - k

#     # Mask region where DP surface is close to zero
#     mask = np.abs(F) < f_tol
#     xs = sig1[mask]
#     ys = sig2[mask]
#     zs = theta[mask]

#     # ----------------------------
#     # 3. Plot DP surface points
#     # ----------------------------
#     ax.scatter(xs, ys, zs, s=4, c=np.maximum(xs, ys), cmap="Blues", alpha=0.15)

#     # ----------------------------
#     # 4. Plot the data points (optional)
#     # ----------------------------
#     if data_df is not None:
#         ax.scatter(
#             data_df["Strength_1"], data_df["Strength_2"], data_df["Theta"],
#             color="black", s=10, alpha=0.8, label="data"
#         )

#     # ----------------------------
#     # 5. Formatting
#     # ----------------------------
#     ax.set_xlim([-5, 130])
#     ax.set_ylim([-5, 130])
#     ax.set_zlim([0, 90])

#     ax.set_xlabel(r"$\sigma_1$ (GPa)", labelpad=0)
#     ax.set_ylabel(r"$\sigma_2$ (GPa)", labelpad=0)
#     ax.set_zlabel(r"$\theta$ (\textdegree)", labelpad=-2)

#     ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
#     ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
#     ax.set_zticks([0, 30, 60, 90])

#     enable_3d_grid(ax)

#     # Camera angle similar to Plotly (diagonal view)
#     ax.view_init(elev=20, azim=55, roll=0)  # 60, 20, 110


#     ax.grid(False)

#     return fig, ax

# def enable_3d_grid(ax, color="lightgray", lw=0.5):
#     # extract current tick positions
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     zticks = ax.get_zticks()

#     xmin, xmax = ax.get_xlim()
#     ymin, ymax = ax.get_ylim()
#     zmin, zmax = ax.get_zlim()

#     # -------------- X–Y bottom plane (z = zmin) --------------
#     for x in xticks:
#         ax.plot([x, x], [ymin, ymax], [zmin, zmin], color=color, lw=lw)
#     for y in yticks:
#         ax.plot([xmin, xmax], [y, y], [zmin, zmin], color=color, lw=lw)

#     # -------------- X–Z side plane (y = ymin) --------------
#     for x in xticks:
#         ax.plot([x, x], [ymin, ymin], [zmin, zmax], color=color, lw=lw)
#     for z in zticks:
#         ax.plot([xmin, xmax], [ymin, ymin], [z, z], color=color, lw=lw)

#     # -------------- Y–Z back plane (x = xmax) --------------
#     for y in yticks:
#         ax.plot([xmin, xmin], [y, y], [zmin, zmax], color=color, lw=lw)
#     for z in zticks:
#         ax.plot([xmin, xmin], [ymin, ymax], [z, z], color=color, lw=lw)

# fo = 4
# defs = "none"
# df = pd.read_csv(f"z_{defs}{fo}_reg1e-1.csv")
# zs = df.drop(columns=["Total Loss", "NRMSE", "norm_z"]).to_numpy()

# mddf = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
# filtered_df = filter_data(mddf, exact_filters={"Defects": 'None'}, or_filters={"Theta Requested": [0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90]}, 
#                           flip_strengths=True, duplic_freq=(0, 91, 10), remove_dupes=True)

# fig, ax = plot_3d_strength_surface(zs, filtered_df)
# plt.tight_layout(pad=0.4)
# fig.subplots_adjust(left=0.15, right=0.98, bottom=0.05, top=0.98)
# fig.savefig('3d_example.pdf')


#########################################################################################################################


from deform_graphene import Relaxation
from mpi4py import MPI
import local_config

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Relaxation(comm, rank, f"{local_config.DATA_DIR}/data_files/data.20_20", 20, 20, sim_length=100000)