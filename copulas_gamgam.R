# This does all of the copulas stuff for alpha and k in R.
# NOTE: MUST CLEAN THIS FILE UP SIGNIFICANTLY ONCE SIMS ARE DONE

library(readr)
library(fitdistrplus)
library(copula)
library(ggplot2)

dataset <- "ZZ_DV"
csvname <- paste0("DPparams_", dataset)

# Load data
data <- read_csv(paste0(csvname, ".csv"))
alpha <- data$alpha
k <- data$k

# Define lower bound for alpha
a0 <- -sqrt(3) / 6  # shift point
# a0 <- -0.1

alpha_shift <- alpha - a0
if (any(alpha_shift < 0)) {
  stop("Some alpha values are below the lower bound a0. Check data or a0.")
}

# Fit marginals - these use maximum likelihood estimation
fit_alpha_gamma <- fitdist(alpha_shift, "gamma", method = "mle")
fit_k_gamma <- fitdist(k, "gamma", method = "mle")

# Convert to pseudo-observations
# Uniforms for copula fit
u1 <- pgamma(alpha_shift,
             shape = fit_alpha_gamma$estimate["shape"],
             rate  = fit_alpha_gamma$estimate["rate"])
u2 <- pgamma(k,
             shape = fit_k_gamma$estimate["shape"],
             rate  = fit_k_gamma$estimate["rate"])

# u is a 2D array of points in [0, 1]^2 to use for copula fitting
# u <- cbind(u1, u2)
eps <- 1e-12
u_fit <- cbind(pmin(pmax(u1, eps), 1 - eps),
               pmin(pmax(u2, eps), 1 - eps))

# Fit Gaussian copula
cop <- normalCopula(dim = 2)
fit_cop <- fitCopula(cop, u_fit, method = "ml")  # uses MLE
cop_fitted <- fit_cop@copula  # extract best fit

# Sample from copula
set.seed(123)
u_sampled <- rCopula(100000, cop_fitted)

# must shift alpha back after sampling
alpha_samp <- a0 + qgamma(u_sampled[, 1],
                          shape = fit_alpha_gamma$estimate["shape"],
                          rate  = fit_alpha_gamma$estimate["rate"])
k_samp <- qgamma(u_sampled[, 2],
                 shape = fit_k_gamma$estimate["shape"],
                 rate  = fit_k_gamma$estimate["rate"])

# Combine original and sampled data
original_df <- data.frame(alpha = alpha, k = k, source = "original")
sampled_df <- data.frame(alpha = alpha_samp, k = k_samp, source = "sampled")
plot_df <- rbind(original_df, sampled_df)

# turn alphas and ks into sigma_ts and sigma_bs (for visualization)
bs <- (3 * k) / (sqrt(3) + 6 * alpha)
ts <- (3 * k) / (sqrt(3) + 3 * alpha)
str_df <- data.frame(sigma_ts = ts, sigma_bs = bs, source = "Sampled Strength")

# Plot strength scatter
ggplot(str_df, aes(x = sigma_ts, y = sigma_bs, color = source)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Sampled Strengths in Stress Space",
    x = "Uniaxial Tensile Strength", y = "Biaxial Tensile Strength"
  ) +
  coord_cartesian(xlim = c(min(ts) - 10, max(ts) + 10),
                  ylim = c(min(bs) - 10, max(bs) + 10)) +
  theme_bw(base_size = 22)

# Plot joint scatter (original and sampled)
ggplot(plot_df, aes(x = alpha, y = k, color = source)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Original vs. Sampled Joint Distribution",
    x = expression(alpha), y = "k"
  ) +
  theme_minimal()


# ====== Full PDF in alpha/k space ====== #
# Generate density grid over domain
alpha_vals <- seq(a0 - 0.1, max(alpha) + 0.1, length.out = 1000)
k_vals <- seq(-5, max(k) + 10, length.out = 1000)
grid <- expand.grid(alpha = alpha_vals, k = k_vals)

# Marginal CDFs on the grid
# transform grid into pseudo-observations
alpha_shift_grid <- grid$alpha - a0
u1_grid <- pgamma(alpha_shift_grid,
                  shape = fit_alpha_gamma$estimate["shape"],
                  rate  = fit_alpha_gamma$estimate["rate"])
u2_grid <- pgamma(grid$k,
                  shape = fit_k_gamma$estimate["shape"],
                  rate  = fit_k_gamma$estimate["rate"])

# Evaluate marginal PDFs at each gridpoint
f_alpha <- dgamma(alpha_shift_grid,
                  shape = fit_alpha_gamma$estimate["shape"],
                  rate  = fit_alpha_gamma$estimate["rate"])
f_alpha[alpha_shift_grid < 0] <- 0  # outside support

f_k <- dgamma(grid$k,
              shape = fit_k_gamma$estimate["shape"],
              rate  = fit_k_gamma$estimate["rate"])

# Joint PDF via copula - evaluate copula density:
# multiply the copula with the marginals to get the true joint PDF
c_uv <- dCopula(cbind(u1_grid, u2_grid), copula = cop_fitted)
grid$density <- c_uv * f_alpha * f_k

# Plot joint PDF
ggplot(grid, aes(x = alpha, y = k, fill = density)) +
  geom_tile() +
  geom_point(data = original_df, aes(x = alpha, y = k),
             color = "red", size = 2, alpha = 0.5, inherit.aes = FALSE) +
  scale_fill_viridis_c(option = "mako") +
  labs(title = paste("Empirical Joint PDF with Original Data:", dataset), x = "α", y = "k") +
  coord_cartesian(xlim = c(min(alpha) - 0.05, 0.5), ylim = c(min(k) - 2, 85)) +
  theme_bw(base_size = 22) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = c(0.82, 0.32),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "white"),
    legend.key = element_rect(fill = NA)
  )

# # Joint PDF with zero-density contour (alpha k space)
# ggplot(grid, aes(x = alpha, y = k)) +
#   geom_tile(aes(fill = density)) +
#   geom_point(data = original_df, aes(x = alpha, y = k),
#              color = "white", size = 0.5, alpha = 0.35, inherit.aes = FALSE) +
#   stat_contour(
#     aes(z = density), breaks = c(1e-15),
#     color = "red", linewidth = 1
#   ) +
#   scale_fill_viridis_c() +
#   labs(
#     title = "Joint PDF with Zero-Density Contour",
#     fill = "Density"
#   ) +
#   coord_cartesian(xlim = c(-0.25, 0.25), ylim = c(min(k) - 5, max(k) + 5)) +
#   theme_bw()

# ====== Marginal Distribution of Alpha ====== #
alpha_plot <- seq(min(alpha) - 0.05, max(alpha) + 0.05, length.out = 600)

pdf_alpha <- dgamma(alpha_plot - a0,
                    shape = fit_alpha_gamma$estimate["shape"],
                    rate  = fit_alpha_gamma$estimate["rate"])
pdf_alpha[alpha_plot < a0] <- 0

analytical_alpha_df <- data.frame(alpha = alpha_plot, pdf = pdf_alpha)

# Plot marginal density of alpha
# KDE vs analytical PDF
ggplot() +
  # KDE from your sample alphas
  stat_density(data = original_df,
               aes(x = alpha, y = after_stat(density), color = "KDE of Data"),
               geom = "line", size = 1) +
  # Analytical shifted-Gamma PDF
  geom_line(data = analytical_alpha_df,
            aes(x = alpha, y = pdf, color = "Fit Gamma"),
            size = 1) +
  scale_color_manual(values = c("KDE of Data" = "skyblue",
                                "Fit Gamma" = "navyblue")) +
  labs(title = paste("Empirical Marginal PDF vs. KDE of Data for α:", dataset),
       x = expression(alpha), y = "Density", color = NULL) +
  coord_cartesian(xlim = c(min(alpha_plot), max(alpha_plot))) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.65, 0.99),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = NA),
        legend.key = element_rect(fill = NA)) +
  guides(colour = guide_legend(override.aes = list(size = 3)))


# ====== Marginal Distribution of k ====== #

k_plot <- seq(0, max(k, na.rm = TRUE) * 1.2, length.out = 600)
pdf_k <- dgamma(k_plot,
                shape = fit_k_gamma$estimate["shape"],
                rate  = fit_k_gamma$estimate["rate"])
analytical_k_df <- data.frame(k = k_plot, pdf = pdf_k)

ggplot() +
  stat_density(data = original_df,
               aes(x = k, y = after_stat(density), color = "KDE of Data"),
               geom = "line", size = 1) +
  geom_line(data = analytical_k_df,
            aes(x = k, y = pdf, color = "Fit Gamma"),
            size = 1) +
  scale_color_manual(values = c("KDE of Data" = "skyblue",
                                "Fit Gamma" = "navyblue")) +
  labs(title = paste("Empirical Marginal PDF vs. KDE of Data for k:", dataset),
       x = "k", y = "Density", color = NULL) +
  theme_bw(base_size = 22) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.65, 0.99),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = NA),
        legend.key = element_rect(fill = NA)) +
  guides(colour = guide_legend(override.aes = list(size = 3)))


# ====== Full PDF in Strength Space ====== #
# all this is to setup the strength grid to plot the PDF in strength space
# ====== Full PDF in Strength Space ====== #
# Grid in strength space
# ts_vals <- seq(10, 140, length.out = 300)
# bs_vals <- seq(10, 140, length.out = 300)
# strength_grid <- expand.grid(ts = ts_vals, bs = bs_vals)

# # Map strengths to parameters
# denom <- 2 * strength_grid$bs - strength_grid$ts
# strength_grid$alpha <- (strength_grid$ts - strength_grid$bs) / (sqrt(3) * denom)
# strength_grid$k <- (strength_grid$bs * strength_grid$ts) / (sqrt(3) * denom)

# # Valid domain: alpha >= a0, k >= 0, finite
# valid <- is.finite(denom) &
#   (strength_grid$alpha - a0) >= 0 &
#   strength_grid$k >= 0 &
#   is.finite(strength_grid$alpha) &
#   is.finite(strength_grid$k)

# # Allocate density, default zero outside support
# strength_grid$density <- 0

# # Pseudo observations under the shifted Gamma marginals
# alpha_shift_grid <- strength_grid$alpha[valid] - a0
# k_grid <- strength_grid$k[valid]

# u1_grid <- pgamma(alpha_shift_grid,
#                   shape = fit_alpha_gamma$estimate["shape"],
#                   rate  = fit_alpha_gamma$estimate["rate"])
# u2_grid <- pgamma(k_grid,
#                   shape = fit_k_gamma$estimate["shape"],
#                   rate  = fit_k_gamma$estimate["rate"])

# # Keep uniforms strictly inside (0,1) for copula density stability
# eps <- 1e-12
# u1_grid <- pmin(pmax(u1_grid, eps), 1 - eps)
# u2_grid <- pmin(pmax(u2_grid, eps), 1 - eps)

# # Copula density on the strength grid
# c_uv_strength <- dCopula(cbind(u1_grid, u2_grid), copula = cop_fitted)

# # Marginal PDFs at mapped points
# f_alpha <- dgamma(alpha_shift_grid,
#                   shape = fit_alpha_gamma$estimate["shape"],
#                   rate  = fit_alpha_gamma$estimate["rate"])
# f_k     <- dgamma(k_grid,
#                   shape = fit_k_gamma$estimate["shape"],
#                   rate  = fit_k_gamma$estimate["rate"])

# # Jacobian for (ts, bs) -> (alpha, k): |det d(alpha, k)/d(ts, bs)|
# # This simplifies to bs*ts / [3 * (2*bs - ts)^3]
# Jabs <- (strength_grid$bs[valid] * strength_grid$ts[valid]) /
#         (3 * (2 * strength_grid$bs[valid] - strength_grid$ts[valid])^3)

# # Final joint PDF in strength space
# strength_grid$density[valid] <- c_uv_strength * f_alpha * f_k * Jabs

# # Optional overlay of original strengths
# orig_strength <- transform(
#   data.frame(alpha = alpha, k = k),
#   ts = (3 * k) / (sqrt(3) + 3 * alpha),
#   bs = (3 * k) / (sqrt(3) + 6 * alpha)
# )

# # Plot the PDF in strength space
# ggplot(strength_grid, aes(x = ts, y = bs, fill = density)) +
#   geom_tile() +
#   geom_point(data = orig_strength,
#              aes(x = ts, y = bs),
#              color = "white", size = 0.5, alpha = 0.35, inherit.aes = FALSE) +
#   scale_fill_viridis_c() +
#   labs(title = "Empirical Joint PDF in Strength Space",
#        x = "Uniaxial Tensile Strength (ts)",
#        y = "Biaxial Tensile Strength (bs)") +
#   coord_cartesian(xlim = c(min(ts) - 5, max(ts) + 5),
#                   ylim = c(min(bs) - 5, max(bs) + 5)) +
#   theme_bw(base_size = 22) +
#   theme(
#     plot.title = element_text(hjust = 0.5),
#     legend.position = c(.82, 0.32),
#     legend.justification = c(0, 1),
#     legend.background = element_rect(fill = "white"),
#     legend.key = element_rect(fill = NA)
#   )



# # ====== Marginal Distribution of Alpha ====== #
# # Extract beta parameters
# a <- fit_beta$estimate["shape1"]
# b <- fit_beta$estimate["shape2"]

# # Rescale from [0,1] back to original domain
# alpha_vals <- seq(lower, upper, length.out = 500)
# alpha_unit_vals <- (alpha_vals - lower) / (upper - lower)

# # Compute scaled PDF (adjust for Jacobian)
# beta_pdf_vals <- dbeta(alpha_unit_vals, a, b) / (upper - lower)

# alpha_pdf_df <- data.frame(alpha = alpha_vals, pdf = beta_pdf_vals)

# # Plot marginal density of alpha
# ggplot() +
#   # KDE as line, no area, with proper legend
#   stat_density(data = original_df, aes(x = alpha, color = "KDE of Data"),
#                geom = "line", position = "identity", size = 1) +
#   # Analytical PDF as line
#   geom_line(data = alpha_pdf_df, aes(x = alpha, y = pdf, color = "Analytical PDF"),
#             size = 1) +
#   # Assign your colors
#   scale_color_manual(values = c("KDE of Data" = "skyblue", "Analytical PDF" = "navyblue")) +
#   labs(
#     title = "Empirical Marginal PDF vs. KDE of Data for α",
#     x = "α",
#     y = "Density",
#     color = ""
#   ) +
#   coord_cartesian(xlim = c(-0.15, 0.15)) +
#   theme_bw(base_size = 22) +
#   theme(
#     plot.title = element_text(hjust = 0.5),
#     legend.position = c(0.65, 0.99),
#     legend.justification = c(0, 1),
#     legend.background = element_rect(fill = NA),
#     legend.key = element_rect(fill = NA)
#   ) +
#   guides(colour = guide_legend(override.aes = list(size = 10)))


# # ====== Marginal Distribution of k ====== #

# # Extract gamma parameters
# shape <- fit_gamma$estimate["shape"]
# rate  <- fit_gamma$estimate["rate"]

# k_vals <- seq(min(sampled_df$k), max(sampled_df$k), length.out = 500)
# gamma_pdf_vals <- dgamma(k_vals, shape = shape, rate = rate)

# k_pdf_df <- data.frame(k = k_vals, pdf = gamma_pdf_vals)


# # Plot marginal density of k
# ggplot() +
#   # KDE as line
#   stat_density(data = original_df, aes(x = k, color = "KDE of Data"),
#                geom = "line", position = "identity", size = 1) +
#   # Analytical PDF as line (using correct data frame)
#   geom_line(data = k_pdf_df, aes(x = k, y = pdf, color = "Analytical PDF"),
#             size = 1) +
#   # Color mapping
#   scale_color_manual(values = c("KDE of Data" = "pink", "Analytical PDF" = "maroon")) +
#   labs(
#     title = "Empirical Marginal PDF vs. KDE of Data for k",
#     x = "k",
#     y = "Density",
#     color = ""
#   ) +
#   # coord_cartesian(xlim = c(-0.15, 0.15)) +
#   theme_bw(base_size = 22) +
#   theme(
#     plot.title = element_text(hjust = 0.5),
#     legend.position = c(0.65, 0.99),
#     legend.justification = c(0, 1),
#     legend.background = element_rect(fill = NA),
#     legend.key = element_rect(fill = NA)
#   ) +
#   guides(colour = guide_legend(override.aes = list(size = 10)))
