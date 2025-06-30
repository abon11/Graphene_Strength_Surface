
library(readr)
library(fitdistrplus)
library(copula)
library(ggplot2)

# Load data
data <- read_csv("drucker_prager_params.csv")
alpha <- data$alpha
k <- data$k

# Define bounds for alpha
lower <- -sqrt(3) / 3
upper <-  sqrt(3) / 3

# Transform alpha to [0, 1]
alpha_unit <- (alpha - lower) / (upper - lower)
# alpha_unit <- pmin(pmax((alpha - lower) / (upper - lower), 0.01), 0.99)


# Fit marginals - these use maximum likelihood estimation
fit_beta <- fitdist(alpha_unit, "beta", method = "mle")
fit_beta$estimate
fit_gamma <- fitdist(k, "gamma", method = "mle")

# Convert to pseudo-observations
# Values transformed to the uniform space via marginal CDFs.
u1 <- pbeta(
  alpha_unit,
  shape1 = fit_beta$estimate["shape1"],
  shape2 = fit_beta$estimate["shape2"]
)
u2 <- pgamma(
  k,
  shape = fit_gamma$estimate["shape"],
  rate = fit_gamma$estimate["rate"]
)

# u is a 2D array of points in [0, 1]^2 to use for copula fitting
u <- cbind(u1, u2)

# Fit Gaussian copula
cop <- normalCopula(dim = 2)
fit_cop <- fitCopula(cop, u, method = "ml")  # uses MLE
cop_fitted <- fit_cop@copula  # extract best fit

# Sample from copula
u_sampled <- rCopula(100000, cop_fitted)

# Invert marginals using quantile function
alpha_sampled_scaled <- qbeta(
  u_sampled[, 1],
  shape1 = fit_beta$estimate["shape1"],
  shape2 = fit_beta$estimate["shape2"]
)
alpha_samp <- alpha_sampled_scaled * (upper - lower) + lower
k_samp <- qgamma(
  u_sampled[, 2],
  shape = fit_gamma$estimate["shape"],
  rate = fit_gamma$estimate["rate"]
)

# Combine original and sampled data
original_df <- data.frame(alpha = alpha, k = k, source = "original")
sampled_df <- data.frame(alpha = alpha_samp, k = k_samp, source = "sampled")
plot_df <- rbind(original_df, sampled_df)
# clipped_alphas <- alpha_unit * (upper - lower) + lower
# clipped_df <- data.frame(alpha = clipped_alphas, k = k, source = "original_clipped")
# plot_clipped_df <- rbind(clipped_df, sampled_df)

# turn alphas and ks into sigma_ts and sigma_cs (for visualization)
cs <- (3 * k_samp) / (sqrt(3) - 3 * alpha_samp)
ts <- (3 * k_samp) / (sqrt(3) + 3 * alpha_samp)
str_df <- data.frame(sigma_cs = cs, sigma_ts = ts, source = "Sampled Strength")

# Plot strength scatter
ggplot(str_df, aes(x = sigma_cs, y = sigma_ts, color = source)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Sampled Strengths in Stress Space",
    x = "Compressive Strength", y = "Tensile Strength"
  ) +
  coord_cartesian(xlim = c(30, 130), ylim = c(30, 130)) +
  theme_bw()

# Plot joint scatter
ggplot(plot_df, aes(x = alpha, y = k, color = source)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Original vs. Sampled Joint Distribution",
    x = expression(alpha), y = "k"
  ) +
  theme_minimal()

# Plot marginal density of alpha
ggplot(plot_df, aes(x = alpha, fill = source)) +
  geom_density(alpha = 0.5) +
  labs(title = "Alpha Marginal: Original vs. Sampled") +
  theme_minimal()

# Plot marginal density of k
ggplot(plot_df, aes(x = k, fill = source)) +
  geom_density(alpha = 0.5) +
  labs(title = "k Marginal: Original vs. Sampled") +
  theme_minimal()

# This is for the full PDF
# Generate density grid over domain
alpha_vals <- seq(lower - 0.1, upper + 0.1, length.out = 1000)
k_vals <- seq(-5, max(k) + 10, length.out = 1000)
grid <- expand.grid(alpha = alpha_vals, k = k_vals)

# Marginal CDFs and PDFs
# transform grid into pseudo-observations
alpha_unit_grid <- (grid$alpha - lower) / (upper - lower)
u1_grid <- pbeta(
  alpha_unit_grid,
  shape1 = fit_beta$estimate["shape1"],
  shape2 = fit_beta$estimate["shape2"]
)
u2_grid <- pgamma(
  grid$k,
  shape = fit_gamma$estimate["shape"],
  rate = fit_gamma$estimate["rate"]
)
# Evaluate marginal PDFs at each gridpoint
f_alpha <- dbeta(
  alpha_unit_grid,
  shape1 = fit_beta$estimate["shape1"],
  shape2 = fit_beta$estimate["shape2"]
) / (upper - lower)
f_k <- dgamma(
  grid$k,
  shape = fit_gamma$estimate["shape"],
  rate = fit_gamma$estimate["rate"]
)

# Joint PDF via copula - evaluate copula density:
# multiply the copula with the marginals to get the true joint PDF
c_uv <- dCopula(cbind(u1_grid, u2_grid), copula = cop_fitted)
grid$density <- c_uv * f_alpha * f_k

# Plot joint PDF
ggplot(grid, aes(x = alpha, y = k, fill = density)) +
  geom_tile() +
  geom_point(data = original_df, aes(x = alpha, y = k),
             color = "white", size = 0.5, alpha = 0.35, inherit.aes = FALSE) +
  scale_fill_viridis_c() +
  labs(title = "True Joint PDF from Copula + Marginals") +
  coord_cartesian(xlim = c(-0.25, 0.25), ylim = c(min(k) - 5, max(k) + 5)) +
  theme_bw(base_size = 22)


# Joint PDF with zero-density contour
ggplot(grid, aes(x = alpha, y = k)) +
  geom_tile(aes(fill = density)) +
  geom_point(data = original_df, aes(x = alpha, y = k),
             color = "white", size = 0.5, alpha = 0.35, inherit.aes = FALSE) +
  stat_contour(
    aes(z = density), breaks = c(1e-15),
    color = "red", linewidth = 1
  ) +
  scale_fill_viridis_c() +
  labs(
    title = "Joint PDF with Zero-Density Contour",
    fill = "Density"
  ) +
  coord_cartesian(xlim = c(-0.25, 0.25), ylim = c(min(k) - 5, max(k) + 5)) +
  theme_bw()



# Extract beta parameters
a <- fit_beta$estimate["shape1"]
b <- fit_beta$estimate["shape2"]

# Rescale from [0,1] back to original domain
alpha_vals <- seq(lower, upper, length.out = 500)
alpha_unit_vals <- (alpha_vals - lower) / (upper - lower)

# Compute scaled PDF (adjust for Jacobian)
beta_pdf_vals <- dbeta(alpha_unit_vals, a, b) / (upper - lower)

alpha_pdf_df <- data.frame(alpha = alpha_vals, pdf = beta_pdf_vals)

ggplot() +
  geom_histogram(data = original_df, aes(x = alpha, y = ..density..),
                 bins = 200, fill = "skyblue", alpha = 0.5) +
  geom_line(data = alpha_pdf_df, aes(x = alpha, y = pdf),
            color = "black", size = 1) +
  labs(
    title = "True Marginal PDF vs. Data for α",
    x = "α",
    y = "Density"
  ) +
  coord_cartesian(xlim = c(-0.15, 0.15)) +
  theme_bw(base_size = 22)



# Extract gamma parameters
shape <- fit_gamma$estimate["shape"]
rate  <- fit_gamma$estimate["rate"]

k_vals <- seq(min(sampled_df$k), max(sampled_df$k), length.out = 500)
gamma_pdf_vals <- dgamma(k_vals, shape = shape, rate = rate)

k_pdf_df <- data.frame(k = k_vals, pdf = gamma_pdf_vals)

ggplot() +
  geom_histogram(data = original_df, aes(x = k, y = ..density..),
                 bins = 50, fill = "salmon", alpha = 0.5) +
  geom_line(data = k_pdf_df, aes(x = k, y = pdf),
            color = "black", size = 1) +
  labs(
    title = "True Marginal PDF vs. Data for k",
    x = "k",
    y = "Density",
  ) +
  theme_bw(base_size = 22)
