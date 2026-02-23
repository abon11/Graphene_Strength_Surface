# Required pkgs
library(readr)
library(fitdistrplus)
library(copula)
library(ggplot2)
library(rlang)

# This script does everything we need from R in clean functions and outputs the necessary parameters.
# This is outdated in the current workflow, we no longer use copulas

#######
# This function actually takes in the data and fits the copula function
# It returns if it trims the data, as well as the gamma marginals and full copula object
#######
fit_copula_model <- function(csv_path,
                             alpha_thresh = 1,
                             k_thresh = 100,
                             a0 = -sqrt(3) / 6) {
  # ---- Load data ----
  data <- read_csv(csv_path, show_col_types = FALSE)

  # ---- Trim outliers ----
  data_trimmed <- subset(data, alpha <= alpha_thresh & k <= k_thresh)
  alpha <- data_trimmed$alpha
  k <- data_trimmed$k

  removed_count <- nrow(data) - nrow(data_trimmed)
  cat("Number of rows removed:", removed_count, "\n")

  # ---- Shift alpha for gamma fitting ----
  alpha_shift <- alpha - a0
  if (any(alpha_shift < 0)) {
    stop("Some alpha values below the lower bound a0. Check data or adjust a0.")
  }

  # ---- Fit marginals ----
  fit_alpha_gamma <- fitdist(alpha_shift, "gamma", method = "mle")
  fit_k_gamma     <- fitdist(k, "gamma", method = "mle")

  # ---- Convert to uniforms ----
  u1 <- pgamma(alpha_shift,
               shape = fit_alpha_gamma$estimate["shape"],
               rate  = fit_alpha_gamma$estimate["rate"])
  u2 <- pgamma(k,
               shape = fit_k_gamma$estimate["shape"],
               rate  = fit_k_gamma$estimate["rate"])

  eps <- 1e-12
  u_fit <- cbind(pmin(pmax(u1, eps), 1 - eps),
                 pmin(pmax(u2, eps), 1 - eps))

  # ---- Fit Gaussian copula ----
  cop <- normalCopula(dim = 2)
  fit_cop <- fitCopula(cop, u_fit, method = "ml")

  # ---- Return results ----
  list(
    file = csv_path,
    removed = nrow(data) - nrow(data_trimmed),
    n_used = nrow(data_trimmed),
    a0 = a0,
    alpha_params = fit_alpha_gamma$estimate,
    k_params = fit_k_gamma$estimate,
    copula_fit = fit_cop,
    dataset = data_trimmed
  )
}

#######
# This function takes in the marginals and the copula, and returns the grid of
# probability densities, primed for plotting.
#######
compute_joint_density_grid <- function(fit_alpha_gamma,
                                       fit_k_gamma,
                                       copula_fit,
                                       a0 = -sqrt(3) / 6,
                                       amin, amax,
                                       kmin, kmax,
                                       n_alpha = 1000,
                                       n_k = 1000) {
  # ---- Generate grid ----
  alpha_vals <- seq(amin - 0.1, amax + 0.1, length.out = n_alpha)
  k_vals     <- seq(kmin - 10, kmax + 10, length.out = n_k)
  grid <- expand.grid(alpha = alpha_vals, k = k_vals)

  # ---- Compute CDFs on grid ----
  alpha_shift <- grid$alpha - a0
  u1 <- pgamma(alpha_shift,
               shape = fit_alpha_gamma["shape"],
               rate  = fit_alpha_gamma["rate"])
  u2 <- pgamma(grid$k,
               shape = fit_k_gamma["shape"],
               rate  = fit_k_gamma["rate"])

  # ---- Compute PDFs on grid ----
  f_alpha <- dgamma(alpha_shift,
                    shape = fit_alpha_gamma["shape"],
                    rate  = fit_alpha_gamma["rate"])
  f_alpha[alpha_shift < 0] <- 0  # outside support

  f_k <- dgamma(grid$k,
                shape = fit_k_gamma["shape"],
                rate  = fit_k_gamma["rate"])

  # ---- Evaluate copula density ----
  # Handle both fitCopula and plain copula objects
  cop_obj <- if (inherits(copula_fit, "fitCopula")) copula_fit@copula else copula_fit
  c_uv <- dCopula(cbind(u1, u2), copula = cop_obj)

  # ---- Compute joint density ----
  grid$density <- c_uv * f_alpha * f_k

  grid
}

#######
# This function takes in the density grid and actually plots the joint density
#######
plot_joint_density <- function(grid,
                               title = "Joint PDF",
                               original_df = NULL,
                               base_size = 22) {
  # infer plot bounds from the grid
  ax <- range(grid$alpha, na.rm = TRUE)
  kx <- range(grid$k,     na.rm = TRUE)

  p <- ggplot(grid, aes(x = .data$alpha, y = .data$k, fill = density)) +
    geom_tile() +
    scale_fill_viridis_c(option = "mako") +
    labs(title = title, x = expression(alpha), y = "k", fill = "Density") +
    coord_cartesian(xlim = ax, ylim = kx) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position.inside = c(0.82, 0.32),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = "white"),
      legend.key = element_rect(fill = NA)
    )

  if (!is.null(original_df) && nrow(original_df) > 0) {
    p <- p + geom_point(
      data = original_df,
      aes(x = .data$alpha, y = .data$k),
      color = "red", linewidth = 2, alpha = 0.5, inherit.aes = FALSE
    )
  }
  p
}

#######
# This function takes in the fit gamma (and data if you want) and plots the marginal
# This does it just for one dataset --> quick and easy
#######
plot_marginal_gamma <- function(data = NULL,
                                fit_gamma,
                                a0 = 0,            # shift for alpha (0 for k)
                                title = "Marginal Distribution",
                                color_fit = "navyblue",
                                color_kde = "skyblue",
                                base_size = 22) {
  # --- Build x grid for smooth curve ---
  if (!is.null(data) && length(data) > 0) {
    x_min <- min(data, na.rm = TRUE)
    x_max <- max(data, na.rm = TRUE)
  } else {
    x_min <- a0
    x_max <- qgamma(0.999, shape = fit_gamma["shape"], rate = fit_gamma["rate"]) + a0
  }

  x_vals <- seq(x_min, x_max, length.out = 600)
  pdf_vals <- dgamma(x_vals - a0, shape = fit_gamma["shape"], rate = fit_gamma["rate"])
  pdf_vals[x_vals < a0] <- 0
  curve_df <- data.frame(x = x_vals, pdf = pdf_vals)

  # --- Start plot with fitted gamma ---
  p <- ggplot(curve_df, aes(x = .data$x, y = pdf, color = "Fit Gamma")) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = c("Fit Gamma" = color_fit, "KDE of Data" = color_kde)) +
    labs(title = title, x = NULL, y = "Density", color = NULL) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = c(0.65, 0.99),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = NA),
      legend.key = element_rect(fill = NA)
    ) +
    guides(colour = guide_legend(override.aes = list(linewidth = 3)))

  # --- Optional KDE overlay if data provided ---
  if (!is.null(data) && length(data) > 0) {
    p <- p +
      stat_density(
        data = data.frame(x = data),
        aes(x = .data$x, y = after_stat(density), color = "KDE of Data"),
        geom = "line", linewidth = 1
      )
  }

  p
}

#######
# This function sets up a new marginal plot and returns it so that you can add onto it
#######
new_marginal_plot <- function(title = "Overlay of marginals", base_size = 22) {
  ggplot() +
    labs(title = title, x = NULL, y = "Density", color = NULL) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = c(0.65, 0.99),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = NA),
      legend.key = element_rect(fill = NA)
    )
}

# params: a named vector/list with elements "shape" and "rate"
# data: optional numeric vector (e.g., original_df$alpha or original_df$k)
add_marginal_gamma <- function(p,
                               params,
                               data = NULL,
                               a0 = 0,
                               label = "fit",
                               n = 600,
                               bw = "nrd0", adjust = 1, kernel = "gaussian",
                               color_fit = NULL, color_kde = NULL) {
  shape <- unname(params[["shape"]])
  rate <- unname(params[["rate"]])

  # x-range from data if available, else from fit
  if (!is.null(data) && length(data) > 0) {
    x_min <- min(data, na.rm = TRUE)
    x_max <- max(data, na.rm = TRUE)
  } else {
    x_min <- a0
    x_max <- qgamma(0.999, shape = shape, rate = rate) + a0
  }

  x_vals <- seq(x_min, x_max, length.out = n)
  pdf_vals <- dgamma(x_vals - a0, shape = shape, rate = rate)
  pdf_vals[x_vals < a0] <- 0
  curve_df <- data.frame(x = x_vals, pdf = pdf_vals, dataset = label, check.names = FALSE)

  # add KDE first (if provided), then the fitted curve
  if (!is.null(data) && length(data) > 0) {
    p <- p +
      stat_density(
        data = data.frame(x = data, dataset = label),
        aes(x = .data$x, y = after_stat(density), color = .data$dataset),
        geom = "line", linewidth = 1, bw = bw, adjust = adjust, kernel = kernel
      )
  }

  p <- p +
    geom_line(
      data = curve_df,
      aes(x = .data$x, y = .data$pdf, color = .data$dataset),
      linewidth = 1
    )

  # optional fixed colors per label
  if (!is.null(color_fit) || !is.null(color_kde)) {
    # if you pass a named vector of colors covering all labels, do it once at the end instead
    p <- p + scale_color_manual(values = c(setNames(color_fit, label)))
  }

  p
}

#######
# This function puts all of the fits you give it on the same marginal plot (for alpha or k)
#######
plot_marginals_from_fits <- function(fits,
                                     variable = c("alpha", "k"),
                                     alpha_thresh = 1,
                                     k_thresh = 100,
                                     title_prefix = "Marginal Gamma Distributions") {
  variable <- match.arg(variable)

  # Base plot
  p <- new_marginal_plot(paste(title_prefix, variable))

  # Loop through all fitted results
  for (fit in fits) {
    # Extract dataset name from CSV path
    csv_path <- fit$file
    name <- gsub("^DPparams_|\\.csv$", "", basename(csv_path))

    # Load and trim raw data for KDE
    # df <- readr::read_csv(csv_path, show_col_types = FALSE)
    # trimmed_data <- subset(df, alpha <= alpha_thresh & k <= k_thresh)[[variable]]

    # Add each dataset’s fit to the plot
    p <- add_marginal_gamma(
      p       = p,
      params  = fit[[paste0(variable, "_params")]],  # alpha_params or k_params
      data    = NULL,  # pass trimmed_data through this if we want the KDE's as well
      a0      = fit$a0 * (variable == "alpha"),      # shift only for alpha
      label   = name
    )
  }

  p
}






#######
# Now we can use those functions here
#######


# fit <- fit_copula_model("DPparams_AC_SV.csv")

# # Define grid bounds
# amin <- -0.15
# amax <- 0.5
# kmin <- 25
# kmax <- 90

# grid_df <- compute_joint_density_grid(
#   fit_alpha_gamma = fit$alpha_params,   # gamma fit for alpha
#   fit_k_gamma     = fit$k_params,       # gamma fit for k
#   copula_fit      = fit$copula_fit,     # full fitCopula object
#   a0              = fit$a0,
#   amin = amin, amax = amax,
#   kmin = kmin, kmax = kmax
# )

# plot_joint_density(grid_df, title = "Empirical Joint PDF: ZZ_MX",
#                    original_df = fit$dataset)

# plot_marginal_gamma(data = fit$dataset$alpha, fit_gamma = fit$alpha_params, a0 = -sqrt(3) / 6)
# plot_marginal_gamma(data = fit$dataset$k, fit_gamma = fit$k_params)

# datasets <- c("ZZ_MX", "ZZ_DV", "ZZ_SV", "AC_MX", "AC_DV", "AC_SV")

# fits <- lapply(datasets, function(name) {
#   csv_path <- paste0("DPparams_", name, ".csv")
#   fit_copula_model(csv_path)
# })

# # now have a list of six fitted results
# fits[[1]]$alpha_params  # gamma params for alpha in first dataset
# fits[[1]]$k_params  # gamma params for k in first dataset
# coef(fits[[1]]$copula_fit)  # copula correlation coefficient (ρ)


datasets <- c("ZZ_MX", "ZZ_DV", "ZZ_SV", "AC_MX", "AC_DV", "AC_SV")

# your existing fits
fits <- lapply(datasets, function(name) {
  csv_path <- paste0("DPparams_", name, ".csv")
  fit_copula_model(csv_path)
})

# Overlay for alpha
p_alpha <- plot_marginals_from_fits(
  fits = fits,
  variable = "alpha",
  alpha_thresh = 1,
  k_thresh = 100,
  title_prefix = "Alpha Marginals:"
)
p_alpha

# Overlay for k
p_k <- plot_marginals_from_fits(
  fits = fits,
  variable = "k",
  alpha_thresh = 1,
  k_thresh = 100,
  title_prefix = "k Marginals:"
)
p_k

# Finally, save all of the parameters for later analysis
fits_summary <- do.call(rbind, lapply(fits, function(f) {
  data.frame(
    dataset = gsub("^DPparams_|\\.csv$", "", basename(f$file)),
    a0 = f$a0,
    alpha_shape = f$alpha_params["shape"],
    alpha_rate  = f$alpha_params["rate"],
    k_shape = f$k_params["shape"],
    k_rate  = f$k_params["rate"],
    rho = coef(f$copula_fit)
  )
}))

write.csv(fits_summary, "fitted_parameters.csv", row.names = FALSE)
