## =============================================================================
## test_split_simulation.R
##
## Purpose:
##   Computes the minimum number of test samples required per class and
##   in total (balanced design) to achieve a desired confidence interval
##   width around the per-class F1 score. Prints results and LaTeX output
##   (formula + kable table) to the console.
##
## The analytical foundation:
##   The F1 score per class behaves like a proportion-based metric.
##   Its standard error can be approximated using the normal approximation
##   for proportions. The worst-case variance occurs at F1 = 0.5, which
##   maximises the product F1 * (1 - F1). This script uses that worst-case
##   assumption to guarantee the CI width holds regardless of actual model
##   performance.
##
## Dependencies:
##   - knitr      (for kable LaTeX table generation)
##   - kableExtra (for enhanced LaTeX table styling with booktabs)
##
## Usage:
##   1. Set k (number of classes) in the configuration section
##   2. Run: Rscript test_split_simulation.R
##   3. All output (summary, LaTeX formula, LaTeX table, simulation) is
##      printed to the console.
## =============================================================================


# =============================================================================
# Install and load required packages if not already available
# =============================================================================
if (!requireNamespace("knitr", quietly = TRUE)) {
  install.packages("knitr", repos = "https://cloud.r-project.org")
}
library(knitr)

if (!requireNamespace("kableExtra", quietly = TRUE)) {
  install.packages("kableExtra", repos = "https://cloud.r-project.org")
}
library(kableExtra)


# =============================================================================
# compute_test_split_table
#
# Computes minimum test samples per class and total test set size
# for a set of target confidence interval half-widths.
#
# Parameters:
#   k          - number of classification categories (integer)
#   ci_targets - numeric vector of desired 95% CI half-widths
#   f1_assumed - assumed per-class F1 for worst-case variance (default 0.5)
#   z          - z-value for the confidence level (default 1.96 for 95%)
#
# Returns:
#   data.frame with columns: ci_width, samples_per_class, total_test_set
# =============================================================================
compute_test_split_table <- function(k,
                                     ci_targets = c(0.03, 0.05, 0.07, 0.10),
                                     f1_assumed  = 0.5,
                                     z           = 1.96) {
  
  ## Derived from:  w = z * sqrt( F1*(1-F1) / n )
  ## Solved for n:  n = z^2 * F1*(1-F1) / w^2
  samples_per_class <- ceiling((z^2 * f1_assumed * (1 - f1_assumed)) / ci_targets^2)
  total_test_set    <- samples_per_class * k
  
  data.frame(
    ci_width          = ci_targets,
    samples_per_class = samples_per_class,
    total_test_set    = total_test_set
  )
}


# =============================================================================
# print_latex_formula
#
# Prints the LaTeX representation of the sample-size formula and its
# variable definitions to the console.
# =============================================================================
print_latex_formula <- function() {
  cat("\\begin{equation}\n")
  cat("  n_{\\text{class}} = ")
  cat("\\frac{z^{2} \\cdot \\hat{F}_{1} \\cdot (1 - \\hat{F}_{1})}{w^{2}}\n")
  cat("\\label{eq:sample_size}\n")
  cat("\\end{equation}\n\n")
  cat("\\noindent where\n")
  cat("\\begin{itemize}\n")
  cat("  \\item $n_{\\text{class}}$ is the minimum number of test samples required per class,\n")
  cat("  \\item $z$ is the critical value of the standard normal distribution ($z = 1.96$ for a 95\\% confidence level),\n")
  cat("  \\item $\\hat{F}_{1}$ is the assumed per-class $F_1$ score (set to 0.5 for maximum variance, i.e.\\ the worst case),\n")
  cat("  \\item $w$ is the desired half-width of the 95\\% confidence interval (e.g.\\ $\\pm 0.05$).\n")
  cat("\\end{itemize}\n\n")
  cat("\\noindent The total test set size for a balanced design with $k$ classes is:\n")
  cat("\\begin{equation}\n")
  cat("  N_{\\text{total}} = n_{\\text{class}} \\cdot k\n")
  cat("\\label{eq:total_size}\n")
  cat("\\end{equation}\n")
}


# =============================================================================
# print_latex_table_kable
#
# Uses knitr::kable with kableExtra styling to print a LaTeX-formatted
# table of the results to the console.
#
# Parameters:
#   df - data.frame returned by compute_test_split_table
#   k  - number of classes (used in the caption)
# =============================================================================
print_latex_table_kable <- function(df, k) {
  
  ## Format the CI width column as plus-minus values for LaTeX display
  df_display <- data.frame(
    `Target CI width` = sprintf("$\\pm %.2f$", df$ci_width),
    `Samples per class` = format(df$samples_per_class, big.mark = ","),
    `Total test set (balanced)` = format(df$total_test_set, big.mark = ","),
    check.names = FALSE
  )
  
  ## The caption summarises the assumptions used in the computation
  caption_text <- paste0(
    "Minimum test set sizes for reliable per-class $F_1$ estimation ",
    "($k = ", k, "$, worst-case $\\hat{F}_{1} = 0.5$, $z = 1.96$)."
  )
  
  ## Generate and print the LaTeX table via kable + kableExtra
  latex_out <- kable(
    df_display,
    format    = "latex",
    booktabs  = TRUE,
    escape    = FALSE,
    align     = c("r", "r", "r"),
    caption   = caption_text,
    label     = "tab:test_split"
  ) |>
    kable_styling(latex_options = c("hold_position"))
  
  cat(as.character(latex_out))
  cat("\n")
}


# =============================================================================
# simulate_ci_width
#
# Runs a Monte-Carlo simulation to empirically verify the analytical formula.
# For each per-class sample size, it simulates binary classification outcomes,
# computes F1, and measures the empirical 95% CI width across repetitions.
#
# Parameters:
#   n_values - vector of per-class sample sizes to simulate
#   f1_true  - the true (population) F1 to simulate around (default 0.5)
#   n_sim    - number of simulation repetitions (default 2000)
#
# Returns:
#   data.frame with columns: n_per_class, analytical_ci, empirical_ci
# =============================================================================
simulate_ci_width <- function(n_values = c(50, 100, 200, 384, 500, 1000),
                              f1_true  = 0.5,
                              n_sim    = 2000) {
  
  z <- 1.96
  
  ## Analytical CI half-width for each n, derived from the normal approximation
  analytical_ci <- z * sqrt(f1_true * (1 - f1_true) / n_values)
  
  ## Empirical CI half-width via Monte-Carlo simulation
  empirical_ci <- numeric(length(n_values))
  
  for (j in seq_along(n_values)) {
    n <- n_values[j]
    f1_samples <- numeric(n_sim)
    
    for (i in seq_len(n_sim)) {
      ## Simulate true positives, false positives, and false negatives
      ## by drawing from binomial distributions
      tp <- rbinom(1, size = n, prob = f1_true)
      fp <- rbinom(1, size = n - tp, prob = 0.5)
      fn <- n - tp
      
      precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
      recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
      f1_samples[i] <- ifelse(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0
      )
    }
    
    ## The empirical 95% CI half-width is half the distance between
    ## the 2.5th and 97.5th percentile of the simulated F1 distribution
    ci_bounds <- quantile(f1_samples, probs = c(0.025, 0.975))
    empirical_ci[j] <- (ci_bounds[2] - ci_bounds[1]) / 2
  }
  
  data.frame(
    n_per_class   = n_values,
    analytical_ci = round(analytical_ci, 4),
    empirical_ci  = round(empirical_ci, 4)
  )
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

## ---- Configuration ----------------------------------------------------------
## Number of classes. Change this value for your use case.
k <- 13

## Target CI half-widths to evaluate
ci_targets <- c(0.03, 0.05, 0.07, 0.10)


## ---- Compute results --------------------------------------------------------
results <- compute_test_split_table(k, ci_targets)


## ---- Console summary --------------------------------------------------------
cat("=== Test Split Size Computation ===\n")
cat(sprintf("Number of classes (k): %d\n", k))
cat(sprintf("Assumed F1 (worst case): 0.5\n"))
cat(sprintf("Confidence level: 95%% (z = 1.96)\n\n"))
print(results, row.names = FALSE)


## ---- LaTeX formula ----------------------------------------------------------
cat("\n\n=== LaTeX Formula ===\n\n")
print_latex_formula()


## ---- LaTeX table via kable --------------------------------------------------
cat("\n\n=== LaTeX Table (kable) ===\n\n")
print_latex_table_kable(results, k)


## ---- Monte-Carlo simulation for verification --------------------------------
cat("\n\n=== Monte-Carlo Simulation (verification) ===\n")
cat("Comparing analytical CI to empirical CI from 2,000 simulations:\n\n")
sim_results <- simulate_ci_width()
print(sim_results, row.names = FALSE)
cat("\n")