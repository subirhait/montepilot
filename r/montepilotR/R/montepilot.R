#' Check whether the MontePilot Python module is available
#' @export
montepilot_available <- function() {
  reticulate::py_module_available("montepilot")
}

#' Install the development MontePilot Python package
#'
#' @param path Path to the Python project root. For the public release this
#'   helper will install from PyPI instead.
#' @export
montepilot_install <- function(path = NULL) {
  if (is.null(path)) {
    stop("The public PyPI release does not exist yet; supply a local project path.")
  }
  reticulate::py_install(normalizePath(path), pip = TRUE)
  invisible(TRUE)
}

#' Report Python and backend availability
#' @export
montepilot_doctor <- function() {
  if (!montepilot_available()) {
    stop("Python module 'montepilot' is unavailable. Run montepilot_install().")
  }
  cli <- reticulate::import("montepilot.cli", delay_load = TRUE)
  reticulate::py_to_r(cli$doctor())
}

#' Run the built-in normal-mean demonstration
#'
#' @param backend One of "auto", "numpy", "torch_cpu", "torch_cuda", or
#'   "torch_xpu".
#' @param precision One of "auto", "float32", or "float64".
#' @param batch_size Number of replications evaluated in one batch.
#' @param min_reps Minimum replications before adaptive stopping.
#' @param max_reps Maximum replications per condition.
#' @param target_mcse Target Monte Carlo standard error of the estimate.
#' @param target_coverage_mcse Optional target Monte Carlo standard error of
#'   the estimated coverage probability.
#' @param stopping_rule Whether "all" or "any" configured precision targets
#'   must be reached.
#' @param seed Master seed.
#' @param checkpoint_dir Optional checkpoint directory.
#' @export
montepilot_demo <- function(
    backend = "auto",
    precision = "auto",
    batch_size = 512L,
    min_reps = 1000L,
    max_reps = 10000L,
    target_mcse = 0.005,
    target_coverage_mcse = NULL,
    stopping_rule = "all",
    seed = 20261002L,
    checkpoint_dir = NULL) {
  if (!montepilot_available()) {
    stop("Python module 'montepilot' is unavailable. Run montepilot_install().")
  }
  mp <- reticulate::import("montepilot", delay_load = TRUE)
  examples <- reticulate::import("montepilot.examples", delay_load = TRUE)
  config <- mp$RunConfig(
    backend = backend,
    precision = precision,
    batch_size = as.integer(batch_size),
    min_reps = as.integer(min_reps),
    max_reps = as.integer(max_reps),
    target_mcse = target_mcse,
    target_coverage_mcse = target_coverage_mcse,
    stopping_rule = stopping_rule,
    seed = as.integer(seed),
    checkpoint_dir = checkpoint_dir
  )
  report <- mp$SimulationRunner(config)$run(examples$normal_mean_design())
  jsonlite::fromJSON(report$to_json(indent = 2L), simplifyVector = FALSE)
}
