# Base-R reference benchmark for MontePilot comparisons.

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag, default) {
  position <- match(flag, args)
  if (is.na(position) || position == length(args)) {
    return(default)
  }
  args[[position + 1L]]
}

reps <- as.integer(arg_value("--reps", "10000"))
n <- as.integer(arg_value("--n", "1000"))
batch_size <- as.integer(arg_value("--batch-size", "1000"))
scalar_reps <- as.integer(arg_value("--scalar-reps", as.character(min(reps, 1000L))))
seed <- as.integer(arg_value("--seed", "20261002"))
timing_repeats <- as.integer(arg_value("--timing-repeats", "1"))
seed_step <- as.integer(arg_value("--seed-step", "100000"))
output <- arg_value("--output", "r_benchmark_results.csv")

if (any(!is.finite(c(reps, n, batch_size, scalar_reps, timing_repeats, seed_step))) ||
    any(c(reps, n, batch_size, scalar_reps, timing_repeats, seed_step) < 1L)) {
  stop(paste(
    "reps, n, batch-size, scalar-reps, timing-repeats, and seed-step",
    "must be positive integers"
  ))
}

run_batched <- function(current_seed, repeat_index) {
  set.seed(current_seed)
  estimates <- numeric(reps)
  elapsed <- system.time({
    start <- 1L
    while (start <= reps) {
      current <- min(batch_size, reps - start + 1L)
      # R fills matrices by column.  Each column therefore receives the same
      # consecutive random draws as one scalar replication, which creates a
      # paired common-random-number comparison without changing the target.
      draws <- matrix(
        rnorm(current * n, mean = 0.5, sd = 1.0),
        nrow = n,
        ncol = current
      )
      estimates[start:(start + current - 1L)] <- colMeans(draws)
      start <- start + current
    }
  })[["elapsed"]]
  data.frame(
    implementation = "base_r_batched",
    timing_repeat = repeat_index,
    seed = current_seed,
    reps = reps,
    n = n,
    batch_size = batch_size,
    elapsed_seconds = elapsed,
    reps_per_second = reps / elapsed,
    estimate_mean = mean(estimates),
    bias = mean(estimates) - 0.5,
    rmse = sqrt(mean((estimates - 0.5)^2)),
    stringsAsFactors = FALSE
  )
}

run_scalar <- function(current_seed, repeat_index) {
  set.seed(current_seed)
  estimates <- numeric(scalar_reps)
  elapsed <- system.time({
    for (index in seq_len(scalar_reps)) {
      estimates[[index]] <- mean(rnorm(n, mean = 0.5, sd = 1.0))
    }
  })[["elapsed"]]
  data.frame(
    implementation = "base_r_scalar_loop",
    timing_repeat = repeat_index,
    seed = current_seed,
    reps = scalar_reps,
    n = n,
    batch_size = 1L,
    elapsed_seconds = elapsed,
    reps_per_second = scalar_reps / elapsed,
    estimate_mean = mean(estimates),
    bias = mean(estimates) - 0.5,
    rmse = sqrt(mean((estimates - 0.5)^2)),
    stringsAsFactors = FALSE
  )
}

results <- do.call(
  rbind,
  lapply(seq_len(timing_repeats), function(repeat_index) {
    current_seed <- seed + (repeat_index - 1L) * seed_step
    rbind(
      run_batched(current_seed, repeat_index),
      run_scalar(current_seed, repeat_index)
    )
  })
)
results$r_version <- R.version.string
results$platform <- R.version$platform
write.csv(results, output, row.names = FALSE)
print(results)
cat("\nSaved:", normalizePath(output, winslash = "/", mustWork = FALSE), "\n")
