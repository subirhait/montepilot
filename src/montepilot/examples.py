"""Built-in designs used by documentation, tests, and the CLI."""

from __future__ import annotations

import numpy as np

from .design import BatchEstimate, SimulationDesign


def normal_mean_design(
    sample_sizes=(50, 200, 1000),
    true_mean: float = 0.5,
    true_sd: float = 1.0,
) -> SimulationDesign:
    conditions = [{"n": int(n), "mean": true_mean, "sd": true_sd} for n in sample_sizes]

    def generator(context, condition, batch_size):
        return context.backend.normal(
            (batch_size, int(condition["n"])),
            seed=context.seed,
            mean=float(condition["mean"]),
            sd=float(condition["sd"]),
        )

    def estimator(data, context, condition):
        estimates = context.backend.mean(data, axis=1)
        sample_sd = context.backend.std(data, axis=1, ddof=1)
        standard_errors = sample_sd / (float(condition["n"]) ** 0.5)
        return BatchEstimate(estimates=estimates, standard_errors=standard_errors)

    return SimulationDesign(
        name="normal-mean",
        description="Vectorized estimation of a normal population mean.",
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=true_mean,
    )


def linear_regression_design(
    sample_sizes=(250, 1000),
    predictors: int = 8,
    coefficient_index: int = 0,
    noise_sd: float = 1.0,
) -> SimulationDesign:
    """Batched ordinary least squares with independent Gaussian predictors."""

    if predictors < 1:
        raise ValueError("predictors must be positive")
    if not 0 <= coefficient_index < predictors:
        raise ValueError("coefficient_index is outside the coefficient vector")
    beta = np.linspace(0.25, 1.0, predictors, dtype=np.float64)
    conditions = [
        {"n": int(n), "predictors": predictors, "noise_sd": noise_sd}
        for n in sample_sizes
    ]

    def generator(context, condition, batch_size):
        n = int(condition["n"])
        p = int(condition["predictors"])
        x = context.backend.normal((batch_size, n, p), seed=context.seed)
        epsilon = context.backend.normal(
            (batch_size, n),
            seed=context.seed + 1,
            sd=float(condition["noise_sd"]),
        )
        coefficients = context.backend.asarray(beta)
        y = context.backend.matmul(x, coefficients) + epsilon
        return x, y

    def estimator(data, context, condition):
        x, y = data
        xt = context.backend.transpose_last2(x)
        xtx = context.backend.matmul(xt, x)
        xty = context.backend.matmul(xt, context.backend.expand_last(y))
        coefficients = context.backend.squeeze_last(context.backend.solve(xtx, xty))
        return BatchEstimate(estimates=coefficients[:, coefficient_index])

    return SimulationDesign(
        name="linear-regression",
        description="Batched OLS coefficient estimation using normal equations.",
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=float(beta[coefficient_index]),
    )


def ipw_ate_design(
    sample_sizes=(500, 2000),
    treatment_effect: float = 1.0,
) -> SimulationDesign:
    """Synthetic stabilized-IPW estimation of an average treatment effect."""

    conditions = [{"n": int(n), "treatment_effect": treatment_effect} for n in sample_sizes]

    def generator(context, condition, batch_size):
        n = int(condition["n"])
        x = context.backend.normal((batch_size, n), seed=context.seed)
        propensity = context.backend.clip(context.backend.sigmoid(0.7 * x), 0.05, 0.95)
        treatment = (
            context.backend.uniform((batch_size, n), seed=context.seed + 1) < propensity
        )
        noise = context.backend.normal((batch_size, n), seed=context.seed + 2)
        outcome = float(condition["treatment_effect"]) * treatment + 0.5 * x + noise
        return treatment, propensity, outcome

    def estimator(data, context, condition):
        treatment, propensity, outcome = data
        treated = treatment * 1.0
        control = 1.0 - treated
        treated_weight = treated / propensity
        control_weight = control / (1.0 - propensity)
        treated_mean = context.backend.sum(treated_weight * outcome, axis=1) / context.backend.sum(
            treated_weight, axis=1
        )
        control_mean = context.backend.sum(control_weight * outcome, axis=1) / context.backend.sum(
            control_weight, axis=1
        )
        return BatchEstimate(estimates=treated_mean - control_mean)

    return SimulationDesign(
        name="ipw-ate",
        description="Vectorized inverse-probability weighted ATE estimation.",
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=treatment_effect,
    )


def bootstrap_mean_design(
    data_size: int = 2000,
    resample_sizes=(500, 2000),
    data_seed: int = 731,
) -> SimulationDesign:
    """Nonparametric bootstrap distribution of a fixed sample mean."""

    if data_size < 2:
        raise ValueError("data_size must be at least 2")
    source = np.random.default_rng(data_seed).normal(0.5, 1.0, size=data_size)
    target = float(source.mean())
    conditions = [
        {"data_size": data_size, "resample_n": int(n)} for n in resample_sizes
    ]

    def generator(context, condition, batch_size):
        indices = context.backend.integers(
            (batch_size, int(condition["resample_n"])),
            seed=context.seed,
            low=0,
            high=int(condition["data_size"]),
        )
        values = context.backend.asarray(source)
        return values[indices]

    def estimator(data, context, condition):
        return BatchEstimate(estimates=context.backend.mean(data, axis=1))

    return SimulationDesign(
        name="bootstrap-mean",
        description="Batched nonparametric bootstrap of a fixed-sample mean.",
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=target,
    )


def congeneric_reliability_design(
    sample_sizes=(250, 1000),
    item_counts=(10, 20),
    loading_low: float = 0.6,
    loading_high: float = 0.9,
) -> SimulationDesign:
    """Cronbach-alpha simulation under a continuous congeneric measurement model.

    Each standardized item follows ``Y_j = lambda_j * theta + epsilon_j`` with
    ``Var(theta) = 1`` and ``Var(epsilon_j) = 1 - lambda_j**2``.  The population
    covariance matrix is therefore known, giving an analytic population alpha
    against which the sample estimator can be evaluated.
    """

    if not 0 < loading_low <= loading_high < 1:
        raise ValueError("loadings must satisfy 0 < loading_low <= loading_high < 1")
    conditions = []
    for n in sample_sizes:
        if int(n) < 3:
            raise ValueError("sample sizes must be at least three")
        for items in item_counts:
            if int(items) < 2:
                raise ValueError("item counts must be at least two")
            conditions.append({"n": int(n), "items": int(items)})

    def parameters(condition):
        loadings = np.linspace(
            loading_low,
            loading_high,
            int(condition["items"]),
            dtype=np.float64,
        )
        residual_sds = np.sqrt(1.0 - loadings**2)
        return loadings, residual_sds

    def generator(context, condition, batch_size):
        n = int(condition["n"])
        items = int(condition["items"])
        loadings, residual_sds = parameters(condition)
        ability = context.backend.normal((batch_size, n, 1), seed=context.seed)
        residuals = context.backend.normal(
            (batch_size, n, items), seed=context.seed + 1
        )
        return (
            ability * context.backend.asarray(loadings)
            + residuals * context.backend.asarray(residual_sds)
        )

    def estimator(data, context, condition):
        n = int(condition["n"])
        items = int(condition["items"])
        item_sums = context.backend.sum(data, axis=1)
        item_sum_squares = context.backend.sum(data * data, axis=1)
        item_variances = (
            item_sum_squares - item_sums * item_sums / float(n)
        ) / float(n - 1)
        total_scores = context.backend.sum(data, axis=2)
        total_sum = context.backend.sum(total_scores, axis=1)
        total_sum_squares = context.backend.sum(total_scores * total_scores, axis=1)
        total_variance = (
            total_sum_squares - total_sum * total_sum / float(n)
        ) / float(n - 1)
        alpha = (items / float(items - 1)) * (
            1.0 - context.backend.sum(item_variances, axis=1) / total_variance
        )
        return BatchEstimate(estimates=alpha)

    def truth(condition):
        loadings, residual_sds = parameters(condition)
        items = int(condition["items"])
        total_variance = float(loadings.sum() ** 2 + np.sum(residual_sds**2))
        trace = float(np.sum(loadings**2 + residual_sds**2))
        return (items / float(items - 1)) * (1.0 - trace / total_variance)

    return SimulationDesign(
        name="congeneric-reliability",
        description=(
            "Vectorized psychometric reliability simulation under a continuous "
            "congeneric measurement model."
        ),
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=truth,
    )


def cluster_randomized_trial_design(
    cluster_counts=(40, 100),
    cluster_sizes=(20, 30),
    treatment_effect: float = 0.2,
    intraclass_correlation: float = 0.15,
) -> SimulationDesign:
    """Balanced two-level educational cluster-randomized trial simulation."""

    if not 0 <= intraclass_correlation < 1:
        raise ValueError("intraclass_correlation must be in [0, 1)")
    conditions = []
    for clusters in cluster_counts:
        clusters = int(clusters)
        if clusters < 4 or clusters % 2:
            raise ValueError("cluster counts must be even and at least four")
        for cluster_size in cluster_sizes:
            if int(cluster_size) < 2:
                raise ValueError("cluster sizes must be at least two")
            conditions.append(
                {
                    "clusters": clusters,
                    "cluster_size": int(cluster_size),
                    "icc": float(intraclass_correlation),
                    "treatment_effect": float(treatment_effect),
                }
            )

    def treatment_vector(clusters: int):
        treatment = np.zeros(clusters, dtype=np.float64)
        treatment[clusters // 2 :] = 1.0
        return treatment

    def generator(context, condition, batch_size):
        clusters = int(condition["clusters"])
        cluster_size = int(condition["cluster_size"])
        icc = float(condition["icc"])
        treatment = context.backend.asarray(
            treatment_vector(clusters).reshape(1, clusters, 1)
        )
        random_intercept = context.backend.normal(
            (batch_size, clusters, 1),
            seed=context.seed,
            sd=icc**0.5,
        )
        residual = context.backend.normal(
            (batch_size, clusters, cluster_size),
            seed=context.seed + 1,
            sd=(1.0 - icc) ** 0.5,
        )
        return (
            float(condition["treatment_effect"]) * treatment
            + random_intercept
            + residual
        )

    def estimator(data, context, condition):
        clusters = int(condition["clusters"])
        group_clusters = clusters // 2
        cluster_means = context.backend.mean(data, axis=2)
        treated = context.backend.asarray(treatment_vector(clusters).reshape(1, clusters))
        control = 1.0 - treated
        treated_sum = context.backend.sum(cluster_means * treated, axis=1)
        control_sum = context.backend.sum(cluster_means * control, axis=1)
        treated_mean = treated_sum / float(group_clusters)
        control_mean = control_sum / float(group_clusters)
        estimates = treated_mean - control_mean

        squares = cluster_means * cluster_means
        treated_variance = (
            context.backend.sum(squares * treated, axis=1)
            - treated_sum * treated_sum / float(group_clusters)
        ) / float(group_clusters - 1)
        control_variance = (
            context.backend.sum(squares * control, axis=1)
            - control_sum * control_sum / float(group_clusters)
        ) / float(group_clusters - 1)
        standard_errors = context.backend.sqrt(
            treated_variance / float(group_clusters)
            + control_variance / float(group_clusters)
        )
        return BatchEstimate(estimates=estimates, standard_errors=standard_errors)

    return SimulationDesign(
        name="cluster-randomized-trial",
        description=(
            "Vectorized two-level educational cluster-randomized trial with "
            "random school intercepts."
        ),
        conditions=conditions,
        generator=generator,
        estimator=estimator,
        truth=treatment_effect,
    )
