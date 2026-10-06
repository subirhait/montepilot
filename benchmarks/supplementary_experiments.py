"""Hardware-independent supplementary experiments for the MontePilot manuscript.

E1 float32 variance stress test on controlled inputs (application estimators)
E2 adaptive stopping: realized replications vs a fixed-R design
E3 interrupted-and-resumed run vs uninterrupted run (checkpoint equivalence)
E4 analytic checks: z-interval coverage and small-sample alpha bias
"""
import json, math, platform, tempfile, time
from dataclasses import replace
import numpy as np
from scipy import stats
import montepilot
from montepilot import RunConfig, SimulationRunner
from montepilot.backends import resolve_backend
from montepilot.design import BatchContext, SimulationDesign
from montepilot.examples import (congeneric_reliability_design,
                                 cluster_randomized_trial_design)

out = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                       "montepilot": montepilot.__version__, "platform": platform.platform()}}

# ---------------- E1 ----------------
def controlled(design_factory, condition, gen64, reps, offsets, seed=2026):
    rows = []
    for off in offsets:
        rng = np.random.default_rng(seed)
        data64 = gen64(rng, reps) + off
        b64 = resolve_backend("numpy", precision="float64")
        b32 = resolve_backend("numpy", precision="float32")
        res = {}
        for method in ("one_pass", "centered"):
            d = design_factory(method)
            ctx64 = BatchContext(b64, 0, 0, 0); ctx32 = BatchContext(b32, 0, 0, 0)
            ref = d.estimator(b64.asarray(data64), ctx64, condition)
            cand = d.estimator(b32.asarray(data64), ctx32, condition)
            for kind in ("estimates", "standard_errors"):
                r = getattr(ref, kind); c = getattr(cand, kind)
                if r is None: continue
                err = np.asarray(c, dtype=np.float64) - np.asarray(r, dtype=np.float64)
                res[f"{method}_{kind}_max_abs_error"] = float(np.max(np.abs(err)))
                res[f"{method}_{kind}_mean_error"] = float(np.mean(err))
        # sampling scale of the estimator (float64, centered)
        d = design_factory("centered")
        ref = d.estimator(resolve_backend("numpy", precision="float64").asarray(data64),
                          BatchContext(resolve_backend("numpy", precision="float64"), 0, 0, 0), condition)
        res["estimator_sd"] = float(np.std(np.asarray(ref.estimates), ddof=1))
        res["offset"] = off
        rows.append(res)
    return rows

lam = np.linspace(0.6, 0.9, 20); rsd = np.sqrt(1 - lam**2)
def gen_alpha(rng, reps):
    th = rng.standard_normal((reps, 1000, 1)); e = rng.standard_normal((reps, 1000, 20))
    return th * lam + e * rsd
def gen_crt(rng, reps):
    J, m, icc = 100, 30, 0.15
    trt = np.zeros((1, J, 1)); trt[:, J // 2:, :] = 1
    return 0.2 * trt + rng.standard_normal((reps, J, 1)) * icc**.5 + rng.standard_normal((reps, J, m)) * (1 - icc)**.5

offsets = [0.0, 10.0, 50.0, 100.0, 500.0]
out["E1_reliability"] = controlled(
    lambda m: congeneric_reliability_design(sample_sizes=[1000], item_counts=[20], variance_method=m),
    {"n": 1000, "items": 20}, gen_alpha, 500, offsets)
out["E1_cluster"] = controlled(
    lambda m: cluster_randomized_trial_design(cluster_counts=[100], cluster_sizes=[30], variance_method=m),
    {"clusters": 100, "cluster_size": 30, "icc": 0.15, "treatment_effect": 0.2}, gen_crt, 500, offsets)

# ---------------- E2 ----------------
def adaptive(design, **kw):
    cfg = RunConfig(backend="numpy", precision="float64", batch_size=500, seed=20261004, **kw)
    t = time.perf_counter(); rep = SimulationRunner(cfg).run(design); el = time.perf_counter() - t
    return rep, el

E2 = {}
for label, design, kw in [
    ("cluster_trial", cluster_randomized_trial_design(cluster_counts=(40, 100), cluster_sizes=(20, 30)),
     dict(min_reps=2000, max_reps=20000, target_mcse=0.001, target_coverage_mcse=0.0025, stopping_rule="all")),
    ("reliability", congeneric_reliability_design(sample_sizes=(250, 1000), item_counts=(10, 20)),
     dict(min_reps=1000, max_reps=20000, target_mcse=0.0001, stopping_rule="all")),
]:
    rep, el = adaptive(design, **kw)
    fixed_rep, fixed_el = adaptive(design, **{**kw, "min_reps": kw["max_reps"], "target_mcse": None,
                                              "target_coverage_mcse": None})
    rows = []
    for c, f in zip(rep.conditions, fixed_rep.conditions):
        d = c.as_dict() if hasattr(c, "as_dict") else c.__dict__
        rows.append({"condition": c.condition, "valid_reps": c.valid_reps, "batches": c.batches,
                     "mcse": c.mcse if hasattr(c, "mcse") else None,
                     "coverage": c.coverage, "coverage_mcse": c.coverage_mcse,
                     "stop_reason": c.stop_reason, "bias": c.bias, "rmse": c.rmse,
                     "fixed_valid_reps": f.valid_reps, "fixed_bias": f.bias, "fixed_rmse": f.rmse,
                     "fixed_coverage": f.coverage})
    E2[label] = {"settings": kw, "conditions": rows,
                 "adaptive_total_reps": sum(r["valid_reps"] for r in rows),
                 "fixed_total_reps": sum(r["fixed_valid_reps"] for r in rows),
                 "adaptive_seconds_this_machine": el, "fixed_seconds_this_machine": fixed_el}
out["E2"] = E2

# ---------------- E3 ----------------
class Interrupt(Exception): pass
def crashing(design, crash_after):
    calls = {"n": 0}
    def est(data, ctx, cond):
        calls["n"] += 1
        if calls["n"] > crash_after: raise Interrupt()
        return design.estimator(data, ctx, cond)
    return SimulationDesign(name=design.name, description=design.description, conditions=design.conditions,
                            generator=design.generator, estimator=est, truth=design.truth)
E3 = []
for every, crash in [(1, 7), (3, 7), (5, 12)]:
    base = cluster_randomized_trial_design(cluster_counts=[100], cluster_sizes=[30])
    common = dict(backend="numpy", precision="float32", batch_size=500, min_reps=10000, max_reps=10000,
                  target_mcse=None, seed=77)
    ref = SimulationRunner(RunConfig(**common)).run(base).conditions[0]
    with tempfile.TemporaryDirectory() as d:
        cfg = RunConfig(**common, checkpoint_dir=d, checkpoint_every_batches=every)
        try:
            SimulationRunner(cfg).run(crashing(base, crash))
        except Interrupt:
            pass
        res = SimulationRunner(cfg).run(base).conditions[0]
    E3.append({"checkpoint_every_batches": every, "crash_after_batches": crash,
               "identical_estimate_mean": res.estimate_mean == ref.estimate_mean,
               "identical_rmse": res.rmse == ref.rmse,
               "identical_coverage": res.coverage == ref.coverage,
               "attempted": res.attempted_reps, "reference_attempted": ref.attempted_reps})
out["E3"] = E3

# ---------------- E4 ----------------
S = np.outer(lam, lam) + np.diag(rsd**2); k = 20
alpha_pop = k / (k - 1) * (1 - np.trace(S) / S.sum())
L = np.linalg.cholesky(S); rng = np.random.default_rng(11); est = []
for _ in range(40):
    X = rng.standard_normal((1000, 1000, k)) @ L.T
    est.append(k / (k - 1) * (1 - X.var(1, ddof=1).sum(1) / X.sum(2).var(1, ddof=1)))
est = np.concatenate(est)
out["E4"] = {"z_interval_expected_coverage_df98": float(1 - 2 * stats.t.sf(1.959964, 98)),
             "t_critical_df98": float(stats.t.ppf(0.975, 98)),
             "alpha_population": float(alpha_pop),
             "alpha_float64_bias": float(est.mean() - alpha_pop),
             "alpha_float64_bias_mcse": float(est.std(ddof=1) / math.sqrt(est.size)),
             "alpha_float64_reps": int(est.size)}
json.dump(out, open("supplementary_experiments.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
