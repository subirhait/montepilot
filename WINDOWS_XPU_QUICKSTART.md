# Windows Intel XPU quick start

This release adds native `torch_xpu` execution for supported Intel Arc GPUs.

## Upgrade an existing MontePilot environment

Open Anaconda Prompt, activate the existing environment, and enter the newly
extracted project folder:

```bat
conda activate montepilot
cd /d "C:\Users\subir\Downloads\montepilot"
python -m pip install -e .
```

The Intel PyTorch wheel should already be installed. If it is not, use:

```bat
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```

## Verify detection

```bat
montepilot doctor
```

The output should contain `torch_xpu` in `available_backends` and should list
the Intel Arc device under `xpu_devices`.

## Run a short XPU smoke test

```bat
montepilot demo --backend torch_xpu --batch-size 100 --min-reps 200 --max-reps 500 --target-mcse 0.01
```

Version 0.5 also supports matched precision and a separate coverage-MCSE
target:

```bat
montepilot demo --backend auto --precision float32 --batch-size 2000 --min-reps 20000 --max-reps 100000 --target-mcse 0.001 --target-coverage-mcse 0.0015 --stopping-rule all
```

## Benchmark after other long computations finish

```bat
python benchmarks\benchmark_backends.py --reps 10000 --n 1000 --batch-size 1000
```

For repeated publication-oriented benchmarking across mean, OLS, causal-IPW,
and bootstrap workloads:

```bat
python benchmarks\benchmark_suite.py --profile full --output-dir benchmark_results
```

For the calibrated v0.4 protocol, which separates warm-state throughput from
fresh-process latency and lengthens very short XPU timings:

```bat
python benchmarks\benchmark_publication.py --profile full --min-block-seconds 0.5 --output-dir benchmark_results_v04
```

The full calibrated run can take several minutes because every backend and
workload receives multiple timing blocks of at least 0.5 seconds.

With automatic precision, Intel client XPU execution uses FP32 while NumPy,
Torch CPU, and Torch CUDA use FP64. Use `--precision float32` for a matched
FP32 comparison, and always report numerical error together with elapsed time.
