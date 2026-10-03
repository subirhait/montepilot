"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

from . import __version__
from .backends import _torch, available_backends, torch_installed
from .config import RunConfig
from .engine import SimulationRunner
from .examples import normal_mean_design


def doctor() -> dict:
    report = {
        "montepilot": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch_installed": torch_installed(),
        "available_backends": available_backends(),
        "precision_modes": ["auto", "float32", "float64"],
    }
    if torch_installed():
        torch = _torch()
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(
            getattr(torch, "cuda", None) and torch.cuda.is_available()
        )
        xpu = getattr(torch, "xpu", None)
        report["xpu_available"] = bool(xpu and xpu.is_available())
        if report["xpu_available"]:
            report["xpu_devices"] = [
                xpu.get_device_name(index) for index in range(xpu.device_count())
            ]
            report["torch_xpu_default_dtype"] = "float32"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="montepilot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="print environment and backend diagnostics")
    demo = sub.add_parser("demo", help="run the built-in normal-mean simulation")
    demo.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "numpy", "torch_cpu", "torch_cuda", "torch_xpu"],
    )
    demo.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "float32", "float64"],
    )
    demo.add_argument("--batch-size", type=int, default=512)
    demo.add_argument("--min-reps", type=int, default=1000)
    demo.add_argument("--max-reps", type=int, default=10000)
    demo.add_argument("--target-mcse", type=float, default=0.005)
    demo.add_argument("--target-coverage-mcse", type=float)
    demo.add_argument("--stopping-rule", choices=["all", "any"], default="all")
    demo.add_argument("--seed", type=int, default=20261002)
    demo.add_argument("--checkpoint-dir", type=str)
    demo.add_argument("--auto-expected-reps", type=int)
    demo.add_argument("--output", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2, sort_keys=True))
        return 0
    config = RunConfig(
        backend=args.backend,
        precision=args.precision,
        batch_size=args.batch_size,
        min_reps=args.min_reps,
        max_reps=args.max_reps,
        target_mcse=args.target_mcse,
        target_coverage_mcse=args.target_coverage_mcse,
        stopping_rule=args.stopping_rule,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        auto_expected_reps=args.auto_expected_reps,
    )
    report = SimulationRunner(config).run(normal_mean_design())
    rendered = report.to_json(indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
