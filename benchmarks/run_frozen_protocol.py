"""Run the prespecified laptop or A100 evidence collection workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}; inspect {log_path}"
        )


def require_protocol_tag(config: dict) -> str:
    command = ["git", "rev-list", "-n", "1", str(config["protocol_tag"])]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    commit = completed.stdout.strip()
    if completed.returncode or not commit:
        raise SystemExit(
            f"full runs require Git tag {config['protocol_tag']!r}; commit and tag the protocol first"
        )
    return commit


def write_checksums(output: Path) -> None:
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(records) + "\n", encoding="utf-8")


def make_zip(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                stream.write(path, arcname=(output.name + "/" + path.relative_to(output).as_posix()))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["laptop", "a100"], required=True)
    parser.add_argument("--config", default="benchmarks/protocol_v062.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_commit = "smoke-not-tag-enforced" if args.smoke else require_protocol_tag(config)
    default_name = f"protocol_v062_{args.phase}_{'smoke' if args.smoke else 'full'}"
    output = (ROOT / (args.output_dir or default_name)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_identity.json").write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "protocol_tag": config["protocol_tag"],
                "protocol_commit": protocol_commit,
                "phase": args.phase,
                "smoke": bool(args.smoke),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run_logged(
        [
            sys.executable,
            "benchmarks/capture_environment.py",
            "--protocol-config",
            str(config_path),
            "--output",
            str(output / "environment.json"),
        ],
        output / "logs" / "environment.log",
    )

    irls_command = [
        sys.executable,
        "benchmarks/benchmark_logistic_irls.py",
        "--config",
        str(config_path),
        "--phase",
        args.phase,
        "--output-dir",
        str(output / "logistic_irls"),
    ]
    if args.smoke:
        irls_command.append("--smoke")
    run_logged(irls_command, output / "logs" / "logistic_irls.log")

    if args.phase == "a100":
        phase_config = config["a100"]
        for precision in phase_config["precisions"]:
            generic_dir = output / f"generic_{precision}"
            if args.smoke:
                publication_command = [
                    sys.executable,
                    "benchmarks/benchmark_publication.py",
                    "--profile",
                    "quick",
                    "--precision",
                    precision,
                    "--repeats",
                    "2",
                    "--warmups",
                    "0",
                    "--min-block-seconds",
                    "0.01",
                    "--max-inner-iterations",
                    "2",
                    "--cold-start-repeats",
                    "0",
                    "--workloads",
                    "mean_n1000_r5000",
                ]
            else:
                publication_command = [
                    sys.executable,
                    "benchmarks/benchmark_publication.py",
                    "--profile",
                    str(phase_config["generic_profile"]),
                    "--precision",
                    precision,
                    "--repeats",
                    str(config["timing"]["warm_block_repeats"]),
                    "--warmups",
                    str(config["timing"]["warmups"]),
                    "--min-block-seconds",
                    str(config["timing"]["minimum_block_seconds"]),
                    "--max-inner-iterations",
                    str(config["timing"]["maximum_inner_iterations"]),
                    "--cold-start-repeats",
                    str(config["timing"]["fresh_process_repeats"]),
                    "--cold-workload",
                    str(phase_config["cold_workload"]),
                    "--workloads",
                    *phase_config["generic_workloads"],
                ]
            publication_command.extend(
                ["--backends", *phase_config["backends"], "--output-dir", str(generic_dir)]
            )
            run_logged(
                publication_command,
                output / "logs" / f"generic_{precision}.log",
            )

            validation_dir = output / f"controlled_generic_{precision}"
            validation_command = [
                sys.executable,
                "benchmarks/validate_precision.py",
                "--profile",
                "quick" if args.smoke else "full",
                "--precision",
                precision,
                "--seeds",
                "1" if args.smoke else str(len(config["master_seeds"])),
                "--seed",
                str(config["master_seeds"][0]),
                "--seed-step",
                "100000",
                "--backends",
                *phase_config["backends"],
                "--protocol-config",
                str(config_path),
                "--output-dir",
                str(validation_dir),
            ]
            run_logged(
                validation_command,
                output / "logs" / f"controlled_generic_{precision}.log",
            )

    write_checksums(output)
    archive = make_zip(output)
    print(f"Results: {output}")
    print(f"Archive: {archive}")
    print(f"Verify: {sys.executable} benchmarks/verify_checksums.py {output}")


if __name__ == "__main__":
    main()
