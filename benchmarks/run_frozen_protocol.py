"""Run the prespecified laptop or A100 evidence collection workflow."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_TAG = "protocol-v0.6.2"
PROTOCOL_COMMIT = "8fd8a3eb61e24fc66f73d4e704e5b699f72ac5cd"
PROTOCOL_CONFIG = "benchmarks/protocol_v062.json"
LAUNCHER = "benchmarks/run_frozen_protocol.py"


def git_bytes(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Git provenance check failed: git {' '.join(arguments)}") from exc
    return result.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protected_path(name: str) -> bool:
    # Only orchestration may evolve after freezing. All calculation modules,
    # benchmark scripts, settings and dependency declarations stay frozen.
    return (
        name.startswith("src/")
        or (name.startswith("benchmarks/") and name != LAUNCHER)
        or name in {"PROTOCOL.md", "pyproject.toml", "requirements-research.txt"}
    )


def source_files(root: Path) -> set[str]:
    names = set()
    for directory in ("src", "benchmarks"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if any(part == "__pycache__" or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if protected_path(relative.as_posix()):
                names.add(relative.as_posix())
    return names


def prepare_execution(config_path: Path, snapshot: Path, *, smoke: bool) -> tuple[dict, bytes]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    protocol_commit = require_protocol_tag(config)
    executing_commit = git_bytes("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    status = git_bytes("status", "--porcelain", "--untracked-files=all").decode()
    if not smoke and git_bytes("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("Full runs require a clean tracked working tree; commit the launcher first.")

    launcher_bytes = Path(__file__).read_bytes()
    launcher_matches_commit = launcher_bytes == git_bytes("show", f"{executing_commit}:{LAUNCHER}")
    if not smoke and not launcher_matches_commit:
        raise SystemExit("Full runs require launcher bytes matching the executing commit.")

    archive = git_bytes("archive", "--format=zip", protocol_commit)
    with zipfile.ZipFile(io.BytesIO(archive)) as stream:
        stream.extractall(snapshot)
    frozen_bytes = (snapshot / PROTOCOL_CONFIG).read_bytes()
    if config_bytes != frozen_bytes:
        raise SystemExit("Protocol configuration differs from the frozen Git blob (including bytes).")

    expected = {
        name for name in git_bytes("ls-tree", "-r", "--name-only", protocol_commit).decode().splitlines()
        if protected_path(name)
    }
    drift = sorted(source_files(ROOT) - expected)
    for name in sorted(expected):
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != (snapshot / name).read_bytes():
            drift.append(name)
    if drift:
        raise SystemExit("Source differs from the frozen revision: " + ", ".join(sorted(set(drift))))

    identity = {
        "schema_version": "2.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol_id": config["protocol_id"],
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": protocol_commit,
        "executing_commit": executing_commit,
        "execution_source_commit": protocol_commit,
        "execution_source_tree": git_bytes("rev-parse", f"{protocol_commit}^{{tree}}").decode().strip(),
        "execution_mode": "verified-tag-snapshot",
        "launcher_sha256": sha256(launcher_bytes),
        "launcher_matches_commit": launcher_matches_commit,
        "launcher_git_status": status,
        "supplied_config_path": str(config_path),
        "config_sha256": sha256(config_bytes),
        "frozen_config_sha256": sha256(frozen_bytes),
        "source_archive_sha256": sha256(archive),
        "source_sha256": {
            name: sha256((snapshot / name).read_bytes()) for name in sorted(expected)
        },
        "smoke": smoke,
        "evidence_class": "smoke-diagnostic" if smoke else "full-protocol-execution",
    }
    return identity, archive


def run_logged(
    command: list[str], log_path: Path, *, cwd: Path = ROOT, env: dict | None = None,
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}; inspect {log_path}"
        )

    return completed.stdout


def require_protocol_tag(config: dict) -> str:
    if config.get("protocol_tag") != PROTOCOL_TAG:
        raise SystemExit(f"Expected protocol tag {PROTOCOL_TAG!r}.")
    ref = f"refs/tags/{PROTOCOL_TAG}"
    if git_bytes("cat-file", "-t", ref).decode().strip() != "tag":
        raise SystemExit(f"{PROTOCOL_TAG} must be the original annotated tag.")
    commit = git_bytes("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    if commit != PROTOCOL_COMMIT:
        raise SystemExit(f"Protocol tag resolves to {commit}; expected {PROTOCOL_COMMIT}.")
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


def execute(args, config: dict, output: Path, snapshot: Path, identity: dict) -> None:
    config_path = snapshot / PROTOCOL_CONFIG
    env = os.environ.copy()
    # Inherited PYTHONPATH or an installed MontePilot must not supply calculation code.
    # This also propagates to fresh-process benchmark workers.
    env["PYTHONPATH"] = str(snapshot / "src")

    def launch(command: list[str], log_path: Path) -> str:
        return run_logged(command, log_path, cwd=snapshot, env=env)

    probe = launch(
        [sys.executable, "-c", "import json, montepilot; "
         "print(json.dumps({'module': montepilot.__file__, 'version': montepilot.__version__}))"],
        output / "logs" / "source_import.log",
    )
    imported = json.loads(probe)
    if Path(imported["module"]).resolve() != (snapshot / "src/montepilot/__init__.py").resolve():
        raise SystemExit("MontePilot import did not resolve to the frozen source snapshot.")
    identity["imported_montepilot"] = imported
    launch(
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

    environment_path = output / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["execution_provenance"] = {
        key: identity[key] for key in (
            "executing_commit", "execution_source_commit", "config_sha256", "execution_mode",
        )
    }
    environment["montepilot_import"] = imported
    environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    launch(irls_command, output / "logs" / "logistic_irls.log")

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
            launch(
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
            launch(
                validation_command,
                output / "logs" / f"controlled_generic_{precision}.log",
            )


def write_identity(output: Path, identity: dict) -> None:
    (output / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["laptop", "a100"], required=True)
    parser.add_argument("--config", default=PROTOCOL_CONFIG)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    default_name = f"protocol_v062_{args.phase}_{'smoke' if args.smoke else 'full'}"
    output = (ROOT / (args.output_dir or default_name)).resolve()
    if output.exists() or output.with_suffix(".zip").exists():
        raise SystemExit("Use a new output directory and archive path; previous evidence cannot be overwritten.")
    with tempfile.TemporaryDirectory(prefix="montepilot-frozen-") as temporary:
        snapshot = Path(temporary)
        identity, archive_bytes = prepare_execution(config_path, snapshot, smoke=bool(args.smoke))
        identity["phase"] = args.phase
        launcher_bytes = Path(__file__).read_bytes()
        if sha256(launcher_bytes) != identity["launcher_sha256"]:
            raise SystemExit("Launcher changed during preflight; retry with a stable checkout.")
        output.mkdir(parents=True)
        (output / "frozen_source.zip").write_bytes(archive_bytes)
        (output / "protocol_config.json").write_bytes((snapshot / PROTOCOL_CONFIG).read_bytes())
        (output / "launcher.py").write_bytes(launcher_bytes)
        write_identity(output, identity)
        try:
            execute(args, json.loads((snapshot / PROTOCOL_CONFIG).read_bytes()), output, snapshot, identity)
            # Detect accidental edits to the executing snapshot before sealing the bundle.
            for name, digest in identity["source_sha256"].items():
                if sha256((snapshot / name).read_bytes()) != digest:
                    raise RuntimeError(f"Executing snapshot changed: {name}")
        except BaseException as exc:
            identity["status"] = "failed"
            identity["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            identity["status"] = "completed"
        finally:
            identity["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_identity(output, identity)
            write_checksums(output)
        # Verify the sealed directory before making its archive.
        verification = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks/verify_checksums.py"), str(output)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        print(verification.stdout.strip())
        result_archive = make_zip(output)
    print(f"Results: {output}")
    print(f"Archive: {result_archive}")
    print(f"Verify: {sys.executable} benchmarks/verify_checksums.py {output}")


if __name__ == "__main__":
    main()
