"""Capture the hardware and software environment for a frozen benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def total_memory_bytes() -> int | None:
    try:
        if os.name == "nt":
            result = command_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ]
            )
            return int(result.get("stdout", ""))
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return None


def cpu_model() -> str:
    if os.name == "nt":
        result = command_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
            ]
        )
        if result.get("stdout"):
            return str(result["stdout"]).splitlines()[0].strip()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_metadata() -> dict:
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "protocol_tag_commit": command_output(
            ["git", "rev-list", "-n", "1", "protocol-v0.6.2"]
        ),
        "status": command_output(["git", "status", "--short"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default="benchmarks/protocol_v062.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol_config)
    protocol_bytes = protocol_path.read_bytes()
    torch_details: dict = {"installed": package_version("torch") is not None}
    if torch_details["installed"]:
        try:
            import torch

            torch_details.update(
                {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()),
                    "cuda_devices": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "xpu_available": bool(
                        hasattr(torch, "xpu") and torch.xpu.is_available()
                    ),
                }
            )
        except Exception as exc:
            torch_details["inspection_error"] = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": "1.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_config": str(protocol_path.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "platform": platform.platform(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": cpu_model(),
            "logical_count": os.cpu_count(),
        },
        "memory": {"total_bytes": total_memory_bytes()},
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "packages": {
            name: package_version(name)
            for name in ("montepilot", "numpy", "scipy", "torch")
        },
        "torch": torch_details,
        "nvidia_smi_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_full": command_output(["nvidia-smi"]),
        "git": git_metadata(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
