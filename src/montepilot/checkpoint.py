"""Atomic JSON checkpoints for resumable condition runs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def checkpoint_path(directory: str | Path, design_name: str, condition_index: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", design_name).strip("_") or "design"
    return Path(directory) / f"{safe}.condition-{condition_index:04d}.json"


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

