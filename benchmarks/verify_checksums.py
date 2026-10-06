"""Verify a frozen protocol output directory against its SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    args = parser.parse_args()
    root = Path(args.result_directory)
    manifest = root / "SHA256SUMS.txt"
    failures: list[str] = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual != expected:
            failures.append(f"mismatch: {relative}")
    if failures:
        raise SystemExit("checksum verification failed:\n" + "\n".join(failures))
    print(f"Verified {checked} files in {root.resolve()}")


if __name__ == "__main__":
    main()
