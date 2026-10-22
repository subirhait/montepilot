# Executing and auditing the frozen v0.6.2 protocol

The statistical design remains frozen at annotated tag `protocol-v0.6.2`,
commit `8fd8a3eb61e24fc66f73d4e704e5b699f72ac5cd`. Do not move or recreate
that tag. `PROTOCOL.md`, `benchmarks/protocol_v062.json`, calculation modules,
and benchmark scripts remain unchanged. The later launcher adds provenance
checks and source isolation; this is an execution safeguard, not a new design.

## Obtain a reviewed launcher revision

Use a fresh clone with byte-preserving checkout on both Windows and Linux:

```text
git clone -c core.autocrlf=false https://github.com/subirhait/montepilot.git
cd montepilot
git fetch origin --tags
git checkout --detach <reviewed-launcher-commit>
git rev-parse HEAD
git cat-file -t refs/tags/protocol-v0.6.2
git rev-list -n 1 refs/tags/protocol-v0.6.2
git status --short
```

Replace `<reviewed-launcher-commit>` with the full SHA of the reviewed commit
containing the hardened `run_frozen_protocol.py` (or its reviewed merge commit).
Record it in the research log. The tag type must be `tag` and its resolved
commit must be the frozen SHA above. The launcher checks both automatically.
Do not check out the old tag to launch experiments: its runner predates these
safeguards. A later documentation/launcher commit is allowed and recorded
separately, while benchmark calculations execute from the original commit.

Full runs require a clean tracked working tree. Modified, deleted, or added
source/configuration files in `src/` or `benchmarks/` are rejected, as are
changes to `PROTOCOL.md`, `pyproject.toml`, and `requirements-research.txt`.
The launcher itself may change in a committed revision; its exact bytes and
SHA-256 are retained, and full execution verifies those bytes against HEAD
even if Git status flags hide an edit. Generated Python caches and package metadata are ignored.
Smoke runs allow a dirty launcher/docs/tests tree, record its status, and still
require the frozen source and configuration. Untracked calculation files are
rejected in either mode.

An alternate `--config` path is accepted only if its bytes equal the frozen
JSON blob. Reformatting or CRLF conversion also fails this check. If an old
Windows checkout has converted line endings, retain your edits separately and
use the fresh clone above. Do not weaken the check or edit the design to make
a run proceed. No uncommitted code changes should be made during execution.

## Install and check the intended hardware

Activate the intended Python environment, install the appropriate PyTorch
build and device drivers for Intel XPU or NVIDIA CUDA, then run:

```text
python -m pip install -e ".[research]"
python -m montepilot.cli doctor
python -m unittest discover -s tests -v
```

The laptop requires `numpy`, `torch_cpu`, and `torch_xpu`; the A100 requires
`numpy`, `torch_cpu`, and `torch_cuda`. Confirm the actual GPU model in the
environment record. A successful test suite or process exit alone does not
establish that all requested hardware or numerical comparisons succeeded.
If a required backend is unavailable, preserve the diagnostic output, fix the
environment, and use a new output directory. Never remove a backend from the
JSON to make a run finish.

## Collect laptop and A100 results

Run the laptop smoke test on the Intel laptop, inspect its logs and all
reported availability/numerical failures, then run the full phase:

```text
python benchmarks/run_frozen_protocol.py --phase laptop --smoke --output-dir protocol_v062_laptop_smoke
python benchmarks/run_frozen_protocol.py --phase laptop --output-dir protocol_v062_laptop_full
python benchmarks/verify_checksums.py protocol_v062_laptop_full
```

On the A100, use the same reviewed launcher revision and repeat the checks,
then run:

```text
python benchmarks/run_frozen_protocol.py --phase a100 --smoke --output-dir protocol_v062_a100_smoke
python benchmarks/run_frozen_protocol.py --phase a100 --output-dir protocol_v062_a100_full
python benchmarks/verify_checksums.py protocol_v062_a100_full
```

The frozen settings still specify five master seeds, all three IRLS conditions,
the original convergence/failure definitions and numerical tolerances, laptop
matched FP32, and A100 matched FP32 and FP64. No statistical or timing
calculation has been changed. Smoke mode applies the original reduced-count
settings internally; its configuration checksum identifies the frozen input
file, and `smoke: true` identifies the reduced execution. Smoke outputs are
never manuscript evidence.

Every invocation requires a new output directory and ZIP path, including
retries. Do not resume into or overwrite a previous collection. If a subprocess
fails, the runner leaves logs, a `status: failed` identity, and checksums of the
partial output; it does not create a completed result ZIP. Retain these failed
attempts in the research log.

## What the provenance fields mean

Before timing, the launcher extracts `git archive` of the verified frozen
commit to a temporary directory. All benchmark subprocesses, including fresh
process workers, use that directory and its `src/` through an explicit
`PYTHONPATH`. A probe checks the imported MontePilot package path. The supplied
configuration is checked once; workers receive the frozen snapshot's copy.
Later working-tree edits therefore cannot replace worker calculations. The
snapshot's protected files are hashed again after execution.

Each result directory includes:

| Record | Audit meaning |
|---|---|
| `run_identity.json: protocol_commit` | Pinned commit resolved from the annotated protocol tag |
| `executing_commit` | HEAD of the checkout that launched the orchestration; may be a later reviewed commit |
| `execution_source_commit` / `execution_source_tree` | Frozen benchmark implementation actually supplied to workers |
| `launcher.py` / `launcher_sha256` / `launcher_matches_commit` / `launcher_git_status` | Exact launcher bytes, digest, and checkout status at preflight |
| `protocol_config.json` / `config_sha256` / `frozen_config_sha256` | Retained supplied configuration, byte-identical to the frozen blob |
| `frozen_source.zip` / `source_archive_sha256` / `source_sha256` | Frozen source archive and individual protected-file SHA-256 values |
| `execution_mode`, `smoke`, `evidence_class`, `status` | Snapshot mode, diagnostic/full classification, and execution outcome |
| `environment.json` | Hardware/packages plus the same execution provenance and actual imported package path |
| `logs/` and benchmark reports | Commands, stdout/stderr, raw outcomes, validation results, and failures |
| `SHA256SUMS.txt` | Manifest of all bundle files except the manifest itself |

The archived source is extracted without `.git`. The legacy `environment.json`
`git` probes consequently report that it is not a Git repository. Use
`execution_provenance` and `run_identity.json` for commit attribution. The
recorded temporary import path will cease to exist after the run; its contents
are preserved in `frozen_source.zip`. Package metadata can reflect an installed
MontePilot distribution, while the import path and source hashes identify the
code actually used. Environment packages and drivers are captured, not pinned
by the frozen source tag.

The runner automatically verifies checksums before creating the result ZIP.
After transfer, verify the ZIP's SHA-256 against the value recorded on the
origin machine, extract it, and rerun `verify_checksums.py` on the extracted
directory. For example, use `sha256sum <archive>.zip` on Linux, or PowerShell
`Get-FileHash <archive>.zip -Algorithm SHA256` on Windows. Verify A100 downloads
before terminating the instance.

Audit `run_identity.json` for `status: completed`, `smoke: false`, both pinned
source-commit fields, the recorded launcher revision, and matching configuration
digests. Check that the retained config/source/launcher hashes agree with the
identity and manifest. Inspect every prespecified backend, precision, condition,
metric, failure, and numerical validation row. `status: completed` means the
commands finished, not that every backend or numerical validation passed.
Keep unavailable backends, numerical failures, and unfavorable results visible.

Keep the laptop and A100 provenance rows pending until their full raw outputs,
environment records, source/configuration identities, and verified manifests
have been archived. Then record the two distinct source and launcher revisions,
archive locations and checksums, and any dated deviations. Never attribute
these collections to the launcher commit alone, or relabel older hardware
results as the new frozen evidence. No existing result is presumed invalid
merely because the former launcher lacked these checks.
