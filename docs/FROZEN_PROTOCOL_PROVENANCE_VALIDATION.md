# Frozen-protocol execution safeguard validation

Validation date: October 21, 2026. These checks assess the execution workflow;
they are not manuscript benchmark evidence. Both full hardware collections
remain pending in `RESULT_PROVENANCE.md`.

## Revisions and unchanged design

The local annotated `protocol-v0.6.2` tag resolves to
`8fd8a3eb61e24fc66f73d4e704e5b699f72ac5cd` (tag object
`5f5ca6e5f6559f70ab79535ce0fa2bd7e3169112`). The baseline main revision was
`f9e8e021db69f6ea331f24f92cc48d9d6cc95cd4`; its only difference from the
frozen commit was the provenance-documentation change.

The implementation leaves `PROTOCOL.md`, `benchmarks/protocol_v062.json`,
`src/`, all other benchmark scripts, `pyproject.toml`, and
`requirements-research.txt` byte-identical to the tag. The only executable
change is the orchestration runner. Its benchmark argument construction and
statistical calculations retain the original design.

The frozen configuration SHA-256 is:

```text
e37007313807b002eabde0a7d3afe3bf715395db8c0d34a61cd2b85495e5b237
```

## Test and smoke outcomes

| Check | Outcome |
|---|---|
| Original tagged-source NumPy suite, extracted from `git archive` with its own `src/` | 45 tests passed |
| Updated `python -m unittest discover -s tests -v` | 65 tests passed, including 20 provenance regressions |
| Python compilation and `git diff --check` | Passed |
| Laptop workflow smoke | Completed on NumPy; requested Torch CPU/XPU unavailability retained in reports |
| A100 workflow smoke | Correctly stopped at the generic FP32 command because Torch CPU/CUDA were unavailable; failed identity and partial-output checksums retained |
| Original generic benchmark, CPU diagnostic, matched FP32 and FP64 | Both completed with NumPy, one quick mean workload, two timing blocks, no fresh-process repeats |
| Original controlled-input validation, CPU diagnostic, FP32 and FP64 | Both passed; four comparisons per precision, no execution failures, one seed |
| Result checksum verification | Laptop directory and ZIP extraction passed; partial A100 directory passed |

The regressions use real temporary Git histories. They cover missing,
lightweight, and moved tags; a supplied tag override; committed and
uncommitted calculation drift; deleted and untracked source; modified protocol
text; modified external configuration and format-only JSON drift; an identical
external configuration; generated caches; clean-launcher enforcement including
edits hidden from Git status; later documentation/launcher commits; snapshot
imports after working-tree edits; full A100 command/settings propagation;
completed/failed identities and archived inputs; and output reuse rejection.

The diagnostic environment was Linux x86_64, Python 3.12.14, NumPy 2.3.5,
MontePilot 0.6.2, without Torch, XPU, or CUDA. No GPU performance claim follows
from these checks. The real laptop and A100 smoke tests must be repeated on
the intended machines before their full runs.

## Provenance inspection

The workflow smoke records distinguish the launcher checkout commit from the
frozen `execution_source_commit`. Their configuration SHA-256 values match
the retained configuration, frozen Git blob, and environment record. The
source archive's embedded Git commit and every recorded protected-file digest
match the frozen revision. The archived launcher bytes match their recorded
SHA-256. The imported package path points to the extracted source snapshot.
Smoke runs performed during development record their dirty launcher state;
full execution instead requires a committed matching launcher.

The successful laptop result ZIP was extracted into a separate directory and
its manifest verified again. The A100 attempt carries `status: failed` and
produces no completed ZIP. These diagnostic collections must not replace the
pending full-hardware evidence rows.

See [the execution procedure](FROZEN_PROTOCOL_EXECUTION.md) for hardware setup,
result attribution, transfer checks, complete reporting, and archive rules.
