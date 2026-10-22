"""Regression tests use real temporary Git histories, not mocked revision checks."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "frozen_runner", REPO / "benchmarks/run_frozen_protocol.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FrozenProtocolTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root).decode().strip()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture revision")
        return self.git("rev-parse", "HEAD")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Protocol tests")
        self.git("config", "user.email", "protocol@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.config = self.root / runner.PROTOCOL_CONFIG
        self.write(runner.PROTOCOL_CONFIG, json.dumps({
            "protocol_tag": runner.PROTOCOL_TAG, "protocol_id": "fixture",
        }) + "\n")
        self.write("src/montepilot/__init__.py", '__version__ = "fixture"\n')
        self.write("benchmarks/calculation.py", "# frozen calculation\n")
        self.write(runner.LAUNCHER, Path(runner.__file__).read_text())
        self.write("benchmarks/verify_checksums.py", (REPO / "benchmarks/verify_checksums.py").read_text())
        self.write("PROTOCOL.md", "# fixed design\n")
        self.write("pyproject.toml", "# fixed dependencies\n")
        self.frozen = self.commit()
        self.git("tag", "-a", runner.PROTOCOL_TAG, "-m", "fixture protocol")
        self.addCleanup(patch.stopall)
        patch.object(runner, "ROOT", self.root).start()
        patch.object(runner, "PROTOCOL_COMMIT", self.frozen).start()
        self.snapshot = Path(self.temporary.name) / "snapshot"
        self.snapshot.mkdir()

    def prepare(self, *, smoke=False, config=None):
        return runner.prepare_execution(config or self.config, self.snapshot, smoke=smoke)

    def test_documentation_and_launcher_commit_recorded_separately(self):
        self.write("RESULT_PROVENANCE.md", "Documentation-only successor\n")
        self.write(runner.LAUNCHER, Path(runner.__file__).read_text() + "# later orchestration\n")
        head = self.commit()
        with patch.object(runner, "__file__", str(self.root / runner.LAUNCHER)):
            identity, archive = self.prepare()
        self.assertEqual(identity["executing_commit"], head)
        self.assertEqual(identity["execution_source_commit"], self.frozen)
        self.assertNotEqual(head, self.frozen)
        self.assertEqual(identity["config_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertEqual(identity["source_archive_sha256"], hashlib.sha256(archive).hexdigest())
        self.assertEqual((self.snapshot / runner.LAUNCHER).read_bytes(), Path(runner.__file__).read_bytes())
        self.assertNotEqual((self.snapshot / runner.LAUNCHER).read_bytes(), (self.root / runner.LAUNCHER).read_bytes())
        self.assertTrue(identity["launcher_matches_commit"])
        for name, digest in identity["source_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((self.snapshot / name).read_bytes()).hexdigest())

    def test_missing_tag_rejected(self):
        self.git("tag", "-d", runner.PROTOCOL_TAG)
        with self.assertRaisesRegex(SystemExit, "Git provenance check failed"):
            self.prepare()

    def test_tag_at_wrong_commit_rejected(self):
        self.write("notes.md", "later revision\n")
        self.commit()
        self.git("tag", "-fa", runner.PROTOCOL_TAG, "-m", "moved")
        with self.assertRaisesRegex(SystemExit, "expected " + self.frozen):
            self.prepare()

    def test_lightweight_tag_rejected(self):
        self.git("tag", "-d", runner.PROTOCOL_TAG)
        self.git("tag", runner.PROTOCOL_TAG)
        with self.assertRaisesRegex(SystemExit, "annotated tag"):
            self.prepare()

    def test_supplied_tag_cannot_select_other_protocol(self):
        self.config.write_text('{"protocol_tag": "HEAD"}', encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "Expected protocol tag"):
            self.prepare(smoke=True)

    def test_committed_calculation_drift_rejected(self):
        self.write("src/montepilot/__init__.py", "# changed calculation\n")
        self.commit()
        with self.assertRaisesRegex(SystemExit, "Source differs.*src/montepilot/__init__.py"):
            self.prepare()

    def test_uncommitted_calculation_drift_rejected_in_smoke(self):
        self.write("benchmarks/calculation.py", "# changed\n")
        with self.assertRaisesRegex(SystemExit, "Source differs.*benchmarks/calculation.py"):
            self.prepare(smoke=True)

    def test_full_run_requires_clean_tracked_tree(self):
        self.write(runner.LAUNCHER, "# uncommitted orchestration\n")
        with self.assertRaisesRegex(SystemExit, "clean tracked working tree"):
            self.prepare()

    def test_launcher_drift_hidden_from_git_status_rejected(self):
        self.git("update-index", "--assume-unchanged", runner.LAUNCHER)
        self.write(runner.LAUNCHER, "# hidden local edit\n")
        # Match __file__ to the edited fixture to model an edited executable script.
        with patch.object(runner, "__file__", str(self.root / runner.LAUNCHER)):
            with self.assertRaisesRegex(SystemExit, "launcher bytes matching"):
                self.prepare()

    def test_deleted_source_rejected(self):
        (self.root / "benchmarks/calculation.py").unlink()
        with self.assertRaisesRegex(SystemExit, "Source differs"):
            self.prepare(smoke=True)

    def test_untracked_source_rejected(self):
        self.write("src/montepilot/shadow.py", "# untracked\n")
        with self.assertRaisesRegex(SystemExit, "Source differs.*shadow.py"):
            self.prepare()

    def test_protocol_file_edit_rejected(self):
        self.write("PROTOCOL.md", "# revised design\n")
        self.commit()
        with self.assertRaisesRegex(SystemExit, "Source differs.*PROTOCOL.md"):
            self.prepare()

    def test_modified_external_config_rejected(self):
        external = Path(self.temporary.name) / "custom.json"
        config = json.loads(self.config.read_bytes())
        config["master_seeds"] = [1]
        external.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "configuration differs"):
            self.prepare(config=external)

    def test_format_only_config_change_rejected(self):
        self.config.write_bytes(self.config.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(SystemExit, "configuration differs"):
            self.prepare(smoke=True)

    def test_identical_external_config_and_generated_cache_allowed(self):
        external = Path(self.temporary.name) / "exact.json"
        external.write_bytes(self.config.read_bytes())
        self.write("src/montepilot/__pycache__/cached.pyc", "ignored cache")
        self.write("src/montepilot.egg-info/PKG-INFO", "generated package metadata")
        identity, _ = self.prepare(config=external)
        self.assertEqual(identity["config_sha256"], identity["frozen_config_sha256"])

    def test_snapshot_isolated_from_later_working_tree_edits(self):
        self.prepare()
        self.write("src/montepilot/__init__.py", "raise RuntimeError('working tree imported')\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.snapshot / "src")
        result = subprocess.check_output(
            [sys.executable, "-c", "import montepilot; print(montepilot.__version__)"],
            cwd=self.snapshot, env=env, text=True,
        )
        self.assertEqual(result.strip(), "fixture")

    def test_failed_execution_seals_failed_identity_and_logs(self):
        output = Path(self.temporary.name) / "failed"
        with patch.object(sys, "argv", ["runner", "--phase", "laptop", "--output-dir", str(output)]):
            with patch.object(runner, "execute", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    runner.main()
        identity = json.loads((output / "run_identity.json").read_bytes())
        self.assertEqual(identity["status"], "failed")
        self.assertIn("fixture failure", identity["error"])
        self.assertTrue((output / "SHA256SUMS.txt").is_file())
        self.assertFalse(output.with_suffix(".zip").exists())

    def test_completed_run_archives_and_checksums_identity_and_exact_inputs(self):
        output = Path(self.temporary.name) / "completed"
        with patch.object(sys, "argv", ["runner", "--phase", "laptop", "--output-dir", str(output)]):
            with patch.object(runner, "execute"):
                runner.main()
        identity = json.loads((output / "run_identity.json").read_bytes())
        self.assertEqual(identity["status"], "completed")
        self.assertEqual(identity["evidence_class"], "full-protocol-execution")
        self.assertEqual((output / "protocol_config.json").read_bytes(), self.config.read_bytes())
        for filename, key in (("frozen_source.zip", "source_archive_sha256"), ("launcher.py", "launcher_sha256")):
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), identity[key])
        manifest = (output / "SHA256SUMS.txt").read_text()
        for filename in ("run_identity.json", "protocol_config.json", "launcher.py", "frozen_source.zip"):
            self.assertIn("  " + filename, manifest)
        with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
            self.assertEqual(archive.read("completed/run_identity.json"), (output / "run_identity.json").read_bytes())

    def test_full_a100_commands_keep_frozen_settings_and_snapshot_imports(self):
        identity, _ = self.prepare()
        config = json.loads((REPO / runner.PROTOCOL_CONFIG).read_bytes())
        output = Path(self.temporary.name) / "commands"
        output.mkdir()
        calls = []

        def logged(command, log_path, *, cwd, env):
            calls.append(command)
            self.assertEqual(cwd, self.snapshot)
            self.assertEqual(env["PYTHONPATH"], str(self.snapshot / "src"))
            if command[1] == "-c":
                return json.dumps({"module": str(self.snapshot / "src/montepilot/__init__.py"), "version": "fixture"})
            if command[1] == "benchmarks/capture_environment.py":
                (output / "environment.json").write_text("{}", encoding="utf-8")
            return ""

        with patch.object(runner, "run_logged", side_effect=logged):
            with patch.dict(os.environ, {"PYTHONPATH": "unrelated-installed-source"}):
                runner.execute(SimpleNamespace(phase="a100", smoke=False), config, output, self.snapshot, identity)
        self.assertEqual(len(calls), 7)  # probe, environment, IRLS, two timing + two validation
        for command in calls[2:]:
            self.assertNotIn("--smoke", command)
            if "--config" in command:
                self.assertEqual(command[command.index("--config") + 1], str(self.snapshot / runner.PROTOCOL_CONFIG))
            if "--protocol-config" in command:
                self.assertEqual(command[command.index("--protocol-config") + 1], str(self.snapshot / runner.PROTOCOL_CONFIG))
            if "--seeds" in command:
                self.assertEqual(command[command.index("--seeds") + 1], "5")
            if "--repeats" in command:
                self.assertEqual(command[command.index("--repeats") + 1], "5")
                self.assertEqual(command[command.index("--warmups") + 1], "1")
                self.assertEqual(command[command.index("--min-block-seconds") + 1], "0.5")
                self.assertEqual(command[command.index("--max-inner-iterations") + 1], "512")
                self.assertEqual(command[command.index("--cold-start-repeats") + 1], "5")
                self.assertEqual(command[command.index("--backends") + 1:command.index("--output-dir")], config["a100"]["backends"])
        self.assertEqual([c[c.index("--precision") + 1] for c in calls if "--precision" in c], ["float32", "float32", "float64", "float64"])

    def test_output_reuse_rejected_before_execution(self):
        output = Path(self.temporary.name) / "already_used"
        output.mkdir()
        with patch.object(sys, "argv", ["runner", "--phase", "laptop", "--output-dir", str(output)]):
            with self.assertRaisesRegex(SystemExit, "new output directory"):
                runner.main()


if __name__ == "__main__":
    unittest.main()
