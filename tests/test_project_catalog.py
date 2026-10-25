"""Synthetic navigation tests; no musical inputs or model execution."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project_catalog", ROOT / "scripts/build_project_catalog.py"
)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def test_catalog_preserves_each_attempt_and_unknown_families(self):
        records = [
            {"experiment_id": "EXP-HCM2-RETRY-002", "status": "failed",
             "kind": "training", "title": "A | B\nnext"},
            {"experiment_id": "EXP-UNKNOWN-001", "status": "running",
             "kind": "diagnostic", "title": "Unclassified"},
        ]
        result = catalog.render_catalog(records)
        for record in records:
            assert result.count(f"[{record['experiment_id']}]") == 1
        assert "failed" in result and "running" in result
        assert "A &#124; B next" in result
        assert "Требуют классификации" in result
        assert "не проверка живых процессов" in result
        assert result == catalog.render_catalog(list(reversed(records)))


    def test_catalog_covers_repository_registry_without_changing_records(self):
        records = catalog.load_records(ROOT / "docs/experiments/records")
        result = catalog.render_catalog(records)
        assert len(records) > 0
        for record in records:
            assert result.count(f"[{record['experiment_id']}]") == 1

    def test_report_index_covers_each_existing_report_once(self):
        reports = [p.name for p in (ROOT / "docs/experiments").glob("*.md")
                   if p.name != "README.md"]
        result = catalog.render_catalog([], reports)
        assert reports
        for name in reports:
            assert result.count(f"[{name}]({name})") == 1
        assert result == catalog.render_catalog([], list(reversed(reports)))


    def test_check_detects_stale_output_without_overwriting(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        output = tmp_path / "catalog.md"
        output.write_text("stale\n")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_project_catalog.py"),
             "--output", str(output), "--check"], capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert output.read_text() == "stale\n"

if __name__ == "__main__":
    unittest.main()
