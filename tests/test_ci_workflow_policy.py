from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = REPO_ROOT / ".github" / "scripts" / "classify-ci-change.sh"
FINGERPRINT = REPO_ROOT / ".github" / "scripts" / "ci-tree-fingerprint.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(repository: Path, relative_path: str, content: str) -> None:
    path = repository / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "ci-policy@example.invalid")
    _git(repository, "config", "user.name", "CI Policy Test")
    _write(repository, "baseline.txt", "baseline\n")
    _git(repository, "add", "baseline.txt")
    _git(repository, "commit", "-q", "-m", "baseline")
    return repository, _git(repository, "rev-parse", "HEAD")


def _classify(
    repository: Path,
    base_sha: str,
    head_sha: str,
) -> tuple[dict[str, str], tuple[str, ...]]:
    changed_files = repository / "changed-files.txt"
    result = subprocess.run(
        ["bash", str(CLASSIFIER), base_sha, head_sha, str(changed_files)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    changed = tuple(changed_files.read_text(encoding="utf-8").splitlines())
    return outputs, changed


def _commit_path(repository: Path, relative_path: str) -> str:
    _write(repository, relative_path, f"changed: {relative_path}\n")
    _git(repository, "add", relative_path)
    _git(repository, "commit", "-q", "-m", f"change {relative_path}")
    return _git(repository, "rev-parse", "HEAD")


def _fingerprint(repository: Path) -> str:
    result = subprocess.run(
        ["bash", str(FINGERPRINT)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", result.stdout.strip())
    return result.stdout.strip()


def test_docs_and_markdown_only_change_is_not_ci_relevant(tmp_path: Path) -> None:
    repository, base_sha = _repository(tmp_path)
    _write(repository, "docs/policy.txt", "documentation\n")
    _write(repository, "README.md", "documentation\n")
    _git(repository, "add", "docs/policy.txt", "README.md")
    _git(repository, "commit", "-q", "-m", "docs only")

    outputs, changed = _classify(
        repository, base_sha, _git(repository, "rev-parse", "HEAD")
    )

    assert outputs["ci_relevant"] == "false"
    assert outputs["change_detection_succeeded"] == "true"
    assert outputs["reason"] == "All changed paths are limited to docs/** or *.md."
    assert changed == ("README.md", "docs/policy.txt")


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/module.py",
        "tests/test_module.py",
        "scripts/tool.sh",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "assets/unknown.bin",
        "README.MD",
    ),
)
def test_every_nonexcluded_path_is_ci_relevant(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, base_sha = _repository(tmp_path)
    head_sha = _commit_path(repository, relative_path)

    outputs, changed = _classify(repository, base_sha, head_sha)

    assert outputs["ci_relevant"] == "true"
    assert outputs["change_detection_succeeded"] == "true"
    assert changed == (relative_path,)


def test_change_detection_failure_and_empty_diff_are_fail_open(tmp_path: Path) -> None:
    repository, head_sha = _repository(tmp_path)

    failed, failed_changed = _classify(repository, "not-a-commit", head_sha)
    empty, empty_changed = _classify(repository, head_sha, head_sha)

    assert failed["ci_relevant"] == "true"
    assert failed["change_detection_succeeded"] == "false"
    assert failed_changed == ()
    assert empty["ci_relevant"] == "true"
    assert empty["change_detection_succeeded"] == "true"
    assert empty_changed == ()


def test_ci_tree_fingerprint_ignores_only_docs_and_markdown(tmp_path: Path) -> None:
    repository, _base_sha = _repository(tmp_path)
    _write(repository, "src/module.py", "value = 1\n")
    _write(repository, "docs/note.txt", "first\n")
    _write(repository, "README.md", "first\n")
    _git(repository, "add", "src/module.py", "docs/note.txt", "README.md")
    _git(repository, "commit", "-q", "-m", "tracked tree")
    original = _fingerprint(repository)

    _write(repository, "docs/note.txt", "second\n")
    _write(repository, "README.md", "second\n")
    _git(repository, "add", "docs/note.txt", "README.md")
    _git(repository, "commit", "-q", "-m", "documentation")
    after_docs = _fingerprint(repository)

    _write(repository, "src/module.py", "value = 2\n")
    _git(repository, "add", "src/module.py")
    _git(repository, "commit", "-q", "-m", "source")

    assert after_docs == original
    assert _fingerprint(repository) != original


def _step(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = workflow.index(marker)
    end = workflow.find("\n      - name: ", start + len(marker))
    return workflow[start:] if end == -1 else workflow[start:end]


def test_workflow_has_one_pr_run_and_push_only_on_main() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "  pull_request:\n" in workflow
    assert "  push:\n    branches:\n      - main\n" in workflow
    assert "paths-ignore" not in workflow
    assert "[skip ci]" not in workflow
    assert "  full-suite:\n    name: full-suite\n" in workflow


def test_docs_only_and_cache_hit_skip_every_python_suite_step() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for name in (
        "Set up Python",
        "Install CPU dependencies",
        "Run full test suite",
        "Compile Python sources",
    ):
        step = _step(workflow, name)
        assert "steps.changes.outputs.ci_relevant == 'true'" in step
        assert "steps.success-cache.outputs.cache-hit != 'true'" in step
        assert "steps.tree.outputs.valid == 'true'" not in step

    restore = _step(workflow, "Find successful full-suite marker")
    create = _step(workflow, "Create successful full-suite marker")
    save = _step(workflow, "Save successful full-suite marker")
    summary = _step(workflow, "Summarize full-suite decision")
    assert "actions/cache/restore@v5" in restore
    assert "lookup-only: true" in restore
    assert "restore-keys" not in restore
    assert "continue-on-error: true" in restore
    assert "actions/cache/save@v5" in save
    assert "success()" in create
    assert "success()" in save
    assert workflow.index("Compile Python sources") < workflow.index(
        "Create successful full-suite marker"
    )
    assert "if: always()" in summary
    assert "### Changed files" in summary
