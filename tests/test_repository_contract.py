from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "music_critic"

REQUIRED_DOCS = {
    "IMPLEMENTATION_PLAN.md",
    "ARCHITECTURE.md",
    "DATA_CONTRACT.md",
    "LEGACY_REFERENCE.md",
    "DECISIONS.md",
    "ROADMAP.md",
    "STATUS.md",
    "HOOKTHEORY_MIGRATION.md",
    "CANONICAL_MIDI_RENDERER.md",
    "legacy_snapshot.json",
}

REQUIRED_PACKAGES = {
    "data",
    "graph",
    "models",
    "ssl",
    "tasks",
    "training",
    "inference",
    "evaluation",
    "adapters",
    "exporters",
}


def test_required_documentation_exists() -> None:
    docs_root = REPO_ROOT / "docs"
    assert REQUIRED_DOCS <= {path.name for path in docs_root.iterdir()}
    for name in REQUIRED_DOCS - {"legacy_snapshot.json"}:
        text = (docs_root / name).read_text(encoding="utf-8")
        assert text.strip()

    plan = (docs_root / "IMPLEMENTATION_PLAN.md").read_text(encoding="utf-8")
    assert "Provenance" in plan
    assert "Music Critic V2" in plan or "BLOCKED" in plan


def test_required_package_structure_exists() -> None:
    assert (PACKAGE_ROOT / "__init__.py").is_file()
    for name in REQUIRED_PACKAGES:
        assert (PACKAGE_ROOT / name / "__init__.py").is_file()


def test_package_has_no_legacy_or_heavy_imports() -> None:
    forbidden_roots = {
        "src",
        "torch",
        "torch_geometric",
        "hydra",
        "mido",
        "pretty_midi",
        "partitura",
    }
    forbidden_text = {
        "Fine-tune-text2midi-llm-with-gnn-theory-critic",
        "/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic",
    }

    for path in PACKAGE_ROOT.rglob("*.py"):
        relative_parts = path.relative_to(PACKAGE_ROOT).parts
        allowed_roots = set()
        if relative_parts[0] in {"adapters", "exporters"}:
            allowed_roots.add("mido")
        if relative_parts[0] == "graph":
            allowed_roots.update({"torch", "torch_geometric"})
        disallowed_roots = forbidden_roots - allowed_roots
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                assert not (roots & disallowed_roots), (
                    f"{path}: {roots & disallowed_roots}"
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                assert root not in disallowed_roots, f"{path}: {node.module}"
        for token in forbidden_text:
            assert token not in source, f"{path} contains legacy path {token}"


def test_legacy_snapshot_contract() -> None:
    snapshot = json.loads(
        (REPO_ROOT / "docs" / "legacy_snapshot.json").read_text(encoding="utf-8")
    )
    required = {
        "legacy_path",
        "head_commit",
        "branch",
        "remote_urls",
        "status_porcelain_before",
        "python_version",
        "captured_at_utc",
    }
    assert required <= snapshot.keys()
    assert isinstance(snapshot["status_porcelain_before"], list)


def test_agents_and_gitignore_safety_rules() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8").lower()
    assert "legacy repository is read-only" in agents
    assert "never import legacy modules" in agents

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "datasets/", "outputs/", "checkpoints/", "*.pt"):
        assert pattern in gitignore
