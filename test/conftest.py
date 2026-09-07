import sys
from pathlib import Path

import pytest

# Same convention as run_benchmark.py / run_validation.py: src/ modules
# import each other by bare name ("from config import ..."), which only
# resolves if src/ itself is on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import tools  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point tools.GENERATED_DIR at a throwaway dir for the duration of a test.

    Every path-taking tool resolves GENERATED_DIR at call time rather than
    capturing it at import, so patching the module attribute is enough to
    keep tests from touching the real src/generated/ sandbox.
    """
    generated = tmp_path / "generated"
    generated.mkdir()
    monkeypatch.setattr(tools, "GENERATED_DIR", str(generated))
    return generated
