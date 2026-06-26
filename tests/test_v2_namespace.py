import ast
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1] / "src" / "cupcast" / "v2"


def test_v2_package_imports():
    import cupcast.compare  # noqa: F401
    import cupcast.v2  # noqa: F401


def _imports_v1(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(a.name == "cupcast.v1" or a.name.startswith("cupcast.v1.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "cupcast.v1" or module.startswith("cupcast.v1."):
                return True
            if module == "cupcast" and any(a.name == "v1" for a in node.names):
                return True
    return False


def test_v2_is_clean_room():
    """v2 must never import v1 — the comparison baseline stays honest."""
    offenders = [
        str(path.relative_to(V2_ROOT))
        for path in V2_ROOT.rglob("*.py")
        if _imports_v1(path.read_text())
    ]
    assert offenders == [], f"v2 modules import v1: {offenders}"
