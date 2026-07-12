"""Unit tests for bridge helpers that don't need a running model."""

from pathlib import Path

from jlens_gguf.server import safe_static_path


def test_safe_static_path_allows_files_inside(tmp_path):
    (tmp_path / "app.js").write_text("//")
    sub = tmp_path / "vendor"
    sub.mkdir()
    (sub / "d3.min.js").write_text("//")
    assert safe_static_path("app.js", tmp_path) == (tmp_path / "app.js").resolve()
    assert safe_static_path("vendor/d3.min.js", tmp_path) == (sub / "d3.min.js").resolve()
    assert safe_static_path("", tmp_path) == tmp_path.resolve()  # -> index.html by caller


def test_safe_static_path_blocks_traversal(tmp_path):
    root = tmp_path / "web"
    root.mkdir()
    (tmp_path / "secret.py").write_text("password = 1")
    assert safe_static_path("../secret.py", root) is None
    assert safe_static_path("../../etc/passwd", root) is None


def test_safe_static_path_blocks_prefixed_sibling(tmp_path):
    """The bug the fix closes: a sibling dir whose name is prefixed by the web
    dir (e.g. `web-backup`) must not be reachable via `..`."""
    root = tmp_path / "web"
    root.mkdir()
    sibling = tmp_path / "web-backup"
    sibling.mkdir()
    (sibling / "leak.txt").write_text("secret")
    # a naive str.startswith(root) guard would allow this; the boundary check must not
    assert safe_static_path("../web-backup/leak.txt", root) is None
