"""post_verify: the declared verifier chains behind the cell command."""

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.run.command import with_post_verify


class _Cell:
    def __init__(self, app):
        self.app = app


def _app(post_verify=None):
    return ResolvedApp(name="x", binary="x", version="base", cls="task",
                       marker="M", scalar_re="M (\\d+)",
                       post_verify=post_verify)


def test_no_hook_returns_argv_unchanged():
    argv = ["./x", "--a", "1"]
    assert with_post_verify(argv, _Cell(_app())) is argv


def test_hook_chains_with_and_and_preserves_quoting():
    argv = ["./x", "a b"]
    out = with_post_verify(argv, _Cell(_app("cmp -s out ref")))
    assert out[:2] == ["sh", "-c"]
    assert "'a b'" in out[2] and "&& { cmp -s out ref ; }" in out[2]
