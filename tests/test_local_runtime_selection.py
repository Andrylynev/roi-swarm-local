"""Launch only the published Prism runtime, never staging or backup trees."""

import os

import pytest

from local import manage


@pytest.fixture(params=[".", "bin", "archive/bin"])
def published_runtime(tmp_path, request):
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    canonical = tmp_path / "runtime/prism" / request.param / name
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"new published executable")
    canonical.chmod(0o755)
    for directory in (".prism-stage-left.previous", ".prism-stage-left", "other"):
        leftover = tmp_path / "runtime" / directory / name
        leftover.parent.mkdir(parents=True)
        leftover.write_bytes(b"old uncommitted executable")
        leftover.chmod(0o755)
    return canonical


def test_plan_selects_canonical_runtime_over_leftover_backup(tmp_path, published_runtime):
    plan = manage.build_plan(tmp_path)
    assert plan["services"]["inference"]["command"][0] == str(published_runtime)
    published_runtime.unlink()
    missing_plan = manage.build_plan(tmp_path)
    expected = tmp_path / "runtime/prism" / published_runtime.name
    assert missing_plan["services"]["inference"]["command"][0] == str(expected)
    assert str(expected) in missing_plan["missing"]
