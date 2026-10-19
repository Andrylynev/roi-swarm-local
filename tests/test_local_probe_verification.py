"""The live probe must not silently normalize an incorrect filesystem result."""

from local.probe_local_swarm import inspect_files


def test_probe_rejects_extra_trailing_newline(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"ALPHA\n")
    result = inspect_files(tmp_path, {"a.txt": "ALPHA"})
    assert result["verified"] is False
    assert result["actual"]["a.txt"] == "ALPHA\n"
    assert result["actual_hex"]["a.txt"] == "414c5048410a"


def test_probe_accepts_only_exact_expected_bytes(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"ALPHA")
    assert inspect_files(tmp_path, {"a.txt": "ALPHA"})["verified"] is True
    assert inspect_files(tmp_path, {"b.txt": "BETA"})["verified"] is False
