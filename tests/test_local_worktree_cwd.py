"""A requested ignored/empty project subfolder must exist inside its worktree."""

import subprocess

from roi_swarm.worktrees import WaveWorkspace


def test_ignored_target_directory_is_recreated_without_copying_live_data(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    folder = root / "data" / "workspace"
    folder.mkdir(parents=True)
    (folder / "private.db").write_text("keep in shared folder", encoding="utf-8")
    workspace = WaveWorkspace.open(str(folder), root=tmp_path / "workers")
    assert workspace is not None
    try:
        workspace.snapshot()
        actual = workspace.prepare("one")
        expected = workspace.root / "one" / "data" / "workspace"
        assert actual == expected
        assert actual.is_dir()
        assert not (actual / "private.db").exists()
        assert workspace.to_worker(str(folder), "one") == str(actual)
    finally:
        workspace.close()
