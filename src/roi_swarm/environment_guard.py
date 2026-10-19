"""Fingerprint the small package metadata surface shared by wave workers."""

import json
import os
from collections.abc import Iterator
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from stat import S_IFDIR, S_IFMT, S_IFREG
from urllib.parse import urlsplit
from urllib.request import url2pathname


def _matching(directory: Path, pattern: str, kind: int) -> Iterator[Path]:
    if "*" not in pattern:
        path = directory / pattern
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            return
        if S_IFMT(mode) == kind:
            yield path
        return
    # Path.glob() suppresses scan OSErrors, which would silently bless an incomplete baseline.
    with os.scandir(directory) as entries:
        for entry in entries:
            if fnmatch(entry.name, pattern) and S_IFMT(entry.stat().st_mode) == kind:
                yield directory / entry.name


def _editable_target(path: Path) -> Path | None:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    directory = metadata.get("dir_info")
    url = metadata.get("url")
    if not isinstance(directory, dict) or directory.get("editable") is not True:
        return None
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme != "file":
        return None
    authority = "" if parsed.netloc.lower() in ("", "localhost") else f"//{parsed.netloc}"
    return Path(url2pathname(authority + parsed.path))


def _broken_target(target: Path) -> bool:
    if not target.is_absolute():
        return False
    if "roi-waves" in {part.casefold() for part in target.parts}:
        return True
    try:
        target.stat()
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return True
    return "roi-waves" in {part.casefold() for part in resolved.parts}


class EnvironmentGuard:
    """Detect environment changes without repairing or accepting them implicitly."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._baseline: dict[str, str] = {}

    def _files(self) -> Iterator[Path]:
        for pattern in (
            ".venv/Lib/site-packages/*.pth",
            ".venv/Lib/site-packages/*.dist-info/direct_url.json",
            ".venv/lib/python*/site-packages/*.pth",
            ".venv/lib/python*/site-packages/*.dist-info/direct_url.json",
            "node_modules/.package-lock.json",
        ):
            parts = pattern.split("/")
            directories = [self.root]
            for part in parts[:-1]:
                directories = [
                    child
                    for directory in directories
                    for child in _matching(directory, part, S_IFDIR)
                ]
            for directory in directories:
                yield from _matching(directory, parts[-1], S_IFREG)

    def _fingerprint(self) -> dict[str, str]:
        return {
            path.relative_to(self.root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in self._files()
        }

    def broken_editables(self) -> list[str]:
        """Return filenames pointing at missing paths or disposable wave worktrees."""
        broken: set[str] = set()
        for path in self._files():
            targets: list[Path] = []
            if path.suffix.lower() == ".pth":
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "import ", "import\t")):
                        targets.append(Path(line))
            elif path.name == "direct_url.json":
                target = _editable_target(path)
                if target is not None:
                    targets.append(target)
            for target in targets:
                if _broken_target(target):
                    broken.add(path.relative_to(self.root).as_posix())
        return sorted(broken)

    def capture(self) -> None:
        """Explicitly record the current environment, e.g. after a brokered install."""
        self._baseline = self._fingerprint()

    def check(self) -> list[str]:
        """Return sorted changed relative filenames, leaving the baseline untouched."""
        current = self._fingerprint()
        return sorted(
            name
            for name in self._baseline.keys() | current.keys()
            if self._baseline.get(name) != current.get(name)
        )
