"""Pinned, SHA-256 verified downloads. Importing this module never uses the network."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = {"User-Agent": "Andrylynev/roi-swarm-local installer"}


def select_assets(system: str, backend: str, *, with_security: bool = False) -> list[dict]:
    if system not in {"Windows", "Linux"} or backend not in {"cpu", "cuda"}:
        raise ValueError("Supported runtimes: Windows/Linux x64 CPU or CUDA12.4")
    assets = json.loads((Path(__file__).with_name("assets.json")).read_text(encoding="utf-8"))
    return [
        a
        for a in assets
        if (a["kind"] == "runtime" and a["system"] == system and a["backend"] == backend)
        or (a["kind"] == "model" and (with_security or not a.get("optional", False)))
    ]


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, asset: dict) -> bool:
    return (
        path.is_file() and path.stat().st_size == asset["bytes"] and sha256(path) == asset["sha256"]
    )


def mutation_path(path: Path) -> Path:
    """Reject links/reparse points before resolving any writable path or parent.

    lstat also detects Windows junctions on Python 3.11 (is_symlink does not).
    This guards installer-owned paths, not interpreter files read for execution.
    """
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Unsafe linked/reparse mutation path: {component}")
    return path


def safe_target(root: Path, name: str) -> Path:
    """Use the same portable path rules on Windows and POSIX, including ADS."""
    parts = name.replace("\\", "/").split("/")
    if (
        not name
        or name.startswith(("/", "\\"))
        or any(part == ".." or ":" in part or part.rstrip(" .") != part for part in parts)
    ):
        raise ValueError("Unsafe asset/archive path: " + name)
    target = mutation_path(root / name.replace("\\", "/")).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("Unsafe asset/archive path: " + name)
    return target


class _TLSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("HTTPS download cannot redirect to plaintext")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(
    asset: dict,
    *,
    root: Path = ROOT,
    asset_cache: Path | None = None,
    opener=None,
    attempts: int = 4,
    timeout: float = 90,
    sleep=time.sleep,
) -> dict:
    """Publish only complete verified bytes; an old destination survives any failure."""
    path = safe_target(Path(root), asset["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if verified(path, asset):
        return {**asset, "verified": True}
    fd, temporary = tempfile.mkstemp(prefix=path.name + "-", suffix=".part", dir=path.parent)
    os.close(fd)
    partial = Path(temporary)
    try:
        if asset_cache is not None:
            cache = Path(asset_cache)
            for candidate in (cache / asset["path"], cache / path.name, cache / asset["sha256"]):
                if verified(candidate, asset):
                    shutil.copyfile(candidate, partial)
                    if not verified(partial, asset):
                        raise ValueError("Cache changed during verified copy: " + path.name)
                    os.replace(partial, path)
                    return {**asset, "verified": True}
        if opener is None and not asset["url"].startswith("https://"):
            raise ValueError("Production downloads require HTTPS")
        opening = opener or urllib.request.build_opener(_TLSRedirect()).open
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(asset["url"], headers=AGENT)
                with opening(request, timeout=timeout) as response, partial.open("wb") as stream:
                    total = 0
                    while block := response.read(min(1024 * 1024, asset["bytes"] - total + 1)):
                        total += len(block)
                        if total > asset["bytes"]:
                            raise ValueError("Response exceeds pinned byte size: " + path.name)
                        stream.write(block)
                if not verified(partial, asset):
                    raise ValueError("SHA256 or byte size mismatch: " + path.name)
                os.replace(partial, path)
                return {**asset, "verified": True}
            except (OSError, ValueError):
                if attempt + 1 == attempts:
                    raise
                sleep(2**attempt)
        raise ValueError("attempts must be positive")
    finally:
        partial.unlink(missing_ok=True)


def unpack(archive: Path, stage: Path) -> None:
    """Extract regular files/directories only; never delegate path handling to tar/zip."""
    stage.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = safe_target(stage, member.filename.rstrip("/"))
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Unsafe ZIP special file: " + member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    else:
        with tarfile.open(archive, "r:*") as source:
            members = source.getmembers()
            names = {m.name.removeprefix("./"): m for m in members}
            for member in members:
                name = member.name.removeprefix("./").rstrip("/")
                if name == "." and member.isdir():
                    continue
                target = safe_target(stage, name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                resolved = member
                seen = set()
                while resolved.issym() or resolved.islnk():
                    if resolved.name in seen:
                        raise ValueError("Unsafe archive link cycle")
                    seen.add(resolved.name)
                    link = resolved.linkname
                    # Materialize in-archive regular bytes, never OS links. Relative
                    # '..' links are rejected even if they might normalize inside.
                    safe_target(stage, link)
                    link_name = (
                        str(Path(resolved.name).parent / link).replace("\\", "/")
                        if resolved.issym()
                        else link
                    )
                    link_name = link_name.removeprefix("./")
                    safe_target(stage, link_name)
                    if link_name not in names:
                        raise ValueError("Unsafe missing archive link target")
                    resolved = names[link_name]
                if not resolved.isfile():
                    raise ValueError("Unsafe unsupported TAR special file: " + member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(resolved) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    for path in stage.rglob("*"):
        if path.is_file() and (path.name.startswith("llama-") or ".so" in path.name):
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def extract_runtime(archives: list[Path], destination: Path, *, validate=None) -> None:
    """Prepare and probe beside the old runtime; roll back failed publication."""
    destination = mutation_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prism-stage-", dir=destination.parent))
    backup = stage.with_name(stage.name + ".previous")
    moved = False
    try:
        for archive in archives:
            unpack(archive, stage)
        if validate is not None:
            validate(stage)
        mutation_path(destination)
        mutation_path(backup)
        if destination.exists():
            destination.rename(backup)
            moved = True
        try:
            stage.rename(destination)
        except BaseException:
            if moved:
                backup.rename(destination)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(mutation_path(stage))
    # Publication is committed. Cleanup failure must not tell callers to undo
    # config/receipt metadata while leaving the new runtime in place.
    if moved:
        try:
            shutil.rmtree(mutation_path(backup))
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning(
                "Runtime committed at %s; previous-runtime cleanup failed (%s). "
                "Retained backup for operator cleanup: %s",
                destination,
                exc,
                backup,
            )
