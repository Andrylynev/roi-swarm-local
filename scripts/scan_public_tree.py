"""Scan the exact Git index, without printing matched secret values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{24,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
]
FORBIDDEN = {".venv", "runtime", "models", "hermes", "data", "runs", "cache", "download", "tools"}


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    findings = []
    count = 0
    for raw in filter(None, names):
        name = raw.decode("utf-8")
        path = Path(name)
        if path.parts[0] in FORBIDDEN or path.name == ".env":
            findings.append((name, "runtime_or_secret_path"))
            continue
        blob = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        count += 1
        if len(blob) > 2 * 1024 * 1024:
            findings.append((name, "unexpected_large_blob"))
        for index, pattern in enumerate(SECRET_PATTERNS):
            if pattern.search(blob):
                findings.append((name, f"credential_pattern_{index}"))
    print(f"Scanned {count} indexed files; findings: {len(findings)}")
    for name, kind in findings:
        print(f"{name}: {kind}")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
