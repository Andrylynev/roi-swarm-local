"""Launch a separate local Hermes chat without altering the active desktop profile."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from local.manage import ROOT, child_environment, hermes_runtime


def main(argv: list[str] | None = None, root: Path = ROOT):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--root", type=Path, default=root)
    args, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    source, python = hermes_runtime(root)
    if not python.is_file():
        raise SystemExit("Hermes interpreter is missing: " + str(python))
    env = child_environment(root)
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{env['ROI_LOCAL_API_PORT']}/v1"
    env["VIRTUAL_ENV"] = str(python.parent.parent)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(source)
    # Explicit CLI provider/model plus child base URL: never reuse desktop cloud route.
    command = [
        str(python),
        "-m",
        "hermes_cli.main",
        "chat",
        "--provider",
        "roi-local",
        "--model",
        "roi-swarm",
        *forwarded,
    ]
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
