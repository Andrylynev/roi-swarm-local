"""Single local entry point; no cloud provider credentials or global profile changes."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        print("roi-local {start|stop|status|inspect|serve|chat|setup|profile}")
        print("Default endpoints: dashboard 127.0.0.1:8775/dashboard, API 127.0.0.1:8775/v1")
        return 0
    command, *rest = args
    if command in {"start", "stop", "status", "inspect", "serve"}:
        from local import manage

        return manage.main(args)
    if command == "chat":
        from local import run_hermes

        return run_hermes.main(rest)
    if command == "setup":
        from local import setup

        return setup.main(rest)
    if command == "profile":
        from local.profiles import apply_profile

        parser = argparse.ArgumentParser(prog="roi-local profile")
        parser.add_argument("profile", choices=("fast", "balanced", "security"))
        options = parser.parse_args(rest)
        apply_profile(options.profile)
        return 0
    raise SystemExit(f"Unknown local command: {command}; use roi-local --help")


if __name__ == "__main__":
    raise SystemExit(main())
