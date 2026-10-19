"""Heuristic routing policy for shared Python/Node package environments.

This is command classification, not a shell security sandbox. Dynamic shell
programs and arbitrary nested scripts require separate execution controls.
"""

import re
from dataclasses import dataclass
from typing import Any

_PIP = re.compile(r"pip(?:\d+(?:\.\d+)*)?\Z")
_PYTHON = re.compile(r"python(?:\d+(?:\.\d+)*)?\Z")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*=")
_MUTATIONS = {
    "poetry": {"install", "add", "remove", "update", "lock"},
    "pipenv": {"install", "uninstall", "sync"},
    "npm": {"install", "i", "ci", "add", "uninstall", "remove", "update"},
    "pnpm": {"install", "i", "add", "remove"},
    "yarn": {"install", "add", "remove"},
    "bun": {"install"},
}
_VALUE_OPTIONS = {
    "pip": {
        "--python",
        "--proxy",
        "--retries",
        "--timeout",
        "--exists-action",
        "--trusted-host",
        "--cert",
        "--client-cert",
        "--cache-dir",
        "--log",
        "--keyring-provider",
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "-i",
        "-f",
    },
    "uv": {
        "-p",
        "--directory",
        "-C",
        "--project",
        "--config-file",
        "--cache-dir",
        "--color",
        "--python",
        "--index",
        "--default-index",
        "--index-url",
        "--extra-index-url",
        "--keyring-provider",
        "--allow-insecure-host",
    },
    "poetry": {"-C", "--directory", "-P", "--project"},
    "pipenv": {"--python", "--pypi-mirror"},
    "conda": {"-n", "--name", "-p", "--prefix", "--root-prefix"},
    "npm": {
        "--prefix",
        "--cache",
        "--registry",
        "--userconfig",
        "--workspace",
        "-w",
        "--location",
    },
    "pnpm": {"--dir", "-C", "--prefix", "--filter", "-F", "--store-dir"},
    "yarn": {"--cwd", "--cache-folder", "--modules-folder", "--use-yarnrc"},
    "bun": {"--cwd", "--config", "-c", "--cache-dir"},
}


class _QuotedAssignment(str):
    """An assignment-looking word whose name/equal sign was shell quoted."""


@dataclass
class _Shell:
    segments: list[list[str]]
    unsafe: bool = False


def _shell(text: str) -> _Shell:
    """Read literal segment heads, preserving quoted Windows backslashes.

    Unlike POSIX shlex, an ordinary backslash before a path letter is retained.
    Shell operators, escapes, comments and quotes still get lexical treatment.
    """
    result = _Shell([])
    words: list[str] = []
    word: list[str] = []
    quote = ""
    started = False
    redirect_target = False
    quoted_name = False

    def flush() -> None:
        nonlocal word, started, redirect_target, quoted_name
        if started:
            if not redirect_target:
                value = "".join(word)
                words.append(_QuotedAssignment(value) if quoted_name else value)
            redirect_target = False
            quoted_name = False
            word = []
            started = False

    def segment() -> None:
        flush()
        if words:
            result.segments.append(words.copy())
            words.clear()

    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1 : index + 2]
        if char == "\\" and quote != "'":
            if not following:
                result.unsafe = True
            escapable = '$`"\\\n' if quote else " \t\r\n\"';&|<>(){}#$`\\"
            if following and following in escapable:
                if following != "\n":
                    quoted_name |= "=" not in word
                    word.append(following)
                    started = True
                index += 2
                continue
        if quote:
            if char == quote:
                quote = ""
            else:
                word.append(char)
                if quote == '"' and (char == "`" or char == "$"):
                    result.unsafe = True
        elif char in "\"'":
            quoted_name |= "=" not in word
            quote = char
            started = True
        elif char == "#" and not started:
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        elif char == "\n":
            segment()
        elif char.isspace():
            flush()
        elif char in ";&|()":
            result.unsafe = True
            segment()
        elif char == "{" and not words and not started and following.isspace():
            # Braces are reserved words only at a command head, not echo data.
            result.unsafe = True
        elif char in "<>":
            result.unsafe = True
            # A leading file descriptor is syntax, not an executable.
            if started and "".join(word).isdigit():
                word.clear()
                started = False
            flush()
            while index + 1 < len(text) and text[index + 1] in "<>&":
                index += 1
            redirect_target = True
        else:
            if char in "$`":
                result.unsafe = True
            word.append(char)
            started = True
        index += 1
    segment()
    result.unsafe |= bool(quote) or redirect_target or len(result.segments) > 1
    return result


def _executable(word: str) -> str:
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"\.(?:exe|cmd|bat)\Z", "", name)


def _value_option(option: str, tool: str) -> bool:
    values = _VALUE_OPTIONS.get(tool, set())
    if option in values:
        return True
    # pip accepts unambiguous long-option abbreviations, including --py.
    return (
        tool == "pip"
        and option.startswith("--")
        and len(option) >= 4
        and ("--python".startswith(option))
    )


# Only options with known arity may be skipped. Guessing "one token" for an
# unknown option lets its value (info/list/test) masquerade as a readonly command.
_FLAG_OPTIONS = {
    "pip": {
        "--isolated",
        "--require-virtualenv",
        "--disable-pip-version-check",
        "--no-input",
        "--no-color",
        "--no-cache-dir",
        "--quiet",
        "-q",
        "--verbose",
        "-v",
    },
    "uv": {
        "--offline",
        "--no-config",
        "--no-cache",
        "-n",
        "--native-tls",
        "--system-certs",
        "--no-python-downloads",
        "--no-managed-python",
        "--managed-python",
        "--quiet",
        "-q",
        "--verbose",
        "-v",
        "--no-progress",
    },
    "npm": {"--global", "-g", "--local", "--quiet", "-q", "--silent", "-s"},
    "poetry": {"--quiet", "-q", "--verbose", "-v", "--no-interaction", "-n", "--no-ansi"},
    "pipenv": {"--verbose", "--quiet", "--bare", "--site-packages", "--system"},
    "conda": {"--quiet", "-q", "--verbose", "-v", "--yes", "-y", "--offline", "--json"},
    "pnpm": {"--global", "-g", "--silent", "--offline"},
    "yarn": {"--silent", "--offline", "--verbose", "--non-interactive"},
    "bun": {"--silent", "--offline", "--verbose"},
    "setup": {"--quiet", "-q", "--verbose", "-v"},
}
_AMBIGUOUS_OPTION = "<unresolved-option>"


def _option_width(args: list[str], index: int, tool: str) -> int:
    option = args[index]
    if "=" in option:
        # An explicitly bound value cannot be confused with the subcommand.
        return 1
    if _value_option(option, tool):
        return 2 if index + 1 < len(args) else 0
    if any(
        len(flag) == 2 and option.startswith(flag) and len(option) > 2
        for flag in _VALUE_OPTIONS.get(tool, set())
    ):
        return 1
    if tool == "npm" and option in {"--global", "-g", "--local"}:
        return 2 if index + 1 < len(args) and args[index + 1].lower() in {"true", "false"} else 1
    if option in _FLAG_OPTIONS.get(tool, set()) | {"--help", "-h", "--version"}:
        return 1
    if option == "-V" and tool in {"pip", "uv", "poetry", "pipenv"}:
        return 1
    if option == "-v" and tool in {"npm", "pnpm", "yarn", "bun"}:
        return 1
    return 0


def _skip_options(args: list[str], tool: str) -> list[str]:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        if args[index] == "--":
            return args[index + 1 :]
        width = _option_width(args, index, tool)
        if not width:
            return [_AMBIGUOUS_OPTION]
        index += width
    return args[index:]


def _unwrap(words: list[str]) -> tuple[list[str], str | None]:
    """Peel literal wrappers; never scan their option values for executable names."""
    assignment_mode = "shell"
    while words:
        while (
            words
            and assignment_mode
            and (assignment_mode == "env" or not isinstance(words[0], _QuotedAssignment))
            and _ASSIGNMENT.match(words[0])
        ):
            words = words[1:]
        if not words:
            break
        wrapper = _executable(words[0])
        if wrapper not in {"env", "command", "exec"}:
            break
        args = words[1:]
        index = 0
        while index < len(args) and args[index].startswith("-"):
            option = args[index]
            index += 1
            if option == "--":
                break
            if wrapper == "env":
                if option in {"-S", "--split-string"}:
                    split = args[index] if index < len(args) else ""
                    return args[index + 1 :], split
                if option.startswith("--split-string=") or option.startswith("-S"):
                    split = option.split("=", 1)[1] if "=" in option else option[2:]
                    return args[index:], split
                if option in {"-u", "--unset", "-C", "--chdir", "--argv0"}:
                    index += 1
                elif option not in {"-i", "--ignore-environment", "-", "-v", "--debug"} and not (
                    option.startswith(("--unset=", "--chdir=", "--argv0=", "-u", "-C"))
                ):
                    return [], None
            elif wrapper == "command":
                if option != "-p":
                    return [], None
            elif option == "-a":
                index += 1
            elif option not in {"-c", "-l"}:
                return [], None
        words = args[index:]
        assignment_mode = "env" if wrapper == "env" else ""
    return words, None


_UV_RUN_VALUES = _VALUE_OPTIONS["uv"] | {
    "--with",
    "--with-editable",
    "--with-requirements",
    "--package",
    "--extra",
    "--group",
    "--env-file",
    "-p",
    "-w",
}


def _uv_run_options(args: list[str]) -> list[str]:
    """Return only uv's switches, stopping at the command/module boundary."""
    options = []
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option == "--":
            break
        options.append(option)
        if option in {"-m", "--module"}:
            break
        if option in _UV_RUN_VALUES:
            width = 2 if index + 1 < len(args) else 0
        elif option in {"--no-sync", "--frozen", "--locked", "--isolated", "--no-project"}:
            width = 1
        else:
            width = _option_width(args, index, "uv")
        if not width:
            # Do not use --no-sync/--help that may actually be an unknown option's
            # value to grant the local/informational fast path.
            return []
        index += width
    return options


def _informational(args: list[str], tool: str) -> bool:
    """Recognize help/version switches, not their appearances as option values."""
    index = 0
    switches = {"--help", "-h", "--version"}
    if tool in {"pip", "uv", "poetry", "pipenv"}:
        switches.add("-V")
    if tool in {"npm", "pnpm", "yarn", "bun"}:
        switches.add("-v")
    while index < len(args):
        arg = args[index]
        if arg == "--":
            break
        if arg in switches:
            return True
        if tool == "uv" and arg == "run":
            return bool({"--help", "-h"}.intersection(_uv_run_options(args[index + 1 :])))
        if arg.startswith("-"):
            width = _option_width(args, index, tool)
            if not width:
                # Its next token may be a value, even '--help'. Never let an
                # unresolved option select the informational/local fast path.
                return False
            index += width
        else:
            index += 1
    return False


def _python(args: list[str]) -> str | None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("-c") or arg in {"--help", "-h", "--version", "-V", "-"}:
            return None
        if arg.startswith("-m") and arg != "-m":
            args = [*args[:index], "-m", arg[2:], *args[index + 1 :]]
            arg = "-m"
        if arg == "-m":
            if index + 1 < len(args):
                if _PIP.fullmatch(args[index + 1]):
                    return _classify(["pip", *args[index + 2 :]])
                if args[index + 1] == "venv":
                    return "venv"
            return None
        if arg in {"-W", "-X", "--check-hash-based-pycs"}:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            if _executable(arg) == "setup.py":
                commands = _skip_options(args[index + 1 :], "setup")
                if commands[:1] == [_AMBIGUOUS_OPTION]:
                    return "ambiguous"
                if commands[:1] in (["install"], ["develop"]):
                    return "install"
            return None
    return None


def _classify(words: list[str]) -> str | None:
    words, split_string = _unwrap(words)
    if split_string is not None:
        # env's second parser has different expansion rules: never forward it.
        parsed = _shell(split_string)
        if parsed.segments:
            parsed.segments[-1].extend(words)
        else:
            parsed.segments.append(words)
        return "env" if any(_classify(segment) for segment in parsed.segments) else None
    if not words:
        return None
    executable = _executable(words[0])
    args = words[1:]
    if _PYTHON.fullmatch(executable) or executable == "py":
        return _python(args)
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        # npm's help/version are configurable booleans, not unconditional exits.
        # Do not emulate each Node manager's override rules: permit only a sole
        # explicit informational switch, never a mixed command or false value.
        plain_info = {"--help", "-h", "--version", "-v"}
        info_options = plain_info | {
            "--usage",
            "-?",
            "--no-help",
            "--no-usage",
            "--no-version",
        }
        if any(arg.partition("=")[0] in info_options for arg in args):
            return None if len(args) == 1 and args[0] in plain_info else "ambiguous"
    option_tool = "pip" if _PIP.fullmatch(executable) else executable
    if _informational(args, option_tool):
        return None
    if _PIP.fullmatch(executable):
        args = _skip_options(args, "pip")
        if args[:1] in (["install"], ["uninstall"]):
            return "install"
        return (
            None
            if not args
            or args[0]
            in {
                "list",
                "show",
                "freeze",
                "check",
                "debug",
                "help",
                "index",
                "download",
                "wheel",
                "cache",
                "config",
                "inspect",
                "completion",
            }
            else "ambiguous"
        )
    if executable == "uv":
        args = _skip_options(args, "uv")
        if args[:1] == ["venv"]:
            return "venv"
        if args[:1] == ["run"]:
            return None if "--no-sync" in _uv_run_options(args[1:]) else "run"
        if args[:1] == ["pip"]:
            subargs = _skip_options(args[1:], "uv")
            if subargs[:1] in (["install"], ["uninstall"], ["sync"]):
                return "install"
            return (
                None
                if not subargs
                or subargs[0]
                in {
                    "compile",
                    "list",
                    "show",
                    "freeze",
                    "check",
                    "tree",
                    "help",
                }
                else "ambiguous"
            )
        if args[:1] in (["sync"], ["add"], ["remove"]):
            return "install"
        if args[:1] == ["tool"]:
            tool_args = _skip_options(args[1:], "uv")
            if tool_args[:1] in (["install"], ["uninstall"], ["upgrade"]):
                return "install"
            return (
                None
                if not tool_args
                or tool_args[0]
                in {
                    "list",
                    "dir",
                    "help",
                    "update-shell",
                }
                else "ambiguous"
            )
        return (
            None
            if not args
            or args[0]
            in {
                "help",
                "tree",
                "export",
                "lock",
                "version",
                "cache",
                "auth",
                "python",
                "build",
                "publish",
                "init",
                "workspace",
                "format",
                "check",
                "audit",
                "self",
            }
            else "ambiguous"
        )
    if executable in {"conda", "mamba", "micromamba"}:
        args = _skip_options(args, "conda")
        if args[:1] == [_AMBIGUOUS_OPTION]:
            return "ambiguous"
        if args[:1] in (["install"], ["remove"], ["update"]):
            return "install"
        return "install" if args[:2] == ["env", "update"] else None
    args = _skip_options(args, executable)
    if executable == "yarn" and not args:
        return "install"
    if executable in _MUTATIONS and args:
        if args[0] in _MUTATIONS[executable]:
            return "install"
        readonly = {
            "test",
            "run",
            "run-script",
            "start",
            "stop",
            "restart",
            "list",
            "ls",
            "show",
            "info",
            "view",
            "why",
            "explain",
            "outdated",
            "audit",
            "config",
            "prefix",
            "root",
            "help",
            "version",
            "cache",
            "exec",
            "dlx",
            "pack",
            "publish",
            "search",
            "ping",
            "build",
            "check",
            "env",
            "shell",
            "lock",
            "export",
            "about",
            "init",
            "bin",
        }
        return None if args[0] in readonly else "ambiguous"
    return None


def is_install(text: Any) -> bool:
    """Whether a command invokes a mutating package/environment operation."""
    return isinstance(text, str) and any(
        _classify(words) is not None for words in _shell(text).segments
    )


def install_error(command: str) -> str | None:
    """Explain why a recognized install cannot run in the shared worktree."""
    parsed = _shell(command)
    kinds = {_classify(words) for words in parsed.segments}
    if "ambiguous" in kinds:
        return (
            "Ambiguous package-manager command: cannot identify its subcommand safely. "
            "Use a plain standalone install command with options after the subcommand; "
            "unknown options must not run locally."
        )
    if "venv" in kinds:
        return (
            "Refusing to recreate or replace the running shared venv. Use a plain package "
            "installer for dependency changes; environment recreation must be done by the "
            "central coordinator after workers stop."
        )
    if "run" in kinds:
        return (
            "Do not forward uv run centrally: it may sync the shared environment and run tests "
            "in the wrong tree. Install dependencies separately, then use uv run --no-sync ... "
            "so tests and other code stay in the worker worktree."
        )
    if "env" in kinds:
        return (
            "Use a single plain installer command, not env --split-string/-S: its second "
            "command parser cannot be safely forwarded centrally."
        )
    if parsed.unsafe:
        return (
            "Run a single standalone installer command in terminal. Shell compounds, pipes, "
            "redirection, substitutions and malformed quoting cannot be forwarded centrally; "
            "run directory changes, tests and other commands separately in the worker worktree."
        )
    return None
