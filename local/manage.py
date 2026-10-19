"""Ownership-checked lifecycle CLI for an isolated Windows/Linux local install."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from ctypes import wintypes
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORTS = {"inference": 8091, "api": 8775}


def read_install(root: Path) -> dict:
    """Read optional setup metadata; malformed or foreign records fail closed."""
    root = Path(root).resolve()
    path = root / "data" / "local-install.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError) as exc:
        raise LifecycleError(f"Cannot read install record {path}: {exc}") from exc
    try:
        record = json.loads(raw)
        if (
            not isinstance(record, dict)
            or type(record.get("version")) is not int
            or record["version"] != 1
        ):
            raise ValueError("unsupported schema")
        if not isinstance(record.get("root"), str) or not same_path(record["root"], str(root)):
            raise ValueError("record belongs to a different installation")
        if record.get("backend") not in {"cpu", "cuda"}:
            raise ValueError("invalid backend")
        if record.get("profile") not in {"fast", "balanced", "security"}:
            raise ValueError("invalid profile")
        ports = [record.get(key) for key in ("api_port", "inference_port", "lock_port")]
        if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports):
            raise ValueError("invalid unprivileged port")
        if len(set(ports)) != 3:
            raise ValueError("ports must be distinct")
        hermes = root / "runtime" / "hermes-agent"
        if not isinstance(record.get("hermes_root"), str) or not same_path(
            record["hermes_root"], str(hermes)
        ):
            raise ValueError("Hermes must be isolated under runtime/hermes-agent")
        python = record.get("hermes_python")
        if not isinstance(python, str) or not any(
            os.path.normcase(os.path.abspath(python))
            == os.path.normcase(os.path.abspath(hermes / ".venv" / suffix))
            for suffix in ("Scripts/python.exe", "bin/python")
        ):
            raise ValueError("Hermes interpreter must belong to its isolated .venv")
        # Resolving the source tree itself must not escape the user-owned root.
        if not hermes.resolve().is_relative_to(root):
            raise ValueError("Hermes source escapes installation root")
        return record
    except (ValueError, TypeError, KeyError) as exc:
        raise LifecycleError(f"Cannot trust install record {path}: {exc}") from exc


def lock_for_root(root: Path):
    """Keep the historical default-lock API while honoring installed overrides."""
    record = read_install(root)
    return lifecycle_lock(record["lock_port"]) if record else lifecycle_lock()


def app_python(root: Path) -> Path:
    suffixes = ("Scripts/pythonw.exe", "Scripts/python.exe", "bin/python")
    if os.name != "nt":
        suffixes = ("bin/python", "Scripts/pythonw.exe", "Scripts/python.exe")
    candidates = [root / ".venv" / suffix for suffix in suffixes]
    return next((path for path in candidates if path.is_file()), candidates[0])


def hermes_runtime(root: Path) -> tuple[Path, Path]:
    """Never fall back to a globally installed or desktop Hermes interpreter."""
    record = read_install(root)
    source = Path(root).resolve() / "runtime" / "hermes-agent"
    python = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return source, Path(record.get("hermes_python", python))


def build_plan(root: Path, models_max: int = 1) -> dict:
    """Inspect launch requirements without creating directories or loading ROI."""
    if models_max < 1:
        raise ValueError("--models-max must be at least 1")
    root = Path(root).resolve()
    record = read_install(root)
    ports = {
        "inference": record.get("inference_port", PORTS["inference"]),
        "api": record.get("api_port", PORTS["api"]),
    }
    filenames = (
        ("llama-server.exe", "llama-server")
        if os.name == "nt"
        else ("llama-server", "llama-server.exe")
    )
    published = root / "runtime" / "prism"
    candidates = [
        path for name in filenames for path in sorted(published.rglob(name)) if path.is_file()
    ]
    runtime = candidates[0] if candidates else published / filenames[0]
    pythonw = app_python(root)
    preset = root / "local" / "models.ini"
    commands = {
        "inference": [
            str(runtime),
            "--host",
            HOST,
            "--port",
            str(ports["inference"]),
            "--models-preset",
            str(preset),
            "--models-max",
            str(models_max),
            "--no-webui",
        ],
        "api": [
            str(pythonw),
            "-m",
            "uvicorn",
            "local.api:create_local_app",
            "--factory",
            "--host",
            HOST,
            "--port",
            str(ports["api"]),
            "--no-access-log",
        ],
    }
    return {
        "root": str(root),
        "state_file": str(root / "data" / "local-services.json"),
        "models_max": models_max,
        "backend": record.get("backend", "cuda"),
        "lock_port": record.get("lock_port", 8776),
        "missing": [str(path) for path in (runtime, pythonw, preset) if not path.is_file()],
        "services": {
            name: {
                "command": command,
                "cwd": str(root),
                "port": ports[name],
                "health_url": f"http://{HOST}:{ports[name]}/health",
                "log": str(
                    root
                    / "data"
                    / "diagnostics"
                    / ("inference.log" if name == "inference" else "local-api.log")
                ),
            }
            for name, command in commands.items()
        },
    }


class LifecycleError(RuntimeError):
    """A safe refusal or a failed local service operation."""


@contextmanager
def process_handle(pid: int):
    """Pin the process object so its PID cannot be recycled during taskkill."""
    if os.name != "nt":
        try:
            process = psutil.Process(pid)
            process.create_time()  # Cache identity before later reuse-safe signals.
        except psutil.NoSuchProcess:
            process = None
        except psutil.Error as exc:
            raise LifecycleError(f"Cannot inspect process PID {pid}: {exc}") from exc
        yield process
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle and ctypes.get_last_error() != 87:  # ERROR_INVALID_PARAMETER: no such PID
        raise LifecycleError(f"Cannot inspect process PID {pid}; ownership is unconfirmed")
    try:
        yield (kernel, handle) if handle else None
    finally:
        if handle:
            kernel.CloseHandle(handle)


def identity_from_handle(pid: int, pinned) -> dict | None:
    if pinned is None:
        return None
    if isinstance(pinned, psutil.Process):
        return identity_from_process(pinned)
    kernel, handle = pinned
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    wait_status = kernel.WaitForSingleObject(handle, 0)
    if wait_status == 0:
        return None
    if wait_status != 0x102:
        raise LifecycleError(f"Cannot inspect process state for PID {pid}")
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(system),
        ctypes.byref(user),
    ):
        # The pinned process can finish after the first liveness check.
        # Only a signaled handle is a confirmed exit; other query failures
        # must still fail closed rather than treating a live process as gone.
        if kernel.WaitForSingleObject(handle, 0) == 0:
            return None
        raise LifecycleError(f"Cannot read creation time for PID {pid}")
    kernel.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    size = wintypes.DWORD(32768)
    image = ctypes.create_unicode_buffer(size.value)
    if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
        if kernel.WaitForSingleObject(handle, 0) == 0:
            return None
        raise LifecycleError(f"Cannot read executable path for PID {pid}")
    return {
        "pid": pid,
        "created": str((created.dwHighDateTime << 32) | created.dwLowDateTime),
        "image": str(Path(image.value).resolve()),
    }


def identity_from_process(process: psutil.Process | None) -> dict | None:
    """Portable identity; start time is an integer microsecond receipt."""
    if process is None:
        return None
    try:
        created = process.create_time()
        image = process.exe()
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {
            "pid": process.pid,
            "created": str(round(created * 1_000_000)),
            "image": str(Path(image).resolve()),
        }
    except psutil.NoSuchProcess:
        return None
    except psutil.Error as exc:
        raise LifecycleError(f"Cannot inspect process PID {process.pid}: {exc}") from exc


def process_identity(pid: int) -> dict | None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    with process_handle(pid) as pinned:
        return identity_from_handle(pid, pinned)


def same_path(left: str, right: str) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def ownership(record: dict | None, service: dict, current: dict | None = None) -> dict:
    """Require the exact saved PID, start time, executable and launch command."""
    result = {"owned": False, "pid": record.get("pid") if isinstance(record, dict) else None}
    if not isinstance(record, dict):
        return {**result, "reason": "no saved process"}
    if not isinstance(record.get("image"), str):
        return {**result, "reason": "invalid saved executable path"}
    if record.get("command") != service["command"]:
        return {**result, "reason": "saved command does not match this installation"}
    pid = record.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return {**result, "reason": "invalid saved PID"}
    live = current if current is not None else process_identity(pid)
    if live is None:
        return {**result, "reason": "process is not running or cannot be inspected"}
    if str(live["created"]) != str(record.get("created")):
        return {**result, "reason": "PID creation time changed; refusing recycled PID"}
    expected = service["command"][0]
    if not same_path(live["image"], expected) or not same_path(record.get("image", ""), expected):
        return {**result, "reason": "executable path does not match this installation"}
    return {**result, "owned": True, "reason": "verified PID, creation time and executable"}


def read_state(root: Path) -> dict:
    path = Path(root) / "data" / "local-services.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": 1, "root": str(Path(root).resolve()), "models_max": 1, "services": {}}
    except (OSError, UnicodeError) as exc:
        raise LifecycleError(
            f"Cannot trust state file {path}: {exc}; no process was stopped"
        ) from exc
    try:
        state = json.loads(raw)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("unsupported schema")
        if not isinstance(state.get("root"), str) or not same_path(state["root"], str(root)):
            raise ValueError("state belongs to a different installation")
        if not isinstance(state.get("services"), dict) or set(state["services"]) - set(PORTS):
            raise ValueError("invalid service records")
        if not isinstance(state.get("models_max", 1), int) or state.get("models_max", 1) < 1:
            raise ValueError("invalid models_max")
        return state
    except (OSError, ValueError, TypeError) as exc:
        raise LifecycleError(
            f"Cannot trust state file {path}: {exc}; no process was stopped"
        ) from exc


def write_state(root: Path, state: dict) -> None:
    path = Path(root) / "data" / "local-services.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="local-services-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def child_environment(root: Path, source: dict | None = None) -> dict[str, str]:
    """Remove inherited provider/credential routing and pin all mutable caches."""
    root = Path(root).resolve()
    record = read_install(root)
    inherited = os.environ if source is None else source
    blocked = (
        "ROI_",
        "OPENAI_",
        "ANTHROPIC_",
        "AZURE_",
        "AWS_",
        "GOOGLE_",
        "GEMINI_",
        "HERMES_",
        "LLAMA_",
        "HF_",
        "HUGGINGFACE_",
        "UV_",
        "XDG_",
        "PYTHON",
        "LITELLM_",
        "OPENROUTER_",
        "GROQ_",
        "COHERE_",
        "MISTRAL_",
        "TOGETHER_",
        "CONDA_",
    )
    env = {
        key: value
        for key, value in inherited.items()
        if not key.upper().startswith(blocked)
        and key.upper()
        not in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "VIRTUAL_ENV",
            "ROI",
            "API_KEY",
        }
        and not key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
    }
    paths = {
        "HERMES_HOME": root / "hermes",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "HF_HOME": root / "cache" / "huggingface",
        "HF_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "cache" / "huggingface" / "transformers",
        "UV_CACHE_DIR": root / "cache" / "uv",
        "TIKTOKEN_CACHE_DIR": root / "cache" / "tiktoken",
        "LLAMA_CACHE": root / "cache" / "llama",
        "TORCH_HOME": root / "cache" / "torch",
        "TEMP": root / "cache" / "tmp",
        "TMP": root / "cache" / "tmp",
        "TMPDIR": root / "cache" / "tmp",
    }
    env.update({key: str(value) for key, value in paths.items()})
    path_key = next((key for key in env if key.upper() == "PATH"), "PATH")
    env[path_key] = str(app_python(root).parent) + os.pathsep + env.get(path_key, "")
    env["VIRTUAL_ENV"] = str(root / ".venv")
    env.update(
        {
            "ROI_PROVIDER": "local",
            "ROI_LOCAL_NATIVE_BUDGET": "1",
            "OPENAI_API_KEY": "local-only",
            "OPENAI_BASE_URL": (
                f"http://{HOST}:{record.get('inference_port', PORTS['inference'])}/v1"
            ),
            "ROI_LOCAL_ROOT": str(root),
            "ROI_LOCAL_API_PORT": str(record.get("api_port", PORTS["api"])),
            "ROI_LOCAL_INFERENCE_PORT": str(record.get("inference_port", PORTS["inference"])),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )
    return env


@contextmanager
def lifecycle_lock(port: int = 8776):
    """A crash-released loopback lock, unrelated to the remote deployment."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as lock:
        if os.name == "nt":
            lock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            lock.bind((HOST, port))
            lock.listen(1)
        except OSError as exc:
            raise LifecycleError(
                f"Cannot acquire local lock {HOST}:{port}; another lifecycle operation is running"
            ) from exc
        yield


def portable_listener_pid(port: int) -> int | None:
    """Inspect listeners only; an inaccessible or ownerless match is not absence."""
    try:
        matches = [
            item
            for item in psutil.net_connections(kind="tcp")
            if item.status == psutil.CONN_LISTEN
            and item.laddr.port == port
            and item.laddr.ip in {HOST, "0.0.0.0", "::", "::1"}
        ]
    except psutil.Error as exc:
        raise LifecycleError(f"Cannot inspect TCP listener ownership: {exc}") from exc
    if any(item.pid is None for item in matches):
        raise LifecycleError(f"Cannot inspect TCP listener ownership on port {port}")
    owners = {item.pid for item in matches}
    if len(owners) > 1:
        raise LifecycleError(f"Ambiguous TCP listener ownership on port {port}")
    return next(iter(owners), None)


def listener_pid(port: int) -> int | None:
    """Get the listener owner without parsing localized netstat output."""
    if os.name != "nt":
        return portable_listener_pid(port)
    api = ctypes.WinDLL("iphlpapi", use_last_error=True)
    api.GetExtendedTcpTable.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.ULONG,
    ]
    api.GetExtendedTcpTable.restype = wintypes.DWORD
    size = wintypes.DWORD()
    for _ in range(4):
        buffer = ctypes.create_string_buffer(max(size.value, 4))
        code = api.GetExtendedTcpTable(buffer, ctypes.byref(size), False, socket.AF_INET, 3, 0)
        if code == 122:
            continue
        if code != 0:
            raise LifecycleError(f"Cannot inspect TCP listener ownership (Windows error {code})")
        count = struct.unpack_from("<I", buffer.raw)[0]
        for index in range(count):
            state, address, raw_port, _, _, pid = struct.unpack_from(
                "<6I", buffer.raw, 4 + index * 24
            )
            if (
                state == 2
                and socket.ntohs(raw_port & 0xFFFF) == port
                and address in (0, int.from_bytes(socket.inet_aton(HOST), "little"))
            ):
                return pid
        return None
    raise LifecycleError("TCP listener table changed repeatedly; retry the command")


class NoHealthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, "Health redirects are disabled", headers, fp)


def health(url: str) -> dict:
    """Read only a local health route; 503 is loading, not ready."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoHealthRedirect())
        with opener.open(url, timeout=2) as response:
            return {"ready": response.status == 200, "status": response.status}
    except urllib.error.HTTPError as exc:
        return {"ready": False, "status": exc.code}
    except (OSError, urllib.error.URLError) as exc:
        return {"ready": False, "status": None, "error": str(exc)}


def status(root: Path = ROOT) -> dict:
    """Report health independently from ownership; never adopt a listening process."""
    root = Path(root).resolve()
    state = read_state(root)
    plan = build_plan(root, state.get("models_max", 1))
    services = {}
    for name, service in plan["services"].items():
        owned = ownership(state["services"].get(name), service)
        listener = listener_pid(service["port"])
        services[name] = {
            **owned,
            "port": service["port"],
            "listener_pid": listener,
            "listener_owned": owned["owned"]
            and listener_belongs_to(listener, state["services"].get(name)),
            "health": health(service["health_url"]),
            "log": service["log"],
        }
    return {
        "root": str(root),
        "models_max": plan["models_max"],
        "state_file": plan["state_file"],
        "running": not _spawn_recovery_path(root).exists()
        and all(
            s["owned"] and s["listener_owned"] and s["health"]["ready"] for s in services.values()
        ),
        "services": services,
    }


class _WindowsLaunchScope:
    """Contain startup, then explicitly release the Job without killing services."""

    def __init__(self):
        from roi_swarm import worker_process_scope as native

        self.native = native
        self.scope = native.WorkerProcessScope()

    def identity(self, process) -> dict | None:
        # The Popen creation handle, not a potentially reused numeric PID.
        return identity_from_handle(process.pid, (self.native._kernel32, process._handle))

    def attach(self, process) -> None:
        self.scope.attach(process.pid)

    def members(self) -> list[dict]:
        kernel = self.native._kernel32
        kernel.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        records = []
        for pid in self.native._job_pids(self.scope._handle):
            with process_handle(pid) as pinned:
                if pinned is None:
                    continue
                member = wintypes.BOOL()
                if not kernel.IsProcessInJob(pinned[1], self.scope._handle, ctypes.byref(member)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if member.value:
                    identity = identity_from_handle(pid, pinned)
                    if identity is not None:
                        records.append(identity)
        return records

    def release(self) -> None:
        # Disarm successful detachment, but retain actual Job ownership until
        # the creation handle is closed and a durable handoff receipt exists.
        limits = self.native._ExtendedLimitInformation()
        kernel = self.native._kernel32
        if not kernel.SetInformationJobObject(
            self.scope._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def detach(self) -> None:
        """Irreversible close; the caller must already have a durable receipt."""
        if not self.native._kernel32.CloseHandle(self.scope._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        self.scope._handle = None
        self.scope._closed = True

    def close(self) -> None:
        self.scope.close()


def _spawn_recovery_path(root: Path) -> Path:
    return Path(root) / "data" / "local-spawn-recovery.json"


def _write_spawn_recovery(root: Path, receipt: dict) -> None:
    path = _spawn_recovery_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="local-spawn-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def spawn(service: dict, root: Path, env: dict) -> dict:
    recovery = _spawn_recovery_path(root)
    if recovery.exists():
        raise LifecycleError(f"Unconfirmed spawn cleanup; inspect recovery evidence: {recovery}")
    log = Path(service["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    scope = _WindowsLaunchScope() if os.name == "nt" else None
    process = None
    receipt = {"command": service["command"]}
    detach_attempted = False
    try:
        with log.open("ab", buffering=0) as output:
            output.write(
                f"\n--- local launcher {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode()
            )
            process = subprocess.Popen(
                service["command"],
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                close_fds=True,
                **(
                    {
                        "creationflags": (
                            subprocess.DETACHED_PROCESS
                            | subprocess.CREATE_NEW_PROCESS_GROUP
                            | subprocess.CREATE_NO_WINDOW
                            | 0x00000004  # CREATE_SUSPENDED: attach before any child can run.
                        )
                    }
                    if os.name == "nt"
                    else {"start_new_session": True}
                ),
            )
        receipt["pid"] = process.pid
        if scope is not None:
            receipt.update(scope.identity(process) or {})
            scope.attach(process)
        else:
            # Cache a reuse-checked identity independently of normal verification.
            receipt.update(identity_from_process(psutil.Process(process.pid)) or {})
            receipt.update(pgid=os.getpgid(process.pid), sid=os.getsid(process.pid))
        identity = process_identity(process.pid)
        if identity is None or not same_path(identity["image"], service["command"][0]):
            raise LifecycleError(f"Started process cannot be verified; inspect {log}")
        group = (
            {}
            if os.name == "nt"
            else {"pgid": os.getpgid(process.pid), "sid": os.getsid(process.pid)}
        )
        record = {**identity, "command": service["command"], **group}
        if scope is not None:
            scope.release()
            # This service is deliberately detached, not an unreaped child of
            # the short-lived CLI. Close the creation handle explicitly and
            # disarm Popen's destructor without inventing an exit status.
            process._handle.Close()
            process._child_created = False
            receipt.update(record)
            receipt["members"] = scope.members()
            receipt["root"] = str(Path(root).resolve())
            receipt["handoff"] = True
            _write_spawn_recovery(root, receipt)
            # Keep this journal until start has durably adopted the receipt.
            # A signal inside/after CloseHandle must not silently lose the tree.
            detach_attempted = True
            scope.detach()
        return record
    except BaseException as exc:
        failures = []
        if detach_attempted:
            # The Job close may already have happened. Its saved member identities
            # are recovery evidence, never permission for blind numeric PID kills.
            raise LifecycleError(
                f"Interrupted detached handoff; recovery evidence: {recovery}"
            ) from exc
        if scope is not None:
            try:
                receipt["members"] = scope.members()
            except Exception as snapshot_error:
                receipt["snapshot_error"] = str(snapshot_error)
        try:
            if scope is not None:
                scope.close()  # OS membership includes grandchildren even if the root exits.
            elif process is not None and not terminate_process_group(
                receipt, service, _launch_identity=True
            ):
                raise LifecycleError("Could not confirm failed launch process-group cleanup")
        except Exception as cleanup_error:
            failures.append(str(cleanup_error))
        try:
            if process is not None and not (os.name == "nt" and process._handle.closed):
                if process.poll() is None:
                    process.kill()  # Only Popen's own creation handle.
                process.wait(timeout=10)
        except Exception as cleanup_error:
            failures.append(str(cleanup_error))
        if failures:
            # A missing root is not proof of subtree exit. This separate durable
            # journal cannot be overwritten by start's ordinary rollback state.
            receipt["stop_error"] = "; ".join(failures)
            receipt["launch_error"] = str(exc)
            receipt["root"] = str(Path(root).resolve())
            recovery.parent.mkdir(parents=True, exist_ok=True)
            with recovery.open("w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            raise LifecycleError(
                f"{exc}; spawn cleanup incomplete: {receipt['stop_error']}; "
                f"recovery evidence: {recovery}"
            ) from exc
        raise
    finally:
        if os.name == "nt" and process is not None:
            process._handle.Close()
            process._child_created = False
        if scope is not None and not scope.scope._closed:
            # close() may fail to confirm quiescence. Do not leak kernel handles;
            # KILL_ON_JOB_CLOSE still provides best-effort containment on failure.
            kernel = scope.native._kernel32
            for handle in scope.scope._wait_handles:
                kernel.CloseHandle(handle)
            if scope.scope._handle:
                kernel.CloseHandle(scope.scope._handle)


def terminate_process_group(record: dict, service: dict, *, _launch_identity=False) -> bool:
    """Quiesce an isolated POSIX session; signal only reuse-checked Process objects.

    There is no persistent kernel handle for a PGID. Never pass an old saved
    number to killpg: after the leader exits it can name an unrelated group.
    Instead discover the live group under its verified leader, stop its members
    to close the fork window, and use psutil's creation-time-checked signals.
    """
    pid = record["pid"]
    if record.get("pgid") != pid or record.get("sid") != pid:
        return False
    try:
        if os.getpgid(pid) != pid or os.getsid(pid) != pid:
            return False
    except ProcessLookupError:
        return False
    if not _launch_identity and not ownership(record, service)["owned"]:
        return False
    members = {}

    def discover():
        for process in psutil.process_iter():
            try:
                if os.getpgid(process.pid) == pid and os.getsid(process.pid) == pid:
                    if process.status() == psutil.STATUS_ZOMBIE:
                        continue
                    process.create_time()
                    members.setdefault(process.pid, process)
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
            except (PermissionError, psutil.AccessDenied) as exc:
                raise LifecycleError(f"Cannot inspect owned process group {pid}: {exc}") from exc

    stopped = []
    try:
        leader = psutil.Process(pid)
        if identity_from_process(leader) != {
            key: record[key] for key in ("pid", "created", "image")
        }:
            return False
        leader.suspend()
        stopped.append(leader)
        members[pid] = leader
        for _ in range(20):
            previous = set(members)
            discover()
            for process in members.values():
                if process not in stopped:
                    try:
                        process.suspend()
                        stopped.append(process)
                    except psutil.NoSuchProcess:
                        pass
            if set(members) == previous:
                break
        else:
            raise LifecycleError(f"Owned process group {pid} kept changing; cleanup refused")
        for process in members.values():
            with suppress(psutil.NoSuchProcess):
                process.terminate()
    finally:
        for process in stopped:
            with suppress(psutil.NoSuchProcess):
                process.resume()
    _, alive = psutil.wait_procs(list(members.values()), timeout=3)
    for process in alive:
        with suppress(psutil.NoSuchProcess):
            process.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise LifecycleError(f"Could not confirm exit of owned process group {pid}")
    # Do not signal late arrivals without fresh ownership. Keep recovery state.
    before = set(members)
    discover()
    if set(members) != before:
        raise LifecycleError(f"Owned process group {pid} has unconfirmed late children")
    return True


def terminate_owned(record: dict, service: dict) -> bool:
    """Stop only a verified saved process tree, with the root PID pinned throughout."""
    if not ownership(record, service)["owned"]:
        return False
    pid = record["pid"]
    with process_handle(pid) as pinned:
        live = identity_from_handle(pid, pinned)
        if live is None or not ownership(record, service, current=live)["owned"]:
            return False
        if os.name != "nt":
            return terminate_process_group(record, service)
        result = subprocess.run(
            [
                str(Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32" / "taskkill.exe"),
                "/PID",
                str(pid),
                "/T",
                "/F",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise LifecycleError(f"taskkill failed for owned PID {pid}: {result.stdout.strip()}")
        # Identity was verified before mutation and this exact process object is
        # pinned. During termination its image path can become unreadable BEFORE
        # its handle is signaled; do not re-query metadata while waiting for exit.
        kernel, handle = pinned
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        if kernel.WaitForSingleObject(handle, 10_000) != 0:
            raise LifecycleError(f"Could not confirm exit of owned PID {pid}")
    return True


def wait_ready(
    name: str, service: dict, record: dict, timeout: float, *, _shutdown_requested=None
) -> None:
    deadline = time.monotonic() + timeout
    last = {"ready": False}
    while True:
        _check_shutdown(_shutdown_requested)
        if not ownership(record, service)["owned"]:
            raise LifecycleError(
                f"{name} exited or lost verified ownership; inspect {service['log']}"
            )
        last = health(service["health_url"])
        _check_shutdown(_shutdown_requested)
        if last["ready"]:
            if listener_belongs_to(listener_pid(service["port"]), record):
                return
            raise LifecycleError(f"{name} health listener is unowned; inspect {service['log']}")
        if time.monotonic() >= deadline:
            raise LifecycleError(
                f"{name} was not ready after {timeout:g}s ({last}); inspect {service['log']}"
            )
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def process_parents() -> dict[int, int]:
    """Snapshot native parent PIDs; venv launchers spawn the actual Python host."""
    if os.name != "nt":
        try:
            return {p.pid: p.info["ppid"] for p in psutil.process_iter(["ppid"])}
        except psutil.Error as exc:
            raise LifecycleError(f"Cannot inspect process tree: {exc}") from exc

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise LifecycleError("Cannot inspect native process tree")
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        parents = {}
        success = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            success = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        return parents
    finally:
        kernel.CloseHandle(snapshot)


def listener_belongs_to(pid: int | None, record: dict | None) -> bool:
    if not isinstance(record, dict) or pid is None:
        return False
    root_pid = record.get("pid")
    if pid == root_pid:
        return True
    # A live root plus monotonically increasing process creation times excludes PID reuse.
    root_identity = process_identity(root_pid)
    if root_identity is None or root_identity["created"] != record.get("created"):
        return False
    parents = process_parents()
    seen = set()
    while pid != root_pid:
        if pid in seen:
            return False
        seen.add(pid)
        parent_pid = parents.get(pid)
        if parent_pid is None or parent_pid <= 0:
            return False
        child_identity = process_identity(pid)
        parent_identity = process_identity(parent_pid)
        if child_identity is None or parent_identity is None:
            return False
        if int(child_identity["created"]) < int(parent_identity["created"]):
            return False
        pid = parent_pid
    return True


def start(
    root: Path = ROOT,
    models_max: int = 1,
    dry_run: bool = False,
    ready_timeout: float = 180,
    *,
    _supervised: bool = False,
    _receipts: dict | None = None,
    _shutdown_requested=None,
) -> dict:
    plan = build_plan(root, models_max)
    if dry_run:
        return {"dry_run": True, **plan}
    if ready_timeout <= 0:
        raise ValueError("--ready-timeout must be positive (readiness only, not provider calls)")
    root = Path(root).resolve()
    with lock_for_root(root):
        state = read_state(root)
        recovery = _spawn_recovery_path(root)
        if recovery.exists():
            raise LifecycleError(
                f"Unconfirmed spawn cleanup; inspect recovery evidence: {recovery}"
            )
        if _supervised and state["services"]:
            raise LifecycleError(
                "Foreground supervisor refuses existing service records; stop first"
            )
        saved_plan = build_plan(root, state.get("models_max", 1))
        existing = {}
        for name, service in saved_plan["services"].items():
            record = state["services"].get(name)
            owned = ownership(record, service)
            if owned["owned"]:
                existing[name] = record
            elif name in state["services"]:
                raise LifecycleError(
                    f"Saved {name} process tree is unconfirmed ({owned['reason']}); "
                    "retaining its recovery record, refusing duplicate startup"
                )
            listener = listener_pid(service["port"])
            if listener is not None and not (
                name in existing and listener_belongs_to(listener, record)
            ):
                raise LifecycleError(
                    f"Port {service['port']} is occupied by unowned PID {listener}; "
                    "refusing to start or stop that process"
                )
        if existing and state.get("models_max", 1) != models_max:
            raise LifecycleError(
                "Owned services use different --models-max; stop them before changing it"
            )
        if plan["missing"]:
            raise LifecycleError("Missing local prerequisites: " + ", ".join(plan["missing"]))
        env = child_environment(root)
        for key in (
            "HERMES_HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "UV_CACHE_DIR",
            "TIKTOKEN_CACHE_DIR",
            "LLAMA_CACHE",
            "TORCH_HOME",
            "TEMP",
        ):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        state = {"version": 1, "root": str(root), "models_max": models_max, "services": existing}
        started = []
        try:
            for name, service in plan["services"].items():
                _check_shutdown(_shutdown_requested)
                if name not in state["services"]:
                    record = spawn(service, root, env)
                    started.append(name)
                    state["services"][name] = record
                    if _receipts is not None:
                        _receipts[name] = record
                    write_state(root, state)  # Save before the potentially long readiness wait.
                    if recovery.exists():
                        handoff = json.loads(recovery.read_text(encoding="utf-8"))
                        keys = ("pid", "created", "image", "command")
                        if not handoff.get("handoff") or any(
                            handoff.get(key) != record.get(key) for key in keys
                        ):
                            raise LifecycleError(f"Unconfirmed spawn recovery evidence: {recovery}")
                        recovery.unlink()  # The ordinary state now durably owns this receipt.
                # Signals only set a flag: the receipt is fully registered and
                # durable before a cancellation can unwind startup.
                _check_shutdown(_shutdown_requested)
                wait_ready(
                    name,
                    service,
                    state["services"][name],
                    ready_timeout,
                    **({"_shutdown_requested": _shutdown_requested} if _supervised else {}),
                )
                _check_shutdown(_shutdown_requested)
            result = status(root)
            if _supervised:
                result["supervised"] = dict(state["services"])
            return result
        except BaseException as exc:
            failures = []
            for name in reversed(started):
                try:
                    if not terminate_owned(state["services"][name], plan["services"][name]):
                        raise LifecycleError(f"Termination of newly started {name} was refused")
                    del state["services"][name]
                    if _receipts is not None:
                        _receipts.pop(name, None)
                except Exception as cleanup_error:
                    state["services"][name]["stop_error"] = str(cleanup_error)
                    failures.append(str(cleanup_error))
            try:
                write_state(root, state)
            except OSError as state_error:
                failures.append(str(state_error))
            if failures:
                raise LifecycleError(f"{exc}; cleanup incomplete: {'; '.join(failures)}") from exc
            raise


def stop(root: Path = ROOT, dry_run: bool = False) -> dict:
    root = Path(root).resolve()
    if dry_run:
        return {"dry_run": True, "action": "stop", **status(root)}
    with lock_for_root(root):
        state = read_state(root)
        plan = build_plan(root, state.get("models_max", 1))
        stopped, skipped = [], []
        recovery = _spawn_recovery_path(root)
        incomplete = (
            {"spawn": f"Unconfirmed spawn cleanup; inspect recovery evidence: {recovery}"}
            if recovery.exists()
            else {}
        )
        for name in reversed(PORTS):
            record = state["services"].get(name)
            if name not in state["services"]:
                continue
            if not isinstance(record, dict):
                incomplete[name] = "Invalid saved process record; retaining recovery evidence"
                skipped.append(name)
                continue
            try:
                if not terminate_owned(record, plan["services"][name]):
                    detail = ownership(record, plan["services"][name])["reason"]
                    raise LifecycleError("Termination refused: " + detail)
            except (LifecycleError, OSError, subprocess.SubprocessError) as exc:
                # A missing/uninspectable root is NOT proof that its children
                # have exited. Keep the exact evidence for manual recovery.
                record["stop_error"] = str(exc)
                incomplete[name] = str(exc)
                skipped.append(name)
            else:
                stopped.append(name)
                del state["services"][name]
            write_state(root, state)
        return {
            "stopped": not incomplete,
            "terminated": stopped,
            "skipped": skipped,
            "incomplete": incomplete,
            "state_file": plan["state_file"],
        }


def cleanup_supervised(root: Path, records: dict, plan: dict) -> dict:
    """Clean only this supervisor's receipts, never processes from a later start."""
    errors = {}
    with lock_for_root(root):
        state = read_state(root)
        for name in reversed(records):
            record = records[name]
            current = state["services"].get(name)
            keys = ("pid", "created", "image", "command")
            if current is None:
                continue
            if not isinstance(current, dict) or any(current.get(k) != record.get(k) for k in keys):
                errors[name] = "Service record changed; refusing replacement process"
                continue
            try:
                if not terminate_owned(record, plan["services"][name]):
                    raise LifecycleError("Supervisor cleanup ownership was lost")
            except (LifecycleError, OSError, psutil.Error) as exc:
                current["stop_error"] = str(exc)
                errors[name] = str(exc)
            else:
                del state["services"][name]
            write_state(root, state)
    return {"stopped": not errors, "incomplete": errors}


class _SupervisorSignal(BaseException):
    pass


def _check_shutdown(requested) -> None:
    """Unwind only at safe checkpoints, never from the asynchronous handler."""
    if requested is not None and requested():
        raise _SupervisorSignal()


def serve(root: Path = ROOT, models_max: int = 1, ready_timeout: float = 180) -> dict:
    """Foreground supervisor for systemd --user; SIGTERM/INT release owned children."""
    root = Path(root).resolve()
    plan = build_plan(root, models_max)
    records = {}
    shutdown_requested = False
    previous = {}

    def shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, shutdown)
        start(
            root,
            models_max,
            ready_timeout=ready_timeout,
            _supervised=True,
            _receipts=records,
            _shutdown_requested=lambda: shutdown_requested,
        )
        while not shutdown_requested:
            for name, record in records.items():
                if not ownership(record, plan["services"][name])["owned"]:
                    raise LifecycleError(
                        f"Supervised {name} exited; see {plan['services'][name]['log']}"
                    )
            time.sleep(0.5)
    except _SupervisorSignal:
        pass
    finally:
        try:
            if records:
                cleanup = cleanup_supervised(root, records, plan)
                if not cleanup["stopped"]:
                    raise LifecycleError(f"Supervisor cleanup incomplete: {cleanup['incomplete']}")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return {"stopped": True, "root": str(root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status", "inspect", "serve"))
    parser.add_argument("--root", type=Path, default=ROOT, help="Isolated project root")
    parser.add_argument("--models-max", type=int, default=1, help="Maximum resident router models")
    parser.add_argument(
        "--dry-run", action="store_true", help="Inspect only; no writes or launches"
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=180,
        help="Readiness deadline in seconds, not an inference/provider timeout",
    )
    args = parser.parse_args(argv)
    try:
        if args.action == "serve" and not args.dry_run:
            result = serve(args.root, args.models_max, args.ready_timeout)
        elif args.action in {"start", "inspect", "serve"}:
            result = start(
                args.root,
                args.models_max,
                args.dry_run or args.action == "inspect",
                args.ready_timeout,
            )
        elif args.action == "stop":
            result = stop(args.root, args.dry_run)
        else:
            result = status(args.root)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        if args.action == "stop" and not args.dry_run and not result.get("stopped"):
            return 1
        return 0
    except (LifecycleError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc), "action": args.action}, indent=2, ensure_ascii=True))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrupted; newly started processes were cleaned up"}))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
