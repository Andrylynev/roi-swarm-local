"""OS-owned worker subtree, attached before the host receives any task input.

Launch with ``**scope.launch_kwargs``, call ``scope.attach(host.pid)``, then send
its task. Windows launch is suspended; attach assigns the Job and resumes the
launcher before it can spawn even a venv Python shim. Merge any other Windows
creationflags with bitwise OR, never discard the scope's CREATE_SUSPENDED flag.
A host which has already exited may be attached as a no-op: this is
safe only because the host must not spawn tools before receiving that task.
Windows Jobs contain descendants even after the host exits. POSIX process
groups cannot contain descendants that deliberately create a new session.

``close()`` is synchronous, idempotent after success, and raises rather than
claiming cleanup succeeded without confirmation. On failure it retains its OS
ownership so the caller can quarantine the worker and retry cleanup.
"""

import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

_CLEANUP_TIMEOUT = 10.0
_POLL_INTERVAL = 0.02

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = wintypes.LONG
    _ntdll.RtlNtStatusToDosError.argtypes = [wintypes.LONG]
    _ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG

    def _active_processes(handle: int) -> int:
        accounting = _BasicAccountingInformation()
        if not _kernel32.QueryInformationJobObject(
            handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return accounting.ActiveProcesses

    def _job_pids(handle: int) -> list[int]:
        # OS-owned membership, not parent-PID traversal (which loses orphans).
        capacity = 32
        while True:
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if _kernel32.QueryInformationJobObject(handle, 3, buffer, len(buffer), None):
                count = wintypes.DWORD.from_buffer(buffer, 4).value
                return list((ctypes.c_size_t * count).from_buffer(buffer, 8))
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, wintypes.DWORD.from_buffer(buffer).value)


def _posix_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    # Linux can retain dead orphans as zombies indefinitely under a non-reaping
    # PID 1. They cannot run tools; don't mistake them for a live process group.
    if sys.platform.startswith("linux"):
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            return False
        except (OSError, ValueError, IndexError):
            pass  # Without /proc visibility require the group to disappear.
    return True


class WorkerProcessScope:
    """Own one worker's descendants; never silently fall back from a Job."""

    def __init__(self) -> None:
        self._closed = False
        self._attached = False
        self._pgid: int | None = None
        self._handle: int | None = None
        self._wait_handles: list[int] = []
        if os.name == "nt":
            handle = _kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            if not _kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                error = ctypes.WinError(ctypes.get_last_error())
                _kernel32.CloseHandle(handle)
                raise error
            # Null SECURITY_ATTRIBUTES makes this handle non-inheritable.
            self._handle = handle

    @property
    def launch_kwargs(self) -> dict[str, bool | int]:
        """Launch suspended on Windows; merge creationflags with bitwise OR.

        ``attach`` assigns the still-suspended launcher to its Job and resumes
        it, so a venv shim cannot create uncontained Python descendants first.
        """
        return {"creationflags": 0x00000004} if os.name == "nt" else {"start_new_session": True}

    def attach(self, pid: int) -> None:
        """Attach once, before task delivery. An already exited host is allowed."""
        if self._closed:
            raise RuntimeError("Worker process scope is closed")
        if self._attached:
            raise RuntimeError("Worker process scope already has a host")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            raise ValueError("Expected a positive worker PID other than the current process")
        if os.name == "nt":
            if pid > 0xFFFFFFFF:
                raise ValueError("Worker PID does not fit a Windows DWORD")
            # SET_QUOTA | TERMINATE | SUSPEND_RESUME, plus SYNCHRONIZE for exit.
            process = _kernel32.OpenProcess(0x00100901, False, pid)
            if not process:
                error = ctypes.get_last_error()
                if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
                    self._attached = True
                    return
                raise ctypes.WinError(error)
            try:
                state = _kernel32.WaitForSingleObject(process, 0)
                if state == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
                if state != 0:
                    if not _kernel32.AssignProcessToJobObject(self._handle, process):
                        error = ctypes.get_last_error()
                        if _kernel32.WaitForSingleObject(process, 0) != 0:
                            raise ctypes.WinError(error)
                    else:
                        self._attached = True
                        # Popen closes the primary thread handle; resume via a
                        # process handle only after OS containment is installed.
                        status = _ntdll.NtResumeProcess(process)
                        if status < 0:
                            raise ctypes.WinError(_ntdll.RtlNtStatusToDosError(status))
                self._attached = True
            finally:
                if not _kernel32.CloseHandle(process):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                self._attached = True
                return
            if pgid != pid or pgid == os.getpgrp():
                raise ValueError("Worker must be launched with start_new_session=True")
            self._pgid = pid
            self._attached = True

    def close(self) -> None:
        """Kill the subtree and confirm quiescence, or raise and retain ownership."""
        if self._closed:
            return
        deadline = time.monotonic() + _CLEANUP_TIMEOUT
        if os.name == "nt":
            # ActiveProcesses reaches zero before all process handles signal.
            # Pin OS Job members before termination, then wait for their exit.
            for pid in _job_pids(self._handle):
                process = _kernel32.OpenProcess(0x00100000, False, pid)
                if not process:
                    error = ctypes.get_last_error()
                    if error == 87:  # Already exited and its PID is gone.
                        continue
                    raise ctypes.WinError(error)
                self._wait_handles.append(process)
            if not _kernel32.TerminateJobObject(self._handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            while _active_processes(self._handle):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker Job still has active processes after termination")
                time.sleep(_POLL_INTERVAL)
            while self._wait_handles:
                process = self._wait_handles[-1]
                milliseconds = max(0, int((deadline - time.monotonic()) * 1000))
                state = _kernel32.WaitForSingleObject(process, milliseconds)
                if state == 0x102:
                    raise TimeoutError("Worker process handle is not signaled after termination")
                if state != 0:
                    raise ctypes.WinError(ctypes.get_last_error())
                if not _kernel32.CloseHandle(process):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._wait_handles.pop()
            if not _kernel32.CloseHandle(self._handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = None
        elif self._pgid is not None:
            with suppress(ProcessLookupError):
                os.killpg(self._pgid, signal.SIGKILL)
            while _posix_group_alive(self._pgid):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker process group still exists after SIGKILL")
                time.sleep(_POLL_INTERVAL)
            self._pgid = None
        self._closed = True
