"""Exercise real installer functions without installing into the host."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

INSTALLER = Path(__file__).parents[1] / "scripts" / "cloud-install.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not installed")


def run_function(script: str, **variables: str) -> subprocess.CompletedProcess:
    definitions = INSTALLER.read_text(encoding="utf-8").rsplit('main "$@"', 1)[0]
    env = {**os.environ, "PY": sys.executable.replace("\\", "/"), **variables}
    return subprocess.run(
        [BASH],
        input=definitions + "\n" + script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        timeout=30,
    )


@pytest.fixture
def provider():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            authorized = (
                self.path == "/health" or self.headers.get("Authorization") == "Bearer fixture-key"
            )
            status = 200 if authorized else 401
            body = json.dumps({"data": [{"id": "fixture-model"}]}).encode()
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_installer_subprocess_is_utf8_under_non_utf8_host_locale(monkeypatch):
    import locale

    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")
    result = run_function("printf 'UTF8_OK'")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "UTF8_OK"


def test_provider_probe_sends_actual_key(provider):
    url, seen = provider
    result = run_function(
        'probe_provider; printf "%s" "$PROVIDER_MODELS"',
        PROVIDER_URL=url,
        PROVIDER_KEY="fixture-key",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "fixture-model"
    assert seen == [("/models", "Bearer fixture-key")]


@pytest.mark.parametrize("saved", [None, "false", "true"])
def test_installer_enables_new_sleep_but_preserves_saved_choice(tmp_path, provider, saved):
    from dotenv import dotenv_values

    url, _ = provider
    env_file = tmp_path / ".env"
    if saved is not None:
        env_file.write_text(f"ROI_SLEEP_ENABLED={saved}\n", encoding="utf-8")
    result = run_function(
        "configure_models() { :; }; configure",
        INSTALL_DIR=tmp_path.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="false",
        PROVIDER_URL=url,
        PROVIDER_KEY="fixture-key",
        SWARM_KEY="fixture-token",
        HOST="127.0.0.1",
        PORT="8765",
    )
    assert result.returncode == 0, result.stderr
    values = dotenv_values(env_file)
    assert values["ROI_SLEEP_ENABLED"] == ("true" if saved is None else saved)
    assert values["ROI_SLEEP_AUTO_IMPROVE"] == "false"


def test_reconfigure_preserves_unknown_env_and_quotes_values(tmp_path, provider):
    from dotenv import dotenv_values

    url, _ = provider
    env_file = tmp_path / ".env"
    env_file.write_text("ROI_CUSTOM_SETTING=keep\nROI_SLEEP_ENABLED=true\n", encoding="utf-8")
    result = run_function(
        "configure_models() { :; }; configure",
        INSTALL_DIR=tmp_path.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="true",
        PROVIDER_URL=url,
        PROVIDER_KEY="fixture-key",
        SWARM_KEY="token with # apostrophe'",
        HOST="127.0.0.1",
        PORT="8765",
    )
    assert result.returncode == 0, result.stderr
    values = dotenv_values(env_file)
    assert values["ROI_CUSTOM_SETTING"] == "keep"
    assert values["ROI_SWARM_API_KEY"] == "token with # apostrophe'"
    assert values["ROI_SLEEP_ENABLED"] == "true"


def test_provider_reconfigure_keeps_client_token_and_bind_settings(tmp_path, provider):
    from dotenv import dotenv_values

    url, _ = provider
    env_file = tmp_path / ".env"
    env_file.write_text(
        'ROI_SWARM_API_KEY="existing-fixture-token"\nROI_HOST="127.0.0.1"\nROI_PORT="9123"\n',
        encoding="utf-8",
    )
    result = run_function(
        "configure_models() { :; }; configure",
        INSTALL_DIR=tmp_path.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="true",
        PROVIDER_URL=url,
        PROVIDER_KEY="fixture-key",
        SWARM_KEY="",
        HOST="",
        PORT="",
    )
    assert result.returncode == 0, result.stderr
    values = dotenv_values(env_file)
    assert values["ROI_SWARM_API_KEY"] == "existing-fixture-token"
    assert values["ROI_HOST"] == "127.0.0.1"
    assert values["ROI_PORT"] == "9123"


def test_invalid_provider_probe_does_not_overwrite_configuration(tmp_path, provider):
    url, _ = provider
    env_file = tmp_path / ".env"
    original = "ROI_CUSTOM_SETTING=keep\n"
    env_file.write_text(original, encoding="utf-8")
    result = run_function(
        "configure_models() { :; }; configure",
        INSTALL_DIR=tmp_path.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="true",
        PROVIDER_URL=url,
        PROVIDER_KEY="wrong-fixture-key",
        SWARM_KEY="fixture-token",
        HOST="127.0.0.1",
        PORT="8765",
    )
    assert result.returncode != 0
    assert env_file.read_text(encoding="utf-8") == original


def test_repeat_install_preserves_all_individual_role_overrides(tmp_path):
    import yaml

    roles = {
        role: {"model": "base"} for role in ("router", "planner", "worker", "critic", "synthesizer")
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config/settings.yaml").write_text(yaml.safe_dump({"models": roles}))
    selection = tmp_path / "data/model-settings.json"
    original = {"roles": {role: f"custom-{role}" for role in roles}}
    selection.write_text(json.dumps(original))
    result = run_function(
        "configure_models",
        INSTALL_DIR=tmp_path.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="false",
        MAIN_MODEL="",
        WORKER_MODEL="",
        PROVIDER_MODELS="",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(selection.read_text()) == original


def test_update_keeps_locally_committed_sleep_improvements(tmp_path):
    def git(folder, *args):
        return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()

    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "Fixture")
    git(source, "config", "user.email", "fixture@example.test")
    (source / "app.txt").write_text("base")
    git(source, "add", "app.txt")
    git(source, "commit", "-m", "base")
    installed = tmp_path / "installed"
    subprocess.run(["git", "clone", str(source), str(installed)], check=True, capture_output=True)
    git(installed, "config", "user.name", "Fixture")
    git(installed, "config", "user.email", "fixture@example.test")
    (installed / "app.txt").write_text("local improvement")
    git(installed, "commit", "-am", "verified local improvement")
    expected = git(installed, "rev-parse", "HEAD")
    result = run_function(
        "fetch_code",
        INSTALL_DIR=installed.as_posix(),
        IS_ROOT="false",
        BRANCH="main",
    )
    assert result.returncode == 0, result.stderr
    assert git(installed, "rev-parse", "HEAD") == expected
    assert (installed / "app.txt").read_text() == "local improvement"


@pytest.mark.parametrize("directory", ["/", "/tmp", "/opt", "/usr", "/home"])
def test_installer_refuses_dangerous_install_directories(directory):
    definitions = INSTALLER.read_text(encoding="utf-8").rsplit('main "$@"', 1)[0]
    result = subprocess.run(
        [BASH],
        input=definitions + '\ncheck_system() { fail "REACHED_INSTALL"; }; main --dir "$BAD_DIR"',
        env={**os.environ, "BAD_DIR": directory},
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "REACHED_INSTALL" not in result.stderr
    assert "каталог" in result.stderr


def test_update_routes_git_through_owner_before_reading_repository(tmp_path):
    installed = tmp_path / "installed"
    (installed / ".git").mkdir(parents=True)
    result = run_function(
        r"""
run_owned() {
    printf 'owned:%s\n' "$*" >>"$TRACE"
    if [ "$1" = git ] && [[ "$*" == *rev-parse* ]]; then printf fixture; fi
}
git() { printf 'ROOT_GIT:%s\n' "$*" >>"$TRACE"; return 1; }
fetch_code
""",
        INSTALL_DIR=installed.as_posix(),
        IS_ROOT="true",
        BRANCH="main",
        TRACE=(tmp_path / "trace").as_posix(),
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "trace").read_text()
    assert "ROOT_GIT" not in calls
    assert "status --porcelain" in calls
    assert "core.fsmonitor=false" in calls
    assert "core.hooksPath=/dev/null" in calls


def test_root_owner_wrapper_sanitizes_home_and_keeps_explicit_credentials(tmp_path):
    result = run_function(
        r"""
runuser() {
    [ "$1 $2 $3" = '-u roi-swarm --' ] || exit 91
    shift 3
    "$@"
}
ROI_PROVIDER_API_KEY='fixture root credential' run_owned "$PY" -c \
    'import json, os; names=["HOME", "PYTHONPATH", "ROI_PROVIDER_API_KEY"]
print(json.dumps({k: os.getenv(k) for k in names}))'
""",
        IS_ROOT="true",
        INSTALL_OWNER="roi-swarm",
        INSTALL_DIR=tmp_path.as_posix(),
        PYTHONPATH="/root/untrusted",
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert Path(values.pop("HOME")) == tmp_path
    assert values == {"PYTHONPATH": None, "ROI_PROVIDER_API_KEY": "fixture root credential"}


def test_sanitized_owner_path_contains_system_admin_executables(tmp_path):
    result = run_function(
        r"""
runuser() {
    case ":$PATH:" in *:/usr/sbin:*) ;; *) exit 91 ;; esac
    case ":$PATH:" in *:/sbin:*) ;; *) exit 92 ;; esac
    shift 3
    "$@"
}
run_owned /bin/true
""",
        IS_ROOT="true",
        INSTALL_OWNER="roi-swarm",
        INSTALL_DIR=tmp_path.as_posix(),
    )
    assert result.returncode == 0, result.stderr


def test_application_python_and_uv_always_use_owner_wrapper(tmp_path, provider):
    url, _ = provider
    trace = tmp_path / "trace"
    result = run_function(
        r"""
run_owned() {
    printf 'owned:%s\n' "$*" >>"$TRACE"
    if [ "$1" = fixture-uv ]; then return; fi
    if [ "$1" = "$PY" ] && [[ "$*" == *'import roi_swarm.api'* ]]; then return; fi
    if [ "$1" = "$PY" ]; then shift; "$TEST_PY" "$@"; else "$@"; fi
}
TEST_PY="$PY"
UV=fixture-uv
install_env
configure_models() { :; }
configure
""",
        IS_ROOT="true",
        INSTALL_DIR=tmp_path.as_posix(),
        TRACE=trace.as_posix(),
        NON_INTERACTIVE="true",
        RECONFIGURE="true",
        PROVIDER_URL=url,
        PROVIDER_KEY="fixture-key",
        SWARM_KEY="fixture-token",
        HOST="127.0.0.1",
        PORT="8765",
    )
    assert result.returncode == 0, result.stderr
    calls = trace.read_text()
    assert "fixture-uv sync" in calls
    assert "import roi_swarm.api" in calls
    assert ".env" in calls


def test_provisioning_refuses_unmarked_install_or_sibling_without_mutation(tmp_path):
    installed = tmp_path / "installed"
    sibling = tmp_path / "installed-worktrees"
    sibling.mkdir()
    sentinel = sibling / "unrelated.txt"
    sentinel.write_text("unrelated")
    result = run_function(
        'state_path() { printf "%s" "$STATE"; }; prepare_owner',
        INSTALL_DIR=installed.as_posix(),
        STATE=(tmp_path / "state").as_posix(),
        IS_ROOT="false",
        SERVICE="roi-swarm",
    )
    assert result.returncode != 0
    assert "немаркирован" in result.stderr
    assert sentinel.read_text() == "unrelated"
    assert not installed.exists()


def test_preexisting_service_account_is_not_adopted(tmp_path):
    trace = tmp_path / "trace"
    result = run_function(
        r"""
state_path() { printf '%s' "$STATE"; }
id() { if [ "${1:-}" = roi-swarm ]; then return 0; fi; command id "$@"; }
userdel() { printf deleted >>"$TRACE"; }
prepare_owner
""",
        INSTALL_DIR=(tmp_path / "installed").as_posix(),
        STATE=(tmp_path / "state").as_posix(),
        IS_ROOT="true",
        SERVICE="roi-swarm",
        TRACE=trace.as_posix(),
    )
    assert result.returncode != 0
    assert "пользователь" in result.stderr
    assert not trace.exists()


def test_preparation_marks_only_new_owned_directories(tmp_path):
    installed = tmp_path / "installed"
    state = tmp_path / "state"
    result = run_function(
        'state_path() { printf "%s" "$STATE"; }; prepare_owner; prepare_owner',
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        IS_ROOT="false",
        SERVICE="roi-swarm",
    )
    assert result.returncode == 0, result.stderr
    assert installed.is_dir()
    assert installed.with_name("installed-worktrees").is_dir()
    assert state.is_file()


def test_main_prepares_owner_before_uv_fetch_or_python(tmp_path):
    trace = tmp_path / "trace"
    result = run_function(
        r"""
validate_install_dir() { :; }
check_system() { :; }
ensure_packages() { :; }
prepare_owner() { printf 'owner\n' >>"$TRACE"; }
ensure_uv() { printf 'uv\n' >>"$TRACE"; }
fetch_code() { printf 'git\n' >>"$TRACE"; }
install_env() { printf 'python\n' >>"$TRACE"; }
configure() { :; }
start_foreground_hint() { :; }
finish() { :; }
main --no-service --non-interactive
""",
        TRACE=trace.as_posix(),
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text().splitlines() == ["owner", "uv", "git", "python"]


def test_root_uv_is_shared_not_discovered_in_root_home(tmp_path):
    result = run_function(
        r"""
provision_uv() { [ "$1" = /usr/local/bin ]; }
validate_trusted_uv() { :; }
run_owned() { printf 'owned:%s\n' "$*"; }
ensure_uv
""",
        IS_ROOT="true",
        INSTALL_DIR=tmp_path.as_posix(),
        HOME="/root",
    )
    assert result.returncode == 0, result.stderr
    assert "owned:/usr/local/bin/uv --version" in result.stdout
    assert "/root/.local" not in result.stdout


def managed_fixture(tmp_path):
    installed = tmp_path / "installed"
    state = tmp_path / "state"
    result = run_function(
        'state_path() { printf "%s" "$STATE"; }; prepare_owner',
        IS_ROOT="false",
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        SERVICE="roi-swarm",
    )
    assert result.returncode == 0, result.stderr
    (installed / "src/roi_swarm").mkdir(parents=True)
    (installed / "src/roi_swarm/server.py").write_text("# fixture")
    (installed / "install.sh").write_text("# fixture")
    return installed, state


def test_uninstall_validates_mismatched_unit_before_stopping_or_deleting(tmp_path):
    installed, state = managed_fixture(tmp_path)
    unit = tmp_path / "roi-swarm.service"
    unit.write_text("# ROI Swarm installer v1\nWorkingDirectory=/unrelated\nUser=someone\n")
    trace = tmp_path / "trace"
    result = run_function(
        r"""
state_path() { printf '%s' "$STATE"; }
system_unit_path() { printf '%s' "$UNIT"; }
user_unit_path() { printf '%s/missing' "$STATE"; }
systemctl() { printf stopped >>"$TRACE"; }
uninstall
""",
        IS_ROOT="false",
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        UNIT=unit.as_posix(),
        TRACE=trace.as_posix(),
        SERVICE="roi-swarm",
        SUDO="",
        NON_INTERACTIVE="true",
    )
    assert result.returncode != 0
    assert installed.exists() and unit.exists() and state.exists()
    assert not trace.exists()


def test_uninstall_preserves_replaced_unrelated_worktree_directory(tmp_path):
    installed, state = managed_fixture(tmp_path)
    sibling = installed.with_name("installed-worktrees")
    sibling.rename(tmp_path / "original-worktrees")
    sibling.mkdir()
    sentinel = sibling / "unrelated"
    sentinel.write_text("keep")
    result = run_function(
        r"""
state_path() { printf '%s' "$STATE"; }
system_unit_path() { printf '%s/missing-unit' "$STATE"; }
user_unit_path() { printf '%s/missing-user-unit' "$STATE"; }
uninstall
""",
        IS_ROOT="false",
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        SERVICE="roi-swarm",
        SUDO="",
        NON_INTERACTIVE="true",
    )
    assert result.returncode != 0
    assert sentinel.read_text() == "keep"
    assert installed.exists() and state.exists()


def test_service_setup_never_chowns_existing_tree(tmp_path):
    result = run_function(
        r"""
validate_units() { :; }
system_unit_path() { printf '%s' "$UNIT"; }
sudo() { if [ "${1:-}" = tee ]; then command tee "$2"; fi; return 0; }
chown() { fail ROOT_CHOWN; }
useradd() { fail LATE_USERADD; }
wait_healthy() { return 0; }
install_service
""",
        IS_ROOT="true",
        INSTALL_OWNER="roi-swarm",
        INSTALL_DIR=tmp_path.as_posix(),
        SERVICE="roi-swarm",
        SUDO="sudo",
        UNIT=(tmp_path / "unit").as_posix(),
    )
    assert result.returncode == 0, result.stderr


def test_finish_uses_encoded_fragment_instead_of_query_token(tmp_path):
    result = run_function(
        "public_host() { printf 127.0.0.1; }; finish",
        INSTALL_DIR=tmp_path.as_posix(),
        SERVICE_SCOPE="none",
        HOST="127.0.0.1",
        PORT="8765",
        SWARM_KEY="fixture token#&",
    )
    assert result.returncode == 0, result.stderr
    assert "/dashboard#token=fixture%20token%23%26" in result.stdout
    assert "/dashboard?token=" not in result.stdout


def test_inaccessible_parent_stops_before_git_uv_or_unit_activation(tmp_path):
    installed = tmp_path / "installed"
    state = tmp_path / "state"
    result = run_function(
        r"""
state_path() { printf '%s' "$STATE"; }
run_owned() { return 1; }
prepare_owner
printf SHOULD_NOT_REACH
""",
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        IS_ROOT="false",
        SERVICE="roi-swarm",
    )
    assert result.returncode != 0
    assert "не имеет доступа" in result.stderr
    assert "SHOULD_NOT_REACH" not in result.stdout


def test_uninstall_removes_only_marked_directories(tmp_path):
    installed, state = managed_fixture(tmp_path)
    result = run_function(
        r"""
state_path() { printf '%s' "$STATE"; }
system_unit_path() { printf '%s/missing-unit' "$STATE"; }
user_unit_path() { printf '%s/missing-user-unit' "$STATE"; }
uninstall
""",
        IS_ROOT="false",
        INSTALL_DIR=installed.as_posix(),
        STATE=state.as_posix(),
        SERVICE="roi-swarm",
        SUDO="",
        NON_INTERACTIVE="true",
    )
    assert result.returncode == 0, result.stderr
    assert not installed.exists() and not state.exists()
    assert not installed.with_name("installed-worktrees").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux UID/permissions")
def test_real_root_wrapper_drops_uid_before_git_or_python():
    import tempfile

    if os.geteuid() != 0 or not shutil.which("runuser"):
        pytest.skip("requires disposable root Linux environment with runuser")
    with tempfile.TemporaryDirectory(prefix="roi-installer-uid-", dir="/tmp") as folder:
        root = Path(folder)
        root.chmod(0o755)
        result = run_function(
            r"""
# No account creation: use the existing nobody account to exercise real setuid.
runuser() { shift 3; command runuser -u nobody -- "$@"; }
run_owned "$PY" -c 'import os; assert os.geteuid() != 0; print("unprivileged")'
""",
            IS_ROOT="true",
            INSTALL_OWNER="roi-swarm",
            INSTALL_DIR=root.as_posix(),
            PY="/usr/bin/python3",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "unprivileged"


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux UID/permissions")
def test_real_private_parent_is_refused_without_chmod():
    import tempfile

    if os.geteuid() != 0 or not shutil.which("runuser"):
        pytest.skip("requires disposable root Linux environment with runuser")
    with tempfile.TemporaryDirectory(prefix="roi-installer-private-", dir="/tmp") as folder:
        root = Path(folder)
        root.chmod(0o700)
        installed = root / "installed"
        result = run_function(
            r"""
state_path() { printf '%s' "$STATE"; }
run_owned() { command runuser -u nobody -- /bin/sh -c 'cd "$1"' sh "$INSTALL_DIR"; }
prepare_owner
""",
            IS_ROOT="false",
            INSTALL_DIR=installed.as_posix(),
            STATE=(root / "state").as_posix(),
            SERVICE="roi-swarm",
        )
        assert result.returncode != 0
        assert "не имеет доступа" in result.stderr
        assert root.stat().st_mode & 0o777 == 0o700


def test_health_probe_sends_actual_swarm_key(provider):
    url, seen = provider
    result = run_function(
        "wait_healthy",
        PORT=url.rsplit(":", 1)[1],
        SWARM_KEY="fixture-key",
    )
    assert result.returncode == 0, result.stderr
    assert seen == [("/health", None), ("/v1/models", "Bearer fixture-key")]
