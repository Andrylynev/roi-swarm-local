#!/usr/bin/env bash
# run_owned deliberately changes HOME/USER only in its privilege-drop subshell.
# Guard chains intentionally fail if ANY predicate is false, not an if/else.
# shellcheck disable=SC2015,SC2030,SC2031
# ROI Swarm — установка одной командой:
#
#   curl -fsSL https://raw.githubusercontent.com/Andrylynev/roi-swarm/main/install.sh | bash
#
# Без вопросов (CI, Ansible, cloud-init):
#
#   curl -fsSL .../install.sh | ROI_PROVIDER_BASE_URL=https://host/v1 \
#     ROI_PROVIDER_API_KEY=sk-... bash -s -- --non-interactive
#
# Что делает: ставит git/curl (если нет), uv и Python 3.11+, клонирует рой,
# создаёт окружение, спрашивает URL и ключ провайдера, проверяет их, пишет .env
# (chmod 600), генерирует токен доступа к рою, регистрирует systemd-сервис и
# запускает сервер. Повторный запуск обновляет установку и сохраняет .env.
#
# Флаги / переменные окружения:
#   --dir PATH            ROI_INSTALL_DIR   каталог (root: /opt/roi-swarm, иначе ~/.roi-swarm)
#   --branch NAME         ROI_BRANCH        ветка или тег (main)
#   --repo URL            ROI_REPO          git-репозиторий
#   --provider-url URL    ROI_PROVIDER_BASE_URL
#   --provider-key KEY    ROI_PROVIDER_API_KEY
#   --main-model ID       ROI_MAIN_MODEL    центральная модель (planner/synthesizer)
#   --worker-model ID     ROI_WORKER_MODEL  модель воркеров (router/worker/critic)
#   --host ADDR           ROI_HOST          0.0.0.0 — доступ из сети, 127.0.0.1 — локально
#   --port N              ROI_PORT          8765
#   --api-key TOKEN       ROI_SWARM_API_KEY токен для /v1 и /dashboard (иначе генерируется)
#   --reconfigure                           заново спросить провайдера, даже если .env есть
#   --no-service                            не регистрировать systemd-сервис
#   --non-interactive                       ничего не спрашивать
#   --uninstall                             остановить сервис и удалить установку
set -euo pipefail

main() {
    REPO="${ROI_REPO:-https://github.com/Andrylynev/roi-swarm.git}"
    BRANCH="${ROI_BRANCH:-main}"
    INSTALL_DIR="${ROI_INSTALL_DIR:-}"
    PROVIDER_URL="${ROI_PROVIDER_BASE_URL:-}"
    PROVIDER_KEY="${ROI_PROVIDER_API_KEY:-}"
    MAIN_MODEL="${ROI_MAIN_MODEL:-}"
    WORKER_MODEL="${ROI_WORKER_MODEL:-}"
    HOST="${ROI_HOST:-}"
    PORT="${ROI_PORT:-}"
    SWARM_KEY="${ROI_SWARM_API_KEY:-}"
    NON_INTERACTIVE=false
    RECONFIGURE=false
    WITH_SERVICE=true
    UNINSTALL=false
    SERVICE=roi-swarm

    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) INSTALL_DIR="$2"; shift 2 ;;
            --branch) BRANCH="$2"; shift 2 ;;
            --repo) REPO="$2"; shift 2 ;;
            --provider-url) PROVIDER_URL="$2"; shift 2 ;;
            --provider-key) PROVIDER_KEY="$2"; shift 2 ;;
            --main-model) MAIN_MODEL="$2"; shift 2 ;;
            --worker-model) WORKER_MODEL="$2"; shift 2 ;;
            --host) HOST="$2"; shift 2 ;;
            --port) PORT="$2"; shift 2 ;;
            --api-key) SWARM_KEY="$2"; shift 2 ;;
            --reconfigure) RECONFIGURE=true; shift ;;
            --no-service) WITH_SERVICE=false; shift ;;
            --non-interactive|-y) NON_INTERACTIVE=true; shift ;;
            --uninstall) UNINSTALL=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) fail "неизвестный флаг: $1" ;;
        esac
    done

    if [ "$(id -u)" -eq 0 ]; then
        IS_ROOT=true; SUDO=""
        INSTALL_DIR="${INSTALL_DIR:-/opt/roi-swarm}"
    else
        IS_ROOT=false; SUDO="sudo"
        INSTALL_DIR="${INSTALL_DIR:-$HOME/.roi-swarm}"
    fi
    if [ "$NON_INTERACTIVE" != true ] && ! { : </dev/tty; } 2>/dev/null; then
        NON_INTERACTIVE=true  # нет терминала (cloud-init, CI): только env/флаги
    fi

    validate_install_dir
    if [ "$UNINSTALL" = true ]; then uninstall; exit 0; fi

    banner
    step "Проверка системы"; check_system
    step "Владелец установки"; prepare_owner
    step "Зависимости"; ensure_packages; ensure_uv
    step "Код роя → $INSTALL_DIR"; fetch_code
    step "Python-окружение"; install_env
    step "Провайдер моделей"; configure
    if [ "$WITH_SERVICE" = true ] && has_systemd; then
        step "Сервис systemd"; install_service
    else
        step "Запуск без systemd"; start_foreground_hint
    fi
    finish
}

usage() {
    cat <<'EOF'
ROI Swarm installer
  curl -fsSL https://raw.githubusercontent.com/Andrylynev/roi-swarm/main/install.sh | bash
  ... | bash -s -- [--provider-url URL] [--provider-key KEY] [--main-model ID]
                   [--worker-model ID] [--host 0.0.0.0|127.0.0.1] [--port 8765]
                   [--api-key TOKEN] [--dir PATH] [--branch NAME] [--reconfigure]
                   [--no-service] [--non-interactive] [--uninstall]
EOF
}

# ─── вывод ────────────────────────────────────────────────────────────────
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else B=""; G=""; Y=""; R=""; C=""; N=""; fi
banner() { printf '\n%s  ROI Swarm — коллективный разум%s\n  установка облачного роя (без локальных моделей)\n\n' "$B$C" "$N"; }
step() { printf '\n%s▸ %s%s\n' "$B" "$1" "$N"; }
ok() { printf '  %s✓%s %s\n' "$G" "$N" "$1"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$1" >&2; }
fail() { printf '\n%s✗ %s%s\n' "$R" "$1" "$N" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

ask() {  # ask VAR "вопрос" "по умолчанию"
    local __var="$1" prompt="$2" default="${3:-}" reply=""
    if [ "$NON_INTERACTIVE" = true ]; then printf -v "$__var" '%s' "$default"; return; fi
    if [ -n "$default" ]; then prompt="$prompt [$default]"; fi
    read -r -p "  $prompt: " reply </dev/tty || true
    printf -v "$__var" '%s' "${reply:-$default}"
}
ask_secret() {  # ask_secret VAR "вопрос" — ввод не отображается
    local __var="$1" reply=""
    if [ "$NON_INTERACTIVE" = true ]; then return; fi
    read -r -s -p "  $2: " reply </dev/tty || true
    printf '\n' >/dev/tty
    printf -v "$__var" '%s' "$reply"
}
confirm() {  # confirm "вопрос" y|n
    local reply
    if [ "$NON_INTERACTIVE" = true ]; then [ "$2" = y ]; return; fi
    read -r -p "  $1 [$( [ "$2" = y ] && echo Y/n || echo y/N )]: " reply </dev/tty || true
    reply="${reply:-$2}"
    case "$reply" in [YyДд]*) return 0 ;; *) return 1 ;; esac
}

# ─── шаги ─────────────────────────────────────────────────────────────────
validate_install_dir() {
    # This same path is chowned and recursively removed on uninstall.
    case "$INSTALL_DIR" in
        /*) ;;
        *) fail "каталог установки должен быть абсолютным" ;;
    esac
    INSTALL_DIR="${INSTALL_DIR%/}"
    case "$INSTALL_DIR" in
        ""|/|/opt|/usr|/home|/root|/tmp|/var|/etc|/bin|/sbin|/lib|/srv|/mnt|/media|/Users|"$HOME")
            fail "небезопасный каталог установки: $INSTALL_DIR" ;;
        *[[:space:]]*|*%*|*\\*|*/../*|*/..|*/./*|*/.|*//* )
            fail "каталог установки содержит недопустимые компоненты" ;;
    esac
    local parent="$INSTALL_DIR"
    while [ "$parent" != / ]; do
        [ ! -L "$parent" ] || fail "каталог установки не должен проходить через symlink: $parent"
        parent="$(dirname "$parent")"
    done
}

# The receipt lives OUTSIDE the service-writable tree. Never source it as shell.
state_path() {
    if [ "$IS_ROOT" = true ]; then printf /var/lib/roi-swarm-installer/installation;
    else printf '%s/.local/state/roi-swarm-installer/installation' "$HOME"; fi
}
state_body() {
    printf 'ROI Swarm installer v1\n%s\n%s\n%s\n%s\n' "$INSTALL_DIR" "$INSTALL_OWNER" \
        "$(id -u "$INSTALL_OWNER")" "$(stat -c '%d:%i' "$INSTALL_DIR" 2>/dev/null || stat -f '%d:%i' "$INSTALL_DIR")"
    stat -c '%d:%i' "$INSTALL_DIR-worktrees" 2>/dev/null || stat -f '%d:%i' "$INSTALL_DIR-worktrees"
}
validate_receipt() {
    local receipt; receipt="$(state_path)"
    [ -f "$receipt" ] && [ ! -L "$receipt" ] || fail "нет маркировки установщика: $receipt"
    [ -d "$INSTALL_DIR" ] && [ ! -L "$INSTALL_DIR" ] \
        && [ -d "$INSTALL_DIR-worktrees" ] && [ ! -L "$INSTALL_DIR-worktrees" ] \
        || fail "каталоги установки не совпадают с маркировкой"
    id "$INSTALL_OWNER" >/dev/null 2>&1 || fail "нет пользователя установки: $INSTALL_OWNER"
    if [ "$IS_ROOT" = true ]; then
        [ "$(stat -c '%u:%a' "$(dirname "$receipt")")" = 0:700 ] \
            && [ ! -L "$(dirname "$receipt")" ] \
            && [ "$(stat -c '%u:%a' "$receipt")" = 0:600 ] \
            || fail "небезопасная маркировка установщика"
    fi
    cmp -s "$receipt" <(state_body) || fail "установка/пользователь не совпадают с маркировкой"
}
prepare_owner() {
    local receipt
    receipt="$(state_path)"
    if [ "$IS_ROOT" = true ]; then INSTALL_OWNER="$SERVICE"; else INSTALL_OWNER="$(id -un)"; fi
    if [ -e "$receipt" ] || [ -L "$receipt" ]; then
        validate_receipt
    else
        # Even an empty pre-existing directory/account may belong to someone else.
        [ ! -e "$INSTALL_DIR" ] && [ ! -L "$INSTALL_DIR" ] \
            && [ ! -e "$INSTALL_DIR-worktrees" ] && [ ! -L "$INSTALL_DIR-worktrees" ] \
            || fail "немаркированный каталог установки или worktrees: отказ без изменений"
        if [ "$IS_ROOT" = true ]; then
            ! id "$SERVICE" >/dev/null 2>&1 || fail "пользователь $SERVICE уже существует без маркировки"
            have runuser || fail "нужен runuser для запуска от пользователя $SERVICE"
        fi
        [ ! -L "$(dirname "$receipt")" ] || fail "каталог маркировки не должен быть symlink"
        mkdir -p "$(dirname "$receipt")" "$(dirname "$INSTALL_DIR")"
        chmod 700 "$(dirname "$receipt")"
        if [ "$IS_ROOT" = true ]; then
            useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE"
        fi
        mkdir -- "$INSTALL_DIR" "$INSTALL_DIR-worktrees"
        if [ "$IS_ROOT" = true ]; then
            # Only the two freshly-created, empty directories; never recursive chown.
            chown "$INSTALL_OWNER:" "$INSTALL_DIR" "$INSTALL_DIR-worktrees"
        fi
        (umask 077; state_body >"$receipt")
    fi
    run_owned /bin/true || fail "пользователь $INSTALL_OWNER не имеет доступа к $INSTALL_DIR; выберите доступный --dir (права родителей не меняются)"
}

check_system() {
    OS="$(uname -s)"
    case "$OS" in
        Linux|Darwin) ok "$OS $(uname -m)" ;;
        *) fail "поддерживаются Linux и macOS (на Windows используйте WSL2)" ;;
    esac
    if [ "$IS_ROOT" = false ] && ! have sudo; then SUDO=""; fi
}

ensure_packages() {
    local missing=()
    for tool in git curl; do have "$tool" || missing+=("$tool"); done
    if [ ${#missing[@]} -eq 0 ]; then ok "git, curl"; return; fi
    warn "нет: ${missing[*]} — ставлю"
    if have apt-get; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${missing[@]}" ca-certificates
    elif have dnf; then $SUDO dnf install -y -q "${missing[@]}"
    elif have yum; then $SUDO yum install -y -q "${missing[@]}"
    elif have pacman; then $SUDO pacman -Sy --noconfirm "${missing[@]}"
    elif have apk; then $SUDO apk add --no-cache "${missing[@]}" bash
    elif have brew; then brew install "${missing[@]}"
    else fail "установите вручную: ${missing[*]}"; fi
    ok "${missing[*]} установлены"
}

provision_uv() {
    if [ ! -e "$1/uv" ]; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env -i HOME="$HOME" PATH=/usr/bin:/bin UV_UNMANAGED_INSTALL="$1" sh >/dev/null
    fi
}
validate_trusted_uv() {
    local path=/usr/local/bin/uv mode
    while [ "$path" != / ]; do
        [ ! -L "$path" ] && [ "$(stat -c %u "$path")" = 0 ] \
            || fail "небезопасный путь trusted uv: $path"
        mode="$(stat -c %a "$path")"
        [ "$((8#$mode & 022))" -eq 0 ] || fail "trusted uv доступен для чужой записи: $path"
        path="$(dirname "$path")"
    done
}
ensure_uv() {
    if [ "$IS_ROOT" = true ]; then
        # Never execute root's ~/.local/bin or a service-controlled uv as root.
        UV=/usr/local/bin/uv
        provision_uv /usr/local/bin
        validate_trusted_uv
    else
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if ! have uv; then provision_uv "$HOME/.local/bin"; fi
        have uv || fail "не удалось установить uv"
        UV="$(command -v uv)"
    fi
    ok "uv $(run_owned "$UV" --version)"
}

# All repository operations, including status/fsmonitor and Git config, are unprivileged.
run_owned() {
    if [ "${IS_ROOT:-false}" = true ]; then
        [ "${INSTALL_OWNER:-}" = roi-swarm ] || fail "владелец установки ещё не подготовлен"
        (
            # Clear inherited root Python/Git/uv/loader config before privilege drop.
            # Keep explicit credentials in the environment, never command arguments.
            local name
            for name in $(compgen -e); do
                case "$name" in
                    ROI_PROVIDER_BASE_URL|ROI_PROVIDER_API_KEY|ROI_SWARM_API_KEY|ROI_HOST|ROI_PORT) ;;
                    *) export -n "${name?}" ;;
                esac
            done
            export HOME="$INSTALL_DIR" USER="$INSTALL_OWNER" LOGNAME="$INSTALL_OWNER"
            export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
            export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/.python" UV_CACHE_DIR="$INSTALL_DIR/.cache/uv"
            # Arguments are intentionally expanded by the unprivileged child shell.
            # shellcheck disable=SC2016
            runuser -u "$INSTALL_OWNER" -- /bin/bash -c \
                'cd "$1" && shift && exec "$@"' bash "$INSTALL_DIR" "$@"
        )
    else
        (cd "${INSTALL_DIR:-$PWD}" && "$@")
    fi
}
owned_python() { run_owned "$PY" "$@"; }
owned_git() {
    run_owned git -c core.hooksPath=/dev/null -c core.fsmonitor=false -C "$INSTALL_DIR" "$@"
}

fetch_code() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        local status
        status="$(owned_git status --porcelain --untracked-files=no)" || fail "не удалось проверить git status"
        if [ -n "$status" ]; then
            warn "в $INSTALL_DIR есть локальные правки — код не обновляю"
        else
            # Never reset a clean branch: sleep may have committed verified local work.
            owned_git fetch -q origin "$BRANCH"
            if owned_git merge-base --is-ancestor HEAD FETCH_HEAD; then
                owned_git merge -q --ff-only FETCH_HEAD
                ok "обновлено до $(owned_git rev-parse --short HEAD)"
            else
                warn "есть локальные коммиты/другая история — сохранены; автоматическое обновление пропущено"
            fi
        fi
    else
        run_owned git -c core.hooksPath=/dev/null -c core.fsmonitor=false \
            clone -q --depth 1 --branch "$BRANCH" "$REPO" "$INSTALL_DIR"
        ok "склонировано $(owned_git rev-parse --short HEAD)"
    fi
    # Режим сна коммитит проверенные улучшения в эту копию.
    owned_git config --local user.name >/dev/null 2>&1 || owned_git config user.name "ROI Swarm"
    owned_git config --local user.email >/dev/null 2>&1 || owned_git config user.email "roi-swarm@localhost"
}

install_env() {
    # Интерпретатор кладём внутрь установки, чтобы сервис-пользователь его видел.
    UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/.python" \
        run_owned "$UV" sync --frozen --managed-python --python 3.11 --extra dev -q \
        || fail "uv sync завершился с ошибкой (проверьте доступ владельца к $INSTALL_DIR)"
    PY="$INSTALL_DIR/.venv/bin/python"
    owned_python -c "import roi_swarm.api" || fail "пакет roi_swarm не импортируется от владельца установки"
    ok "$(owned_python --version), зависимости установлены"
}

env_get() {
    owned_python - "$ENV_FILE" "$1" <<'EOF'
import sys
from dotenv import dotenv_values
print(dotenv_values(sys.argv[1], interpolate=False).get(sys.argv[2]) or "", end="")
EOF
}

probe_provider() {  # → PROVIDER_MODELS (список id) или пусто
    PROVIDER_MODELS=""
    local body
    body="$(curl -fsS --max-time 20 -H "Authorization: Bearer $PROVIDER_KEY" \
        "${PROVIDER_URL%/}/models" 2>/dev/null)" || return 1
    PROVIDER_MODELS="$(printf '%s' "$body" | owned_python -c '
import json, sys
try:
    data = json.load(sys.stdin).get("data", [])
except Exception:
    sys.exit(1)
print("\n".join(sorted(str(m.get("id")) for m in data if m.get("id"))))
')" || return 1
}

configure() {
    ENV_FILE="$INSTALL_DIR/.env"
    if [ -f "$ENV_FILE" ]; then
        # Changing upstream credentials must not invalidate existing client connections.
        SWARM_KEY="${SWARM_KEY:-$(env_get ROI_SWARM_API_KEY)}"
        HOST="${HOST:-$(env_get ROI_HOST)}"
        PORT="${PORT:-$(env_get ROI_PORT)}"
        if [ "$RECONFIGURE" != true ]; then
            PROVIDER_URL="${PROVIDER_URL:-$(env_get ROI_PROVIDER_BASE_URL)}"
            PROVIDER_KEY="${PROVIDER_KEY:-$(env_get ROI_PROVIDER_API_KEY)}"
            ok "найден $ENV_FILE — настройки сохранены (--reconfigure, чтобы изменить)"
        fi
    fi

    while :; do
        [ -n "$PROVIDER_URL" ] || ask PROVIDER_URL "URL провайдера (OpenAI-совместимый, обычно …/v1)" ""
        [ -n "$PROVIDER_URL" ] || fail "нужен URL провайдера: --provider-url или ROI_PROVIDER_BASE_URL"
        PROVIDER_URL="${PROVIDER_URL%/}"
        case "$PROVIDER_URL" in http://*|https://*) ;; *) PROVIDER_URL="https://$PROVIDER_URL" ;; esac
        [ -n "$PROVIDER_KEY" ] || ask_secret PROVIDER_KEY "API-ключ провайдера (ввод скрыт)"
        [ -n "$PROVIDER_KEY" ] || fail "нужен ключ провайдера: --provider-key или ROI_PROVIDER_API_KEY"
        if probe_provider; then
            ok "провайдер отвечает: $(printf '%s\n' "$PROVIDER_MODELS" | grep -c . || true) моделей"
            break
        fi
        warn "провайдер не ответил на GET $PROVIDER_URL/models с этим ключом"
        [ "$NON_INTERACTIVE" != true ] || fail "не удалось проверить провайдера"
        if confirm "Сохранить всё равно?" n; then break; fi
        PROVIDER_URL=""; PROVIDER_KEY=""
    done

    configure_models

    if [ -z "$HOST" ]; then
        if confirm "Открыть доступ к рою из сети (0.0.0.0, по токену)?" y; then HOST=0.0.0.0; else HOST=127.0.0.1; fi
    fi
    PORT="${PORT:-8765}"
    if [ -z "$SWARM_KEY" ]; then
        SWARM_KEY="roi-$(owned_python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        ok "сгенерирован токен доступа ROI_SWARM_API_KEY"
    fi

    # Pass credentials through the child environment, never argv. Preserve custom keys.
    ROI_PROVIDER_BASE_URL="$PROVIDER_URL" ROI_PROVIDER_API_KEY="$PROVIDER_KEY" \
    ROI_SWARM_API_KEY="$SWARM_KEY" ROI_HOST="$HOST" ROI_PORT="$PORT" \
    owned_python - "$ENV_FILE" <<'EOF'
import os
import tempfile
from pathlib import Path
from dotenv import dotenv_values
path = Path(__import__("sys").argv[1])
values = dict(dotenv_values(path, interpolate=False)) if path.exists() else {}
for name in ("ROI_PROVIDER_BASE_URL", "ROI_PROVIDER_API_KEY", "ROI_SWARM_API_KEY", "ROI_HOST", "ROI_PORT"):
    values[name] = os.environ[name]
values.setdefault("ROI_SLEEP_ENABLED", "true")
values.setdefault("ROI_SLEEP_AUTO_IMPROVE", "false")
fd, temporary = tempfile.mkstemp(prefix=".env-", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("# ROI Swarm — secrets. Do not publish.\n")
        for name, value in values.items():
            if value is None:
                stream.write(f"{name}\n")
                continue
            if any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError(f"Control characters are not allowed in {name}")
            # Double quotes work with both python-dotenv and systemd EnvironmentFile.
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            stream.write(f'{name}="{escaped}"\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
finally:
    Path(temporary).unlink(missing_ok=True)
EOF
    ok "$ENV_FILE записан (chmod 600)"
}

configure_models() {
    # Модели по ролям хранятся в data/model-settings.json (без секретов), их же меняет дашборд.
    local selection="$INSTALL_DIR/data/model-settings.json"
    local default_main default_worker
    default_main="$(owned_python - "$INSTALL_DIR" <<'EOF'
import sys, yaml, json, pathlib
root = pathlib.Path(sys.argv[1])
roles = {k: v["model"] for k, v in yaml.safe_load((root / "config/settings.yaml").read_text())["models"].items()}
saved = root / "data/model-settings.json"
if saved.exists():
    roles.update(json.loads(saved.read_text())["roles"])
print(roles["planner"], roles["worker"])
EOF
)"
    default_worker="${default_main#* }"; default_main="${default_main%% *}"

    if [ -f "$selection" ] && [ "$RECONFIGURE" != true ] \
        && [ -z "$MAIN_MODEL" ] && [ -z "$WORKER_MODEL" ]; then
        MAIN_MODEL="$default_main"; WORKER_MODEL="$default_worker"
        ok "сохранены все индивидуальные модели по ролям"
        return
    fi
    if [ -z "$MAIN_MODEL" ] || [ -z "$WORKER_MODEL" ]; then
        if [ -n "$PROVIDER_MODELS" ] && [ "$NON_INTERACTIVE" != true ]; then
            printf '  Модели провайдера:\n'
            printf '%s\n' "$PROVIDER_MODELS" | sed -n '1,60s/^/    · /p'
        fi
        [ -n "$MAIN_MODEL" ] || ask MAIN_MODEL "Центральная модель (план и синтез)" "$default_main"
        [ -n "$WORKER_MODEL" ] || ask WORKER_MODEL "Модель воркеров (роутер, воркеры, критик)" "$default_worker"
    fi
    for model in "$MAIN_MODEL" "$WORKER_MODEL"; do
        if [ -n "$PROVIDER_MODELS" ] && ! printf '%s\n' "$PROVIDER_MODELS" | grep -qxF -- "$model"; then
            warn "модели «$model» нет в списке провайдера — проверьте имя"
        fi
    done
    run_owned mkdir -p "$INSTALL_DIR/data"
    owned_python - "$selection" "$MAIN_MODEL" "$WORKER_MODEL" <<'EOF'
import sys
from pathlib import Path
from roi_swarm.model_settings import ModelSelection, save_selection
path, main, worker = sys.argv[1], sys.argv[2], sys.argv[3]
roles = {"planner": main, "synthesizer": main, "router": worker, "worker": worker, "critic": worker}
save_selection(Path(path), ModelSelection(roles=roles))
EOF
    ok "центр: $MAIN_MODEL · воркеры: $WORKER_MODEL"
}

has_systemd() { [ "$OS" = Linux ] && have systemctl && [ -d /run/systemd/system ]; }

unit_body() {  # unit_body WANTED_BY [USER]
    cat <<EOF
# ROI Swarm installer v1
[Unit]
Description=ROI Swarm — коллективный разум (OpenAI-совместимый провайдер)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
${2:+User=$2}
WorkingDirectory=$INSTALL_DIR
Environment=ROI_SUPERVISED=1
Environment=PYTHONUNBUFFERED=1
ExecStart=$INSTALL_DIR/.venv/bin/roi-swarm-server
Restart=always
RestartSec=3
NoNewPrivileges=true

[Install]
WantedBy=$1
EOF
}

system_unit_path() { printf '/etc/systemd/system/%s.service' "$SERVICE"; }
user_unit_path() { printf '%s/.config/systemd/user/%s.service' "$HOME" "$SERVICE"; }
validate_unit() {
    local unit="$1" expected="$2"
    if [ -e "$unit" ] || [ -L "$unit" ]; then
        [ -f "$unit" ] && [ ! -L "$unit" ] \
            && grep -qxF '# ROI Swarm installer v1' "$unit" \
            && [ "$(grep '^WorkingDirectory=' "$unit")" = "WorkingDirectory=$INSTALL_DIR" ] \
            && [ "$(grep '^User=' "$unit" || true)" = "$expected" ] \
            && [ "$(grep '^ExecStart=' "$unit")" = "ExecStart=$INSTALL_DIR/.venv/bin/roi-swarm-server" ] \
            || fail "чужой или несовпадающий unit: $unit (WorkingDirectory/User/ExecStart)"
    fi
}
validate_units() {
    validate_unit "$(system_unit_path)" "User=$INSTALL_OWNER"
    validate_unit "$(user_unit_path)" ""
}
install_service() {
    validate_units
    if [ "$IS_ROOT" = true ] || { have sudo && sudo -n true 2>/dev/null; }; then
        SERVICE_SCOPE=system
        unit_body multi-user.target "$INSTALL_OWNER" | $SUDO tee "$(system_unit_path)" >/dev/null
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable -q "$SERVICE"
        $SUDO systemctl restart "$SERVICE"
        CTL="${SUDO:+sudo }systemctl"
        LOGS="${SUDO:+sudo }journalctl -u $SERVICE -f"
    else
        SERVICE_SCOPE=user
        mkdir -p "$HOME/.config/systemd/user"
        unit_body default.target >"$HOME/.config/systemd/user/$SERVICE.service"
        systemctl --user daemon-reload
        systemctl --user enable -q "$SERVICE"
        systemctl --user restart "$SERVICE"
        loginctl enable-linger "$USER" 2>/dev/null \
            || warn "без linger сервис остановится после выхода: sudo loginctl enable-linger $USER"
        CTL="systemctl --user"
        LOGS="journalctl --user -u $SERVICE -f"
    fi
    ok "сервис $SERVICE ($SERVICE_SCOPE) включён в автозапуск"
    wait_healthy || { $CTL status "$SERVICE" --no-pager -n 30 >&2 || true; fail "сервер не поднялся, логи: $LOGS"; }
}

wait_healthy() {
    local url="http://127.0.0.1:$PORT"
    for _ in $(seq 1 60); do
        if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
            curl -fsS --max-time 5 -H "Authorization: Bearer $SWARM_KEY" "$url/v1/models" >/dev/null \
                || fail "/v1/models отклонил токен"
            ok "сервер отвечает на $url (health и /v1/models с токеном)"
            return 0
        fi
        sleep 1
    done
    return 1
}

start_foreground_hint() {
    SERVICE_SCOPE=none
    warn "systemd недоступен — запускайте сервер вручную:"
    printf '    cd %s && .venv/bin/roi-swarm-server\n' "$INSTALL_DIR"
}

public_host() {
    if [ "$HOST" = 127.0.0.1 ] || [ "$HOST" = localhost ]; then echo 127.0.0.1; return; fi
    local ip
    ip="$(curl -fsS --max-time 3 https://api.ipify.org 2>/dev/null || true)"
    [ -n "$ip" ] || ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "${ip:-<IP-сервера>}"
}

finish() {
    local addr; addr="$(public_host)"
    local base="http://$addr:$PORT" token_fragment
    token_fragment="$(ROI_SWARM_API_KEY="$SWARM_KEY" owned_python -c \
        'import os; from urllib.parse import quote; print(quote(os.environ["ROI_SWARM_API_KEY"], safe=""))')"
    cat <<EOF

${B}${G}✓ ROI Swarm установлен${N}

  Дашборд:      $base/dashboard#token=$token_fragment
  API (OpenAI): $base/v1      модель: roi-swarm
  Токен:        $SWARM_KEY
  Каталог:      $INSTALL_DIR   (секреты в .env)

  ${B}Подключить к Hermes${N} (~/.hermes/config.yaml → providers):
    roi-swarm:
      name: ROI Swarm
      base_url: $base/v1
      key_env: ROI_SWARM_API_KEY
      api_mode: chat_completions
      default_model: roi-swarm
  и в ~/.hermes/.env:  ROI_SWARM_API_KEY=$SWARM_KEY

EOF
    if [ "$SERVICE_SCOPE" != none ]; then
        cat <<EOF
  Управление:   $CTL {status|restart|stop} $SERVICE
  Логи:         $LOGS
EOF
    fi
    cat <<EOF
  Обновить:     повторите команду установки (настройки сохранятся)
  Провайдер:    ... | bash -s -- --reconfigure
  Удалить:      ... | bash -s -- --uninstall

EOF
    if [ "$HOST" != 127.0.0.1 ] && [ "$addr" != 127.0.0.1 ]; then
        printf '  %s!%s Порт %s открыт наружу по HTTP. Для продакшена поставьте перед ним HTTPS\n    (Caddy/nginx) или закройте firewall и ходите через VPN/SSH-туннель.\n\n' "$Y" "$N" "$PORT"
    fi
}

uninstall() {
    if [ "$IS_ROOT" = true ]; then INSTALL_OWNER="$SERVICE"; else INSTALL_OWNER="$(id -un)"; fi
    # Validate EVERY target before stopping any service or deleting anything.
    validate_receipt
    validate_units
    local system_unit user_unit receipt
    system_unit="$(system_unit_path)"; user_unit="$(user_unit_path)"; receipt="$(state_path)"
    if [ "$NON_INTERACTIVE" != true ] && ! confirm "Удалить $INSTALL_DIR вместе с .env и данными?" n; then
        ok "установка и сервис оставлены без изменений"; return
    fi
    if [ -f "$system_unit" ]; then
        $SUDO systemctl disable --now -q "$SERVICE"
        $SUDO rm -f -- "$system_unit"; $SUDO systemctl daemon-reload
    fi
    if [ -f "$user_unit" ]; then
        systemctl --user disable --now -q "$SERVICE"
        rm -f -- "$user_unit"; systemctl --user daemon-reload
    fi
    # rm does not traverse symlinks inside these two verified directory identities.
    rm -rf -- "$INSTALL_DIR" "$INSTALL_DIR-worktrees"
    if [ "$IS_ROOT" = true ]; then userdel "$SERVICE"; fi
    rm -f -- "$receipt"
    ok "удалено $INSTALL_DIR"
}

main "$@"
