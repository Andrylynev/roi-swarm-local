#!/usr/bin/env bash
# User-owned bootstrap. No root, distro packages, service registration or API keys.
set -euo pipefail
usage() {
    printf '%s\n' 'roi-swarm-local: user-owned local inference, no provider keys' \
        '  --dir PATH (default: ~/.roi-swarm-local)' \
        '  --backend auto|cuda|cpu' '  --profile balanced|fast' \
        '  --with-security' '  --no-start' '  --non-interactive' '  --help' \
        '  --accept-whiterabbitneo-terms (optional model license acceptance)' \
        'Requires git and curl. Run as your normal account, never root.'
}
fail() { printf 'ROI Local: %s\n' "$*" >&2; exit 1; }
check_path() {
    local path="$1" component current='' rest
    case "$path" in /*) ;; *) fail '--dir must be an absolute path' ;; esac
    case "$path" in /|"$HOME"|*/../*|*/..|*/./*|*/.|*//*|*$'\n'*|*$'\r'*)
        fail 'Unsafe installation path' ;; esac
    rest="${path#/}"
    while [ -n "$rest" ]; do
        component="${rest%%/*}"
        current="$current/$component"
        [ ! -L "$current" ] || fail "Symlink component refused: $current"
        if [ "$rest" = "$component" ]; then break; fi
        rest="${rest#*/}"
    done
}
main() {
    local root="$HOME/.roi-swarm-local" backend=auto profile=balanced
    local security=false no_start=false non_interactive=false arg
    local accept_terms=false
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --help|-h) usage; return ;;
            --dir|--backend|--profile)
                [ "$#" -ge 2 ] && [ -n "$2" ] || fail "Missing value for $1"
                case "$1" in --dir) root="$2" ;; --backend) backend="$2" ;; --profile) profile="$2" ;; esac
                shift 2 ;;
            --with-security) security=true; shift ;;
            --accept-whiterabbitneo-terms) accept_terms=true; shift ;;
            --no-start) no_start=true; shift ;;
            --non-interactive) non_interactive=true; shift ;;
            *) fail "Unknown argument: $1 (see --help)" ;;
        esac
    done
    case "$backend" in auto|cuda|cpu) ;; *) fail '--backend must be auto, cuda or cpu' ;; esac
    case "$profile" in balanced|fast) ;; *) fail '--profile must be balanced or fast' ;; esac
    [ "$(id -u)" -ne 0 ] || fail 'Do not run as root; use your normal account (without sudo).'
    for arg in git curl; do
        command -v "$arg" >/dev/null 2>&1 || fail "Required prerequisite: $arg. Install it first as appropriate for your system."
    done
    root="${root%/}"
    check_path "$root"
    local repo='https://github.com/Andrylynev/roi-swarm-local.git'
    local marker="$root/.roi-swarm-local-install" existing=false expected origin python
    expected=$(printf 'roi-swarm-local-v1\n%s\n%s' "$repo" "$root")
    if [ -e "$root" ]; then
        [ -d "$root" ] || fail 'Installation path is not a directory'
        if [ -n "$(command ls -A "$root")" ]; then
            [ -f "$marker" ] && [ ! -L "$marker" ] || fail 'Existing directory is unmarked; choose a new empty directory.'
            [ "$(<"$marker")" = "$expected" ] || fail 'Installation ownership marker does not match this path/repository.'
            [ -d "$root/.git" ] && [ ! -L "$root/.git" ] || fail 'Expected a standalone Git checkout'
            origin=$(git -C "$root" config --local --get-all remote.origin.url) || fail 'Origin is missing'
            [ "$origin" = "$repo" ] || fail 'Untrusted origin; expected the public HTTPS repository.'
            [ "$(git -C "$root" rev-parse --show-toplevel)" = "$root" ] || fail 'Checkout root mismatch'
            existing=true
        fi
    fi
    if [ "$existing" = true ]; then
        check_path "$root/.venv"
        check_path "$root/.venv/bin"
        python="$root/.venv/bin/python"
        # A venv interpreter is normally a symlink on POSIX; allow only this
        # interpreter link, not redirected install directories (checked above).
        if [ -x "$python" ]; then
            (cd "$root" && "$python" -m local.manage stop) || fail 'Stop incomplete; refusing update/dependency reinstall.'
        elif [ -e "$root/data/local-services.json" ]; then
            fail 'Cannot prove services stopped: application interpreter is missing.'
        fi
    fi
    mkdir -p "$root"
    if [ "$existing" = true ]; then
        git -C "$root" -c core.hooksPath=/dev/null -c core.fsmonitor=false fetch --no-tags origin main
        git -C "$root" -c core.hooksPath=/dev/null -c core.fsmonitor=false merge --ff-only FETCH_HEAD ||
            fail 'Fast-forward update refused; local files/commits preserved. Resolve manually.'
    else
        git -c core.hooksPath=/dev/null clone --branch main -- "$repo" "$root"
        printf '%s\n' "$expected" > "$marker"
        printf '\n/.roi-swarm-local-install\n' >> "$root/.git/info/exclude"
    fi
    # Keep managed Python, uv cache and application venv inside the chosen root.
    export UV_CACHE_DIR="$root/cache/uv" UV_PYTHON_INSTALL_DIR="$root/tools/python"
    export UV_PROJECT_ENVIRONMENT="$root/.venv"
    unset VIRTUAL_ENV CONDA_PREFIX UV_PYTHON PYTHONHOME PYTHONPATH
    local uv
    if command -v uv >/dev/null 2>&1; then
        uv=$(command -v uv)
    else
        mkdir -p "$root/tools/uv"
        local installer="$root/tools/uv-install.sh"
        curl --proto '=https' --tlsv1.2 -fsSL https://astral.sh/uv/install.sh -o "$installer"
        UV_INSTALL_DIR="$root/tools/uv" UV_NO_MODIFY_PATH=1 sh "$installer"
        uv="$root/tools/uv/uv"
        [ -x "$uv" ] || fail 'uv user-scoped installation failed'
    fi
    printf 'Creating/updating application environment in %s\n' "$root"
    (cd "$root" && "$uv" sync --frozen --python 3.11)
    local setup=(--backend "$backend" --profile "$profile")
    [ "$security" = false ] || setup+=(--with-security)
    [ "$accept_terms" = false ] || setup+=(--accept-whiterabbitneo-terms)
    [ "$no_start" = false ] || setup+=(--no-start)
    [ "$non_interactive" = false ] || setup+=(--non-interactive)
    (cd "$root" && "$root/.venv/bin/python" -m local.setup "${setup[@]}")
    printf '\nROI Swarm Local: %s\nManage: cd "%s" && .venv/bin/python -m local.manage status\n' "$root" "$root"
}
main "$@"
