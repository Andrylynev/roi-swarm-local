"""Offline regression tests for worker package-install routing."""

import pytest

from roi_swarm.worker_install_policy import install_error, is_install


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "pip uninstall requests",
        "python -m pip install -e .",
        "uv pip install -e .",
        "uv pip sync requirements.txt",
        "uv sync",
        "uv add requests",
        "uv remove requests",
        "uv tool install ruff",
        "poetry install",
        "poetry add requests",
        "poetry remove requests",
        "poetry update",
        "poetry lock",
        "pipenv install",
        "pipenv uninstall requests",
        "pipenv sync",
        "conda install numpy",
        "mamba remove numpy",
        "micromamba update numpy",
        "conda env update -f environment.yml",
        "npm install",
        "npm i lodash",
        "npm ci",
        "npm add lodash",
        "npm uninstall lodash",
        "npm remove lodash",
        "npm update",
        "pnpm install",
        "pnpm i",
        "pnpm add lodash",
        "pnpm remove lodash",
        "yarn",
        "yarn install",
        "yarn add lodash",
        "yarn remove lodash",
        "bun install",
    ],
)
def test_plain_installers_are_routed_and_safe(command):
    assert is_install(command) is True
    assert install_error(command) is None


@pytest.mark.parametrize(
    "command",
    [
        None,
        42,
        {},
        ["pip install x"],
        "",
        "   ",
        "echo pip install x",
        'git commit -m "pip install fix"',
        'grep "npm install" .',
        "pip list",
        "pip show requests",
        "uv pip compile requirements.in",
        "npm test",
        "yarn test",
        '"pip install x"',
        "python -c 'print(\"pip install x\")'",
    ],
)
def test_noninstall_commands_stay_in_worker(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "pip --disable-pip-version-check install requests",
        "pip --proxy http://proxy.example:8080 install requests",
        "pip --timeout 30 uninstall requests",
        "pip3.11.exe --isolated --require-virtualenv install requests",
        "py -m pip install -e .",
        "py -3.11 -m pip install -e .",
        "python3.exe -m pip install -e .",
        "python -I -m pip install -e .",
        "python -W ignore -m pip install -e .",
        '"C:\\Program Files\\Python311\\python.exe" -m pip install -e .',
        '"D:\\ROI\\.venv\\Scripts\\pip.exe" install -e .',
        "'/d/Program Files/Python/python.exe' -m pip install -e .",
        '"pip" install requests',
        "python setup.py develop",
        "python setup.py install",
        "python3.exe -u ./setup.py --quiet develop",
        "uv --offline pip --python .venv/Scripts/python.exe install requests",
        "python -m pip --py D:/foreign/python.exe install requests",
        "pip --pyth=D:/foreign/python.exe install requests",
        "uv tool --offline install ruff",
        "uv tool --color always install ruff",
        "uv tool --allow-insecure-host localhost install ruff",
        "uv --allow-insecure-host localhost tool install ruff",
        "uv tool --allow-insecure-host localhost uninstall ruff",
        "uv --allow-insecure-host localhost tool upgrade ruff",
        "uv -p D:/env/Scripts/python.exe pip install requests",
        "uv -pD:/env/Scripts/python.exe pip install requests",
        "uv --directory app sync",
        "poetry --directory app install",
        "pipenv --python 3.11 install",
        "conda -n shared install numpy",
        "npm --prefix app install",
        "npm --location global install x",
        "npm --global true install x",
        "npm --global false install x",
        "npm --global=true install x",
        "pnpm --dir app add lodash",
        "yarn --cwd app",
        "bun.exe --cwd app install",
    ],
)
def test_installer_executable_paths_and_global_options(command):
    assert is_install(command) is True
    assert install_error(command) is None


@pytest.mark.parametrize(
    "command",
    [
        'pip --proxy "install" list',
        "pip --log install show requests",
        "python -c 'pip install requests'",
        "python script.py pip install requests",
        "python -m other_module pip install requests",
        "python setup.py --help",
        "python setup.py sdist",
        "uv --directory install pip compile requirements.in",
        "npm --prefix install test",
        "yarn --cwd install test",
        'echo "C:\\Program Files\\Python\\python.exe" -m pip install requests',
    ],
)
def test_option_values_and_script_arguments_are_not_subcommands(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "cd app && npm install && pytest",
        "echo ready || pip install requests",
        "echo ready; uv sync",
        "echo ready\nnpm ci",
        "pip install requests && pytest",
        "pip install requests || true",
        "pip install requests;",
        "pip install requests &",
        "cat requirements.txt | pip install -r /dev/stdin",
        "npm install | tee install.log",
        "pip install requests > install.log",
        "pip install requests 2>&1",
        "2>install.log pip install requests",
        "pip install requests < requirements.txt",
        "pip install $(cat requirements.txt)",
        'pip install "$(cat requirements.txt)"',
        "pip install `cat requirements.txt`",
        'pip install "`cat requirements.txt`"',
        "pip install <(cat requirements.txt)",
        "(pip install requests)",
        "{ pip install requests; }",
        'pip install "requests',
    ],
)
def test_unsafe_shell_install_is_detected_but_not_forwarded(command):
    assert is_install(command) is True
    error = install_error(command)
    assert isinstance(error, str) and error
    assert any(word in error.lower() for word in ("single", "standalone", "plain"))


@pytest.mark.parametrize(
    "command",
    [
        "pip install 'thing;with&&shell|characters'",
        'pip install "thing;with&&shell|characters"',
        "pip install 'thing>name'",
        "pip install 'thing$(literal)'",
        "pip install 'thing`literal`'",
        "pip install requests # && pytest",
        "# npm install\npip install requests\n",
        "\n  pip install requests\n",
        "pip install req\\\nuests",
        "pip install foo\\;bar",
        "pip install foo\\|bar",
        'pip install "foo\\";bar"',
    ],
)
def test_quoted_escaped_or_comment_metacharacters_do_not_split_commands(command):
    assert is_install(command) is True
    assert install_error(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "# pip install requests",
        "echo ready # pip install requests",
        "echo 'one; pip install requests'",
        'git commit -m "fix; pip install requests"',
        "grep 'npm install' . && pytest",
        "echo foo\\;pip install requests",
        "echo pip\\\n install requests",
        "echo ready\n# npm install",
        "echo ready | grep 'pip install'",
        "echo file{pip install requests}",
        "echo {pip install requests}",
        "echo { pip install requests; }",
    ],
)
def test_shell_data_is_not_a_segment_head(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install requests",
        "A=1 B='two words' npm install",
        "env pip install requests",
        "env -i FOO=1 uv sync",
        "env --ignore-environment --unset PYTHONPATH pip install requests",
        "env -u PYTHONPATH --chdir app npm ci",
        "env --unset=PYTHONPATH --chdir=app npm install",
        "env -C app -- npm install",
        "env -uPYTHONPATH pip install requests",
        "env FOO=1 env BAR=2 python -m pip install -e .",
        "command pip install requests",
        "command -p -- npm install",
        "exec pip install requests",
        "exec -a package-installer python -m pip install requests",
        "exec -c env -i npm install",
    ],
)
def test_literal_environment_and_execution_wrappers(command):
    assert is_install(command) is True
    assert install_error(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "A='pip install requests' echo okay",
        "env FOO='npm install' echo okay",
        "env -u pip install requests",
        "env --unset pip install requests",
        "env --chdir pip echo install requests",
        "env -C pip echo install requests",
        "command -v pip install requests",
        "command -V npm install",
        "exec -a pip echo install requests",
        "env -i",
        "A=1",
        "env --help pip install requests",
    ],
)
def test_wrapper_values_are_not_invocations(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "env -S 'pip install requests'",
        "env --split-string='npm install'",
        "env --split-string 'uv sync'",
        "env -S pip install requests",
        "env -Spip install requests",
        "env --split-string=pip install requests",
        "env -S 'pip install' requests",
    ],
)
def test_split_string_env_wrapper_is_recognized_but_rejected(command):
    assert is_install(command) is True
    error = install_error(command)
    assert error and "env" in error.lower()
    assert "single" in error.lower() or "plain" in error.lower()


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest",
        "uv run --frozen pytest",
        "uv --offline run python -m pytest",
        "env FOO=1 uv run pytest",
        "uv run pytest --no-sync",
        "uv run --with --no-sync pytest",
        "uv run --unknown-value --no-sync pytest",
        "uv run --unknown-value --help pytest",
    ],
)
def test_uv_run_sync_is_never_moved_to_shared_tree(command):
    assert is_install(command) is True
    error = install_error(command)
    assert error and "uv run --no-sync" in error
    assert "worker" in error.lower()


@pytest.mark.parametrize(
    "command",
    [
        "uv run --no-sync pytest",
        "uv run --no-sync -- python -m pytest",
        "uv run --directory app --no-sync pytest",
        "uv run --with requests --no-sync pytest",
        "env -i uv run --no-sync pytest",
        "uv run --no-sync pytest && echo okay",
    ],
)
def test_uv_run_explicit_no_sync_stays_local(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "uv venv",
        "uv venv --clear .venv",
        "uv --directory app venv .venv",
        "python -m venv .venv",
        "python3.exe -I -m venv --clear .venv",
        "py -3.11 -m venv .venv",
        '"C:\\Program Files\\Python\\python.exe" -m venv .venv',
        "env -u VIRTUAL_ENV uv venv .venv",
    ],
)
def test_shared_venv_recreation_is_always_refused(command):
    assert is_install(command) is True
    error = install_error(command)
    assert error and "shared" in error.lower()
    assert "venv" in error.lower()
    assert "recreat" in error.lower() or "replace" in error.lower()


@pytest.mark.parametrize(
    "command",
    [
        "pip --help install",
        "pip --version install",
        "pip install --help",
        "uv --help sync",
        "uv pip install --help",
        "uv run --help",
        "npm --version",
        "yarn --help",
        "yarn --version",
        '"A=1" pip install requests',
        "'A'=1 pip install requests",
        "env -S 'echo pip install requests'",
        "env --unknown pip install requests",
        "python -cpass -m pip install requests",
        "python -mnot_pip -m pip install requests",
    ],
)
def test_informational_options_and_quoted_assignments_are_not_installs(command):
    assert is_install(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "pip --index-url https://packages.example/simple install requests",
        "pip -i https://packages.example/simple install requests",
        "pip --proxy=http://proxy.example install requests",
        "pip --timeout=30 install requests",
        "uv -C app sync",
        "uv pip --python=3.11 sync requirements.txt",
        "python -mpip install requests",
        "python -mvenv .venv",
        'A="two words" pip install requests',
        "env -- FOO=1 pip install requests",
    ],
)
def test_additional_argument_spellings_are_recognized(command):
    assert is_install(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "env -S 'echo okay' pip install requests",
        "env -S 'echo okay' && npm install",
        "pip install ${PACKAGE}",
        'pip install "$PACKAGE"',
        "pip install requests\\",
    ],
)
def test_ambiguous_shell_or_env_commands_are_never_forwarded(command):
    if is_install(command):
        assert install_error(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "npm --future-global-mode enabled install x",
        "npm --global true --loglevel info install x",
        "npm --unknown-value info install x",
        "npm --unknown-value test install x",
        "uv --unknown-value tree tool install x",
        "uv tool --unknown-value list install x",
        "pip --unknown-value list install x",
        "npm --unknown-value --help install x",
        "npm --global true --help false install x",
        "npm --version false install x",
        "npm -v false install x",
        "npm --version --no-version install x",
        "npm install x --help false",
        "npm --help=false install x",
        "npm --usage false install x",
        "npm install x --help",
        "npm --version install",
        "conda --unknown-value list install x",
        "python setup.py --unknown-value info install",
        "uv --future-target some-env pip install x",
        "uv tool --future-target some-env install x",
        "pip --future-python some-env install x",
    ],
)
def test_unresolved_installer_subcommand_never_runs_locally(command):
    assert is_install(command)
    assert "ambiguous" in install_error(command).lower()


def test_metadata_looks_like_windows_path_but_is_not_executable_prose():
    assert not is_install('"notes about C:\\Python\\pip.exe install" requests')


@pytest.mark.parametrize(
    "command",
    [
        "env 'FOO=1' pip install requests",
        'env "FOO=two words" pip install requests',
        "env -- 'FOO=1' pip install requests",
        "pip -V install",
        "yarn -v",
        "npm -v",
        "uv -V sync",
        "poetry -V install",
        "command FOO=1 pip install requests",
        "exec FOO=1 pip install requests",
        "uv run -m pytest --no-sync",
        "uv run --module pytest --no-sync",
        "uv run pytest --help",
        "uv run --with --no-sync pytest",
        "uv run --with --help pytest",
        "uv run --with requests --help",
    ],
)
def test_wrapper_and_uv_run_argument_boundaries(command):
    expected_install = command.startswith("env ") or command in {
        "uv run -m pytest --no-sync",
        "uv run --module pytest --no-sync",
        "uv run pytest --help",
        "uv run --with --no-sync pytest",
        "uv run --with --help pytest",
    }
    assert is_install(command) is expected_install
    if expected_install:
        if command.startswith("env "):
            assert install_error(command) is None
        else:
            assert "uv run --no-sync" in install_error(command)
