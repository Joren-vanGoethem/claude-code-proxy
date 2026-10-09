import os
from pathlib import Path
import socket
import subprocess

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "claude-proxy"

FAKE_DOCKER = """#!/usr/bin/env bash
echo "docker $* PROFILE_DIR=${PROFILE_DIR:-}" >> "$FAKE_LOG"
if [[ " $* " == *" ps "* && -n "${FAKE_RUNNING:-}" ]]; then echo container-id; fi
"""
FAKE_CLAUDE = """#!/usr/bin/env bash
echo "claude $* ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL" >> "$FAKE_LOG"
"""


@pytest.fixture
def sandbox(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("claude", FAKE_CLAUDE)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    profiles = tmp_path / "profiles"
    for name, port in (("spark", "8082"), ("z", "8083")):
        (profiles / name).mkdir(parents=True)
        (profiles / name / ".env").write_text(f'PROXY_PORT="{port}"\n')
        (profiles / name / "model_mapping.json").write_text("{}")
    return tmp_path


def run(sandbox: Path, *args: str, running: bool = False):
    env = {
        "PATH": f"{sandbox / 'bin'}:{os.environ['PATH']}",
        "FAKE_LOG": str(sandbox / "log"),
        "CLAUDE_PROXY_PROFILES_DIR": str(sandbox / "profiles"),
    }
    if running:
        env["FAKE_RUNNING"] = "1"
    completed = subprocess.run(
        [str(SCRIPT), *args], env=env, capture_output=True, text=True, check=False
    )
    log = sandbox / "log"
    return completed, log.read_text() if log.exists() else ""


def test_run_passes_port_and_arguments_to_claude_for_a_running_profile(sandbox: Path):
    completed, log = run(sandbox, "run", "z", "--dangerously-skip-permissions", "-p", "hi", running=True)

    assert completed.returncode == 0, completed.stderr
    assert "claude --dangerously-skip-permissions -p hi ANTHROPIC_BASE_URL=http://127.0.0.1:8083" in log
    assert " up " not in log


def test_run_starts_a_stopped_profile_before_launching_claude(sandbox: Path):
    # An open local port satisfies the script's readiness wait after `up`.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        (sandbox / "profiles" / "z" / ".env").write_text(f"PROXY_PORT={port}\n")
        completed, log = run(sandbox, "run", "z")

    assert completed.returncode == 0, completed.stderr
    lines = log.splitlines()
    up = next(i for i, line in enumerate(lines) if " up -d --build " in f"{line} ")
    launch = next(i for i, line in enumerate(lines) if line.startswith("claude"))
    assert up < launch
    assert "-p claude-proxy-z" in lines[up]
    assert f"PROFILE_DIR={sandbox / 'profiles' / 'z'}" in lines[up]


def test_up_uses_a_project_name_and_env_file_per_profile(sandbox: Path):
    completed, log = run(sandbox, "up", "spark")

    assert completed.returncode == 0, completed.stderr
    assert "-p claude-proxy-spark up -d --build" in log
    assert f"--env-file {sandbox / 'profiles' / 'spark' / '.env'}" in log


def test_ls_shows_each_profile_with_its_port_and_state(sandbox: Path):
    completed, _ = run(sandbox, "ls", running=True)

    assert completed.returncode == 0, completed.stderr
    rows = [line.split() for line in completed.stdout.splitlines()[1:]]
    assert rows == [["spark", "8082", "running"], ["z", "8083", "running"]]


def test_missing_profile_names_the_directory_to_create(sandbox: Path):
    completed, _ = run(sandbox, "up", "missing")

    assert completed.returncode != 0
    assert "no profile 'missing'" in completed.stderr
    assert str(sandbox / "profiles" / "missing") in completed.stderr


def test_missing_env_file_names_the_file_to_create(sandbox: Path):
    (sandbox / "profiles" / "z" / ".env").unlink()

    completed, _ = run(sandbox, "up", "z")

    assert completed.returncode != 0
    assert f"{sandbox / 'profiles' / 'z' / '.env'}" in completed.stderr


def test_missing_port_names_the_variable_to_add(sandbox: Path):
    (sandbox / "profiles" / "z" / ".env").write_text("OPENAI_API_KEY=x\n")

    completed, _ = run(sandbox, "run", "z")

    assert completed.returncode != 0
    assert "PROXY_PORT" in completed.stderr
