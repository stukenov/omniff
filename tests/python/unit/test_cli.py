import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "omniff.cli", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "python", "PATH": ""},
        cwd=str(__import__("pathlib").Path(__file__).parents[3]),
    )


def test_cli_no_args():
    result = run_cli()
    assert result.returncode == 0
    assert "omniff" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_doctor():
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "Python:" in result.stdout
    assert "Dependencies:" in result.stdout


def test_cli_models_list():
    result = run_cli("models", "list")
    assert result.returncode == 0


def test_cli_models_remove_nonexistent():
    result = run_cli("models", "remove", "--model-id", "nonexistent/model-xyz-999")
    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()


def test_cli_models_pull_no_id():
    result = run_cli("models", "pull")
    assert result.returncode != 0
