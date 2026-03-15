from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def test_cli_train_inspect_playback(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "cli-ckpt.json"

    train = _run(
        [
            sys.executable,
            "-m",
            "self_parking.cli.main",
            "train",
            "--generations",
            "2",
            "--generation-size",
            "8",
            "--episode-seconds",
            "0.6",
            "--output",
            str(checkpoint),
        ],
        cwd=repo,
    )
    assert train.returncode == 0, train.stderr
    assert checkpoint.exists()

    inspect_cmd = _run(
        [
            sys.executable,
            "-m",
            "self_parking.cli.main",
            "checkpoint-inspect",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=repo,
    )
    assert inspect_cmd.returncode == 0, inspect_cmd.stderr
    assert "generation_size" in inspect_cmd.stdout

    playback = _run(
        [
            sys.executable,
            "-m",
            "self_parking.cli.main",
            "playback",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=repo,
    )
    assert playback.returncode == 0, playback.stderr
    assert "Playback loss" in playback.stdout
