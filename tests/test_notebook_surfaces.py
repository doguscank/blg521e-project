from __future__ import annotations

import json
from pathlib import Path


def test_notebook_files_exist_and_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for notebook_name in ["evolution_training.ipynb", "automatic_playback.ipynb"]:
        notebook_path = root / "notebooks" / notebook_name
        assert notebook_path.exists(), f"missing notebook: {notebook_name}"
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert len(payload["cells"]) >= 2
