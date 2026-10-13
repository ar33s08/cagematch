"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cagematch.agent import CellConfig
from cagematch.cells import scripted_transcript
from cagematch.scripted import ScriptedProvider


@pytest.fixture()
def data_dir() -> Path:
    return REPO_ROOT / "data"


def make_cell(name: str, task_id: str, **kw) -> CellConfig:
    provider = ScriptedProvider(name, scripted_transcript(task_id))
    return CellConfig(name=name, provider=provider, model="scripted",
                      system_prompt="Use tools. Finish with the answer object.",
                      parallel_tools=kw.get("parallel_tools", True),
                      max_steps=kw.get("max_steps", 16),
                      max_cost_usd=kw.get("max_cost_usd", 0.50),
                      pricing=kw.get("pricing", {"prompt": 0.0, "completion": 0.0}))
