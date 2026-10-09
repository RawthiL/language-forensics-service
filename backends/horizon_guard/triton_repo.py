"""Horizon-guard backend hooks for the shared tooling (`tools/make_fixtures.py`,
`tools/verify_repo.py`). Needs only `onnx`, `tokenizers` and `numpy` — no
weights, no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import common  # noqa: E402

FAMILY = "horizon-guard"
FIXTURE_NAME = "horizon-guard-fixture"


def check(conf, meta: dict) -> None:
    """Family-specific config.pbtxt checks (raise AssertionError on failure)."""
    assert len(meta["labels"]) == 2, f"horizon-guard is a binary head: {meta['labels']}"
    assert [meta["labels"][p].upper() for p in meta["positive"]] == ["INJECTION"], meta
    assert set(meta["inputs"]) == {"input_ids", "attention_mask"}, meta["inputs"]


def make_fixture(repo: Path, workdir: Path) -> str:
    """Trigger fixture (fires on "ignore") with Horizon's labels; a 16-token
    window with stride 4, so a few sentences already span several windows."""
    return common.write_fixture(repo, workdir, name=FIXTURE_NAME, family=FAMILY,
                                labels=["SAFE", "INJECTION"], positive=1)
