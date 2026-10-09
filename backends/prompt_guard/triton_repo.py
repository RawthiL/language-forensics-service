"""Prompt Guard backend hooks for the shared tooling (`tools/make_fixtures.py`,
`tools/verify_repo.py`). Needs only `onnx`, `tokenizers` and `numpy` — no
weights, no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import common  # noqa: E402

FAMILY = "prompt-guard"
FIXTURE_NAME = "prompt-guard-fixture"


def check(conf, meta: dict) -> None:
    """Family-specific config.pbtxt checks (raise AssertionError on failure)."""
    assert len(meta["labels"]) in (2, 3), f"expected a 2- or 3-label head: {meta['labels']}"
    assert all(meta["labels"][p] != "BENIGN" for p in meta["positive"]), meta
    assert meta["max_tokens"] <= 512, f"Prompt Guard context is 512, got {meta['max_tokens']}"
    assert set(meta["inputs"]) == {"input_ids", "attention_mask"}, meta["inputs"]
    if not meta.get("source", {}).get("fixture"):
        assert meta.get("notice") == "Built with Llama", "Llama license notice missing from metadata"


def make_fixture(repo: Path, workdir: Path) -> str:
    """Trigger fixture with Prompt Guard 2's labels."""
    return common.write_fixture(repo, workdir, name=FIXTURE_NAME, family=FAMILY,
                                labels=["BENIGN", "MALICIOUS"], positive=1)
