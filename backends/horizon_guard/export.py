#!/usr/bin/env python3
"""Export a Horizon-Labs prompt-injection-guard model to a Triton repo.

Upstream publishes ONNX weights next to the PyTorch ones, so no torch and no
conversion are needed: this script fetches (or copies) the ONNX graph and the
tokenizer, checks the graph's contract, and writes
`triton/model_repository/<name>/{config.pbtxt, tokenizer.json, 1/model.onnx}`.

Two checkpoint sources, mirroring the sibling services:

  1. By NAME, from the catalog below (pinned to an upstream commit sha, fetched
     into the Hugging Face cache):
       uv run --extra horizon-guard python backends/horizon_guard/export.py --variant base
       uv run --extra horizon-guard python backends/horizon_guard/export.py --variant base --quantized

  2. By PATH, a local directory with the upstream layout (config.json,
     tokenizer.json, onnx/model.onnx[_data]); no network access at all:
       uv run --extra horizon-guard python backends/horizon_guard/export.py \
           --checkpoint /models/prompt-injection-guard-base

`--quantized` serves upstream's `onnx/model_quantized.onnx` (int8 embedding
gather, about half the size). Upstream measured 99.75% decision agreement
with fp32 on 400 inputs (`onnx/quantization_check.json`); it is a different
model fingerprint and should be evaluated as such.

LICENSE: the Horizon-Labs weights are Apache-2.0 (ungated, trained on
permissively licensed data). Fine-tunes inherit the license of their base
model and data. See backends/horizon_guard/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common  # noqa: E402

FAMILY = "horizon-guard"

# Pinned upstream revisions. Bump deliberately: a new revision is a new model.
CATALOG = {
    "base": {
        "repo": "Horizon-Labs/prompt-injection-guard-base",
        "revision": "e4fbfc6f603e24c675cdf310c3b8d7abaa825666",
    },
    "large": {
        "repo": "Horizon-Labs/prompt-injection-guard-large",
        "revision": "bbc67bafa71c791ad802d4df0ae7639d4e880dea",
    },
}

# Upstream's evaluation scores long inputs over windows of min(2048, max)
# tokens with a quarter-window stride (code/train/evaluate.py).
DEFAULT_WINDOW = 2048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--variant", choices=sorted(CATALOG), help="catalog model (pinned revision)")
    src.add_argument("--checkpoint", type=Path, help="local directory with the upstream layout")
    p.add_argument("--quantized", action="store_true", help="serve onnx/model_quantized.onnx")
    p.add_argument("--name", default=None,
                   help="Triton model name (default: horizon-guard-<variant|dir>[-q8])")
    p.add_argument("--repo", type=Path, default=ROOT / "triton" / "model_repository")
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--queue-delay-us", type=int, default=2000)
    p.add_argument("--window", type=int, default=None,
                   help=f"recommended window in tokens (default min({DEFAULT_WINDOW}, model max))")
    p.add_argument("--stride", type=int, default=None, help="window overlap (default window // 4)")
    return p.parse_args()


def fetch(args) -> tuple[Path, dict]:
    """Return (local dir with the upstream layout, source metadata)."""
    onnx_file = "onnx/model_quantized.onnx" if args.quantized else "onnx/model.onnx"
    if args.checkpoint:
        d = args.checkpoint.resolve()
        for f in ("config.json", "tokenizer.json", onnx_file):
            if not (d / f).exists():
                sys.exit(f"{d} lacks {f} (expected the upstream repository layout)")
        return d, {"path": str(d), "file": onnx_file}

    from huggingface_hub import hf_hub_download, list_repo_files

    entry = CATALOG[args.variant]
    repo, rev = entry["repo"], entry["revision"]
    files = set(list_repo_files(repo, revision=rev))
    wanted = ["config.json", "tokenizer.json", onnx_file]
    data = f"{onnx_file}_data"
    if data in files:  # large fp32 keeps its weights in external data
        wanted.append(data)
    paths = [Path(hf_hub_download(repo, f, revision=rev)) for f in wanted]
    print(f"fetched {repo}@{rev[:12]}: {', '.join(wanted)}")
    # hf_hub_download keeps the repo layout inside the snapshot directory.
    return paths[0].parent, {"repo": repo, "revision": rev, "file": onnx_file}


def main() -> None:
    args = parse_args()
    src_dir, source = fetch(args)
    onnx_path = src_dir / source["file"]

    cfg = json.loads((src_dir / "config.json").read_text())
    id2label = {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
    labels = [id2label.get(i, f"LABEL_{i}") for i in range(len(id2label))]
    positive = [i for i, name in enumerate(labels) if name.upper() == "INJECTION"]
    if labels != ["SAFE", "INJECTION"]:
        sys.exit(f"unexpected labels {labels}; horizon-guard expects ['SAFE', 'INJECTION']")
    max_tokens = int(cfg.get("max_position_embeddings") or 512)
    pad_id = int(cfg.get("pad_token_id") or 0)

    window = args.window or min(DEFAULT_WINDOW, max_tokens)
    stride = window // 4 if args.stride is None else args.stride
    if not 0 < window <= max_tokens or not 0 <= stride < window:
        sys.exit(f"invalid window {window} / stride {stride} (model max {max_tokens})")

    info = common.inspect_onnx(onnx_path)
    in_names = [i["name"] for i in info["inputs"]]
    if set(in_names) != {"input_ids", "attention_mask"}:
        sys.exit(f"ONNX inputs {in_names}, expected input_ids + attention_mask")
    if not all(common.has_dynamic_batch(i["dims"]) for i in info["inputs"]):
        sys.exit(f"ONNX inputs {info['inputs']} need a dynamic batch dim")
    inputs, outputs = common.classifier_io_specs(info, len(labels), args.max_batch)

    if args.name:
        name = args.name
    else:
        stem = args.variant or src_dir.name.removeprefix("prompt-injection-guard-")
        name = f"horizon-guard-{stem}" + ("-q8" if args.quantized else "")

    tok_path = src_dir / "tokenizer.json"
    print("hashing weights ...")
    metadata = {
        "family": FAMILY,
        "task": common.TASK,
        "labels": labels,
        "positive": positive,
        "inputs": in_names,
        "outputs": [o["name"] for o in outputs],
        "tokenizer": "tokenizer.json",
        "tokenizer_sha256": common.sha256_file(tok_path),
        "max_tokens": max_tokens,
        "window": window,
        "stride": stride,
        "pad_id": pad_id,
        "onnx_sha256": common.sha256_file(onnx_path),
        "source": source,
        "license": "Apache-2.0",
    }
    config = common.build_config_pbtxt(
        model_name=name,
        metadata=metadata,
        input_specs=inputs,
        output_specs=outputs,
        max_batch_size=args.max_batch,
        queue_delay_us=args.queue_delay_us,
    )
    out = common.write_repo(args.repo, name, onnx_path, config, tokenizer_path=tok_path)
    print(f"OK: {out}/config.pbtxt + tokenizer.json + 1/model.onnx — '{name}' "
          f"labels={labels} window={window}/{stride} max_tokens={max_tokens}")


if __name__ == "__main__":
    main()
