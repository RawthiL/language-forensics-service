# Horizon-Labs guard backend (`family: "horizon-guard"`)

Serves the [Horizon-Labs prompt-injection-guard](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base)
classifiers. They detect prompt injection and jailbreak attempts in direct
user messages and in indirect sources (emails, documents, tool outputs),
across 30 languages.

## License

| Component | License |
|-----------|---------|
| This folder's code (`src/`, `export.py`, `triton_repo.py`) | MIT |
| Horizon-Labs `prompt-injection-guard-base` / `-large` weights | **Apache-2.0** ✅ commercial use OK, ungated |
| Training data (per the model card) | Permissively licensed sources only |

Serving these weights commercially, including staking a Pocket Network
supplier, is fine subject to the Apache-2.0 notice requirements. Custom
fine-tunes inherit the license of the base checkpoint and their own data.

## Catalog

`export.py --variant <name>` resolves to a pinned upstream commit. A new
revision is a new model: bump it deliberately and re-evaluate.

| Variant | Upstream repo | Revision | Backbone | Params | Context | ONNX files |
|---------|---------------|----------|----------|--------|---------|------------|
| `base` | `Horizon-Labs/prompt-injection-guard-base` | `e4fbfc6f603e…` | mmBERT-base (ModernBERT) | 308M | 8192 | `model.onnx` 1.2 GB, `model_quantized.onnx` 641 MB |
| `large` | `Horizon-Labs/prompt-injection-guard-large` | `bbc67bafa71c…` | bge-m3 (XLM-R) | 568M | 8192 | `model.onnx` + `model.onnx_data`, `model_quantized.onnx` |

`--quantized` serves upstream's int8 build (embedding gather only).
Upstream's `onnx/quantization_check.json` reports 99.75% decision agreement
with fp32 over 400 inputs. Max probability difference was 0.025. It has its
own model fingerprint; evaluate it as a separate model.

## Model contract

| | |
|-|-|
| Inputs | `input_ids`, `attention_mask` INT64 `[batch, seq]`, seq ≤ 8192 |
| Output | `logits` FP32 `[batch, 2]` |
| Labels | `SAFE` (0), `INJECTION` (1). The score is softmax P(INJECTION) |
| Tokenizer | Upstream `tokenizer.json` (Gemma-style BPE, `<bos> $A <eos>`) |
| Normalization | Inside the tokenizer: decodes Unicode tag characters and variation-selector bytes to ASCII, NFKC, strips invisible/bidi/format characters. Homoglyphs are not mapped. The Rust `tokenizers` crate runs it natively, so the gateway sees exactly what the model was trained with |
| Default window | 2048 tokens, stride 512. This is upstream's own evaluation setting (`code/train/evaluate.py`: windows of min(2048, max), stride window/4, max over windows) |
| Triton | `max_batch_size: 16`, dynamic batching `[4, 8, 16]`, 2 ms queue delay |

## Export

```bash
uv sync --extra horizon-guard
uv run python backends/horizon_guard/export.py --variant base
uv run python backends/horizon_guard/export.py --variant base --quantized   # -> horizon-guard-base-q8
uv run python backends/horizon_guard/export.py --variant large
uv run python backends/horizon_guard/export.py --checkpoint /path/to/prompt-injection-guard-base   # no network
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

Upstream ships ONNX, so the exporter needs no torch. It checks the graph's
inputs and outputs, reads labels and the context size from `config.json`,
hashes the ONNX file and the tokenizer, and writes the Triton repository.

## Known limitations (from the model card and independent evaluation)

- A single defense layer. Adaptive attackers can evade it.
- It flags harmless role-play: ~16% false positives on Qualifire (large,
  self-reported).
- It is weaker on bare out-of-place tasks embedded in documents (BIPIA:
  37.7% detection at 1% FPR, [arXiv:2610.03448](https://arxiv.org/abs/2610.03448)).
- Short benign imperatives can land near 0.5 (see the root README,
  "Detection quality").
