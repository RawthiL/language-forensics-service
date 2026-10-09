# Model backends

Each folder here is one **model family**: everything specific to that model,
and nothing else. The gateway (`../gateway/`) is model-agnostic. It only talks
to backends through the `Backend` trait in [`../backend-api`](../backend-api/src/lib.rs).

```
backends/<name>/
├── Cargo.toml        # crate `backend-<name>`, depends on backend-api
├── src/lib.rs        # FAMILY + create() + impl Backend
├── export.py         # weights -> ONNX + tokenizer.json + Triton config.pbtxt (metadata.family = "<name>")
├── triton_repo.py    # FAMILY, check(conf, meta), make_fixture(repo, workdir)
└── README.md         # model contract + LICENSE of the weights
```

## The contract

### Rust (`src/lib.rs`)

```rust
pub const FAMILY: &str = "<name>";                         // == metadata "family"
pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>>; // parse metadata, load tokenizer, validate

impl Backend for MyModel {
    fn family(&self) -> &'static str;
    fn labels(&self) -> &[String];            // indexed by logit position
    fn positive(&self) -> &[usize];           // logits summed into the injection score
    fn output_names(&self) -> &[String];      // Triton outputs to request
    fn max_tokens(&self) -> usize;
    fn default_window(&self) -> WindowParams; // exporter's recommendation
    fn preprocess(&self, text: &str, params: &WindowParams) -> Result<Vec<Window>>;  // covers ALL the text
    fn pack(&self, windows: &[&Window], seq_len: usize) -> Vec<InputTensor>;
    fn postprocess(&self, outputs: &[OutputTensor], n: usize) -> Result<Vec<f64>>;   // one score per window
}
```

- `backend_api::text::TextClassifier` implements all of this for a standard
  `[batch, seq] -> [batch, labels]` sequence classifier. It loads
  `tokenizer.json` from `info.model_dir`, checks its sha256 against the
  metadata, cuts overlapping windows (with code-point spans into the original
  text), packs `input_ids` / `attention_mask` / `token_type_ids`, and applies
  softmax in f64. A family crate usually wraps it and adds only validation of
  its own label conventions (see [`horizon_guard`](horizon_guard/src/lib.rs)).
- `preprocess` runs on a blocking thread. It must **never truncate**: the
  windows must cover every token. The window policy (size, stride) belongs to
  the gateway; the backend only rejects values its model cannot take.
- The model's normalization (NFKC, invisible-character stripping, smuggled-text
  decoding) belongs in `tokenizer.json`'s normalizer, not in Rust code. The
  `tokenizers` crate then runs exactly what the model was trained with, and
  offsets still map back to the original text.

### Metadata (`config.pbtxt` → `parameters.metadata.string_value`)

The exporter embeds a JSON blob. These fields are common to all backends
(`backend_api::CommonMeta`, checked by `tools/verify_repo.py`):

```json
{ "family": "<name>", "task": "prompt-injection",
  "labels": ["SAFE", "INJECTION"], "positive": [1],
  "inputs": ["input_ids", "attention_mask"], "outputs": ["logits"],
  "tokenizer": "tokenizer.json", "tokenizer_sha256": "…",
  "max_tokens": 8192, "window": 2048, "stride": 512, "pad_id": 0,
  "onnx_sha256": "…", "source": {"repo": "…", "revision": "<commit sha>", "file": "…"},
  "license": "Apache-2.0" }
```

Add any extra fields your backend needs and parse them in `create`. Use
`tools/common.py` (`build_config_pbtxt`, `inspect_onnx`,
`classifier_io_specs`, `write_repo`) to write the repository. `write_repo`
copies `tokenizer.json` next to `config.pbtxt` and any ONNX external-data
files next to `model.onnx`.

**Pin upstream revisions.** A catalog name must resolve to a commit sha, never
to "the latest `main`". The revision is part of the model fingerprint.

### Python hooks (`triton_repo.py`)

`tools/make_fixtures.py` and `tools/verify_repo.py` discover every
`backends/*/triton_repo.py` automatically:

```python
FAMILY = "<name>"
def check(conf, meta) -> None: ...                       # assert family-specific config.pbtxt shape
def make_fixture(repo: Path, workdir: Path) -> str: ...  # tiny ONNX + tokenizer, returns model name
```

A fixture should be small but behave like a detector (react to some token),
so the contract tests can check windowing, spans and aggregation for real.

## Registering a new backend

The workspace already includes `backends/*`. Then:

1. Root `Cargo.toml`, under `[workspace.dependencies]`:
   `backend-<name> = { path = "backends/<name>" }`
2. `gateway/Cargo.toml`: add `backend-<name> = { workspace = true, optional = true }`,
   a feature `<name> = ["dep:backend-<name>"]`, and add it to `default` if it
   should be built in by default.
3. `gateway/src/registry.rs`, in `BACKENDS`:
   ```rust
   #[cfg(feature = "<name>")]
   (backend_<name>::FAMILY, backend_<name>::create),
   ```
4. Add a `<name>` extra in `pyproject.toml` for the exporter's dependencies.
5. Add the feature to `scripts/test-all.sh` (feature isolation build) and to
   `FEATURES` in `docker/` and `.github/workflows/release.yml`.

Before opening a PR, document the **weights license** in your backend's README,
and add it to the "Supported backends" and "Model licensing" sections of the
root README. Pocket Network suppliers rely on that information. Add the
model's independent evaluation numbers (and their caveats) to
"Detection quality".
