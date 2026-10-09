//! Horizon-Labs prompt-injection-guard backend (base: mmBERT, large: bge-m3).
//!
//! Model contract (written by `backends/horizon_guard/export.py`):
//! - inputs: `input_ids`, `attention_mask` INT64 `[batch, seq]`, seq <= 8192;
//! - output: `logits` FP32 `[batch, 2]`, labels `SAFE` (0) / `INJECTION` (1);
//! - tokenizer: the upstream `tokenizer.json`, whose normalizer already
//!   decodes smuggled text (tag characters, variation-selector bytes), applies
//!   NFKC and strips invisible characters, so the Rust `tokenizers` crate runs
//!   the exact normalization the model was trained with.
//!
//! The upstream evaluation scores long inputs as the max over windows of
//! min(2048, model max) tokens with a quarter-window stride; the exporter
//! writes that as the default window policy.
//!
//! LICENSE NOTE: this crate is MIT. The Horizon-Labs weights are Apache-2.0
//! — see `backends/horizon_guard/README.md`.

use backend_api::text::TextClassifier;
use backend_api::{
    Backend, BackendError, InputTensor, ModelInfo, OutputTensor, Result, Window, WindowParams,
};

pub const FAMILY: &str = "horizon-guard";

pub struct HorizonGuard {
    inner: TextClassifier,
}

pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>> {
    let inner = TextClassifier::load(info)?;
    let m = &inner.meta;
    if m.labels.len() != 2 {
        return Err(BackendError::InvalidModel(format!(
            "model '{}': horizon-guard is a binary head, got labels {:?}",
            info.name, m.labels
        )));
    }
    if m.positive
        .iter()
        .any(|&p| !m.labels[p].eq_ignore_ascii_case("INJECTION"))
    {
        return Err(BackendError::InvalidModel(format!(
            "model '{}': positive {:?} must point at the INJECTION label of {:?}",
            info.name, m.positive, m.labels
        )));
    }
    Ok(Box::new(HorizonGuard { inner }))
}

impl Backend for HorizonGuard {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn labels(&self) -> &[String] {
        &self.inner.meta.labels
    }

    fn positive(&self) -> &[usize] {
        &self.inner.meta.positive
    }

    fn output_names(&self) -> &[String] {
        &self.inner.meta.outputs
    }

    fn max_tokens(&self) -> usize {
        self.inner.meta.max_tokens
    }

    fn default_window(&self) -> WindowParams {
        self.inner.default_window()
    }

    fn preprocess(&self, text: &str, params: &WindowParams) -> Result<Vec<Window>> {
        self.inner.windows(text, params)
    }

    fn pack(&self, windows: &[&Window], seq_len: usize) -> Vec<InputTensor> {
        self.inner.pack(windows, seq_len)
    }

    fn postprocess(&self, outputs: &[OutputTensor], n: usize) -> Result<Vec<f64>> {
        self.inner.scores(outputs, n)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use backend_api::text::sha256_hex;
    use serde_json::json;

    use super::*;

    const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
      "added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
      "post_processor":null,"decoder":null,
      "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hi":1},"unk_token":"[UNK]"}}"#;

    fn info_with(
        dir: &Path,
        labels: serde_json::Value,
        positive: serde_json::Value,
    ) -> serde_json::Value {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), TOKENIZER).unwrap();
        json!({
            "family": FAMILY, "labels": labels, "positive": positive,
            "inputs": ["input_ids", "attention_mask"], "outputs": ["logits"],
            "tokenizer": "tokenizer.json", "tokenizer_sha256": sha256_hex(TOKENIZER.as_bytes()),
            "max_tokens": 8192, "window": 2048, "stride": 512, "pad_id": 0
        })
    }

    fn try_create(
        labels: serde_json::Value,
        positive: serde_json::Value,
    ) -> Result<Box<dyn Backend>> {
        let dir = std::env::temp_dir().join(format!("lfs-horizon-{}", std::process::id()));
        let meta = info_with(&dir, labels, positive);
        create(&ModelInfo {
            name: "hg",
            metadata: &meta,
            max_batch_size: 16,
            model_dir: &dir,
        })
    }

    #[test]
    fn accepts_upstream_labels() {
        let b = try_create(json!(["SAFE", "INJECTION"]), json!([1])).unwrap();
        assert_eq!(b.family(), FAMILY);
        assert_eq!(
            b.default_window(),
            WindowParams {
                window: 2048,
                stride: 512
            }
        );
    }

    #[test]
    fn rejects_positive_pointing_at_safe() {
        let err = try_create(json!(["SAFE", "INJECTION"]), json!([0]))
            .err()
            .unwrap();
        assert!(err.to_string().contains("INJECTION"), "{err}");
    }

    #[test]
    fn rejects_non_binary_heads() {
        assert!(try_create(json!(["BENIGN", "INJECTION", "JAILBREAK"]), json!([1])).is_err());
    }
}
