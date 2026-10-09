//! Meta Llama Prompt Guard 2 backend (22M: DeBERTa-v3-xsmall, 86M:
//! mDeBERTa-v3-base).
//!
//! Model contract (written by `backends/prompt_guard/export.py`):
//! - inputs: `input_ids`, `attention_mask` INT64 `[batch, seq]`, seq <= 512;
//! - output: `logits` FP32 `[batch, 2]`, labels `BENIGN` (0) / `MALICIOUS` (1);
//! - tokenizer: the upstream `tokenizer.json` (SentencePiece Unigram,
//!   `[CLS] $A [SEP]`).
//!
//! The model card says to split inputs longer than 512 tokens into segments
//! and scan them in parallel; the gateway's window policy does exactly that.
//! There is no obfuscation normalizer in the upstream tokenizer (only Strip
//! and SentencePiece's precompiled NFKC-like charsmap).
//!
//! LICENSE NOTE: this crate is MIT. The weights are under the Llama 4
//! Community License — see `backends/prompt_guard/README.md`.

use backend_api::text::TextClassifier;
use backend_api::{
    Backend, BackendError, InputTensor, ModelInfo, OutputTensor, Result, Window, WindowParams,
};

pub const FAMILY: &str = "prompt-guard";

/// DeBERTa's absolute context; the 2.x models were trained at 512.
const MAX_CONTEXT: usize = 512;

pub struct PromptGuard {
    inner: TextClassifier,
}

pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>> {
    let inner = TextClassifier::load(info)?;
    let m = &inner.meta;
    let bad = |msg: String| {
        Err(BackendError::InvalidModel(format!(
            "model '{}': {msg}",
            info.name
        )))
    };
    if !(2..=3).contains(&m.labels.len()) {
        return bad(format!(
            "expected a 2-label (v2) or 3-label (v1) head, got {:?}",
            m.labels
        ));
    }
    if m.positive
        .iter()
        .any(|&p| m.labels[p].eq_ignore_ascii_case("BENIGN"))
    {
        return bad(format!(
            "positive {:?} includes the BENIGN label of {:?}",
            m.positive, m.labels
        ));
    }
    if m.max_tokens > MAX_CONTEXT {
        return bad(format!(
            "max_tokens {} exceeds Prompt Guard's {MAX_CONTEXT}-token context",
            m.max_tokens
        ));
    }
    Ok(Box::new(PromptGuard { inner }))
}

impl Backend for PromptGuard {
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
    use backend_api::text::sha256_hex;
    use serde_json::{Value, json};

    use super::*;

    const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
      "added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
      "post_processor":null,"decoder":null,
      "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hi":1},"unk_token":"[UNK]"}}"#;

    fn try_create(labels: Value, positive: Value, max_tokens: usize) -> Result<Box<dyn Backend>> {
        let dir = std::env::temp_dir().join(format!("lfs-pg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), TOKENIZER).unwrap();
        let meta = json!({
            "family": FAMILY, "labels": labels, "positive": positive,
            "inputs": ["input_ids", "attention_mask"], "outputs": ["logits"],
            "tokenizer": "tokenizer.json", "tokenizer_sha256": sha256_hex(TOKENIZER.as_bytes()),
            "max_tokens": max_tokens, "window": max_tokens, "stride": max_tokens / 4, "pad_id": 0
        });
        create(&ModelInfo {
            name: "pg",
            metadata: &meta,
            max_batch_size: 16,
            model_dir: &dir,
        })
    }

    #[test]
    fn accepts_v2_labels() {
        let b = try_create(json!(["BENIGN", "MALICIOUS"]), json!([1]), 512).unwrap();
        assert_eq!(b.family(), FAMILY);
        assert_eq!(
            b.default_window(),
            WindowParams {
                window: 512,
                stride: 128
            }
        );
    }

    #[test]
    fn accepts_v1_three_label_head() {
        assert!(
            try_create(
                json!(["BENIGN", "INJECTION", "JAILBREAK"]),
                json!([1, 2]),
                512
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_benign_as_positive() {
        let err = try_create(json!(["BENIGN", "MALICIOUS"]), json!([0]), 512)
            .err()
            .unwrap();
        assert!(err.to_string().contains("BENIGN"), "{err}");
    }

    #[test]
    fn rejects_context_beyond_512() {
        assert!(try_create(json!(["BENIGN", "MALICIOUS"]), json!([1]), 1024).is_err());
    }
}
