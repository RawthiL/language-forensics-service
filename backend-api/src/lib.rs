//! The contract between the model-agnostic gateway and a model backend.
//!
//! The gateway owns everything client-facing (HTTP API, auth, limits, the
//! window policy, batching, Triton transport, aggregation, response
//! formatting). A backend owns everything model-specific:
//!
//! 1. reading its own metadata from the Triton model config and loading the
//!    tokenizer the exporter wrote next to it ([`Factory`]),
//! 2. turning a text into overlapping, model-ready token windows that cover
//!    all of it ([`Backend::preprocess`]),
//! 3. packing windows into input tensors ([`Backend::pack`]),
//! 4. decoding the raw output tensors into one injection probability per
//!    window ([`Backend::postprocess`]).
//!
//! A new model family is a new crate under `backends/<name>/` that exports a
//! `FAMILY` string and a `create` function matching [`Factory`], registered
//! in `gateway/src/registry.rs`. Most text classifiers can delegate to
//! [`text::TextClassifier`].

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

pub use tokenizers;

pub mod text;
pub mod util;

/// Errors a backend can report. The gateway maps `InvalidModel` to a startup
/// failure, `BadInput` to a 400 and `BadOutput` to an upstream (503) error.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The Triton model config / metadata / tokenizer is not usable.
    #[error("{0}")]
    InvalidModel(String),
    /// The text cannot be tokenized.
    #[error("{0}")]
    BadInput(String),
    /// Triton returned tensors that do not match what the backend expects.
    #[error("{0}")]
    BadOutput(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// What the gateway knows about the served model when constructing a backend.
pub struct ModelInfo<'a> {
    /// Triton model name.
    pub name: &'a str,
    /// Parsed JSON of `parameters.metadata.string_value` from `config.pbtxt`
    /// (written by the backend's own exporter).
    pub metadata: &'a Value,
    /// Triton `max_batch_size` (0 = the graph carries an explicit batch dim).
    pub max_batch_size: u64,
    /// This model's directory in the gateway's read-only view of the Triton
    /// model repository. Tokenizer files live here, next to `config.pbtxt`.
    pub model_dir: &'a Path,
}

/// Constructor every backend crate exports as `pub fn create`.
pub type Factory = fn(&ModelInfo) -> Result<Box<dyn Backend>>;

/// Tensor payload. Text models take integer token ids; FP32 exists for
/// models with float inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    I64(Vec<i64>),
    F32(Vec<f32>),
}

impl TensorData {
    /// Triton datatype name.
    pub fn datatype(&self) -> &'static str {
        match self {
            TensorData::I64(_) => "INT64",
            TensorData::F32(_) => "FP32",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::I64(v) => v.len(),
            TensorData::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Little-endian bytes (Triton binary tensor extension).
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// One input tensor ready to send to Triton.
#[derive(Debug, Clone)]
pub struct InputTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub data: TensorData,
}

/// One output tensor returned by Triton, converted to f32.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

/// Window policy, chosen by the gateway (it is part of the pipeline
/// fingerprint). Both values count tokens and include special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowParams {
    /// Tokens per window, special tokens included.
    pub window: usize,
    /// Tokens shared by consecutive windows.
    pub stride: usize,
}

/// One window of a text: model-ready token ids (special tokens included)
/// plus the span of the ORIGINAL text it covers, in Unicode code points,
/// end-exclusive.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    pub ids: Vec<u32>,
    pub start: usize,
    pub end: usize,
    /// Content tokens (excluding special tokens) in this window.
    pub content_tokens: usize,
}

/// A model backend. Implementations must be cheap to share across threads;
/// `preprocess` runs on a blocking worker thread.
pub trait Backend: Send + Sync {
    /// Family identifier (matches the `family` metadata field).
    fn family(&self) -> &'static str;
    /// Output labels, indexed by logit position.
    fn labels(&self) -> &[String];
    /// Logit positions whose probabilities sum to the injection score.
    fn positive(&self) -> &[usize];
    /// Triton output tensor names to request, in order.
    fn output_names(&self) -> &[String];
    /// Longest window the model accepts (tokens, special tokens included).
    fn max_tokens(&self) -> usize;
    /// The exporter's recommended window policy.
    fn default_window(&self) -> WindowParams;
    /// Text -> windows that together cover ALL of it. Never truncates.
    fn preprocess(&self, text: &str, params: &WindowParams) -> Result<Vec<Window>>;
    /// Windows -> Triton input tensors, each row right-padded to `seq_len`.
    fn pack(&self, windows: &[&Window], seq_len: usize) -> Vec<InputTensor>;
    /// Raw outputs for a batch of `n` windows -> injection probability per
    /// window, in batch order.
    fn postprocess(&self, outputs: &[OutputTensor], n: usize) -> Result<Vec<f64>>;
}

/// Metadata fields shared by all backends (each backend may read more).
#[derive(Debug, Clone, Deserialize)]
pub struct CommonMeta {
    /// Output labels, indexed by logit position.
    pub labels: Vec<String>,
    /// Logit positions counted as "injection".
    pub positive: Vec<usize>,
    /// ONNX input tensor names (`input_ids`, `attention_mask`, optional
    /// `token_type_ids`).
    pub inputs: Vec<String>,
    /// ONNX output tensor names (one logits output).
    pub outputs: Vec<String>,
    /// Tokenizer file name inside the model directory.
    pub tokenizer: String,
    /// sha256 (hex) of that file, checked at load.
    pub tokenizer_sha256: String,
    /// Longest sequence the model accepts.
    pub max_tokens: usize,
    /// Recommended window policy.
    pub window: usize,
    pub stride: usize,
    /// Padding token id.
    pub pad_id: u32,
}

impl CommonMeta {
    pub fn parse(info: &ModelInfo) -> Result<Self> {
        let meta: CommonMeta = serde_json::from_value(info.metadata.clone()).map_err(|e| {
            BackendError::InvalidModel(format!("model '{}' metadata: {e}", info.name))
        })?;
        let bad = |msg: String| {
            Err(BackendError::InvalidModel(format!(
                "model '{}': {msg}",
                info.name
            )))
        };
        if meta.labels.is_empty() {
            return bad("metadata has no labels".into());
        }
        if meta.positive.is_empty() || meta.positive.iter().any(|&p| p >= meta.labels.len()) {
            return bad(format!(
                "positive {:?} must be non-empty indices into labels {:?}",
                meta.positive, meta.labels
            ));
        }
        if meta.outputs.len() != 1 {
            return bad(format!(
                "expected exactly one logits output, got {:?}",
                meta.outputs
            ));
        }
        if meta.max_tokens == 0 || meta.window == 0 || meta.window > meta.max_tokens {
            return bad(format!(
                "window {} must be within 1..={} (max_tokens)",
                meta.window, meta.max_tokens
            ));
        }
        if meta.stride >= meta.window {
            return bad(format!(
                "stride {} must be smaller than window {}",
                meta.stride, meta.window
            ));
        }
        Ok(meta)
    }
}
