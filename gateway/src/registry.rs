//! Compile-time registry of model backends.
//!
//! To add a backend: create `backends/<name>/` (see `backends/README.md`),
//! add it to the workspace + a `<name>` feature in `gateway/Cargo.toml`, and
//! add one line below.

use std::path::Path;
use std::sync::Arc;

use backend_api::{Backend, Factory, ModelInfo};
use serde_json::Value;

use crate::error::{ApiError, Result};

/// `(family, constructor)` for every backend compiled into this binary.
pub const BACKENDS: &[(&str, Factory)] = &[
    #[cfg(feature = "horizon-guard")]
    (backend_horizon_guard::FAMILY, backend_horizon_guard::create),
    #[cfg(feature = "prompt-guard")]
    (backend_prompt_guard::FAMILY, backend_prompt_guard::create),
];

/// Parse the exporter's metadata JSON embedded in a Triton model config.
pub fn metadata(model: &str, config: &Value) -> Result<Value> {
    let raw = config
        .pointer("/parameters/metadata/string_value")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::Internal(format!(
                "model '{model}' has no parameters.metadata string_value; \
                 re-export it with backends/<family>/export.py"
            ))
        })?;
    serde_json::from_str(raw)
        .map_err(|e| ApiError::Internal(format!("model '{model}' metadata is not valid JSON: {e}")))
}

/// Pick and construct the backend for a Triton model config, using the
/// `family` field of its embedded `parameters.metadata` JSON.
pub fn backend_for(
    model: &str,
    config: &Value,
    metadata: &Value,
    model_dir: &Path,
) -> Result<Arc<dyn Backend>> {
    let family = metadata
        .get("family")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Internal(format!("model '{model}' metadata has no 'family'")))?
        .to_ascii_lowercase();

    let factory = BACKENDS
        .iter()
        .find(|(f, _)| *f == family)
        .map(|(_, factory)| *factory)
        .ok_or_else(|| {
            let names: Vec<&str> = BACKENDS.iter().map(|(f, _)| *f).collect();
            ApiError::Internal(format!(
                "model '{model}' has family '{family}', but this gateway was built with {names:?}"
            ))
        })?;

    let info = ModelInfo {
        name: model,
        metadata,
        max_batch_size: config
            .get("max_batch_size")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        model_dir,
    };
    let backend = factory(&info).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(Arc::from(backend))
}
