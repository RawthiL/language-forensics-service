use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{ApiError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerCfg,
    pub triton: TritonCfg,
    /// The gateway's read-only view of the Triton model repository. Tokenizer
    /// files are read from `<model_repository>/<model>/`.
    #[serde(default = "default_model_repository")]
    pub model_repository: PathBuf,
    /// Public model served by this deployment (== Triton model name unless
    /// `triton_model_name` overrides it).
    pub default_model: String,
    /// Accepted API keys (flat list). Omitted or empty = no authentication:
    /// every route is open (startup logs a warning).
    #[serde(default, deserialize_with = "null_as_empty")]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub scoring: Scoring,
    #[serde(default)]
    pub api: ApiCfg,
}

/// `api_keys:` with no value is YAML null: treat it like an omitted field.
fn null_as_empty<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<String>, D::Error> {
    Ok(Option::<Vec<String>>::deserialize(d)?.unwrap_or_default())
}

fn default_model_repository() -> PathBuf {
    PathBuf::from("triton/model_repository")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCfg {
    pub listen: String,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TritonCfg {
    pub url: String,
    /// Optional bearer key for an authenticated Triton front. Not required
    /// when Triton is network-internal.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Optional explicit Triton model name if it differs from `default_model`.
    #[serde(default)]
    pub triton_model_name: Option<String>,
}

fn default_timeout_ms() -> u64 {
    30_000
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// Largest accepted request body.
    pub max_body_kb: usize,
    /// Texts per request (`inputs`, messages, documents).
    pub max_inputs: usize,
    /// Total characters (code points) per request, across all texts.
    pub max_chars: usize,
    /// Total windows per request. A request needing more is refused with
    /// 413 rather than truncated: an unscanned tail is exactly where an
    /// injection would hide.
    pub max_windows: usize,
    /// Triton requests in flight per client request.
    pub max_concurrent_batches: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body_kb: 1024,
            max_inputs: 32,
            max_chars: 100_000,
            max_windows: 64,
            max_concurrent_batches: 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scoring {
    /// Tokens per window, special tokens included. Unset = the exporter's
    /// recommendation for this model.
    pub window_tokens: Option<usize>,
    /// Tokens shared by consecutive windows. Unset = the exporter's value, or
    /// a quarter of `window_tokens` when that is overridden.
    pub stride_tokens: Option<usize>,
    /// Score at or above which a text is flagged (native requests may
    /// override it; the compatibility layers always use this).
    pub threshold: f64,
}

impl Default for Scoring {
    fn default() -> Self {
        Self {
            window_tokens: None,
            stride_tokens: None,
            threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiCfg {
    /// `POST /v2/guard` (Lakera Guard v2 shape).
    pub lakera: bool,
    /// `POST /contentsafety/text:shieldPrompt` (Azure Prompt Shields shape).
    pub azure: bool,
}

impl Default for ApiCfg {
    fn default() -> Self {
        Self {
            lakera: true,
            azure: true,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            ApiError::Internal(format!("cannot read config {}: {e}", path.display()))
        })?;
        let cfg: Config = serde_yaml::from_str(&text)
            .map_err(|e| ApiError::Internal(format!("invalid config {}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(ApiError::Internal(m.into()));
        // An empty *list* means "no auth"; an empty *key* is a typo (e.g. an
        // unset env var templated into the file), never a credential.
        if self.api_keys.iter().any(String::is_empty) {
            return bad("api_keys contains an empty key");
        }
        if !(0.0..=1.0).contains(&self.scoring.threshold) {
            return bad("scoring.threshold must be between 0 and 1");
        }
        let l = &self.limits;
        if l.max_body_kb == 0
            || l.max_inputs == 0
            || l.max_chars == 0
            || l.max_windows == 0
            || l.max_concurrent_batches == 0
        {
            return bad("limits.* must all be positive");
        }
        Ok(())
    }

    /// True when requests must carry one of `api_keys`.
    pub fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    pub fn triton_model(&self) -> &str {
        self.triton
            .triton_model_name
            .as_deref()
            .unwrap_or(&self.default_model)
    }

    pub fn max_body_bytes(&self) -> usize {
        self.limits.max_body_kb * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(keys: &str) -> Result<Config> {
        let cfg: Config = serde_yaml::from_str(&format!(
            "triton: {{ url: \"http://t:8000\" }}\ndefault_model: m\n{keys}"
        ))
        .map_err(|e| ApiError::Internal(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    #[test]
    fn omitted_null_or_empty_keys_disable_auth() {
        for keys in ["", "api_keys:\n", "api_keys: []\n"] {
            let cfg = parse(keys).unwrap_or_else(|e| panic!("{keys:?}: {e}"));
            assert!(!cfg.auth_enabled(), "{keys:?}");
        }
    }

    #[test]
    fn listed_keys_enable_auth() {
        assert!(parse("api_keys: [\"k\"]\n").unwrap().auth_enabled());
    }

    #[test]
    fn empty_string_key_is_an_error() {
        let err = parse("api_keys: [\"\"]\n").err().unwrap().to_string();
        assert!(err.contains("empty key"), "{err}");
    }
}
