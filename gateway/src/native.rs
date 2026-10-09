//! `POST /v1/detect` and `GET /v1/models`: this service's own API, described
//! by `docs/openapi.json`.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::api::{AppState, authorize, read_json};
use crate::error::{ApiError, Result};
use crate::pipeline::{self, Span, Speed};

/// Where a text came from. Encoder classifiers do not read it; it is echoed
/// back so callers can tell direct (user) from indirect (document, tool)
/// injection in the results.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    System,
    Assistant,
    Document,
    Tool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectRequest {
    /// Shorthand for a single `{"text": ..., "role": "user"}` input.
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub inputs: Option<Vec<InputItem>>,
    /// Override `scoring.threshold` for this request.
    #[serde(default)]
    pub threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputItem {
    pub text: String,
    #[serde(default)]
    pub role: Option<Role>,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DetectResponse {
    pub id: String,
    pub flagged: bool,
    /// Max score over all inputs.
    pub score: f64,
    pub threshold: f64,
    pub results: Vec<InputResult>,
    pub model: String,
    pub model_fingerprint: String,
    pub pipeline_fingerprint: String,
    pub usage: Usage,
    pub speed: Speed,
}

#[derive(Debug, Serialize)]
pub struct InputResult {
    pub index: usize,
    pub id: Option<String>,
    pub role: Role,
    pub score: f64,
    pub flagged: bool,
    pub windows: usize,
    /// Code-point span of the original text covered by the top window.
    pub span: Span,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub characters: usize,
    pub tokens: usize,
    pub windows: usize,
}

pub async fn detect(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<DetectResponse>> {
    authorize(&headers, &state.cfg)?;
    let ready = state.ready()?;
    let body: DetectRequest = read_json(req, &state.cfg).await?;

    let threshold = body.threshold.unwrap_or(state.cfg.scoring.threshold);
    if !(0.0..=1.0).contains(&threshold) {
        return Err(ApiError::BadRequest(
            "threshold must be between 0 and 1".into(),
        ));
    }
    let inputs = match (body.text, body.inputs) {
        (Some(text), None) => vec![InputItem {
            text,
            role: Some(Role::User),
            id: None,
        }],
        (None, Some(inputs)) if !inputs.is_empty() => inputs,
        (None, Some(_)) => return Err(ApiError::BadRequest("'inputs' must not be empty".into())),
        (Some(_), Some(_)) => {
            return Err(ApiError::BadRequest(
                "provide either 'text' or 'inputs', not both".into(),
            ));
        }
        (None, None) => {
            return Err(ApiError::BadRequest(
                "one of 'text' or 'inputs' is required".into(),
            ));
        }
    };

    let characters = inputs.iter().map(|i| i.text.chars().count()).sum();
    let meta: Vec<(Option<String>, Role)> = inputs
        .iter()
        .map(|i| (i.id.clone(), i.role.unwrap_or(Role::User)))
        .collect();
    let texts = inputs.into_iter().map(|i| i.text).collect();
    let scored = pipeline::score_texts(&state.cfg, &state.triton, ready, texts).await?;

    let results: Vec<InputResult> = scored
        .texts
        .iter()
        .zip(meta)
        .enumerate()
        .map(|(index, (s, (id, role)))| InputResult {
            index,
            id,
            role,
            score: round(s.score),
            flagged: s.score >= threshold,
            windows: s.windows,
            span: s.span,
        })
        .collect();
    let score = scored.texts.iter().map(|s| s.score).fold(0.0, f64::max);
    Ok(Json(DetectResponse {
        id: format!("det_{}", uuid::Uuid::new_v4().simple()),
        flagged: score >= threshold,
        score: round(score),
        threshold,
        usage: Usage {
            characters,
            tokens: scored.texts.iter().map(|s| s.tokens).sum(),
            windows: scored.texts.iter().map(|s| s.windows).sum(),
        },
        results,
        model: state.cfg.default_model.clone(),
        model_fingerprint: ready.model_fingerprint.clone(),
        pipeline_fingerprint: ready.pipeline_fingerprint.clone(),
        speed: scored.speed,
    }))
}

/// Scores are rounded to 6 decimals: enough to threshold on, stable across
/// float noise in the last bits.
fn round(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

pub async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Result<Json<Value>> {
    authorize(&headers, &state.cfg)?;
    let ready = state.ready()?;
    let md = &ready.metadata;
    Ok(Json(json!({
        "model": state.cfg.default_model,
        "triton_model": state.cfg.triton_model(),
        "family": ready.backend.family(),
        "labels": ready.backend.labels(),
        "positive": ready.backend.positive(),
        "max_tokens": ready.backend.max_tokens(),
        "window": { "tokens": ready.window.window, "stride": ready.window.stride },
        "aggregation": "max",
        "threshold": state.cfg.scoring.threshold,
        "batch_limit": ready.batch_limit,
        "source": md.get("source"),
        "license": md.get("license"),
        "notice": md.get("notice"),
        "model_fingerprint": ready.model_fingerprint,
        "pipeline_fingerprint": ready.pipeline_fingerprint,
        "version": {
            "gateway": env!("CARGO_PKG_VERSION"),
            "triton": ready.triton_version,
        },
        "uptime_s": state.started.elapsed().as_secs_f64(),
    })))
}
