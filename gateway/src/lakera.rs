//! `POST /v2/guard`: the Lakera Guard v2 screening shape, so a client written
//! against `https://api.lakera.ai/v2/guard` can swap its base URL and API key.
//! Not normative; `docs/API_COMPAT.md` records what is and is not reproduced.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{AppState, authorize, read_json};
use crate::error::{ApiError, Result};
use crate::pipeline;

/// Upstream request. Unknown fields are ignored (forward compatible).
#[derive(Debug, Deserialize)]
pub struct GuardRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub breakdown: bool,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    /// A string, or OpenAI-style content parts (`[{"type":"text","text":...}]`).
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct GuardResponse {
    pub flagged: bool,
    pub metadata: GuardMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breakdown: Option<Vec<Detection>>,
}

#[derive(Debug, Serialize)]
pub struct GuardMetadata {
    pub request_uuid: String,
}

#[derive(Debug, Serialize)]
pub struct Detection {
    pub project_id: Option<String>,
    pub policy_id: String,
    pub detector_id: String,
    pub detector_type: &'static str,
    pub detected: bool,
    pub message_id: usize,
}

/// Roles screened for prompt attacks. System prompts are the developer's own
/// instructions (scanning them only produces false positives); assistant
/// turns are model output.
fn screened(role: &str) -> bool {
    matches!(role, "user" | "tool" | "function")
}

fn text_of(content: &Value) -> Result<String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => Ok(parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n")),
        _ => Err(ApiError::BadRequest(
            "message content must be a string or an array of content parts".into(),
        )),
    }
}

pub async fn guard(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<GuardResponse>> {
    authorize(&headers, &state.cfg)?;
    let ready = state.ready()?;
    let body: GuardRequest = read_json(req, &state.cfg).await?;
    if body.messages.is_empty() {
        return Err(ApiError::BadRequest("'messages' must not be empty".into()));
    }

    let mut ids = Vec::new();
    let mut texts = Vec::new();
    for (i, m) in body.messages.iter().enumerate() {
        if screened(&m.role) {
            ids.push(i);
            texts.push(text_of(&m.content)?);
        }
    }
    let threshold = state.cfg.scoring.threshold;
    let scores = if texts.is_empty() {
        Vec::new()
    } else {
        pipeline::score_texts(&state.cfg, &state.triton, ready, texts)
            .await?
            .texts
    };
    let flagged = scores.iter().any(|s| s.score >= threshold);
    let breakdown = body.breakdown.then(|| {
        ids.iter()
            .zip(&scores)
            .map(|(&message_id, s)| Detection {
                project_id: body.project_id.clone(),
                policy_id: "language-forensics-default".into(),
                detector_id: format!("language-forensics-{}", state.cfg.default_model),
                detector_type: "prompt_attack",
                detected: s.score >= threshold,
                message_id,
            })
            .collect()
    });
    Ok(Json(GuardResponse {
        flagged,
        metadata: GuardMetadata {
            request_uuid: uuid::Uuid::new_v4().to_string(),
        },
        breakdown,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_parts_are_joined() {
        let v = json!([{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]);
        assert_eq!(text_of(&v).unwrap(), "a\nb");
        assert!(text_of(&json!(3)).is_err());
    }

    #[test]
    fn system_and_assistant_are_not_screened() {
        assert!(screened("user") && screened("tool"));
        assert!(!screened("system") && !screened("assistant"));
    }
}
