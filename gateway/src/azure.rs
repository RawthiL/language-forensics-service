//! `POST /contentsafety/text:shieldPrompt`: the Azure AI Content Safety
//! Prompt Shields shape, so a client written against
//! `{endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01` can
//! swap its endpoint and subscription key. Any `api-version` is accepted.
//! Not normative; `docs/API_COMPAT.md` records what is and is not reproduced.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::api::{AppState, authorize, read_json};
use crate::error::Result;
use crate::pipeline;

/// Upstream request. Unknown fields are ignored.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShieldRequest {
    pub user_prompt: String,
    #[serde(default)]
    pub documents: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShieldResponse {
    pub user_prompt_analysis: Analysis,
    pub documents_analysis: Vec<Analysis>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub attack_detected: bool,
}

pub async fn shield_prompt(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<ShieldResponse>> {
    authorize(&headers, &state.cfg)?;
    let ready = state.ready()?;
    let body: ShieldRequest = read_json(req, &state.cfg).await?;

    let mut texts = Vec::with_capacity(1 + body.documents.len());
    texts.push(body.user_prompt);
    texts.extend(body.documents);
    let scored = pipeline::score_texts(&state.cfg, &state.triton, ready, texts).await?;
    let threshold = state.cfg.scoring.threshold;
    let mut analyses = scored.texts.iter().map(|s| Analysis {
        attack_detected: s.score >= threshold,
    });
    let user_prompt_analysis = analyses.next().expect("user prompt is always scored");
    Ok(Json(ShieldResponse {
        user_prompt_analysis,
        documents_analysis: analyses.collect(),
    }))
}
