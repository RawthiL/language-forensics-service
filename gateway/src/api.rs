use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use subtle::ConstantTimeEq;

use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::pipeline::Ready;
use crate::triton::TritonClient;
use crate::{azure, lakera, native, registry};

/// The OpenAPI document for every route, served at `/openapi.json`.
pub const OPENAPI: &str = include_str!("../../docs/openapi.json");

pub struct AppState {
    pub cfg: Arc<Config>,
    pub triton: Arc<TritonClient>,
    pub started: Instant,
    /// Set once the model is loaded from Triton (see [`spawn_model_loader`]).
    /// Until then every request gets a 503 JSON error instead of no answer.
    pub ready: OnceLock<Ready>,
    /// Why the model is not ready yet (last loader error), for 503 bodies.
    pub load_error: Mutex<Option<String>>,
}

impl AppState {
    pub fn new(cfg: Arc<Config>, triton: Arc<TritonClient>) -> Self {
        Self {
            cfg,
            triton,
            started: Instant::now(),
            ready: OnceLock::new(),
            load_error: Mutex::new(None),
        }
    }

    /// The loaded model, or a 503 explaining why it is not available yet.
    pub fn ready(&self) -> Result<&Ready> {
        self.ready.get().ok_or_else(|| {
            let reason = self
                .load_error
                .lock()
                .ok()
                .and_then(|e| e.clone())
                .unwrap_or_else(|| "still loading".into());
            ApiError::Upstream(format!(
                "model '{}' is not ready: {reason}",
                self.cfg.default_model
            ))
        })
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let mut app = Router::new()
        .route("/v1/detect", post(native::detect))
        .route("/v1/models", get(native::models))
        .route("/health", get(health))
        .route("/openapi.json", get(openapi));
    if state.cfg.api.lakera {
        app = app.route("/v2/guard", post(lakera::guard));
    }
    if state.cfg.api.azure {
        app = app.route(
            "/contentsafety/text:shieldPrompt",
            post(azure::shield_prompt),
        );
    }
    app.with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let model = &state.cfg.default_model;
    if let Err(e) = state.ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                serde_json::json!({ "status": "loading", "model": model, "error": e.to_string() }),
            ),
        );
    }
    // Readiness: confirm Triton still serves the model.
    match state.triton.model_config(state.cfg.triton_model()).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "healthy", "model": model })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                serde_json::json!({ "status": "unavailable", "model": model, "error": e.to_string() }),
            ),
        ),
    }
}

async fn openapi() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], OPENAPI)
}

/// Accept the deployment key from any of the headers the supported client
/// libraries send: `Authorization: Bearer` (native, Lakera SDKs), `X-API-Key`,
/// or `Ocp-Apim-Subscription-Key` (Azure SDKs). Keys are compared in
/// constant time against every configured key. With no keys configured,
/// every request is allowed (and any key a client sends is ignored).
pub fn authorize(headers: &HeaderMap, cfg: &Config) -> Result<()> {
    if !cfg.auth_enabled() {
        return Ok(());
    }
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let bearer = get(header::AUTHORIZATION.as_str()).and_then(|raw| {
        raw.strip_prefix("Bearer ")
            .or_else(|| raw.strip_prefix("bearer "))
    });
    let token = bearer
        .or_else(|| get("x-api-key"))
        .or_else(|| get("ocp-apim-subscription-key"))
        .ok_or(ApiError::Unauthorized)?;
    let mut ok = 0u8;
    for key in &cfg.api_keys {
        ok |= u8::from(bool::from(token.as_bytes().ct_eq(key.as_bytes())));
    }
    if ok == 1 {
        Ok(())
    } else {
        Err(ApiError::Unauthorized)
    }
}

/// Read a JSON body with the configured size cap; every failure becomes a
/// JSON error (axum's own extractor rejections are plain text).
pub async fn read_json<T: DeserializeOwned>(req: Request, cfg: &Config) -> Result<T> {
    let is_json = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|c| {
            c.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false);
    if !is_json {
        return Err(ApiError::BadRequest(
            "expected an application/json body".into(),
        ));
    }
    let limit = cfg.max_body_bytes();
    let body = axum::body::to_bytes(req.into_body(), limit)
        .await
        .map_err(|_| {
            ApiError::TooLarge(format!(
                "body too large (max {} KB)",
                cfg.limits.max_body_kb
            ))
        })?;
    serde_json::from_slice(&body)
        .map_err(|e| ApiError::BadRequest(format!("invalid JSON body: {e}")))
}

/// Load the model in the background, retrying with backoff until Triton
/// serves it. The HTTP server runs meanwhile and answers 503 with the reason.
pub fn spawn_model_loader(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let model = state.cfg.triton_model().to_string();
        let model_dir = state.cfg.model_repository.join(&model);
        let mut delay = Duration::from_millis(500);
        loop {
            let attempt = async {
                let config = state.triton.model_config(&model).await?;
                let metadata = registry::metadata(&model, &config)?;
                let backend = registry::backend_for(&model, &config, &metadata, &model_dir)?;
                let version = state.triton.server_metadata().await;
                Ready::new(&state.cfg, &model, backend, &config, metadata, version)
            }
            .await;
            match attempt {
                Ok(ready) => {
                    tracing::info!(
                        "model '{model}' ready -> backend '{}' (labels {:?}, window {} / stride {}, batch {}): {}",
                        ready.backend.family(),
                        ready.backend.labels(),
                        ready.window.window,
                        ready.window.stride,
                        ready.batch_limit,
                        ready.pipeline_fingerprint,
                    );
                    let _ = state.ready.set(ready);
                    if let Ok(mut e) = state.load_error.lock() {
                        *e = None;
                    }
                    return;
                }
                Err(e) => {
                    let msg = e.to_string();
                    if let Ok(mut last) = state.load_error.lock()
                        && last.as_deref() != Some(msg.as_str())
                    {
                        tracing::warn!("model '{model}' not ready (retrying): {msg}");
                        *last = Some(msg);
                    }
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(10));
                }
            }
        }
    })
}
