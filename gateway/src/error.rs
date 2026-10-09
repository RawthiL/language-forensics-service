use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// Every error, on every route, is serialized as
/// `{"error": {"code": "<code>", "message": "<message>"}}` with a fitting
/// status code. (The same envelope Azure Content Safety uses.)
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),

    #[error("Missing or invalid API key")]
    Unauthorized,

    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    TooLarge(String),

    #[error("{0}")]
    Upstream(String),

    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::Upstream(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::Unauthorized => "unauthorized",
            ApiError::NotFound(_) => "not_found",
            ApiError::TooLarge(_) => "too_large",
            ApiError::Upstream(_) => "unavailable",
            ApiError::Internal(_) => "internal",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "code": self.code(), "message": self.to_string() } });
        (self.status(), Json(body)).into_response()
    }
}
