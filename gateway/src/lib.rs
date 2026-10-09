//! Model-agnostic prompt-injection detection gateway in front of Triton.
//! Serves the native `/v1/detect` API plus Lakera Guard and Azure Prompt
//! Shields compatibility layers. All model-specific logic lives in
//! `backends/<family>/` crates and is selected at startup through
//! [`registry`]; the window policy, batching and aggregation live in
//! [`pipeline`].

pub mod api;
pub mod azure;
pub mod config;
pub mod error;
pub mod lakera;
pub mod native;
pub mod pipeline;
pub mod registry;
pub mod triton;
