//! The scoring pipeline, shared by every API surface: text -> windows
//! (backend) -> length-bucketed batches -> Triton -> per-window scores
//! (backend) -> max per text.
//!
//! Everything here is service-level policy that no backend knows about, and
//! all of it is part of the pipeline fingerprint: the window size and stride,
//! and max-aggregation over windows. Texts are never truncated; a request
//! that would need more windows than `limits.max_windows` is refused.

use std::sync::Arc;
use std::time::Instant;

use backend_api::{Backend, Window, WindowParams};
use futures_util::{StreamExt, TryStreamExt, stream};
use serde::Serialize;
use serde_json::Value;

use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::triton::TritonClient;

/// The loaded model plus the resolved scoring policy.
pub struct Ready {
    pub backend: Arc<dyn Backend>,
    pub window: WindowParams,
    /// Windows per Triton request (Triton `max_batch_size`, or 1).
    pub batch_limit: usize,
    pub model_fingerprint: String,
    pub pipeline_fingerprint: String,
    pub triton_version: Option<String>,
    /// Exporter metadata, for `/v1/models`.
    pub metadata: Value,
}

impl Ready {
    /// Resolve the window policy (config overrides the exporter's default),
    /// validate it against the model, and compute fingerprints.
    pub fn new(
        cfg: &Config,
        model: &str,
        backend: Arc<dyn Backend>,
        triton_config: &Value,
        metadata: Value,
        triton_version: Option<String>,
    ) -> Result<Self> {
        let default = backend.default_window();
        let window = cfg.scoring.window_tokens.unwrap_or(default.window);
        let stride = cfg
            .scoring
            .stride_tokens
            .unwrap_or(match cfg.scoring.window_tokens {
                Some(w) => w / 4,
                None => default.stride,
            });
        let window = WindowParams { window, stride };
        if window.window > backend.max_tokens() {
            return Err(ApiError::Internal(format!(
                "scoring.window_tokens {} exceeds model '{model}' maximum of {} tokens",
                window.window,
                backend.max_tokens()
            )));
        }
        // Dry run: the backend validates the policy against its special tokens.
        backend
            .preprocess("", &window)
            .map_err(|e| ApiError::Internal(format!("scoring policy for '{model}': {e}")))?;

        let batch_limit = triton_config
            .get("max_batch_size")
            .and_then(Value::as_u64)
            .filter(|&b| b > 0)
            .unwrap_or(1) as usize;

        let short = |key: &str| {
            metadata
                .get(key)
                .and_then(Value::as_str)
                .map(|s| s.chars().take(12).collect::<String>())
                .unwrap_or_else(|| "unknown".into())
        };
        let revision = metadata
            .pointer("/source/revision")
            .and_then(Value::as_str)
            .map(|s| s.chars().take(12).collect::<String>())
            .unwrap_or_else(|| "local".into());
        let model_fingerprint = format!(
            "{}@{model}+{revision}/onnx:{}/tok:{}",
            backend.family(),
            short("onnx_sha256"),
            short("tokenizer_sha256"),
        );
        let pipeline_fingerprint = format!(
            "{model_fingerprint}/w{}s{}+max",
            window.window, window.stride
        );
        Ok(Self {
            backend,
            window,
            batch_limit,
            model_fingerprint,
            pipeline_fingerprint,
            triton_version,
            metadata,
        })
    }
}

/// Score of one text.
#[derive(Debug, Clone, Serialize)]
pub struct TextScore {
    /// Max injection probability over the text's windows.
    pub score: f64,
    pub windows: usize,
    /// Content tokens (overlaps counted once per window).
    pub tokens: usize,
    /// Code-point span of the original text covered by the top window.
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Milliseconds per stage.
#[derive(Debug, Clone, Serialize)]
pub struct Speed {
    pub preprocess: f64,
    pub inference: f64,
    pub postprocess: f64,
}

pub struct Scored {
    pub texts: Vec<TextScore>,
    pub speed: Speed,
}

/// Score every text, enforcing the request limits.
pub async fn score_texts(
    cfg: &Config,
    triton: &TritonClient,
    ready: &Ready,
    texts: Vec<String>,
) -> Result<Scored> {
    let limits = &cfg.limits;
    if texts.len() > limits.max_inputs {
        return Err(ApiError::TooLarge(format!(
            "{} texts in one request (max {})",
            texts.len(),
            limits.max_inputs
        )));
    }
    let chars: usize = texts.iter().map(|t| t.chars().count()).sum();
    if chars > limits.max_chars {
        return Err(ApiError::TooLarge(format!(
            "{chars} characters in one request (max {})",
            limits.max_chars
        )));
    }

    // Tokenization is CPU-bound: keep it off the async workers.
    let t0 = Instant::now();
    let backend = ready.backend.clone();
    let params = ready.window;
    let per_text: Vec<Vec<Window>> = tokio::task::spawn_blocking(move || {
        texts
            .iter()
            .map(|t| backend.preprocess(t, &params))
            .collect::<backend_api::Result<Vec<_>>>()
    })
    .await
    .map_err(|e| ApiError::Internal(format!("preprocess task failed: {e}")))?
    .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let preprocess = elapsed_ms(t0);

    let total: usize = per_text.iter().map(Vec::len).sum();
    if total > limits.max_windows {
        return Err(ApiError::TooLarge(format!(
            "request needs {total} windows of {} tokens (max {}); split it into smaller requests",
            ready.window.window, limits.max_windows
        )));
    }

    // Flatten, then group windows of similar length so each Triton request
    // pads to a small power-of-two bucket. Equal shapes also let Triton's
    // dynamic batcher merge batches from concurrent requests.
    let flat: Vec<&Window> = per_text.iter().flatten().collect();
    let batches = plan_batches(&flat, ready.batch_limit, ready.window.window);

    let t1 = Instant::now();
    let model = cfg.triton_model();
    let outputs = ready.backend.output_names();
    let results: Vec<(Vec<usize>, Vec<f64>)> = stream::iter(batches)
        .map(|(idx, seq_len)| {
            let rows: Vec<&Window> = idx.iter().map(|&i| flat[i]).collect();
            let inputs = ready.backend.pack(&rows, seq_len);
            async move {
                let out = triton.infer(model, &inputs, outputs).await?;
                let scores = ready
                    .backend
                    .postprocess(&out, idx.len())
                    .map_err(|e| ApiError::Upstream(e.to_string()))?;
                Ok::<_, ApiError>((idx, scores))
            }
        })
        .buffer_unordered(limits.max_concurrent_batches)
        .try_collect()
        .await?;
    let inference = elapsed_ms(t1);

    let t2 = Instant::now();
    let mut window_scores = vec![0f64; flat.len()];
    for (idx, scores) in results {
        for (i, s) in idx.into_iter().zip(scores) {
            window_scores[i] = s;
        }
    }
    let mut texts_out = Vec::with_capacity(per_text.len());
    let mut offset = 0;
    for windows in &per_text {
        let scores = &window_scores[offset..offset + windows.len()];
        offset += windows.len();
        texts_out.push(aggregate(windows, scores));
    }
    Ok(Scored {
        texts: texts_out,
        speed: Speed {
            preprocess,
            inference,
            postprocess: elapsed_ms(t2),
        },
    })
}

/// Max over windows; the top window's span is reported.
fn aggregate(windows: &[Window], scores: &[f64]) -> TextScore {
    let (best, score) =
        scores
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f64::NEG_INFINITY),
                |acc, (i, s)| if s > acc.1 { (i, s) } else { acc },
            );
    let w = &windows[best];
    TextScore {
        score: score.max(0.0),
        windows: windows.len(),
        tokens: windows.iter().map(|w| w.content_tokens).sum(),
        span: Span {
            start: w.start,
            end: w.end,
        },
    }
}

/// Group window indices into batches of at most `limit` windows sharing one
/// padded length (a power of two, at least 16, at most `cap`).
pub fn plan_batches(windows: &[&Window], limit: usize, cap: usize) -> Vec<(Vec<usize>, usize)> {
    let bucket = |len: usize| len.next_power_of_two().max(16).min(cap.max(len));
    let mut order: Vec<usize> = (0..windows.len()).collect();
    order.sort_by_key(|&i| windows[i].ids.len());
    let mut batches: Vec<(Vec<usize>, usize)> = Vec::new();
    for i in order {
        let b = bucket(windows[i].ids.len());
        match batches.last_mut() {
            Some((idx, seq)) if *seq == b && idx.len() < limit => idx.push(i),
            _ => batches.push((vec![i], b)),
        }
    }
    batches
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(len: usize) -> Window {
        Window {
            ids: vec![1; len],
            start: 0,
            end: len,
            content_tokens: len,
        }
    }

    #[test]
    fn batches_share_power_of_two_buckets_and_respect_limit() {
        let ws = [w(3), w(40), w(20), w(17), w(5), w(33)];
        let refs: Vec<&Window> = ws.iter().collect();
        let plan = plan_batches(&refs, 2, 512);
        // 3,5 -> 16 ; 17,20 -> 32 ; 33,40 -> 64
        assert_eq!(
            plan,
            vec![(vec![0, 4], 16), (vec![3, 2], 32), (vec![5, 1], 64)]
        );
        let plan = plan_batches(&refs, 1, 512);
        assert_eq!(plan.len(), 6);
    }

    #[test]
    fn bucket_never_exceeds_window_cap() {
        let ws = [w(300)];
        let refs: Vec<&Window> = ws.iter().collect();
        assert_eq!(plan_batches(&refs, 8, 300)[0].1, 300);
    }

    #[test]
    fn aggregate_takes_max_and_its_span() {
        let mut a = w(4);
        a.start = 10;
        a.end = 20;
        let agg = aggregate(&[w(4), a], &[0.1, 0.9]);
        assert_eq!(agg.score, 0.9);
        assert_eq!(agg.span, Span { start: 10, end: 20 });
        assert_eq!((agg.windows, agg.tokens), (2, 8));
    }
}
