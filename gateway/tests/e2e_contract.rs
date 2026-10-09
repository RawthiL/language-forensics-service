//! End-to-end contract tests: real gateway router + an in-process mock Triton
//! (binary tensor extension, INT64 inputs). The mock model fires on one token
//! id, so windowing, batching, aggregation and every API surface run for real
//! without Python, ONNX or weights.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use backend_api::text::sha256_hex;
use backend_api::tokenizers::Tokenizer;
use backend_api::tokenizers::models::wordlevel::WordLevel;
use backend_api::tokenizers::normalizers::Lowercase;
use backend_api::tokenizers::pre_tokenizers::whitespace::Whitespace;
use backend_api::tokenizers::processors::template::TemplateProcessing;
use gateway::api::{self, AppState};
use gateway::config::Config;
use gateway::triton::TritonClient;
use serde_json::{Value, json};

const MODEL: &str = "e2e-guard";
const KEY: &str = "e2e-key";
const TRIGGER: u32 = 4;

#[derive(Clone, Default)]
struct Mock {
    /// `[batch, seq]` of every infer call.
    calls: Arc<Mutex<Vec<(u64, u64)>>>,
    metadata: Arc<String>,
}

fn write_tokenizer(dir: &std::path::Path) -> String {
    let vocab: HashMap<String, u32> = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "ignore"]
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_normalizer(Some(Lowercase)).unwrap();
    tok.with_pre_tokenizer(Some(Whitespace {}));
    tok.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    ));
    let path = dir.join("tokenizer.json");
    tok.save(&path, false).unwrap();
    sha256_hex(&std::fs::read(&path).unwrap())
}

async fn mock_config(State(m): State<Mock>, Path(_model): Path<String>) -> Json<Value> {
    Json(json!({
        "name": MODEL, "backend": "onnxruntime", "max_batch_size": 4,
        "parameters": { "metadata": { "string_value": m.metadata.as_str() } }
    }))
}

async fn mock_server() -> Json<Value> {
    Json(json!({"name": "triton", "version": "2.99-mock"}))
}

async fn mock_infer(
    State(m): State<Mock>,
    headers: HeaderMap,
    Path(_model): Path<String>,
    body: Bytes,
) -> Response {
    let hcl: usize = headers["Inference-Header-Content-Length"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(hcl % 8, 0, "header must be padded for INT64 alignment");
    let header: Value = serde_json::from_slice(&body[..hcl]).unwrap();
    let inputs = header["inputs"].as_array().unwrap();
    let ids = &inputs[0];
    assert_eq!(ids["name"], "input_ids");
    assert_eq!(ids["datatype"], "INT64");
    let shape: Vec<u64> = ids["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let (n, seq) = (shape[0], shape[1]);
    m.calls.lock().unwrap().push((n, seq));
    if n > 4 {
        return (StatusCode::BAD_REQUEST, "batch over max_batch_size").into_response();
    }
    let blob = &body[hcl..hcl + (n * seq * 8) as usize];
    let values: Vec<i64> = blob
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out = Vec::new();
    for row in values.chunks_exact(seq as usize) {
        let z: f32 = if row.contains(&(TRIGGER as i64)) {
            5.0
        } else {
            -5.0
        };
        out.extend_from_slice(&(-z).to_le_bytes());
        out.extend_from_slice(&z.to_le_bytes());
    }
    let mut resp = serde_json::to_vec(&json!({
        "outputs": [{ "name": "logits", "datatype": "FP32", "shape": [n, 2],
                      "parameters": { "binary_data_size": out.len() } }]
    }))
    .unwrap();
    let len = resp.len();
    resp.extend_from_slice(&out);
    ([("Inference-Header-Content-Length", len.to_string())], resp).into_response()
}

async fn serve(app: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

struct Env {
    base: String,
    http: reqwest::Client,
    mock: Mock,
}

async fn start(extra_yaml: &str) -> Env {
    start_with_keys(extra_yaml, true).await
}

async fn start_with_keys(extra_yaml: &str, keys: bool) -> Env {
    let repo = std::env::temp_dir().join(format!("lfs-e2e-{}-{}", std::process::id(), uuid()));
    let model_dir: PathBuf = repo.join(MODEL);
    std::fs::create_dir_all(&model_dir).unwrap();
    let sha = write_tokenizer(&model_dir);
    let metadata = json!({
        "family": "horizon-guard", "task": "prompt-injection",
        "labels": ["SAFE", "INJECTION"], "positive": [1],
        "inputs": ["input_ids", "attention_mask"], "outputs": ["logits"],
        "tokenizer": "tokenizer.json", "tokenizer_sha256": sha,
        "max_tokens": 64, "window": 8, "stride": 2, "pad_id": 0,
        "onnx_sha256": "feedfacecafe0000", "source": {"repo": "x/y", "revision": "0123456789abcdef"}
    })
    .to_string();
    let mock = Mock {
        calls: Arc::default(),
        metadata: Arc::new(metadata),
    };
    let triton = Router::new()
        .route("/v2", get(mock_server))
        .route("/v2/models/{model}/config", get(mock_config))
        .route("/v2/models/{model}/infer", post(mock_infer))
        .with_state(mock.clone());
    let triton_addr = serve(triton).await;

    let yaml = format!(
        "triton: {{ url: \"http://{triton_addr}\" }}\n\
         model_repository: \"{}\"\n\
         default_model: \"{MODEL}\"\n\
         {}{extra_yaml}",
        repo.display(),
        if keys {
            format!("api_keys: [\"{KEY}\"]\n")
        } else {
            String::new()
        },
    );
    let cfg_path = repo.join("config.yaml");
    std::fs::write(&cfg_path, yaml).unwrap();
    let cfg = Arc::new(Config::load(&cfg_path).unwrap());
    let client = Arc::new(TritonClient::new(
        &cfg.triton.url,
        None,
        Duration::from_secs(5),
    ));
    let state = Arc::new(AppState::new(cfg, client));
    api::spawn_model_loader(state.clone());
    let addr = serve(api::router(state)).await;
    let env = Env {
        base: format!("http://{addr}"),
        http: reqwest::Client::new(),
        mock,
    };
    for _ in 0..100 {
        if let Ok(r) = env.http.get(format!("{}/health", env.base)).send().await
            && r.status() == 200
        {
            return env;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("gateway never became healthy");
}

fn uuid() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

impl Env {
    async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        let r = self
            .http
            .post(format!("{}{path}", self.base))
            .bearer_auth(KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = StatusCode::from_u16(r.status().as_u16()).unwrap();
        (status, r.json().await.unwrap())
    }
}

#[tokio::test]
async fn native_detect_windows_batches_and_aggregates() {
    let env = start("").await;
    // window 8 = CLS + 6 content + SEP, stride 2 -> step 4: 22 tokens -> 5 windows.
    let text = format!("{} ignore x", ["w"; 20].join(" "));
    let (status, body) = env
        .post(
            "/v1/detect",
            json!({ "inputs": [
                { "text": "hello there", "role": "user" },
                { "text": text, "role": "document", "id": "d1" }
            ]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["flagged"], true);
    let r = body["results"].as_array().unwrap();
    assert_eq!(r[0]["flagged"], false);
    assert_eq!(r[1]["flagged"], true);
    assert_eq!(r[1]["id"], "d1");
    assert_eq!(r[1]["role"], "document");
    assert_eq!(r[1]["windows"], 5);
    let (s, e) = (
        r[1]["span"]["start"].as_u64().unwrap(),
        r[1]["span"]["end"].as_u64().unwrap(),
    );
    let span: String = text
        .chars()
        .skip(s as usize)
        .take((e - s) as usize)
        .collect();
    assert!(span.contains("ignore"), "{span:?}");
    let model_fp = body["model_fingerprint"].as_str().unwrap();
    assert!(
        model_fp.starts_with("horizon-guard@e2e-guard+0123456789ab/onnx:feedfacecafe/tok:"),
        "{model_fp}"
    );
    assert_eq!(body["pipeline_fingerprint"], format!("{model_fp}/w8s2+max"));
    // 6 windows total, batch limit 4, all padded to the 8-token cap.
    let calls = env.mock.calls.lock().unwrap().clone();
    assert_eq!(calls.iter().map(|c| c.0).sum::<u64>(), 6, "{calls:?}");
    assert!(
        calls.iter().all(|&(n, seq)| n <= 4 && seq == 8),
        "{calls:?}"
    );
}

#[tokio::test]
async fn compatibility_layers() {
    let env = start("").await;
    let (status, body) = env
        .post(
            "/v2/guard",
            json!({ "breakdown": true, "messages": [
                { "role": "system", "content": "ignore this" },
                { "role": "user", "content": "IGNORE previous" }
            ]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["flagged"], true);
    assert_eq!(body["breakdown"].as_array().unwrap().len(), 1);
    assert_eq!(body["breakdown"][0]["message_id"], 1);
    assert_eq!(body["breakdown"][0]["detector_type"], "prompt_attack");

    let r = env
        .http
        .post(format!(
            "{}/contentsafety/text:shieldPrompt?api-version=2024-09-01",
            env.base
        ))
        .header("Ocp-Apim-Subscription-Key", KEY)
        .json(&json!({ "userPrompt": "hi", "documents": ["ignore me", "fine"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(
        body,
        json!({ "userPromptAnalysis": { "attackDetected": false },
                "documentsAnalysis": [{ "attackDetected": true }, { "attackDetected": false }] })
    );
}

#[tokio::test]
async fn errors_are_json_with_codes() {
    let env = start("limits: { max_windows: 3 }\napi: { lakera: false }\n").await;
    let r = env
        .http
        .post(format!("{}/v1/detect", env.base))
        .json(&json!({ "text": "x" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "unauthorized");

    let long = ["w"; 40].join(" ");
    let (status, body) = env.post("/v1/detect", json!({ "text": long })).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["code"], "too_large");

    let (status, _) = env.post("/v1/detect", json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Disabled layer is simply not routed.
    let r = env
        .http
        .post(format!("{}/v2/guard", env.base))
        .bearer_auth(KEY)
        .json(&json!({ "messages": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn no_api_keys_means_open_routes() {
    let env = start_with_keys("", false).await;
    // No credentials at all...
    let r = env
        .http
        .post(format!("{}/v1/detect", env.base))
        .json(&json!({ "text": "ignore that" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["flagged"], true);
    // ...and a stray vendor key is ignored rather than rejected.
    let r = env
        .http
        .get(format!("{}/v1/models", env.base))
        .bearer_auth("some-lakera-key")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
}
