# Prompt-injection detection: models, APIs, effectiveness

Research notes for the first `language-forensics-service` detector. Gathered 2026-10-08.
The service should take the same shape as `../image-forensics-service`: a native `/v1/detect` API, vendor
compatibility layers, and one plugin folder per model under `detectors/backends/`.

---

## 1. TL;DR

- **Detectors are a filter, not a defense.** Every independent 2025–2026 evaluation says the same thing.
  Adaptive attackers beat detectors more than 90% of the time. Which detector looks best depends on the benchmark.
- **The default backend should be an Apache-2.0, ungated, long-context encoder trained on agent-shaped inputs.**
  Today that is **Horizon-Labs `prompt-injection-guard-base`** (308M, mmBERT, 8k ctx). Ship
  **Llama Prompt Guard 2 86M** as a second backend: it has the lowest false-positive rate on tool outputs,
  but its license is more restrictive and it only reads 512 tokens.
- **Publish our own API** (`/v1/detect`, OpenAPI spec) and add **two compatibility layers**:
  **Lakera Guard v2** (`POST /v2/guard`) and **Azure Prompt Shields** (`POST /contentsafety/text:shieldPrompt`).
  AWS Bedrock Guardrails and Google Model Armor depend on server-side config objects (guardrail IDs and
  templates), so a drop-in clone isn't practical.
- **Report a score and thresholds calibrated at a fixed FPR**, never a bare boolean. Treat direct input
  (user prompt) and indirect input (documents, tool outputs) as separate fields. Scan the whole text, not a
  truncated prefix.

---

## 2. Candidate models (open weights)

| Model | Base / size | Context | License | Gated | Notes |
|---|---|---|---|---|---|
| [Horizon-Labs/prompt-injection-guard-base](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) | mmBERT-base, 308M | 8k | **Apache-2.0** | No | Updated 2026-09-23 (v2). Built-in obfuscation normalizer (decodes smuggled text, NFKC, strips invisible chars). 30 languages. ONNX int8 and transformers.js available. Labels `SAFE`/`INJECTION` (same as ProtectAI v2). |
| [Horizon-Labs/prompt-injection-guard-large](https://huggingface.co/Horizon-Labs/prompt-injection-guard-large) | bge-m3, 568M | 8k (512 stride sliding window beyond) | Apache-2.0 | No | Self-reported: NotInject 93.2%, OR-Bench-hard 99.3%, BIPIA F1 0.68. Self-reported **~16% FPR on harmless role-play**. |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | mDeBERTa-base, 86M | **512** | Llama 4 Community License | Yes | Self-reported: 97.5% recall at 1% FPR, AUC .998 (EN). Lowest tool-output FPR in independent tests. Requires chunking. |
| Llama-Prompt-Guard-2-22M | DeBERTa-xsmall, 22M | 512 | Llama 4 Community License | Yes | 88.7% recall at 1% FPR (self-reported). ~19 ms on A100. Weaker multilingual (AUC .942). |
| [leolee99/PIGuard](https://huggingface.co/leolee99/PIGuard) (formerly InjecGuard) | DeBERTa-v3-base, 184M | 512 | MIT | No | Trained to reduce over-defense (NotInject). **Training data overlaps BIPIA verbatim**, so its BIPIA score is inflated. |
| protectai/deberta-v3-base-prompt-injection-v2 | DeBERTa-v3-base, 184M | 512 | Apache-2.0 | No | The common baseline. **30–67% FPR on benign tool outputs.** Do not use it as a default. |
| qualifire/prompt-injection-sentinel (v1/v2) | ModernBERT-large, 396–596M | 8k | Custom/"other" | Yes | Strong on its vendor's own benchmark. The license needs review before commercial serving. |
| [@stackone/defender](https://www.stackone.com/platform/prompt-injection-guard/) | MLP, 22 MB | — | Apache-2.0 | No | JS/npm only. ~4 ms on CPU. All published numbers are vendor-measured. |

---

## 3. How effective is detection today?

### 3.1 Vendor and model-card numbers (optimistic)
Model cards and vendor pages report 95–99% accuracy or recall, usually on held-out splits that resemble the
training data. Examples: Prompt Guard 2 at 97.5% recall at 1% FPR, Sentinel at 97.6% on Qualifire's own
benchmark, StackOne at 95.9% balanced accuracy, and Lakera claiming 98%+.

### 3.2 Independent: "Passing the Test You Trained On" (arXiv [2610.03448](https://arxiv.org/abs/2610.03448), Oct 2026)
The paper tests 15 detectors (plus 2 LLM judges, out of scope here) on AgentDojo, τ-bench, BIPIA and general text.

**False positives on benign tool outputs** (these count more than detection rate in production):

| Detector | AgentDojo FPR | τ-bench FPR | Tasks blocked |
|---|---|---|---|
| Horizon-Labs | 0.0% | 0.7% | 0% |
| Prompt Guard 2 (86M) | 0.6% | 0.0% | 2.1% |
| Wolf Defender | 0.9% | 0.0% | 3.1% |
| Prismor (1.5B) | 6.5% | 46.2% | 17.5% |
| ProtectAI v2 | 30.1% | 66.9% | 69.1% |

**Detection (TPR) at 1% FPR:**

| Detector | AgentDojo | τ-bench | BIPIA |
|---|---|---|---|
| Horizon-Labs | 82.2% | 100.0% | 37.7% |
| Wolf Defender | 73.8% | 90.8% | 3.5% |
| Prismor | 72.2% | 15.2% | 4.0% |
| PIGuard | **2.1%** | 52.7% | **95.1%** |

Key conclusions:
- **Rankings do not transfer between benchmarks.** Kendall's τ is 0.01 between BIPIA and AgentDojo.
  None of the pairs is statistically significant.
- **Detectors work on inputs whose form resembles their training inputs.** Format matters more than attack
  strings. Horizon-Labs had no overlap with any test set and still did best on agent benchmarks, because it
  was trained on agent-shaped data.
- Task-irrelevant injections (which make the agent do something unrelated) got past nearly every detector.
- The paper's own recommendations: measure FPR on real tool outputs, report TPR at a fixed low FPR, audit
  training data for overlap, and prefer detectors trained on agent-like inputs.

### 3.3 Adaptive attacks
- **"The Attacker Moves Second"** (Nasr, Carlini et al., arXiv [2510.09023](https://arxiv.org/abs/2510.09023)).
  Adaptive attacks (gradient, RL, search, human red-teaming) broke 12 recent defenses, with ASR above 90% for
  most. Many of those defenses had originally reported near-zero ASR.
- **Zhan et al. 2025.** Adaptive attacks against indirect prompt injection reached >90% ASR, including
  against stacked detectors.
- **Prompt Overflow** (arXiv [2605.23196](https://arxiv.org/abs/2605.23196)). Prompt Guard and Granite
  Guardian truncate or segment long inputs. Spreading an injection across filler text gets past them while
  the full prompt still works on the target LLM. **Our service must scan the whole input and aggregate across
  windows**, not just score the first 512 tokens.
- **PIArena** (ACL 2026, [arXiv 2604.08499](https://arxiv.org/abs/2604.08499),
  [code](https://github.com/sleeepeer/PIArena)) found limited generalization across tasks and vulnerability to
  adaptive attacks. Detection is fundamentally hard when the injected task looks like the legitimate one.
  We could use it as our evaluation harness.

### 3.4 Agent-level impact
Meta's LlamaFirewall paper ([arXiv 2505.03574](https://arxiv.org/abs/2505.03574)) measured on AgentDojo:
- Prompt Guard 2 86M: **−57% ASR** with 1.5% utility loss.
- Prompt Guard 2 22M: −41% ASR with no utility loss.

These numbers are self-reported, but they are the honest way to frame it. The detector cuts attacks roughly
in half. It does not eliminate them.

**Bottom line for the README's "Detection quality" section:** expect 50–100% detection at 1% FPR on inputs
shaped like the training data. Expect single-digit detection on unfamiliar formats. Expect near-zero against
an adaptive attacker. The operator has to calibrate the threshold on their own traffic.

---

## 4. Existing commercial APIs

| Vendor | Endpoint | Request | Response | Clone-friendly? |
|---|---|---|---|---|
| **Lakera Guard v2** (now part of Check Point) | `POST https://api.lakera.ai/v2/guard`, `Authorization: Bearer` | `{messages:[{role,content}], project_id?, breakdown?, payload?, tools?}` (OpenAI chat format) | `{flagged, metadata:{request_uuid}}`. With `breakdown:true`, a per-detector list (`detector_type`, `detected`). `/v2/guard/results` returns confidence levels L1 (confident) to L5 (unlikely). | **Yes.** It takes plain chat messages and the policy is just an opaque `project_id`. Detector-type names need confirming against the live docs before we clone the breakdown. |
| **Azure AI Content Safety – Prompt Shields** | `POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01`, `Ocp-Apim-Subscription-Key` | `{userPrompt: str, documents: [str]}` | `{userPromptAnalysis:{attackDetected}, documentsAnalysis:[{attackDetected}]}` | **Yes.** It is tiny, and it already separates direct from indirect injection. Boolean only, so our score has to go in an extension field. |
| AWS Bedrock Guardrails | `ApplyGuardrail` (SigV4), filter `PROMPT_ATTACK`, strength NONE/LOW/MEDIUM/HIGH | references a pre-created `guardrailIdentifier` and version | assessments per policy | No. It needs SigV4 and a server-side guardrail resource. |
| Google Model Armor | `templates/{t}:sanitizeUserPrompt` | references a pre-created template | `filterResults.pi_and_jailbreak.matchState` and confidence level | No. It needs GCP auth and a template resource. |

### Recommendation: publish ours, and copy two
Do what the image service does:
1. **A native `/v1/detect` with a published OpenAPI spec.** This is where scores, per-segment results,
   per-window spans, the model id and the pipeline fingerprint live. Neither Lakera nor Azure exposes
   calibrated scores, and Pocket callers will want them.
2. **A Lakera `/v2/guard` dialect.** This is the most common standalone guardrail API, and the
   chat-messages input fits LLM gateway integrations.
3. **An Azure `text:shieldPrompt` dialect.** It is trivial to implement, and its `userPrompt` + `documents`
   split matches the direct/indirect distinction we need anyway.

Neither vendor publishes an OpenAPI spec we can reuse verbatim. Like `docs/API_COMPAT.md` in the image
service, we should document field-by-field what we do and do not reproduce.

---

## 5. Sketch of the native contract

> **Implemented (2026-10-08):** the shipped contract is [`docs/openapi.json`](../openapi.json). It differs from this sketch in one way: `threshold` is a number. Named presets wait for calibration data, which this repository does not have yet.

```jsonc
// POST /v1/detect
{
  "inputs": [
    {"role": "user",     "text": "..."},             // direct
    {"role": "document", "text": "...", "id": "doc1"}, // indirect: RAG chunk, email, web page
    {"role": "tool",     "text": "...", "id": "call_42"}
  ],
  "context": {"task": "Summarise my inbox"},         // optional; reserved, unused by encoder backends
  "threshold": "balanced"                            // or "strict" / "permissive" / float
}
// 200
{
  "model": "horizon-guard-base@<sha>",
  "pipeline": "<fingerprint>",
  "flagged": true,
  "results": [
    {"index": 0, "score": 0.02, "flagged": false},
    {"index": 1, "score": 0.97, "flagged": true,
     "spans": [{"start": 1840, "end": 2350, "score": 0.97}]}   // window that fired
  ]
}
```

Implications for the service layer, the equivalent of the image service's crop policy:
- **Normalization** before scoring: NFKC, strip zero-width and bidi characters, decode common smuggling
  (base64, leetspeak, homoglyphs). Horizon's tokenizer does some of this already. Keep it in the service layer
  so every backend gets it.
- **Window policy** (the analogue of crops): run a sliding window over the full text and aggregate with
  `max`. Never truncate. This is the Prompt Overflow lesson.
- **Threshold presets calibrated per backend at 0.1% / 1% / 5% FPR**, measured on benign tool-output data.
  Ship them in each backend's catalog.
- **A `stub` backend** for CI, as in the image service.

---

## 6. Suggested backend roadmap

1. `horizon_guard`: base (default) and large. Apache-2.0, no gating, 8k ctx, ONNX for CPU.
2. `prompt_guard`: 86M and 22M. Gated. The Llama license needs "Built with Llama" attribution and has the
   700M-MAU clause. 512 ctx, so it relies on the window policy.
3. `piguard`: MIT. Useful for an ensemble on document-style (BIPIA-like) inputs, where it complements the
   agent-trained models.

Evaluation: wire up **PIArena**, **AgentDojo** and **BIPIA**, plus a benign tool-output set. Report TPR at
1% FPR per benchmark in the README, the same way the image service reports its benchmark caveats.

## 7. Serving architecture

> **Implemented (2026-10-08)** as described below. One finding changed the plan: Horizon's obfuscation normalizer *is* serialized in its `tokenizer.json` (192 `Replace` rules plus NFKC), so the Rust `tokenizers` crate runs it natively and nothing had to be reimplemented. Gateway scores match a Python `tokenizers` + ORT reference to 1e-6 (`tests/test_real_model.py`).

Scope: small encoder classifiers only (22M–568M). LLM-as-judge is out of scope.

**Decision: Rust gateway + Triton (ONNX Runtime backend), the same shape as `../image-inference-server`. No vLLM.**

| Option | Verdict | Why |
|---|---|---|
| **Triton + onnxruntime** | ✅ | Runs any architecture that exports to ONNX: DeBERTa-v2/v3 (Prompt Guard 2, PIGuard), mmBERT/ModernBERT (Horizon), bge-m3/XLM-R. Dynamic batching matters because one request turns into many windows. Reuses the sibling's exporter, `verify_repo.py` and `mock_triton.py`, and runs on CPU or GPU. |
| vLLM | ❌ | Built for autoregressive decoding (paged KV cache, continuous batching of decode steps). An encoder classifier does a single forward pass, so none of that helps. DeBERTa-v2/v3 support in its pooling runner is not established. |
| HF Text Embeddings Inference | ❌ | Sequence classification is listed only for XLM-RoBERTa/CamemBERT-style models (plus ModernBERT/GTE re-rankers). **No DeBERTa**, which rules out Prompt Guard 2 and PIGuard. |
| ORT in-process in the gateway (`ort` crate) | Fallback | Simplest for CPU-only operators: one binary. No cross-request batching across replicas, and less operational parity with the image stack. Keep as a possible later `transport` option. |

**The mapping onto the `Backend` trait** (pre/post-processing in Rust, forward pass in Triton):
- `preprocess`: normalize (NFKC, invisible and bidi characters, decoding of smuggled text), tokenize with the
  HF `tokenizers` crate from the model's `tokenizer.json`, and cut into overlapping windows sized to the
  model's limit (512 or 8k).
- Triton: `input_ids` and `attention_mask` (INT64, dynamic `[-1,-1]`) in, `logits [-1,2]` out, with dynamic
  batching enabled.
- `postprocess`: softmax and label mapping (`INJECTION`, `MALICIOUS`, `LABEL_1` differ per family), take the
  max across windows, keep the span of the window that fired, and apply the calibrated threshold presets.

**Text-specific gotchas:**
- **The tokenizer is part of the model.** The exporter must write `tokenizer.json` (and the label map and
  threshold presets) into the model's metadata/directory. The pipeline fingerprint must hash it.
- **The Horizon obfuscation normalizer** ships as tokenizer-side code. Check whether it survives
  `tokenizer.json`. If it doesn't, reimplement it in the gateway's shared normalization step, which every
  backend benefits from anyway.
- **Long windows are expensive.** 8k-token windows are slow on CPU, because attention cost grows with
  sequence length. Make window size an operator setting (e.g. 1024 or 2048 even for 8k models). Bucket by
  length before batching so padding stays low.
- **Batch limits:** cap total windows per request (`limits.max_windows`) so a single huge document can't
  monopolize Triton.

---

## Sources
- Re-evaluating PI detectors for agents: https://arxiv.org/abs/2610.03448
- The Attacker Moves Second: https://arxiv.org/abs/2510.09023
- Prompt Overflow: https://arxiv.org/abs/2605.23196
- PIArena (ACL 2026): https://arxiv.org/abs/2604.08499 · https://github.com/sleeepeer/PIArena
- LlamaFirewall / Prompt Guard 2 AgentDojo numbers: https://arxiv.org/abs/2505.03574
- InjecGuard / PIGuard: https://arxiv.org/abs/2410.22770 · https://huggingface.co/leolee99/PIGuard
- Llama Prompt Guard 2: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
- Horizon-Labs guards: https://huggingface.co/Horizon-Labs/prompt-injection-guard-base · https://huggingface.co/Horizon-Labs/prompt-injection-guard-large
- Qualifire Sentinel: https://arxiv.org/abs/2506.05446 · https://huggingface.co/qualifire/prompt-injection-sentinel
- StackOne Defender: https://www.stackone.com/platform/prompt-injection-guard/
- Lakera Guard API: https://docs.lakera.ai/docs/api/guard · https://docs.lakera.ai/docs/api/results.md
- Azure Prompt Shields: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak
- AWS Bedrock prompt attack filter: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html
- Google Model Armor: https://docs.cloud.google.com/model-armor/sanitize-prompts-responses
