# API compatibility

**These are compatibility layers, not this service's API.** `POST /v1/detect`
is the API this project specifies and versions; see
[`openapi.json`](openapi.json). The routes below reshape the same scores into
a commercial vendor's envelope so an existing client can swap its base URL
and key, and they inherit whatever that envelope cannot express. In
particular, **both vendors return booleans**: these layers apply
`scoring.threshold` and drop the scores.

This document is normative for the layers. It records what we reproduce,
what we approximate, and what we deliberately omit. The mappings were
derived from the vendors' published documentation (links below), checked on
2026-10-08.

---

## Lakera Guard v2

**Upstream:** [`POST https://api.lakera.ai/v2/guard`](https://docs.lakera.ai/docs/api/guard)
(Lakera is now part of Check Point.)

### Request

| Field | Upstream | Here |
|---|---|---|
| `Authorization: Bearer <key>` | Lakera API key | **the deployment key** (set it as the client's Lakera key) |
| `messages[]` | OpenAI-style `{role, content}` | same; `content` may be a string or content parts (`[{"type":"text","text":…}]`, non-text parts ignored) |
| `project_id` | selects a policy | **accepted, echoed in `breakdown`, no policy effect** |
| `breakdown` | per-detector results | supported |
| `payload` | PII / profanity match locations | **accepted and ignored** (no such detectors) |
| `tools`, `metadata`, `dev_info` | various | **accepted and ignored** |

**Which messages are screened:** `user`, `tool` and `function` messages.
`system` messages are the developer's own instructions, and screening them
only produces false positives. `assistant` turns are model output.

### Response

| Field | Fidelity |
|---|---|
| `flagged` | `true` if any screened message scores ≥ `scoring.threshold` |
| `metadata.request_uuid` | a fresh UUIDv4 |
| `breakdown[]` | one entry per **screened** message: `project_id` (echoed), `policy_id` (`language-forensics-default`), `detector_id` (`language-forensics-<model>`), `detector_type` (`prompt_attack`), `detected`, `message_id` (index into `messages`) |

**Not verified:** the exact `detector_type` vocabulary and the id formats
Lakera uses in `breakdown`. `prompt_attack` is our reading of the name of
Lakera's prompt-defense detector type; confirm it against the live docs
before relying on it. Clients that match on Lakera's own `policy_id` /
`detector_id` strings will not match ours.

**Not reproduced:** Detect vs. Enforce modes (we always return the decision;
`flagged` is never forced to `false`); content moderation, PII, unknown-links
and custom detectors; `/v2/guard/results` (L1 to L5 confidence levels); the
project, policy and dashboard APIs.

---

## Azure AI Content Safety — Prompt Shields

**Upstream:** [`POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01`](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak)

### Request

| Field | Upstream | Here |
|---|---|---|
| `Ocp-Apim-Subscription-Key` | Azure resource key | **the deployment key** (also accepted as Bearer / `X-API-Key`) |
| `api-version` query | required | **accepted and ignored** (any value, or absent) |
| `userPrompt` | string, required | same |
| `documents` | array of strings | same; optional here (defaults to `[]`) |

Azure documents its own input limits (prompt length, document count). We do
not reproduce them; this deployment's `limits.*` apply instead.

### Response

```json
{ "userPromptAnalysis": { "attackDetected": false },
  "documentsAnalysis": [ { "attackDetected": true } ] }
```

| Field | Fidelity |
|---|---|
| `userPromptAnalysis.attackDetected` | `userPrompt` score ≥ `scoring.threshold` |
| `documentsAnalysis[i].attackDetected` | `documents[i]` score ≥ `scoring.threshold`, in order |

Azure trains separate classifiers for user-prompt attacks and document
attacks. Here **one model scores both**; the distinction is only which field
the result lands in.

**Errors** use `{"error": {"code", "message"}}`, the same envelope Azure
uses, but our `code` values (`bad_request`, `unauthorized`, `too_large`,
`unavailable`, …) are not Azure's (`InvalidRequestBody`, …).

**Not reproduced:** every other Content Safety API (text/image moderation,
groundedness, protected material, custom categories); blocklists.

---

## Why these two, and not AWS or Google

AWS Bedrock Guardrails (`ApplyGuardrail`, `PROMPT_ATTACK` filter) and Google
Model Armor (`sanitizeUserPrompt`) both reference server-side resources
(a guardrail id and version, or a template) and cloud-specific auth (SigV4,
GCP IAM). A drop-in clone would have to fake those resources. Lakera and
Azure take plain text plus a key, so a base-URL swap is enough. See
[`research/prompt-injection-detection.md`](research/prompt-injection-detection.md).
