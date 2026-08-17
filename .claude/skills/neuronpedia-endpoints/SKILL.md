---
name: neuronpedia-endpoints
description: Neuronpedia API discipline for striatica — re-enumerate the live API surface on every use, then consult the dated snapshot (graph/source-set/explanation endpoints, L0 identity validation, field gotchas, rate limits) and the weekly delta-check procedure. Use whenever fetching Neuronpedia data, validating dictionary identity, working on traced circuits, or checking what's new on Neuronpedia.
---

# Neuronpedia endpoints — striatica working discipline

**Prime rule: the API surface is constantly expanding. Nothing in this file is truth —
it is a dated snapshot. Step 0 is always re-enumeration.** A skill that hardcodes
endpoints rots; this one hardcodes the *procedure* and dates its facts.

## Step 0 — Re-enumerate the live surface (every session that touches Neuronpedia)

1. **Snapshot the OpenAPI spec.** There is no standalone spec URL — Neuronpedia generates
   the spec server-side (next-swagger-doc over `app/api`) and embeds it inline in the
   api-doc HTML as the body of `<script id="api-reference" type="application/json">`
   (verified 2026-08-01: openapi 3.0.0, "Neuronpedia API" v1.0, **50 paths**). Extract it:

   ```bash
   curl -s https://www.neuronpedia.org/api-doc | python3 -c "import sys,re,json; \
   m=re.search(r'id=\"api-reference\"[^>]*>(.*?)</script>', sys.stdin.read(), re.S); \
   print(json.dumps(json.loads(m.group(1)), indent=2))" \
   > .claude/skills/neuronpedia-endpoints/openapi-snapshot.json
   ```

   Diff the **extracted JSON**, never the raw page (Scalar UI markup changes independently
   of the spec). Commit the refreshed snapshot next to this skill; `git diff` on it IS the
   API delta check. Fallback if extraction breaks: the route tree in the open-source repo
   (`github.com/hijohnnylin/neuronpedia`, `apps/webapp/app/api/**` — `@swagger` JSDoc
   annotations are the spec source).
   Path inventory at snapshot (tag level): `feature`, `activation`, `explanation/*`
   (search/generate/score), `graph/*` (generate, list, tokenize, **subgraph/**),
   `steer` / `steer-chat` / `steer-logits`, `search-all`, `search-topk-by-token`,
   `lens/*`, `nla/*`, `sparsity/connected-neurons`, `list/*`, `vector/*`, `bookmark/*`,
   `model/new`.
2. Diff what you find against the **Snapshot** section below (endpoints, params, rate
   limits, nav NEW/UPDATE tags).
3. Any drift on a surface striatica consumes → re-probe with the fixture fetch
   (`tests/fixtures/`), shape-diff, and update BOTH this snapshot (with a new
   as-of date) and the delta log in `img/correction/v5_consolidated/v5-master-plan.md` §7.
4. New endpoints/tools that don't map to current work still get one line in the delta
   log. The periphery IS the ask: enumerate everything first, filter by relevance second.
   A plan-delta review is not a platform review.

Use the first-party clients in `pipeline/graph_fetch.py` — never a third-party wrapper
(a community MCP server exists, `manncodes/neuronpedia-mcp`; exploration only — it
bypasses our identity validation).

---

## Snapshot — as of 2026-08-01 (live-verified; re-verify per Step 0 before relying on it)

Nav tags at snapshot: Jacobian Lens NEW · NLA NEW · Assistant Axis NEW ·
**Circuit Tracer UPDATE** (post-7/28 — fixture re-probe required before the next
production traced-circuit run).

### 1. Graph record (keyless)
```
GET https://www.neuronpedia.org/api/graph/{model}/{slug}
```
Small JSON record: **`sourceSetName` is here**, plus `url` → full graph JSON on S3
(`neuronpedia-attrib.s3.us-east-1.amazonaws.com`). Two-step fetch: record, then S3
payload; both keyless. No public graph-list endpoint (only auth'd
`/api/graph/list-owned`) — the CLI is slug-driven by design.

### 2. Source set — identity validation (keyless)
```
GET https://www.neuronpedia.org/api/source-set/{model}/{name}
```
Per-layer sources with `saelensSaeId` (**contains the L0 variant**) and `hfRepoId`.
Ground truth at snapshot: deployed gemma-2-2b layer-12 transcoder =
`google/gemma-scope-2b-pt-transcoders/layer_12/width_16k/average_l0_6`.
**L0 mismatch = different dictionary = the same feature index means different features.**
The validator hard-fails contradictions; `--allow-l0-mismatch` rescues only
*unverifiable*, never *contradicted*. Never weaken this.

### ⛔ Write-surface guardrail (confirmed 2026-08-01)

The API is NOT read-only. The api-doc enumerates authenticated **mutating endpoints** —
e.g. `POST /api/explanation/{explanationId}/delete` — that act on live platform data
under the account behind `NEURONPEDIA_API_KEY`. Hard allowlist for agents:

- **Allowed:** all keyless GETs (graph record, S3 payloads, source-set, exports).
- **Allowed with explicit approval per run:** `POST /api/graph/generate`.
- **Forbidden, always, no exceptions:** every other authenticated endpoint — anything
  that deletes, edits, votes, saves, steers, or otherwise mutates platform state. If a
  task appears to need one, stop and escalate to Laura. Deleting or altering data on a
  public scientific platform is an outward, destructive action; it is never an agent's
  call.

### 3. Graph generation (API key)
```
POST https://www.neuronpedia.org/api/graph/generate     header: x-api-key
{ prompt (≤64 tokens), slug, sourceSetName (optional → model default),
  maxNLogits, desiredLogitProb, maxFeatureNodes }
```
Slug must be unique ("Model + Slug/ID Exists"). Key = `NEURONPEDIA_API_KEY` in `.env`;
never attach it to keyless GETs, never log it.

### 4. Explanations / features (bulk, keyless)
Per-source S3 JSONL batches (pipeline Step 1; e.g.
`gemma-2-2b/12-gemmascope-transcoder-16k`, 64 batches; ~46% explanation coverage is
normal). Full dataset exports: `neuronpedia-datasets.s3.us-east-1.amazonaws.com`
(`?prefix=v1/`).

### Field gotchas (confirmed on real graphs)
- Feature encoding `layer*100000 + local`; locals are 0–16383 **at every layer** —
  strict layer filtering is what prevents silent atlas collisions. Cross-layer members
  go in metadata only, never as `featureIndex`.
- `layer` is a **string**, can be non-numeric (`"E"` = embedding).
- `feature_type` seen: `cross layer transcoder`, `mlp reconstruction error`,
  `embedding`, `logit`.
- Link weights **signed**, unnormalized (seen −41.8…+68.9) — normalize before use.
- Graphs arrive **pre-pruned**; `metadata.pruning_settings` records `node_threshold` /
  `edge_threshold`; no raw scores. Robustness = |weight| margin above the recorded edge
  threshold. A local "threshold sweep" is a no-op — don't propose it.
- `metadata.transcoder_list` is empty in real graphs — identity comes from the record +
  source-set endpoint, not the graph body.
- **Same-layer edges cannot exist** (attribution is a strictly-forward DAG);
  single-layer cross-sections ship `edges: []` with layerFilter accounting. Do NOT
  invent same-layer edge semantics — that was the Jaccard mistake.
- Supernodes span layers; a layer-12 node gets a role label only if it is actually a
  supernode member; otherwise `unassigned`.

### Rate limits (at snapshot)
`/api/nla/completion` 240/hr · `/api/lens/prompt` 120/hr · `/api/steer` 120/hr.
Graph record/S3 reads keyless — still cache (`data/graphs/{model}/`, gitignored).

### Platform tools shipped 2026 (so reviews stop missing them)
Assistant Axis (Jan) · Circuit Tracing w/ Attention + Interp Explorer (Apr) ·
NLA/activation verbalizers (May, Anthropic collab) · HeadVis (Jun) · Jacobian Lens (Jul).
Triage rule: striatica maps anything exposing a per-feature vector dictionary
(W_dec-equivalent). NLA = verbalizer, no static dictionary (complementary); J-Lens =
operators (not a loader); Assistant Axis = one direction (plottable overlay).

---

## Labels — one rule, no essays

Handling is provenance-tiered and already decided (Laura ruling 2026-08-01):
public-tier labels — Gemma-2-2B foremost — are **affirmatively displayed and published**
in the atlas; showing them is the proof of 1:1 alignment with Neuronpedia's deployed
dictionary. Everything beyond public-tier goes out only as **salted SHA-256 commitments**
(`pipeline/commitment.py` — digest now, reveal to vetted reviewers later). Do not add
semantics/dual-use prose beyond this anywhere; the commitment mechanism IS the posture.

## Claims-ledger lockstep

Any Neuronpedia-derived change that touches a load-bearing claim (identity, circuits,
labels, metrics) updates `paper-v2-claims-ledger.md` **in the same commit-set** — the
ledger tracks the code so paper v2 can publish simultaneously with the v1.0.0 release,
not after a reconciliation scramble.

## Weekly delta check (~10 min; log in v5-master-plan.md §7)

1. Blog: any post newer than the last logged entry?
2. Homepage nav NEW/UPDATE tags: changed vs the last log row?
3. UPDATE touching a consumed API (currently: graph API) → immediate fixture re-probe
   before the next production run.
4. Log the result either way. Real delta that changes phases/gates/ledger rows bumps
   the plan version; a no-delta check just logs.
