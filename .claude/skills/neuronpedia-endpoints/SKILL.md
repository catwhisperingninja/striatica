---
name: "neuronpedia-endpoints"
description: "Neuronpedia + interp-engine API discipline for striatica — re-enumerate the live API surface on every use, then consult the dated snapshot (graph/source-set/explanation endpoints, L0 identity validation, field gotchas, rate limits, write-surface guardrail) and the daily/weekly delta-check cadence. Use whenever fetching Neuronpedia data, validating dictionary identity, working on traced circuits, or checking what's new on Neuronpedia / interp-engine.org."
---

# Neuronpedia endpoints — striatica working discipline

**v4 · 2026-09-08** (changelog at end)

**Prime rule: the API surface is constantly expanding. Nothing in this file is truth —
it is a dated snapshot. Step 0 is always re-enumeration.** A skill that hardcodes
endpoints rots; this one hardcodes the *procedure* and dates its facts.

**Scope boundary.** This skill owns the *procedure, surfaces, and dated snapshot*. The
stateful daily watch — the carried arXiv triage queue, the Slack destination, what was
already reported — lives in the `neuronpedia-whats-new` **scheduled task**, and only
there. Do not copy the carried list into this file; two copies of a queue that gets
pruned is a guaranteed contradiction.

## Search path — surfaces and cadence

| # | Surface | Cadence |
|---|---|---|
| 1 | `neuronpedia.org` api-doc (spec), model pages, source-set endpoint, homepage nav NEW/UPDATE tags | daily |
| 2 | **`neuronpedia.org/blog`** — post index; compare newest post date against the baseline table below | **weekly** |
| 3 | **`interp-engine.org`** + `neuronpedia.org/blog/interp-engine` | **daily** |
| 4 | PyPI: `neuronpedia`, `interp-engine` | daily (cheap) |
| 5 | arXiv `cs.LG` / `cs.AI` / **`cs.CL`** — SAE / transcoder / feature-geometry / intrinsic-dimension lane | daily |

**Why the blog URL is in here:** the newsletter is the announcement channel, and relying
on a forwarded email means a missed forward is a missed release. `/blog` is the
authoritative index and is fetchable without a subscription. Check the index, not the inbox.

**arXiv query hygiene:** `all:"transcoder"` pulls networking/codec false positives
(e.g. 2607.28100 PCAP-LM, where "transcoded" means format conversion). Filter on primary
category in {cs.LG, cs.AI, cs.CL} — and note cs.CR occasionally carries real
transcoder-deployment work, so don't hard-exclude it.

### Blog post-date baseline (as of 2026-09-03)
Newest = **interp-engine, 2026-08-31**. Prior: J-Space/Jacobian Lens 2026-07-10 ·
HeadVis 2026-06-09 · NLA 2026-05-17 · Circuit Tracing w/ Attention + Interp Explorer
2026-04-21 · Assistant Axis + Gemma Scope 2 + SAELens + NNsight 2026-01-28 ·
Llama 3.3 70B / gpt-oss-20b 2025-11-12. A post newer than the top entry = delta; update
this row when you log it.

## Step 0 — Re-enumerate the live surface (every session that touches Neuronpedia)

1. **Snapshot the OpenAPI spec.** There is no standalone spec URL — Neuronpedia generates
   the spec server-side (next-swagger-doc over `app/api`) and embeds it inline in the
   api-doc HTML as the body of `<script id="api-reference" type="application/json">`
   (verified 2026-08-01: openapi 3.0.0, "Neuronpedia API" v1.0, **50 paths**;
   re-verified live 2026-08-27 and 2026-09-03: 50 paths; **2026-09-06: 51 paths** —
   `POST /api/model/lookup` added, sole net change vs the 2026-09-03 inventory; nothing the
   inventory below listed was removed). Extract it:

   ```bash
   curl -s https://www.neuronpedia.org/api-doc | python3 -c "import sys,re,json; \
   m=re.search(r'id=\"api-reference\"[^>]*>(.*?)</script>', sys.stdin.read(), re.S); \
   print(json.dumps(json.loads(m.group(1)), indent=2))" \
   > <striatica-repo>/data/api-snapshots/neuronpedia-openapi.json
   ```

   **The snapshot goes in the striatica repo, never beside this skill.** The skill
   directory is a read-only cache that gets re-synced — a file written there is silently
   discarded, and it is not under git, so `git diff` on it is meaningless. The whole
   delta check depends on the snapshot being versioned in the repo.

   (**Fetch capability is runtime-dependent — dated 2026-09-08.** Watch/scheduled runtime: direct
   HTTPS works (the 2026-09-06 watch run fetched the spec via curl, no browser); the v3 blanket
   "sandbox cannot curl" was over-broad — it is environment-specific. Cowork interactive sandbox
   (this session): `WebFetch` is **robots-disallowed** for `neuronpedia.org` and `pypi.org`, and
   scripting around a WebFetch block is out of policy — use the **in-app browser** (keyless GETs
   render fine; source-set + api-doc + graph record were read this way to verify the Gemma-3
   inventory this session) or `WebSearch` for package versions. Blog renders via `get_page_text`;
   for arXiv open `export.arxiv.org/api/query` in its own tab (the XML does not parse via
   querySelector). The in-app browser is the reliable cross-runtime path.)

   Diff the **extracted JSON**, never the raw page (Scalar UI markup changes independently
   of the spec). Fallback if extraction breaks: the route tree in the open-source repo
   (`github.com/hijohnnylin/neuronpedia`, `apps/webapp/app/api/**` — `@swagger` JSDoc
   annotations are the spec source).
   Path inventory at snapshot (tag level): `feature`, `activation`, `explanation/*`
   (search/generate/score), `graph/*` (generate, list, tokenize, **subgraph/**),
   `steer` / `steer-chat` / `steer-logits`, `search-all`, `search-topk-by-token`,
   `lens/*`, `nla/*`, `sparsity/connected-neurons`, `list/*`, `vector/*`, `bookmark/*`,
   `model/lookup` (NEW 2026-09-06), `model/new`.
   **Not in the spec, but load-bearing:** `/api/source-set/{model}/{name}` — the identity
   endpoint (§2 below) — is NOT enumerated by the OpenAPI and never has been. "51 paths" is the
   documented surface, not the full *consumed* surface. Never prune anything for "not in the spec."
2. Diff what you find against the **Snapshot** section below (endpoints, params, rate
   limits, nav NEW/UPDATE tags).
3. Any drift on a surface striatica consumes → re-probe with the fixture fetch
   (`tests/fixtures/`), shape-diff, and update BOTH this snapshot (with a new
   as-of date) and the striatica plan delta log.
4. New endpoints/tools that don't map to current work still get one line in the delta
   log. The periphery IS the ask: enumerate everything first, filter by relevance second.
   A plan-delta review is not a platform review.

Use the first-party clients in `pipeline/graph_fetch.py` — never a third-party wrapper
(a community MCP server exists, `manncodes/neuronpedia-mcp`; exploration only — it
bypasses our identity validation).

---

## Snapshot — spec as of 2026-09-06 · inventory re-verified in part 2026-09-08 (re-verify per Step 0 before relying on it)

Nav tags at snapshot: Jacobian Lens NEW · NLA NEW · Assistant Axis NEW ·
**Circuit Tracer UPDATE** (post-7/28 — ⚠️ OPEN GATE: fixture re-probe required before the
next production traced-circuit run).

### 1. Graph record (keyless)
```
GET https://www.neuronpedia.org/api/graph/{model}/{slug}
```
Small JSON record: **`sourceSetName` is here**, plus `url` → full graph JSON on S3
(`neuronpedia-attrib.s3.us-east-1.amazonaws.com`). Two-step fetch: record, then S3
payload; both keyless. The spec exposes `/api/graph/list`; treat listing as
account-scoped — the CLI is slug-driven by design.

### Model lookup — HuggingFace repo → Neuronpedia model id (keyless · NEW 2026-09-06)
```
POST https://www.neuronpedia.org/api/model/lookup     { hfRepoId: "namespace/name" }
```
Keyless, **non-mutating**. NP model ids are slash-free and are NOT the HF repo id with the
namespace stripped (`openai-community/gpt2` → `gpt2-small`, not `gpt2`); this resolves one to the
other, one repo → at most one model. Returns id, layers, neuronsPerLayer, dimension,
instruct/thinking/inferenceEnabled. Watch run cites landing 2026-09-05 (commit `df03cfee` /
PR #234 — unverified here). Useful **upstream** of identity validation (repo → model id), but it
is model-level: it does NOT return `saelensSaeId`/L0. `source-set` (below) stays the only source
of dictionary identity. A `model/lookup` hit is not dictionary validation.

### 2. Source set — identity validation (keyless)
```
GET https://www.neuronpedia.org/api/source-set/{model}/{name}
```
Per-layer sources with `saelensSaeId` (**contains the L0 variant**) and `hfRepoId`.
Ground truth, re-confirmed live 2026-09-03: deployed gemma-2-2b layer-12 transcoder =
`google/gemma-scope-2b-pt-transcoders/layer_12/width_16k/average_l0_6`; the set
`gemma-2-2b/gemmascope-transcoder-16k` carries 26 sources (layers 0–25), all
`inferenceEnabled: true`, `graphEnabled: true`.
**L0 mismatch = different dictionary = the same feature index means different features.**
The validator hard-fails contradictions; `--allow-l0-mismatch` rescues only
*unverifiable*, never *contradicted*. Never weaken this.

Useful set-level flags on this endpoint: `graphEnabled`, `hasGraphs`,
`allowInferenceSearch`; per-source: `inferenceEnabled`. **These are independent** — a set
can be graph-enabled while every source is inference-disabled (see Gemma-3 below).

### Gemma-3 / Gemma Scope 2 inventory (base rows re-verified 2026-09-08; -it / 27b rows as of 2026-09-03)

Release: "Gemma Scope 2: Suite of SAEs and Transcoders for Gemma 3" (Google DeepMind,
Dec 2025; surfaced on the Neuronpedia blog 2026-01-28). Source-set naming is
`gemmascope-2-{res|transcoder}-{16k|262k}`; transcoder SAE ids use the `l0_small_affine`
variant, residual sets use `l0_medium`.

| model | set | type | layers | graphEnabled | any source inference-enabled |
|---|---|---|---|---|---|
| gemma-3-4b-it | gemmascope-2-transcoder-262k | Transcoder 262k | 34 (0–33) | **true** | no |
| gemma-3-4b-it | gemmascope-2-transcoder-16k | Transcoder 16k | 10 | false | no |
| gemma-3-4b-it | gemmascope-2-res-16k / -262k | Residual | 4 | false | no |
| gemma-3-27b-it | gemmascope-2-transcoder-262k | Transcoder 262k | 62 | false | no |
| gemma-3-27b-it | gemmascope-2-res-16k / -262k | Residual | 5 | false | no |
| gemma-3-4b (base) | gemmascope-2-res-16k | Residual 16k | 4 | false | **yes** ✓09-08 |
| gemma-3-4b (base) | gemmascope-2-res-262k | Residual 262k | 4 | false | **yes** ✓09-08 |
| gemma-3-12b (base) | gemmascope-2-res-16k | Residual 16k | 4 | false | **yes** ✓09-08 |
| gemma-3-12b (base) | gemmascope-2-res-262k | Residual 262k | 4 | false | no ✓09-08 |
| gemma-3-27b (base) | gemmascope-2-res-16k / -262k | Residual | 4 | false | no (v3) |

**Verification:** `✓09-08` rows were live-checked via `source-set` this session (2026-09-08:
base-4b res-16k layers 9/17/22/29 and res-262k same four, all `inferenceEnabled: true`; base-12b
res-16k layers 12/24/31/41 all `true`, res-262k same four all `false`). Unmarked rows carry
forward from the 2026-09-03 inventory, not re-verified this session.

Standing rule: transcoders exist only on `-it` variants; the only inference-enabled
gemma-3 dictionaries are **residual-only, on the base models** — base 4b (both 16k **and** 262k)
and base 12b (**16k only**; 12b's 262k residual is inference-disabled), live-verified 2026-09-08.
(v3 said "base 4b only" — corrected; the watch flagged 12b, this session verified it and found the
16k/262k asymmetry. Base 27b residual stays "no" per 2026-09-03, not re-verified.) Record gemma-3
transcoder status as "transcoder present, graphs enabled, inference not enabled" until a
transcoder source flips `inferenceEnabled: true`. **That flip is the R1.5 evidence gate** —
lead with it the day it happens.

### ⛔ Write-surface guardrail (v4: re-keyed on MUTATION, not HTTP method — 2026-09-08)

The API is NOT read-only, and **HTTP method is not the safety axis.** Neuronpedia serves many
non-mutating reads over POST (`activation/get`, `activation/source`, `graph/tokenize`,
`model/lookup`, `search-topk-by-token`, `vector/get`, `lens/prompt`, the `explanation/search-*`
family), and its one graph-writing endpoint (`graph/generate`) is itself a keyless POST. "Keyless
GET = safe" is false in both directions. **The axis is whether the call persists or publishes
anything to the platform.** Hard allowlist for agents:

- **Allowed (auto):** anything that only reads or computes-and-returns, persisting and publishing
  nothing — any method, key or no key. Covers everything striatica consumes: `source-set` GET,
  graph record GET + S3 payloads, `feature` GET, `activation/*`, `model/lookup`, `search-*`,
  `explanation/search-*`, `graph/tokenize`, `sparsity/connected-neurons`, `nla/sources` +
  `nla/explain` + `nla/completion`, `vector/get`, `list/get` + `list/list`, `subgraph/list`,
  `lens/prompt`, blog, exports.
- **Allowed with explicit human approval, per run:** `POST /api/graph/generate` — it publishes a
  persistent public graph, so it is a write despite being keyless. **Never in an unattended or
  scheduled run** — "approval per run" presupposes a human in the loop, and a cron job has none.
  Scheduled watches are read-only.
- **Forbidden, always, no exceptions:** anything that creates, edits, deletes, saves, votes, or
  **publishes** persistent platform state or public content — every `*/delete`, `*/save`,
  `*/new`, `list/add-features` · `edit-feature` · `remove` · `update`, `vector/new`, `bookmark/*`,
  `model/new`, `subgraph/save`, **and `lens/share`** (it publishes a permanent public snapshot — a
  write, not a read). If a task appears to need one, stop and escalate to Laura. Altering or
  publishing on a public scientific platform is an outward action; it is never an agent's call.
- **Steering** (`steer`, `steer-chat`, `steer-logits`) is transient — it intervenes at inference
  and returns output without persisting. The one genuinely ambiguous case. **Default: keep it
  gated as write-surface** (v3 posture); do not auto-allow it pending an explicit Laura ruling.

### 3. Graph generation — MUTATING (publishes a public graph; key + per-run approval)
```
POST https://www.neuronpedia.org/api/graph/generate
{ prompt (≤64 tokens), modelId (REQUIRED — spec: "only gemma-2-2b supported"),
  slug, sourceSetName (optional → model default),
  maxNLogits, desiredLogitProb, nodeThreshold, edgeThreshold, maxFeatureNodes,
  qkTopFraction, qkTopk (Lorsa models only) }
```
`required: [prompt, modelId, slug]` per the 2026-09-06 spec — **`modelId` was missing from the
v3 summary and is required.** Slug must be unique ("Model + Slug/ID Exists").
⚠️ The 2026-09-06 spec marks this endpoint **keyless** (no `security` block → inherits global
`security: []`). Do NOT act on that — it is almost certainly a missing `@swagger` annotation, and
graph/generate **persists a public graph to S3+DB**, a write regardless of auth. Keep sending
`NEURONPEDIA_API_KEY`, keep the per-run human-approval gate, never log the key. The approval
requirement rests on **mutation, not on whether a key is demanded** (see guardrail).

### 4. Explanations / features (bulk, keyless)
Per-source S3 JSONL batches (pipeline Step 1; e.g.
`gemma-2-2b/12-gemmascope-transcoder-16k`, 64 batches; ~46% explanation coverage is
normal). Full dataset exports: `neuronpedia-datasets.s3.us-east-1.amazonaws.com`
(`?prefix=v1/`). HeadVis attention-head metrics export under
`[model]/headvis/[dataset_used]` (pile-uncopyrighted for all HeadVis data).

### Field gotchas (confirmed on real graphs)
- Feature encoding `layer*100000 + local`; locals are 0–16383 **at every layer** —
  strict layer filtering is what prevents silent atlas collisions. Cross-layer members
  go in metadata only, never as `featureIndex`. (Note: 262k-width sets break the 16384
  local bound — recheck the encoding before touching Gemma Scope 2 262k dictionaries.)
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

### Rate limits (⚠️ STALE as of 2026-09-03 — treat as unmeasured)
Last measured: `/api/nla/completion` 240/hr · `/api/lens/prompt` 120/hr ·
`/api/steer` 120/hr. The interp-engine launch post states the throughput gain
"allows us to increase rate limits on Neuronpedia, host larger models, and increase max
number of tokens in each application" — that is **announced capability, not a published
new number**. Do not substitute a guess; re-measure before relying on any figure. Graph
record/S3 reads keyless — still cache (`data/graphs/{model}/`, gitignored).

### interp-engine (daily surface)
Announced on the Neuronpedia blog **2026-08-31** (author Johnny Lin; post is marked
human-written). PyPI `interp-engine` **1.6.0** — confirmed present on PyPI 2026-09-08 (version
page live; exact upload time not captured this session; watch run cited 1.6.0 on 2026-09-06).
Prior: 1.5.1, 2026-09-01T22:28 UTC (1.3.3 → 1.5.1). Cite 2026-08-31 as the announcement date.
Hosted-model roster signal (2026-09-06 spec): lens/nla enums now include `deepseek-v4-flash`
(lens `numCompletionTokens` cap 2048 vs 1024 for all other models) and `qwen3.6-27b`; NLA pairs
are `gemma-3-27b-it`/`kitft-l41`, `llama3.3-70b-it`/`kitft-l53`, `qwen2.5-1.5b-it`/`andyxu-l18`.
Roster growth is itself the rate-limit / hosted-model re-measure trigger.
Sub-surfaces to check: `interp-engine.org` · `/docs` · GitHub repo ·
`results-latest.md` (full cross-model benchmarks) · `interp-engine/validator` ·
GPU-sizer (also exposed via API; supports private models via HF-token override) ·
architecture visualizer + Gemma-2-vs-Gemma-3 style "Compare" view · "Ask Riz" docs bot.

Claims, stated precisely (the headline and the measurement are different numbers):
- **~40x** throughput vs HF transformers — headline, for *large* models, e.g.
  Deepseek-V4-Flash.
- **~7x** faster — the actual worked benchmark, Jacobian Lens on Deepseek V4 Flash, same
  outputs. Cite 7x when citing a measurement; 40x is the marketing figure.
- vLLM by default; stock vLLM exposes only the residual stream, interp-engine adds
  read/write across **34 standardized hook points**, GPT-2 → Gemma 4, incl.
  multi-residual-stream.
- Validated against other engines + HF Transformers on **50+ models**; the process found
  and fixed bugs in *other* engines.

**No arXiv preprint exists for interp-engine** (checked 2026-09-03) — it is tool-only.
Two consequences: a competitor can adopt it with no citation obligation, and if
Neuronpedia later publishes one carrying the benchmark corpus, that is a "new data"
event worth flagging.

Striatica relevance: it is an *activation/generation* engine, not a static dictionary
loader — it does not replace Gemma Scope transcoder loading, but it is the fastest path
to reproducible activation capture and the thing a competitor would use to move quickly.
Treat a minor-version bump as a trigger to re-check rate limits and the hosted-model roster.

### Platform tools shipped 2026 (so reviews stop missing them)
Assistant Axis (Jan 28) · Circuit Tracing w/ Attention + Interp Explorer (Apr 21) ·
NLA/activation verbalizers (May 17, Anthropic collab) · HeadVis (Jun 9 — 36,000+ attention
head dashboards across 37 models, Anthropic collab; Head Finder by induction / prev-token /
attention-entropy / self-attention score) · Jacobian Lens (Jul 10) · interp-engine (Aug 31).
Jun newsletter also shipped 14 new SAEs (Gemma 4 31B/E2B/E4B, Olmo 3 7B/32B, Qwen 3
1.7B/8B/14B/32B, Qwen 3.5 0.8B/2B/4B/9B/27B) — **all residual-stream, zero new
transcoders**, plus cross-model NLA explanations (Zaffino) and Gemma 4 E2B NLAs (DeLeeuw).
Triage rule: striatica maps anything exposing a per-feature vector dictionary
(W_dec-equivalent). NLA = verbalizer, no static dictionary (complementary); J-Lens =
operators (not a loader); Assistant Axis = one direction (plottable overlay);
HeadVis = attention heads, no feature dictionary (complementary);
interp-engine = runtime, no dictionary (complementary, but competitively enabling).

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

## Delta check (log in the current striatica v6 plan delta log)

**Daily:** nav NEW/UPDATE tags · interp-engine.org + PyPI versions · arXiv lane.
**Weekly (adds):** `/blog` index vs the baseline table · full source-set/dictionary
inventory · api-doc spec re-extract and diff.

1. Blog index: any post newer than the baseline row above?
2. Homepage nav NEW/UPDATE tags: changed vs the last log row?
3. UPDATE touching a consumed API (currently: graph API) → immediate fixture re-probe
   before the next production run.
4. PyPI: `neuronpedia` (last known **1.2.1**, 2026-08-08 — a newer release trips the R1
   Circuit-Tracer re-probe gate) and `interp-engine` (**1.6.0**, PyPI-confirmed 2026-09-08;
   watch cited it 2026-09-06 — a newer release trips a rate-limit / hosted-model re-measure).
5. Gemma-3 gate: re-run the source-set flags table above; the gate opens only when a
   `gemmascope-2-transcoder-*` source reports `inferenceEnabled: true`.
6. Log the result either way. Real delta that changes phases/gates/ledger rows bumps
   the plan version; a no-delta check just logs.

---

## Changelog

- **v4 — 2026-09-08.** Independent review→implement cycle — reviewer/implementer separate from
  the `neuronpedia-whats-new` watch agent that proposed the changes (external `CHANGELOG.md`
  carries the audit trail). api-doc **50→51 paths**: added `model/lookup` (keyless, non-mutating,
  HF repo → NP model id) to the inventory + a subsection. **Re-keyed the write-surface guardrail on
  mutation, not HTTP method** — "keyless GET" was both too narrow (misses POST reads striatica
  needs) and too loose (never caught `lens/share` as a publish); `model/lookup` is now an
  auto-allowed read, `graph/generate` stays gated because it *persists*, steering stays gated as
  transient-but-write-surface pending a ruling. Fixed the `graph/generate` summary (required
  `modelId`; added `nodeThreshold`/`edgeThreshold`/`qkTopFraction`/`qkTopk`) and flagged its
  spec-"keyless" marking as a likely annotation gap — keep key + approval. Recorded that
  `source-set` is not in the spec, so "51 paths" ≠ full consumed surface. Corrected the Gemma-3
  inference-enabled rule (base 4b: 16k **and** 262k; base 12b: 16k **only**), live-verified via
  `source-set` 2026-09-08. Bumped interp-engine 1.5.1 → 1.6.0 (PyPI-confirmed) and noted the
  deepseek-v4-flash / qwen3.6-27b roster signal. Reframed the "can't curl in Cowork" note as
  runtime-dependent-and-dated instead of deleting it. **Routed to the watch task, not this skill
  (scope boundary):** the "read the prior watch note before reporting delta" step.
- **v3 — 2026-09-03.** Added `/blog` (weekly) and `/blog/interp-engine` (daily) as
  explicit surfaces so a missed newsletter forward can't cause a missed release; added
  blog post-date baseline table. Corrected interp-engine announcement to 2026-08-31
  (blog) vs 2026-09-01 (PyPI). Split the 40x headline from the 7x measured benchmark.
  Softened the rate-limit claim to "announced, not published". Added cs.CL to the arXiv
  lane + false-positive filter note. Moved the OpenAPI snapshot out of the read-only
  skill directory into the striatica repo. Closed the unattended-run hole in the
  write-surface guardrail. Recorded that interp-engine has no arXiv preprint. Declared
  the scope boundary against the `neuronpedia-whats-new` scheduled task. Added this
  changelog and an in-band version line.
- **v2 — 2026-09-03.** Manual edit: added interp-engine as a search-path surface and
  snapshot section; refreshed Gemma-3 / Gemma Scope 2 inventory table.
- **v1 — 2026-08-01.** Initial: Step 0 re-enumeration procedure, graph/source-set
  endpoints, L0 identity validation, field gotchas, write-surface guardrail, labels
  ruling, claims-ledger lockstep.

