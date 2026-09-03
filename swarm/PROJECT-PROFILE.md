# PROJECT PROFILE — striatica (DRAFT — NOT AUTHORITATIVE, WORKFLOW NOT APPROVED)

> ⛔ **STATUS (Laura ruling 2026-08-01): this document and the entire `docs/swarm/`
> workflow are STAGED FOR ITERATION ONLY. The swarm workflow is NOT approved for
> striatica and may not be used — read-only or otherwise — until Laura explicitly
> approves it. Nothing here is authoritative: where this file restates CLAUDE.md or the
> v5 plan, those sources govern.** The pre-filled content below exists so that IF the
> workflow is ever approved, no one has to author a profile from scratch — it is a
> convenience draft, not a config in force.
>
> Pre-filled 2026-08-01 from CLAUDE.md, the v5 master plan (`img/correction/v5_consolidated/`),
> ERRATA.md, and the 2026-08-01 repo audit. 🧑 sections are drafts of Laura's already-stated
> doctrine, not new policy.

Leave a field as `UNKNOWN` rather than guessing. Use `N/A` when a field genuinely does not apply.
Reviewers skip `N/A` and probe `UNKNOWN`.

---

## 1. Identity

- **Project name:** striatica
- **Repo root:** repo root (this file lives at `swarm/PROJECT-PROFILE.md` — tracked and public as of
  2026-09-03; it was previously under the gitignored `docs/**`)
- **Default branch:** `main`
- **Working branch for this swarm:** `<set per run>` 🧑
- **HEAD SHA under review:** `<run git rev-parse HEAD at dispatch>` 🧑

## 2. Stack

- **Language / runtime:** Python 3.12 (pipeline); TypeScript + React + React Three Fiber (frontend)
- **Package manager:** Poetry (pipeline — **never pip in the host env, never `poetry lock/update/export`**, see footguns); pnpm (frontend). Docker images install pinned deps with pip **by design** — that is not a violation, it is the reproducibility mechanism.
- **Backend framework:** N/A (no server; CLI pipeline + static-file viewer)
- **Frontend framework:** React + React Three Fiber + Vite; custom GLSL shaders
- **Database / ORM:** N/A
- **Test runner:** pytest — `poetry run pytest tests/ -m "not slow"`; frontend E2E: Playwright via pnpm (24 tests); audit lane: `audit/test_audit_euclidicity.py` in its own venv (`requirements-audit.txt`)
- **Typecheck command:** frontend `pnpm tsc --noEmit` (verify script name in `frontend/package.json` before claiming); Python: none configured (N/A)
- **Build command:** `docker build -t striatica .` (CPU — **the canonical test path**); `docker build -f Dockerfile.gpu -t striatica-gpu .`; frontend `pnpm build`
- **CI system + config path:** **no workflow file in-tree** (`.github/` holds only issue templates), but CI *does* run on pull requests: CodeQL default setup (`Analyze (python)`, `Analyze (javascript-typescript)`) configured in repo settings, plus the GitGuardian Security Checks app. Both observed green on PR #12, 2026-09-03. There is no build/test/lint CI — that absence is known and tracked, and is not a finding. **Do not conclude "no CI" from the absence of `.github/workflows/`.**

**Known footguns:**
- **Never run `poetry lock` anywhere, ever** — UMAP output changes across dependency versions even with `random_state=42`; a lockfile regeneration silently moves every 3D position (the 2026-03-21 incident). `poetry export` is also currently broken; the pinned Docker images and `requirements-audit.txt` are the workaround, by design.
- The Docker images are `striatica` / `striatica-gpu`; the host console script is `striat`. `docker run striat …` always fails (`pull access denied`).
- **L0 variant identity:** same width ≠ same dictionary (`average_l0_6` vs `average_l0_604` are different feature sets; index 2082 means different features in each). The identity validator hard-fails contradictions — never bypass, never suggest `--allow-l0-mismatch` for a *contradicted* case.
- Feature indices are **local per layer** (encoding `layer*100000 + local`; locals 0–16383 at every layer). Strict layer filtering is load-bearing.
- The UMAP "n_jobs overridden to 1" warning on every run is known noise, not a bug.
- `striat demo` is host-only and currently generates retracted-methodology circuits (R0 fixes this — in scope).
- No hardcoded rendering values: every visual parameter lives in `frontend/src/config/rendering.ts`.
- Sidecar JSONs (`*-metadata.json`, `*-validation.json`) are not datasets; the Vite dev dropdown currently mis-lists them (known, R0).

## 3. Trust boundary — WHO MAY DO WHAT 🧑 (draft from CLAUDE.md + v5 §6; confirm)

- **Agents may:** read everything; run tests and read-only pipeline commands; run Docker builds; commit to their **own branch**; prepare diffs and MR/PR descriptions.
- **Agents may NEVER:** push to `main` or any remote (all pushes are Laura's, explicitly); force-push; `reset --hard`; `clean -fd`; `branch -D`; delete remote branches; regenerate or modify anything under `frontend/public/data/` (production data — Laura-gated, always); run `poetry lock/update/export`; commit, bake, or log semantic labels for non-public-tier models.
- **Who merges:** Laura. Always.
- **Shared-git hazard:** **YES** — one checkout, parallel agents share `.git`. All preamble git rules apply at full strength.
- **Commit identity to use:** repo default.

## 4. Data safety 🧑 (draft; confirm)

- **Which environment does the default config point at?** No database, no remote writes. The hazard here is **data files, not databases**: `frontend/public/data/*.json` + sidecars ARE production data — the published atlases. Treat any write there as a production write.
- **Safe-to-write:** agent's own branch; `img/correction/**` (plans, reports — gitignored); `data/` cache (re-fetchable raw downloads); scratch dirs.
- **Writes requiring explicit human approval:** anything under `frontend/public/data/`; any dataset regeneration (Laura runs + visually verifies — tests are necessary but not sufficient); any `.gitignore`, Dockerfile, or lockfile change.
- **Read-only operations always fine:** everything else, including keyless Neuronpedia/HF GETs (cache them).

## 5. Secrets 🧑 (confirm)

- **Secret manager:** `.env` at repo root (python-dotenv). Single secret: `NEURONPEDIA_API_KEY`. (Under the Doppler threshold; no Doppler here.)
- **Invocation pattern:** loaded via `load_dotenv()`; the key is used ONLY for authenticated POSTs (e.g. `/api/graph/generate`) — never attached to keyless GETs, per `pipeline/graph_fetch.py`.
- **Absolute rule:** never print, `cat`, echo, log, or commit a secret value. Names and presence-checks only. No exceptions.
- **Key blast radius (confirmed 2026-08-01):** the Neuronpedia API key is NOT read-only — the API exposes authenticated mutating endpoints (e.g. `POST /api/explanation/{id}/delete`). Agents may use the key ONLY for `POST /api/graph/generate`, with per-run approval. Every other authenticated endpoint — anything that deletes, edits, votes, or saves on the platform — is **forbidden**; escalate to Laura instead. (Also listed in §6.)

## 6. Kill list — OUT OF SCOPE, DO NOT TOUCH 🧑 (draft from v5 doctrine; confirm)

- **TDA / stratified-geometry / persistence / Mapper code in `pipeline/`** — gated until the Phase-0 euclidicity memo exists AND the v5 plan has 2× external validation. `audit/` is the only permitted home for that work.
- **Any inbound-security hardening** — auth, TLS, CORS/CSP, rate limiting, or any exposure mitigation. striatica is localhost-only, fixed threat model (v5 §6.1). In scope for security review: validation of fetched remote content, cache-path safety, API-key hygiene, label handling. Nothing else.
- **Jaccard co-activation pipeline** — no work on it except its R0 quarantine/removal.
- **Mock data**, anywhere, in any form.
- **Coverage-percentage raising** and mutation-tool tuning (mutmut *pilot* arrives at R1, not before).
- **New rendering magic numbers** (config/rendering.ts or nothing).
- **Gemma 3 work** before R0/R1 acceptance criteria are all closed (v5 R1.5 deferral).
- **Semantics/dual-use prose additions** to docs, UI, or plans (v5 §6.6 — the commitment mechanism is the posture).
- **Regenerating the poetry lockfile** (also listed in §2 because it keeps happening).
- **Any authenticated Neuronpedia endpoint other than `POST /api/graph/generate`** — the API has live delete/edit/vote surfaces (see §5 key blast radius); mutating a public scientific platform is never an agent's call.

**Explicit exceptions:** the R0 Jaccard quarantine IS in scope; the R1 mutmut pilot on the traced-circuits lane IS in scope when R1 opens; `swarm/**` and `img/correction/**` edits are normal ops.

## 7. Scope of this review cycle 🧑 (draft; confirm/update per cycle)

- **What ships this cycle:** v5 **Phase R0 — Integrity** (see `img/correction/v5_consolidated/v5-master-plan.md` §1 and `01-readme-release-coder-plan.md`): version unification, README v2, CLAUDE.md repair + doctrine, Jaccard quarantine, small-defect sweep, release plumbing → tag v0.4.0.
- **End state:** a cloner sees no retracted methodology anywhere, one pipeline version everywhere, and GitHub/Docker Hub tags that match.
- **Severity scale:** `P0 / P1 / P2 / P3` (highest → lowest). Every reviewer uses exactly this.
- **Severity routing:** P0/P1 → Laura immediately (via gatekeeper). P2/P3 → logged in the report, deferred to the gatekeeper's queue.
- **Accepted-risk policy (do NOT report these as defects):**
  - Trustworthiness 0.8104 < 0.85 scorecard bar **rides by design** — it is the measured cost of flattening ~dim-20 structure into 3D, i.e. the v5 thesis expressed as a number. L2 is a scorecard; only L1 hard-gates.
  - `poetry export` broken → worked around by pinned Docker + `requirements-audit.txt`; repair is P1-era housekeeping on Laura's machine only.
  - O(n²) trustworthiness memory (OOM at width_65k) is known and tracked; it gates 65k runs only.
  - The absence of **build/test/lint** CI is a known state, not a finding. Security-scan CI does
    exist and runs on every PR (see § 2) — a *failure* there is a real finding, and this bullet is
    not cover for ignoring one.
  - `docs/**` and `img/**` being gitignored is deliberate (private plans); flag only if a *public-facing* asset (e.g. README image) depends on an ignored path.

## 8. Where reports go

- **Report directory:** `img/correction/agent_reports/<yyyymmdd>-<topic>/` (gitignored — correct for internal findings)
- **One file per role.** Never chat scrollback.
- **Naming:** `<role-name>.md`, three-line SHA header per `_SHARED-PREAMBLE.md`.

## 9. Project governance docs a reviewer must read first

Authority order on conflict (highest → lowest). Flag conflicts; never resolve them.

1. `CLAUDE.md` (repo root) — standing rules. *(Known defect until R0: its first ~50 lines describe a different repo; skip to the striatica content.)*
2. `img/correction/v5_consolidated/v5-master-plan.md` — current plan, phases, doctrine, delta log
3. `img/correction/v5_consolidated/01-readme-release-coder-plan.md` — active R0 workstreams
4. `ERRATA.md` — the retraction record
5. `.claude/skills/neuronpedia-endpoints/SKILL.md` — API discipline (re-enumeration rule)
