# Striatica v6.2.2 Consolidated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Status:** REVIEW DRAFT · UNRATIFIED · NO IMPLEMENTATION AUTHORIZED  
**Prepared:** 2026-09-01  
**Goal:** Repair the scientific, provenance, reproducibility, audit, claims,
Neuronpedia, documentation, and release-control defects identified during the
v6.2 review, then produce one independently reviewed checkpoint candidate
without regenerating production data or changing shared state by implication.

**Architecture:** The detailed requirements, gates, node contracts, file scopes,
and acceptance criteria live in `6.2.2-implementation-control.md`. This document
is the cross-lane execution plan: it fixes the decision order, identifies safe
parallelism, and prevents any lane from promoting another lane's evidence.
Implementation proceeds from a small control bootstrap, through truth
establishment and principal method gates, into bounded production/audit/source
work, and only then into claims, public documentation, checkpoint, and release.

**Tech Stack:** Python/Poetry, Docker CPU and GPU images, isolated `audit/` venv,
React/TypeScript frontend, YAML/JSON control records, pytest, frontend
build/lint/browser checks, GitHub Actions, Neuronpedia HTTP/OpenAPI sources.

**Spec:** `img/correction/v6_plans/6.2.2-implementation-control.md`  
**Scientific review:** `img/correction/v6_plans/6.2.1-amp-adv-review.md`  
**Review inputs:**
`v6.2-plan-0831-deepseekavdrev-scievidlane-pt1.md` and
`v6.2-plan-0831-deepseekavdrev-engrexecutionlane-pt2.md`

## Authority and basis

1. Laura's current explicit decision outranks every plan, review, or memory
   mirror.
2. `CLAUDE.md` safety/privacy doctrine applies throughout.
3. The recovered DAG specification is byte-identical to the orb and Basic
   Memory copy at SHA-256
   `4830997ba2a8f35c17cfd4e7249879f23db66039adfcb6aef6a94510146c91e8`.
4. Its inspected repository basis is
   `13ff88b48aeeee0cc9c4d7dc45a4feba224a3dd3`; the local repository was at
   `081f34e9c0abdb82144ba72d04334eaef1a53493` when this consolidation was
   written. `K-BASIS-01` must record and reconcile the actual execution basis;
   this plan does not silently treat the older review SHA as current.
5. Basic Memory and Slack are redundant indexes, not execution authorities.

## Global constraints

- No node is dispatchable before Laura ratifies the exact DAG specification
  hash through `G-RATIFY`.
- Semantic labels must never be committed, pushed, baked into images, logged,
  copied into evidence, or persisted to a version-controllable file.
- No production file under `frontend/public/data/` may be regenerated without
  the separate, exact `G-DATA` authorization. User visual acceptance requires
  the later and separate `G-VISUAL` gate.
- No synthetic application dataset, placeholder circuit, or hardcoded feature
  index may replace real data. Synthetic known-geometry controls are allowed
  only for method validation inside `audit/`.
- Dockerfiles and CI consume the committed `poetry.lock`; they never run
  `poetry lock`. CPU Docker is canonical; GPU support is separately evidenced.
- The Euclidicity/TDA audit remains isolated from the main Poetry environment.
- Code inspection, static review, one execution, two fresh reproductions, and
  scientific validation remain different evidence types.
- Static review never counts as an execution. Two invocations in one mutable
  environment never count as two fresh reproductions.
- A ticket changes only its permitted files. Additional files require a
  controlled amendment before editing.
- No push, PR mutation, merge, tag, release, publication, branch deletion, or
  other shared-state mutation occurs without exact, current approval.
- Private plan changes are recorded in `6.2-private-changelog.md`. Future public
  shipped changes belong in root `CHANGELOG.md`; the two histories are not
  interchangeable.

---

## Decisions incorporated from review

### Scientific and evidence corrections

- PR, exact TWO-NN, density, VGT curves, VGT slopes, and Euclidicity are
  separate estimands with separate inputs, assumptions, provenance, and claims.
- Current `_vgt_single` values such as `231.9` and the paper's `0.14–21` are
  slopes pending immutable forensics, not validated intrinsic dimensions.
- The defective-VGT-versus-PR referee contest is dissolved, but PR is not
  appointed universal scientific referee. It is currently a decoder-weight
  local effective-rank baseline.
- Geometric Wall supports TWO-NN on activation manifolds, not PR on decoder
  weights. Bhalla supports the SAE fragmentation arm, not a transcoder result.
  arXiv `2509.02565` is empirical adjacent work, not theory-only background.
- The current convergence-singularity interpretation is contradicted/inverted
  by the best supplied measurement and stays on hold until immutable
  remeasurement. No causal explanation follows from radius spread or the
  activation-frequency correlations.
- The historical claim that the VGT scalar would have crashed on 2026-03-02 is
  false: the relevant bound landed on 2026-03-24.
- Deterministic repetition establishes determinism only, not correctness or
  robustness.

### DeepSeek engineering corrections, adjudicated rather than copied

- Accepted: full `prepare_json` tests, curve-content validation, upper/lower
  method-bound tests, SAE dispatch coverage, documentation correction, isolated
  audit-environment evidence, and frontend/CI coverage.
- Corrected: public JSON `dimMethod` and internal Python `dim_method` are valid
  layer-specific casing, not a landed mismatch. The mismatch appeared in
  proposed PR #8 prose.
- Corrected: no in-tree production caller currently proves live metric data
  loss; inventory precedes metric rename or retirement.
- Rejected: permanent numerical assertions that methods must differ. Tests pin
  method identity and gate-approved known-geometry behavior instead.
- Rejected: matching eight handwritten direct pins proves full dependency
  closure. Current Dockerfiles still do not consume the committed lock.
- Rejected: the claim that open Dependabot PRs/branches existed on the reviewed
  date.

### Briefer-agent suggestions retained

- Use a four-field mutable state projection: `state`, `sha`, `evidence`, and
  `note`. Git and reviewed artifacts retain history; duplicate event, finding,
  gate, and per-evidence ledgers are not introduced.
- Keep a narrow structural checker and required CI. A repository-wide
  banned-vocabulary pre-commit grep is too weak and too noisy; scoped retired-
  claim checks may supplement structural enforcement.
- Public-safe neutral status/gap maps are legitimate. PR #8 remains unsuitable
  because that specific tracked file contains private strategy/findings prose.
- Preserve Neuronpedia per-use enumeration, Monday weekly delta logging,
  immediate re-probe of changed consumed APIs, and the monthly full review.
  Daily checks are optional, not a replacement cadence.
- The repository Neuronpedia skill is canonical by location but was not more
  expanded than the reviewed copies. It already uses `curl`; browser use is a
  fallback. A path count of 50 cannot establish zero drift without a preserved
  baseline.

---

## Program map

```diagram
G-RATIFY
    │
    ▼
Lane K: control bootstrap
    │
    ├───────────────┬────────────────┬────────────────┐
    ▼               ▼                ▼                ▼
Lane B0         Lane N           Lane C0          basis/checks
methods         source truth     artifacts
    │               │                │
    ▼               │                ▼
G-VGT / G-EUC       │            claims ledger
    │               │                │
    ├───────┐       │                ├──────▶ ERRATA / citations / paper
    ▼       ▼       ▼                │
Lane A   Lane B1  NP review           └──────▶ README / changelogs / CI
pipeline audit
    │       │                           │
    └───┬───┴───────────────┬───────────┘
        ▼                   ▼
 integration reviews   G-DATA/G-VISUAL if needed
        │
        ▼
 G-CHECKPOINT ─────────▶ final release review ─────────▶ G-RELEASE
```

The authoritative node graph contains 52 nodes, 95 dependency edges, and 73
requirements. The DAG specification owns exact edge and acceptance semantics.

---

## Phase 0: principal review and ratification

### Task 0.1: Review the planning package

**Files:**
- Review: `striatica-v6.2.2-consolidated-plan.md`
- Review: `6.2.2-implementation-control.md`
- Context: `6.2.1-amp-adv-review.md`
- Context: both DeepSeek adversarial-review files
- Review: `6.2-private-changelog.md`

- [ ] Confirm that the consolidated plan includes every project lane and does
      not collapse scientific choices into engineering defaults.
- [ ] Confirm the nine principal gates remain independent: `G-RATIFY`,
      `G-VGT`, `G-COMP`, `G-EUC`, `G-DATA`, `G-VISUAL`, `G-CHECKPOINT`,
      `G-PAPER`, and `G-RELEASE`.
- [ ] Confirm checkpoint means an annotated tag after acceptance, not a parked
      branch.
- [ ] Confirm that ratification authorizes bounded local implementation only,
      not data regeneration or shared-state actions.
- [ ] If accepted, record `G-RATIFY` against the exact post-review DAG hash. If
      amended, update the private changelog, rerun structural validation, and
      review the new hash before recording the gate.

**Exit:** Exact DAG hash ratified or a finite amendment list returned. No code
work begins from an unratified draft.

---

## Phase 1: Lane K — control bootstrap

### Task 1.1: Materialize the machine-readable control contract

**Nodes:** `K-SCHEMA-01`  
**Produces:** `control/dag.yaml`, `control/state.yaml`, and schemas under
`control/schemas/` as specified by the ratified DAG.

- [ ] Write schema-negative tests before schemas or examples.
- [ ] Materialize all 52 nodes, all gates, all requirements, acceptance IDs,
      file scopes, and dependency edges without weakening conjunctions.
- [ ] Keep mutable state to exactly `state`, `sha`, `evidence`, and `note`.
- [ ] Validate malformed enums/references fail; graph is acyclic; every
      requirement and acceptance ID resolves.
- [ ] Independently review the exact result before any dependent node starts.

### Task 1.2: Establish basis, integration, and structural enforcement

**Nodes:** `K-BASIS-01`, `K-INTEGRATE-01`, `K-CHECK-01`  
**Dependency:** accepted `K-SCHEMA-01`.

- [ ] Implement the deterministic, atomic basis manifest and prove missing
      inputs, unsafe output paths, dirty-state handling, and hash changes fail or
      invalidate as specified.
- [ ] Define worktree/branch ownership, non-overlap, integrator, exact-candidate,
      and conflict-resolution rules; distinguish local `main` from
      `origin/main` and orb workspaces.
- [ ] Implement the narrow checker for schema, one-to-one state rows,
      refs/cycles, parent/gate states, SHA forms, and required evidence pointers.
- [ ] Review and accept each node independently; do not bundle bootstrap with
      product or scientific work.

**Exit:** A machine-enforced control substrate exists at an exact reviewed
basis. This establishes process integrity, not scientific correctness.

---

## Phase 2: truth-establishment wave

These nodes may run in parallel only after Lane K permits dispatch and their
ticket file sets do not overlap.

### Task 2.1: Lane B0 — define estimands and reproduce evidence

**Nodes:** `B-SPEC-01`, `B-VGT-FORENSICS-01`, `B-EUC-REFERENCE-01`

- [ ] Define each method's estimand, input space, units, algorithm/version,
      output, assumptions, supported claims, contrary evidence, and falsifier.
- [ ] Reproduce current VGT slope behavior on known-dimensional, degenerate,
      density, clump, outlier, and sensitivity controls.
- [ ] Remeasure `231.9`, `0.14–21`, convergence, radius-spread, and correlation
      claims against immutable artifacts and producer hashes.
- [ ] Reproduce the reference Euclidicity equations and published controls;
      enumerate and test every local deviation rather than inheriting its name.
- [ ] Obtain independent scientific/source review of each exact artifact.

### Task 2.2: Lane C0 — inventory immutable claim-bearing artifacts

**Node:** `C-ARTIFACTS-01`

- [ ] Hash the exact paper, figures, captions, datasets, sidecars, scripts,
      configs, and producer code used by every affected claim.
- [ ] Record missing or non-reconstructable provenance as contrary evidence,
      not as an implementation inconvenience.
- [ ] Exclude semantic payloads from tracked or public evidence.

### Task 2.3: Lane N — establish current Neuronpedia source truth

**Node:** `N-INVENTORY-01`

- [ ] Re-enumerate the live API and preserve the first reviewable baseline
      before replacement.
- [ ] Re-probe the layer-12 `average_l0_6` source-set identity chain.
- [ ] Inventory blog and navigation NEW/UPDATE surfaces before relevance
      filtering.
- [ ] Record graph listing/generation fields and feature-node versus total-node
      counts without authenticated mutation.

**Exit:** Laura receives decision packages for `G-VGT` and `G-EUC`; comparator
design can proceed toward `G-COMP`. No method choice is made by an implementer.

---

## Phase 3: principal method gates

### Task 3.1: Decide VGT disposition

**Gate:** `G-VGT`  
**Choices:** repair as a calibrated dimension estimator; retire as dimension;
or retain only approved descriptive curves.

- [ ] Review `B-VGT-FORENSICS-01` evidence, including contrary results and
      downstream schema/claim consequences.
- [ ] Record one exact disposition and its allowed terminology/outputs.
- [ ] Dispatch `B-VGT-APPLY-01` to implement only that disposition.

### Task 3.2: Decide Euclidicity disposition

**Gate:** `G-EUC`

- [ ] Review reference reproduction and every documented deviation.
- [ ] Select a reference-faithful/justified implementation or halt Euclidicity
      claim use.
- [ ] If retained, run `B-EUC-IMPLEMENT-01` then held-out
      `B-EUC-ROBUSTNESS-01`; never treat repeated identical arrays as robustness.

### Task 3.3: Decide comparator/concordance disposition

**Nodes/gate:** `B-COMP-SPEC-01` → `G-COMP`

- [ ] Require explicit method/version, input space, identity/order/value hashes,
      threshold rationale, artifact producer, and permitted claim.
- [ ] Choose the accepted comparator contract or remove the concordance gate.
- [ ] Do not preserve a scientifically invalid gate merely because renaming it
      is easier than deleting it.

---

## Phase 4: Lane A — production pipeline and reproducibility

### Task 4.1: Metric/provenance contract and producer validation

**Nodes:** `A-SCHEMA-01`, `A-CURVES-01`, `A-VALIDATE-01`, `A-FRONTEND-01`,
`A-OUTPUT-01`

- [ ] Keep canonical camelCase `dimMethod` at the public JSON boundary and
      explicit versioned scalar/curve provenance; internal Python may use
      `dim_method`.
- [ ] Test the full `prepare_json` writer and all ingestion consumers, including
      missing, partial, unknown, stale, mismatched, and non-finite provenance.
- [ ] Add method-specific lower, upper, accepted-boundary, and interior tests;
      no universal slope bound is invented before `G-VGT`.
- [ ] Wire curves into production validation and reject missing keys,
      misalignment, unequal axis lengths, non-finite values, unordered radii,
      insufficient dynamic range, degeneracy, and method/version mismatch.
- [ ] Wire or explicitly disposition the activation-frequency L2 path so claims
      cite checks that actually execute.
- [ ] Reject bad provenance before frontend rendering; remove default-to-PR and
      misleading VGT/local-dimension labels while preserving existing UI
      behavior.
- [ ] Route all tests and smokes to disposable output outside
      `frontend/public/data/`.

### Task 4.2: Canonical CPU reproducibility

**Nodes:** `A-DOCKER-CPU-01`, `A-DOCKER-CPU-02`

- [ ] Make the CPU image consume the committed lock without lock regeneration
      or a parallel dependency graph.
- [ ] Record image digest, installed distribution report, architecture, lock
      hash, commands, and output hashes.
- [ ] Execute two independently provisioned fresh CPU environments on one basis;
      compare with appropriate exactness/tolerances rather than unsupported
      cross-hardware bit-identity claims.

### Task 4.3: Flagship l0=6 and SAE resume paths

**Nodes:** `A-L06-STATIC-01`, `A-L06-RUN-01`, `A-SAE-FIXTURE-01`,
`A-SAE-RUN-01`

- [ ] Migrate the flagship default and identity chain to deployed layer-12,
      width-16k, `average_l0_6` without semantic-label persistence.
- [ ] Run the real flagship CPU smoke twice into disposable paths and record
      identity, dimensions, schema, sidecars, redaction, and output hashes.
- [ ] Add the fast SAE `cmd_model` resolution/dispatch test plus a real-derived,
      semantics-free fixture.
- [ ] Run two fresh SAE resume smokes through the shared contracts; remove stale
      expected-failure setup guidance only when the accepted fixture replaces
      the skip deferral.

### Task 4.4: Metrics, GPU, and production integration review

**Nodes:** `A-METRIC-01`, `A-DOCKER-GPU-01`, `A-INTEGRATION-REVIEW-01`

- [ ] Inventory actual in-tree and retained external metrics consumers before
      choosing atomic rename, retirement, or time-bounded compatibility.
- [ ] Repair and evidence the GPU image separately only for claims the release
      intends to make; never substitute it for canonical CPU acceptance.
- [ ] Integrate only independently accepted commits, run combined checks at one
      exact candidate, and reopen owning nodes for failures.

**Exit:** One exact production candidate is independently reviewed. This does
not authorize production data regeneration, visual acceptance, or a checkpoint.

---

## Phase 5: Lane B1 — audit comparator, artifacts, environment, and execution

### Task 5.1: Build or remove the comparator path

**Nodes:** `B-COMP-ARTIFACT-01`, `B-COMP-CONSUMER-01`

- [ ] Produce the accepted semantics-free comparator sidecar with identity,
      feature-order, method/version, and value hashes—or remove the gate and all
      dependent claims under the approved disposition.
- [ ] Fail closed on NaN/Inf, constant input, too few pairs, low all-feature
      coverage, count/order/identity mismatch, stale schema, and wrong method.

### Task 5.2: Make audit artifacts atomic and the environment isolated

**Nodes:** `B-AUDIT-ARTIFACT-01`, `B-AUDIT-ENV-01`

- [ ] Restrict run IDs to safe basenames with containment assertions.
- [ ] Write into a temporary run directory, hash all outputs, write completion
      last, and atomically rename; prove failed runs cannot leave a stale
      valid-looking memo.
- [ ] Hash-pin the audit environment and prove both import directions remain
      isolated from main Poetry.
- [ ] Reproduce the audit environment twice from scratch.

### Task 5.3: Execute and independently review claim-grade audit

**Nodes:** `B-AUDIT-RUN-01`, `B-AUDIT-REVIEW-01`

- [ ] Execute the accepted method/comparator protocol twice on one exact real
      input and basis.
- [ ] Require independent method review and exact-code/artifact review.
- [ ] Treat null, disagreement, or failed calibration as scientific outcomes,
      not reasons to weaken a gate.

---

## Phase 6: complete Lane N — Neuronpedia source truth

### Task 6.1: Correct the skill, source records, and platform delta

**Nodes:** `N-SKILL-01`, `N-SOURCE-01`, `N-PLATFORM-01`, `N-REVIEW-01`

- [ ] Make the repo skill the current complete canonical version; preserve
      `curl` enumeration, add the missing OpenAPI baseline, correct stale graph
      fields/count wording, and repair the private v6 delta-log locator.
- [ ] Recompute graph source and l0=6 identity evidence under the accepted
      source contract.
- [ ] Enumerate the full platform periphery before relevance filtering,
      including current NEW/UPDATE surfaces.
- [ ] Independently review exact source snapshots and identity conclusions.

No authenticated generation or other mutation is implied by this lane.

---

## Phase 7: Lane C — claims ledger, ERRATA, citations, and paper v2

### Task 7.1: Normalize the private claims ledger and public correction

**Nodes:** `C-LEDGER-01`, `C-ERRATA-01`

- [ ] Create one live row per stable claim ID with separate paper disposition,
      evidence maturity/class, method/population, immutable artifacts, citation
      entailment, contrary evidence, uncertainty/falsifier, public surfaces,
      owner decision, and supersession.
- [ ] Remove amendment-over-stale-base behavior; preserve history through
      supersession.
- [ ] Add a newer dated ERRATA entry above the circuit erratum. Preserve the
      circuit correction while separating producer independence from scientific
      validation and placing dimension/convergence claims in their current
      evidence states.
- [ ] Update `CLAUDE.md` doctrine and the README errata callout only in the later
      principal-reviewed public change.

### Task 7.2: Verify citations and adjudicate every paper surface

**Nodes:** `C-CITATIONS-01`, `C-PAPER-DISPOSITION-01` → `G-PAPER`

- [ ] Attach exact source version, page/section/quote, supported proposition,
      and not-supported proposition to every load-bearing citation.
- [ ] Mark each affected paper sentence, figure, table, and caption as retain,
      rewrite, remove, or pending, tied to exact artifacts and claim IDs.
- [ ] Obtain Laura's approval of the exact disposition package through
      `G-PAPER` before any private paper rewrite.

### Task 7.3: Rewrite and cross-check paper v2

**Nodes:** `C-PAPER-REWRITE-01`, `C-CROSS-SURFACE-01`

- [ ] Rewrite only approved dispositions from the normalized ledger.
- [ ] Cross-check paper, ledger, ERRATA, README, changelogs, public manifest,
      code terminology, figures, and immutable artifacts.
- [ ] Reopen owning nodes for contradictions; never patch around them in prose.

---

## Phase 8: Lane D — README, changelogs, CI, checkpoint, and release

### Task 8.1: Inventory and rebuild the public truth surface

**Nodes:** `D-README-INVENTORY-01`, `D-README-02`

- [ ] Execute/classify every README command and inventory every claim, status
      phrase, citation, screenshot, and asset against the exact candidate.
- [ ] Produce a concise audience-first README: identity/paper version, current
      correction status, canonical CPU quickstart, implemented capabilities,
      method/limits, data-semantic policy, releases, and contribution guidance.
- [ ] Remove blanket unaffected wording, method conflation, unexecuted command
      claims, unsupported l0=6/traced/audit status, and broken/unapproved hero
      assets.
- [ ] Synchronize only the necessary factual corrections in `CONTRIBUTING.md`.

### Task 8.2: Maintain private and public changelogs and projection

**Nodes:** `D-CHANGELOG-PRIVATE-01`, `D-CHANGELOG-PUBLIC-01`,
`D-PUBLIC-MANIFEST-01`

- [ ] Keep this private planning/correction history current in
      `6.2-private-changelog.md` with document hashes, drivers, adjudications,
      reviewer state, and supersession.
- [ ] Create root `CHANGELOG.md` for shipped product/publication history only,
      with `Unreleased` and Keep-a-Changelog categories including scientific
      corrections.
- [ ] Project only neutral public-safe claim/evidence IDs, hashes, changelog
      linkage, and exact candidate into tracked manifests.

### Task 8.3: Enforce public structure and document release procedure

**Nodes:** `D-CI-01`, `D-RELEASE-PROCEDURE-01`

- [ ] Run the structural/public checker, Python tests, frontend build/lint and
      required browser tests, lock policy, changelog/manifest linkage, and
      privacy checks in required GitHub Actions.
- [ ] Keep private ignored checks local and content-safe; CI must not require or
      reconstruct `img/**`.
- [ ] Write a check-first `RELEASING.md` with exact candidate/image/artifact
      hashes, commands, data/visual conditions, shared-action prompts, and
      rollback. Validation must not tag, push, publish, or regenerate.

### Task 8.4: Prepare checkpoint and release candidates

**Nodes/gates:** `D-CHECKPOINT-01` → `G-CHECKPOINT`;
`D-RELEASE-REVIEW-01` → `G-RELEASE`

- [ ] Prepare the immutable checkpoint manifest only after production
      integration acceptance. Laura separately authorizes the annotated tag;
      no parked branch is maintained.
- [ ] If any production-visible data changes are proposed, obtain one-command
      `G-DATA`, regenerate only the named files, then obtain user visual review
      through `G-VISUAL`.
- [ ] Review one exact release candidate across source, tests, images, audit,
      claims, docs, changelogs, public manifest, data, and visuals.
- [ ] List each intended push/tag/release/publication action explicitly and
      obtain `G-RELEASE`; approval of the plan or checkpoint does not authorize
      those actions.

---

## Verification strategy

For each node:

1. Acceptance tests and negative cases are written in its generated ticket
   before implementation.
2. The implementer runs the narrow failing test, implements the minimum accepted
   behavior, reruns narrow checks, then the node's broader required checks.
3. An independent reviewer inspects the exact SHA/hash and evidence pointers.
4. A node reaches acceptance only under the lifecycle/evidence rules in the DAG.
5. Dependency changes invalidate affected descendants; integrated checks rerun
   at the exact candidate.

Program-level verification requires:

- the machine DAG/state checker;
- targeted positive and negative Python tests;
- full non-slow Python suite where required;
- frontend build, lint, and required browser checks;
- canonical lock-consuming CPU build and two fresh executions;
- separate GPU checks only for GPU claims;
- two fresh isolated audit environments and claim-grade audit executions;
- independent method, artifact, source, and integration reviews;
- cross-surface claims/documentation consistency;
- user visual acceptance for any regenerated visible artifacts.

Green tests alone do not establish scientific validity or visual correctness.

## Explicitly unresolved principal decisions

These are gates, not missing plan content:

1. `G-RATIFY` — approve/amend the exact DAG specification.
2. `G-VGT` — fix, retire-as-dimension, or descriptive-curves-only.
3. `G-COMP` — accepted comparator/artifact/threshold or concordance removal.
4. `G-EUC` — accepted Euclidicity implementation/deviations or halt claim use.
5. `G-PAPER` — exact paper text/figure dispositions.
6. `G-DATA` — one named production regeneration, if proposed.
7. `G-VISUAL` — acceptance of named regenerated visible artifacts.
8. `G-CHECKPOINT` — exact accepted candidate checkpoint tag.
9. `G-RELEASE` — exact shared-state actions for one release candidate.

## Completion condition

The v6.2.2 program is complete only when the DAG's chosen non-superseded nodes
are accepted at one exact candidate; method and paper decisions match their
evidence; CPU/audit reproduction requirements are met; claims, code, paper,
ERRATA, README, changelogs, manifest, data, and visuals agree; privacy and
semantic-label boundaries hold; and Laura authorizes each shared release action.

Planning completion is not implementation completion, test completion is not
scientific verification, and ratification is not release authorization.

## Document change history

- **2026-09-01 · v6.2.2 review draft created locally.** Consolidated the
  immutable v6.2 planning basis, Amp v6.2.1 adversarial review, adjudicated
  DeepSeek science and engineering findings, retained briefer-agent control and
  cadence suggestions, recovered the revised 52-node DAG from the paused orb,
  and enumerated every program lane and principal gate. No implementation or
  shared-state action authorized.
