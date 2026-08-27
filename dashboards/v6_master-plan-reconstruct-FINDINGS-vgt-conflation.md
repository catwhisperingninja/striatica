# FINDINGS — VGT / participation-ratio conflation in the Striatica audit lane

**Grounding:** static analysis of repomix snapshot `0827-state.1.xml` (136 files, repo
state 2026-08-27). Nothing was executed. Line numbers are as of this snapshot.
**Status:** PROPOSED. Written to be attacked — hand to Amp / Fabled / DeepSeek for
independent falsification before any code lands.
**Scope decision:** surgical repair of the `audit/` lane. The geometry pipeline is
**PINNED and untouched**. No revert.

---

## TL;DR

The production pipeline computes **participation ratio (PR)** and stores it — honestly —
as the JSON field `localDimensions`, stamped `dim_method="pr"`. The Euclidicity audit's
`load_vgt()` then reads that same `localDimensions` array and treats it as **VGT**,
feeding a hard-stop "VGT concordance" gate. Real *scalar* VGT is computed and thrown
away; only VGT *growth curves* are persisted. So every artifact that says "VGT dimension"
**downstream of the audit** is actually PR.

This is a **provenance / labeling defect, not corrupted geometry.** The numbers are right;
they are called the wrong name. The fix changes **zero numbers**.

---

## 1. Data flow (verified, file:line)

| Stage | Code | Space it runs in |
|---|---|---|
| PCA → UMAP → 3-D coords | `reduce.py::reduce_to_3d` (`cli.py:598`) | original vectors |
| HDBSCAN clustering | `cluster.py::cluster_points` (`cli.py:604`) | **3-D UMAP coords** |
| Local dimension = **PR** | `estimate_local_dim(vectors, method="pr")` (`cli.py:614`, `process_gpt2_small.py:63`) | original high-D vectors |
| VGT | `estimate_local_dim_vgt(vectors, return_curves=True)` → `_, growth_curves` (`cli.py:619`) | **scalar dims discarded**; curves kept |
| Write JSON | `prepare_json(..., local_dimensions=local_dims, dim_method="pr", growth_curves=...)` (`cli.py:656-657`) → `prepare.py:124` `result["localDimensions"] = local_dimensions.tolist()` | — |

So the dataset JSON's `localDimensions` **is participation ratio**, and the sibling field
`dim_method="pr"` records that truthfully at the source.

## 2. The defect (verified)

`audit/run_audit.py`:

- `load_vgt()` (`:124-131`): `vgt = np.asarray(data["localDimensions"], dtype=np.float64)`
  — reads the PR array; docstring calls them "per-feature **VGT** local-dimension scores."
  **It ignores `dim_method`.**
- `:241`: `vgt = load_vgt(args.dataset_json)`
- `vgt_concordance_gate(min_per_feature, vgt)`: Pearson r between `(1 − Euclidicity)` and
  the PR-array-called-VGT. **Hard stop** — `raise SystemExit(EXIT_VGT_DISCORDANT)` (code 3)
  if `r < VGT_GATE_R (0.3)`. Its own detail text asserts "(1 − Euclidicity) and VGT both
  measure non-manifoldness."

The gate that certifies the entire Euclidicity audit is correlating topological
singularity against **participation ratio**, while asserting — in code, in `audit_memo.yaml`
(`vgt_correlation_pearson_r`), and presumably in the paper — that it is VGT. This matches
the `20260819-pr-vgt-conflation.md` retro exactly: *real scalar VGT is persisted nowhere.*

## 3. Blast radius — every place "VGT" silently means PR

- **Audit concordance gate + `audit_memo.yaml`** (`vgt_correlation_pearson_r`) — the
  load-bearing "is this trustworthy" decision.
- **Metrics dashboard:** `striatica_vgt_mean` / "Mean VGT dimension" (`metrics.py:465-474`)
  and the `"vgt"` stats block (`metrics.py:300-307`).
  - *Caveat (unverified):* the only `set_vgt_stats` reference in the snapshot is the
    docstring example at `metrics.py:24`. No production caller filling `vgt_values` was
    found — so the dashboard's VGT is either empty or populated somewhere outside the
    snapshot. **Verify before claiming this surface is affected.**
- **`ERRATA.md`** lists "VGT dimensional structure" as an independent clean geometric
  finding — but with the scalar discarded, only growth *curves* exist under that name.

## 4. What is NOT affected (bounding the damage)

- **Geometry math is sound.** PCA/UMAP/HDBSCAN/PR compute correctly; no number is wrong.
- **VGT *growth curves* are genuinely VGT.** `cli.py:619` keeps the curves; the machinery
  (`local_dim.py::_vgt_single`) predates the gap and was hardened `2026-03-21`. If the
  paper's "VGT convergence" figures are built from these curves, they most likely stand —
  see the §11 check.
- **PR itself is exact** (Gram-matrix trick, `local_dim.py::_pr_single`).

## 5. Origin, and why surgical — not revert

Commit timeline (from the pack's git logs):

- Activity gap **2026-04-27 → 2026-07-28** (~3 months).
- Dense burst **Jul 28–30**. The defect enters in one commit:
  `2026-07-28 "Bite 1 audit lane: vendored euclidicity in isolated venv"` — adds
  `audit/run_audit.py` with the mislabeled `load_vgt`.
- The Neuronpedia recon that "started the problems" is the same burst
  (`2026-07-28 "traced circuits: Neuronpedia attribution-graph pipeline (141 new tests)"`).

A wholesale revert to the pre-gap April state would **delete work you want to keep**: the
Neuronpedia attribution pipeline, the SSRF/traversal/L0 security guards, the
semantics hashing/salting (`commitment.py`), and the geometry pins themselves
(`umap-learn==0.5.11`, `hdbscan==0.8.41`, numba/llvmlite/pynndescent). The rot is
concentrated in the `audit/` lane + CL-31, both from the Jul-28 burst. **Fix in place.**

## 6. Cross-check against Amp's 0818 review

- **Consistent:** Amp's bottom line — "nothing showed the existing PCA/UMAP, clustering,
  VGT, or local-dimension *geometry* was corrupted" — is true. This finding is orthogonal:
  provenance, not math.
- **Gap in Amp's pass:** Amp reviewed plan-vs-repo parity and the *new* Euclidicity/traced
  evidence; it did not check whether the gate's "VGT" input is actually VGT. Amp's own
  recommended controls (§B "consumer evidence"; §A `SPECIFIED→…→VERIFIED`; §G mutation
  testing) are exactly what catch `load_vgt`.
- **Sibling defect Amp *did* catch, same lane:** the **Euclidicity cross-seed robustness
  check is vacuous** — the scorer ignores `seed`, so cross-seed correlation is 1.0 by
  construction (`audit/euclidicity.py:52-58, 238-245, 295-310`;
  `audit/test_audit_euclidicity.py:219-225, 377-388`). Determinism dressed as robustness.
- **Net:** the `audit/` lane carries **two independent defects** (Amp's cross-seed vacuity +
  this VGT/PR mislabel), both introduced in the same Jul-28 commit, and **neither review
  alone caught both** — the argument for independent adversarial passes, in miniature.

## 7. Independent second problem — VGT is degenerate in the flagship regime

Your own `scripts/0821-vgt-forensics.py` tests this: at D=2304 with unit-norm vectors,
distances concentrate, so `_vgt_single`'s log-count-vs-log-radius slope degenerates toward
the mechanical `log(49) − log(2)` span (E4). In high dimension, VGT-as-implemented largely
measures distance concentration, not intrinsic dimension. **Consequence:** switching the
gate to "real VGT" would concord against a degenerate quantity — so the honest rename
(§8.2, option 1) is preferable to persisting VGT for the gate.

## 8. Proposed Phase-1 remediation (geometry PINNED)

### 8.1 Fail loud on provenance — `load_vgt` guard
Drop-in guard. As-is, it will **correctly refuse** to run the current mislabeled gate,
forcing the §8.2 decision rather than silently continuing:

```python
def load_vgt(dataset_json: Path) -> np.ndarray:
    data = json.load(open(dataset_json))
    method = data.get("dim_method")
    if method != "vgt":
        raise SystemExit(
            f"Refusing to load localDimensions as VGT: dim_method={method!r}. "
            f"That field is {method}, not VGT (FINDINGS-vgt-conflation §2)."
        )
    return np.asarray(data["localDimensions"], dtype=np.float64)
```

### 8.2 Rename to truth (RECOMMENDED, given §7)
Rename everything that says "VGT" but holds PR:

```python
def load_local_dim(dataset_json: Path) -> tuple[np.ndarray, str]:
    data = json.load(open(dataset_json))
    return np.asarray(data["localDimensions"], np.float64), data.get("dim_method", "unknown")
```

- `vgt_concordance_gate` → `local_dim_concordance_gate(min_scores, local_dim, method)`;
  report "(1 − Euclidicity) vs **{method}** concordance r=…". A PR–Euclidicity concordance
  is a legitimate check; it simply must not be *called* VGT.
- Update `audit_memo.yaml` key `vgt_correlation_pearson_r` → `localdim_correlation_pearson_r`
  plus a `localdim_method` field.
- `metrics.py`: `striatica_vgt_mean` → `striatica_localdim_mean` (label carries the method).
- `ERRATA.md`: "VGT dimensional structure" → separate "participation-ratio local dimension"
  (scalar) from "VGT growth curves" (the only persisted VGT).

*(Alternative — option 2: actually persist scalar VGT at `cli.py:619` and gate on it. §7
says it's degenerate at 2304-D, so this still lands on option 1 for the gate; keep VGT
curves as a separately-scoped artifact.)*

### 8.3 Replace the vacuous cross-seed check (Amp §1)
Delete cross-seed correlation as evidence; implement the ε-perturbation protocol (jittered
copies, deterministic scorer, Spearman/rank stability + low-Euclidicity set overlap,
preregistered ε/replicates/thresholds; anti-vacuity assertions). Bit-identical output is
reported **uninformative**, never PASS.

### 8.4 Provenance as a first-class output field
Make `dim_method` (and an analogous `metric_provenance` on audit outputs) **required**, and
validate it on both producer and consumer. No field may be trusted by *name* again.

## 9. Tests to prevent recurrence (no mock data — committed real/synthetic fixtures)

Adopt Amp §A–§G wholesale (evidence state machine, claim-evidence bundle, anti-vacuity
checklist, contract gates, mutation testing). Plus, targeting *this* bug:

1. **Provenance guard:** `load_*` on a `dim_method="pr"` JSON must raise, never silently
   return PR as VGT.
2. **Array-matches-its-method:** recompute `estimate_local_dim(vectors, method=dim_method)`
   on a committed fixture; stored `localDimensions` must match within tolerance.
3. **Estimator distinguishability (metamorphic):** on synthetic known-*d* geometry, PR /
   VGT / TwoNN produce distinct signatures — swapping one for another fails.
4. **0821 forensics as regression:** VGT on isotropic D=2304 collapses to the `log(49)−log(2)`
   span → assert VGT is flagged non-trustworthy in the flagship regime.
5. **Gate anti-vacuity:** the concordance gate fails on two deliberately-discordant metrics;
   cannot pass by construction; a mutation that swaps its input array must flip the result.
6. **Cross-space contract:** local dim runs on original high-D vectors, clustering on 3-D
   coords; fail if 3-D coords are handed to `estimate_local_dim`.
7. **Mutation testing (Stryker rule + Amp §G):** mutmut/cosmic-ray on `local_dim.py`,
   `prepare.py`, `audit/run_audit.py`, `metrics.py`. Mutants like `method="pr"→"vgt"` or
   dropping the guard **must be killed** by tests 1–6. Stryker on the frontend contract.
   Surviving mutant = the test is not evidence.

## 10. Reviewer attack surface — "how to falsify this"

For Amp / Fabled / DeepSeek. Try to break each claim:

- **C1 — "`localDimensions` is PR."** Falsify by finding a production path that writes VGT
  scalars into `localDimensions` (search every `prepare_json`/writer caller for
  `dim_method != "pr"`). *Predicted:* none in snapshot.
- **C2 — "scalar VGT is discarded."** Falsify by finding any persistence of the first return
  of `estimate_local_dim_vgt`. *Predicted:* only `growth_curves` (the 2nd return) survives.
- **C3 — "the gate consumes PR."** Falsify by showing `load_vgt` reads a field other than
  `data["localDimensions"]`, or that a different array reaches the gate.
- **C4 — "geometry math is uncorrupted."** Falsify by a numeric defect in PCA/UMAP/HDBSCAN/PR
  (not labeling). *Predicted:* none.
- **C5 — "VGT curves are real / paper likely stands."** Falsify by showing a paper figure
  labeled "VGT dimension" that is actually the PR scalar (see §11). This is the one most
  worth attacking.
- **C6 — "surgical beats revert."** Falsify by showing the Jul-28 burst is contaminated
  beyond `load_vgt` + the cross-seed check.

## 11. Open decisions / checks for Laura

1. **Rename (§8.2 opt 1) vs persist VGT (opt 2).** Recommendation: rename, per §7.
2. **Verify the metrics caller** (`set_vgt_stats`) — is the dashboard "VGT" populated at all?
3. **Confirm no paper figure** captioned "VGT dimension" is the PR scalar. (Curves are safe;
   the scalar is the exposure.)
4. **CL-31 and the cross-seed protocol** are Amp's blockers — track them alongside, but they
   are separate from this label fix.
5. **Phase 3 (later):** transcoder mapping off Neuronpedia, no SAELens, into an invertible /
   reverse-engineerable 3-D transform to retire UMAP.
