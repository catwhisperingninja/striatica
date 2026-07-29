# striatica/pipeline/semantics_merge.py
"""Merge Neuronpedia explanations into an existing dataset JSON, locally.

This is a *local-only* post-processing step. The geometry pipeline
(`process_gpt2_small.py` / `striat model`) produces a dataset JSON whose
`features[].explanation` fields may be empty — either because the model was
processed with semantics redacted, or because Neuronpedia had not published
explanations at generation time. `merge_explanations` patches those
explanation strings in place from a Neuronpedia explanations JSONL bulk export,
touching NOTHING else in the file.

Why this exists as its own unit (not folded into `prepare.py`):
    - Semantic labels are SECRET for non-public-tier models (see CLAUDE.md).
      The merge hard-refuses any dataset whose model is not `is_public_tier`,
      so restricted-model geometry can never gain semantic labels by accident.
    - The geometry (positions, clusters, local dimensions, VGT growth curves)
      is expensive and reproducibility-sensitive (UMAP is not reproducible
      across library versions — see CLAUDE.md). Re-running the pipeline just to
      add explanations would risk changing every 3D position. This merge is a
      byte-preserving patch: only `features[].explanation` and the top-level
      `semanticsRedacted` flag may change.

The explanations JSONL is the real Neuronpedia S3 export line shape: one JSON
object per line with a string `"index"` and a `"description"`, plus other
fields the merge ignores. First non-empty description for an index wins (the
same first-occurrence rule `prepare.py` uses); empty descriptions never erase
an existing explanation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pipeline.banner import detail, error, info, step_header, success
from pipeline.config import is_public_tier

# Module-global handle so `main --download` can be intercepted in tests by
# monkeypatching `pipeline.semantics_merge.download_explanations` — no network.
from pipeline.download import download_explanations

# Subtrees the merge must NEVER touch (IRON RULE). Documented here so the
# contract is visible at the call site; the code simply never assigns to them.
_IRON_RULE_KEYS = ("positions", "clusterLabels", "localDimensions", "growthCurves", "clusters")

# "layer12-l0604" -> layer index 12; "layer5-l088" -> 5. Transcoder datasets
# name their layer this way (see cli.py `_run_process_pipeline`); anything that
# does not match is assumed to already be a Neuronpedia source/layer id.
_TRANSCODER_LAYER_RE = re.compile(r"^layer(\d+)-l0(\d+)$")

# The l0 dictionary Neuronpedia has DEPLOYED per source id — the dictionary its
# published explanations are indexed by. The gemma-2-2b layer-12
# gemmascope-transcoder-16k source is average_l0_6, while local datasets were
# built from average_l0_604 (a DIFFERENT dictionary — see CLAUDE.md /
# graph_fetch identity). Merging l0_6-indexed labels positionally onto l0_604
# features silently mislabels them, so a mismatch is refused. A source id absent
# from this map is unrecognized -> the guard stays dormant (safe no-op).
_DEPLOYED_L0_VARIANTS = {"12-gemmascope-transcoder-16k": 6}


def neuronpedia_layer_id(layer: str) -> str:
    """Map a dataset "layer" string to its Neuronpedia source/layer id.

    Transcoder datasets store layer as ``layer{N}-l0{M}`` (e.g. ``layer12-l0604``);
    Neuronpedia's matching source id is ``{N}-gemmascope-transcoder-16k``. Any
    other value (e.g. a SAELens id like ``6-res-jb``) is already a source id and
    passes through unchanged.
    """
    m = _TRANSCODER_LAYER_RE.match(layer)
    if m:
        return f"{int(m.group(1))}-gemmascope-transcoder-16k"
    return layer


def _load_explanations(explanations_jsonl_path: Path, num_features: int) -> dict[int, str]:
    """Read a Neuronpedia explanations JSONL into {local_index: description}.

    Rules (pinned by tests/test_semantics_merge.py):
      - index via int(line["index"]) — real exports use STRINGS.
      - description via line["description"].
      - First NON-EMPTY description for an index wins (first-occurrence rule).
      - Skipped without error: blank lines, empty descriptions (never erase an
        existing explanation), and indices outside [0, num_features).
      - Extra line fields ignored.
    """
    explanations: dict[int, str] = {}
    with open(explanations_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = int(record["index"])
            if idx < 0 or idx >= num_features:
                continue
            description = record.get("description", "")
            if not description:
                continue
            if idx not in explanations:  # first non-empty occurrence wins
                explanations[idx] = description
    return explanations


def _check_l0_variant(dataset_json_path: Path, data: dict) -> None:
    """Sidecar-gated L0-dictionary guard (raises ValueError on a mismatch).

    Engages ONLY when a ``{stem}-metadata.json`` sidecar is discoverable next to
    the dataset AND its ``transcoder.l0_variant`` differs from the DEPLOYED
    Neuronpedia dictionary (``_DEPLOYED_L0_VARIANTS`` keyed by the dataset's
    Neuronpedia source id). No sidecar, no recorded variant, or an unrecognized
    source id -> no-op: a dataset we cannot identify is merged unchanged.
    """
    sidecar = dataset_json_path.parent / f"{dataset_json_path.stem}-metadata.json"
    if not sidecar.exists():
        return
    meta = json.loads(sidecar.read_text())
    dataset_l0 = meta.get("transcoder", {}).get("l0_variant")
    if dataset_l0 is None:
        return
    np_layer_id = neuronpedia_layer_id(data.get("layer", ""))
    deployed_l0 = _DEPLOYED_L0_VARIANTS.get(np_layer_id)
    if deployed_l0 is None:
        return
    if int(dataset_l0) != int(deployed_l0):
        raise ValueError(
            f"Refusing to merge semantics: this dataset was built from the "
            f"average_l0_{dataset_l0} transcoder dictionary, but Neuronpedia's "
            f"deployed '{np_layer_id}' source is average_l0_{deployed_l0} — "
            f"different dictionaries whose feature indices are not comparable, "
            f"so positional merging would mislabel features. Pass "
            f"allow_l0_mismatch=True (CLI: --allow-l0-mismatch) to override."
        )


def merge_explanations(
    dataset_json_path: str | Path,
    explanations_jsonl_path: str | Path,
    out_path: str | Path | None = None,
    allow_l0_mismatch: bool = False,
) -> dict:
    """Patch features[].explanation in an existing dataset JSON from a JSONL.

    Args:
        dataset_json_path: EXISTING dataset JSON (prepare_json shape).
        explanations_jsonl_path: Neuronpedia explanations JSONL (S3 line shape).
        out_path: Destination. None overwrites dataset_json_path in place;
            otherwise the source file's bytes are left untouched.
        allow_l0_mismatch: Skip the sidecar-gated L0-dictionary guard (see
            _check_l0_variant) and merge anyway.

    Returns:
        The merged dataset dict (identical to what was written to disk).

    Raises:
        ValueError: if the dataset's model is not public tier, or if the L0
            dictionary guard trips — writes NOTHING in either case.
        FileNotFoundError: if the explanations JSONL is missing — writes NOTHING.

    IRON RULE: positions, clusterLabels, localDimensions, growthCurves and
    clusters are byte-identical after the merge; only features[].explanation and
    the top-level semanticsRedacted flag may change.
    """
    dataset_json_path = Path(dataset_json_path)
    explanations_jsonl_path = Path(explanations_jsonl_path)

    with open(dataset_json_path) as f:
        data = json.load(f)

    # Safety tier gate FIRST — a restricted model must never gain semantic
    # labels, and must never even read the explanations file. Write nothing.
    model = data.get("model", "")
    if not is_public_tier(model):
        raise ValueError(
            f"Refusing to merge semantics: model '{model}' is not public tier. "
            "Semantic labels are only merged for public-tier models."
        )

    # L0-dictionary guard (sidecar-gated) — refuse mismatched dictionaries so
    # l0_6-indexed Neuronpedia labels are never merged onto l0_604 features.
    if not allow_l0_mismatch:
        _check_l0_variant(dataset_json_path, data)

    if not explanations_jsonl_path.exists():
        raise FileNotFoundError(f"Explanations JSONL not found: {explanations_jsonl_path}")

    num_features = data["numFeatures"]
    explanations = _load_explanations(explanations_jsonl_path, num_features)

    # Patch ONLY matched local indices; every other feature is left verbatim.
    features = data["features"]
    for idx, description in explanations.items():
        features[idx]["explanation"] = description

    data["semanticsRedacted"] = False

    target = Path(out_path) if out_path is not None else dataset_json_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(data, f)

    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="striat semantics-merge",
        description=(
            "Merge Neuronpedia explanations into a local dataset JSON, patching "
            "only features[].explanation. Public-tier models only; never commit "
            "the merged output (see CLAUDE.md — semantic labels are secret)."
        ),
    )
    parser.add_argument("dataset", help="Existing dataset JSON to patch (prepare_json shape)")
    parser.add_argument("explanations", help="Neuronpedia explanations JSONL (S3 export line shape)")
    parser.add_argument(
        "--out",
        default=None,
        help="Write merged JSON here (default: overwrite the dataset in place)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch explanations from Neuronpedia S3 into EXPLANATIONS first "
             "(model + layer derived from the dataset), then merge.",
    )
    parser.add_argument(
        "--allow-l0-mismatch",
        action="store_true",
        help="Merge even when the dataset's transcoder l0 dictionary differs "
             "from Neuronpedia's deployed source (indices may not be "
             "comparable). Mirrors circuits --traced.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `striat semantics-merge`.

    usage: DATASET_JSON EXPLANATIONS_JSONL [--out OUT_JSON] [--download]

    Returns 0 on success; a nonzero int (no uncaught exception) on missing
    files or a tier-gate failure.
    """
    args = _build_parser().parse_args(argv)

    dataset_path = Path(args.dataset)
    explanations_path = Path(args.explanations)
    out_path = Path(args.out) if args.out else None

    step_header("assemble", "Merging semantic explanations")
    info("Dataset", str(dataset_path), emoji="📁")

    try:
        if args.download:
            # Derive Neuronpedia model + layer from the dataset itself, gate on
            # tier BEFORE any network fetch, then download into EXPLANATIONS.
            with open(dataset_path) as f:
                meta = json.load(f)
            model_id = meta.get("model", "")
            if not is_public_tier(model_id):
                raise ValueError(
                    f"Refusing to download semantics: model '{model_id}' is not public tier."
                )
            np_layer = neuronpedia_layer_id(meta.get("layer", ""))
            info("Download", f"{model_id}/{np_layer}", emoji="🛰️")
            download_explanations(
                model_id=model_id,
                layer=np_layer,
                output_path=explanations_path,
            )

        merge_explanations(
            dataset_path,
            explanations_path,
            out_path=out_path,
            allow_l0_mismatch=args.allow_l0_mismatch,
        )
    except FileNotFoundError as e:
        error(str(e))
        return 1
    except ValueError as e:
        error(str(e))
        return 1

    written = out_path if out_path is not None else dataset_path
    detail(f"Explanations merged from {explanations_path.name}")
    success(f"Merged semantics → {written}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
