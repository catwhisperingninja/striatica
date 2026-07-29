#!/usr/bin/env python3
"""Generate traced circuits from Neuronpedia attribution graphs.

Thin argparse orchestrator around pipeline.graph_fetch (network) and
pipeline.traced_circuits (build + write). Invoked either directly or via
``striat circuits --traced ...`` (see pipeline/cli.py cmd_circuits).

Graph sources (at least one required):
  --slug SLUG            fetch an existing public graph (repeatable)
  --slugs-file FILE      newline-separated slugs to fetch
  --list                 list graphs owned by the API key (requires
                         NEURONPEDIA_API_KEY)
  --generate --prompt P  generate a new graph (requires NEURONPEDIA_API_KEY;
                         consumes --slug as the new graph's slug)

Outputs (wet run only): one traced circuit JSON per slug plus a manifest under
{out_dir}/circuits/{dataset_stem}/, written in a single
write_traced_outputs() call (the writer owns the manifest). --dry-run
fetches, parses, and validates but writes nothing anywhere.

Dataset identity: the local dataset metadata is read from
{out_dir}/{dataset_stem}-metadata.json and passed to build_traced_circuit,
whose identity validator hard-fails on model / source-set / L0 mismatches
(see pipeline.graph_fetch.validate_graph_identity for the three-state L0
logic and --allow-l0-mismatch semantics).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.banner import (  # noqa: E402
    detail,
    error,
    info,
    step_header,
    success,
    warn,
)
from pipeline.graph_fetch import (  # noqa: E402
    NEURONPEDIA_BASE,
    fetch_graph,
    fetch_graph_record,
    fetch_source_set,
    generate_graph,
    load_neuronpedia_api_key,
)
from pipeline.traced_circuits import (  # noqa: E402
    build_traced_circuit,
    write_traced_outputs,
)

DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_LAYER = 12
DEFAULT_DATASET_STEM = "gemma-2-2b-layer12-l06"
DEFAULT_OUT_DIR = "frontend/public/data"


# ── Parser ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="striat circuits --traced",
        description=(
            "Generate traced circuits from Neuronpedia attribution graphs "
            "(causal attribution, layer-filtered to the local dataset's "
            "transcoder layer)."
        ),
    )
    src = parser.add_argument_group("graph sources (at least one required)")
    src.add_argument(
        "--slug", action="append", default=None, metavar="SLUG",
        help="Neuronpedia graph slug to fetch (repeatable). With --generate, "
             "the slug for the newly generated graph.",
    )
    src.add_argument(
        "--slugs-file", default=None, metavar="FILE",
        help="File with one graph slug per line (blank lines and # comments "
             "ignored).",
    )
    src.add_argument(
        "--list", action="store_true",
        help="List graphs owned by your API key (requires "
             "NEURONPEDIA_API_KEY).",
    )
    src.add_argument(
        "--generate", action="store_true",
        help="Generate a new attribution graph via the Neuronpedia API "
             "(requires NEURONPEDIA_API_KEY and --prompt; rate limit "
             "30/hour). Consumes --slug as the new graph's slug.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt text for --generate (max 64 tokens).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Neuronpedia model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help=f"Transformer layer to keep feature nodes from (default: "
             f"{DEFAULT_LAYER}).",
    )
    parser.add_argument(
        "--dataset-stem", default=DEFAULT_DATASET_STEM,
        help=f"Local dataset stem; metadata is read from "
             f"{{out-dir}}/{{stem}}-metadata.json (default: "
             f"{DEFAULT_DATASET_STEM}).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Public data root (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Circuit name override (default: the graph slug).",
    )
    parser.add_argument(
        "--description", default=None,
        help="Circuit description override.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the local graph cache and re-fetch from Neuronpedia.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch, parse, and validate, then print a summary — write "
             "nothing.",
    )
    parser.add_argument(
        "--allow-l0-mismatch", action="store_true",
        help="Proceed when the source-set L0 cannot be verified against the "
             "dataset (l0Verified=false). A PROVEN mismatch still hard-fails.",
    )
    parser.add_argument(
        "--redact-roles", action="store_true",
        help="Replace supernode role labels with group-N placeholders.",
    )
    return parser


# ── Helpers ──────────────────────────────────────────────────────────────


def _read_slugs_file(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"Slugs file not found: {path}")
    slugs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            slugs.append(line)
    if not slugs:
        raise RuntimeError(f"Slugs file {path} contains no slugs.")
    return slugs


def _load_dataset_metadata(out_dir: Path, dataset_stem: str) -> dict:
    metadata_path = out_dir / f"{dataset_stem}-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Dataset metadata not found: {metadata_path}. The traced "
            f"circuit builder validates graph/dataset identity against "
            f"{{out-dir}}/{{dataset-stem}}-metadata.json — check "
            f"--dataset-stem and --out-dir."
        )
    return json.loads(metadata_path.read_text())


def _require_api_key(flag: str):
    """Return the API key or raise with an actionable message."""
    api_key = load_neuronpedia_api_key()
    if not api_key:
        raise RuntimeError(
            f"{flag} requires a Neuronpedia API key, but NEURONPEDIA_API_KEY "
            f"is not set. Add NEURONPEDIA_API_KEY=<key> to the .env file at "
            f"the project root. Without a key, existing public graphs can "
            f"still be fetched with --slug."
        )
    return api_key


def _list_owned_graphs(api_key: str) -> list[dict]:
    """GET /api/graph/list-owned (x-api-key required)."""
    url = f"{NEURONPEDIA_BASE}/api/graph/list-owned"
    request = urllib.request.Request(url, headers={"x-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Neuronpedia list-owned request failed with HTTP {e.code} "
            f"({url}). Check that NEURONPEDIA_API_KEY is valid."
        ) from e
    data = json.loads(payload.decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("graphs", data.get("data", []))
    return data if isinstance(data, list) else []


def _cmd_list(args: argparse.Namespace) -> int:
    api_key = _require_api_key("--list")
    graphs = _list_owned_graphs(api_key)
    if not graphs:
        warn("No graphs owned by this API key.")
        return 0
    info("owned graphs", str(len(graphs)), emoji="🔗")
    for g in graphs:
        slug = g.get("slug", "?") if isinstance(g, dict) else str(g)
        model = g.get("modelId", "?") if isinstance(g, dict) else "?"
        detail(f"{model}/{slug}")
    return 0


def _build_one(
    args: argparse.Namespace, slug: str, dataset_metadata: dict
) -> dict:
    """Fetch record -> full graph -> source set, then build the circuit."""
    info("slug", slug, emoji="🔗")
    record = fetch_graph_record(args.model, slug)
    raw_graph = fetch_graph(args.model, slug, force=args.force)
    source_set_name = record.get("sourceSetName")
    if not source_set_name:
        raise RuntimeError(
            f"Graph record for slug '{slug}' has no sourceSetName — cannot "
            f"verify dictionary identity."
        )
    source_set = fetch_source_set(args.model, source_set_name)
    circuit = build_traced_circuit(
        raw_graph,
        record,
        source_set,
        dataset_metadata,
        name=args.name,
        description=args.description,
        layer=args.layer,
        allow_l0_mismatch=args.allow_l0_mismatch,
        redact_roles=args.redact_roles,
    )
    detail(
        f"{slug}: {len(circuit['nodes'])} layer-{args.layer} nodes, "
        f"{len(circuit['edges'])} edges"
    )
    return circuit


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    step_header("circuits", "Traced circuits (Neuronpedia attribution graphs)")

    if args.list:
        return _cmd_list(args)

    if args.generate:
        if not args.prompt:
            parser.error(
                "--generate requires --prompt (the text to trace, "
                "max 64 tokens)"
            )
        if not args.slug or len(args.slug) != 1:
            parser.error(
                "--generate requires exactly one --slug naming the new graph"
            )

    slugs: list[str] = list(args.slug or [])
    if args.slugs_file:
        slugs.extend(_read_slugs_file(Path(args.slugs_file)))
    if not slugs:
        parser.error(
            "no graph slugs to process — provide --slug or a non-empty "
            "--slugs-file"
        )

    if args.generate:
        api_key = _require_api_key("--generate")
        info("generate", f"'{args.prompt}' -> {slugs[0]}", emoji="⚡")
        generate_graph(args.model, args.prompt, slugs[0], api_key)
        success(f"graph '{slugs[0]}' generated")

    out_dir = Path(args.out_dir)
    dataset_metadata = _load_dataset_metadata(out_dir, args.dataset_stem)
    info("dataset", args.dataset_stem, emoji="📋")
    if args.name and len(slugs) > 1:
        warn(
            "--name applies to every slug — multiple circuits will share "
            "one name."
        )

    circuits = [_build_one(args, slug, dataset_metadata) for slug in slugs]

    if args.dry_run:
        warn("dry run — nothing written")
        success(
            f"dry run complete: {len(circuits)} circuit(s) fetched and "
            f"validated"
        )
        return 0

    dest = write_traced_outputs(circuits, out_dir, args.dataset_stem)
    success(f"wrote {len(circuits)} traced circuit(s) -> {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.slug or args.slugs_file or args.list or args.generate):
        parser.error(
            "at least one graph source is required: --slug, --slugs-file, "
            "--list, or --generate"
        )

    try:
        return _run(args, parser)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — CLI boundary: no raw tracebacks
        error(str(exc) or exc.__class__.__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
