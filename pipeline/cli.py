#!/usr/bin/env python3
"""striat — CLI entry point for the striatica data pipeline.

Subcommands:
    striat demo       Run the GPT-2 Small demo (generate data + circuits + launch frontend)
    striat model      Generate data for any SAELens-compatible model
    striat discover   Discover available models from SAELens + Neuronpedia
    striat batch      Process multiple models sequentially
    striat circuits   Generate circuit data (co-activation or similarity)
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"


# ── Utilities ────────────────────────────────────────────────────────────

def _port_available(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_port(start: int = 5173, attempts: int = 20) -> int:
    """Find the first available port starting from `start`."""
    for offset in range(attempts):
        port = start + offset
        if _port_available(port):
            return port
    raise RuntimeError(f"No available port found in range {start}–{start + attempts - 1}")


def _launch_frontend(port: int) -> None:
    """Install deps and start the Vite dev server on the given port."""
    from pipeline.banner import info, separator

    separator()
    info("Frontend", f"http://localhost:{port}", emoji="🖥️")

    # Ensure pnpm is available
    if not shutil.which("pnpm"):
        print("  pnpm not found, installing via corepack...")
        subprocess.run(["corepack", "enable"], check=True)

    subprocess.run(["pnpm", "install"], cwd=FRONTEND_DIR, check=True)

    # Launch Vite via Popen so we control shutdown cleanly.
    # subprocess.run + check=True lets pnpm print ELIFECYCLE on SIGINT;
    # managing the process ourselves avoids that noise.
    proc = subprocess.Popen(
        ["pnpm", "dev", "--port", str(port)],
        cwd=FRONTEND_DIR,
        # Give pnpm its own process group so SIGINT doesn't race
        preexec_fn=os.setpgrp,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        # Send SIGTERM to the process group (pnpm + vite), then wait
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already exited
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def _detect_device(requested: str = "auto") -> str:
    """Resolve torch device: auto-detect cuda → mps → cpu, or use explicit."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Simple y/n prompt with a default."""
    suffix = " [Y/n] " if default else " [y/N] "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


# ── Demo ─────────────────────────────────────────────────────────────────

def cmd_demo(args: argparse.Namespace) -> None:
    """Run the GPT-2 Small demo end-to-end."""
    from pipeline.banner import (
        print_banner, step_header, step_done, step_cached,
        detail, reset_step_counter,
    )
    from pipeline.config import GPT2_SMALL_L6, DATA_DIR

    print_banner()
    reset_step_counter()

    cfg = GPT2_SMALL_L6
    dataset_path = OUTPUT_DIR / f"{cfg.model_id}-{cfg.layer}.json"
    circuits_dir = OUTPUT_DIR / "circuits"
    manifest_path = circuits_dir / "manifest.json"

    # Step 1: Generate point cloud data (skip if cached)
    if dataset_path.exists() and not args.regenerate:
        step_cached(dataset_path.name)
        detail("(use --regenerate to rebuild)")
    else:
        detail("First run detected — downloading ~2 GB from Neuronpedia S3")
        _run_process_pipeline(cfg, DATA_DIR)

    # Step 2: Generate circuits
    if manifest_path.exists() and not args.regenerate:
        n_circuits = len(list(circuits_dir.glob("*.json"))) - 1
        step_cached(f"circuits ({n_circuits} circuits)")
        detail("(use --regenerate to rebuild)")
    else:
        if _ask_yes_no("  🔗  Generate default circuits? (5 co-activation + 5 similarity)"):
            from scripts.generate_circuits import generate_batch_defaults
            generate_batch_defaults()
        else:
            detail("Skipping circuit generation.")

    # Step 3: Launch frontend
    port = args.port if args.port is not None else _find_port()
    _launch_frontend(port)


# ── Discover ───────────────────────────────────────────────────────────

def cmd_discover(args: argparse.Namespace) -> None:
    """Discover available models from SAELens + Neuronpedia."""
    from pipeline.banner import print_banner, info, success
    from pipeline.config import DATA_DIR
    from pipeline.discovery import (
        discover_models, save_catalog, catalog_summary, generate_readme_table,
    )

    print_banner()
    info("Discover", "Pulling SAELens registry...", emoji="🔍")
    print()

    sae_types = [s.strip() for s in args.sae_types.split(",") if s.strip()] if args.sae_types else None
    model_families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None

    catalog = discover_models(
        probe_s3=args.probe_s3,
        sae_types=sae_types,
        model_families=model_families,
        require_neuronpedia=not args.include_no_neuronpedia,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        save_catalog(catalog, output_path)

    if args.readme:
        table = generate_readme_table(catalog)
        print(table)
    else:
        summary = catalog_summary(catalog)
        print(summary)

    success(f"{len(catalog)} SAE configurations discovered")


# ── Model ──────────────────────────────────────────────────────────────

def cmd_model(args: argparse.Namespace) -> None:
    """Generate data for any SAELens-compatible model.

    Accepts either explicit parameters (--sae-release, --sae-hook, etc.)
    or a Neuronpedia ID shorthand (--np-id "gpt2-small/6-res-jb") which
    auto-resolves all parameters from the SAELens registry.
    """
    from pipeline.banner import (
        print_banner, info, success, error, warn, detail, separator,
        reset_step_counter,
    )
    from pipeline.config import SAEConfig, DATA_DIR, is_public_tier

    print_banner()
    reset_step_counter()

    device = _detect_device(args.device)

    # ── Resolve config from Neuronpedia ID or explicit params ──
    if args.np_id:
        cfg = _resolve_from_np_id(args.np_id, args)
    elif args.sae_release and args.sae_hook:
        cfg = SAEConfig(
            model_id=args.model,
            layer=args.layer,
            sae_release=args.sae_release,
            sae_hook=args.sae_hook,
            num_batches=args.num_batches,
            features_per_batch=args.features_per_batch,
        )
    else:
        error("Provide either --np-id or (--sae-release + --sae-hook)")
        detail("Run 'striat discover' to see available models.")
        sys.exit(1)

    # Safety: determine whether semantic labels should be included
    public = is_public_tier(cfg.model_id)
    include_semantics = args.include_semantics if args.include_semantics else public
    redact = not include_semantics

    info("Model", cfg.model_id, emoji="🧬")
    info("Layer", cfg.layer, emoji="📐")
    info("SAE release", cfg.sae_release, emoji="🛰️")
    info("SAE hook", cfg.sae_hook, emoji="🔗")
    info("Features", f"{cfg.num_batches * cfg.features_per_batch:,}", emoji="⚡")
    info("Device", device, emoji="🖥️")
    if redact:
        info("Semantics", "REDACTED (model not in public tier)", emoji="🔒")
        detail("Use --include-semantics to override")
    else:
        info("Semantics", "included", emoji="📖")
    print()

    _run_process_pipeline(cfg, DATA_DIR, device=device, redact_semantics=redact)

    dataset_file = f"{cfg.model_id}-{cfg.layer}.json"
    dataset_path = OUTPUT_DIR / dataset_file

    if args.json_export:
        success(f"JSON exported: {dataset_path}")
        detail("Transfer to local machine:")
        detail(f"scp remote:{dataset_path} ./frontend/public/data/")
        detail(f"Then open: http://localhost:5173/?dataset={dataset_file}")
    else:
        success(f"Output: {dataset_path}")
        detail("Launch the frontend with:")
        detail("cd frontend && pnpm dev")
        detail(f"Then open: http://localhost:5173/?dataset={dataset_file}")


def _resolve_from_np_id(np_id: str, args: argparse.Namespace):
    """Resolve a Neuronpedia ID (e.g. 'gpt2-small/6-res-jb') to a full SAEConfig.

    Looks up the SAELens registry to find the matching release and hook point,
    then probes S3 for batch count.
    """
    from pipeline.config import SAEConfig
    from pipeline.discovery import _load_saelens_registry, _probe_s3_batch_count

    from pipeline.banner import info, error, warn, detail

    parts = np_id.split("/", 1)
    if len(parts) != 2:
        error(f"Invalid Neuronpedia ID format: '{np_id}'")
        detail("Expected: 'model-id/layer-id' (e.g. 'gpt2-small/6-res-jb')")
        sys.exit(1)

    np_model_id, np_layer = parts

    info("Resolving", np_id, emoji="🔍")
    registry = _load_saelens_registry()

    # Search registry for matching neuronpedia_id
    match_release = None
    match_sae_id = None
    for release_key, entry in registry.items():
        np_ids = getattr(entry, "neuronpedia_id", None) or {}
        if not isinstance(np_ids, dict):
            continue
        for sae_id, this_np_id in np_ids.items():
            if this_np_id == np_id:
                match_release = getattr(entry, "release", release_key)
                match_sae_id = sae_id
                break
        if match_release:
            break

    if not match_release:
        error(f"Neuronpedia ID '{np_id}' not found in SAELens registry.")
        detail("Run 'striat discover' to see available models.")
        sys.exit(1)

    info("Found", f"release={match_release}, hook={match_sae_id}", emoji="✅")

    # Probe S3 for batch count
    info("Probing", "S3 batch count...", emoji="🛰️")
    num_batches = _probe_s3_batch_count(np_model_id, np_layer)
    if num_batches is None:
        warn(f"Could not probe S3 batches. Using --num-batches={args.num_batches}")
        num_batches = args.num_batches

    detail(f"{num_batches} batches found")

    return SAEConfig(
        model_id=np_model_id,
        layer=np_layer,
        sae_release=match_release,
        sae_hook=match_sae_id,
        num_batches=num_batches,
        features_per_batch=args.features_per_batch,
    )


# ── Batch ──────────────────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> None:
    """Process multiple models sequentially."""
    from pipeline.banner import (
        print_banner, info, success, error, warn, detail,
        separator, reset_step_counter,
    )
    from pipeline.config import DATA_DIR, SAEConfig, is_public_tier
    from pipeline.discovery import load_catalog, ModelInfo, _probe_s3_batch_count

    print_banner()

    device = _detect_device(args.device)

    # Load model list from catalog or explicit np-ids
    if args.catalog:
        catalog = load_catalog(Path(args.catalog))
        info("Catalog", f"{len(catalog)} models loaded", emoji="📋")
    elif args.np_ids:
        # Parse comma-separated Neuronpedia IDs
        np_id_list = [x.strip() for x in args.np_ids.split(",") if x.strip()]
        catalog = []
        for np_id in np_id_list:
            # Create a minimal ModelInfo — we'll resolve fully during processing
            parts = np_id.split("/", 1)
            if len(parts) == 2:
                catalog.append(ModelInfo(
                    sae_release="", sae_id="", model="", repo_id="",
                    neuronpedia_id=np_id, np_model_id=parts[0], np_layer=parts[1],
                    conversion_func="",
                ))
            else:
                warn(f"Skipping invalid Neuronpedia ID: '{np_id}' (expected 'model/layer')")
        info("Batch", f"{len(catalog)} models specified", emoji="📋")
    else:
        error("Provide --catalog <path> or --np-ids 'id1,id2,...'")
        sys.exit(1)

    results = []
    failed = []

    for i, model_info in enumerate(catalog, 1):
        np_id = model_info.neuronpedia_id
        separator()
        info("Model", f"{i}/{len(catalog)}: {np_id}", emoji="📦")

        # Check for existing output (resume capability)
        output_file = OUTPUT_DIR / f"{model_info.np_model_id}-{model_info.np_layer}.json"
        if output_file.exists() and not args.force:
            info("Status", "cached, skipping (use --force to reprocess)", emoji="💾")
            results.append((np_id, "skipped", 0))
            continue

        try:
            reset_step_counter()

            # Resolve full config via registry lookup
            class _FakeArgs:
                num_batches = args.num_batches
                features_per_batch = args.features_per_batch
            cfg = _resolve_from_np_id(np_id, _FakeArgs())

            public = is_public_tier(cfg.model_id)
            redact = not public

            t_start = time.time()
            _run_process_pipeline(cfg, DATA_DIR, device=device, redact_semantics=redact)
            elapsed = time.time() - t_start

            results.append((np_id, "success", elapsed))
        except Exception as e:
            error(f"Failed: {e}")
            failed.append((np_id, str(e)))
            results.append((np_id, "failed", 0))
            if not args.continue_on_error:
                detail("Stopping batch (use --continue-on-error to skip failures)")
                break

    # Summary
    separator()
    info("Summary", "Batch Results", emoji="📊")
    separator()
    for np_id, status, elapsed in results:
        if status == "success":
            m, s = divmod(int(elapsed), 60)
            success(f"{np_id} — {m}m {s}s")
        elif status == "skipped":
            info("Cached", np_id, emoji="💾")
        else:
            error(f"{np_id} — failed")

    total_time = sum(e for _, _, e in results)
    m, s = divmod(int(total_time), 60)
    detail(f"Total processing time: {m}m {s}s")
    if failed:
        warn(f"{len(failed)} failures:")
        for np_id, err in failed:
            detail(f"{np_id}: {err}")


def _run_process_pipeline(cfg, data_dir: Path, device: str = "cpu", redact_semantics: bool = False) -> None:
    """Run the full data pipeline for a given SAEConfig."""
    from pipeline.banner import step_header, step_done, step_cached, detail, separator, success
    from pipeline.download import download_features, download_explanations
    from pipeline.vectors import load_decoder_vectors
    from pipeline.reduce import reduce_to_3d
    from pipeline.cluster import cluster_points
    from pipeline.prepare import prepare_json

    data_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # Step 1: Download metadata
    features_path = data_dir / f"{cfg.model_id}_{cfg.layer}_features.jsonl"
    if not features_path.exists():
        t0 = time.time()
        step_header("download", "Step 1a/6 · Downloading feature metadata")
        batch_indices = list(range(cfg.num_batches))
        download_features(cfg.model_id, cfg.layer, batch_indices=batch_indices, output_path=features_path)
        step_done(time.time() - t0)
    else:
        step_cached(features_path.name)

    explanations_path = data_dir / f"{cfg.model_id}_{cfg.layer}_explanations.jsonl"
    if not explanations_path.exists():
        t0 = time.time()
        step_header("download", "Step 1b/6 · Downloading explanations")
        batch_indices = list(range(cfg.num_batches))
        download_explanations(cfg.model_id, cfg.layer, batch_indices=batch_indices, output_path=explanations_path)
        step_done(time.time() - t0)
    else:
        step_cached(explanations_path.name)

    # Step 2: Load decoder vectors
    t0 = time.time()
    step_header("vectors", "Step 2/6 · Loading SAELens decoder vectors")
    vectors = load_decoder_vectors(cfg.sae_release, cfg.sae_hook, device=device)
    detail(f"{vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
    step_done(time.time() - t0)

    # Step 3: Dimensionality reduction
    t0 = time.time()
    step_header("reduce", "Step 3/6 · PCA + UMAP → 3D")
    coords = reduce_to_3d(vectors, pca_dim=50)
    step_done(time.time() - t0)

    # Step 4: Clustering
    t0 = time.time()
    step_header("cluster", "Step 4/6 · HDBSCAN clustering")
    labels = cluster_points(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    detail(f"{n_clusters} clusters found")
    step_done(time.time() - t0)

    # Step 5: Local dimension estimation
    from pipeline.local_dim import estimate_local_dim, estimate_local_dim_vgt

    t0 = time.time()
    step_header("dimension", "Step 5a/6 · Local intrinsic dimension (Participation Ratio)")
    local_dims = estimate_local_dim(vectors, method="pr")
    step_done(time.time() - t0)

    t0 = time.time()
    step_header("vgt", "Step 5b/6 · VGT growth curves")
    _, growth_curves = estimate_local_dim_vgt(vectors, return_curves=True)
    step_done(time.time() - t0)

    # Step 6: Assemble JSON
    t0 = time.time()
    step_header("assemble", "Step 6/6 · Preparing JSON for frontend")
    output = OUTPUT_DIR / f"{cfg.model_id}-{cfg.layer}.json"
    prepare_json(
        coords, labels, features_path, explanations_path, output,
        local_dimensions=local_dims, dim_method="pr", growth_curves=growth_curves,
        model=cfg.model_id, layer=cfg.layer,
        redact_semantics=redact_semantics,
    )
    step_done(time.time() - t0)

    elapsed = time.time() - t_total
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Write metadata.json alongside the dataset for output packaging
    import datetime
    metadata = {
        "model_id": cfg.model_id,
        "layer": cfg.layer,
        "sae_release": cfg.sae_release,
        "sae_hook": cfg.sae_hook,
        "num_features": cfg.num_batches * cfg.features_per_batch,
        "processing_time_seconds": round(elapsed),
        "pipeline_version": "0.2.0",
        "device": device,
        "redact_semantics": redact_semantics,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    metadata_path = OUTPUT_DIR / f"{cfg.model_id}-{cfg.layer}-metadata.json"
    import json as _json
    with open(metadata_path, "w") as _f:
        _json.dump(metadata, _f, indent=2)

    separator()
    success(f"Complete · {minutes}m {seconds}s")
    detail(f"📁  {output}")
    detail(f"📋  {metadata_path}")


# ── Circuits ─────────────────────────────────────────────────────────────

def cmd_circuits(args: argparse.Namespace) -> None:
    """Delegate to the existing generate_circuits script."""
    from pipeline.banner import print_banner

    print_banner()

    # The generate_circuits script lives outside the installed package
    # (striatica/scripts/), so we add it to sys.path and invoke its main().
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    sys.argv = ["striat circuits"] + args.circuit_args
    from generate_circuits import main as circuits_main
    circuits_main()


# ── Main parser ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="striat",
        description="striatica — geometric atlas for machine intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  striat demo                                  # GPT-2 Small demo, launches browser\n"
            "  striat discover                              # show all available SAEs\n"
            "  striat discover --families gpt2,gemma2       # filter by model family\n"
            "  striat discover --sae-types res              # residual stream only\n"
            "  striat discover --readme                     # output Markdown table\n"
            "  striat model --np-id gpt2-small/6-res-jb    # process by Neuronpedia ID\n"
            "  striat model --np-id gemma-2-2b/12-gemmascope-res-16k --device cuda\n"
            "  striat model --model gpt2-small \\            # explicit params\n"
            "    --layer 6-res-jb \\\n"
            "    --sae-release gpt2-small-res-jb \\\n"
            "    --sae-hook blocks.6.hook_resid_pre\n"
            "  striat batch --np-ids 'gpt2-small/6-res-jb,gpt2-small/8-res-jb'\n"
            "  striat circuits --batch-defaults             # generate default circuits\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # ── demo ──
    p_demo = sub.add_parser("demo", help="Run the GPT-2 Small demo end-to-end")
    p_demo.add_argument("--port", type=int, help="Frontend port (default: auto-detect from 5173)")
    p_demo.add_argument("--regenerate", action="store_true", help="Regenerate data even if cached")
    p_demo.set_defaults(func=cmd_demo)

    # ── discover ──
    p_discover = sub.add_parser("discover", help="Discover available models from SAELens + Neuronpedia")
    p_discover.add_argument("--sae-types", help="Filter by SAE type: res, mlp, att, transcoder (comma-separated)")
    p_discover.add_argument("--families", help="Filter by model family: gpt2, gemma2, gemma3, llama, pythia, mistral, qwen (comma-separated)")
    p_discover.add_argument("--probe-s3", action="store_true", help="Probe S3 for batch counts (slow but accurate)")
    p_discover.add_argument("--include-no-neuronpedia", action="store_true", help="Include models without Neuronpedia data")
    p_discover.add_argument("--output-json", help="Save full catalog to JSON file")
    p_discover.add_argument("--readme", action="store_true", help="Output Markdown table for README")
    p_discover.set_defaults(func=cmd_discover)

    # ── model ──
    p_model = sub.add_parser(
        "model",
        help="Generate data for any SAELens-compatible model",
        description=(
            "Process a single model. Provide either a Neuronpedia ID (--np-id) for\n"
            "auto-resolution, or explicit SAELens parameters.\n\n"
            "Neuronpedia ID mode (recommended):\n"
            "  striat model --np-id gpt2-small/6-res-jb\n\n"
            "Explicit mode (for models not in registry):\n"
            "  striat model --model gpt2-small --layer 6-res-jb \\\n"
            "    --sae-release gpt2-small-res-jb --sae-hook blocks.6.hook_resid_pre"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Neuronpedia ID shorthand (preferred path)
    p_model.add_argument("--np-id", help="Neuronpedia ID (e.g. 'gpt2-small/6-res-jb'). Auto-resolves all params.")
    # Explicit SAELens parameters (fallback path)
    p_model.add_argument("--model", help="Model ID (e.g., gpt2-small, gemma-2b)")
    p_model.add_argument("--layer", help="Layer identifier (e.g., 6-res-jb)")
    p_model.add_argument("--sae-release", help="SAELens release name")
    p_model.add_argument("--sae-hook", help="SAELens hook point (e.g., blocks.6.hook_resid_pre)")
    # Batch configuration
    p_model.add_argument("--num-batches", type=int, default=24, help="S3 batch file count (default: 24)")
    p_model.add_argument("--features-per-batch", type=int, default=1024, help="Features per batch (default: 1024)")
    # Device and output
    p_model.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps, or cpu (default: auto)")
    p_model.add_argument("--json-export", action="store_true", help="Export JSON only, skip frontend launch instructions")
    p_model.add_argument(
        "--include-semantics", action="store_true", default=False,
        help="Include semantic labels for non-public-tier models. "
             "WARNING: Semantic labels for frontier models may contain safety-relevant "
             "feature interpretations. Do not publish without institutional review.",
    )
    p_model.set_defaults(func=cmd_model)

    # ── batch ──
    p_batch = sub.add_parser("batch", help="Process multiple models sequentially")
    p_batch.add_argument("--catalog", help="Path to discovery catalog JSON")
    p_batch.add_argument("--np-ids", help="Comma-separated Neuronpedia IDs")
    p_batch.add_argument("--device", default="auto", help="Torch device (default: auto)")
    p_batch.add_argument("--num-batches", type=int, default=24, help="S3 batch file count per model (default: 24)")
    p_batch.add_argument("--features-per-batch", type=int, default=1024, help="Features per batch (default: 1024)")
    p_batch.add_argument("--force", action="store_true", help="Reprocess models even if output exists")
    p_batch.add_argument("--continue-on-error", action="store_true", help="Continue to next model on failure")
    p_batch.set_defaults(func=cmd_batch)

    # ── circuits ──
    p_circuits = sub.add_parser(
        "circuits",
        help="Generate circuit data (co-activation or similarity)",
        add_help=False,  # let generate_circuits handle its own --help
    )
    p_circuits.set_defaults(func=cmd_circuits)

    # Parse known args — circuits subcommand passes everything through
    args, unknown = parser.parse_known_args()

    if args.command is None:
        # No subcommand — show banner + help
        from pipeline.banner import print_banner
        print_banner()
        parser.print_help()
        sys.exit(0)

    if args.command == "circuits":
        args.circuit_args = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\n\n  👋 Interrupted. See you next time.\n\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
