#!/usr/bin/env python3
"""striat — CLI entry point for the striatica data pipeline.

Subcommands:
    striat demo       Run the GPT-2 Small demo (generate data + circuits + launch frontend)
    striat model      Bring your own model (generate data for any SAELens-compatible model)
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
    from pipeline.banner import STEP_EMOJIS

    print(f"\n  {STEP_EMOJIS['frontend']}  Launching frontend → http://localhost:{port}")
    print(f"  {'━' * 48}\n")

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
    from pipeline.banner import print_banner, step_header, step_done, step_cached, STEP_EMOJIS
    from pipeline.config import GPT2_SMALL_L6, DATA_DIR

    print_banner()

    cfg = GPT2_SMALL_L6
    dataset_path = OUTPUT_DIR / f"{cfg.model_id}-{cfg.layer}.json"
    circuits_dir = OUTPUT_DIR / "circuits"
    manifest_path = circuits_dir / "manifest.json"

    # Step 1: Generate point cloud data (skip if cached)
    if dataset_path.exists() and not args.regenerate:
        step_cached(dataset_path.name)
        print("     (use --regenerate to rebuild)\n")
    else:
        print(f"  🧬  First run detected — downloading ~2 GB from Neuronpedia S3\n")
        _run_process_pipeline(cfg, DATA_DIR)

    # Step 2: Generate circuits
    if manifest_path.exists() and not args.regenerate:
        n_circuits = len(list(circuits_dir.glob("*.json"))) - 1
        step_cached(f"circuits ({n_circuits} circuits)")
        print("     (use --regenerate to rebuild)\n")
    else:
        if _ask_yes_no("  🔗  Generate default circuits? (5 co-activation + 5 similarity)"):
            from scripts.generate_circuits import generate_batch_defaults
            generate_batch_defaults()
        else:
            print("     Skipping circuit generation.")

    # Step 3: Launch frontend
    port = args.port if args.port is not None else _find_port()
    _launch_frontend(port)


# ── Model (BYO) ─────────────────────────────────────────────────────────

def cmd_model(args: argparse.Namespace) -> None:
    """Generate data for a custom SAELens-compatible model."""
    from pipeline.banner import print_banner, STEP_EMOJIS
    from pipeline.config import SAEConfig, DATA_DIR

    print_banner()

    device = _detect_device(args.device)

    cfg = SAEConfig(
        model_id=args.model,
        layer=args.layer,
        sae_release=args.sae_release,
        sae_hook=args.sae_hook,
        num_batches=args.num_batches,
        features_per_batch=args.features_per_batch,
    )

    print(f"  🧬  Model:       {cfg.model_id}")
    print(f"  📐  Layer:       {cfg.layer}")
    print(f"  🛰️  SAE release: {cfg.sae_release}")
    print(f"  🔗  SAE hook:    {cfg.sae_hook}")
    print(f"  ⚡  Features:    {cfg.num_batches * cfg.features_per_batch:,}")
    print(f"  🖥️  Device:      {device}")
    print()

    _run_process_pipeline(cfg, DATA_DIR, device=device)

    dataset_file = f"{cfg.model_id}-{cfg.layer}.json"
    dataset_path = OUTPUT_DIR / dataset_file

    if args.json_export:
        print(f"\n  ✅  JSON exported: {dataset_path}")
        print(f"  Transfer to local machine:")
        print(f"    scp remote:{dataset_path} ./frontend/public/data/")
        print(f"  Then open: http://localhost:5173/?dataset={dataset_file}")
    else:
        print(f"\n  ✅  Output: {dataset_path}")
        print(f"  Launch the frontend with:")
        print(f"    cd frontend && pnpm dev")
        print(f"  Then open: http://localhost:5173/?dataset={dataset_file}")


def _run_process_pipeline(cfg, data_dir: Path, device: str = "cpu") -> None:
    """Run the full data pipeline for a given SAEConfig."""
    from pipeline.banner import step_header, step_done, step_cached
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
    print(f"     {vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
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
    print(f"     {n_clusters} clusters found")
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
    )
    step_done(time.time() - t0)

    elapsed = time.time() - t_total
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n  {'━' * 48}")
    print(f"  ✅  Complete · {minutes}m {seconds}s")
    print(f"  📁  {output}")


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
            "  striat demo                               # GPT-2 Small demo, launches browser\n"
            "  striat demo --port 8080                   # use a specific port\n"
            "  striat model --model gemma-2b \\             # bring your own model\n"
            "    --layer 12-res-jb \\\n"
            "    --sae-release gemma-2b-res-jb \\\n"
            "    --sae-hook blocks.12.hook_resid_pre \\\n"
            "    --device auto\n"
            "  striat model ... --json-export              # export JSON without frontend\n"
            "  striat circuits --batch-defaults           # generate default circuits\n"
            "  striat circuits --type coactivation \\      # custom circuit\n"
            "    --prompt 'The capital of France is'\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # ── demo ──
    p_demo = sub.add_parser("demo", help="Run the GPT-2 Small demo end-to-end")
    p_demo.add_argument("--port", type=int, help="Frontend port (default: auto-detect from 5173)")
    p_demo.add_argument("--regenerate", action="store_true", help="Regenerate data even if cached")
    p_demo.set_defaults(func=cmd_demo)

    # ── model ──
    p_model = sub.add_parser("model", help="Generate data for any SAELens-compatible model")
    p_model.add_argument("--model", required=True, help="Model ID (e.g., gpt2-small, gemma-2b)")
    p_model.add_argument("--layer", required=True, help="Layer identifier (e.g., 6-res-jb)")
    p_model.add_argument("--sae-release", required=True, help="SAELens release name")
    p_model.add_argument("--sae-hook", required=True, help="SAELens hook point (e.g., blocks.6.hook_resid_pre)")
    p_model.add_argument("--num-batches", type=int, default=24, help="S3 batch file count (default: 24)")
    p_model.add_argument("--features-per-batch", type=int, default=1024, help="Features per batch (default: 1024)")
    p_model.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps, or cpu (default: auto)")
    p_model.add_argument("--json-export", action="store_true", help="Export JSON only, skip frontend launch instructions")
    p_model.set_defaults(func=cmd_model)

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
