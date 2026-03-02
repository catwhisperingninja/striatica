# striatica/pipeline/download.py
"""Download feature data from Neuronpedia S3 bulk exports."""

import gzip
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.config import S3_BASE
from pipeline.banner import ProgressBar


def _download_batch(model_id: str, layer: str, category: str, batch_idx: int) -> list[dict]:
    """Download and decompress a single batch file from S3."""
    url = f"{S3_BASE}/v1/{model_id}/{layer}/{category}/batch-{batch_idx}.jsonl.gz"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = gzip.decompress(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    except gzip.BadGzipFile as e:
        raise RuntimeError(f"Corrupted gzip data from {url}: {e}") from e
    lines = data.decode().strip().split("\n")
    results = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping malformed JSON at line {i} in batch-{batch_idx}: {e}", file=sys.stderr)
    return results


def download_features(
    model_id: str,
    layer: str,
    batch_indices: list[int] | None = None,
    output_path: Path | None = None,
) -> int:
    """Download feature metadata from Neuronpedia S3.

    Args:
        model_id: e.g. "gpt2-small"
        layer: e.g. "6-res-jb"
        batch_indices: Which batches to download (default: all 24)
        output_path: Where to save JSONL output

    Returns:
        Number of features downloaded.
    """
    if batch_indices is None:
        batch_indices = list(range(24))

    all_features = []
    total = len(batch_indices)
    bar = ProgressBar(total=total, label="Features", emoji="🛰️")
    for i, idx in enumerate(batch_indices):
        batch = _download_batch(model_id, layer, "features", idx)
        all_features.extend(batch)
        bar.update(i + 1, suffix=f"({len(all_features):,} features)")
    bar.finish()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for feat in all_features:
                f.write(json.dumps(feat) + "\n")

    return len(all_features)


def download_explanations(
    model_id: str,
    layer: str,
    batch_indices: list[int] | None = None,
    output_path: Path | None = None,
) -> int:
    """Download feature explanations from Neuronpedia S3.

    Returns:
        Number of explanations downloaded.
    """
    if batch_indices is None:
        batch_indices = list(range(24))

    all_explanations = []
    total = len(batch_indices)
    bar = ProgressBar(total=total, label="Explanations", emoji="🛰️")
    for i, idx in enumerate(batch_indices):
        batch = _download_batch(model_id, layer, "explanations", idx)
        all_explanations.extend(batch)
        bar.update(i + 1, suffix=f"({len(all_explanations):,} entries)")
    bar.finish()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for expl in all_explanations:
                f.write(json.dumps(expl) + "\n")

    return len(all_explanations)
