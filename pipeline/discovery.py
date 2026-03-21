# striatica/pipeline/discovery.py
"""Discover available SAEs from SAELens registry + Neuronpedia availability.

This module pulls the authoritative SAELens pretrained registry and cross-
references with Neuronpedia S3 batch availability to produce a catalog of
models that Striatica can fully process (decoder weights + feature metadata
+ explanations).

Usage:
    from pipeline.discovery import discover_models, ModelInfo
    catalog = discover_models()
    for model in catalog:
        print(model.display_name, model.num_features, model.device_recommendation)
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from pipeline.config import S3_BASE


# ── Data Structures ────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """A fully-resolved model configuration that Striatica can process."""

    # SAELens identifiers
    sae_release: str          # e.g. "gpt2-small-res-jb"
    sae_id: str               # e.g. "blocks.6.hook_resid_pre"
    model: str                # e.g. "gpt2-small" or "gemma-2-2b"
    repo_id: str              # HuggingFace repo ID

    # Neuronpedia identifiers (for S3 downloads)
    neuronpedia_id: str       # e.g. "gpt2-small/6-res-jb"
    np_model_id: str          # e.g. "gpt2-small"
    np_layer: str             # e.g. "6-res-jb"

    # SAE metadata
    conversion_func: str      # e.g. "gemma_2", "default"
    expected_var_explained: Optional[float] = None
    expected_l0: Optional[float] = None

    # Derived / computed fields
    num_features: Optional[int] = None       # estimated from width or probed
    hidden_dim: Optional[int] = None
    num_batches: Optional[int] = None        # S3 batch count (probed)

    # Classification
    model_family: str = ""       # "gpt2", "gemma2", "gemma3", "llama", "pythia", etc.
    sae_type: str = "res"        # "res", "mlp", "att", "transcoder"
    width_label: str = ""        # "16k", "65k", "131k" etc. (Gemma Scope style)
    is_canonical: bool = False   # Gemma Scope canonical variant
    hook_location: str = ""      # "pre" or "post"
    layer_num: Optional[int] = None

    # Hardware recommendation
    device_recommendation: str = "cpu"   # "cpu", "gpu_recommended", "gpu_required"
    intensity: str = "light"             # "light", "medium", "heavy", "extreme"

    @property
    def display_name(self) -> str:
        """Human-readable name for display in tables and CLI."""
        parts = [self.model]
        if self.layer_num is not None:
            parts.append(f"L{self.layer_num}")
        if self.sae_type != "res":
            parts.append(self.sae_type.upper())
        if self.width_label:
            parts.append(self.width_label)
        if self.is_canonical:
            parts.append("canonical")
        return " / ".join(parts)

    @property
    def striat_model_id(self) -> str:
        """The model_id used in Striatica file naming."""
        return self.np_model_id

    @property
    def striat_layer(self) -> str:
        """The layer string used in Striatica file naming."""
        return self.np_layer

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        d = asdict(self)
        d["display_name"] = self.display_name
        d["striat_model_id"] = self.striat_model_id
        d["striat_layer"] = self.striat_layer
        return d


# ── Registry Access ────────────────────────────────────────────────────

def _load_saelens_registry() -> dict:
    """Pull the pretrained SAEs directory from SAELens.

    Returns the raw registry dict: {release_key: {release, repo_id, model,
    conversion_func, sae_ids, neuronpedia_id, expected_var_explained, ...}}

    Handles multiple SAELens versions — the function moved between releases:
      >=6.x  : sae_lens.loading.pretrained_saes_directory
      older  : sae_lens.toolkit.pretrained_saes
    """
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:
        try:
            from sae_lens.toolkit.pretrained_saes import get_pretrained_saes_directory
        except ImportError:
            raise ImportError(
                "Cannot find get_pretrained_saes_directory in sae-lens. "
                "Ensure sae-lens >= 6.0.0 is installed: poetry install --extras ml"
            )

    return get_pretrained_saes_directory()


def _parse_model_family(model: str, conversion_func: str) -> str:
    """Determine model family from model name and conversion function."""
    model_lower = model.lower()
    if "gpt2" in model_lower or "gpt-2" in model_lower:
        return "gpt2"
    if "gemma-3" in model_lower or conversion_func == "gemma_3":
        return "gemma3"
    if "gemma-2" in model_lower or "gemma-2b" in model_lower or conversion_func == "gemma_2":
        return "gemma2"
    if "gemma" in model_lower:
        return "gemma"
    if "llama" in model_lower:
        return "llama"
    if "pythia" in model_lower:
        return "pythia"
    if "mistral" in model_lower:
        return "mistral"
    if "qwen" in model_lower:
        return "qwen"
    if "deepseek" in model_lower:
        return "deepseek"
    return "other"


def _parse_sae_type(release: str, sae_id: str) -> str:
    """Determine SAE type: res, mlp, att, or transcoder."""
    combined = f"{release} {sae_id}".lower()
    if "transcoder" in combined:
        return "transcoder"
    if "-mlp" in combined or "mlp" in combined:
        return "mlp"
    if "-att" in combined or "attn" in combined or "hook_z" in combined:
        return "att"
    return "res"


def _parse_hook_location(sae_id: str) -> str:
    """Determine hook location: pre, post, or unknown."""
    if "resid_pre" in sae_id or "hook_resid_pre" in sae_id:
        return "pre"
    if "resid_post" in sae_id or "hook_resid_post" in sae_id:
        return "post"
    return ""


def _parse_layer_num(sae_id: str) -> Optional[int]:
    """Extract layer number from hook point string."""
    import re
    # Pattern: blocks.N.hook or layer_N/ or lNr or LNR
    patterns = [
        r"blocks\.(\d+)\.",
        r"layer_(\d+)",
        r"^l(\d+)[arm]",
        r"^L(\d+)[ARM]",
    ]
    for pat in patterns:
        m = re.search(pat, sae_id)
        if m:
            return int(m.group(1))
    return None


def _parse_width_label(sae_id: str) -> str:
    """Extract width label (e.g. '16k', '65k') from sae_id."""
    import re
    m = re.search(r"width[_-]?(\d+k)", sae_id, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(\d+k)-", sae_id)
    if m:
        return m.group(1)
    return ""


def _is_canonical(sae_id: str) -> bool:
    """Check if this is a Gemma Scope canonical variant."""
    return "canonical" in sae_id.lower()


def _estimate_features_from_width(width_label: str) -> Optional[int]:
    """Estimate feature count from width label."""
    mapping = {
        "1k": 1024,
        "4k": 4096,
        "16k": 16384,
        "32k": 32768,
        "65k": 65536,
        "131k": 131072,
        "262k": 262144,
        "524k": 524288,
        "1m": 1048576,
    }
    return mapping.get(width_label.lower())


def _classify_hardware(model: str, num_features: Optional[int], hidden_dim: Optional[int]) -> tuple[str, str]:
    """Classify hardware requirements and intensity.

    Returns (device_recommendation, intensity).
    """
    model_lower = model.lower()

    # Rough sizing based on model name
    if "70b" in model_lower or "65b" in model_lower:
        return "gpu_required", "extreme"
    if "27b" in model_lower:
        return "gpu_required", "heavy"
    if "14b" in model_lower or "9b" in model_lower:
        return "gpu_recommended", "heavy"
    if "8b" in model_lower or "7b" in model_lower:
        return "gpu_recommended", "medium"
    if "4b" in model_lower or "3b" in model_lower or "2.8b" in model_lower:
        return "cpu", "medium"
    if "2b" in model_lower or "1.4b" in model_lower or "1b" in model_lower:
        return "cpu", "medium"

    # Feature count heuristic
    if num_features:
        if num_features > 131072:
            return "gpu_recommended", "heavy"
        if num_features > 65536:
            return "cpu", "medium"

    return "cpu", "light"


# ── S3 Batch Probing ──────────────────────────────────────────────────

def _probe_s3_batch_count(np_model_id: str, np_layer: str, max_probe: int = 512) -> Optional[int]:
    """Probe Neuronpedia S3 to find how many feature batches exist.

    Uses binary search: try batch-0, then double until 404, then binary search.
    Returns the batch count, or None if batch-0 doesn't exist.
    """
    def _batch_exists(idx: int) -> bool:
        url = f"{S3_BASE}/v1/{np_model_id}/{np_layer}/features/batch-{idx}.jsonl.gz"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False

    # Check batch-0 exists at all
    if not _batch_exists(0):
        return None

    # Exponential search for upper bound
    upper = 1
    while upper < max_probe and _batch_exists(upper):
        upper *= 2
    upper = min(upper, max_probe)

    # Binary search for exact count
    lo, hi = upper // 2, upper
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _batch_exists(mid):
            lo = mid
        else:
            hi = mid - 1

    return lo + 1  # count = highest valid index + 1


# ── Main Discovery Function ───────────────────────────────────────────

def discover_models(
    probe_s3: bool = False,
    sae_types: Optional[list[str]] = None,
    model_families: Optional[list[str]] = None,
    require_neuronpedia: bool = True,
) -> list[ModelInfo]:
    """Discover all available models from SAELens registry.

    Args:
        probe_s3: If True, probe S3 for batch counts (slow but accurate).
        sae_types: Filter by SAE type (e.g. ["res"] for residual stream only).
        model_families: Filter by model family (e.g. ["gpt2", "gemma2"]).
        require_neuronpedia: If True, only return models with Neuronpedia data.

    Returns:
        List of ModelInfo objects, sorted by model family and layer.
    """
    registry = _load_saelens_registry()
    catalog: list[ModelInfo] = []

    for release_key, entry in registry.items():
        # entry is a PretrainedSAELookup dataclass (not a dict)
        release = getattr(entry, "release", release_key)
        model = getattr(entry, "model", "")
        repo_id = getattr(entry, "repo_id", "")
        conversion_func = getattr(entry, "conversion_func", "default")
        saes_map = getattr(entry, "saes_map", None) or {}
        np_ids = getattr(entry, "neuronpedia_id", None) or {}
        var_explained = getattr(entry, "expected_var_explained", None) or {}
        l0_values = getattr(entry, "expected_l0", None) or {}

        for sae_id in saes_map:
            # Get Neuronpedia ID for this specific sae_id
            np_id = np_ids.get(sae_id) if isinstance(np_ids, dict) else None

            if require_neuronpedia and not np_id:
                continue

            # Parse Neuronpedia ID into model_id / layer
            if np_id:
                parts = np_id.split("/", 1)
                np_model_id = parts[0]
                np_layer = parts[1] if len(parts) > 1 else ""
            else:
                np_model_id = ""
                np_layer = ""

            # Parse metadata
            family = _parse_model_family(model, conversion_func)
            sae_type = _parse_sae_type(release, sae_id)
            hook_loc = _parse_hook_location(sae_id)
            layer_num = _parse_layer_num(sae_id)
            width = _parse_width_label(sae_id)
            canonical = _is_canonical(sae_id)
            num_features = _estimate_features_from_width(width)

            # Apply filters
            if sae_types and sae_type not in sae_types:
                continue
            if model_families and family not in model_families:
                continue

            # Hardware classification
            device_rec, intensity = _classify_hardware(model, num_features, None)

            info = ModelInfo(
                sae_release=release,
                sae_id=sae_id,
                model=model,
                repo_id=repo_id,
                neuronpedia_id=np_id or "",
                np_model_id=np_model_id,
                np_layer=np_layer,
                conversion_func=conversion_func,
                expected_var_explained=var_explained.get(sae_id),
                expected_l0=l0_values.get(sae_id),
                num_features=num_features,
                model_family=family,
                sae_type=sae_type,
                width_label=width,
                is_canonical=canonical,
                hook_location=hook_loc,
                layer_num=layer_num,
                device_recommendation=device_rec,
                intensity=intensity,
            )

            # Optionally probe S3 for batch count
            if probe_s3 and np_model_id and np_layer:
                info.num_batches = _probe_s3_batch_count(np_model_id, np_layer)

            catalog.append(info)

    # Sort: family → model → layer_num → sae_type → width
    catalog.sort(key=lambda m: (
        m.model_family,
        m.model,
        m.layer_num or 0,
        m.sae_type,
        m.width_label,
    ))

    return catalog


# ── Catalog Persistence ────────────────────────────────────────────────

def save_catalog(catalog: list[ModelInfo], path: Path) -> None:
    """Save discovery catalog to JSON."""
    data = [m.to_dict() for m in catalog]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(catalog)} models to {path}", file=sys.stderr)


def load_catalog(path: Path) -> list[ModelInfo]:
    """Load discovery catalog from JSON."""
    with open(path) as f:
        data = json.load(f)
    catalog = []
    for d in data:
        # Remove computed properties that aren't constructor args
        d.pop("display_name", None)
        d.pop("striat_model_id", None)
        d.pop("striat_layer", None)
        catalog.append(ModelInfo(**d))
    return catalog


# ── Summary / Table Generation ─────────────────────────────────────────

def catalog_summary(catalog: list[ModelInfo]) -> str:
    """Generate a human-readable summary table of the catalog."""
    lines = []
    lines.append(f"{'Model':<35} {'Layer':>5} {'Type':<6} {'Width':<8} "
                 f"{'Features':>10} {'Device':<16} {'Intensity':<8} {'Neuronpedia ID'}")
    lines.append("─" * 120)

    for m in catalog:
        feat_str = f"{m.num_features:,}" if m.num_features else "?"
        layer_str = str(m.layer_num) if m.layer_num is not None else "?"
        lines.append(
            f"{m.model:<35} {layer_str:>5} {m.sae_type:<6} {m.width_label:<8} "
            f"{feat_str:>10} {m.device_recommendation:<16} {m.intensity:<8} {m.neuronpedia_id}"
        )

    lines.append("─" * 120)
    lines.append(f"Total: {len(catalog)} SAE configurations")

    # Summary by family
    families: dict[str, int] = {}
    for m in catalog:
        families[m.model_family] = families.get(m.model_family, 0) + 1
    fam_str = ", ".join(f"{k}: {v}" for k, v in sorted(families.items()))
    lines.append(f"By family: {fam_str}")

    return "\n".join(lines)


def generate_readme_table(catalog: list[ModelInfo]) -> str:
    """Generate a Markdown table for the README showing available models.

    Groups by model, shows key info: layers available, SAE types, widths,
    device recommendation.
    """
    lines = []
    lines.append("## Available Models")
    lines.append("")
    lines.append(f"*Auto-generated from SAELens registry. {len(catalog)} SAE configurations available.*")
    lines.append("")
    lines.append("| Model | Family | Layers | SAE Types | Widths | Device | Intensity |")
    lines.append("|-------|--------|--------|-----------|--------|--------|-----------|")

    # Group by (model, sae_type)
    from collections import defaultdict
    groups: dict[str, dict] = {}
    for m in catalog:
        key = m.model
        if key not in groups:
            groups[key] = {
                "family": m.model_family,
                "layers": set(),
                "types": set(),
                "widths": set(),
                "device": m.device_recommendation,
                "intensity": m.intensity,
            }
        g = groups[key]
        if m.layer_num is not None:
            g["layers"].add(m.layer_num)
        g["types"].add(m.sae_type)
        if m.width_label:
            g["widths"].add(m.width_label)

    for model, g in sorted(groups.items(), key=lambda x: (x[1]["family"], x[0])):
        layers = sorted(g["layers"])
        if len(layers) > 3:
            layer_str = f"{min(layers)}-{max(layers)} ({len(layers)} layers)"
        else:
            layer_str = ", ".join(str(l) for l in layers) or "—"
        types_str = ", ".join(sorted(g["types"]))
        widths_str = ", ".join(sorted(g["widths"], key=lambda w: int(w.replace("k", "000").replace("m", "000000")) if w.replace("k", "").replace("m", "").isdigit() else 0)) or "—"
        device_emoji = {"cpu": "🟢 CPU", "gpu_recommended": "🟡 GPU rec.", "gpu_required": "🔴 GPU req."}.get(g["device"], g["device"])
        lines.append(f"| {model} | {g['family']} | {layer_str} | {types_str} | {widths_str} | {device_emoji} | {g['intensity']} |")

    return "\n".join(lines)
