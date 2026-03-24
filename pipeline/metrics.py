"""
Metrics emission for Striatica pipeline runs.

Two modes:
1. JSON sidecar (default, zero deps) — writes run_metrics.json alongside output
2. Prometheus push (optional) — pushes to Prometheus Pushgateway or Grafana OTLP

Usage in pipeline code:
    from pipeline.metrics import PipelineMetrics

    m = PipelineMetrics(run_id="abc123", model_id="gemma-2-2b")
    m.set_feature_count(16384)

    m.step_start("pca")
    # ... do PCA ...
    m.step_end("pca")

    m.step_start("umap")
    m.set_umap_progress(epoch=50, total=200, loss=1.23)
    # ... UMAP runs ...
    m.step_end("umap")

    m.set_cluster_stats(cluster_count=12, uncategorized_fraction=0.35, sizes={0: 500, 1: 300, ...})
    m.set_vgt_stats(mean=15.2, values=[...])

    m.finalize(output_path="path/to/output.json")
    # -> writes run_metrics.json next to it
    # -> optionally pushes to Prometheus
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_dep_versions() -> dict[str, str]:
    """Collect installed versions of key dependencies."""
    versions = {}
    packages = [
        "numpy", "scipy", "sklearn", "umap", "hdbscan",
        "numba", "pynndescent", "torch", "sae_lens",
        "transformer_lens", "transformers", "huggingface_hub",
    ]
    # Mapping of import names to pip package names for display
    display_names = {
        "sklearn": "scikit-learn",
        "umap": "umap-learn",
        "sae_lens": "sae-lens",
        "transformer_lens": "transformer-lens",
        "huggingface_hub": "huggingface-hub",
    }
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            name = display_names.get(pkg, pkg)
            versions[name] = ver
        except ImportError:
            pass
    return versions


def _get_gpu_info() -> dict[str, Any]:
    """Get GPU info if torch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_bytes": torch.cuda.get_device_properties(0).total_mem,
                "cuda_version": torch.version.cuda or "unknown",
                "gpu_count": torch.cuda.device_count(),
            }
    except Exception:
        pass
    return {"gpu_name": "none", "gpu_memory_total_bytes": 0}


@dataclass
class StepTiming:
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class PipelineMetrics:
    """Collects and emits pipeline run metrics."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    pipeline_mode: str = "sae"  # "sae" or "transcoder"
    pipeline_version: str = "0.3.0"

    # Provider info (set by caller or auto-detected)
    provider: str = ""
    instance_type: str = ""

    # Timing
    _steps: dict[str, StepTiming] = field(default_factory=dict)
    _run_start: float = field(default_factory=time.time)

    # Counts
    feature_count: int = 0

    # UMAP
    umap_random_state: int = 42
    umap_epochs: list[dict[str, float]] = field(default_factory=list)

    # VGT
    vgt_features_completed: int = 0
    vgt_values: list[float] = field(default_factory=list)

    # HDBSCAN
    cluster_count: int = 0
    uncategorized_fraction: float = 0.0
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    # PCA
    pca_variance_ratios: list[float] = field(default_factory=list)

    # Correlation
    vgt_activation_pearson_r: float = 0.0

    # Memory snapshots
    _memory_snapshots: list[dict] = field(default_factory=list)

    # --- Step timing ---

    def step_start(self, name: str) -> None:
        self._steps[name] = StepTiming(name=name, start_time=time.time())

    def step_end(self, name: str) -> None:
        if name in self._steps:
            step = self._steps[name]
            step.end_time = time.time()
            step.duration_seconds = step.end_time - step.start_time

    # --- Setters ---

    def set_feature_count(self, count: int) -> None:
        self.feature_count = count

    def set_umap_progress(self, epoch: int, total: int, loss: float) -> None:
        self.umap_epochs.append({
            "epoch": epoch,
            "total": total,
            "loss": loss,
            "timestamp": time.time(),
        })

    def set_vgt_progress(self, completed: int) -> None:
        self.vgt_features_completed = completed

    def set_vgt_stats(self, mean: float, values: list[float] | None = None) -> None:
        self.vgt_values = values or []
        # Compute correlation if we have activation data
        # (caller can set vgt_activation_pearson_r directly)

    def set_cluster_stats(
        self,
        cluster_count: int,
        uncategorized_fraction: float,
        sizes: dict[int, int] | None = None,
    ) -> None:
        self.cluster_count = cluster_count
        self.uncategorized_fraction = uncategorized_fraction
        self.cluster_sizes = sizes or {}

    def set_pca_variance(self, ratios: list[float]) -> None:
        self.pca_variance_ratios = ratios

    def snapshot_memory(self) -> None:
        """Capture current memory usage."""
        import os
        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "rss_bytes": 0,
        }
        # RSS from /proc on Linux
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        snapshot["rss_bytes"] = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass
        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                snapshot["gpu_memory_used_bytes"] = torch.cuda.memory_allocated(0)
                snapshot["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(0)
                snapshot["gpu_utilization_percent"] = -1  # needs nvidia-smi
        except Exception:
            pass
        self._memory_snapshots.append(snapshot)

    # --- Output ---

    def finalize(
        self,
        output_path: str | Path | None = None,
        reference_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Finalize metrics and write JSON sidecar.

        Args:
            output_path: Path to the pipeline output JSON. Hash will be computed.
            reference_path: Path to a reference dataset for drift comparison.

        Returns:
            The full metrics dict.
        """
        total_duration = time.time() - self._run_start
        output_path = Path(output_path) if output_path else None

        # Compute output hash
        output_sha256 = ""
        if output_path and output_path.exists():
            output_sha256 = _sha256_file(output_path)

        # Compute drift if reference exists
        drift_stats = self._compute_drift(output_path, reference_path)

        # Lockfile hash
        lockfile_sha256 = ""
        lockfile = Path("poetry.lock")
        if lockfile.exists():
            lockfile_sha256 = _sha256_file(lockfile)

        metrics: dict[str, Any] = {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "pipeline_mode": self.pipeline_mode,
            "pipeline_version": self.pipeline_version,
            "provider": self.provider,
            "instance_type": self.instance_type,
            "platform_arch": platform.machine(),
            "python_version": platform.python_version(),
            "timestamp": time.time(),
            "total_duration_seconds": total_duration,
            "feature_count": self.feature_count,
            "output_sha256": output_sha256,
            "lockfile_sha256": lockfile_sha256,
            "umap_random_state": self.umap_random_state,

            # Dependencies
            "dependency_versions": _get_dep_versions(),
            "gpu_info": _get_gpu_info(),

            # Step timings
            "steps": {
                name: {
                    "duration_seconds": step.duration_seconds,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                }
                for name, step in self._steps.items()
            },

            # UMAP convergence
            "umap_convergence": self.umap_epochs[-10:] if self.umap_epochs else [],

            # Cluster stats
            "hdbscan": {
                "cluster_count": self.cluster_count,
                "uncategorized_fraction": self.uncategorized_fraction,
                "cluster_sizes": self.cluster_sizes,
            },

            # PCA
            "pca_variance_ratios": self.pca_variance_ratios,

            # VGT
            "vgt": {
                "mean": float(np.mean(self.vgt_values)) if self.vgt_values else 0.0,
                "std": float(np.std(self.vgt_values)) if self.vgt_values else 0.0,
                "median": float(np.median(self.vgt_values)) if self.vgt_values else 0.0,
                "features_computed": len(self.vgt_values),
            },

            "vgt_activation_pearson_r": self.vgt_activation_pearson_r,

            # Drift
            "drift": drift_stats,

            # Memory
            "memory_snapshots": self._memory_snapshots,

            # Cost (computed by consumer based on provider hourly rate)
            "cost_inputs": {
                "total_duration_seconds": total_duration,
                "provider": self.provider,
                "instance_type": self.instance_type,
            },
        }

        # Write JSON sidecar
        if output_path:
            sidecar_path = output_path.parent / f"{output_path.stem}_metrics.json"
            with open(sidecar_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        # Push to Prometheus if configured
        self._push_prometheus(metrics)

        return metrics

    def _compute_drift(
        self,
        output_path: Path | None,
        reference_path: str | Path | None,
    ) -> dict[str, Any]:
        """Compare positions against a reference dataset."""
        if not output_path or not reference_path:
            return {"status": "no_reference"}

        reference_path = Path(reference_path)
        if not output_path.exists() or not reference_path.exists():
            return {"status": "file_not_found"}

        try:
            with open(output_path) as f:
                new_data = json.load(f)
            with open(reference_path) as f:
                ref_data = json.load(f)

            new_features = {f["index"]: f for f in new_data.get("features", [])}
            ref_features = {f["index"]: f for f in ref_data.get("features", [])}

            common = set(new_features.keys()) & set(ref_features.keys())
            if not common:
                return {"status": "no_common_features"}

            drifts = []
            for idx in common:
                new_pos = new_features[idx].get("pos", [0, 0, 0])
                ref_pos = ref_features[idx].get("pos", [0, 0, 0])
                dist = float(np.linalg.norm(
                    np.array(new_pos) - np.array(ref_pos)
                ))
                drifts.append({"feature_index": idx, "distance": dist})

            drifts.sort(key=lambda x: x["distance"], reverse=True)
            distances = [d["distance"] for d in drifts]

            return {
                "status": "computed",
                "features_compared": len(common),
                "max_drift": float(np.max(distances)),
                "mean_drift": float(np.mean(distances)),
                "p95_drift": float(np.percentile(distances, 95)),
                "features_affected": int(np.sum(np.array(distances) > 0.001)),
                "top_10_drifted": drifts[:10],
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _push_prometheus(self, metrics: dict[str, Any]) -> None:
        """Push metrics to Prometheus Pushgateway if configured."""
        import os
        gateway = os.environ.get("PROMETHEUS_PUSHGATEWAY")
        if not gateway:
            return

        try:
            from prometheus_client import CollectorRegistry, Gauge, Info, push_to_gateway

            registry = CollectorRegistry()
            job_name = "striatica_pipeline"

            # Run info
            info = Info(
                "striatica_pipeline_run",
                "Pipeline run metadata",
                registry=registry,
            )
            info.info({
                "run_id": self.run_id,
                "model_id": self.model_id,
                "pipeline_mode": self.pipeline_mode,
                "pipeline_version": self.pipeline_version,
                "platform_arch": metrics["platform_arch"],
                "output_sha256": metrics["output_sha256"],
                "provider": self.provider,
                "instance_type": self.instance_type,
            })

            # Duration
            g = Gauge(
                "striatica_pipeline_total_duration_seconds",
                "Total pipeline wall time",
                ["model_id", "run_id"],
                registry=registry,
            )
            g.labels(self.model_id, self.run_id).set(
                metrics["total_duration_seconds"]
            )

            # Feature count
            g2 = Gauge(
                "striatica_pipeline_feature_count",
                "Number of features processed",
                ["model_id", "run_id"],
                registry=registry,
            )
            g2.labels(self.model_id, self.run_id).set(self.feature_count)

            # Step durations
            g3 = Gauge(
                "striatica_step_duration_seconds",
                "Duration per pipeline step",
                ["model_id", "run_id", "step"],
                registry=registry,
            )
            for name, step in self._steps.items():
                g3.labels(self.model_id, self.run_id, name).set(
                    step.duration_seconds
                )

            # Cluster stats
            g4 = Gauge(
                "striatica_hdbscan_cluster_count",
                "Number of HDBSCAN clusters",
                ["model_id", "run_id"],
                registry=registry,
            )
            g4.labels(self.model_id, self.run_id).set(self.cluster_count)

            g5 = Gauge(
                "striatica_hdbscan_uncategorized_fraction",
                "Fraction of uncategorized features",
                ["model_id", "run_id"],
                registry=registry,
            )
            g5.labels(self.model_id, self.run_id).set(
                self.uncategorized_fraction
            )

            # VGT
            g6 = Gauge(
                "striatica_vgt_mean",
                "Mean VGT dimension",
                ["model_id", "run_id"],
                registry=registry,
            )
            if self.vgt_values:
                g6.labels(self.model_id, self.run_id).set(
                    float(np.mean(self.vgt_values))
                )

            # Drift
            drift = metrics.get("drift", {})
            if drift.get("status") == "computed":
                g7 = Gauge(
                    "striatica_position_drift_max",
                    "Maximum position drift from reference",
                    ["model_id", "run_id"],
                    registry=registry,
                )
                g7.labels(self.model_id, self.run_id).set(drift["max_drift"])

            # Dep versions
            dep_info = Info(
                "striatica_dep_version",
                "Dependency versions",
                registry=registry,
            )
            dep_info.info(metrics["dependency_versions"])

            push_to_gateway(gateway, job=job_name, registry=registry)

        except ImportError:
            pass  # prometheus_client not installed, silent skip
        except Exception:
            pass  # Don't crash the pipeline over metrics
