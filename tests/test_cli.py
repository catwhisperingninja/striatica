# striatica/tests/test_cli.py
"""Tests for CLI utilities: device detection, argument parsing."""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest
from pipeline.cli import _detect_device, cmd_model, _resolve_transcoder, _run_process_pipeline
from pipeline.config import TranscoderConfig


class TestDetectDevice:
    """Tests for _detect_device() auto-detection and explicit passthrough."""

    def test_explicit_cuda_passthrough(self):
        """Explicit device string should be returned as-is."""
        assert _detect_device("cuda") == "cuda"

    def test_explicit_mps_passthrough(self):
        assert _detect_device("mps") == "mps"

    def test_explicit_cpu_passthrough(self):
        assert _detect_device("cpu") == "cpu"

    def test_auto_with_cuda_available(self):
        """auto should resolve to cuda when torch.cuda.is_available()."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # Patch torch and all submodules that might be cached in sys.modules
        patches = {"torch": mock_torch, "torch.backends": mock_torch.backends}
        with patch.dict("sys.modules", patches):
            assert _detect_device("auto") == "cuda"

    def test_auto_with_mps_available(self):
        """auto should fall through to mps if cuda is unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        patches = {"torch": mock_torch, "torch.backends": mock_torch.backends}
        with patch.dict("sys.modules", patches):
            assert _detect_device("auto") == "mps"

    def test_auto_with_nothing_available(self):
        """auto should fall back to cpu when no GPU backend is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        patches = {"torch": mock_torch, "torch.backends": mock_torch.backends}
        with patch.dict("sys.modules", patches):
            assert _detect_device("auto") == "cpu"

    def test_auto_without_torch_installed(self):
        """auto should fall back to cpu if torch is not importable.

        Setting a module to None in sys.modules causes ImportError on import.
        We also block submodule keys to prevent stale references from leaking
        into later tests (e.g., similarity circuit imports).
        """
        block = {k: None for k in list(sys.modules) if k == "torch" or k.startswith("torch.")}
        block["torch"] = None  # ensure base key present even if torch was never imported
        with patch.dict("sys.modules", block):
            assert _detect_device("auto") == "cpu"


class TestModelArgparse:
    """Verify that --device, --json-export, and --include-semantics flags parse."""

    def _parse_model_args(self, extra_args: list[str] | None = None) -> argparse.Namespace:
        """Run the argparser with model subcommand + required flags."""
        base = [
            "model",
            "--model", "gemma-2b",
            "--layer", "12-res-jb",
            "--sae-release", "gemma-2b-res-jb",
            "--sae-hook", "blocks.12.hook_resid_post",
        ]
        if extra_args:
            base.extend(extra_args)

        # Mirror the real CLI parser structure — keep in sync with pipeline/cli.py
        parser = argparse.ArgumentParser(prog="striat")
        sub = parser.add_subparsers(dest="command")
        p_model = sub.add_parser("model")
        p_model.add_argument("--model", required=True)
        p_model.add_argument("--layer", required=True)
        p_model.add_argument("--sae-release", required=True)
        p_model.add_argument("--sae-hook", required=True)
        p_model.add_argument("--num-batches", type=int, default=24)
        p_model.add_argument("--features-per-batch", type=int, default=1024)
        p_model.add_argument("--device", default="auto")
        p_model.add_argument("--json-export", action="store_true")
        p_model.add_argument("--pca-dim", default="auto")
        p_model.add_argument("--include-semantics", action="store_true", default=False)
        return parser.parse_args(base)

    def test_device_default_is_auto(self):
        args = self._parse_model_args()
        assert args.device == "auto"

    def test_device_explicit_cuda(self):
        args = self._parse_model_args(["--device", "cuda"])
        assert args.device == "cuda"

    def test_json_export_default_false(self):
        args = self._parse_model_args()
        assert args.json_export is False

    def test_json_export_flag(self):
        args = self._parse_model_args(["--json-export"])
        assert args.json_export is True

    def test_device_and_json_export_together(self):
        args = self._parse_model_args(["--device", "cpu", "--json-export"])
        assert args.device == "cpu"
        assert args.json_export is True

    def test_include_semantics_default_false(self):
        """--include-semantics defaults to False (safety: redact by default)."""
        args = self._parse_model_args()
        assert args.include_semantics is False

    def test_include_semantics_flag(self):
        """--include-semantics can be explicitly set."""
        args = self._parse_model_args(["--include-semantics"])
        assert args.include_semantics is True

    def test_include_semantics_with_json_export(self):
        """Both flags can be combined for remote compute with semantics."""
        args = self._parse_model_args(["--include-semantics", "--json-export"])
        assert args.include_semantics is True
        assert args.json_export is True

    def test_pca_dim_default_is_auto(self):
        """--pca-dim defaults to 'auto' for adaptive selection."""
        args = self._parse_model_args()
        assert args.pca_dim == "auto"

    def test_pca_dim_explicit_int(self):
        """--pca-dim accepts an explicit integer value."""
        args = self._parse_model_args(["--pca-dim", "300"])
        assert args.pca_dim == "300"  # argparse returns string, caller converts to int

    def test_pca_dim_explicit_50(self):
        """--pca-dim=50 should be parseable for backwards compat."""
        args = self._parse_model_args(["--pca-dim", "50"])
        assert args.pca_dim == "50"


class TestTranscoderCLI:
    """Tests for transcoder-specific CLI behavior."""

    def _transcoder_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            transcoder="gemma-2-2b/12/604",
            transcoder_repo="google/gemma-scope-2b-pt-transcoders",
            transcoder_width="width_16k",
            np_id=None,
            sae_release=None,
            sae_hook=None,
            model=None,
            layer=None,
            num_batches=24,
            features_per_batch=1024,
            device="cpu",
            include_semantics=False,
            json_export=True,
            pca_dim="auto",
        )

    def test_cmd_model_uses_transcoder_resolution_and_pipeline(self, monkeypatch):
        """cmd_model should resolve --transcoder and run the pipeline with that config."""
        args = self._transcoder_args()
        cfg = TranscoderConfig(model_id="gemma-2-2b", layer=12, l0_variant=604)
        calls = {}

        monkeypatch.setattr("pipeline.cli._detect_device", lambda *_: "cpu")
        monkeypatch.setattr("pipeline.cli._resolve_transcoder", lambda _args: cfg)
        monkeypatch.setattr(
            "pipeline.cli._run_process_pipeline",
            lambda c, data_dir, device, redact_semantics, pca_dim="auto": calls.update(
                cfg=c, data_dir=data_dir, device=device, redact_semantics=redact_semantics, pca_dim=pca_dim
            ),
        )

        cmd_model(args)

        assert calls["cfg"] == cfg
        assert calls["device"] == "cpu"
        assert calls["redact_semantics"] is True  # non-public model defaults to redaction

    def test_resolve_transcoder_parses_valid_spec(self):
        """_resolve_transcoder should parse model/layer/l0 into TranscoderConfig."""
        args = argparse.Namespace(
            transcoder="gemma-2-2b/12/604",
            transcoder_repo="org/custom-repo",
            transcoder_width="width_32k",
        )
        cfg = _resolve_transcoder(args)
        assert isinstance(cfg, TranscoderConfig)
        assert cfg.model_id == "gemma-2-2b"
        assert cfg.layer == 12
        assert cfg.l0_variant == 604
        assert cfg.repo_id == "org/custom-repo"
        assert cfg.width == "width_32k"

    def test_resolve_transcoder_rejects_invalid_format(self):
        """_resolve_transcoder should exit for malformed transcoder spec."""
        args = argparse.Namespace(
            transcoder="gemma-2-2b/12",
            transcoder_repo="google/gemma-scope-2b-pt-transcoders",
            transcoder_width="width_16k",
        )
        with pytest.raises(SystemExit) as exc:
            _resolve_transcoder(args)
        assert exc.value.code == 1

    @pytest.mark.slow
    def test_run_process_pipeline_dispatches_transcoder_loader(self, monkeypatch, tmp_path):
        """_run_process_pipeline should call load_transcoder_vectors for TranscoderConfig."""
        cfg = TranscoderConfig(model_id="gemma-2-2b", layer=12, l0_variant=604)

        called = {"load_transcoder_vectors": False}

        monkeypatch.setattr("pipeline.cli.OUTPUT_DIR", tmp_path / "out")
        monkeypatch.setattr("pipeline.vectors.load_transcoder_vectors", lambda **kwargs: (
            called.__setitem__("load_transcoder_vectors", True),
            kwargs,
            __import__("numpy").ones((3, 2)),
        )[2])
        monkeypatch.setattr("pipeline.reduce.reduce_to_3d", lambda _v, **kwargs: (__import__("numpy").zeros((3, 3)), 0.5) if kwargs.get("return_pca_variance") else __import__("numpy").zeros((3, 3)))
        monkeypatch.setattr("pipeline.cluster.cluster_points", lambda _coords: __import__("numpy").array([0, 0, 1]))
        monkeypatch.setattr("pipeline.local_dim.estimate_local_dim", lambda _v, method: __import__("numpy").array([1.0, 1.1, 1.2]))
        monkeypatch.setattr(
            "pipeline.local_dim.estimate_local_dim_vgt",
            lambda _v, return_curves: (__import__("numpy").array([1.0, 1.1, 1.2]), {0: [1.0], 1: [1.0], 2: [1.0]}),
        )
        monkeypatch.setattr("pipeline.prepare.prepare_json", lambda *args, **kwargs: None)

        # Skip validation — this test verifies dispatch, not validation
        class _FakeReport:
            passed = True
            def print_scorecard(self): pass
        monkeypatch.setattr("pipeline.validate.validate_level1_arrays", lambda *a, **kw: _FakeReport())
        monkeypatch.setattr("pipeline.validate.validate_level2", lambda *a, **kw: _FakeReport())
        monkeypatch.setattr("pipeline.validate.write_validation_sidecar", lambda *a, **kw: None)

        _run_process_pipeline(cfg, tmp_path / "data", device="cpu", redact_semantics=True)

        assert called["load_transcoder_vectors"] is True
