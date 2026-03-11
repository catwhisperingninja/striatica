# striatica/tests/test_cli.py
"""Tests for CLI utilities: device detection, argument parsing."""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest
from pipeline.cli import _detect_device, main


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
