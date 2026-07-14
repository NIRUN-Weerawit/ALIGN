#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Weights & Biases utilities for ALIGN training scripts.

Usage:
    from training.wandb_utils import init_wandb, log_metrics

    config = {"lr": 1e-4, "batch_size": 64, "epochs": 50, "dataset": "LIBERO"}
    trainer = init_wandb(project="align-pretrain", name="run-001", config=config)

    # In training loop:
    log_metrics({"loss": 0.5, "cos_vt": 0.8, "epoch": 1}, step=1)

    trainer.finish()

Set WANDB_MODE=disabled to run without W&B (no-op).
"""

import os
import sys
from typing import Any, Dict, Optional

# W&B is optional — import gracefully
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandBTrainer:
    """Wrapper around wandb that gracefully handles disabled mode."""

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None):
        self._enabled = False
        self._run = None

        if not _WANDB_AVAILABLE:
            return

        # Check environment override
        mode = os.environ.get("WANDB_MODE", "").lower()
        if mode in ("disabled", "offline", "dryrun") or mode == "" and not self._check_api_key():
            self._enabled = False
            return

        try:
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                reinit=True,
            )
            self._enabled = True
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}", file=sys.stderr)
            self._enabled = False

    @staticmethod
    def _check_api_key() -> bool:
        """Check if W&B API key is configured."""
        if os.environ.get("WANDB_API_KEY"):
            return True
        # Check common paths
        import pathlib
        key_paths = [
            pathlib.Path.home() / ".netrc",
            pathlib.Path.home() / ".wandb" / "settings",
        ]
        for p in key_paths:
            if p.exists():
                content = p.read_text()
                if "wandb" in content.lower() or "api" in content.lower():
                    return True
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run(self):
        return self._run

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self._enabled and self._run is not None:
            try:
                self._run.log(metrics, step=step)
            except Exception:
                pass  # silently ignore if connection drops mid-training

    def watch(self, model, log="gradients", log_freq=100, log_graph=False):
        """Watch a model's gradients/parameters.

        Args:
            model: torch.nn.Module to watch
            log: "gradients", "parameters", "all", or None
            log_freq: how often to log (in batches)
            log_graph: whether to log the computation graph
        """
        if self._enabled and self._run is not None:
            try:
                self._run.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
            except Exception as e:
                print(f"[W&B] watch failed: {e}", file=sys.stderr)

    def save(self, path: str):
        if self._enabled and self._run is not None:
            try:
                self._run.save(path)
            except Exception:
                pass

    def finish(self):
        if self._enabled and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass


def init_wandb(
    project: str = "align",
    name: Optional[str] = None,
    config: Optional[Dict] = None,
) -> WandBTrainer:
    """Initialize W&B trainer with project, name, and config.

    Returns dummy trainer if W&B is disabled / not installed.
    """
    return WandBTrainer(project=project, name=name, config=config)


def log_metrics(trainer: WandBTrainer, metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to W&B (no-op if disabled)."""
    trainer.log(metrics, step=step)
