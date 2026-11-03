"""Training configuration from YAML files and command-line overrides."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from rega_perturb.model.train import PerturbTrainingConfig

# Fields that cannot be set from a YAML file (non-serialisable).
_NOT_CONFIGURABLE = {"dtype"}


def configurable_fields() -> list[str]:
    """Names of training parameters that can be set in a YAML file."""
    return [f.name for f in dataclasses.fields(PerturbTrainingConfig)
            if f.name not in _NOT_CONFIGURABLE]


def make_training_config(
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PerturbTrainingConfig:
    """Build a training configuration.

    Precedence: package defaults < ``config_file`` (YAML mapping, optionally under a
    ``train:`` key) < ``overrides`` (entries set to ``None`` are ignored).
    Unknown keys raise ``ValueError``.
    """
    values: dict[str, Any] = {}
    if config_file is not None:
        data = yaml.safe_load(Path(config_file).read_text()) or {}
        if "train" in data and isinstance(data["train"], dict):
            data = data["train"]
        values.update(data)
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    allowed = set(configurable_fields())
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown training parameter(s): {unknown}. Allowed: {sorted(allowed)}")
    return PerturbTrainingConfig(**values)
