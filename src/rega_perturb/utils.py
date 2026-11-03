"""Shared helpers: logging, run records and small array utilities."""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rega_perturb._version import __version__

LOGGER_NAME = "rega_perturb"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger (or a child logger)."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure package logging to stderr and, optionally, a file."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    """MD5 checksum of a file (used to record inputs in run records)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _plain(value: Any) -> Any:
    """Convert values to YAML-friendly built-in types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def write_run_record(
    out_dir: str | Path,
    step: str,
    params: dict[str, Any],
    inputs: dict[str, str | Path] | None = None,
    checksum_inputs: bool = False,
) -> Path:
    """Write ``<out_dir>/<step>.run.yaml`` with parameters, versions and inputs.

    Every command writes one record so that a result directory documents how it
    was produced (parameters, random seeds, package and dependency versions).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = {"rega_perturb": __version__, "python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "anndata", "h5py", "torch"):
        try:
            versions[mod] = importlib.metadata.version(mod)
        except Exception:  # pragma: no cover - optional dependency
            pass
    record: dict[str, Any] = {
        "step": step,
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "versions": versions,
        "params": _plain(params),
    }
    if inputs:
        rec_inputs = {}
        for key, p in inputs.items():
            entry: dict[str, Any] = {"path": str(p)}
            if checksum_inputs and p is not None and Path(p).is_file():
                entry["md5"] = file_md5(p)
            rec_inputs[key] = entry
        record["inputs"] = rec_inputs
    path = out_dir / f"{step}.run.yaml"
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def to_str_array(x) -> np.ndarray:
    """Return a numpy array of Python strings (handles bytes / object arrays)."""
    return np.asarray(x).astype(str)


def chrom_sort_key(chrom: str) -> tuple:
    """Sort key for chromosome names: chr1..chr22, chrX, chrY, chrM, others."""
    c = str(chrom).replace("chr", "")
    if c.isdigit():
        return (0, int(c), "")
    if c == "X":
        return (1, 0, "")
    if c == "Y":
        return (2, 0, "")
    if c in ("M", "MT"):
        return (3, 0, "")
    return (4, 0, c)
