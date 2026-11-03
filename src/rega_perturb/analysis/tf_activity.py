"""Perturbation-induced changes in TF activity.

For every perturbation target, the inferred activity of each TF is compared with
its activity in the non-targeting control and reported as a log2 fold change
(target x TF matrix).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rega_perturb.analysis.model_io import TrainedModel, detect_control
from rega_perturb.analysis.motif_tf import build_motif_tf_matrix
from rega_perturb.utils import get_logger

logger = get_logger("analysis.tf_activity")


def tf_activity(model: TrainedModel, motif_tf_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Inferred TF activity per perturbation target: ``(U, tfs)`` with U of shape (n_TFs, n_targets)."""
    C, tfs = build_motif_tf_matrix(motif_tf_file, model.motifs, model.genes)
    U = C.T @ (model.V.astype(np.float64) @ model.H.astype(np.float64))
    return U, tfs


def compute_tf_activity_changes(
    model: TrainedModel,
    motif_tf_file: str | Path,
    control: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(log2fc, activity)``.

    ``log2fc`` is a target x TF DataFrame of log2(activity under the target /
    activity under the control); the control row is 0. ``activity`` holds the
    inferred activities (target x TF). Non-positive activities give NaN.
    """
    control = control or detect_control(model.targets)
    U, tfs = tf_activity(model, motif_tf_file)
    ci = model.target_index(control)
    n_bad = int((U <= 0).sum())
    if n_bad:
        logger.warning(f"{n_bad} non-positive TF activities; their log2 fold changes are NaN.")
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2(U / U[:, [ci]])
    L[~np.isfinite(L)] = np.nan
    idx = pd.Index(model.targets, name="perturb_target")
    log2fc = pd.DataFrame(L.T, index=idx, columns=tfs)
    activity = pd.DataFrame(U.T, index=idx, columns=tfs)
    logger.info(f"[tf_activity] {len(model.targets)} targets x {len(tfs)} TFs, control={control!r}, "
                f"min activity={U.min():.3g}")
    return log2fc, activity


def write_tf_activity_changes(
    model: TrainedModel,
    motif_tf_file: str | Path,
    out_dir: str | Path,
    control: str | None = None,
    threshold: float | None = None,
) -> dict[str, Path]:
    """Write the target x TF log2 fold-change matrix and a long table.

    Files: ``tf_activity_log2fc.tsv`` (wide; rows = targets, columns = TFs) and
    ``tf_activity_changes.tsv.gz`` (long: ``target, TF, activity_target,
    activity_control, log2fc`` and, with ``threshold``, ``changed`` =
    |log2fc| >= threshold).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    control = control or detect_control(model.targets)
    log2fc, activity = compute_tf_activity_changes(model, motif_tf_file, control)

    wide = out_dir / "tf_activity_log2fc.tsv"
    log2fc.to_csv(wide, sep="\t", float_format="%.6g")

    long = log2fc.stack(future_stack=True).rename("log2fc").reset_index()
    long.columns = ["target", "TF", "log2fc"]
    act = activity.stack(future_stack=True).to_numpy()
    ctrl = np.tile(activity.loc[control].to_numpy(), len(activity))
    long.insert(2, "activity_target", act)
    long.insert(3, "activity_control", ctrl)
    if threshold is not None:
        long["changed"] = long["log2fc"].abs() >= threshold
    path = out_dir / "tf_activity_changes.tsv.gz"
    long.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info(f"[write] {wide}; {path} ({len(long):,} rows)")
    return {"wide": wide, "long": path}
