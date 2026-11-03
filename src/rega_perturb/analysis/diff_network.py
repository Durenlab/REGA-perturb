"""Perturbation-specific and differential TF-TG regulatory networks.

For each perturbation target the package infers a perturbation-specific TF-TG
network and its difference from the non-targeting control network (edge-level
``delta_w``; positive = strengthened, negative = weakened inferred regulation).
Per target gene two summaries are reported:

* ``RRS`` (regulatory rewiring score) -- total magnitude of the edge changes;
* ``NRS`` (net regulatory shift score) -- signed sum of the edge changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from rega_perturb.analysis.model_io import TrainedModel, detect_control
from rega_perturb.analysis.motif_tf import build_motif_tf_matrix
from rega_perturb.utils import get_logger

logger = get_logger("analysis.diff_network")


@dataclass
class NetworkFactors:
    """Pre-computed quantities shared by all targets."""

    genes: np.ndarray
    tfs: np.ndarray
    targets: np.ndarray
    control: str
    G: np.ndarray
    T: np.ndarray
    H: np.ndarray

    def program_shift(self, target: str) -> np.ndarray:
        if target not in set(self.targets.tolist()):
            raise KeyError(f"perturbation target not in model: {target!r}")
        t = int(np.flatnonzero(self.targets == target)[0])
        c = int(np.flatnonzero(self.targets == self.control)[0])
        return self.H[:, t] - self.H[:, c]


def network_factors(
    model: TrainedModel,
    motif_tf_file: str | Path,
    control: str | None = None,
) -> NetworkFactors:
    control = control or detect_control(model.targets)
    model.target_index(control)
    C, tfs = build_motif_tf_matrix(motif_tf_file, model.motifs, model.genes)
    V = model.V
    G = (model.W @ (model.B @ V)).astype(np.float32)
    s = V.sum(axis=0).astype(np.float32)
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    G = (G / s[None, :]).astype(np.float32)
    T = (V.T @ C).astype(np.float32)
    return NetworkFactors(genes=model.genes, tfs=tfs, targets=model.targets, control=control,
                          G=G, T=T, H=model.H)


def specific_network(f: NetworkFactors, target: str) -> np.ndarray:
    """Perturbation-specific network of ``target`` (gene x TF)."""
    t = int(np.flatnonzero(f.targets == target)[0])
    return ((f.G * f.H[:, t][None, :]) @ f.T).astype(np.float32)


def delta_network(f: NetworkFactors, target: str) -> np.ndarray:
    """Differential network of ``target`` versus the control (gene x TF ``delta_w``)."""
    return ((f.G * f.program_shift(target)[None, :]) @ f.T).astype(np.float32)


def rewiring_scores(
    f: NetworkFactors,
    targets: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(RRS, NRS, targets)``; RRS and NRS have shape (n_genes, n_targets)."""
    if targets is None:
        targets = [t for t in f.targets if t != f.control]
    targets = list(targets)
    tf_sum = f.T.sum(axis=1)
    rrs = np.empty((len(f.genes), len(targets)), dtype=np.float32)
    nrs = np.empty((len(f.genes), len(targets)), dtype=np.float32)
    for j, t in enumerate(targets):
        dh = f.program_shift(t)
        rrs[:, j] = np.abs(delta_network(f, t)).sum(axis=1)
        nrs[:, j] = f.G @ (dh * tf_sum)
        if (j + 1) % 200 == 0:
            logger.info(f"  rewiring scores: {j + 1}/{len(targets)} targets")
    return rrs, nrs, np.asarray(targets)


def write_rewiring_scores(f: NetworkFactors, out_file: str | Path,
                          targets: list[str] | None = None) -> Path:
    """Write ``genes``, ``targets``, ``RRS`` and ``NRS`` (gene x target, float32) to HDF5."""
    rrs, nrs, targets = rewiring_scores(f, targets)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as h5:
        h5.create_dataset("genes", data=np.asarray(f.genes, dtype="S"))
        h5.create_dataset("targets", data=np.asarray(targets, dtype="S"))
        h5.create_dataset("TFs", data=np.asarray(f.tfs, dtype="S"))
        for name, arr in (("RRS", rrs), ("NRS", nrs)):
            h5.create_dataset(name, data=arr, compression="gzip", chunks=(len(f.genes), 1))
        h5.attrs["control"] = f.control
    logger.info(f"[write] {out_file}: {rrs.shape[0]} genes x {rrs.shape[1]} targets")
    return out_file


def network_long_table(
    f: NetworkFactors,
    target: str,
    kind: str = "delta",
    min_abs: float | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Long table of one target: ``target, TF, TG, delta_w`` (or ``weight`` for kind='specific').

    ``min_abs`` keeps edges with |value| >= min_abs; ``top_k`` keeps the k edges
    with the largest |value|.
    """
    M = delta_network(f, target) if kind == "delta" else specific_network(f, target)
    col = "delta_w" if kind == "delta" else "weight"
    vals = M.ravel()
    gi, fi = np.divmod(np.arange(vals.size), M.shape[1])
    keep = np.ones(vals.shape, dtype=bool)
    if min_abs is not None:
        keep &= np.abs(vals) >= min_abs
    idx = np.flatnonzero(keep)
    if top_k is not None and top_k < idx.size:
        idx = idx[np.argsort(-np.abs(vals[idx]), kind="stable")[:top_k]]
    return pd.DataFrame({"target": target, "TF": f.tfs[fi[idx]], "TG": f.genes[gi[idx]],
                         col: vals[idx]})


def write_networks(
    f: NetworkFactors,
    targets: list[str],
    out_dir: str | Path,
    kind: str = "delta",
    min_abs: float | None = None,
    top_k: int | None = None,
    fmt: str = "tsv",
) -> list[Path]:
    """Write per-target networks.

    ``fmt="tsv"``: one ``<kind>_network.<target>.tsv.gz`` long table per target
    (filters apply). ``fmt="h5"``: one HDF5 file ``<kind>_networks.h5`` with a dense
    gene x TF float32 matrix per target (``/networks/<target>``; filters ignored).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_vals = len(f.genes) * len(f.tfs)
    logger.info(f"[networks] {len(targets)} targets, {n_vals:,} edges each "
                f"(~{len(targets) * n_vals * 4 / 1e9:.2f} GB uncompressed as float32)")
    written = []
    if fmt == "h5":
        path = out_dir / f"{kind}_networks.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("genes", data=np.asarray(f.genes, dtype="S"))
            h5.create_dataset("TFs", data=np.asarray(f.tfs, dtype="S"))
            h5.attrs["control"] = f.control
            grp = h5.create_group("networks")
            for t in targets:
                M = delta_network(f, t) if kind == "delta" else specific_network(f, t)
                grp.create_dataset(str(t), data=M, compression="gzip")
        written.append(path)
    elif fmt == "tsv":
        for t in targets:
            df = network_long_table(f, t, kind=kind, min_abs=min_abs, top_k=top_k)
            path = out_dir / f"{kind}_network.{t}.tsv.gz"
            df.to_csv(path, sep="\t", index=False, float_format="%.7g")
            written.append(path)
    else:
        raise ValueError("fmt must be 'tsv' or 'h5'")
    logger.info(f"[write] {len(written)} file(s) in {out_dir}")
    return written
