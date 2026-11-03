"""Motif-to-TF mapping shared by all TF-level analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from rega_perturb.utils import get_logger

logger = get_logger("analysis.motif_tf")


def build_motif_tf_matrix(
    motif_tf_file: str | Path,
    motif_names: Iterable[str],
    genes: Iterable[str],
    motif_col: int = 0,
    tf_col: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the binary motif x TF matrix and the TF names.

    ``motif_tf_file`` is a tab-separated table without header (one motif and one TF
    symbol per row). A TF is kept when at least one of its motifs is among
    ``motif_names`` and its symbol is among the modeled ``genes``. TFs are sorted
    alphabetically.
    """
    motif_names = np.asarray(list(motif_names)).astype(str)
    gene_set = set(np.asarray(list(genes)).astype(str).tolist())

    mt = pd.read_csv(motif_tf_file, sep="\t", header=None, dtype=str)
    mt = mt.iloc[:, [motif_col, tf_col]].dropna()
    mt.columns = ["motif", "TF"]
    mt["motif"] = mt["motif"].str.strip()
    mt["TF"] = mt["TF"].str.strip()

    motif_to_idx = {m: i for i, m in enumerate(motif_names)}
    mt = mt[mt["motif"].isin(motif_to_idx) & mt["TF"].isin(gene_set)]
    tf_list = sorted(mt["TF"].unique().tolist())
    if not tf_list:
        raise ValueError(
            "No TF of the motif-TF table matches both the model motifs and the modeled genes; "
            "check that motif names and gene symbols use the same naming scheme."
        )
    tf_to_idx = {t: i for i, t in enumerate(tf_list)}

    C = np.zeros((len(motif_names), len(tf_list)), dtype=np.float32)
    C[mt["motif"].map(motif_to_idx).to_numpy(), mt["TF"].map(tf_to_idx).to_numpy()] = 1.0
    logger.info(f"[motif_tf] motifs={len(motif_names)}, TFs={len(tf_list)}, pairs={int(C.sum())}")
    return C, np.asarray(tf_list)
