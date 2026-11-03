"""TF-target gene (TF-TG) regulatory scores.

Two scores are available:

* ``"influence"`` -- functional influence of each TF on each target gene;
* ``"connectivity"`` -- regulatory connectivity between each TF and each target gene.

Both are returned as a gene x TF matrix (non-negative) and can be written as a long
table with columns ``TF``, ``TG`` and ``score``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from rega_perturb.analysis.model_io import TrainedModel
from rega_perturb.analysis.motif_tf import build_motif_tf_matrix
from rega_perturb.utils import get_logger

logger = get_logger("analysis.tf_tg_scores")

SCORES = ("influence", "connectivity")


def _influence(model: TrainedModel, C: np.ndarray) -> np.ndarray:
    BV = (model.B @ model.V).astype(np.float32)
    WBV = (model.W @ BV).astype(np.float32)
    net_motif = (WBV @ model.V.T).astype(np.float32)
    return (net_motif @ C).astype(np.float32)


def _connectivity(model: TrainedModel, C: np.ndarray) -> np.ndarray:
    V = model.V
    VtC = V.T @ C
    mask_all = np.asarray(((model.B_prior @ sparse.csr_matrix(C)) > 0).todense())
    out = np.zeros((len(model.genes), C.shape[1]), dtype=np.float64)
    for W_b, B_b, gi, ri in model.blocks:
        BV = B_b @ V
        WBV = W_b @ BV
        tf_re = (BV @ VtC).astype(np.float64)
        sub = model.W_prior[gi][:, ri].tocoo()
        re_tg = sparse.csr_matrix(
            (np.einsum("ik,ik->i", WBV[sub.row], BV[sub.col]).astype(np.float64),
             (sub.row, sub.col)),
            shape=sub.shape,
        )
        out[gi] = re_tg @ (tf_re * mask_all[ri])
    return out


def compute_tf_tg_scores(
    model: TrainedModel,
    motif_tf_file: str | Path,
    score: str = "influence",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scores, genes, tfs)``; ``scores`` has shape (n_genes, n_TFs).

    TFs are those of ``motif_tf_file`` whose motif is in the model and whose gene is
    among the modeled genes (sorted alphabetically).
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    C, tfs = build_motif_tf_matrix(motif_tf_file, model.motifs, model.genes)
    S = _influence(model, C) if score == "influence" else _connectivity(model, C)
    logger.info(f"[tf_tg] score={score}: {S.shape[0]} genes x {S.shape[1]} TFs")
    return S, model.genes.copy(), tfs


def tf_tg_long_table(
    scores: np.ndarray,
    genes: np.ndarray,
    tfs: np.ndarray,
    top_per_tf: int | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Long table ``TF, TG, score`` (rows ordered by gene, then TF).

    ``top_per_tf`` keeps the highest-scoring target genes of each TF; ``min_score``
    drops rows below a threshold.
    """
    n_gene, n_tf = scores.shape
    gi, fi = np.meshgrid(np.arange(n_gene), np.arange(n_tf), indexing="ij")
    df = pd.DataFrame({"TF": np.asarray(tfs)[fi.ravel()], "TG": np.asarray(genes)[gi.ravel()],
                       "score": scores.ravel()})
    if top_per_tf is not None and top_per_tf > 0:
        df = (df.sort_values(["TF", "score"], ascending=[True, False], kind="stable")
                .groupby("TF", sort=False).head(top_per_tf))
    if min_score is not None:
        df = df[df["score"] >= min_score]
    return df.reset_index(drop=True)


def write_tf_tg_scores(
    model: TrainedModel,
    motif_tf_file: str | Path,
    out_dir: str | Path,
    scores: tuple[str, ...] = SCORES,
    top_per_tf: int | None = None,
    min_score: float | None = None,
) -> list[Path]:
    """Write ``tf_tg_<score>.tsv.gz`` for each requested score."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in scores:
        S, genes, tfs = compute_tf_tg_scores(model, motif_tf_file, name)
        df = tf_tg_long_table(S, genes, tfs, top_per_tf=top_per_tf, min_score=min_score)
        path = out_dir / f"tf_tg_{name}.tsv.gz"
        df.to_csv(path, sep="\t", index=False, float_format="%.7g")
        logger.info(f"[write] {path} ({len(df):,} rows)")
        written.append(path)
    return written
