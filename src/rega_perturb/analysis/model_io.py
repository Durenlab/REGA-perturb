"""Load a trained model together with its graph inputs for downstream analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from rega_perturb.utils import get_logger

logger = get_logger("analysis.model_io")

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required to read model checkpoints.") from e


def _h5_strings(node) -> np.ndarray:
    """String array from an AnnData element: plain dataset or nullable-string-array group."""
    if isinstance(node, h5py.Group):
        if "mask" in node and np.asarray(node["mask"][()]).any():
            raise ValueError(f"{node.name} contains missing values.")
        node = node["values"]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in node[()]])


def _h5_names(group) -> np.ndarray:
    return _h5_strings(group[group.attrs["_index"]])


def _h5_csr(group) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=tuple(group.attrs["shape"]),
    )


@dataclass
class TrainedModel:
    """A trained model aligned to the gene / RE / motif order of its graph AnnData."""

    genes: np.ndarray                 # modeled genes (graph obs order)
    res: np.ndarray                   # regulatory elements (graph var order)
    motifs: np.ndarray
    targets: np.ndarray               # perturbation targets
    V: np.ndarray
    H: np.ndarray
    W: sparse.csr_matrix
    B: np.ndarray
    W_prior: sparse.csr_matrix
    B_prior: sparse.csr_matrix
    blocks: list = field(repr=False, default_factory=list)

    def target_index(self, target: str) -> int:
        idx = np.flatnonzero(self.targets == target)
        if idx.size == 0:
            raise KeyError(f"perturbation target not in model: {target!r}")
        return int(idx[0])


def detect_control(targets, candidates=("non-targeting", "NTC", "CTRL", "control")) -> str:
    """Return the control target label (first of ``candidates`` present in ``targets``)."""
    present = set(np.asarray(targets).astype(str).tolist())
    for c in candidates:
        if c in present:
            return c
    raise ValueError(f"No control target found among {list(candidates)}; pass the label explicitly.")


def load_trained_model(checkpoint: str | Path, graph_h5ad: str | Path) -> TrainedModel:
    """Read ``final_perturb.pt`` and the graph AnnData (``REGA_graph_without_GEX.h5ad``)."""
    logger.info(f"[load] {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    with h5py.File(graph_h5ad, "r") as f:
        genes, res = _h5_names(f["obs"]), _h5_names(f["var"])
        motifs_h5 = _h5_strings(f["uns/motif_names"])
        W_prior = _h5_csr(f["X"])
        B_prior = _h5_csr(f["varm/Peak_Motif"]).astype(np.float32)

    V = ckpt["V"].detach().cpu().numpy().astype(np.float32)
    H = ckpt["H"].detach().cpu().numpy().astype(np.float32)
    motifs = np.asarray(ckpt["motifs"]).astype(str) if "motifs" in ckpt else motifs_h5
    if len(motifs) != V.shape[0] or not np.array_equal(motifs, motifs_h5):
        raise ValueError("Motif order of the checkpoint and the graph AnnData differ.")
    targets = np.asarray(ckpt["perturb_targets"]).astype(str)

    g2i = {g: i for i, g in enumerate(genes)}
    r2i = {r: i for i, r in enumerate(res)}
    rows, cols, data, blocks = [], [], [], []
    B = np.zeros((len(res), V.shape[0]), dtype=np.float32)
    for W_b, B_b, g_b, r_b in zip(ckpt["W_blocks"], ckpt["B_blocks"], ckpt["genes_blocks"],
                                  ckpt["RE_blocks"]):
        Wd = (W_b.to_dense() if W_b.is_sparse else W_b).detach().cpu().numpy().astype(np.float32)
        Bd = (B_b.to_dense() if B_b.is_sparse else B_b).detach().cpu().numpy().astype(np.float32)
        gi = np.array([g2i[g] for g in np.asarray(g_b).astype(str)], dtype=np.int64)
        ri = np.array([r2i[r] for r in np.asarray(r_b).astype(str)], dtype=np.int64)
        lr, lc = np.nonzero(Wd)
        rows.append(gi[lr])
        cols.append(ri[lc])
        data.append(Wd[lr, lc])
        B[ri, :] = Bd
        blocks.append((Wd, Bd, gi, ri))
    W = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(genes), len(res)), dtype=np.float32,
    ).tocsr()
    logger.info(f"[load] genes={len(genes)}, REs={len(res)}, motifs={len(motifs)}, "
                f"targets={len(targets)}, K={V.shape[1]}")
    return TrainedModel(genes=genes, res=res, motifs=motifs, targets=targets, V=V, H=H, W=W, B=B,
                        W_prior=W_prior, B_prior=B_prior, blocks=blocks)
