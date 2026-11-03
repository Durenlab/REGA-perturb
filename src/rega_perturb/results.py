"""Collect a trained model into a result AnnData (``rega_perturb_result.h5ad``).

The result AnnData has genes as ``obs`` and REs as ``var``; module-, motif-,
perturbation-target- and cell-level results are stored in ``obsm``, ``varm`` and
``uns`` (see :func:`build_result_anndata` for the keys). No expression data is
stored.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from rega_perturb.utils import get_logger, write_run_record

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required to read model checkpoints.") from e

logger = get_logger("results")


def _to_str_array(x) -> np.ndarray:
    """Convert a name array to a str ndarray (bytes / object safe)."""
    return np.asarray(x).astype(str)


def _to_numpy(t) -> np.ndarray:
    """Convert a torch.Tensor or array to numpy float32."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _to_dense_f32(M) -> np.ndarray:
    """Convert a scipy sparse matrix / np.matrix / ndarray to a dense float32 array."""
    if sparse.issparse(M):
        M = M.toarray()
    M = np.asarray(M)
    # np.matrix -> ndarray
    if isinstance(M, np.matrix):
        M = np.asarray(M)
    return M.astype(np.float32)


def normalize_embedding(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize an embedding matrix for storage in the result AnnData."""
    E = np.asarray(E, dtype=np.float32)
    col_sum = np.abs(E).sum(axis=0, keepdims=True) + eps
    E = E / col_sum
    row_sum = np.abs(E).sum(axis=1, keepdims=True) + eps
    E = E / row_sum
    return E.astype(np.float32)


def report_missing(name, missing_list, top=20):
    logger.info(f"\n  Missing {name} ({len(missing_list)} total). First {min(top, len(missing_list))}:")
    for x in missing_list[:top]:
        logger.info(f"    - {x}")


# ============================================================
# Step 1: load the input (graph) AnnData
# ============================================================

def load_input_anndata(input_h5ad):
    logger.info("\n[Step 1] Loading input AnnData ...")
    adata = ad.read_h5ad(input_h5ad)
    logger.info(f"  AnnData shape : {adata.X.shape}")
    logger.info(f"  obs           : {list(adata.obs.columns)}")
    logger.info(f"  var           : {list(adata.var.columns)}")

    # motif names are required; GEX is optional
    if "motif_names" not in adata.uns:
        raise ValueError("adata.uns['motif_names'] missing.")
    if "Peak_Motif" not in adata.varm:
        warnings.warn("adata.varm['Peak_Motif'] missing; not required here, continuing.")

    genes_master = _to_str_array(adata.obs_names)
    RE_master    = _to_str_array(adata.var_names)
    motif_names  = _to_str_array(adata.uns["motif_names"])

    has_gex = "GEX" in adata.obsm

    logger.info(f"  genes  : {len(genes_master)}")
    logger.info(f"  REs    : {len(RE_master)}")
    logger.info(f"  motifs : {len(motif_names)}")
    logger.info(f"  GEX present      : {has_gex}")

    return adata, genes_master, RE_master, motif_names, has_gex


# ============================================================
# Step 2: load final_perturb.pt
# ============================================================

def load_perturb_result(rega_result_path):
    logger.info("\n[Step 2] Loading REGA-Perturb result (final_perturb.pt) ...")
    ckpt = torch.load(rega_result_path, map_location="cpu", weights_only=False)

    required = [
        "W_blocks", "B_blocks", "V", "H", "p_values",
        "target_idx", "perturb_targets",
        "genes_blocks", "RE_blocks", "cells", "motifs",
    ]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"final_perturb.pt missing field '{k}'")

    logger.info(f"  model          : {ckpt.get('model', 'N/A')}")
    logger.info(f"  blocks         : {len(ckpt['W_blocks'])}")
    logger.info(f"  V              : {tuple(ckpt['V'].shape)}")
    logger.info(f"  H (K x target) : {tuple(ckpt['H'].shape)}")
    logger.info(f"  p_values       : {len(ckpt['p_values'])}")
    logger.info(f"  perturb targets: {len(ckpt['perturb_targets'])}")
    logger.info(f"  cells          : {len(ckpt['cells'])}")
    return ckpt


# ============================================================
# Step 3: consistency checks
# ============================================================

def consistency_checks(ckpt, genes_master, RE_master, motif_names):
    logger.info("\n[Step 3] Consistency checks ...")

    W_blocks     = ckpt["W_blocks"]
    B_blocks     = ckpt["B_blocks"]
    genes_blocks = [_to_str_array(g) for g in ckpt["genes_blocks"]]
    RE_blocks    = [_to_str_array(r) for r in ckpt["RE_blocks"]]
    V            = _to_numpy(ckpt["V"])
    H            = _to_numpy(ckpt["H"])
    rega_motifs  = _to_str_array(ckpt["motifs"])

    perturb_targets = _to_str_array(ckpt["perturb_targets"])
    cells           = _to_str_array(ckpt["cells"])
    target_idx      = np.asarray(ckpt["target_idx"]).astype(np.int64)
    p_values        = _to_numpy(ckpt["p_values"]).astype(np.float32)

    # number of blocks
    assert len(W_blocks) == len(B_blocks) == len(genes_blocks) == len(RE_blocks), (
        f"block counts mismatch: W={len(W_blocks)} B={len(B_blocks)} "
        f"genes={len(genes_blocks)} RE={len(RE_blocks)}"
    )

    # shape of every block
    for i, (W_b, B_b, g_b, r_b) in enumerate(zip(W_blocks, B_blocks, genes_blocks, RE_blocks)):
        W_arr = _to_numpy(W_b)
        B_arr = _to_numpy(B_b)
        assert W_arr.shape == (len(g_b), len(r_b)), \
            f"block {i}: W shape {W_arr.shape} != (genes={len(g_b)}, RE={len(r_b)})"
        assert B_arr.shape == (len(r_b), len(motif_names)), \
            f"block {i}: B shape {B_arr.shape} != (RE={len(r_b)}, motifs={len(motif_names)})"

    # motif order
    if not np.array_equal(rega_motifs, motif_names):
        raise ValueError("checkpoint 'motifs' differ from input AnnData uns['motif_names'].")

    # V / H dimensions
    K = V.shape[1]
    assert V.shape[0] == len(motif_names), \
        f"V.shape[0]={V.shape[0]} != len(motif_names)={len(motif_names)}"
    assert H.shape[0] == K, f"H.shape[0]={H.shape[0]} != K={K}"
    assert H.shape[1] == len(perturb_targets), \
        f"H.shape[1]={H.shape[1]} != n_targets={len(perturb_targets)}"

    # compact P representation
    assert len(target_idx) == len(cells), \
        f"len(target_idx)={len(target_idx)} != n_cells={len(cells)}"
    assert len(p_values) == len(cells), \
        f"len(p_values)={len(p_values)} != n_cells={len(cells)}"
    if target_idx.size > 0:
        assert int(target_idx.max()) < len(perturb_targets), \
            "target_idx contains out-of-range perturbation-target indices."
        assert int(target_idx.min()) >= 0, "target_idx contains negative indices."

    # every block gene / RE must exist in the input AnnData
    gene_set = set(genes_master.tolist())
    re_set   = set(RE_master.tolist())

    missing_genes = []
    for g_b in genes_blocks:
        missing_genes += [g for g in g_b.tolist() if g not in gene_set]
    if missing_genes:
        report_missing("genes (block not in input AnnData)", missing_genes)
        raise ValueError(f"{len(missing_genes)} genes from blocks are not in input AnnData obs_names.")

    missing_REs = []
    for r_b in RE_blocks:
        missing_REs += [r for r in r_b.tolist() if r not in re_set]
    if missing_REs:
        report_missing("REs (block not in input AnnData)", missing_REs)
        raise ValueError(f"{len(missing_REs)} REs from blocks are not in input AnnData var_names.")

    logger.info(f"  All checks passed. K={K}, n_targets={len(perturb_targets)}, n_cells={len(cells)}")
    return {
        "W_blocks": W_blocks, "B_blocks": B_blocks,
        "genes_blocks": genes_blocks, "RE_blocks": RE_blocks,
        "V": V, "H": H, "K": K,
        "perturb_targets": perturb_targets, "cells": cells,
        "target_idx": target_idx, "p_values": p_values,
    }


# ============================================================
# Step 4: assemble global W_learned (sparse) and B_learned (dense)
# ============================================================

def assemble_W_learned(W_blocks, genes_blocks, RE_blocks, genes_master, RE_master):
    logger.info("\n[Step 4a] Assembling W_learned (sparse) ...")
    gene_to_idx = {g: i for i, g in enumerate(genes_master)}
    re_to_idx   = {r: i for i, r in enumerate(RE_master)}

    rows, cols, data = [], [], []
    for W_b, g_b, r_b in zip(W_blocks, genes_blocks, RE_blocks):
        W_arr = _to_numpy(W_b)
        gi = np.array([gene_to_idx[g] for g in g_b], dtype=np.int64)
        ri = np.array([re_to_idx[r]   for r in r_b], dtype=np.int64)

        local_rows, local_cols = np.where(W_arr != 0)
        if local_rows.size == 0:
            continue
        rows.append(gi[local_rows])
        cols.append(ri[local_cols])
        data.append(W_arr[local_rows, local_cols])

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data).astype(np.float32)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float32)

    W_learned = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(genes_master), len(RE_master)),
        dtype=np.float32,
    ).tocsr()

    logger.info(f"  W_learned shape : {W_learned.shape}, nnz = {W_learned.nnz}")
    return W_learned


def assemble_B_learned(B_blocks, RE_blocks, RE_master, motif_names):
    logger.info("\n[Step 4b] Assembling B_learned (dense) ...")
    re_to_idx = {r: i for i, r in enumerate(RE_master)}

    B_learned = np.zeros((len(RE_master), len(motif_names)), dtype=np.float32)
    for B_b, r_b in zip(B_blocks, RE_blocks):
        B_arr = _to_numpy(B_b)
        ri = np.array([re_to_idx[r] for r in r_b], dtype=np.int64)
        B_learned[ri, :] = B_arr

    logger.info(f"  B_learned shape : {B_learned.shape}")
    return B_learned


# ============================================================
# Step 5a: per-cell perturbation assignment and target statistics
# ============================================================

def build_P(target_idx, p_values, perturb_targets, cells):
    """Per-cell perturbation assignment (cell x target, one non-zero per row) and
    per-target cell counts, mean and summed cell weights."""
    logger.info("\n[Step 5a] Building P (cell x target) and target statistics ...")
    n_cells = len(cells)
    n_targets = len(perturb_targets)

    rows = np.arange(n_cells, dtype=np.int64)
    cols = target_idx.astype(np.int64)
    P_sparse = sparse.coo_matrix(
        (p_values.astype(np.float32), (rows, cols)),
        shape=(n_cells, n_targets), dtype=np.float32,
    ).tocsr()

    target_n_cells = np.bincount(cols, minlength=n_targets).astype(np.int64)
    target_sum_p   = np.bincount(cols, weights=p_values, minlength=n_targets).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        target_mean_p = np.where(
            target_n_cells > 0,
            target_sum_p / np.maximum(target_n_cells, 1),
            0.0,
        ).astype(np.float32)

    logger.info(f"  P_sparse        : {P_sparse.shape}, nnz = {P_sparse.nnz}")
    logger.info(f"  targets with >=1 cell : {(target_n_cells > 0).sum()} / {n_targets}")
    logger.info(f"  mean cells/target     : {target_n_cells[target_n_cells>0].mean():.1f}"
          if (target_n_cells > 0).any() else "  (no assigned cells)")
    return {
        "P_sparse": P_sparse,
        "target_n_cells": target_n_cells,
        "target_mean_p": target_mean_p,
        "target_sum_p": target_sum_p,
    }


# ============================================================
# Step 5b: derived quantities
# ============================================================

def compute_downstream(adata, W_learned, B_learned, V, H,
                       motif_names, has_gex,
                       P_info, perturb_targets, cells, target_idx, p_values,
                       cell_embeddings, max_cells_dense, peak_activity=False):
    """Compute the derived quantities stored in the result AnnData."""
    logger.info("\n[Step 5b] Computing downstream results ...")

    # ---- Embedding ----
    BV  = (B_learned @ V).astype(np.float32)
    WBV = (W_learned @ BV).astype(np.float32)
    logger.info(f"  BV  : {BV.shape}")
    logger.info(f"  WBV : {WBV.shape}")

    # ---- Activity (target-level) ----
    VH  = (V @ H).astype(np.float32)
    BVH = (B_learned @ VH).astype(np.float32)
    logger.info(f"  VH  (motif x target) : {VH.shape}")
    logger.info(f"  BVH (RE x target)    : {BVH.shape}"
                + ("" if peak_activity else " (not stored)"))

    # ---- RE importance ----
    w = np.asarray(W_learned.sum(axis=0)).flatten().astype(np.float32)
    y = BVH.sum(axis=1).astype(np.float32)
    d = (w * y).astype(np.float32)
    if not peak_activity:
        BVH = None   # RE x target matrix is large for genome-scale screens; kept only on request
    logger.info(f"  RE importance d : shape {d.shape}, "
          f"min={float(d.min()):.3e}, max={float(d.max()):.3e}, mean={float(d.mean()):.3e}")

    # ---- Net_RE_TG (sparse, on the non-zero pattern of W_learned) ----
    logger.info("  Computing Net_RE_TG (sparse, on W_learned nonzero pattern) ...")
    W_coo = W_learned.tocoo()
    gene_idx = W_coo.row
    re_idx   = W_coo.col
    if gene_idx.size > 0:
        scores = np.einsum("ik,ik->i", WBV[gene_idx], BV[re_idx]).astype(np.float32)
    else:
        scores = np.array([], dtype=np.float32)
    Net_RE_TG = sparse.coo_matrix(
        (scores, (gene_idx, re_idx)),
        shape=W_learned.shape, dtype=np.float32,
    ).tocsr()
    logger.info(f"  Net_RE_TG : shape {Net_RE_TG.shape}, nnz = {Net_RE_TG.nnz}")

    Net_motif_TG = (WBV @ V.T).astype(np.float32)

    # ---- target-level perturbation embeddings ----
    H_T = H.T.astype(np.float32)
    logger.info(f"  H_T (target x K) : {H_T.shape}")

    # ---- optional cell-level embedding ----
    cell_module = None
    if cell_embeddings:
        n_cells = len(cells)
        if n_cells > max_cells_dense:
            warnings.warn(
                f"n_cells={n_cells} > max_cells_dense={max_cells_dense}; "
                "skipping dense cell x K embedding. Raise --max-cells-dense to force."
            )
        else:
            cell_module = (H[:, target_idx] * p_values[None, :]).T.astype(np.float32)
            logger.info(f"  cell_module (cell x K) : {cell_module.shape}")
    else:
        logger.info("  cell_module : skipped (use --cell-embeddings to enable).")

    # ---- normalize embeddings ----
    logger.info("  Normalizing embeddings (column-then-row L1) ...")
    WBV_norm = normalize_embedding(WBV)
    BV_norm  = normalize_embedding(BV)
    V_norm   = normalize_embedding(V)
    H_T_norm = normalize_embedding(H_T)

    # ---- Motif_TF_PCC: not defined for perturbation-target columns; skipped ----
    Motif_TF_PCC = None
    logger.info("  Motif_TF_PCC : skipped (perturb columns are targets, not samples).")

    results = {
        "BV": BV, "WBV": WBV,
        "VH": VH, "BVH": BVH,
        "w": w, "y": y, "d": d,
        "Net_RE_TG": Net_RE_TG,
        "Net_motif_TG": Net_motif_TG,
        "H_T": H_T,
        "cell_module": cell_module,
        "WBV_norm": WBV_norm, "BV_norm": BV_norm,
        "V_norm": V_norm, "H_T_norm": H_T_norm,
        "Motif_TF_PCC": Motif_TF_PCC,
    }
    return results


# ============================================================
# Step 6: assemble and save the result AnnData
# ============================================================

def build_result_anndata(adata, W_learned, B_learned, V, H,
                         results, P_info, ckpt, out_h5ad,
                         motif_names, TF_names, tf_tg,
                         perturb_targets, cells, target_idx, p_values):
    logger.info("\n[Step 6] Building result AnnData ...")

    # copy obs / var of the input AnnData; no expression data is stored
    adata_result = ad.AnnData(
        X=W_learned,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    n_genes = adata_result.n_obs
    n_REs   = adata_result.n_vars
    n_K     = V.shape[1]
    n_motif = len(motif_names)
    n_target = len(perturb_targets)
    n_cells = len(cells)

    # ---- layers ----
    adata_result.layers["Net_RE_TG"] = results["Net_RE_TG"]

    # ---- var: RE statistics ----
    adata_result.var["RE_strength_w"] = results["w"].astype(np.float32)
    adata_result.var["RE_activity_y"] = results["y"].astype(np.float32)
    adata_result.var["RE_importance"] = results["d"].astype(np.float32)

    # ---- obsm (gene × *) ----
    adata_result.obsm["Gene_Embedding"]      = results["WBV"]
    adata_result.obsm["Gene_Embedding_norm"] = results["WBV_norm"]
    adata_result.obsm["Net_motif_TG"] = results["Net_motif_TG"]
    for name, S in tf_tg.items():
        adata_result.obsm[f"TF_TG_{name}"] = S

    # ---- varm (RE × *) ----
    adata_result.varm["B_learned"]         = B_learned
    adata_result.varm["RE_Embedding"]      = results["BV"]
    adata_result.varm["RE_Embedding_norm"] = results["BV_norm"]
    if results["BVH"] is not None:
        adata_result.varm["Peak_Activity"] = results["BVH"]   # RE × target

    # ---- uns: module / motif / target level results ----
    adata_result.uns["Motif_Embedding"]      = V.astype(np.float32)
    adata_result.uns["Motif_Embedding_norm"] = results["V_norm"]
    adata_result.uns["H"]                 = H.astype(np.float32)          # K × target
    adata_result.uns["Target_Embedding"]     = results["H_T"]               # target × K
    adata_result.uns["Target_Embedding_norm"] = results["H_T_norm"]
    adata_result.uns["Motif_Activity"] = results["VH"]                # motif × target
    if results["Motif_TF_PCC"] is not None:
        adata_result.uns["Motif_TF_PCC"]  = results["Motif_TF_PCC"]

    adata_result.uns["Module_names"]    = np.array([f"Module_{i+1}" for i in range(n_K)])
    adata_result.uns["motif_names"]     = motif_names
    adata_result.uns["perturb_targets"] = perturb_targets
    if TF_names is not None:
        adata_result.uns["TF_names"] = TF_names

    # ---- per-cell perturbation assignment ----
    adata_result.uns["P_cells"]            = cells
    adata_result.uns["P_target_idx"]       = target_idx.astype(np.int64)
    adata_result.uns["P_cell_target_name"] = perturb_targets[target_idx]
    adata_result.uns["P_cell_strength"]    = p_values.astype(np.float32)
    # sparse cell x target matrix
    adata_result.uns["P_sparse"]           = P_info["P_sparse"]
    # target-level statistics
    adata_result.uns["target_n_cells"]     = P_info["target_n_cells"]
    adata_result.uns["target_mean_p"]      = P_info["target_mean_p"]
    adata_result.uns["target_sum_p"]       = P_info["target_sum_p"]

    # optional cell x K embedding
    if results["cell_module"] is not None:
        adata_result.uns["Cell_Embedding"] = results["cell_module"]

    # ---- training metadata ----
    cfg = ckpt.get("config", {}) or {}
    adata_result.uns["REGA_model"] = str(ckpt.get("model", "REGA-Perturb"))
    adata_result.uns["REGA_K"]     = int(cfg.get("K", n_K))
    adata_result.uns["REGA_perturb_config"] = {
        k: (v if isinstance(v, (int, float, str, bool)) else str(v))
        for k, v in cfg.items()
    }

    # ---- shape checks ----
    assert adata_result.X.shape == (n_genes, n_REs), "X shape mismatch"
    assert adata_result.layers["Net_RE_TG"].shape == (n_genes, n_REs), "Net_RE_TG shape mismatch"
    assert adata_result.obsm["Gene_Embedding"].shape == (n_genes, n_K), "Gene_Embedding shape mismatch"
    assert adata_result.obsm["Net_motif_TG"].shape == (n_genes, n_motif), "Net_motif_TG shape mismatch"
    for name in tf_tg:
        assert adata_result.obsm[f"TF_TG_{name}"].shape == (n_genes, len(TF_names)), name
    assert adata_result.varm["B_learned"].shape == (n_REs, n_motif), "B_learned shape mismatch"
    assert adata_result.varm["RE_Embedding"].shape == (n_REs, n_K), "RE_Embedding shape mismatch"
    if results["BVH"] is not None:
        assert adata_result.varm["Peak_Activity"].shape == (n_REs, n_target), "BVH shape mismatch"
    assert adata_result.uns["Motif_Embedding"].shape == (n_motif, n_K), "Motif_Embedding shape mismatch"
    assert adata_result.uns["H"].shape == (n_K, n_target), "H shape mismatch"
    assert adata_result.uns["Motif_Activity"].shape == (n_motif, n_target), "VH shape mismatch"
    assert adata_result.uns["P_sparse"].shape == (n_cells, n_target), "P_sparse shape mismatch"
    assert len(adata_result.uns["P_cells"]) == n_cells, "P_cells length mismatch"
    assert len(adata_result.uns["target_n_cells"]) == n_target, "target_n_cells length mismatch"

    # ---- save ----
    out_h5ad = Path(out_h5ad)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata_result.write_h5ad(out_h5ad)
    logger.info(f"  Saved -> {out_h5ad}")
    return adata_result


# ============================================================
# Step 7: summary
# ============================================================

def print_summary(W_learned, B_learned, results, P_info, out_h5ad, tf_tg):
    logger.info("\n========== Summary ==========")
    logger.info(f"  W_learned          : shape={W_learned.shape}, nnz={W_learned.nnz} (sparse csr)")
    logger.info(f"  B_learned          : shape={B_learned.shape} (dense float32)")
    logger.info(f"  WBV (gene x K)     : {results['WBV'].shape}")
    logger.info(f"  BV  (RE x K)       : {results['BV'].shape}")
    logger.info(f"  VH  (motif x targ) : {results['VH'].shape}")
    if results["BVH"] is not None:
        logger.info(f"  BVH (RE x target)  : {results['BVH'].shape}")
    logger.info(f"  H_T (target x K)   : {results['H_T'].shape}")
    logger.info(f"  Net_RE_TG          : shape={results['Net_RE_TG'].shape}, nnz={results['Net_RE_TG'].nnz}")
    for name, S in tf_tg.items():
        logger.info(f"  TF_TG_{name:<12s}: {S.shape}")
    if results["cell_module"] is not None:
        logger.info(f"  Cell_Embedding     : {results['cell_module'].shape}")

    tnc = P_info["target_n_cells"]
    logger.info(f"  P_sparse           : {P_info['P_sparse'].shape}, nnz={P_info['P_sparse'].nnz}")
    logger.info(f"  targets w/ cells   : {(tnc>0).sum()} / {len(tnc)}")

    d = results["d"]
    logger.info(f"  RE_importance (d)  : min={float(d.min()):.3e}, max={float(d.max()):.3e}, "
          f"mean={float(d.mean()):.3e}, median={float(np.median(d)):.3e}")
    logger.info(f"  Output h5ad        : {out_h5ad}")
    logger.info("==============================")


# ============================================================
# Main
# ============================================================


def collect_results(
    input_h5ad: str | Path,
    checkpoint: str | Path,
    out_h5ad: str | Path,
    cell_embeddings: bool = False,
    max_cells_dense: int = 200000,
    peak_activity: bool = False,
    motif_tf: str | Path | None = None,
) -> Path:
    """Build ``out_h5ad`` from the graph AnnData and a trained checkpoint.

    ``peak_activity=True`` also stores the RE x target activity matrix
    (``varm['Peak_Activity']``), which is large for genome-scale screens.
    ``motif_tf`` (motif-to-TF table) also stores both TF-target gene scores
    (``obsm['TF_TG_influence']``, ``obsm['TF_TG_connectivity']``, gene x TF; TF names in
    ``uns['TF_names']``), identical to those of
    :func:`rega_perturb.analysis.tf_tg_scores.compute_tf_tg_scores`.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

    logger.info(f"Input  h5ad     : {input_h5ad}")
    logger.info(f"Checkpoint      : {checkpoint}")
    logger.info(f"Output h5ad     : {out_h5ad}")

    adata, genes_master, RE_master, motif_names, has_gex = load_input_anndata(input_h5ad)

    ckpt = load_perturb_result(checkpoint)

    chk = consistency_checks(ckpt, genes_master, RE_master, motif_names)
    W_blocks = chk["W_blocks"]
    B_blocks = chk["B_blocks"]
    genes_blocks = chk["genes_blocks"]
    RE_blocks = chk["RE_blocks"]
    V = chk["V"]
    H = chk["H"]
    perturb_targets = chk["perturb_targets"]
    cells = chk["cells"]
    target_idx = chk["target_idx"]
    p_values = chk["p_values"]

    W_learned = assemble_W_learned(W_blocks, genes_blocks, RE_blocks, genes_master, RE_master)
    B_learned = assemble_B_learned(B_blocks, RE_blocks, RE_master, motif_names)

    P_info = build_P(target_idx, p_values, perturb_targets, cells)

    results = compute_downstream(
        adata, W_learned, B_learned, V, H,
        motif_names, has_gex,
        P_info, perturb_targets, cells, target_idx, p_values,
        cell_embeddings=cell_embeddings,
        max_cells_dense=max_cells_dense,
        peak_activity=peak_activity,
    )

    tf_tg, TF_names = {}, None
    if motif_tf is not None:
        from rega_perturb.analysis.model_io import load_trained_model
        from rega_perturb.analysis.tf_tg_scores import SCORES, compute_tf_tg_scores

        logger.info("\n[Step 5c] Computing TF-target gene scores ...")
        model = load_trained_model(checkpoint, input_h5ad)
        if not np.array_equal(model.genes, genes_master):
            raise ValueError("Gene order of the model and the graph AnnData differ.")
        for name in SCORES:
            S, _, TF_names = compute_tf_tg_scores(model, motif_tf, name)
            tf_tg[name] = S.astype(np.float32)
        del model

    build_result_anndata(
        adata, W_learned, B_learned, V, H,
        results, P_info, ckpt, out_h5ad,
        motif_names, TF_names, tf_tg,
        perturb_targets, cells, target_idx, p_values,
    )

    print_summary(W_learned, B_learned, results, P_info, out_h5ad, tf_tg)
    write_run_record(
        Path(out_h5ad).parent,
        "assemble",
        params=dict(cell_embeddings=cell_embeddings, max_cells_dense=max_cells_dense,
                    peak_activity=peak_activity),
        inputs={"input_h5ad": input_h5ad, "checkpoint": checkpoint,
                **({"motif_tf": motif_tf} if motif_tf is not None else {})},
    )
    return Path(out_h5ad)
