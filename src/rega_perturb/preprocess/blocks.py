"""Stream the raw expression matrix into per-chromosome model input blocks.

For each chromosome shared by genes and REs, writes ``<chrom>.npz`` containing the
expression block (genes x retained cells; log-CPM by default), the RE-TG prior
block and the RE x motif block. Also writes the per-cell perturbation assignment
(``P_perturb_target.npz``, ``P_cells.txt``, ``P_targets.txt``) and
``blocks_manifest.csv``. The raw h5ad is read in backed mode, one chromosome at a
time.
"""

from __future__ import annotations

import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import sparse

from rega_perturb.utils import chrom_sort_key as chrom_key
from rega_perturb.utils import get_logger, write_run_record

logger = get_logger("preprocess.blocks")


def to_csr(x) -> sp.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))


def get_peak_motif_matrix(graph: ad.AnnData) -> tuple[sp.csr_matrix, np.ndarray]:
    if "Peak_Motif" in graph.varm:
        B = graph.varm["Peak_Motif"]
    elif "Peak_Motif" in graph.uns:
        B = graph.uns["Peak_Motif"]
    else:
        raise ValueError("Missing Peak_Motif in graph.varm or graph.uns.")

    B = to_csr(B)

    if B.shape[0] != graph.n_vars:
        raise ValueError(
            f"Peak_Motif rows {B.shape[0]} != graph peaks {graph.n_vars}"
        )

    if "motif_names" not in graph.uns:
        raise ValueError("graph.uns['motif_names'] missing.")

    motif_names = np.asarray(graph.uns["motif_names"]).astype(str)

    if B.shape[1] != len(motif_names):
        raise ValueError(
            f"Peak_Motif columns {B.shape[1]} != motif_names {len(motif_names)}"
        )

    return B.tocsr(), motif_names


def save_full_chr_block(
    path: Path,
    X: sp.csr_matrix,
    W: sp.csr_matrix,
    B: sp.csr_matrix,
    genes,
    peaks,
    cells,
    motifs,
    gene_chr,
    peak_chr,
) -> None:
    X = X.tocsr().astype(np.float32)
    W = W.tocsr().astype(np.float32)
    B = B.tocsr().astype(np.float32)

    payload = {
        "X_data": X.data.astype(np.float32),
        "X_indices": X.indices.astype(np.int64),
        "X_indptr": X.indptr.astype(np.int64),
        "X_shape": np.asarray(X.shape, dtype=np.int64),

        "W_data": W.data.astype(np.float32),
        "W_indices": W.indices.astype(np.int64),
        "W_indptr": W.indptr.astype(np.int64),
        "W_shape": np.asarray(W.shape, dtype=np.int64),

        "B_data": B.data.astype(np.float32),
        "B_indices": B.indices.astype(np.int64),
        "B_indptr": B.indptr.astype(np.int64),
        "B_shape": np.asarray(B.shape, dtype=np.int64),

        "genes": np.asarray(genes, dtype=np.str_),
        "RE": np.asarray(peaks, dtype=np.str_),
        "cells": np.asarray(cells, dtype=np.str_),
        "motifs": np.asarray(motifs, dtype=np.str_),
        "gene_chr": np.asarray(gene_chr, dtype=np.str_),
        "re_chr": np.asarray(peak_chr, dtype=np.str_),
    }

    np.savez_compressed(str(path), **payload)


def build_and_save_P(
    cell_meta: pd.DataFrame,
    perturb_col: str,
    output_dir: Path,
) -> tuple[sp.csr_matrix, np.ndarray]:
    if perturb_col not in cell_meta.columns:
        raise ValueError(f"cell_metadata missing perturbation column '{perturb_col}'.")

    labels = cell_meta[perturb_col].astype(str).fillna("NA").values
    targets = sorted(pd.unique(labels).tolist())

    target_to_idx = {t: i for i, t in enumerate(targets)}

    row_idx = np.arange(len(labels), dtype=np.int64)
    col_idx = np.array([target_to_idx[x] for x in labels], dtype=np.int64)
    data = np.ones(len(labels), dtype=np.int8)

    P = sp.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(labels), len(targets)),
        dtype=np.int8,
    ).tocsr()

    sp.save_npz(str(output_dir / "P_perturb_target.npz"), P)

    (output_dir / "P_cells.txt").write_text(
        "\n".join(cell_meta.index.astype(str)) + "\n"
    )

    (output_dir / "P_targets.txt").write_text(
        "\n".join(targets) + "\n"
    )

    logger.info(f"[P] shape={P.shape}, nnz={P.nnz}")
    logger.info(f"[write] {output_dir / 'P_perturb_target.npz'}")

    return P, np.asarray(targets, dtype=np.str_)


def map_cells_to_raw_indices(cell_meta: pd.DataFrame, raw: ad.AnnData) -> np.ndarray:
    if "raw_obs_idx" in cell_meta.columns:
        raw_cell_idx = cell_meta["raw_obs_idx"].astype(int).values

        if np.any(raw_cell_idx < 0) or np.any(raw_cell_idx >= raw.n_obs):
            raise ValueError("Some raw_obs_idx values are outside raw.n_obs.")

        raw_names_check = raw.obs_names[raw_cell_idx].astype(str)
        cell_names = cell_meta.index.astype(str).values

        if not np.array_equal(raw_names_check, cell_names):
            mismatch = np.where(raw_names_check != cell_names)[0]
            first = int(mismatch[0])
            raise ValueError(
                "raw_obs_idx does not match cell metadata index.\n"
                f"First mismatch: metadata={cell_names[first]}, raw={raw_names_check[first]}"
            )

        return raw_cell_idx.astype(np.int64)

    raw_cell_to_idx = {
        c: i for i, c in enumerate(raw.obs_names.astype(str))
    }

    missing = [c for c in cell_meta.index.astype(str) if c not in raw_cell_to_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} cells in metadata not found in raw h5ad. "
            f"Examples: {missing[:10]}"
        )

    return np.array(
        [raw_cell_to_idx[c] for c in cell_meta.index.astype(str)],
        dtype=np.int64,
    )


def map_graph_genes_to_raw_indices(
    graph_genes: np.ndarray,
    gene_meta: pd.DataFrame,
    raw: ad.AnnData,
    gene_col: str,
) -> np.ndarray:
    if "raw_var_idx" in gene_meta.columns and gene_col in gene_meta.columns:
        gm = gene_meta.copy()
        gm[gene_col] = gm[gene_col].astype(str)

        duplicated = gm[gm[gene_col].duplicated()][gene_col].unique()
        if len(duplicated) > 0:
            logger.info(
                f"[gene mapping] WARNING: {len(duplicated)} duplicated gene symbols "
                "in gene metadata. Keeping first."
            )

        gene_to_raw_idx = (
            gm.drop_duplicates(gene_col, keep="first")
            .set_index(gene_col)["raw_var_idx"]
            .astype(int)
            .to_dict()
        )
    else:
        if gene_col in raw.var.columns:
            raw_symbols = raw.var[gene_col].astype(str).values
        else:
            raw_symbols = raw.var_names.astype(str)

        gene_to_raw_idx = (
            pd.DataFrame({
                "gene_name": raw_symbols,
                "raw_var_idx": np.arange(raw.n_vars, dtype=np.int64),
            })
            .drop_duplicates("gene_name", keep="first")
            .set_index("gene_name")["raw_var_idx"]
            .to_dict()
        )

    missing = [g for g in graph_genes if g not in gene_to_raw_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} graph genes not found in raw/gene metadata. "
            f"Examples: {missing[:10]}"
        )

    raw_gene_idx = np.array([gene_to_raw_idx[g] for g in graph_genes], dtype=np.int64)

    if np.any(raw_gene_idx < 0) or np.any(raw_gene_idx >= raw.n_vars):
        raise ValueError("Some raw gene indices are outside raw.n_vars.")

    return raw_gene_idx


def read_raw_expression_chunk(
    raw: ad.AnnData,
    raw_cell_idx: np.ndarray,
    raw_gene_idx: np.ndarray,
) -> sp.csr_matrix:
    # raw.X is cell x gene
    chunk = raw.X[raw_cell_idx, :]

    if sparse.issparse(chunk):
        chunk = chunk.tocsr()[:, raw_gene_idx]
    else:
        chunk = np.asarray(chunk)[:, raw_gene_idx]
        chunk = sp.csr_matrix(chunk.astype(np.float32, copy=False))

    return chunk.tocsr().astype(np.float32)


def transform_expression(
    X_cell_gene: sp.csr_matrix,
    library_size: np.ndarray,
    expr_mode: str,
    target_sum: float,
) -> sp.csr_matrix:
    if expr_mode in ("counts", "as_is"):
        return X_cell_gene.tocsr().astype(np.float32)

    if expr_mode in ("logcpm", "logcp10k"):
        lib = library_size.astype(np.float32)
        lib_safe = np.where(lib == 0, 1.0, lib)
        scale = (target_sum / lib_safe).astype(np.float32)

        X = sp.diags(scale).dot(X_cell_gene).tocsr()
        X.data = np.log1p(X.data).astype(np.float32)

        return X

    raise ValueError(f"Unknown expr_mode: {expr_mode}")


def build_blocks(
    graph_h5ad: str | Path,
    raw_h5ad: str | Path,
    cell_metadata: str | Path,
    gene_metadata: str | Path,
    output_dir: str | Path,
    expr_mode: str = "logcp10k",
    target_sum: float | None = None,
    chunk_size: int = 5000,
    perturb_col: str = "gene",
    gene_col: str = "gene_name",
    chroms: list[str] | None = None,
    skip_p: bool = False,
) -> Path:
    """Write model input blocks to ``output_dir`` and return it.

    ``expr_mode``:

    * ``"logcp10k"`` (default) -- ``log1p(count / library_size * 1e4)``, with
      ``library_size`` taken from the cell metadata;
    * ``"logcpm"`` -- the same with a scaling of ``1e6`` (log-CPM);
    * ``"as_is"`` -- use the values of the input matrix unchanged (for data that is
      already normalized; see ``input_type`` of :func:`rega_perturb.preprocess.qc.run_qc`);
    * ``"counts"`` -- raw counts unchanged.

    ``target_sum`` overrides the scaling of the two log modes (default: 1e4 for
    ``"logcp10k"``, 1e6 for ``"logcpm"``). ``perturb_col``
    is the cell-metadata column holding the perturbation target of each cell.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[load] graph AnnData")
    graph = ad.read_h5ad(graph_h5ad)

    logger.info("[load] raw h5ad backed")
    raw = ad.read_h5ad(raw_h5ad, backed="r")

    logger.info("[load] metadata")
    cell_meta = pd.read_csv(cell_metadata, index_col=0)
    gene_meta = pd.read_csv(gene_metadata, index_col=0)

    if expr_mode not in ("logcp10k", "logcpm", "as_is", "counts"):
        raise ValueError(f"Unknown expr_mode: {expr_mode}")
    if target_sum is None:
        target_sum = 1e6 if expr_mode == "logcpm" else 1e4
    log_mode = expr_mode in ("logcp10k", "logcpm")

    if log_mode and "library_size" not in cell_meta.columns:
        raise ValueError(
            "cell_metadata must contain 'library_size' for log normalization "
            "(use expr_mode='as_is' for already normalized data)."
        )

    if graph.obs_names.has_duplicates:
        raise ValueError("graph.obs_names contains duplicate genes.")

    if graph.var_names.has_duplicates:
        raise ValueError("graph.var_names contains duplicate peaks.")

    if "chrom" not in graph.obs.columns:
        raise ValueError("graph.obs missing 'chrom'.")

    if "peak_chrom" not in graph.var.columns:
        raise ValueError("graph.var missing 'peak_chrom'.")

    W_all = graph.X
    if not sparse.issparse(W_all):
        raise ValueError("graph.X must be sparse.")

    W_all = W_all.tocsr().astype(np.float32)

    B_all, motif_names = get_peak_motif_matrix(graph)
    B_all = B_all.tocsr().astype(np.float32)

    graph_genes = graph.obs_names.astype(str).values
    graph_peaks = graph.var_names.astype(str).values

    raw_gene_idx_all = map_graph_genes_to_raw_indices(
        graph_genes=graph_genes,
        gene_meta=gene_meta,
        raw=raw,
        gene_col=gene_col,
    )

    raw_cell_idx_all = map_cells_to_raw_indices(
        cell_meta=cell_meta,
        raw=raw,
    )

    cell_names = cell_meta.index.astype(str).values

    if log_mode:
        library_size_all = cell_meta["library_size"].astype(float).values
    else:
        library_size_all = np.ones(len(cell_names), dtype=np.float32)

    logger.info("\n[check] Dimensions")
    logger.info(f"  graph W: {W_all.shape}, nnz={W_all.nnz:,}")
    logger.info(f"  B: {B_all.shape}, nnz={B_all.nnz:,}")
    logger.info(f"  raw: cells={raw.n_obs:,}, genes={raw.n_vars:,}")
    logger.info(f"  retained cells={len(raw_cell_idx_all):,}")
    logger.info(f"  graph genes={len(graph_genes):,}")
    logger.info(f"  motifs={len(motif_names):,}")

    if not skip_p:
        build_and_save_P(
            cell_meta=cell_meta,
            perturb_col=perturb_col,
            output_dir=output_dir,
        )

    gene_chrom = graph.obs["chrom"].astype(str).values
    peak_chrom = graph.var["peak_chrom"].astype(str).values

    common_chroms = sorted(set(gene_chrom) & set(peak_chrom), key=chrom_key)

    if chroms is not None:
        wanted = set(chroms)
        process_chroms = [c for c in common_chroms if c in wanted]
    else:
        process_chroms = common_chroms

    if len(process_chroms) == 0:
        raise ValueError(f"No chromosomes selected. Common chromosomes: {common_chroms}")

    manifest_records = []

    for chrom in process_chroms:
        logger.info(f"\n[chrom] {chrom}")

        g_idx = np.where(gene_chrom == chrom)[0]
        p_idx = np.where(peak_chrom == chrom)[0]

        if len(g_idx) == 0 or len(p_idx) == 0:
            logger.info(f"  skipped empty chrom={chrom}")
            continue

        raw_gene_idx_chr = raw_gene_idx_all[g_idx]

        W_chr = W_all[g_idx, :][:, p_idx].tocsr().astype(np.float32)
        B_chr = B_all[p_idx, :].tocsr().astype(np.float32)

        if W_chr.shape != (len(g_idx), len(p_idx)):
            raise ValueError(f"{chrom}: W_chr shape error.")

        if B_chr.shape != (len(p_idx), len(motif_names)):
            raise ValueError(f"{chrom}: B_chr shape error.")

        X_chunks = []

        for start in range(0, len(raw_cell_idx_all), chunk_size):
            end = min(start + chunk_size, len(raw_cell_idx_all))

            raw_cell_idx = raw_cell_idx_all[start:end]
            library_chunk = library_size_all[start:end]

            X_cell_gene = read_raw_expression_chunk(
                raw=raw,
                raw_cell_idx=raw_cell_idx,
                raw_gene_idx=raw_gene_idx_chr,
            )

            X_cell_gene = transform_expression(
                X_cell_gene=X_cell_gene,
                library_size=library_chunk,
                expr_mode=expr_mode,
                target_sum=target_sum,
            )

            # Convert cell x gene_chr to gene_chr x cell_chunk.
            X_gene_cell = X_cell_gene.T.tocsr().astype(np.float32)

            if X_gene_cell.shape != (len(g_idx), end - start):
                raise ValueError(
                    f"{chrom} cells {start}-{end}: "
                    f"X shape {X_gene_cell.shape} expected {(len(g_idx), end - start)}"
                )

            X_chunks.append(X_gene_cell)

            logger.info(
                f"  chunk cells {start:,}-{end:,}: "
                f"X{X_gene_cell.shape}, nnz={X_gene_cell.nnz:,}"
            )

            del X_cell_gene, X_gene_cell
            gc.collect()

        logger.info(f"  stacking {len(X_chunks)} X chunks for {chrom}...")

        X_chr = sp.hstack(X_chunks, format="csr").astype(np.float32)

        del X_chunks
        gc.collect()

        if X_chr.shape != (len(g_idx), len(raw_cell_idx_all)):
            raise ValueError(
                f"{chrom}: final X shape {X_chr.shape} expected "
                f"{(len(g_idx), len(raw_cell_idx_all))}"
            )

        if X_chr.shape[0] != W_chr.shape[0]:
            raise ValueError(f"{chrom}: X genes != W genes.")

        if W_chr.shape[1] != B_chr.shape[0]:
            raise ValueError(f"{chrom}: W peaks != B peaks.")

        out_file = output_dir / f"{chrom}.npz"

        save_full_chr_block(
            path=out_file,
            X=X_chr,
            W=W_chr,
            B=B_chr,
            genes=graph_genes[g_idx],
            peaks=graph_peaks[p_idx],
            cells=cell_names,
            motifs=motif_names,
            gene_chr=gene_chrom[g_idx],
            peak_chr=peak_chrom[p_idx],
        )

        logger.info(
            f"  saved {out_file.name}: "
            f"X{X_chr.shape}, nnz={X_chr.nnz:,}; "
            f"W{W_chr.shape}, nnz={W_chr.nnz:,}; "
            f"B{B_chr.shape}, nnz={B_chr.nnz:,}"
        )

        manifest_records.append({
            "chrom": chrom,
            "file": out_file.name,
            "n_genes": len(g_idx),
            "n_peaks": len(p_idx),
            "n_cells": len(raw_cell_idx_all),
            "n_motifs": len(motif_names),
            "X_nnz": int(X_chr.nnz),
            "W_nnz": int(W_chr.nnz),
            "B_nnz": int(B_chr.nnz),
            "expr_mode": expr_mode,
        })

        del X_chr, W_chr, B_chr
        gc.collect()

    manifest = pd.DataFrame(manifest_records)
    manifest_file = output_dir / "blocks_manifest.csv"
    manifest.to_csv(manifest_file, index=False)

    raw.file.close()

    write_run_record(
        output_dir,
        "build_blocks",
        params=dict(expr_mode=expr_mode, target_sum=target_sum, chunk_size=chunk_size,
                    perturb_col=perturb_col, gene_col=gene_col, chroms=chroms, skip_p=skip_p),
        inputs={"graph_h5ad": graph_h5ad, "raw_h5ad": raw_h5ad,
                "cell_metadata": cell_metadata, "gene_metadata": gene_metadata},
    )

    logger.info("\nDone.")
    logger.info(f"Saved manifest: {manifest_file}")
    logger.info(f"Output dir: {output_dir}")
    return output_dir
