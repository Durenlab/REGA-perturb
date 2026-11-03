"""Cell and gene quality control for a raw Perturb-seq AnnData (streamed).

The raw count matrix is read in chunks (backed mode), so the full matrix is never
loaded into memory. Outputs, written to ``out_dir`` with file prefix ``prefix``:

* ``<prefix>_REGA_qc_cell_metadata.csv`` -- retained cells with QC metrics,
  ``library_size`` and size factors;
* ``<prefix>_REGA_qc_gene_metadata.csv`` -- retained genes;
* ``<prefix>_REGA_qc_summary.csv`` -- number of cells / genes after each step;
* ``<prefix>_REGA_qc_gene_TSS_300kb.txt`` -- TSS +/- window of each retained gene
  (BED-like, no header), used to build the RE-TG table.

The mitochondrial content is handled as a fraction in [0, 1] (column
``mitopercent`` of ``adata.obs`` in the Replogle et al. format).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from rega_perturb.utils import get_logger, write_run_record

logger = get_logger("preprocess.qc")

STANDARD_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def compute_row_qc(X, n_obs: int, chunk_size: int, require_counts: bool = True):
    """Compute n_genes_by_counts and total_counts_from_X in chunks.

    With ``require_counts`` the first chunk must be non-negative and integer-like
    (raw UMI counts); otherwise only non-negativity is required.
    """
    n_genes_by_counts = np.zeros(n_obs, dtype=np.int32)
    total_counts_from_X = np.zeros(n_obs, dtype=np.float64)

    logger.info("\n[preprocess] Computing per-cell QC from X...")

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        idx = np.arange(start, end)

        chunk = X[idx, :]

        if sparse.issparse(chunk):
            chunk = chunk.tocsr()
            n_genes_by_counts[start:end] = np.asarray((chunk > 0).sum(axis=1)).ravel()
            total_counts_from_X[start:end] = np.asarray(chunk.sum(axis=1)).ravel()

            if start == 0:
                vals = chunk.data
        else:
            chunk = np.asarray(chunk)
            n_genes_by_counts[start:end] = np.count_nonzero(chunk > 0, axis=1)
            total_counts_from_X[start:end] = chunk.sum(axis=1)

            if start == 0:
                vals = chunk.ravel()

        if start == 0:
            vals = vals[~np.isnan(vals)]
            logger.info(f"Sample min value: {vals.min()}")
            logger.info(f"Sample max value: {vals.max()}")
            logger.info(f"Integer-like: {np.allclose(vals, np.round(vals))}")

            if vals.min() < 0:
                raise ValueError("X contains negative values; not raw counts.")
            if require_counts and not np.allclose(vals, np.round(vals)):
                raise ValueError(
                    "X is not integer-like; not raw UMI counts. "
                    "Use input_type='normalized' for already normalized data."
                )

        logger.info(f"  processed cells {start:,}-{end:,}")

        del chunk
        gc.collect()

    return n_genes_by_counts, total_counts_from_X


def compute_mt_fraction(X, mt_gene_idx, total_counts, chunk_size: int):
    """Fraction of counts from mitochondrial genes per cell (0-1)."""
    n_obs = len(total_counts)
    mt_counts = np.zeros(n_obs, dtype=np.float64)

    logger.info("\n[preprocess] Computing mitochondrial fraction from X...")

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        idx = np.arange(start, end)

        chunk = X[idx, :]

        if sparse.issparse(chunk):
            chunk = chunk.tocsr()[:, mt_gene_idx]
            mt_counts[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
        else:
            chunk = np.asarray(chunk)[:, mt_gene_idx]
            mt_counts[start:end] = chunk.sum(axis=1)

        logger.info(f"  processed cells {start:,}-{end:,}")

        del chunk
        gc.collect()

    total_safe = np.where(total_counts == 0, 1.0, total_counts)
    return mt_counts / total_safe


def compute_gene_n_cells(X, keep_cells_mask, n_vars: int, chunk_size: int):
    """Compute number of retained cells expressing each gene."""
    keep_idx = np.where(keep_cells_mask)[0]
    gene_n_cells = np.zeros(n_vars, dtype=np.int64)

    logger.info("\n[preprocess] Computing n_cells per gene among retained cells...")

    for start in range(0, len(keep_idx), chunk_size):
        end = min(start + chunk_size, len(keep_idx))
        idx = keep_idx[start:end]

        chunk = X[idx, :]

        if sparse.issparse(chunk):
            chunk = chunk.tocsr()
            gene_n_cells += np.asarray((chunk > 0).sum(axis=0)).ravel()
        else:
            chunk = np.asarray(chunk)
            gene_n_cells += np.count_nonzero(chunk > 0, axis=0)

        logger.info(f"  processed retained cells {start:,}-{end:,}")

        del chunk
        gc.collect()

    return gene_n_cells


def compute_final_cell_qc(X, keep_cells_mask, keep_genes_mask, chunk_size: int):
    """Compute final n_genes and library size after both cell and gene filtering."""
    keep_idx = np.where(keep_cells_mask)[0]
    keep_genes = np.where(keep_genes_mask)[0]

    n_genes_final = np.zeros(len(keep_idx), dtype=np.int32)
    library_size = np.zeros(len(keep_idx), dtype=np.float64)

    logger.info("\n[preprocess] Computing final cell QC after gene filtering...")

    out_pos = 0

    for start in range(0, len(keep_idx), chunk_size):
        end = min(start + chunk_size, len(keep_idx))
        idx = keep_idx[start:end]

        chunk = X[idx, :]

        if sparse.issparse(chunk):
            chunk = chunk.tocsr()[:, keep_genes]
            n_genes_final[out_pos:out_pos + len(idx)] = np.asarray((chunk > 0).sum(axis=1)).ravel()
            library_size[out_pos:out_pos + len(idx)] = np.asarray(chunk.sum(axis=1)).ravel()
        else:
            chunk = np.asarray(chunk)[:, keep_genes]
            n_genes_final[out_pos:out_pos + len(idx)] = np.count_nonzero(chunk > 0, axis=1)
            library_size[out_pos:out_pos + len(idx)] = chunk.sum(axis=1)

        logger.info(f"  processed retained cells {start:,}-{end:,}")

        out_pos += len(idx)

        del chunk
        gc.collect()

    return n_genes_final, library_size


def record_step(records, step, n_cells, n_genes):
    records.append({
        "step": step,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    })


GENE_COORD_COLUMNS = ["chr", "start", "end", "strand"]


def annotate_genes_from_tss_reference(var: pd.DataFrame, tss_reference: str | Path) -> pd.DataFrame:
    """Fill ``chr``, ``start``, ``end`` and ``strand`` of ``var`` from a TSS reference table.

    ``tss_reference`` is tab-separated without header: chrom, TSS position, gene symbol,
    strand (e.g. ``reference/tss_hg38.txt``). Genes are matched by symbol (``gene_name``
    when present, otherwise the var index); ``start`` and ``end`` are both set to the TSS.
    Genes absent from the reference get missing coordinates and ``tss_annotated = False``.
    """
    ref = pd.read_csv(tss_reference, sep="\t", header=None, usecols=[0, 1, 2, 3],
                      names=["chr", "tss", "symbol", "strand"],
                      dtype={"chr": str, "symbol": str, "strand": str})
    dup = ref["symbol"][ref["symbol"].duplicated()].unique()
    if len(dup) > 0:
        raise ValueError(f"TSS reference has duplicated gene symbols, e.g. {list(dup[:5])}")
    ref = ref.set_index("symbol")

    var = var.copy()
    if "gene_name" in var.columns:
        symbols = var["gene_name"].astype(str)
    else:
        symbols = pd.Series(var.index.astype(str), index=var.index)
    hit = symbols.isin(ref.index).values
    matched = ref.reindex(symbols.values)
    var["chr"] = matched["chr"].values
    var["start"] = matched["tss"].values
    var["end"] = matched["tss"].values
    var["strand"] = matched["strand"].values
    var["tss_annotated"] = hit
    logger.info(f"[preprocess] TSS reference {tss_reference}: {int(hit.sum()):,} of {len(var):,} "
                f"genes annotated; {int((~hit).sum()):,} genes not found are removed.")
    return var


def write_tss_file(var_final: pd.DataFrame, output_tss: str, tss_window: int):
    logger.info("\n[preprocess] Generating TSS +/- window file...")

    required_cols = ["chr", "start", "end", "strand"]
    missing = [c for c in required_cols if c not in var_final.columns]
    if missing:
        raise ValueError(f"Cannot generate TSS file; missing var columns: {missing}")

    gene_anno = var_final.copy()

    if "gene_name" in gene_anno.columns:
        gene_anno["tss_gene_name"] = gene_anno["gene_name"].astype(str)
    else:
        gene_anno["tss_gene_name"] = gene_anno.index.astype(str)

    gene_anno["chr"] = gene_anno["chr"].astype(str)
    gene_anno["strand"] = gene_anno["strand"].astype(str)
    gene_anno["start"] = pd.to_numeric(gene_anno["start"], errors="coerce")
    gene_anno["end"] = pd.to_numeric(gene_anno["end"], errors="coerce")

    valid = (
        gene_anno["chr"].notna()
        & gene_anno["start"].notna()
        & gene_anno["end"].notna()
        & gene_anno["strand"].isin(["+", "-"])
    )

    n_before = gene_anno.shape[0]
    gene_anno = gene_anno.loc[valid].copy()

    logger.info(f"Genes before TSS filtering: {n_before:,}")
    logger.info(f"Genes with valid TSS: {gene_anno.shape[0]:,}")
    logger.info(f"Genes removed from TSS file: {n_before - gene_anno.shape[0]:,}")

    gene_anno["start"] = gene_anno["start"].astype(int)
    gene_anno["end"] = gene_anno["end"].astype(int)

    gene_anno["TSS"] = np.where(
        gene_anno["strand"] == "+",
        gene_anno["start"],
        gene_anno["end"]
    ).astype(int)

    gene_anno["region_start"] = (gene_anno["TSS"] - tss_window).clip(lower=0).astype(int)
    gene_anno["region_end"] = (gene_anno["TSS"] + tss_window).astype(int)

    out = gene_anno[
        ["chr", "region_start", "region_end", "tss_gene_name", "TSS", "strand"]
    ].copy()

    out = out.sort_values(["chr", "region_start", "region_end", "tss_gene_name"])

    out.to_csv(output_tss, sep="\t", header=False, index=False)

    logger.info(f"Saved TSS file: {output_tss}")


def run_qc(
    input_h5ad: str | Path,
    out_dir: str | Path,
    prefix: str,
    chunk_size: int = 5000,
    min_umi: float = 1000,
    max_umi: float = 50000,
    min_genes: float = 500,
    max_genes: float | None = None,
    max_mt: float = 0.2,
    min_cells_per_gene: int = 30,
    keep_mt_genes: bool = False,
    keep_nonstandard_chromosomes: bool = False,
    tss_window: int = 300000,
    input_type: str = "counts",
    tss_reference: str | Path | None = None,
) -> dict[str, Path]:
    """Run cell / gene QC on a raw Perturb-seq h5ad and write metadata tables.

    Parameters
    ----------
    input_h5ad
        Raw AnnData (cells x genes, integer UMI counts). ``var`` must contain
        ``chr``, ``start``, ``end`` and ``strand``; ``gene_name`` is used as the
        gene symbol when present.
    max_genes
        Optional maximum number of detected genes per cell; ``None`` (default)
        applies no upper limit.
    max_mt
        Maximum mitochondrial fraction (0-1).
    input_type
        ``"counts"`` (raw UMI counts; default) or ``"normalized"`` (values already
        normalized by the user, e.g. log library-size normalized, non-negative).
        For normalized input no cell filtering is applied (cells are assumed to be
        quality controlled already); gene filtering and the TSS file are produced
        as for counts. Use ``expr_mode="as_is"`` in
        :func:`rega_perturb.preprocess.blocks.build_blocks` for such data.
    tss_reference
        Optional TSS table (chrom, TSS, gene symbol, strand; no header), used only when
        ``var`` lacks ``chr`` / ``start`` / ``end`` / ``strand``: gene coordinates are then
        taken from this table by gene symbol, and genes not found in it are removed.

    Returns
    -------
    dict
        Paths of the written files (``summary``, ``cell_metadata``,
        ``gene_metadata``, ``tss``).
    """
    if input_type not in ("counts", "normalized"):
        raise ValueError(f"input_type must be 'counts' or 'normalized', got {input_type!r}")
    normalized = input_type == "normalized"

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    output_summary = out_dir / f"{prefix}_REGA_qc_summary.csv"
    output_obs = out_dir / f"{prefix}_REGA_qc_cell_metadata.csv"
    output_var = out_dir / f"{prefix}_REGA_qc_gene_metadata.csv"
    output_tss = out_dir / f"{prefix}_REGA_qc_gene_TSS_300kb.txt"

    logger.info("[preprocess] Loading AnnData in backed mode...")
    adata_b = ad.read_h5ad(input_h5ad, backed="r")

    n_obs = adata_b.n_obs
    n_vars = adata_b.n_vars

    logger.info(str(adata_b))
    logger.info(f"Initial shape: cells={n_obs:,}, genes={n_vars:,}")
    logger.info(f"X type: {type(adata_b.X)}")

    obs = adata_b.obs.copy()
    var = adata_b.var.copy()

    obs.index = adata_b.obs_names.astype(str)
    var.index = adata_b.var_names.astype(str)

    obs["raw_obs_name"] = obs.index.astype(str)
    obs["raw_obs_idx"] = np.arange(n_obs, dtype=np.int64)

    var["raw_var_name"] = var.index.astype(str)
    var["raw_var_idx"] = np.arange(n_vars, dtype=np.int64)

    use_tss_reference = False
    if tss_reference is not None:
        if all(c in var.columns for c in GENE_COORD_COLUMNS):
            logger.info("var already has gene coordinates; tss_reference is not used.")
        else:
            var = annotate_genes_from_tss_reference(var, tss_reference)
            use_tss_reference = True

    summary_records = []
    record_step(summary_records, "raw_loaded_backed", n_obs, n_vars)

    logger.info("\n[preprocess] Annotating genes...")

    if "gene_name" in var.columns:
        gene_symbol = var["gene_name"].astype(str)
    else:
        gene_symbol = pd.Series(var.index.astype(str), index=var.index)

    gene_symbol_upper = gene_symbol.str.upper()

    var["mt"] = gene_symbol_upper.str.startswith("MT-")
    var["ribo"] = gene_symbol_upper.str.startswith(("RPS", "RPL"))

    logger.info(f"MT genes: {int(var['mt'].sum())}")
    logger.info(f"Ribosomal genes: {int(var['ribo'].sum())}")

    n_genes_by_counts, total_counts_from_X = compute_row_qc(
        X=adata_b.X,
        n_obs=n_obs,
        chunk_size=chunk_size,
        require_counts=not normalized,
    )

    obs["n_genes_by_counts"] = n_genes_by_counts
    obs["total_counts_from_X"] = total_counts_from_X

    if "UMI_count" not in obs.columns:
        obs["UMI_count"] = obs["total_counts_from_X"].values

    if normalized:
        if "mitopercent" not in obs.columns:
            obs["mitopercent"] = np.nan
    elif "mitopercent" not in obs.columns:
        mt_idx = np.where(var["mt"].values)[0]
        obs["mitopercent"] = compute_mt_fraction(
            X=adata_b.X,
            mt_gene_idx=mt_idx,
            total_counts=total_counts_from_X,
            chunk_size=chunk_size,
        )
    elif float(np.nanmax(obs["mitopercent"].astype(float).values)) > 1.0:
        logger.warning(
            "obs['mitopercent'] has values > 1 and looks like a percentage; "
            "it is divided by 100 so that max_mt is applied as a fraction."
        )
        obs["mitopercent"] = obs["mitopercent"].astype(float) / 100.0

    logger.info("\n[preprocess] QC metric summary before filtering:")
    logger.info(
        "\n" + str(obs[
            ["UMI_count", "total_counts_from_X", "n_genes_by_counts", "mitopercent"]
        ].describe())
    )

    logger.info("\n[preprocess] Filtering cells...")

    if normalized:
        logger.info("input_type='normalized': cell filtering skipped (cells assumed QC'd).")
        cell_filter = np.ones(n_obs, dtype=bool)
    else:
        cell_filter = (
            (obs["UMI_count"].astype(float).values >= min_umi)
            & (obs["UMI_count"].astype(float).values <= max_umi)
            & (obs["n_genes_by_counts"].astype(float).values >= min_genes)
            & (obs["mitopercent"].astype(float).values <= max_mt)
        )
        if max_genes is not None:
            cell_filter &= obs["n_genes_by_counts"].astype(float).values <= max_genes

    n_cells_after = int(cell_filter.sum())
    if n_cells_after == 0:
        raise ValueError("No cells passed QC; check the thresholds (max_mt is a fraction, 0-1).")

    logger.info(f"Cells before QC: {n_obs:,}")
    logger.info(f"Cells after QC: {n_cells_after:,}")
    logger.info(f"Removed cells: {n_obs - n_cells_after:,}")

    record_step(summary_records, "after_cell_qc", n_cells_after, n_vars)

    gene_n_cells = compute_gene_n_cells(
        X=adata_b.X,
        keep_cells_mask=cell_filter,
        n_vars=n_vars,
        chunk_size=chunk_size,
    )

    var["n_cells"] = gene_n_cells

    gene_filter = gene_n_cells >= min_cells_per_gene

    logger.info("\n[preprocess] Filtering genes...")
    logger.info(f"Genes before gene QC: {n_vars:,}")
    logger.info(f"Genes after min_cells >= {min_cells_per_gene}: {int(gene_filter.sum()):,}")
    logger.info(f"Removed genes: {n_vars - int(gene_filter.sum()):,}")

    record_step(
        summary_records,
        f"after_gene_qc_min_cells_{min_cells_per_gene}",
        n_cells_after,
        int(gene_filter.sum()),
    )

    if use_tss_reference:
        gene_filter = gene_filter & var["tss_annotated"].values
        logger.info(f"Genes after keeping genes found in the TSS reference: {int(gene_filter.sum()):,}")
        record_step(summary_records, "after_tss_reference", n_cells_after, int(gene_filter.sum()))

    if not keep_mt_genes:
        before = int(gene_filter.sum())
        gene_filter = gene_filter & (~var["mt"].values)

        logger.info("\n[preprocess] Removing MT genes...")
        logger.info(f"Genes before removing MT: {before:,}")
        logger.info(f"Genes after removing MT: {int(gene_filter.sum()):,}")
        logger.info(f"Removed MT genes: {before - int(gene_filter.sum()):,}")

        record_step(summary_records, "after_remove_mt_genes", n_cells_after, int(gene_filter.sum()))

    if not keep_nonstandard_chromosomes:
        if "chr" not in var.columns:
            raise ValueError("var['chr'] is missing; cannot filter standard chromosomes.")

        before = int(gene_filter.sum())

        var["chr"] = var["chr"].astype(str)
        gene_filter = gene_filter & var["chr"].isin(STANDARD_CHROMOSOMES).values

        logger.info("\n[preprocess] Keeping only standard chromosomes...")
        logger.info(f"Genes before standard chromosome filtering: {before:,}")
        logger.info(f"Genes after standard chromosome filtering: {int(gene_filter.sum()):,}")
        logger.info(f"Removed non-standard chromosome genes: {before - int(gene_filter.sum()):,}")

        record_step(
            summary_records,
            "after_keep_standard_chromosomes",
            n_cells_after,
            int(gene_filter.sum()),
        )

    n_genes_final, library_size_final = compute_final_cell_qc(
        X=adata_b.X,
        keep_cells_mask=cell_filter,
        keep_genes_mask=gene_filter,
        chunk_size=chunk_size,
    )

    obs_final = obs.loc[cell_filter].copy()
    var_final = var.loc[gene_filter].copy()

    obs_final["n_genes_by_counts_final"] = n_genes_final
    if not normalized:
        obs_final["library_size"] = library_size_final

        median_library_size = np.median(library_size_final)

        if median_library_size <= 0:
            raise ValueError("Median library size is zero after final filtering.")

        obs_final["size_factor"] = library_size_final / median_library_size
        obs_final["log_size_factor"] = np.log(obs_final["size_factor"].values)

    for col in ["gem_group", "gene", "gene_id", "transcript", "gene_transcript", "sgID_AB"]:
        if col in obs_final.columns:
            obs_final[col] = obs_final[col].astype("category")

    record_step(summary_records, "final_qc_metadata", obs_final.shape[0], var_final.shape[0])

    adata_b.file.close()

    write_tss_file(
        var_final=var_final,
        output_tss=output_tss,
        tss_window=tss_window,
    )

    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv(output_summary, index=False)
    obs_final.to_csv(output_obs)
    var_final.to_csv(output_var)

    write_run_record(
        out_dir,
        "qc",
        params=dict(prefix=prefix, chunk_size=chunk_size, min_umi=min_umi, max_umi=max_umi,
                    min_genes=min_genes, max_genes=max_genes, max_mt=max_mt,
                    min_cells_per_gene=min_cells_per_gene, keep_mt_genes=keep_mt_genes,
                    keep_nonstandard_chromosomes=keep_nonstandard_chromosomes,
                    tss_window=tss_window, input_type=input_type,
                    tss_reference_used=use_tss_reference),
        inputs={"input_h5ad": input_h5ad,
                **({"tss_reference": tss_reference} if use_tss_reference else {})},
    )

    logger.info("\n[preprocess] Done.")
    logger.info(f"Saved QC summary: {output_summary}")
    logger.info(f"Saved cell metadata: {output_obs}")
    logger.info(f"Saved gene metadata: {output_var}")
    logger.info(f"Saved TSS annotation: {output_tss}")

    return {
        "summary": output_summary,
        "cell_metadata": output_obs,
        "gene_metadata": output_var,
        "tss": output_tss,
    }
