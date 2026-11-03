"""Build the regulatory graph inputs (RE-TG prior and RE x motif matrix).

Only gene metadata is needed (no expression matrix). Inputs:

* gene metadata CSV from :func:`rega_perturb.preprocess.qc.run_qc`;
* RE-TG table (tab-separated, with header ``chrom region_start region_end
  gene_name TSS strand peak_chrom peak_start peak_end peak_name``), see
  :func:`rega_perturb.preprocess.reference.build_re_tg`;
* peak-motif table (two tab-separated columns ``peak motif``, no header), see
  :func:`rega_perturb.preprocess.reference.run_motif_scan`.

Outputs in ``output_dir``: ``re_tg_matrix.h5``, ``input_binding.npz``,
``input_binding_peaks.txt``, ``input_binding_motifs.txt``, ``re_tg_genes.txt`` and
``REGA_graph_without_GEX.h5ad`` (genes x REs AnnData used by later steps).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix

from rega_perturb.utils import chrom_sort_key, get_logger, write_run_record

logger = get_logger("preprocess.graph")

GRAPH_H5AD = "REGA_graph_without_GEX.h5ad"


def sort_genes_by_coord(genes: set[str], re_tg: pd.DataFrame) -> list[str]:
    gene_coord = (
        re_tg[["gene_name", "chrom", "TSS"]]
        .drop_duplicates("gene_name", keep="first")
        .set_index("gene_name")
    )

    def key(g):
        if g in gene_coord.index:
            row = gene_coord.loc[g]
            return (chrom_sort_key(row["chrom"]), int(row["TSS"]), g)
        return ((9, 0, ""), 0, g)

    return sorted(genes, key=key)


def sort_peaks_by_coord(peaks: set[str]) -> list[str]:
    def key(p):
        parts = str(p).split("_")

        if len(parts) < 3:
            return ((9, 0, ""), 0, p)

        try:
            return (chrom_sort_key(parts[0]), int(parts[1]), p)
        except ValueError:
            return ((9, 0, ""), 0, p)

    return sorted(peaks, key=key)


def load_expression_gene_set(gene_metadata: Path, gene_col: str) -> tuple[set[str], pd.DataFrame]:
    gene_meta = pd.read_csv(gene_metadata, index_col=0)

    if gene_col in gene_meta.columns:
        genes = set(gene_meta[gene_col].astype(str).values)
    else:
        genes = set(gene_meta.index.astype(str))

    logger.info(f"[gene_metadata] loaded genes={len(genes):,}")

    return genes, gene_meta


def filter_common_features(
    expr_genes: set[str],
    re_tg: pd.DataFrame,
    peak_motif: pd.DataFrame,
):
    re_tg = re_tg.copy()
    peak_motif = peak_motif.copy()

    re_tg["gene_name"] = re_tg["gene_name"].astype(str)
    re_tg["peak_name"] = re_tg["peak_name"].astype(str)

    peak_motif["peak"] = peak_motif["peak"].astype(str)
    peak_motif["motif"] = peak_motif["motif"].astype(str)

    genes_valid = expr_genes & set(re_tg["gene_name"].unique())
    re_tg = re_tg[re_tg["gene_name"].isin(genes_valid)].copy()

    peaks_common = set(re_tg["peak_name"].unique()) & set(peak_motif["peak"].unique())

    re_tg = re_tg[re_tg["peak_name"].isin(peaks_common)].copy()
    peak_motif = peak_motif[peak_motif["peak"].isin(peaks_common)].copy()

    genes_valid = expr_genes & set(re_tg["gene_name"].unique())
    re_tg = re_tg[re_tg["gene_name"].isin(genes_valid)].copy()

    genes_sorted = sort_genes_by_coord(genes_valid, re_tg)
    peaks_sorted = sort_peaks_by_coord(peaks_common)

    logger.info(
        f"[filter_common_features] genes={len(genes_sorted):,}, "
        f"peaks={len(peaks_sorted):,}, re_tg_rows={len(re_tg):,}"
    )

    return genes_sorted, peaks_sorted, re_tg.reset_index(drop=True), peak_motif.reset_index(drop=True)


def build_re_tg_matrix(re_tg: pd.DataFrame, genes: list[str], peaks: list[str], tau: float):
    peak_mid = (
        re_tg["peak_start"].values.astype(float)
        + re_tg["peak_end"].values.astype(float)
    ) / 2.0

    dist = np.abs(peak_mid - re_tg["TSS"].values.astype(float))

    df = re_tg[["gene_name", "peak_name"]].copy()
    df["weight"] = np.exp(-dist / tau)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    df["i"] = df["gene_name"].map(gene_to_idx)
    df["j"] = df["peak_name"].map(peak_to_idx)

    df = df.dropna(subset=["i", "j"]).copy()
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    df = df.groupby(["i", "j"], as_index=False)["weight"].max()

    W = coo_matrix(
        (df["weight"].values.astype(np.float64), (df["i"].values, df["j"].values)),
        shape=(len(genes), len(peaks)),
    ).tocsr()

    logger.info(f"[build_re_tg_matrix] shape={W.shape}, nnz={W.nnz:,}")

    return W


def build_peak_motif_matrix(peak_motif: pd.DataFrame, peaks: list[str]):
    peak_motif = peak_motif.copy()

    motif_list = sorted(peak_motif["motif"].astype(str).unique())

    peak_to_idx = {p: i for i, p in enumerate(peaks)}
    motif_to_idx = {m: i for i, m in enumerate(motif_list)}

    sub = peak_motif[
        peak_motif["peak"].isin(peak_to_idx.keys())
        & peak_motif["motif"].isin(motif_to_idx.keys())
    ].copy()

    pairs = sub[["peak", "motif"]].drop_duplicates()

    i_idx = pairs["peak"].map(peak_to_idx).values.astype(int)
    j_idx = pairs["motif"].map(motif_to_idx).values.astype(int)

    B = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (i_idx, j_idx)),
        shape=(len(peaks), len(motif_list)),
    ).tocsr()

    logger.info(f"[build_peak_motif_matrix] shape={B.shape}, nnz={B.nnz:,}")

    return B, motif_list


def write_re_tg_h5(W: csr_matrix, genes: list[str], peaks: list[str], output_file: Path):
    coo = W.tocoo()

    with h5py.File(output_file, "w") as h5:
        h5.attrs["indexing"] = "0-based"
        h5.attrs["created_by"] = "rega_perturb.preprocess.graph"
        h5.create_dataset("i", data=coo.row.astype(np.int32))
        h5.create_dataset("j", data=coo.col.astype(np.int32))
        h5.create_dataset("x", data=coo.data.astype(np.float64))
        h5.create_dataset("dims", data=np.array(W.shape, dtype=np.int32))
        h5.create_dataset("rownames", data=np.array(genes, dtype="S"))
        h5.create_dataset("colnames", data=np.array(peaks, dtype="S"))

    logger.info(f"[write] {output_file}")


def build_graph_anndata(
    W: csr_matrix,
    B: csr_matrix,
    genes: list[str],
    peaks: list[str],
    motif_list: list[str],
    re_tg_f: pd.DataFrame,
    output_file: Path,
):
    gene_cols = ["gene_name", "chrom", "region_start", "region_end", "TSS", "strand"]
    peak_cols = ["peak_name", "peak_chrom", "peak_start", "peak_end"]

    missing_gene = [c for c in gene_cols if c not in re_tg_f.columns]
    missing_peak = [c for c in peak_cols if c not in re_tg_f.columns]

    if missing_gene:
        raise ValueError(f"RE-TG file missing gene metadata columns: {missing_gene}")
    if missing_peak:
        raise ValueError(f"RE-TG file missing peak metadata columns: {missing_peak}")

    gene_meta = (
        re_tg_f[gene_cols]
        .drop_duplicates("gene_name", keep="first")
        .set_index("gene_name")
        .loc[genes]
        .copy()
    )

    peak_meta = (
        re_tg_f[peak_cols]
        .drop_duplicates("peak_name", keep="first")
        .set_index("peak_name")
        .loc[peaks]
        .copy()
    )

    adata = ad.AnnData(
        X=W.tocsr(),
        obs=gene_meta,
        var=peak_meta,
    )

    adata.varm["Peak_Motif"] = B.tocsr()
    adata.uns["motif_names"] = np.array(motif_list, dtype=object)
    adata.uns["contains_GEX"] = False
    adata.uns["created_by"] = "rega_perturb.preprocess.graph"

    adata.write_h5ad(output_file, compression="gzip")

    logger.info(f"[write] {output_file}")


def build_graph(
    gene_metadata: str | Path,
    re_tg: str | Path,
    peak_motif: str | Path,
    output_dir: str | Path,
    tau: float = 1e5,
    gene_col: str = "gene_name",
) -> Path:
    """Build graph inputs; returns the path of ``REGA_graph_without_GEX.h5ad``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expr_genes, gene_meta = load_expression_gene_set(
        gene_metadata=Path(gene_metadata),
        gene_col=gene_col,
    )

    re_tg_df = pd.read_csv(re_tg, sep="\t", header=0)
    re_tg_df.columns = [c.strip() for c in re_tg_df.columns]

    required_re_tg = {"gene_name", "peak_name", "chrom", "TSS", "peak_start", "peak_end"}
    missing = required_re_tg - set(re_tg_df.columns)
    if missing:
        raise ValueError(f"RE-TG file missing required columns: {sorted(missing)}")

    for col in ["TSS", "peak_start", "peak_end"]:
        re_tg_df[col] = pd.to_numeric(re_tg_df[col], errors="coerce")

    re_tg_df = re_tg_df.dropna(subset=["TSS", "peak_start", "peak_end"]).copy()

    peak_motif_df = pd.read_csv(
        peak_motif,
        sep="\t",
        header=None,
        names=["peak", "motif"],
    )

    genes, peaks, re_tg_f, pm_f = filter_common_features(
        expr_genes=expr_genes,
        re_tg=re_tg_df,
        peak_motif=peak_motif_df,
    )

    if len(genes) == 0:
        raise ValueError("No common genes between gene metadata and RE-TG.")
    if len(peaks) == 0:
        raise ValueError("No common peaks between RE-TG and peak-motif.")

    W = build_re_tg_matrix(
        re_tg=re_tg_f,
        genes=genes,
        peaks=peaks,
        tau=tau,
    )

    B, motif_list = build_peak_motif_matrix(
        peak_motif=pm_f,
        peaks=peaks,
    )

    re_tg_h5 = output_dir / "re_tg_matrix.h5"
    binding_npz = output_dir / "input_binding.npz"
    binding_peaks = output_dir / "input_binding_peaks.txt"
    binding_motifs = output_dir / "input_binding_motifs.txt"
    graph_h5ad = output_dir / GRAPH_H5AD
    re_tg_genes = output_dir / "re_tg_genes.txt"

    write_re_tg_h5(W, genes, peaks, re_tg_h5)

    scipy.sparse.save_npz(str(binding_npz), B)
    binding_peaks.write_text("\n".join(peaks) + "\n")
    binding_motifs.write_text("\n".join(motif_list) + "\n")
    re_tg_genes.write_text("\n".join(genes) + "\n")

    logger.info(f"[write] {binding_npz}")
    logger.info(f"[write] {binding_peaks}")
    logger.info(f"[write] {binding_motifs}")
    logger.info(f"[write] {re_tg_genes}")

    build_graph_anndata(
        W=W,
        B=B,
        genes=genes,
        peaks=peaks,
        motif_list=motif_list,
        re_tg_f=re_tg_f,
        output_file=graph_h5ad,
    )

    write_run_record(
        output_dir,
        "build_graph",
        params=dict(tau=tau, gene_col=gene_col),
        inputs={"gene_metadata": gene_metadata, "re_tg": re_tg, "peak_motif": peak_motif},
    )
    logger.info("Done.")
    return graph_h5ad
