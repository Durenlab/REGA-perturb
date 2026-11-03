"""Tiny synthetic Perturb-seq dataset for tests.

Layout follows the public Replogle et al. h5ad format: ``obs`` has the perturbation
target in ``gene`` and the mitochondrial fraction in ``mitopercent``; ``var`` has
``gene_name``, ``chr``, ``start``, ``end`` and ``strand``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

RE_TG_COLUMNS = [
    "chrom", "region_start", "region_end", "gene_name", "TSS", "strand",
    "peak_chrom", "peak_start", "peak_end", "peak_name",
]


def make_tiny_dataset(out_dir: Path, seed: int = 0, window: int = 300000) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)

    # ---- genes: 2 chromosomes + one mitochondrial gene
    genes = []
    for chrom, n in (("chr1", 18), ("chr2", 14)):
        for i in range(n):
            start = 1_000_000 + i * 150_000
            genes.append(dict(gene_name=f"{chrom.upper()}G{i}", chr=chrom, start=start,
                              end=start + 20_000, strand="+" if i % 2 == 0 else "-"))
    genes.append(dict(gene_name="MT-CO1", chr="chrM", start=5000, end=6000, strand="+"))
    # perturbation targets that are also modeled genes (TF-like names used by the motif-TF table)
    genes[0]["gene_name"] = "TFA"
    genes[3]["gene_name"] = "TFB"
    genes[20]["gene_name"] = "TFC"
    var = pd.DataFrame(genes)
    var.index = [f"ENSG{i:05d}" for i in range(len(var))]

    # ---- cells: non-targeting + 4 targets
    targets = ["non-targeting", "TFA", "TFB", "TFC", "CHR2G5"]
    n_per = [60, 40, 40, 40, 40]
    labels = np.concatenate([[t] * n for t, n in zip(targets, n_per)])
    rng.shuffle(labels)
    n_cells, n_genes = len(labels), len(var)
    base = rng.gamma(2.0, 2.0, size=n_genes)
    lam = np.tile(base, (n_cells, 1))
    for t_i, t in enumerate(targets[1:]):
        mask = labels == t
        affected = rng.choice(n_genes - 1, size=6, replace=False)
        lam[np.ix_(mask, affected)] *= rng.uniform(0.2, 3.0, size=6)
    counts = rng.poisson(lam * rng.uniform(0.5, 1.5, size=(n_cells, 1))).astype(np.float32)
    X = sp.csr_matrix(counts)
    obs = pd.DataFrame(
        {"gene": pd.Categorical(labels),
         "mitopercent": counts[:, -1] / np.maximum(counts.sum(1), 1)},
        index=[f"cell{i:04d}" for i in range(n_cells)],
    )
    raw_h5ad = out_dir / "raw.h5ad"
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(raw_h5ad)

    # ---- reference peaks near gene TSSs
    peaks = []
    for _, g in var[var["chr"] != "chrM"].iterrows():
        tss = g["start"] if g["strand"] == "+" else g["end"]
        for off in rng.choice(np.arange(-250_000, 250_000, 5000), size=3, replace=False):
            s = int(tss + off)
            peaks.append((g["chr"], s, s + 500))
    peaks = sorted(set(peaks))
    peaks_df = pd.DataFrame(peaks, columns=["chrom", "start", "end"])
    peaks_df["name"] = (peaks_df["chrom"] + "_" + peaks_df["start"].astype(str) + "_"
                        + peaks_df["end"].astype(str))
    peaks_bed = out_dir / "peaks.bed"
    peaks_df[["chrom", "start", "end"]].to_csv(peaks_bed, sep="\t", header=False, index=False)
    named_bed = out_dir / "peaks_named.bed"
    peaks_df.to_csv(named_bed, sep="\t", header=False, index=False)

    # ---- RE-TG table (TSS +/- window overlap), same columns as `rega-perturb retg`
    rows = []
    for _, g in var[var["chr"] != "chrM"].iterrows():
        tss = g["start"] if g["strand"] == "+" else g["end"]
        lo, hi = max(tss - window, 0), tss + window
        sub = peaks_df[(peaks_df["chrom"] == g["chr"]) & (peaks_df["end"] > lo) & (peaks_df["start"] < hi)]
        for _, p in sub.iterrows():
            rows.append([g["chr"], lo, hi, g["gene_name"], tss, g["strand"],
                         p["chrom"], p["start"], p["end"], p["name"]])
    re_tg = out_dir / "re_tg.tsv"
    pd.DataFrame(rows, columns=RE_TG_COLUMNS).to_csv(re_tg, sep="\t", index=False)

    # ---- peak-motif table and motif-TF table
    motifs = [f"MOTIF{k}" for k in range(8)]
    pm = [(p, m) for p in peaks_df["name"] for m in motifs if rng.random() < 0.35]
    peak_motif = out_dir / "peak_motif.txt"
    pd.DataFrame(sorted(set(pm))).to_csv(peak_motif, sep="\t", header=False, index=False)
    motif_tf = out_dir / "motif_tf.txt"
    pd.DataFrame([("MOTIF0", "TFA"), ("MOTIF1", "TFA"), ("MOTIF2", "TFB"), ("MOTIF3", "TFC"),
                  ("MOTIF4", "NOT_MODELED"), ("MOTIF_X", "TFB")]).to_csv(
        motif_tf, sep="\t", header=False, index=False)

    return dict(dir=out_dir, raw_h5ad=raw_h5ad, peaks_bed=peaks_bed, named_bed=named_bed,
                re_tg=re_tg, peak_motif=peak_motif, motif_tf=motif_tf)
