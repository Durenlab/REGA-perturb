"""Reference annotation: peak naming, motif scanning (HOMER) and RE-TG pairs (bedtools).

External tools (not installed with the package):

* HOMER ``findMotifsGenome.pl`` with the genome package (e.g. ``hg38``) and a motif file;
* ``bedtools``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from rega_perturb.utils import get_logger

logger = get_logger("preprocess.reference")

STANDARD_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
RE_TG_COLUMNS = [
    "chrom", "region_start", "region_end", "gene_name", "TSS", "strand",
    "peak_chrom", "peak_start", "peak_end", "peak_name",
]


def name_peaks(peaks_bed: str | Path, output_bed: str | Path, standard_only: bool = True) -> Path:
    """Write a 4-column BED whose name column is ``<chrom>_<start>_<end>``.

    Only the first three columns of ``peaks_bed`` are used. Peaks on non-standard
    chromosomes are dropped when ``standard_only`` is True. Duplicate peaks are removed.
    """
    df = pd.read_csv(peaks_bed, sep="\t", header=None, usecols=[0, 1, 2], comment="#",
                     names=["chrom", "start", "end"], dtype={"chrom": str})
    if standard_only:
        df = df[df["chrom"].isin(STANDARD_CHROMOSOMES)]
    df = df.drop_duplicates()
    df["name"] = df["chrom"] + "_" + df["start"].astype(str) + "_" + df["end"].astype(str)
    output_bed = Path(output_bed)
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_bed, sep="\t", header=False, index=False)
    logger.info(f"[name_peaks] {len(df):,} peaks -> {output_bed}")
    return output_bed


def build_re_tg(
    tss_file: str | Path,
    named_peaks_bed: str | Path,
    output_tsv: str | Path,
    bedtools: str = "bedtools",
    standard_only: bool = True,
) -> Path:
    """Pair every gene window with the peaks it overlaps (``bedtools intersect -wa -wb``).

    ``tss_file`` is the TSS +/- window file written by :func:`rega_perturb.preprocess.qc.run_qc`
    (columns ``chrom region_start region_end gene_name TSS strand``, no header).
    The output is tab-separated with header :data:`RE_TG_COLUMNS`.
    """
    if shutil.which(bedtools) is None:
        raise FileNotFoundError(f"bedtools executable not found: {bedtools}")
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    tss = pd.read_csv(tss_file, sep="\t", header=None, dtype={0: str})
    if standard_only:
        tss = tss[tss[0].isin(STANDARD_CHROMOSOMES)]
    tss_clean = output_tsv.with_suffix(".tss_clean.tmp")
    tss.to_csv(tss_clean, sep="\t", header=False, index=False)

    cmd = [bedtools, "intersect", "-a", str(tss_clean), "-b", str(named_peaks_bed), "-wa", "-wb"]
    logger.info(f"[build_re_tg] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    tss_clean.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"bedtools failed ({result.returncode}): {result.stderr[-2000:]}")

    lines = result.stdout.splitlines()
    # Drop adjacent duplicate lines (same as piping through `uniq`).
    out_lines = [ln for i, ln in enumerate(lines) if i == 0 or ln != lines[i - 1]]
    with open(output_tsv, "w") as f:
        f.write("\t".join(RE_TG_COLUMNS) + "\n")
        for ln in out_lines:
            f.write(ln + "\n")
    logger.info(f"[build_re_tg] {len(out_lines):,} RE-TG pairs -> {output_tsv}")
    return output_tsv


def run_motif_scan(
    named_peaks_bed: str | Path,
    genome: str,
    motif_file: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    homer_bin: str = "findMotifsGenome.pl",
    threads: int = 1,
    keep_intermediate: bool = False,
) -> Path:
    """Scan peaks for motifs with HOMER and write ``<output_prefix>_peak_motif.txt``.

    The output has two tab-separated columns (peak name, motif name), no header,
    one row per distinct pair, sorted by peak name.
    """
    if shutil.which(homer_bin) is None:
        raise FileNotFoundError(f"HOMER executable not found: {homer_bin}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / f"{output_prefix}_motif_output"
    work_dir.mkdir(exist_ok=True)
    raw_output = output_dir / f"{output_prefix}_homer_find.txt"

    cmd = [homer_bin, str(named_peaks_bed), genome, str(work_dir),
           "-size", "given", "-p", str(threads), "-find", str(motif_file)]
    logger.info(f"[run_motif_scan] {' '.join(cmd)}")
    with open(raw_output, "w") as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HOMER failed ({result.returncode}): {result.stderr[-2000:]}")
    if raw_output.stat().st_size == 0:
        raise RuntimeError(f"HOMER produced empty output: {raw_output}")

    pairs = set()
    with open(raw_output) as fh:
        next(fh)  # header: PositionID Offset Sequence MotifName Strand MotifScore
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                pairs.add((fields[0], fields[3]))
    peak_motif = output_dir / f"{output_prefix}_peak_motif.txt"
    with open(peak_motif, "w") as out:
        for peak, motif in sorted(pairs):
            out.write(f"{peak}\t{motif}\n")
    logger.info(f"[run_motif_scan] {len(pairs):,} (peak, motif) pairs -> {peak_motif}")

    if not keep_intermediate:
        raw_output.unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    return peak_motif
