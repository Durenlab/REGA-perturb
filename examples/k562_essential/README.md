# Example: K562 essential-gene Perturb-seq screen

This folder runs the full workflow on a K562 CRISPRi Perturb-seq screen
(essential-gene screen, Replogle et al., 2022).

## Mini example (minutes on a CPU)

A small subset of the screen, included in [`mini/`](mini): the non-targeting control and
35 perturbation targets (about 3,500 cells), genes on chromosomes 14, 19, 21, 22 and X,
and the reference peaks on these chromosomes.

| File | Content |
|---|---|
| `mini/k562_essential_mini_raw.h5ad` | raw UMI counts (cells x genes); `obs['gene']` = perturbation target |
| `mini/k562_mini_peaks.bed` | reference regulatory elements (K562 ATAC-seq peaks) on these chromosomes |
| `mini/k562_mini_peak_motif.txt.gz` | motif occurrences in the peaks (peak, motif) |

The motif-to-TF table is `reference/motif_tf_human.txt` at the top of the repository.

```bash
bash run_mini.sh            # data from mini/, results in ./output
bash run_mini.sh <data_dir> <out_dir>
```

Requirements: `rega-perturb` with PyTorch, and `bedtools` on `PATH`.

## Full data

* Raw counts: `K562_essential_raw_singlecell_01.h5ad` (about 10.7 GB) from the
  Replogle et al. (2022) data release on figshare
  ([doi:10.25452/figshare.plus.20029387](https://doi.org/10.25452/figshare.plus.20029387);
  direct download: <https://ndownloader.figshare.com/files/35773219>).
* Reference peaks: [`examples/k562_essential/reference/k562_peaks.bed.gz`](reference) (K562 ATAC-seq peaks;
  chrom, start, end).
* Motif occurrences: name the peaks and scan them with HOMER using the motif collection
  `reference/motifs_homer.txt` (commands run from the top of the repository):

```bash
rega-perturb peaks -i examples/k562_essential/reference/k562_peaks.bed.gz -o out/reference/peaks_named.bed
rega-perturb motif -i out/reference/peaks_named.bed --genome hg38 \
    --motif-file reference/motifs_homer.txt -o out/reference --prefix k562
```

The remaining commands are the same as in `run_mini.sh`. For a screen of this size a GPU is
strongly recommended for `rega-perturb train` (`--device cuda`).

## Outputs

| Folder / file | Content |
|---|---|
| `qc/` | retained cells and genes, QC summary, TSS windows |
| `reference/re_tg.tsv` | gene - regulatory element pairs |
| `graph/`, `blocks/` | model inputs |
| `model/final_perturb.pt`, `model/rega_perturb_result.h5ad` | trained model and result AnnData |
| `tf_tg/tf_tg_influence.tsv.gz`, `tf_tg/tf_tg_connectivity.tsv.gz` | TF-target gene scores (`TF, TG, score`) |
| `tf_activity/tf_activity_log2fc.tsv` | TF activity change per perturbation target (log2 fold change vs control) |
| `tf_activity/tf_activity_changes.tsv.gz` | the same as a long table (`target, TF, activity_target, activity_control, log2fc`) |
| `diff_network/rewiring_scores.h5` | per-gene rewiring scores (RRS, NRS) for every target |
| `diff_network/delta_networks.h5` | differential TF-TG network of every target (`/networks/<target>`, gene x TF) |
| `*/<step>.run.yaml` | parameters, seeds and software versions of each step |

`diff-network` options: `--target <TARGET>` (repeatable) or `--all-targets` selects the targets;
`--format tsv` writes `delta_network.<target>.tsv.gz` long tables (`target, TF, TG, delta_w`;
filter with `--min-abs` / `--top-k`) and `--format h5` one dense gene x TF matrix per target;
`--kind specific` writes perturbation-specific instead of differential networks.
