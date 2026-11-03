# REGA-perturb Tutorial

This tutorial walks through a complete REGA-perturb analysis using the bundled mini
example (`examples/k562_essential/mini/`, a subset of a K562 CRISPRi Perturb-seq
screen; Replogle et al., 2022). The mini example runs on a CPU in a few minutes.

---

## 1. Overview

REGA-perturb is a Reference REgulatory Path-Guided Analysis framework for
resolving how genetic perturbations act through gene regulatory paths. Given a
Perturb-seq count matrix and a reference set of regulatory elements (REs) of the
same cell type, the workflow:

1. performs cell and gene quality control;
2. annotates the REs with TF binding motifs (HOMER) and pairs them with genes in the
   TSS window of each gene;
3. trains the model on the perturbation data;
4. runs the downstream analyses: GRN inference (TF-target gene scores),
   perturbation-induced TF activity changes and differential regulatory networks.

### Pipeline structure

| Step | CLI command | Module | Input → Output |
|------|-------------|--------|----------------|
| 1 | `rega-perturb qc` | `rega_perturb.preprocess.qc` | raw h5ad → cell / gene metadata, TSS windows |
| 2 | `rega-perturb peaks` | `rega_perturb.preprocess.reference` | BED → named BED |
| 3 | `rega-perturb motif` | `rega_perturb.preprocess.reference` | named BED → peak-motif table |
| 4 | `rega-perturb retg` | `rega_perturb.preprocess.reference` | TSS windows + named BED → RE-TG table |
| 5 | `rega-perturb build-graph` | `rega_perturb.preprocess.graph` | gene metadata + RE-TG + peak-motif → graph h5ad |
| 6 | `rega-perturb build-blocks` | `rega_perturb.preprocess.blocks` | graph h5ad + raw h5ad → per-chromosome model inputs |
| 7 | `rega-perturb train` | `rega_perturb.model` | model inputs → `final_perturb.pt` |
| 8 | `rega-perturb assemble` | `rega_perturb.results` | graph h5ad + checkpoint → result h5ad |
| 9 | `rega-perturb tf-tg` | `rega_perturb.analysis.tf_tg_scores` | checkpoint + graph h5ad → TF-target gene scores |
| 10 | `rega-perturb tf-activity` | `rega_perturb.analysis.tf_activity` | checkpoint + graph h5ad → TF activity changes |
| 11 | `rega-perturb diff-network` | `rega_perturb.analysis.diff_network` | checkpoint + graph h5ad → differential networks, RRS / NRS |

Every command writes `<step>.run.yaml` in its output folder with the parameters,
random seeds and software versions of the run. Global options (before the command):
`--log-level`, `--log-file`, `--debug`.

---

## 2. Data preparation

### 2.1 Perturb-seq data

**Format**: AnnData `.h5ad`, cells x genes.

* `X`: raw UMI counts (default) or already-normalized values (see 3.1).
* `obs`: the perturbation target of each cell (default column `gene`); control
  cells are labelled `non-targeting`. A column `mitopercent` (mitochondrial
  fraction) is used when present, otherwise it is computed from `MT-` genes.
* `var`: `gene_name`, `chr`, `start`, `end`, `strand`. Without gene coordinates,
  use `qc --tss-reference reference/tss_hg38.txt` (see 3.1).

The mini example file `examples/k562_essential/mini/k562_essential_mini_raw.h5ad`
contains about 3,500 cells (non-targeting control and 35 perturbation targets) and
the genes on chromosomes 14, 19, 21, 22 and X.

### 2.2 Reference regulatory elements

**Format**: BED (chrom, start, end); additional columns are ignored; `.bed.gz` is
accepted by `rega-perturb peaks`.

The mini example uses `examples/k562_essential/mini/k562_mini_peaks.bed` (K562
ATAC-seq peaks on the same chromosomes); the full peak list is
`examples/k562_essential/reference/k562_peaks.bed.gz`.

### 2.3 Reference data (provided)

| File | Used by |
|------|---------|
| `reference/motifs_homer.txt` — motif collection in HOMER format | `rega-perturb motif --motif-file` |
| `reference/motif_tf_human.txt` — motif-to-TF table | `--motif-tf` of `assemble`, `tf-tg`, `tf-activity`, `diff-network` |
| `reference/tss_hg38.txt` — hg38 TSS table | `rega-perturb qc --tss-reference` (data without gene coordinates) |

---

## 3. Step-by-step usage

```bash
MINI=examples/k562_essential/mini
WORKDIR=/path/to/my_run
mkdir -p $WORKDIR
```

The Python examples use the same paths (run from the top of the repository):

```python
MINI = "examples/k562_essential/mini"
WORKDIR = "/path/to/my_run"
```

### 3.1 `rega-perturb qc` — Quality control

Filters cells and genes and writes the TSS window of every retained gene. The raw
matrix is read in chunks, so large screens do not need to fit in memory.

**CLI**:

```bash
rega-perturb qc -i $MINI/k562_essential_mini_raw.h5ad -o $WORKDIR/qc --prefix k562_mini \
    --min-genes 100 --max-genes 5000
```

**Python API**:

```python
from rega_perturb.preprocess.qc import run_qc
run_qc(f"{MINI}/k562_essential_mini_raw.h5ad", f"{WORKDIR}/qc", "k562_mini",
       min_genes=100, max_genes=5000)
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--min-umi` / `--max-umi` | 1000 / 50000 | UMI count per cell |
| `--min-genes` | 500 | minimum detected genes per cell |
| `--max-genes` | none | optional maximum detected genes per cell (no upper limit unless set) |
| `--max-mt` | 0.2 | maximum mitochondrial fraction (0-1) |
| `--min-cells-per-gene` | 30 | genes detected in fewer cells are removed |
| `--keep-mt-genes` | off | keep mitochondrial genes |
| `--keep-nonstandard-chromosomes` | off | keep genes outside chr1-22, X, Y |
| `--tss-window` | 300000 | TSS +/- window (bp) for pairing genes with REs |
| `--input-type` | `counts` | `normalized`: already-normalized input (no cell filtering) |
| `--tss-reference` | — | TSS table used when `var` lacks gene coordinates |

**Output** (`<prefix>_REGA_qc_*`): `cell_metadata.csv` (retained cells, library
size), `gene_metadata.csv` (retained genes), `summary.csv` (cells / genes after
each step), `gene_TSS_300kb.txt` (TSS windows).

---

### 3.2 `rega-perturb peaks` — Peak naming

Names each RE `<chrom>_<start>_<end>` so that REs can be referenced by name in the
following steps; REs outside chr1-22, X, Y and duplicates are removed.

```bash
rega-perturb peaks -i $MINI/k562_mini_peaks.bed -o $WORKDIR/reference/peaks_named.bed
```

```python
from rega_perturb.preprocess.reference import name_peaks
name_peaks(f"{MINI}/k562_mini_peaks.bed", f"{WORKDIR}/reference/peaks_named.bed")
```

**Output**: 4-column BED — `chrom  start  end  chrom_start_end`

---

### 3.3 `rega-perturb motif` — HOMER motif scan

```bash
rega-perturb motif -i $WORKDIR/reference/peaks_named.bed --genome hg38 \
    --motif-file reference/motifs_homer.txt -o $WORKDIR/reference --prefix k562_mini --threads 8
```

```python
from rega_perturb.preprocess.reference import run_motif_scan
run_motif_scan(f"{WORKDIR}/reference/peaks_named.bed", "hg38", "reference/motifs_homer.txt",
               f"{WORKDIR}/reference", "k562_mini", threads=8)
```

**Output**: `<prefix>_peak_motif.txt` — two columns (peak name, motif), no header.

For the mini example the scan result is provided
(`$MINI/k562_mini_peak_motif.txt.gz`), so this step can be skipped.

---

### 3.4 `rega-perturb retg` — RE-TG pairs

Pairs every gene with the REs in its TSS window (`bedtools intersect`).

```bash
rega-perturb retg --tss-file $WORKDIR/qc/k562_mini_REGA_qc_gene_TSS_300kb.txt \
    --peaks-bed $WORKDIR/reference/peaks_named.bed -o $WORKDIR/reference/re_tg.tsv
```

```python
from rega_perturb.preprocess.reference import build_re_tg
build_re_tg(f"{WORKDIR}/qc/k562_mini_REGA_qc_gene_TSS_300kb.txt",
            f"{WORKDIR}/reference/peaks_named.bed", f"{WORKDIR}/reference/re_tg.tsv")
```

**Output**: tab-separated table with columns `chrom region_start region_end
gene_name TSS strand peak_chrom peak_start peak_end peak_name`.

---

### 3.5 `rega-perturb build-graph` — Graph inputs

```bash
rega-perturb build-graph --gene-metadata $WORKDIR/qc/k562_mini_REGA_qc_gene_metadata.csv \
    --re-tg $WORKDIR/reference/re_tg.tsv --peak-motif $MINI/k562_mini_peak_motif.txt.gz \
    -o $WORKDIR/graph
```

```python
from rega_perturb.preprocess.graph import build_graph
build_graph(f"{WORKDIR}/qc/k562_mini_REGA_qc_gene_metadata.csv", f"{WORKDIR}/reference/re_tg.tsv",
            f"{MINI}/k562_mini_peak_motif.txt.gz", f"{WORKDIR}/graph")
```

**Output**: `REGA_graph_without_GEX.h5ad` (genes x REs AnnData used by the later
steps), `re_tg_matrix.h5`, `input_binding.npz`, `input_binding_peaks.txt`,
`input_binding_motifs.txt`, `re_tg_genes.txt`.

---

### 3.6 `rega-perturb build-blocks` — Model inputs

Streams the expression matrix into one input file per chromosome and writes the
perturbation assignment of every cell.

```bash
rega-perturb build-blocks --graph-h5ad $WORKDIR/graph/REGA_graph_without_GEX.h5ad \
    --raw-h5ad $MINI/k562_essential_mini_raw.h5ad \
    --cell-metadata $WORKDIR/qc/k562_mini_REGA_qc_cell_metadata.csv \
    --gene-metadata $WORKDIR/qc/k562_mini_REGA_qc_gene_metadata.csv -o $WORKDIR/blocks
```

```python
from rega_perturb.preprocess.blocks import build_blocks
build_blocks(f"{WORKDIR}/graph/REGA_graph_without_GEX.h5ad", f"{MINI}/k562_essential_mini_raw.h5ad",
             f"{WORKDIR}/qc/k562_mini_REGA_qc_cell_metadata.csv",
             f"{WORKDIR}/qc/k562_mini_REGA_qc_gene_metadata.csv", f"{WORKDIR}/blocks")
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--expr-mode` | `logcp10k` | `logcp10k`: log1p(count / library size x 1e4); `logcpm`: x 1e6; `as_is`: input values unchanged (already-normalized data); `counts`: raw counts |
| `--perturb-col` | `gene` | cell-metadata column with the perturbation target |
| `--chroms` | all | restrict to these chromosomes (repeatable) |

**Output**: `<chrom>.npz` per chromosome, `P_perturb_target.npz`, `P_cells.txt`,
`P_targets.txt`, `blocks_manifest.csv`.

---

### 3.7 `rega-perturb train` — Model training

> **A GPU is strongly recommended for training.** The mini example also trains on a
> CPU in a few minutes. Check GPU availability with
> `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`; on HPC,
> allocate a GPU node first (see [installation](installation.md#6-hpc-tips)).

Training parameters are read from a YAML file (`train:` section); command-line
options override the file. `examples/k562_essential/train_mini.yaml` holds the
settings of the mini example; adapt them for larger data sets.

**Training length** is fixed and not configurable: 15 epochs in a first phase, then at
least 60 epochs in a second phase, which continues until the full-data reconstruction
error has converged (evaluated every 5 epochs; relative change below 1e-4 in two
consecutive evaluations). There is no upper limit on the number of epochs. The epochs
actually trained are written to the log and stored in the checkpoint
(`epochs_trained`); `perturb_rre_log.csv` lists every evaluation.

```bash
rega-perturb train --block-dir $WORKDIR/blocks -o $WORKDIR/model \
    --config examples/k562_essential/train_mini.yaml --device cpu
```

```python
from rega_perturb.model import load_perturb_blocks, train_rega_perturb
from rega_perturb.model.config import make_training_config

config = make_training_config("examples/k562_essential/train_mini.yaml")
data, p_init = load_perturb_blocks(f"{WORKDIR}/blocks")
train_rega_perturb(data, p_init, f"{WORKDIR}/model", config, device="cpu")
```

**Key options**:

| Option | Default | Notes |
|--------|---------|-------|
| `--config` | — | YAML file with training parameters |
| `--device` | auto | `cuda` or `cpu` (default: `cuda` if available) |
| `--seed`, `--V-seed`, `--H-seed` | from config | random seeds (recorded in `train.run.yaml`) |
| `--chroms` | all | train on these chromosomes only (repeatable) |
| `--progress / --no-progress` | on | progress bars |

**Output**: `final_perturb.pt` (trained model), `perturb_loss_log.csv` and
`perturb_rre_log.csv` (training logs), `perturb_training_curves.png` (if matplotlib
is installed), `train.run.yaml`.

---

### 3.8 `rega-perturb assemble` — Result h5ad

```bash
rega-perturb assemble --graph-h5ad $WORKDIR/graph/REGA_graph_without_GEX.h5ad \
    --checkpoint $WORKDIR/model/final_perturb.pt -o $WORKDIR/model/rega_perturb_result.h5ad \
    --motif-tf reference/motif_tf_human.txt
```

```python
from rega_perturb.results import collect_results
collect_results(f"{WORKDIR}/graph/REGA_graph_without_GEX.h5ad", f"{WORKDIR}/model/final_perturb.pt",
                f"{WORKDIR}/model/rega_perturb_result.h5ad",
                motif_tf="reference/motif_tf_human.txt")
```

**Output**: `rega_perturb_result.h5ad` — AnnData with the trained model results:
genes as `obs`, regulatory elements (REs) as `var`; "target" = a perturbation target
(including the control label). No expression data is stored.

| Slot | Key | Shape | Content |
|------|-----|-------|---------|
| `layers` | `Net_RE_TG` | gene × RE (sparse) | RE → target gene regulatory scores |
| `obsm` | `Net_motif_TG` | gene × motif | motif → target gene scores (motif-level counterpart of `TF_TG_influence`) |
| `obsm` | `TF_TG_influence`, `TF_TG_connectivity` | gene × TF | the two TF → target gene scores (same values as `rega-perturb tf-tg`); only with `--motif-tf` |
| `var` | `RE_importance` | RE | RE importance score |
| `uns` | `Motif_Activity` | motif × target | motif activity per perturbation target |
| `varm` | `Peak_Activity` | RE × target | RE activity per perturbation target; only with `--peak-activity` (several GB for genome-scale screens) |
| `obsm` | `Gene_Embedding`, `Gene_Embedding_norm` | gene × K | gene embedding (`K` = embedding dimension, training parameter `--K`; `_norm`: normalized version, likewise for the other embeddings) |
| `varm` | `RE_Embedding`, `RE_Embedding_norm` | RE × K | RE embedding |
| `uns` | `Motif_Embedding`, `Motif_Embedding_norm` | motif × K | motif embedding |
| `uns` | `Target_Embedding`, `Target_Embedding_norm` | target × K | perturbation target embedding |
| `uns` | `Cell_Embedding` | cell × K | cell embedding; only with `--cell-embeddings` (up to `--max-cells-dense` cells) |
| `uns` | `perturb_targets`, `motif_names`, `TF_names`, `Module_names` | | row / column names of the matrices above (`TF_names` only with `--motif-tf`) |
| `uns` | `P_cells`, `P_cell_target_name` | cell | cell barcodes and the perturbation target of each cell |
| `uns` | `REGA_perturb_config` | | training configuration |

Other keys store learned model components and per-cell assignments used internally.

```python
import anndata as ad
res = ad.read_h5ad(f"{WORKDIR}/model/rega_perturb_result.h5ad")
targets = list(res.uns["perturb_targets"])
motif_act = res.uns["Motif_Activity"]           # motif x target
tf_tg = res.obsm["TF_TG_influence"]             # gene x TF (with --motif-tf)
```

---

## 4. Downstream analysis

The analysis commands read the checkpoint, the graph h5ad and the motif-to-TF table:

```bash
MODEL="--checkpoint $WORKDIR/model/final_perturb.pt \
       --graph-h5ad $WORKDIR/graph/REGA_graph_without_GEX.h5ad \
       --motif-tf reference/motif_tf_human.txt"
```

In Python, load the model once and reuse it:

```python
from rega_perturb.analysis.model_io import load_trained_model
model = load_trained_model(f"{WORKDIR}/model/final_perturb.pt",
                           f"{WORKDIR}/graph/REGA_graph_without_GEX.h5ad")
MOTIF_TF = "reference/motif_tf_human.txt"
```

TFs are the genes of the motif-to-TF table whose motif is in the model and that are
among the modeled genes. The control is detected automatically (`non-targeting`,
`NTC`, `CTRL` or `control`) or set with `--control`.

### 4.1 `rega-perturb tf-tg` — GRN inference (TF-target gene scores)

```bash
rega-perturb tf-tg $MODEL -o $WORKDIR/tf_tg
```

```python
from rega_perturb.analysis.tf_tg_scores import compute_tf_tg_scores, tf_tg_long_table
scores, genes, tfs = compute_tf_tg_scores(model, MOTIF_TF, score="influence")
table = tf_tg_long_table(scores, genes, tfs, top_per_tf=500)
```

| Option | Default | Notes |
|--------|---------|-------|
| `--score` | both | `influence` and / or `connectivity` (repeatable) |
| `--top-per-tf` | all | keep the top N target genes per TF |
| `--min-score` | — | drop rows below this score |

**Output**: `tf_tg_influence.tsv.gz`, `tf_tg_connectivity.tsv.gz` — columns
`TF, TG, score`.

---

### 4.2 `rega-perturb tf-activity` — TF activity changes

TF activity under every perturbation target compared with the control.

```bash
rega-perturb tf-activity $MODEL -o $WORKDIR/tf_activity
```

```python
from rega_perturb.analysis.tf_activity import compute_tf_activity_changes
log2fc, activity = compute_tf_activity_changes(model, MOTIF_TF)
```

| Option | Default | Notes |
|--------|---------|-------|
| `--control` | auto | control target label |
| `--threshold` | — | adds a `changed` column (\|log2 fold change\| ≥ threshold) |

**Output**: `tf_activity_log2fc.tsv` (perturbation target x TF, log2 fold change vs
control; the control row is 0) and `tf_activity_changes.tsv.gz` (long table:
`target, TF, activity_target, activity_control, log2fc`).

---

### 4.3 `rega-perturb diff-network` — Differential regulatory networks

Perturbation-specific TF-target gene networks, their difference from the control
network, and per-gene rewiring scores.

```bash
rega-perturb diff-network $MODEL -o $WORKDIR/diff_network --all-targets --format h5
rega-perturb diff-network $MODEL -o $WORKDIR/diff_network --target <TARGET> --min-abs 1e-3
```

```python
from rega_perturb.analysis.diff_network import delta_network, network_factors, rewiring_scores
f = network_factors(model, MOTIF_TF)
rrs, nrs, targets = rewiring_scores(f)          # gene x target
dw = delta_network(f, targets[0])               # gene x TF, for one target in `targets`
```

| Option | Default | Notes |
|--------|---------|-------|
| `--target` / `--all-targets` | — | targets whose edge-level networks are written |
| `--kind` | `delta` | `delta`: change versus control; `specific`: perturbation-specific network |
| `--format` | `tsv` | `tsv`: one long table per target; `h5`: one dense gene x TF matrix per target |
| `--min-abs` / `--top-k` | — | `tsv` only: edge filters |
| `--no-scores` | off | skip `rewiring_scores.h5` |

**Output**: `rewiring_scores.h5` (`RRS`, `NRS`: gene x target, with `genes` and
`targets`) for all targets; `delta_network.<target>.tsv.gz` (`target, TF, TG,
delta_w`) or `delta_networks.h5` (`/networks/<target>`); with `--kind specific`,
`specific_network.*` files with a `weight` column.

Writing the edge-level networks of all targets as `tsv` can produce large files;
the command logs the number of edges per target first. Use `--format h5`, `--min-abs` or
`--top-k` for large screens.

---

## 5. Complete example

`examples/k562_essential/run_mini.sh` runs steps 3.1–4.3 on the mini data (using the
provided motif scan instead of 3.3):

```bash
bash examples/k562_essential/run_mini.sh            # results in ./output
```

For the full K562 essential screen (raw counts from figshare, full peak list,
HOMER scan), see [`examples/k562_essential/README.md`](../examples/k562_essential/README.md).
