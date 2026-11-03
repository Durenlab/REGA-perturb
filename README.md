# REGA-perturb

> Reference **RE**gulatory Path-**G**uided **A**nalysis
> Resolving how genetic perturbations act through gene regulatory paths

## Overview

REGA-perturb is a Reference REgulatory Path-Guided Analysis framework for resolving how
genetic perturbations act through gene regulatory paths.

## Key Features

- **Complete workflow** — raw Perturb-seq counts + reference peaks → regulatory outputs
- **Modular CLI** — each step runs on its own (`rega-perturb qc`, `peaks`, `motif`, …)
- **Flexible input** — raw UMI counts or already-normalized data
- **Large-scale data** — applicable to million-cell Perturb-seq screens (streamed preprocessing, mini-batch training)
- **Built-in downstream analysis** — GRN inference, perturbation-induced TF activity changes, differential regulatory networks
- **Reproducible** — every step records its parameters, random seeds and software versions
- **GPU strongly recommended** — model training on a GPU is strongly recommended; the bundled mini example also runs on a CPU

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | |
| **PyTorch** | Required for model training (CPU or CUDA build) |
| bedtools | For gene - regulatory element pairs |
| HOMER (with genome package) | For motif scanning |

See [docs/installation.md](docs/installation.md) for full setup instructions.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Durenlab/REGA-perturb.git
cd REGA-perturb

# 2. Create an environment (optional)
conda create -n rega-perturb python=3.11
conda activate rega-perturb

# 3. Install PyTorch — choose the build that matches your hardware
# Visit https://pytorch.org/get-started/locally/ for the exact command.
# Examples:
#   GPU (CUDA 12.4):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   CPU-only:         pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install REGA-perturb
pip install .
```

### Install HOMER and bedtools (separately)

HOMER: follow the official guide
http://homer.ucsd.edu/homer/introduction/install.html, then install the genome package:

```bash
configureHomer.pl -install hg38
```

bedtools: https://bedtools.readthedocs.io (e.g. `conda install -c bioconda bedtools`).

### Verify

```bash
rega-perturb --version
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Quick Start

```bash
# Option 1: bundled mini example (K562 essential-gene screen subset; minutes on a CPU)
bash examples/k562_essential/run_mini.sh

# Option 2: step-by-step
rega-perturb qc -i data.h5ad -o out/qc --prefix sample
rega-perturb peaks -i peaks.bed -o out/reference/peaks_named.bed
rega-perturb motif -i out/reference/peaks_named.bed --genome hg38 \
    --motif-file reference/motifs_homer.txt -o out/reference --prefix sample
rega-perturb retg --tss-file out/qc/sample_REGA_qc_gene_TSS_300kb.txt \
    --peaks-bed out/reference/peaks_named.bed -o out/reference/re_tg.tsv
rega-perturb build-graph --gene-metadata out/qc/sample_REGA_qc_gene_metadata.csv \
    --re-tg out/reference/re_tg.tsv --peak-motif out/reference/sample_peak_motif.txt -o out/graph
rega-perturb build-blocks --graph-h5ad out/graph/REGA_graph_without_GEX.h5ad --raw-h5ad data.h5ad \
    --cell-metadata out/qc/sample_REGA_qc_cell_metadata.csv \
    --gene-metadata out/qc/sample_REGA_qc_gene_metadata.csv -o out/blocks
rega-perturb train --block-dir out/blocks -o out/model \
    --config examples/k562_essential/train_mini.yaml --device cuda
rega-perturb assemble --graph-h5ad out/graph/REGA_graph_without_GEX.h5ad \
    --checkpoint out/model/final_perturb.pt -o out/model/rega_perturb_result.h5ad \
    --motif-tf reference/motif_tf_human.txt
```

Training parameters are read from a YAML file; `examples/k562_essential/train_mini.yaml`
holds the settings of the mini example (adapt them for larger data sets).

Normalization (`build-blocks --expr-mode`): `logcp10k` (default), `logcpm`,
`as_is` (already normalized input; use `qc --input-type normalized`), `counts`.

Input formats and the provided reference files: see
[tutorial §2](docs/tutorial.md#2-data-preparation).

## Downstream Analysis

```bash
MODEL="--checkpoint out/model/final_perturb.pt --graph-h5ad out/graph/REGA_graph_without_GEX.h5ad --motif-tf reference/motif_tf_human.txt"

# TF-target gene scores
rega-perturb tf-tg $MODEL -o out/tf_tg

# Perturbation-induced TF activity changes
rega-perturb tf-activity $MODEL -o out/tf_activity

# Differential networks and per-gene rewiring scores
rega-perturb diff-network $MODEL -o out/diff_network --target <TARGET>
```

`diff-network` writes the per-gene RRS / NRS scores of all targets (unless `--no-scores`); `--target` (repeatable)
or `--all-targets` selects the targets whose edge-level networks are written.

## Example

[`examples/k562_essential`](examples/k562_essential): the complete workflow on a K562
CRISPRi Perturb-seq screen (Replogle et al., 2022), with bundled mini data and
download instructions for the full data.

## Documentation

- [Installation guide](docs/installation.md)
- [Step-by-step tutorial](docs/tutorial.md)
- Command help: `rega-perturb <command> --help`
- API reference: `help(rega_perturb.<module>)` or docstrings in `src/rega_perturb/`

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

Citation information will be added on publication.

## Author

**Lixin Ren**
Duren Lab, Indiana University School of Medicine
Email: rlxmath017@gmail.com

## License

MIT — see [LICENSE](LICENSE)
