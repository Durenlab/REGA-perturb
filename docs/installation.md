# Installation

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python | ≥ 3.10 | conda recommended |
| PyTorch | ≥ 2.0 | install separately (CPU or CUDA build) |
| bedtools | — | for `rega-perturb retg` |
| HOMER | — | for `rega-perturb motif`; genome package required |

Python dependencies installed with the package: numpy ≥ 1.24, scipy ≥ 1.10,
pandas ≥ 2.1, h5py ≥ 3.8, anndata ≥ 0.10, click ≥ 8.0, pyyaml ≥ 6.0, tqdm ≥ 4.60.

---

## 1. Clone and install REGA-perturb

```bash
# 1. Clone the repository
git clone https://github.com/Durenlab/REGA-perturb.git
cd REGA-perturb

# 2. Create and activate an environment
conda create -n rega-perturb python=3.11
conda activate rega-perturb

# 3. Install PyTorch (see section 2)

# 4. Install REGA-perturb
pip install .
# or, for development (editable install with test tools):
# pip install -e ".[dev]"
```

> **HPC users**: if your home directory has a tight quota, create the environment
> in project storage:
> ```bash
> conda create --prefix /path/to/project/envs/rega-perturb python=3.11
> conda activate /path/to/project/envs/rega-perturb
> ```

---

## 2. PyTorch — install separately

PyTorch is not installed automatically because the correct build (CPU or a
specific CUDA version) depends on your hardware and driver.

**Step 1** — check your CUDA version (GPU users):

```bash
nvidia-smi    # CUDA version shown in the top-right corner
```

**Step 2** — get the install command for your system from
https://pytorch.org/get-started/locally/. Common examples:

```bash
# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 3** — verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**A GPU is strongly recommended** for model training. The bundled mini example
also runs on a CPU in minutes.

---

## 3. bedtools

bedtools is used by `rega-perturb retg` to pair genes with the regulatory elements
in their TSS window.

```bash
conda install -c bioconda bedtools   # conda
sudo apt install bedtools             # Ubuntu/Debian
module load bedtools                  # HPC module system
```

Verify:

```bash
bedtools --version
```

A different executable can be given with `rega-perturb retg --bedtools /path/to/bedtools`.

---

## 4. HOMER (motif scanning)

HOMER is used by `rega-perturb motif` to scan regulatory elements for TF binding
motifs. Follow the official guide at
http://homer.ucsd.edu/homer/introduction/install.html:

```bash
mkdir homer && cd homer
wget http://homer.ucsd.edu/homer/configureHomer.pl
perl configureHomer.pl -install
perl configureHomer.pl -install hg38     # genome package (human)
export PATH=$PATH:/path/to/homer/bin     # add to ~/.bashrc for persistence
```

Verify:

```bash
findMotifsGenome.pl    # should print HOMER usage
```

> **HPC users**: HOMER may be available as a module (`module avail homer`).

---

## 5. Verify the full installation

```bash
# REGA-perturb CLI
rega-perturb --version
rega-perturb --help

# PyTorch + GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# bedtools and HOMER
bedtools --version
findMotifsGenome.pl 2>&1 | head -2

# end-to-end check on the bundled mini example (CPU, a few minutes)
bash examples/k562_essential/run_mini.sh
```

---

## 6. HPC tips

### Long-running jobs

Use `tmux` or `screen` to keep interactive sessions alive:

```bash
tmux new -s rega_perturb
# ... run your commands ...
# Ctrl+B, D  to detach;  tmux attach -t rega_perturb  to reconnect
```

### GPU node allocation (SLURM)

Interactive session:

```bash
srun -p gpu --gres=gpu:1 --mem=64G --time=8:00:00 --pty bash
conda activate rega-perturb
rega-perturb train --block-dir out/blocks -o out/model --config examples/k562_essential/train_mini.yaml --device cuda
```

Batch job:

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=rega_perturb_train

conda activate /path/to/envs/rega-perturb
rega-perturb train --block-dir out/blocks -o out/model --config examples/k562_essential/train_mini.yaml --device cuda
```

`train_mini.yaml` holds the settings of the mini example; use your own configuration for
larger data sets. Partition names, memory and time limits depend on your cluster and data size.
