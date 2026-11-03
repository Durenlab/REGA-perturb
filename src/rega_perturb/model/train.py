"""Model training for REGA-perturb.

Trains the REGA-perturb model on per-chromosome input blocks (``chr*.npz``) and
per-cell perturbation assignments (``P_perturb_target.npz``, ``P_cells.txt``,
``P_targets.txt``) produced by :func:`rega_perturb.preprocess.blocks.build_blocks`.

Outputs written to ``output_dir``:

* ``final_perturb.pt`` -- trained model (see :func:`load_perturb_checkpoint`);
* ``perturb_loss_log.csv`` and ``perturb_rre_log.csv`` -- training logs;
* ``perturb_training_curves.png`` -- training curves (if matplotlib is installed).

"""

from __future__ import annotations

import csv
import glob
import itertools
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for REGA-Perturb. "
        "Install PyTorch from https://pytorch.org/get-started/locally/."
    ) from e

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


from rega_perturb.utils import get_logger

logger = get_logger("model")

__all__ = [
    "PerturbTrainingConfig",
    "PerturbBlockData",
    "load_perturb_blocks",
    "train_rega_perturb",
    "save_perturb_checkpoint",
    "load_perturb_checkpoint",
]


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class _TrainingSchedule:
    """Fixed training schedule (not user-configurable).

    Phase 1 keeps P fixed for `phase1_epochs` epochs. Phase 2 also updates P;
    it runs at least `phase2_min_epochs` epochs and then continues until the
    full-dataset RRE has converged: its relative change between consecutive
    full evaluations (every `full_eval_every` epochs) is below `conv_tol` on
    `early_stop_patience` consecutive evaluations after the minimum.
    `phase2_max_epochs` = None means no upper limit.
    """

    phase1_epochs: int = 15
    phase2_min_epochs: int = 60
    phase2_max_epochs: int | None = None
    full_eval_every: int = 5
    conv_tol: float = 1e-4
    early_stop_patience: int = 2


@dataclass
class PerturbTrainingConfig:
    """Hyperparameters for mini-batch REGA-Perturb training."""

    # Model
    K: int = 20

    # Numerics
    eps: float = 1e-6
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    # Update stability:
    #   den_floor   : lower bound on every denominator.
    #   ratio_max   : bound on the per-step change of each parameter.
    #   check_nan   : if True, detect non-finite parameters each batch and raise
    #                 with the offending step/block for fast diagnosis.
    den_floor: float = 1e-6
    ratio_max: float = 1e4
    check_nan: bool = True
    # Optional hard element-wise upper bound on parameters after each update
    # (0/None disables).
    clamp_max: float = 0.0

    # Mini-batch
    batch_size: int = 1024
    shuffle_cells: bool = True
    seed: int = 123

    # Parameter update switches
    update_W: bool = True
    update_B: bool = True
    update_V: bool = True
    update_H: bool = True

    # Re-impose the input sparsity patterns explicitly (off by default).
    enforce_W_support: bool = False
    enforce_B_support: bool = False

    # Phase-2 settings
    update_P_in_phase2: bool = True
    p_update_every: int = 10
    p_min: float = 1e-4
    p_max: float = 10.0
    rescale_P_by_target: bool = True

    # Regularization
    lam_W: float = 0.0
    lam_B: float = 0.0
    lam_V: float = 0.0
    lam_H: float = 0.0
    lam_P: float = 0.0

    # Lower bound on parameters
    clamp_min: float = 0.0

    # Logging / progress
    log_every_epoch: int = 1
    eval_batches: int = 5
    show_progress: bool = True
    # At the end of training, render batch-loss and RRE curves to a PNG in the
    # output directory (requires matplotlib; failure is non-fatal).
    make_plot: bool = True

    # Initialization
    V_seed: int = 123
    H_seed: int = 1234

    # Performance
    # x_on_device selects the host-side X layout used for fast mini-batch
    # gather (no sparse tensor lives on the GPU, because column gather on CUDA
    # sparse tensors is unreliable across PyTorch versions):
    #   True  -> store X^T as CSR and slice batch ROWS (cheap; recommended).
    #   False -> store X as CSC and slice batch COLUMNS (equivalent, a bit
    #            slower to gather).
    # In both cases only the small dense mini-batch is moved to the device.
    x_on_device: bool = True
    pin_memory: bool = True            # pin + async H2D copy of the mini-batch
    allow_tf32: bool = True            # enable TF32 matmul on Ampere+ GPUs
    use_inference_mode: bool = True    # use torch.inference_mode for updates
    log1p_x: bool = False              # apply log1p to X (stabilizes raw counts)

    # Initialization scaling: rescale the random initialization to the scale of
    # the input data before training.
    scale_init_to_data: bool = True
    scale_init_n_batches: int = 4      # how many batches to estimate scale from

    # Periodic rescaling of parameters for numerical stability (on by default).
    rebalance_WB: bool = True
    rebalance_every: int = 0           # also rebalance every N batches (0=off)

    # Diagnostics.
    # If diag_every > 0, log per-parameter min/max/norm statistics every
    # `diag_every` batches. Pure diagnostics; does not change training.
    diag_every: int = 0


@dataclass
class PerturbBlockData:
    """Input data for REGA-Perturb."""

    X_blocks: list[sp.spmatrix]        # gene_chr x cell  (stored as CSC)
    W_blocks: list[sp.csr_matrix]      # gene_chr x RE_chr
    B_blocks: list[sp.csr_matrix]      # RE_chr x motif
    chr_names: list[str]

    genes_blocks: list[np.ndarray]
    RE_blocks: list[np.ndarray]
    cells: np.ndarray
    motifs: np.ndarray

    target_idx: np.ndarray             # length = n_cells
    perturb_targets: np.ndarray        # length = n_targets


# ============================================================
# I/O utilities
# ============================================================

def _load_csr_from_npz_components(z: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    """Load a CSR matrix from component arrays stored by build_blocks."""
    data = z[f"{prefix}_data"]
    indices = z[f"{prefix}_indices"]
    indptr = z[f"{prefix}_indptr"]
    shape = tuple(z[f"{prefix}_shape"])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _read_name_file(path: Path) -> np.ndarray:
    """Read a one-name-per-line text file, ignoring blank lines / trailing whitespace."""
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln != ""]
    return np.asarray(lines).astype(str)


def _load_target_idx_from_P(P_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sparse cell x perturbation-target P and convert to target_idx and p_values."""
    P = sp.load_npz(str(P_file)).tocsr()

    if P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError(f"P matrix is empty: {P_file}")

    nnz_per_row = np.diff(P.indptr)

    if not np.all(nnz_per_row == 1):
        bad = np.where(nnz_per_row != 1)[0][:10]
        raise ValueError(
            "This implementation expects exactly one nonzero perturbation target per cell. "
            f"Found {np.sum(nnz_per_row != 1)} invalid rows. "
            f"Example invalid rows: {bad.tolist()}"
        )

    target_idx = P.indices[P.indptr[:-1]].astype(np.int64)
    p_init = P.data[P.indptr[:-1]].astype(np.float32)

    if not np.all(np.isfinite(p_init)):
        raise ValueError("P contains NaN or Inf values.")

    if np.any(p_init <= 0):
        raise ValueError("Initial nonzero P values must be positive.")

    return target_idx, p_init


def load_perturb_blocks(
    block_dir: Path,
    chroms: list[str] | None = None,
) -> tuple[PerturbBlockData, np.ndarray]:
    """Load chromosome blocks and the perturbation assignment written by build_blocks."""
    block_dir = Path(block_dir)

    npz_files = sorted(glob.glob(str(block_dir / "chr*.npz")))
    if not npz_files:
        raise ValueError(f"No chr*.npz files found in {block_dir}")

    X_blocks: list[sp.spmatrix] = []
    W_blocks: list[sp.csr_matrix] = []
    B_blocks: list[sp.csr_matrix] = []
    chr_names: list[str] = []
    genes_blocks: list[np.ndarray] = []
    RE_blocks: list[np.ndarray] = []

    cells_ref: np.ndarray | None = None
    motifs_ref: np.ndarray | None = None

    for f in npz_files:
        chr_name = Path(f).stem

        if chroms is not None and chr_name not in chroms:
            continue

        z = np.load(f, allow_pickle=False)

        required = [
            "X_data", "X_indices", "X_indptr", "X_shape",
            "W_data", "W_indices", "W_indptr", "W_shape",
            "B_data", "B_indices", "B_indptr", "B_shape",
            "genes", "RE", "cells", "motifs",
        ]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{chr_name}.npz missing keys: {missing}")

        # X is stored as CSR components but we keep it as CSC, because the
        # training loop slices columns (cells) heavily and CSC column
        # slicing is far faster than CSR.
        X = _load_csr_from_npz_components(z, "X").astype(np.float32).tocsc()
        W = _load_csr_from_npz_components(z, "W").astype(np.float32)
        B = _load_csr_from_npz_components(z, "B").astype(np.float32)

        genes = np.asarray(z["genes"]).astype(str)
        RE = np.asarray(z["RE"]).astype(str)
        cells = np.asarray(z["cells"]).astype(str)
        motifs = np.asarray(z["motifs"]).astype(str)

        if X.shape[0] != len(genes):
            raise ValueError(f"{chr_name}: X rows {X.shape[0]} != genes {len(genes)}")
        if X.shape[1] != len(cells):
            raise ValueError(f"{chr_name}: X cols {X.shape[1]} != cells {len(cells)}")
        if W.shape[0] != len(genes):
            raise ValueError(f"{chr_name}: W rows {W.shape[0]} != genes {len(genes)}")
        if W.shape[1] != len(RE):
            raise ValueError(f"{chr_name}: W cols {W.shape[1]} != RE {len(RE)}")
        if B.shape[0] != len(RE):
            raise ValueError(f"{chr_name}: B rows {B.shape[0]} != RE {len(RE)}")
        if B.shape[1] != len(motifs):
            raise ValueError(f"{chr_name}: B cols {B.shape[1]} != motifs {len(motifs)}")

        if cells_ref is None:
            cells_ref = cells
        elif not np.array_equal(cells_ref, cells):
            raise ValueError(f"{chr_name}: cell order inconsistent across blocks")

        if motifs_ref is None:
            motifs_ref = motifs
        elif not np.array_equal(motifs_ref, motifs):
            raise ValueError(f"{chr_name}: motif order inconsistent across blocks")

        X_blocks.append(X)
        W_blocks.append(W)
        B_blocks.append(B)
        chr_names.append(chr_name)
        genes_blocks.append(genes)
        RE_blocks.append(RE)

    if not X_blocks:
        raise ValueError("No valid chromosome blocks loaded.")

    if cells_ref is None or motifs_ref is None:
        raise ValueError("Failed to infer cells or motifs from chromosome blocks.")

    P_file = block_dir / "P_perturb_target.npz"
    P_cells_file = block_dir / "P_cells.txt"
    P_targets_file = block_dir / "P_targets.txt"

    if not P_file.exists():
        raise ValueError(f"Missing {P_file}")
    if not P_cells_file.exists():
        raise ValueError(f"Missing {P_cells_file}")
    if not P_targets_file.exists():
        raise ValueError(f"Missing {P_targets_file}")

    p_cells = _read_name_file(P_cells_file)
    perturb_targets = _read_name_file(P_targets_file)

    if not np.array_equal(cells_ref, p_cells):
        raise ValueError("Cell order mismatch between chromosome blocks and P_cells.txt")

    target_idx, p_init = _load_target_idx_from_P(P_file)

    if len(target_idx) != len(cells_ref):
        raise ValueError("P row count does not match number of cells")

    if np.max(target_idx) >= len(perturb_targets):
        raise ValueError("P contains perturbation target index outside P_targets.txt")

    block_data = PerturbBlockData(
        X_blocks=X_blocks,
        W_blocks=W_blocks,
        B_blocks=B_blocks,
        chr_names=chr_names,
        genes_blocks=genes_blocks,
        RE_blocks=RE_blocks,
        cells=cells_ref,
        motifs=motifs_ref,
        target_idx=target_idx,
        perturb_targets=perturb_targets,
    )

    logger.info(
        "[load_perturb_blocks]\n"
        f"  block_dir={block_dir}\n"
        f"  blocks={len(X_blocks)}\n"
        f"  cells={len(cells_ref)}\n"
        f"  motifs={len(motifs_ref)}\n"
        f"  perturb targets={len(perturb_targets)}"
    )

    for name, X, W, B in zip(chr_names, X_blocks, W_blocks, B_blocks):
        logger.info(
            f"  {name}: X{X.shape} nnz={X.nnz}; "
            f"W{W.shape} nnz={W.nnz}; "
            f"B{B.shape} nnz={B.nnz}"
        )

    return block_data, p_init


# ============================================================
# Training utilities
# ============================================================

def _make_batches(
    n_cells: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    """Yield mini-batches of cell indices."""
    idx = np.arange(n_cells)
    if shuffle:
        rng.shuffle(idx)

    for start in range(0, n_cells, batch_size):
        yield idx[start:start + batch_size]


def _progress_iter(iterable, total: int | None, desc: str, enabled: bool):
    """Return tqdm iterator if available and enabled."""
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def _no_grad_ctx(config: "PerturbTrainingConfig"):
    """Return inference_mode or no_grad context based on config."""
    if getattr(config, "use_inference_mode", False) and hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


def _safe_ratio(
    num: torch.Tensor,
    den: torch.Tensor,
    den_floor: float,
    ratio_max: float,
) -> torch.Tensor:
    """Numerically safe update ratio.

    - Floors the denominator magnitude so it can never be ~0 (avoids 0/0).
    - Replaces any non-finite entries with 1.0 (no update for that entry).
    - Clamps the ratio to [1/ratio_max, ratio_max] so one step cannot blow a
      factor up or collapse it to ~0 in a single update.
    """
    den = den.clamp_min(den_floor)
    r = num / den
    r = torch.nan_to_num(r, nan=1.0, posinf=ratio_max, neginf=0.0)
    if ratio_max is not None and ratio_max > 0:
        r = r.clamp_(min=1.0 / ratio_max, max=ratio_max)
    return r


def _assert_finite(name: str, *tensors: torch.Tensor) -> None:
    """Raise if any tensor contains NaN/Inf, identifying the offending factor."""
    for i, t in enumerate(tensors):
        if not torch.isfinite(t).all():
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            raise FloatingPointError(
                f"Non-finite values in {name}"
                f"{'' if len(tensors) == 1 else f'[{i}]'}: "
                f"{n_nan} NaN, {n_inf} Inf. "
                "Likely divergence: lower the learning signal "
                "(normalize X, reduce p_max, increase eps/den_floor, "
                "or disable TF32 with --no-tf32)."
            )


def _maybe_enable_tf32(config: "PerturbTrainingConfig", device: str) -> None:
    """Enable TF32 matmul on CUDA if requested (near-free speedup on Ampere+)."""
    if getattr(config, "allow_tf32", False) and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _prepare_x_blocks(
    X_blocks: list[sp.spmatrix],
    device: str,
    dtype: torch.dtype,
    x_on_device: bool,
    log1p: bool = False,
) -> tuple[list[Any], bool]:
    """Prepare X blocks for mini-batch column gather.

    Column gather / index_select / CSR@CSR spmm on sparse CUDA tensors is not
    reliably implemented across PyTorch versions, so we do NOT keep X as a
    device sparse tensor. Instead we always slice on the host and transfer the
    (small) dense mini-batch to the device.

    If log1p is True, X is transformed with log1p (applied to the sparse
    nonzeros only, which is exact since log1p(0)=0).

    Two host layouts are supported:

      "fast"   (x_on_device=True): store X^T as scipy CSR (cell x gene). A
               mini-batch is a set of ROWS, and CSR row-slicing only touches
               the nnz of those rows -- cheap and cache-friendly. The sliced
               (batch x gene) block is transferred (pinned/async on CUDA) and
               transposed on-device to (gene x batch).

      "csc"    (x_on_device=False): keep scipy CSC (gene x cell) and column-
               slice per batch. Equivalent result, slightly slower gather.

    The flag name is kept for backward compatibility; in both cases the heavy
    matrices stay sparse on the host and only the dense mini-batch lives on the
    device.

    Returns (prepared_blocks, use_fast_row_layout).
    """
    def _maybe_log1p(X: sp.spmatrix) -> sp.spmatrix:
        if not log1p:
            return X
        X = X.copy()
        X.data = np.log1p(X.data.astype(np.float64)).astype(np.float32)
        return X

    fast = bool(x_on_device)

    if fast:
        # Store X^T as CSR for fast row (=cell) slicing.
        return [_maybe_log1p(X).tocsr().T.tocsr() for X in X_blocks], True

    return [_maybe_log1p(X).tocsc() for X in X_blocks], False


def _gather_batch_dense(
    X_block: Any,
    cols_np: np.ndarray,
    device: str,
    dtype: torch.dtype,
    fast_row_layout: bool,
    pin_memory: bool,
) -> torch.Tensor:
    """Return X[:, cols] as a dense gene_chr x batch tensor on `device`.

    fast_row_layout=True : X_block is scipy CSR holding X^T (cell x gene).
                           Slice the batch ROWS, transfer, transpose on-device.
    fast_row_layout=False: X_block is scipy CSC (gene x cell). Slice columns.
    """
    is_cuda = str(device).startswith("cuda")

    if fast_row_layout:
        # batch x gene on host (row slice of X^T)
        arr = X_block[cols_np, :].toarray().astype(np.float32, copy=False)
        t = torch.from_numpy(arr)
        if pin_memory and is_cuda:
            t = t.pin_memory()
            t = t.to(device, dtype=dtype, non_blocking=True)
        else:
            t = t.to(device, dtype=dtype)
        return t.t().contiguous()                       # gene x batch

    # CSC column slice: gene x batch on host
    arr = X_block[:, cols_np].toarray().astype(np.float32, copy=False)
    t = torch.from_numpy(arr)
    if pin_memory and is_cuda:
        t = t.pin_memory()
        return t.to(device, dtype=dtype, non_blocking=True)
    return t.to(device, dtype=dtype)


def _densify_batch(
    X_blocks: list[Any],
    cols_np: np.ndarray,
    device: str,
    dtype: torch.dtype,
    fast_row_layout: bool,
    pin_memory: bool,
) -> list[torch.Tensor]:
    """Densify the requested columns for every block exactly once."""
    return [
        _gather_batch_dense(
            X_block, cols_np, device, dtype, fast_row_layout, pin_memory
        )
        for X_block in X_blocks
    ]


def _compute_batch_loss(
    Xb_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    batch_idx: torch.Tensor,
    dtype: torch.dtype,
    device: str,
) -> float:
    """Compute the training loss for one mini-batch from pre-densified blocks."""
    targets = target_idx[batch_idx]
    p_batch = p_values[batch_idx]

    Q = H[:, targets] * p_batch.unsqueeze(0)
    Z = V @ Q

    loss = torch.tensor(0.0, device=device, dtype=dtype)

    with torch.no_grad():
        for Xb, Wc, Bc in zip(Xb_blocks, W_blocks, B_blocks):
            Y = Wc @ (Bc @ Z)
            diff = Xb - Y
            loss += (diff * diff).sum()

    return float(loss.detach().cpu())


def _compute_full_rre(
    X_prepared: list[Any],
    n_cells: int,
    batch_size: int,
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    fast_row_layout: bool,
    pin_memory: bool,
) -> dict[str, float]:
    """Full-dataset relative reconstruction error (RRE) over ALL cells.

    Sweeps every cell exactly once (no sampling) and returns ``rre``, ``sse``
    (sum of squared residuals) and ``xnorm`` (norm of the data). This is the
    quantity to track for convergence; the per-batch training loss is a biased
    post-update estimate and must not be used as the criterion.
    """
    sse = 0.0
    xnorm2 = 0.0
    n_batches = int(np.ceil(n_cells / batch_size))

    with torch.no_grad():
        for bstart in range(0, n_cells, batch_size):
            cols = np.arange(bstart, min(bstart + batch_size, n_cells))
            idx = torch.as_tensor(cols, device=device, dtype=torch.long)
            Xb_blocks = _densify_batch(
                X_prepared, cols, device, dtype,
                fast_row_layout=fast_row_layout, pin_memory=pin_memory,
            )
            targets = target_idx[idx]
            p_batch = p_values[idx]
            Q = H[:, targets] * p_batch.unsqueeze(0)
            Z = V @ Q
            for Xb, Wc, Bc in zip(Xb_blocks, W_blocks, B_blocks):
                Y = Wc @ (Bc @ Z)
                diff = Xb - Y
                sse += float((diff * diff).sum())
                xnorm2 += float((Xb * Xb).sum())

    rre = float(np.sqrt(sse / xnorm2)) if xnorm2 > 0 else float("nan")
    return {"rre": rre, "sse": sse, "xnorm": float(np.sqrt(xnorm2))}


def _reconstruction_scale(
    Xb_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[float, float]:
    """Return the norm of the data and of its reconstruction over the mini-batch."""
    targets = target_idx[batch_idx]
    p_batch = p_values[batch_idx]
    Q = H[:, targets] * p_batch.unsqueeze(0)
    Z = V @ Q
    x_sq = 0.0
    y_sq = 0.0
    for Xb, Wc, Bc in zip(Xb_blocks, W_blocks, B_blocks):
        Y = Wc @ (Bc @ Z)
        x_sq += float((Xb * Xb).sum())
        y_sq += float((Y * Y).sum())
    return float(np.sqrt(x_sq)), float(np.sqrt(y_sq))


def _scale_init_VH_(
    Xb_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    batch_idx: torch.Tensor,
    eps: float,
) -> float:
    """Rescale the initialization in place to match the data scale.

    Returns the applied scaling factor.
    """
    x_norm, y_norm = _reconstruction_scale(
        Xb_blocks, W_blocks, B_blocks, V, H, target_idx, p_values, batch_idx
    )
    if y_norm <= eps or x_norm <= eps:
        return 1.0
    alpha = float(np.sqrt(x_norm / y_norm))
    V.mul_(alpha)
    H.mul_(alpha)
    return alpha


def _factor_stats(name: str, t: torch.Tensor) -> str:
    """Format min/max/Fro-norm of a tensor for diagnostics."""
    with torch.no_grad():
        tmin = float(t.min())
        tmax = float(t.max())
        fro = float(t.norm())
    return f"{name}[min={tmin:.2e} max={tmax:.2e} fro={fro:.2e}]"


def _blocks_stats(name: str, blocks: list[torch.Tensor]) -> str:
    """Format min/max/Fro-norm aggregated across a list of block tensors."""
    with torch.no_grad():
        tmin = min(float(b.min()) for b in blocks)
        tmax = max(float(b.max()) for b in blocks)
        fro = float(torch.sqrt(sum((b * b).sum() for b in blocks)))
    return f"{name}[min={tmin:.2e} max={tmax:.2e} fro={fro:.2e}]"


def _minmaxfro(blocks_or_tensor) -> tuple[float, float, float]:
    """Return (min, max, fro) for a tensor or a list of block tensors."""
    with torch.no_grad():
        if isinstance(blocks_or_tensor, (list, tuple)):
            bs = blocks_or_tensor
            tmin = min(float(b.min()) for b in bs)
            tmax = max(float(b.max()) for b in bs)
            fro = float(torch.sqrt(sum((b * b).sum() for b in bs)))
        else:
            t = blocks_or_tensor
            tmin = float(t.min())
            tmax = float(t.max())
            fro = float(t.norm())
    return tmin, tmax, fro


def _print_diagnostics(
    tag: str,
    Xb_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Log per-parameter statistics and the reconstruction-to-data scale ratio."""
    x_norm, y_norm = _reconstruction_scale(
        Xb_blocks, W_blocks, B_blocks, V, H, target_idx, p_values, batch_idx
    )
    ratio = (y_norm / x_norm) if x_norm > 0 else float("nan")
    p_batch = p_values[batch_idx]
    parts = [
        _factor_stats("W0", W_blocks[0]),
        _factor_stats("B0", B_blocks[0]),
        _factor_stats("V", V),
        _factor_stats("H", H),
        _factor_stats("p_b", p_batch),
        f"||X||={x_norm:.2e} ||recon||={y_norm:.2e} recon/X={ratio:.2e}",
    ]
    logger.info(f"  [diag {tag}] " + "  ".join(parts))


def _rebalance_chain_(
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    eps: float,
) -> None:
    """Rescale parameters in place for numerical stability (the fit is unchanged)."""
    with torch.no_grad():
        for Wc, Bc in zip(W_blocks, B_blocks):
            w_norm = Wc.norm(dim=0)
            b_norm = Bc.norm(dim=1)
            valid = (w_norm > eps) & (b_norm > eps)
            s = torch.ones_like(w_norm)
            s[valid] = torch.sqrt(b_norm[valid] / w_norm[valid])
            Wc.mul_(s.unsqueeze(0))
            Bc.mul_((1.0 / s).unsqueeze(1))

        n_motif = V.shape[0]
        b_motif_sq = torch.zeros(n_motif, device=V.device, dtype=V.dtype)
        for Bc in B_blocks:
            b_motif_sq += (Bc * Bc).sum(dim=0)
        b_motif = torch.sqrt(b_motif_sq)
        v_motif = V.norm(dim=1)
        valid = (b_motif > eps) & (v_motif > eps)
        s = torch.ones_like(b_motif)
        s[valid] = torch.sqrt(b_motif[valid] / v_motif[valid])
        for Bc in B_blocks:
            Bc.mul_((1.0 / s).unsqueeze(0))
        V.mul_(s.unsqueeze(1))

        v_mod = V.norm(dim=0)
        h_mod = H.norm(dim=1)
        valid = (v_mod > eps) & (h_mod > eps)
        s = torch.ones_like(v_mod)
        s[valid] = torch.sqrt(h_mod[valid] / v_mod[valid])
        V.mul_(s.unsqueeze(0))
        H.mul_((1.0 / s).unsqueeze(1))

        wn = torch.sqrt(sum((Wc * Wc).sum() for Wc in W_blocks)).clamp_min(eps)
        bn = torch.sqrt(sum((Bc * Bc).sum() for Bc in B_blocks)).clamp_min(eps)
        vn = V.norm().clamp_min(eps)
        hn = H.norm().clamp_min(eps)

        log_target = (torch.log(wn) + torch.log(bn) + torch.log(vn) + torch.log(hn)) / 4.0
        target = torch.exp(log_target)

        a = target / wn
        b = target / bn
        c = target / vn
        d = target / hn
        for Wc in W_blocks:
            Wc.mul_(a)
        for Bc in B_blocks:
            Bc.mul_(b)
        V.mul_(c)
        H.mul_(d)


def _rescale_p_by_target_(
    H: torch.Tensor,
    p_values: torch.Tensor,
    target_idx: torch.Tensor,
    n_targets: int,
    eps: float,
    p_min: float,
    p_max: float,
) -> None:
    """Normalize the cell-level perturbation weights per target (the fit is unchanged)."""
    device = H.device
    dtype = H.dtype

    with torch.no_grad():
        sums = torch.zeros(n_targets, device=device, dtype=dtype)
        counts = torch.zeros(n_targets, device=device, dtype=dtype)

        sums.index_add_(0, target_idx, p_values)
        counts.index_add_(0, target_idx, torch.ones_like(p_values))

        mean_p = sums / (counts + eps)
        valid = counts > 0

        H[:, valid] = H[:, valid] * mean_p[valid].unsqueeze(0)
        p_values /= mean_p[target_idx].clamp_min(eps)

        p_values.clamp_(min=p_min, max=p_max)


# ============================================================
# Mini-batch updates
# ============================================================

def _update_one_batch(
    Xb_blocks: list[torch.Tensor],
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    W_masks: list[torch.Tensor],
    B_masks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    target_idx: torch.Tensor,
    p_values: torch.Tensor,
    batch_idx: torch.Tensor,
    config: PerturbTrainingConfig,
    device: str,
    update_P: bool = False,
) -> None:
    """Run one mini-batch update (in place).

    `Xb_blocks` are the pre-densified data blocks of this mini-batch, in block
    order.
    """
    dtype = config.dtype

    targets = target_idx[batch_idx]
    p_batch = p_values[batch_idx]

    Q = H[:, targets] * p_batch.unsqueeze(0)
    Z = V @ Q
    ZZt = Z @ Z.t()

    M = V.shape[0]
    T = H.shape[1]
    n_batch = batch_idx.shape[0]

    S1 = torch.zeros(M, n_batch, device=device, dtype=dtype)
    S2 = torch.zeros(M, M, device=device, dtype=dtype)

    with _no_grad_ctx(config):
        for Xb, Wc, Bc, Wmask, Bmask in zip(
            Xb_blocks, W_blocks, B_blocks, W_masks, B_masks
        ):
            if config.update_W:
                Y_re = Bc @ Z
                W_num = Xb @ Y_re.t()
                W_den = (Wc @ Y_re) @ Y_re.t() + config.lam_W * Wc
                Wc.mul_(_safe_ratio(W_num, W_den, config.den_floor, config.ratio_max))
                Wc.clamp_(min=config.clamp_min)
                if config.clamp_max and config.clamp_max > 0:
                    Wc.clamp_(max=config.clamp_max)
                if config.enforce_W_support and Wmask is not None:
                    Wc.mul_(Wmask)

            if config.update_B:
                WtX = Wc.t() @ Xb
                B_num = WtX @ Z.t()
                WB = Wc @ Bc
                B_den = Wc.t() @ (WB @ ZZt) + config.lam_B * Bc
                Bc.mul_(_safe_ratio(B_num, B_den, config.den_floor, config.ratio_max))
                Bc.clamp_(min=config.clamp_min)
                if config.clamp_max and config.clamp_max > 0:
                    Bc.clamp_(max=config.clamp_max)
                if config.enforce_B_support and Bmask is not None:
                    Bc.mul_(Bmask)

            WB = Wc @ Bc
            S1 += WB.t() @ Xb
            S2 += WB.t() @ WB

        if config.update_V:
            QQt = Q @ Q.t()
            V_num = S1 @ Q.t()
            V_den = S2 @ V @ QQt + config.lam_V * V
            V.mul_(_safe_ratio(V_num, V_den, config.den_floor, config.ratio_max))
            V.clamp_(min=config.clamp_min)
            if config.clamp_max and config.clamp_max > 0:
                V.clamp_(max=config.clamp_max)

        if config.update_H:
            G = V.t() @ S2 @ V
            A = V.t() @ S1

            H_num = torch.zeros_like(H)
            weighted_A = A * p_batch.unsqueeze(0)
            H_num.index_add_(1, targets, weighted_A)

            p2_sum = torch.zeros(T, device=device, dtype=dtype)
            p2_sum.index_add_(0, targets, p_batch * p_batch)

            H_den = (G @ H) * p2_sum.unsqueeze(0) + config.lam_H * H

            present = p2_sum > 0
            ratio = torch.ones_like(H)
            ratio[:, present] = _safe_ratio(
                H_num[:, present], H_den[:, present],
                config.den_floor, config.ratio_max,
            )
            H.mul_(ratio)
            H.clamp_(min=config.clamp_min)
            if config.clamp_max and config.clamp_max > 0:
                H.clamp_(max=config.clamp_max)

        if update_P:
            P_num = torch.zeros(n_batch, device=device, dtype=dtype)
            P_den = torch.zeros(n_batch, device=device, dtype=dtype)

            for Xb, Wc, Bc in zip(Xb_blocks, W_blocks, B_blocks):
                WB = Wc @ Bc
                C_gene = WB @ V

                D = C_gene @ H[:, targets]

                P_num += (Xb * D).sum(dim=0)
                P_den += (D * D).sum(dim=0)

            P_den = (P_den + config.lam_P).clamp_min(config.den_floor)
            p_direct = torch.nan_to_num(P_num / P_den, nan=config.p_min)
            p_direct.clamp_(min=config.p_min, max=config.p_max)

            p_values[batch_idx] = p_direct

        # --------------------------------------------------------
        # Optional non-finite guard (cheap; helps localize divergence)
        # --------------------------------------------------------
        if config.check_nan:
            if config.update_W:
                _assert_finite("W_blocks", *W_blocks)
            if config.update_B:
                _assert_finite("B_blocks", *B_blocks)
            if config.update_V:
                _assert_finite("V", V)
            if config.update_H:
                _assert_finite("H", H)


# ============================================================
# Training loop
# ============================================================

def train_rega_perturb(
    block_data: PerturbBlockData,
    p_init: np.ndarray,
    output_dir: Path,
    config: PerturbTrainingConfig | None = None,
    device: str | None = None,
    _schedule: _TrainingSchedule | None = None,
) -> Path:
    """Train REGA-Perturb with mini-batch updates.

    The training schedule is fixed (see `_TrainingSchedule`); `_schedule` is
    for internal tests only.
    """
    if config is None:
        config = PerturbTrainingConfig()
    schedule = _TrainingSchedule() if _schedule is None else _schedule
    for name in ("batch_size", "p_update_every", "log_every_epoch"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be >= 1")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    n_cells = len(block_data.cells)
    n_targets = len(block_data.perturb_targets)
    n_motifs = len(block_data.motifs)

    logger.info(
        "[train_rega_perturb]\n"
        f"  device={device}\n"
        f"  K={config.K}\n"
        f"  cells={n_cells}\n"
        f"  targets={n_targets}\n"
        f"  motifs={n_motifs}\n"
        f"  batch_size={config.batch_size}\n"
        f"  update_W={config.update_W}\n"
        f"  update_B={config.update_B}\n"
        f"  update_V={config.update_V}\n"
        f"  update_H={config.update_H}\n"
        f"  enforce_W_support={config.enforce_W_support}\n"
        f"  enforce_B_support={config.enforce_B_support}\n"
        f"  update_P_in_phase2={config.update_P_in_phase2}\n"
    )

    _maybe_enable_tf32(config, device)

    # Convert W and B to dense torch tensors.
    # X remains sparse and is gathered/densified only for mini-batches.
    W_blocks = [
        torch.tensor(W.toarray(), device=device, dtype=config.dtype)
        for W in block_data.W_blocks
    ]

    B_blocks = [
        torch.tensor(B.toarray(), device=device, dtype=config.dtype)
        for B in block_data.B_blocks
    ]

    # Dense support masks are only needed if the caller explicitly wants them.
    if config.enforce_W_support:
        W_masks = [
            torch.tensor((W.toarray() > 0).astype(np.float32), device=device, dtype=config.dtype)
            for W in block_data.W_blocks
        ]
    else:
        W_masks = [None for _ in block_data.W_blocks]

    if config.enforce_B_support:
        B_masks = [
            torch.tensor((B.toarray() > 0).astype(np.float32), device=device, dtype=config.dtype)
            for B in block_data.B_blocks
        ]
    else:
        B_masks = [None for _ in block_data.B_blocks]

    # Prepare X blocks: host-side sparse layout, dense mini-batch to device.
    X_prepared, fast_row_layout = _prepare_x_blocks(
        block_data.X_blocks, device, config.dtype, config.x_on_device,
        log1p=config.log1p_x,
    )
    logger.info(
        f"  x_fast_row_layout={fast_row_layout}  allow_tf32={config.allow_tf32}"
        f"  log1p_x={config.log1p_x}  eps={config.eps}  den_floor={config.den_floor}"
    )

    target_idx = torch.tensor(block_data.target_idx, device=device, dtype=torch.long)
    p_values = torch.tensor(p_init, device=device, dtype=config.dtype)

    gen_V = torch.Generator(device=device).manual_seed(config.V_seed)
    V = torch.rand(n_motifs, config.K, generator=gen_V, device=device, dtype=config.dtype)

    gen_H = torch.Generator(device=device).manual_seed(config.H_seed)
    H = torch.rand(config.K, n_targets, generator=gen_H, device=device, dtype=config.dtype)

    loss_log = output_dir / "perturb_loss_log.csv"
    with open(loss_log, "w", newline="") as f:
        csv.writer(f).writerow([
            "phase", "epoch", "batch_loss_mean", "time_sec",
            "W_min", "W_max", "W_fro",
            "B_min", "B_max", "B_fro",
            "V_min", "V_max", "V_fro",
            "H_min", "H_max", "H_fro",
            "p_min", "p_max", "p_fro",
        ])

    # Separate, unbiased full-dataset RRE log -- this is the convergence metric.
    rre_log = output_dir / "perturb_rre_log.csv"
    with open(rre_log, "w", newline="") as f:
        csv.writer(f).writerow(
            ["phase", "epoch", "stage", "rre", "sse", "xnorm",
             "rel_change", "plateaued"]
        )

    # Mutable holder for the previous RRE so we can report relative change.
    _rre_state = {"prev": None, "plateaued": False}

    def eval_full_rre(phase_name: str, epoch: int, stage: str) -> float:
        """Compute, print, and log the full-dataset RRE. Returns the RRE."""
        stats = _compute_full_rre(
            X_prepared, n_cells, config.batch_size,
            W_blocks, B_blocks, V, H, target_idx, p_values,
            device, config.dtype, fast_row_layout, config.pin_memory,
        )
        rre = stats["rre"]
        if not np.isfinite(rre):
            raise FloatingPointError(
                f"Full-dataset RRE is not finite ({rre}) at {phase_name} epoch {epoch}; "
                "check the input data (all-zero blocks?) and training stability."
            )
        prev = _rre_state["prev"]
        if prev is None:
            rel_change = float("nan")
            plateaued = False
        elif prev > 0:
            rel_change = abs(rre - prev) / prev
            plateaued = rel_change < schedule.conv_tol
        else:   # previous RRE exactly 0: perfect reconstruction
            rel_change = 0.0 if rre == 0 else float("inf")
            plateaued = rre == 0
        _rre_state["prev"] = rre
        _rre_state["plateaued"] = plateaued

        logger.info(
            f"  [full RRE] {phase_name} e{epoch:04d} ({stage}): "
            f"rre={rre:.6f}  sse={stats['sse']:.4e}  "
            f"rel_change={rel_change:.2e}"
            + ("  [plateaued]" if plateaued else "")
        )
        with open(rre_log, "a", newline="") as f:
            csv.writer(f).writerow(
                [phase_name, epoch, stage, f"{rre:.8f}",
                 f"{stats['sse']:.6e}", f"{stats['xnorm']:.6e}",
                 f"{rel_change:.6e}", int(plateaued)]
            )
        return rre

    # Optional: rescale the initialization to the data scale.
    if config.scale_init_to_data:
        with _no_grad_ctx(config):
            n_sample = max(1, min(config.scale_init_n_batches,
                                  int(np.ceil(n_cells / config.batch_size))))
            sample_cols = np.arange(min(n_cells, config.batch_size * n_sample))
            sample_idx = torch.as_tensor(sample_cols, device=device, dtype=torch.long)
            sample_X = _densify_batch(
                X_prepared, sample_cols, device, config.dtype,
                fast_row_layout=fast_row_layout, pin_memory=config.pin_memory,
            )
            x_norm0, y_norm0 = _reconstruction_scale(
                sample_X, W_blocks, B_blocks, V, H, target_idx, p_values, sample_idx
            )
            alpha = _scale_init_VH_(
                sample_X, W_blocks, B_blocks, V, H, target_idx, p_values,
                sample_idx, config.eps,
            )
            x_norm1, y_norm1 = _reconstruction_scale(
                sample_X, W_blocks, B_blocks, V, H, target_idx, p_values, sample_idx
            )
            logger.info(
                f"  [scale_init] alpha={alpha:.3e}  "
                f"recon/X before={ (y_norm0/x_norm0) if x_norm0>0 else float('nan'):.2e}  "
                f"after={ (y_norm1/x_norm1) if x_norm1>0 else float('nan'):.2e}"
            )
            del sample_X

    def run_phase(phase_name: str, min_epochs: int, max_epochs: int | None,
                  early_stop: bool, allow_P_update: bool) -> int:
        """Run one phase; returns the number of epochs trained."""
        n_batches = int(np.ceil(n_cells / config.batch_size))
        if not early_stop:
            max_epochs = min_epochs
        if early_stop:
            epochs_desc = (f">= {min_epochs}, until converged" if max_epochs is None
                           else f"{min_epochs}-{max_epochs}, until converged")
        else:
            epochs_desc = f"{max_epochs}"
        epoch_total = f"{max_epochs}" if max_epochs is not None else f"{min_epochs}+"

        logger.info(
            f"\n[{phase_name}] epochs={epochs_desc}, batches/epoch={n_batches}, "
            f"allow_P_update={allow_P_update}"
        )
        plateau_streak = 0

        # Baseline full-dataset RRE at the start of the phase (epoch 0).
        eval_full_rre(phase_name, 0, "phase_start")

        epoch_iter = _progress_iter(
            itertools.count(1) if max_epochs is None else range(1, max_epochs + 1),
            total=max_epochs,
            desc=phase_name,
            enabled=config.show_progress,
        )

        for epoch in epoch_iter:
            t0 = time.time()
            batch_losses: list[float] = []

            batches = list(
                _make_batches(
                    n_cells=n_cells,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle_cells,
                    rng=rng,
                )
            )

            batch_iter = _progress_iter(
                enumerate(batches, start=1),
                total=len(batches),
                desc=f"{phase_name} epoch {epoch}",
                enabled=config.show_progress,
            )

            for batch_counter, batch_idx_np in batch_iter:
                update_P_now = (
                    allow_P_update
                    and config.update_P_in_phase2
                    and (batch_counter % config.p_update_every == 0)
                )

                # Densify this mini-batch once and reuse it for all updates.
                batch_idx = torch.as_tensor(batch_idx_np, device=device, dtype=torch.long)
                Xb_blocks = _densify_batch(
                    X_prepared,
                    batch_idx_np,
                    device,
                    config.dtype,
                    fast_row_layout=fast_row_layout,
                    pin_memory=config.pin_memory,
                )

                _update_one_batch(
                    Xb_blocks=Xb_blocks,
                    W_blocks=W_blocks,
                    B_blocks=B_blocks,
                    W_masks=W_masks,
                    B_masks=B_masks,
                    V=V,
                    H=H,
                    target_idx=target_idx,
                    p_values=p_values,
                    batch_idx=batch_idx,
                    config=config,
                    device=device,
                    update_P=update_P_now,
                )

                if config.diag_every > 0 and (batch_counter % config.diag_every == 0):
                    _print_diagnostics(
                        f"{phase_name} e{epoch} b{batch_counter}",
                        Xb_blocks, W_blocks, B_blocks, V, H,
                        target_idx, p_values, batch_idx,
                    )

                if (config.rebalance_WB and config.rebalance_every > 0
                        and batch_counter % config.rebalance_every == 0):
                    if allow_P_update and config.rescale_P_by_target:
                        _rescale_p_by_target_(
                            H=H, p_values=p_values, target_idx=target_idx,
                            n_targets=n_targets, eps=config.eps,
                            p_min=config.p_min, p_max=config.p_max,
                        )
                    _rebalance_chain_(W_blocks, B_blocks, V, H, config.eps)

                if len(batch_losses) < config.eval_batches:
                    bl = _compute_batch_loss(
                        Xb_blocks=Xb_blocks,
                        W_blocks=W_blocks,
                        B_blocks=B_blocks,
                        V=V,
                        H=H,
                        target_idx=target_idx,
                        p_values=p_values,
                        batch_idx=batch_idx,
                        dtype=config.dtype,
                        device=device,
                    )
                    batch_losses.append(bl)

                if tqdm is not None and config.show_progress:
                    batch_iter.set_postfix(
                        {
                            "eval_loss": f"{np.mean(batch_losses):.3e}" if batch_losses else "NA",
                            "P": "on" if update_P_now else "off",
                        }
                    )

            if allow_P_update and config.rescale_P_by_target:
                _rescale_p_by_target_(
                    H=H,
                    p_values=p_values,
                    target_idx=target_idx,
                    n_targets=n_targets,
                    eps=config.eps,
                    p_min=config.p_min,
                    p_max=config.p_max,
                )

            # Periodic rescaling for numerical stability.
            if config.rebalance_WB and (config.update_W or config.update_B
                                        or config.update_V or config.update_H):
                _rebalance_chain_(W_blocks, B_blocks, V, H, config.eps)

            elapsed = time.time() - t0
            mean_loss = float(np.mean(batch_losses)) if batch_losses else np.nan

            if epoch % config.log_every_epoch == 0:
                msg = (
                    f"[{phase_name}][epoch {epoch:04d}/{epoch_total}] "
                    f"batch_loss_mean={mean_loss:.6e} time={elapsed:.2f}s\n"
                    f"    "
                    + _blocks_stats("W", W_blocks) + " "
                    + _blocks_stats("B", B_blocks) + " "
                    + _factor_stats("V", V) + " "
                    + _factor_stats("H", H) + " "
                    + _factor_stats("p", p_values)
                )
                logger.info(msg)

            with open(loss_log, "a", newline="") as f:
                Wmm = _minmaxfro(W_blocks)
                Bmm = _minmaxfro(B_blocks)
                Vmm = _minmaxfro(V)
                Hmm = _minmaxfro(H)
                pmm = _minmaxfro(p_values)
                csv.writer(f).writerow(
                    [phase_name, epoch, f"{mean_loss:.6e}", f"{elapsed:.3f}"]
                    + [f"{v:.6e}" for v in (*Wmm, *Bmm, *Vmm, *Hmm, *pmm)]
                )

            # Unbiased full-dataset RRE every `full_eval_every` epochs and on
            # the final epoch of the phase. This is the convergence metric;
            # only evaluations after `min_epochs` count towards stopping.
            if epoch % schedule.full_eval_every == 0 or epoch == max_epochs:
                eval_full_rre(phase_name, epoch, "epoch_end")
                if early_stop and epoch > min_epochs and _rre_state["plateaued"]:
                    plateau_streak += 1
                else:
                    plateau_streak = 0
                if early_stop and plateau_streak >= schedule.early_stop_patience:
                    logger.info(
                        f"[{phase_name}] converged at epoch {epoch} "
                        f"({plateau_streak} consecutive evals with "
                        f"rel_change < {schedule.conv_tol:g}); stopping."
                    )
                    return epoch

        if early_stop:
            logger.info(
                f"[{phase_name}] reached {max_epochs} epochs "
                f"without meeting the convergence criterion."
            )
        return max_epochs

    n_phase1 = run_phase("phase1_freezeP", schedule.phase1_epochs, schedule.phase1_epochs,
                         early_stop=False, allow_P_update=False)

    n_phase2 = run_phase("phase2_P_update", schedule.phase2_min_epochs,
                         schedule.phase2_max_epochs, early_stop=True, allow_P_update=True)
    epochs_trained = {"phase1": int(n_phase1), "phase2": int(n_phase2),
                      "total": int(n_phase1 + n_phase2)}
    logger.info(f"[train] epochs trained: phase 1 = {n_phase1}, phase 2 = {n_phase2}, "
                f"total = {n_phase1 + n_phase2}")

    final_ckpt = output_dir / "final_perturb.pt"
    save_perturb_checkpoint(
        final_ckpt,
        W_blocks=W_blocks,
        B_blocks=B_blocks,
        V=V,
        H=H,
        p_values=p_values,
        block_data=block_data,
        config=config,
        epochs_trained=epochs_trained,
    )

    logger.info(f"[train_rega_perturb done] saved {final_ckpt}")

    if config.make_plot:
        plot_training_curves(
            loss_log, rre_log, output_dir / "perturb_training_curves.png"
        )

    return final_ckpt


# ============================================================
# Checkpointing
# ============================================================

def plot_training_curves(
    loss_log: Path,
    rre_log: Path,
    out_png: Path,
) -> Path | None:
    """Render batch-loss and full-dataset RRE curves to a PNG.

    Reads the two CSV logs written during training and produces a two-panel
    figure: (left) per-epoch batch loss, (right) full-dataset RRE, both with
    phase boundaries marked. Returns the PNG path, or None if plotting is
    unavailable. Never raises -- a plotting failure must not abort training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.info(f"[plot] matplotlib unavailable ({exc}); skipping plot.")
        return None

    try:
        # --- read loss log ---
        with open(loss_log, newline="") as f:
            loss_rows = list(csv.DictReader(f))
        # global x = cumulative epoch index across phases (in log order)
        loss_x, loss_y, loss_phase = [], [], []
        for i, r in enumerate(loss_rows):
            try:
                loss_y.append(float(r["batch_loss_mean"]))
            except (ValueError, KeyError):
                loss_y.append(np.nan)
            loss_x.append(i)
            loss_phase.append(r.get("phase", ""))

        # --- read RRE log (epoch_end rows only for the main curve) ---
        rre_x, rre_y, rre_phase = [], [], []
        if rre_log.exists():
            with open(rre_log, newline="") as f:
                rre_rows = list(csv.DictReader(f))
            for i, r in enumerate(rre_rows):
                try:
                    rre_y.append(float(r["rre"]))
                except (ValueError, KeyError):
                    rre_y.append(np.nan)
                rre_x.append(i)
                rre_phase.append(r.get("phase", ""))

        # phase color map
        phases = []
        for p in loss_phase + rre_phase:
            if p and p not in phases:
                phases.append(p)
        cmap = {p: c for p, c in zip(
            phases, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )}

        def _phase_spans(phase_seq):
            spans = []
            if not phase_seq:
                return spans
            start = 0
            cur = phase_seq[0]
            for i, p in enumerate(phase_seq[1:], start=1):
                if p != cur:
                    spans.append((start, i - 1, cur))
                    start, cur = i, p
            spans.append((start, len(phase_seq) - 1, cur))
            return spans

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # left: batch loss
        ax = axes[0]
        ax.plot(loss_x, loss_y, color="#444", lw=1.2, zorder=3)
        for s, e, p in _phase_spans(loss_phase):
            ax.axvspan(s - 0.5, e + 0.5, color=cmap.get(p, "#ccc"),
                       alpha=0.12, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("logged epoch (across phases)")
        ax.set_ylabel("batch loss mean (log scale)")
        ax.set_title("Training batch loss")
        ax.grid(True, which="both", alpha=0.3)

        # right: full RRE
        ax = axes[1]
        if rre_x:
            ax.plot(rre_x, rre_y, marker="o", ms=4, color="#444",
                    lw=1.4, zorder=3)
            for s, e, p in _phase_spans(rre_phase):
                ax.axvspan(s - 0.5, e + 0.5, color=cmap.get(p, "#ccc"),
                           alpha=0.12, zorder=0, label=p)
            # de-duplicate legend labels
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
            ax.set_ylabel("full-dataset RRE")
            ax.set_title("Full-dataset reconstruction error (unbiased)")
        else:
            ax.text(0.5, 0.5, "no RRE log",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("full-eval index")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        logger.info(f"[plot] saved training curves to {out_png}")
        return out_png
    except Exception as exc:  # pragma: no cover
        logger.info(f"[plot] failed to render curves ({exc}); skipping.")
        return None


def save_perturb_checkpoint(
    path: Path,
    W_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
    V: torch.Tensor,
    H: torch.Tensor,
    p_values: torch.Tensor,
    block_data: PerturbBlockData,
    config: PerturbTrainingConfig,
    epochs_trained: dict[str, int] | None = None,
) -> None:
    """Save REGA-Perturb checkpoint."""
    ckpt: dict[str, Any] = {
        "model": "REGA-Perturb",
        "W_blocks": [w.detach().cpu() for w in W_blocks],
        "B_blocks": [b.detach().cpu() for b in B_blocks],
        "V": V.detach().cpu(),
        "H": H.detach().cpu(),
        "p_values": p_values.detach().cpu(),
        "target_idx": block_data.target_idx,
        "perturb_targets": block_data.perturb_targets,
        "chr_names": block_data.chr_names,
        "genes_blocks": block_data.genes_blocks,
        "RE_blocks": block_data.RE_blocks,
        "cells": block_data.cells,
        "motifs": block_data.motifs,
        "config": asdict(config),
    }
    if epochs_trained is not None:
        ckpt["epochs_trained"] = dict(epochs_trained)
    torch.save(ckpt, path)
    logger.info(f"[save_perturb_checkpoint] {path}")


def load_perturb_checkpoint(path: Path) -> dict[str, Any]:
    """Load REGA-Perturb checkpoint.

    Note: this uses weights_only=False because the checkpoint stores numpy
    arrays and a config dict. Only load checkpoints you trust.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")

    logger.info(f"[load_perturb_checkpoint] loaded {path}")
    return ckpt
