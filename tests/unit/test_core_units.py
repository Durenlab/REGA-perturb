from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rega_perturb.analysis.motif_tf import build_motif_tf_matrix
from rega_perturb.cli import main
from rega_perturb.model.config import configurable_fields, make_training_config
from rega_perturb.preprocess.reference import RE_TG_COLUMNS, build_re_tg, name_peaks


def test_motif_tf_matrix(tiny_dataset):
    motifs = [f"MOTIF{k}" for k in range(8)]
    genes = ["TFA", "TFB", "GENE1"]
    C, tfs = build_motif_tf_matrix(tiny_dataset["motif_tf"], motifs, genes)
    assert list(tfs) == ["TFA", "TFB"]          # TFC not modeled; NOT_MODELED not a gene
    assert C.shape == (8, 2) and C.dtype == np.float32
    assert C[0, 0] == C[1, 0] == C[2, 1] == 1 and C.sum() == 3   # MOTIF_X not a model motif


def test_motif_tf_matrix_no_match(tiny_dataset):
    with pytest.raises(ValueError):
        build_motif_tf_matrix(tiny_dataset["motif_tf"], ["MOTIF0"], ["XYZ"])


def test_training_config(tmp_path):
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text("train:\n  K: 7\n  batch_size: 32\n  seed: 5\n")
    cfg = make_training_config(cfg_file, {"batch_size": 64, "seed": None})
    assert (cfg.K, cfg.batch_size, cfg.seed) == (7, 64, 5)
    assert "dtype" not in configurable_fields()
    with pytest.raises(ValueError):
        make_training_config(None, {"not_a_field": 1})


def test_name_peaks(tmp_path):
    bed = tmp_path / "p.bed"
    bed.write_text("chr1\t10\t20\nchr1\t10\t20\nchrUn_x\t1\t5\nchr2\t5\t9\n")
    out = name_peaks(bed, tmp_path / "named.bed")
    df = pd.read_csv(out, sep="\t", header=None)
    assert df[3].tolist() == ["chr1_10_20", "chr2_5_9"]


@pytest.mark.skipif(shutil.which("bedtools") is None, reason="bedtools not installed")
def test_build_re_tg(tmp_path):
    tss = tmp_path / "tss.txt"
    tss.write_text("chr1\t0\t1000\tG1\t500\t+\nchr1\t5000\t6000\tG2\t5500\t-\nchrUn\t0\t9\tG3\t5\t+\n")
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\tchr1_100_200\nchr1\t5100\t5200\tchr1_5100_5200\n")
    out = build_re_tg(tss, peaks, tmp_path / "re_tg.tsv")
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == RE_TG_COLUMNS
    assert df["gene_name"].tolist() == ["G1", "G2"]
    assert df["peak_name"].tolist() == ["chr1_100_200", "chr1_5100_5200"]


def test_qc_mito_fraction(tiny_dataset, tmp_path):
    from rega_perturb.preprocess.qc import run_qc

    out = run_qc(tiny_dataset["raw_h5ad"], tmp_path, "t", min_umi=10, max_umi=1e7,
                 min_genes=5, max_genes=1e6, max_mt=0.2, min_cells_per_gene=5)
    cells = pd.read_csv(out["cell_metadata"], index_col=0)
    assert cells["mitopercent"].max() <= 0.2 and len(cells) > 0
    genes = pd.read_csv(out["gene_metadata"], index_col=0)
    assert "MT-CO1" not in set(genes["gene_name"])
    assert (tmp_path / "qc.run.yaml").exists()


def test_cli_help():
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    for cmd in ["qc", "peaks", "motif", "retg", "build-graph", "build-blocks",
                "train", "assemble"]:
        assert cmd in res.output


def test_cli_missing_file_exit_code(tmp_path):
    runner = CliRunner()
    res = runner.invoke(main, ["build-graph", "--gene-metadata", str(tmp_path / "missing.csv"),
                               "--re-tg", "x", "--peak-motif", "y", "-o", str(tmp_path)])
    assert res.exit_code == 2


def test_normalized_input_path(tiny_dataset, tmp_path):
    """Already-normalized input: no cell filtering, blocks keep the provided values."""
    import anndata as ad
    import scipy.sparse as sp

    from rega_perturb.preprocess.blocks import build_blocks
    from rega_perturb.preprocess.graph import build_graph
    from rega_perturb.preprocess.qc import run_qc

    raw = ad.read_h5ad(tiny_dataset["raw_h5ad"])
    counts = raw.X.toarray()
    norm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    norm_h5ad = tmp_path / "norm.h5ad"
    ad.AnnData(X=sp.csr_matrix(norm), obs=raw.obs, var=raw.var).write_h5ad(norm_h5ad)

    with pytest.raises(ValueError):
        run_qc(norm_h5ad, tmp_path / "qc_bad", "n")  # counts mode rejects non-integer data
    out = run_qc(norm_h5ad, tmp_path / "qc", "n", input_type="normalized", min_cells_per_gene=5)
    cells = pd.read_csv(out["cell_metadata"], index_col=0)
    assert len(cells) == raw.n_obs and "library_size" not in cells.columns

    graph = build_graph(out["gene_metadata"], tiny_dataset["re_tg"], tiny_dataset["peak_motif"],
                        tmp_path / "graph")
    blocks = build_blocks(graph, norm_h5ad, out["cell_metadata"], out["gene_metadata"],
                          tmp_path / "blocks", expr_mode="as_is", chunk_size=50)
    z = np.load(blocks / "chr1.npz")
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    gpos = {g: i for i, g in enumerate(raw.var["gene_name"].astype(str))}
    expect = norm[:, [gpos[g] for g in z["genes"]]].T
    assert np.array_equal(X.toarray(), expect)


def test_log_normalization_modes(tiny_dataset, tmp_path):
    """Default blocks use log-CP10K; log-CPM is optional."""
    import scipy.sparse as sp

    from rega_perturb.preprocess.blocks import build_blocks, transform_expression
    from rega_perturb.preprocess.graph import build_graph
    from rega_perturb.preprocess.qc import run_qc

    X = sp.csr_matrix(np.array([[1.0, 3.0], [0.0, 2.0]], dtype=np.float32))
    lib = np.array([4.0, 2.0])
    Y = transform_expression(X, lib, "logcp10k", 1e4).toarray()
    assert np.allclose(Y, np.log1p(X.toarray() / lib[:, None] * 1e4))

    out = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "qc", "t", min_umi=10, max_umi=1e7,
                 min_genes=5, max_genes=1e6, min_cells_per_gene=5)
    graph = build_graph(out["gene_metadata"], tiny_dataset["re_tg"], tiny_dataset["peak_motif"],
                        tmp_path / "graph")
    blocks = {}
    for mode in ("logcp10k", "logcpm"):
        d = build_blocks(graph, tiny_dataset["raw_h5ad"], out["cell_metadata"],
                         out["gene_metadata"], tmp_path / mode, expr_mode=mode)
        blocks[mode] = np.load(d / "chr1.npz")["X_data"]
    a = np.expm1(blocks["logcp10k"].astype(np.float64))
    b = np.expm1(blocks["logcpm"].astype(np.float64))
    assert np.allclose(b / a, 100.0, rtol=1e-4)
    rec = (tmp_path / "logcp10k" / "build_blocks.run.yaml").read_text()
    assert "expr_mode: logcp10k" in rec


def test_tss_reference_annotation(tiny_dataset, tmp_path):
    """Data without gene coordinates: coordinates come from a TSS reference table."""
    import anndata as ad

    from rega_perturb.preprocess.qc import run_qc

    kw = dict(min_umi=10, max_umi=1e7, min_genes=5, max_genes=1e6, min_cells_per_gene=5)
    raw = ad.read_h5ad(tiny_dataset["raw_h5ad"])
    var = raw.var
    tss = np.where(var["strand"] == "+", var["start"], var["end"])
    ref = pd.DataFrame({"chr": var["chr"], "tss": tss, "gene": var["gene_name"], "strand": var["strand"]})
    dropped = "CHR1G5"
    ref_file = tmp_path / "tss.txt"
    ref[ref["gene"] != dropped].to_csv(ref_file, sep="\t", header=False, index=False)

    # reference ignored when the data have coordinates: outputs unchanged
    base = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "base", "t", **kw)
    same = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "same", "t", tss_reference=ref_file, **kw)
    for key in ("tss", "gene_metadata", "cell_metadata", "summary"):
        assert same[key].read_bytes() == base[key].read_bytes(), key

    # data without coordinates: error without reference, annotation with it
    bare_h5ad = tmp_path / "bare.h5ad"
    ad.AnnData(X=raw.X, obs=raw.obs, var=var[["gene_name"]]).write_h5ad(bare_h5ad)
    with pytest.raises(ValueError):
        run_qc(bare_h5ad, tmp_path / "bad", "t", **kw)
    out = run_qc(bare_h5ad, tmp_path / "ref", "t", tss_reference=ref_file, **kw)

    cols = ["chr", "region_start", "region_end", "gene", "TSS", "strand"]
    t_base = pd.read_csv(base["tss"], sep="\t", header=None, names=cols)
    t_ref = pd.read_csv(out["tss"], sep="\t", header=None, names=cols)
    expect = t_base[t_base["gene"] != dropped].reset_index(drop=True)
    pd.testing.assert_frame_equal(t_ref.reset_index(drop=True), expect)
    genes = pd.read_csv(out["gene_metadata"], index_col=0)["gene_name"]
    base_genes = pd.read_csv(base["gene_metadata"], index_col=0)["gene_name"]
    assert dropped not in set(genes) and set(genes) == set(base_genes) - {dropped}


# Phase 2: minimum 3 epochs, evaluations every epoch, 2 converged evaluations after the
# minimum needed -> stops at epoch 5 when always converged; never converged -> cap 6.
@pytest.mark.parametrize("conv_tol, n_expected", [(1.0, 5), (0.0, 6)])
def test_phase2_early_stopping(tiny_model, conv_tol, n_expected, tmp_path):
    from helpers.schedule import short_schedule

    from rega_perturb.model.train import load_perturb_blocks, train_rega_perturb

    cfg = make_training_config(None, dict(K=3, batch_size=64, show_progress=False,
                                          make_plot=False))
    schedule = short_schedule(phase1_epochs=1, phase2_min_epochs=3, phase2_max_epochs=6,
                              full_eval_every=1, conv_tol=conv_tol, early_stop_patience=2)
    data, p_init = load_perturb_blocks(tiny_model["dir"] / "blocks")
    train_rega_perturb(data, p_init, tmp_path, cfg, device="cpu", _schedule=schedule)
    rre = pd.read_csv(tmp_path / "perturb_rre_log.csv")
    p2 = rre[(rre["phase"] == "phase2_P_update") & (rre["stage"] == "epoch_end")]
    assert p2["epoch"].max() == n_expected


@pytest.mark.parametrize("bad, match", [
    (dict(p_update_every=0), "p_update_every"),
    (dict(batch_size=0), "batch_size"),
])
def test_training_config_bounds(tiny_model, tmp_path, bad, match):
    from rega_perturb.model.train import load_perturb_blocks, train_rega_perturb

    cfg = make_training_config(None, bad)
    data, p_init = load_perturb_blocks(tiny_model["dir"] / "blocks")
    with pytest.raises(ValueError, match=match):
        train_rega_perturb(data, p_init, tmp_path, cfg, device="cpu")


def test_fixed_training_schedule():
    from rega_perturb.model.train import _TrainingSchedule

    s = _TrainingSchedule()
    assert (s.phase1_epochs, s.phase2_min_epochs, s.phase2_max_epochs) == (15, 60, None)
    assert (s.full_eval_every, s.conv_tol, s.early_stop_patience) == (5, 1e-4, 2)
    for key in ("phase1_epochs", "phase2_min_epochs", "phase2_max_epochs", "full_eval_every",
                "conv_tol", "early_stop_patience"):
        with pytest.raises(ValueError, match="Unknown training parameter"):
            make_training_config(None, {key: 1})


def _train_short(tiny_model, tmp_path, **schedule_kw):
    from helpers.schedule import short_schedule

    from rega_perturb.model.train import load_perturb_blocks, train_rega_perturb

    cfg = make_training_config(None, dict(K=3, batch_size=64, show_progress=True,
                                          make_plot=False))
    data, p_init = load_perturb_blocks(tiny_model["dir"] / "blocks")
    return train_rega_perturb(data, p_init, tmp_path, cfg, device="cpu",
                              _schedule=short_schedule(**schedule_kw))


def test_phase2_no_upper_limit(tiny_model, tmp_path):
    """Phase 2 without an epoch limit (as in the fixed schedule) stops once converged."""
    import torch

    ckpt = _train_short(tiny_model, tmp_path, phase1_epochs=1, phase2_min_epochs=3,
                        phase2_max_epochs=None, full_eval_every=1, conv_tol=1.0,
                        early_stop_patience=2)
    rre = pd.read_csv(tmp_path / "perturb_rre_log.csv")
    p2 = rre[(rre["phase"] == "phase2_P_update") & (rre["stage"] == "epoch_end")]
    assert p2["epoch"].max() == 5
    trained = torch.load(ckpt, map_location="cpu", weights_only=False)["epochs_trained"]
    assert trained == {"phase1": 1, "phase2": 5, "total": 6}


def test_rre_not_finite_raises(tiny_model, tmp_path, monkeypatch):
    """A non-finite full-data RRE stops training with an error instead of looping forever."""
    import rega_perturb.model.train as tr

    monkeypatch.setattr(tr, "_compute_full_rre",
                        lambda *a, **k: {"rre": float("nan"), "sse": 0.0, "xnorm": 0.0})
    with pytest.raises(FloatingPointError, match="not finite"):
        _train_short(tiny_model, tmp_path, phase1_epochs=1, phase2_min_epochs=3,
                     phase2_max_epochs=None, full_eval_every=1)


def test_rre_zero_counts_as_converged(tiny_model, tmp_path, monkeypatch):
    """An RRE of exactly 0 counts as converged (no upper limit, stops at min + patience)."""
    import rega_perturb.model.train as tr

    monkeypatch.setattr(tr, "_compute_full_rre",
                        lambda *a, **k: {"rre": 0.0, "sse": 0.0, "xnorm": 1.0})
    _train_short(tiny_model, tmp_path, phase1_epochs=1, phase2_min_epochs=3,
                 phase2_max_epochs=None, full_eval_every=1, conv_tol=1e-4)
    rre = pd.read_csv(tmp_path / "perturb_rre_log.csv")
    p2 = rre[(rre["phase"] == "phase2_P_update") & (rre["stage"] == "epoch_end")]
    assert p2["epoch"].max() == 5


def test_qc_max_genes_optional(tiny_dataset, tmp_path):
    """--max-genes has no default: without it no upper limit is applied."""
    import pandas as pd

    from rega_perturb.preprocess.qc import run_qc

    kw = dict(min_umi=10, max_umi=1e7, min_genes=5, min_cells_per_gene=5)
    none = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "none", "t", **kw)
    wide = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "wide", "t", max_genes=1e9, **kw)
    assert none["cell_metadata"].read_bytes() == wide["cell_metadata"].read_bytes()
    n_genes = pd.read_csv(none["cell_metadata"], index_col=0)["n_genes_by_counts"]
    cap = float(n_genes.median())
    capped = run_qc(tiny_dataset["raw_h5ad"], tmp_path / "cap", "t", max_genes=cap, **kw)
    kept = pd.read_csv(capped["cell_metadata"], index_col=0)["n_genes_by_counts"]
    assert (kept <= cap).all() and len(kept) < len(n_genes)
    assert "max_genes: null" in (tmp_path / "none" / "qc.run.yaml").read_text()


def test_cli_max_genes_has_no_default():
    from click.testing import CliRunner

    from rega_perturb.cli import main

    out = CliRunner().invoke(main, ["qc", "--help"]).output
    assert "--max-genes" in out and "no upper limit" in out
