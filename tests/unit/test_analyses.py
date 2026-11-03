"""Downstream analyses checked against direct (naive) computations on a small model."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rega_perturb.analysis.diff_network import (
    delta_network,
    network_factors,
    rewiring_scores,
    specific_network,
)
from rega_perturb.analysis.model_io import detect_control, load_trained_model
from rega_perturb.analysis.motif_tf import build_motif_tf_matrix
from rega_perturb.analysis.tf_activity import compute_tf_activity_changes
from rega_perturb.analysis.tf_tg_scores import compute_tf_tg_scores, tf_tg_long_table
from rega_perturb.cli import main


@pytest.fixture(scope="module")
def model(tiny_model):
    return load_trained_model(tiny_model["checkpoint"], tiny_model["graph"])


@pytest.fixture(scope="module")
def C(model, tiny_model):
    return build_motif_tf_matrix(tiny_model["motif_tf"], model.motifs, model.genes)


def test_model_alignment(model):
    assert model.W.shape == (len(model.genes), len(model.res))
    assert model.B.shape == (len(model.res), len(model.motifs))
    assert model.W_prior.shape == model.W.shape
    # learned weights stay on the prior support
    assert not np.any((model.W.toarray() != 0) & (model.W_prior.toarray() == 0))
    assert detect_control(model.targets) == "non-targeting"


def test_influence_score(model, C, tiny_model):
    Cm, tfs = C
    S, genes, tfs2 = compute_tf_tg_scores(model, tiny_model["motif_tf"], "influence")
    W, B, V = model.W.toarray().astype(np.float64), model.B.astype(np.float64), model.V.astype(np.float64)
    expect = W @ B @ V @ V.T @ Cm
    assert list(tfs2) == list(tfs) and S.shape == expect.shape
    assert np.allclose(S, expect, rtol=1e-5, atol=1e-6 * np.abs(expect).max())
    assert (S >= 0).all()


def test_connectivity_score(model, C, tiny_model):
    Cm, _ = C
    S, _, _ = compute_tf_tg_scores(model, tiny_model["motif_tf"], "connectivity")
    W, B, V = model.W.toarray().astype(np.float64), model.B.astype(np.float64), model.V.astype(np.float64)
    BV = B @ V
    WBV = W @ BV
    prior = model.W_prior.toarray() > 0
    motif_hit = (model.B_prior.toarray() @ Cm) > 0                 # RE x TF
    expect = np.zeros_like(S)
    for g in range(S.shape[0]):
        for r in np.flatnonzero(prior[g]):
            link = WBV[g] @ BV[r]
            expect[g] += link * (BV[r] @ V.T @ Cm) * motif_hit[r]
    assert np.allclose(S, expect, rtol=1e-5, atol=1e-8 * max(np.abs(expect).max(), 1))


def test_long_table(model, tiny_model):
    S, genes, tfs = compute_tf_tg_scores(model, tiny_model["motif_tf"], "influence")
    df = tf_tg_long_table(S, genes, tfs)
    assert list(df.columns) == ["TF", "TG", "score"] and len(df) == S.size
    top = tf_tg_long_table(S, genes, tfs, top_per_tf=3)
    assert top.groupby("TF").size().max() == 3
    best = top.groupby("TF")["score"].max()
    for t, v in best.items():
        assert np.isclose(v, S[:, list(tfs).index(t)].max())


def test_tf_activity_changes(model, C, tiny_model):
    Cm, tfs = C
    log2fc, act = compute_tf_activity_changes(model, tiny_model["motif_tf"])
    U = Cm.T.astype(np.float64) @ model.V.astype(np.float64) @ model.H.astype(np.float64)
    c = list(model.targets).index("non-targeting")
    assert np.allclose(act.to_numpy(), U.T)
    assert np.allclose(log2fc.to_numpy(), np.log2(U / U[:, [c]]).T, equal_nan=True)
    assert np.allclose(log2fc.loc["non-targeting"].to_numpy(), 0)
    assert list(log2fc.columns) == list(tfs) and list(log2fc.index) == list(model.targets)


def test_diff_network(model, tiny_model):
    f = network_factors(model, tiny_model["motif_tf"])
    t = "TFA"
    d = delta_network(f, t)
    assert np.allclose(d, specific_network(f, t) - specific_network(f, f.control), atol=1e-5)
    rrs, nrs, targets = rewiring_scores(f)
    assert "non-targeting" not in set(targets) and rrs.shape == (len(f.genes), len(targets))
    j = list(targets).index(t)
    assert np.allclose(rrs[:, j], np.abs(d).sum(axis=1), rtol=1e-5)
    assert np.allclose(nrs[:, j], d.sum(axis=1), rtol=1e-4, atol=1e-6 * np.abs(d).sum(1).max())
    assert (rrs >= np.abs(nrs) - 1e-5).all()


def test_cli_analyses(tiny_model, tmp_path):
    common = ["--checkpoint", str(tiny_model["checkpoint"]), "--graph-h5ad", str(tiny_model["graph"]),
              "--motif-tf", str(tiny_model["motif_tf"])]
    runner = CliRunner()

    res = runner.invoke(main, ["tf-tg", *common, "-o", str(tmp_path / "tftg")])
    assert res.exit_code == 0, res.output
    for s in ("influence", "connectivity"):
        df = pd.read_csv(tmp_path / "tftg" / f"tf_tg_{s}.tsv.gz", sep="\t")
        assert list(df.columns) == ["TF", "TG", "score"]

    res = runner.invoke(main, ["tf-activity", *common, "-o", str(tmp_path / "act"), "--threshold", "0.25"])
    assert res.exit_code == 0, res.output
    wide = pd.read_csv(tmp_path / "act" / "tf_activity_log2fc.tsv", sep="\t", index_col=0)
    long = pd.read_csv(tmp_path / "act" / "tf_activity_changes.tsv.gz", sep="\t")
    assert list(long.columns) == ["target", "TF", "activity_target", "activity_control", "log2fc", "changed"]
    assert len(long) == wide.size

    res = runner.invoke(main, ["diff-network", *common, "-o", str(tmp_path / "dn"), "--target", "TFA",
                               "--target", "TFB", "--min-abs", "0"])
    assert res.exit_code == 0, res.output
    with h5py.File(tmp_path / "dn" / "rewiring_scores.h5") as h5:
        assert h5["RRS"].shape == h5["NRS"].shape == (len(h5["genes"]), len(h5["targets"]))
    df = pd.read_csv(tmp_path / "dn" / "delta_network.TFA.tsv.gz", sep="\t")
    assert list(df.columns) == ["target", "TF", "TG", "delta_w"]

    res = runner.invoke(main, ["diff-network", *common, "-o", str(tmp_path / "dn2"), "--all-targets",
                               "--format", "h5", "--kind", "specific", "--no-scores"])
    assert res.exit_code == 0, res.output
    with h5py.File(tmp_path / "dn2" / "specific_networks.h5") as h5:
        assert len(h5["networks"]) == 4

    res = runner.invoke(main, ["diff-network", *common, "-o", str(tmp_path / "dn3"), "--target", "NOPE"])
    assert res.exit_code == 1


def test_assemble_peak_activity_optional(tiny_model, tmp_path):
    import anndata as ad

    from rega_perturb.results import collect_results

    a = ad.read_h5ad(collect_results(tiny_model["graph"], tiny_model["checkpoint"],
                                     tmp_path / "default.h5ad"))
    b = ad.read_h5ad(collect_results(tiny_model["graph"], tiny_model["checkpoint"],
                                     tmp_path / "full.h5ad", peak_activity=True))
    assert "Peak_Activity" not in a.varm
    assert b.varm["Peak_Activity"].shape[0] == b.n_vars
    pd.testing.assert_frame_equal(a.var, b.var)


def test_delta_network_unknown_target(tiny_model):
    from rega_perturb.analysis.model_io import load_trained_model

    f = network_factors(load_trained_model(tiny_model["checkpoint"], tiny_model["graph"]),
                        tiny_model["motif_tf"])
    with pytest.raises(KeyError, match="not in model"):
        delta_network(f, "NOT_A_TARGET")


def test_assemble_tf_tg_scores(tiny_model, tmp_path):
    """assemble --motif-tf stores both TF-TG scores, identical to compute_tf_tg_scores."""
    import anndata as ad
    from click.testing import CliRunner

    from rega_perturb.analysis.model_io import load_trained_model
    from rega_perturb.analysis.tf_tg_scores import SCORES, compute_tf_tg_scores
    from rega_perturb.cli import main

    out = tmp_path / "res.h5ad"
    res = CliRunner().invoke(main, ["assemble", "--graph-h5ad", str(tiny_model["graph"]),
                                    "--checkpoint", str(tiny_model["checkpoint"]),
                                    "-o", str(out), "--motif-tf", str(tiny_model["motif_tf"])])
    assert res.exit_code == 0, res.output
    a = ad.read_h5ad(out)
    model = load_trained_model(tiny_model["checkpoint"], tiny_model["graph"])
    for name in SCORES:
        S, genes, tfs = compute_tf_tg_scores(model, tiny_model["motif_tf"], name)
        assert np.array_equal(a.obs_names.values, genes)
        assert np.array_equal(np.asarray(a.uns["TF_names"]).astype(str), tfs)
        assert np.array_equal(a.obsm[f"TF_TG_{name}"], S.astype(np.float32)), name
    assert "motif_tf" in (tmp_path / "assemble.run.yaml").read_text()

    plain = ad.read_h5ad(collect_results_default(tiny_model, tmp_path))
    assert not any(k.startswith("TF_TG_") for k in plain.obsm) and "TF_names" not in plain.uns


def collect_results_default(tiny_model, tmp_path):
    from rega_perturb.results import collect_results

    return collect_results(tiny_model["graph"], tiny_model["checkpoint"], tmp_path / "plain" / "r.h5ad")
