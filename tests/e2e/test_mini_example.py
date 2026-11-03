"""End-to-end run of examples/k562_essential/run_mini.sh on the bundled mini data (slow; needs bedtools).

REGA_PERTURB_MINI_DATA can point to another copy of the mini example files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = os.environ.get("REGA_PERTURB_MINI_DATA", str(ROOT / "examples/k562_essential/mini"))
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (Path(DATA) / "k562_essential_mini_raw.h5ad").exists(),
                       reason="mini example data not found"),
    pytest.mark.skipif(shutil.which("bedtools") is None, reason="bedtools not installed"),
]


def test_mini_example(tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    exe = shim / "rega-perturb"
    exe.write_text(f'#!/bin/bash\nPYTHONPATH="{ROOT / "src"}" exec "{sys.executable}" -m rega_perturb.cli "$@"\n')
    exe.chmod(0o755)
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}")
    res = subprocess.run(["bash", str(ROOT / "examples/k562_essential/run_mini.sh"), DATA, str(out)],
                         env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]

    for step in ["qc/qc", "graph/build_graph", "blocks/build_blocks", "model/train", "model/assemble",
                 "tf_tg/tf_tg", "tf_activity/tf_activity", "diff_network/diff_network"]:
        assert (out / f"{step}.run.yaml").exists(), step
    rre = pd.read_csv(out / "model/perturb_rre_log.csv")
    assert rre["rre"].iloc[-1] < rre["rre"].iloc[0]
    act = pd.read_csv(out / "tf_activity/tf_activity_log2fc.tsv", sep="\t", index_col=0)
    assert act.shape[1] > 0 and act.notna().all().all()
    assert (act.loc["non-targeting"] == 0).all()
    res = ad.read_h5ad(out / "model/rega_perturb_result.h5ad")
    for s in ("influence", "connectivity"):
        df = pd.read_csv(out / f"tf_tg/tf_tg_{s}.tsv.gz", sep="\t")
        assert list(df.columns) == ["TF", "TG", "score"] and (df["score"] >= 0).all()
        assert res.obsm[f"TF_TG_{s}"].shape == (res.n_obs, df["TF"].nunique())
    with h5py.File(out / "diff_network/rewiring_scores.h5") as h5:
        assert h5["RRS"].shape[1] == len(act) - 1
    with h5py.File(out / "diff_network/delta_networks.h5") as h5:
        assert len(h5["networks"]) == len(act) - 1
