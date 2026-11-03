import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """Small synthetic Perturb-seq dataset with reference tables (see helpers/synthetic.py)."""
    from helpers.synthetic import make_tiny_dataset

    return make_tiny_dataset(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tiny_model(tiny_dataset, tmp_path_factory):
    """A small model trained on the synthetic dataset (CPU, a few epochs)."""
    from helpers.schedule import short_schedule

    from rega_perturb.model.config import make_training_config
    from rega_perturb.model.train import load_perturb_blocks, train_rega_perturb
    from rega_perturb.preprocess.blocks import build_blocks
    from rega_perturb.preprocess.graph import build_graph
    from rega_perturb.preprocess.qc import run_qc

    d = tmp_path_factory.mktemp("tiny_model")
    qc = run_qc(tiny_dataset["raw_h5ad"], d / "qc", "t", min_umi=10, max_umi=1e7, min_genes=5,
                max_genes=1e6, min_cells_per_gene=5)
    graph = build_graph(qc["gene_metadata"], tiny_dataset["re_tg"], tiny_dataset["peak_motif"],
                        d / "graph")
    blocks = build_blocks(graph, tiny_dataset["raw_h5ad"], qc["cell_metadata"],
                          qc["gene_metadata"], d / "blocks")
    cfg = make_training_config(None, dict(K=3, batch_size=64, p_update_every=1, show_progress=False,
                                          make_plot=False))
    data, p_init = load_perturb_blocks(blocks)
    ckpt = train_rega_perturb(data, p_init, d / "model", cfg, device="cpu",
                              _schedule=short_schedule())
    return dict(dir=d, graph=graph, checkpoint=ckpt, motif_tf=tiny_dataset["motif_tf"])
