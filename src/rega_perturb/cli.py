"""Command-line interface: ``rega-perturb <command> [options]``."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
from pathlib import Path

import click

from rega_perturb._version import __version__
from rega_perturb.utils import get_logger, setup_logging, write_run_record

logger = get_logger("cli")


@contextlib.contextmanager
def _handle_errors(debug: bool):
    """Exit with code 2 for missing files and 1 for other errors (re-raise with --debug)."""
    try:
        yield
    except FileNotFoundError as exc:
        if debug:
            raise
        logger.error(str(exc))
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        if debug:
            raise
        logger.error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)


@click.group()
@click.version_option(__version__, prog_name="rega-perturb")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.option("--log-file", type=click.Path(dir_okay=False), default=None,
              help="Also write the log to this file.")
@click.option("--debug", is_flag=True, help="Show full tracebacks on errors.")
@click.pass_context
def main(ctx, log_level, log_file, debug):
    """REGA-perturb: Reference REgulatory Path-Guided Analysis for resolving how genetic perturbations act through gene regulatory paths."""
    setup_logging(log_level, log_file)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ---------------------------------------------------------------- preprocessing


@main.command("qc")
@click.option("-i", "--input-h5ad", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Perturb-seq AnnData (cells x genes; raw UMI counts or normalized values, "
                   "see --input-type).")
@click.option("-o", "--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--prefix", required=True, help="Prefix of the output files.")
@click.option("--chunk-size", default=5000, show_default=True, type=int)
@click.option("--min-umi", default=1000, show_default=True, type=float)
@click.option("--max-umi", default=50000, show_default=True, type=float)
@click.option("--min-genes", default=500, show_default=True, type=float)
@click.option("--max-genes", default=None, type=float,
              help="Optional maximum number of detected genes per cell (default: no upper limit).")
@click.option("--max-mt", default=0.2, show_default=True, type=float,
              help="Maximum mitochondrial fraction (0-1).")
@click.option("--min-cells-per-gene", default=30, show_default=True, type=int)
@click.option("--keep-mt-genes", is_flag=True)
@click.option("--keep-nonstandard-chromosomes", is_flag=True)
@click.option("--tss-window", default=300000, show_default=True, type=int)
@click.option("--input-type", default="counts", show_default=True,
              type=click.Choice(["counts", "normalized"]),
              help="counts: raw UMI counts; normalized: already normalized values "
                   "(no cell filtering; use build-blocks --expr-mode as_is).")
@click.option("--tss-reference", type=click.Path(exists=True, dir_okay=False), default=None,
              help="TSS table (chrom, TSS, gene symbol, strand), used only when the data lack "
                   "gene coordinates (e.g. reference/tss_hg38.txt).")
@click.pass_context
def qc_cmd(ctx, **kw):
    """Cell / gene quality control and TSS window file."""
    from rega_perturb.preprocess.qc import run_qc

    with _handle_errors(ctx.obj["debug"]):
        run_qc(**kw)


@main.command("peaks")
@click.option("-i", "--peaks-bed", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-bed", required=True, type=click.Path(dir_okay=False))
@click.option("--keep-nonstandard-chromosomes", is_flag=True)
@click.pass_context
def name_peaks_cmd(ctx, peaks_bed, output_bed, keep_nonstandard_chromosomes):
    """Name reference peaks as <chrom>_<start>_<end>."""
    from rega_perturb.preprocess.reference import name_peaks

    with _handle_errors(ctx.obj["debug"]):
        name_peaks(peaks_bed, output_bed, standard_only=not keep_nonstandard_chromosomes)


@main.command("motif")
@click.option("-i", "--peaks-bed", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Named peaks (output of peaks).")
@click.option("--genome", required=True, help="HOMER genome, e.g. hg38.")
@click.option("--motif-file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--prefix", required=True)
@click.option("--homer-bin", default="findMotifsGenome.pl", show_default=True)
@click.option("--threads", default=1, show_default=True, type=int)
@click.option("--keep-intermediate", is_flag=True)
@click.pass_context
def motif_scan_cmd(ctx, peaks_bed, genome, motif_file, output_dir, prefix, homer_bin, threads,
                   keep_intermediate):
    """Scan peaks for motifs with HOMER."""
    from rega_perturb.preprocess.reference import run_motif_scan

    with _handle_errors(ctx.obj["debug"]):
        run_motif_scan(peaks_bed, genome, motif_file, output_dir, prefix, homer_bin=homer_bin,
                       threads=threads, keep_intermediate=keep_intermediate)


@main.command("retg")
@click.option("--tss-file", required=True, type=click.Path(exists=True, dir_okay=False),
              help="TSS window file written by `qc`.")
@click.option("--peaks-bed", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Named peaks (output of peaks).")
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False))
@click.option("--bedtools", default="bedtools", show_default=True)
@click.pass_context
def re_tg_cmd(ctx, tss_file, peaks_bed, output, bedtools):
    """Pair genes with the peaks in their TSS window."""
    from rega_perturb.preprocess.reference import build_re_tg

    with _handle_errors(ctx.obj["debug"]):
        build_re_tg(tss_file, peaks_bed, output, bedtools=bedtools)


@main.command("build-graph")
@click.option("--gene-metadata", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--re-tg", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--peak-motif", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--tau", default=1e5, show_default=True, type=float)
@click.option("--gene-col", default="gene_name", show_default=True)
@click.pass_context
def build_graph_cmd(ctx, **kw):
    """Build the RE-TG / RE-motif graph inputs."""
    from rega_perturb.preprocess.graph import build_graph

    with _handle_errors(ctx.obj["debug"]):
        build_graph(**kw)


@main.command("build-blocks")
@click.option("--graph-h5ad", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--raw-h5ad", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--cell-metadata", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--gene-metadata", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--expr-mode", default="logcp10k", show_default=True,
              type=click.Choice(["logcp10k", "logcpm", "as_is", "counts"]),
              help="logcp10k: log1p(count / library_size * 1e4); logcpm: same with 1e6; "
                   "as_is: use the input values unchanged (already normalized data); "
                   "counts: raw counts.")
@click.option("--chunk-size", default=5000, show_default=True, type=int)
@click.option("--perturb-col", default="gene", show_default=True,
              help="Cell-metadata column with the perturbation target of each cell.")
@click.option("--gene-col", default="gene_name", show_default=True)
@click.option("--chroms", multiple=True, help="Restrict to these chromosomes (repeatable).")
@click.option("--skip-p", is_flag=True, help="Do not write the perturbation assignment files.")
@click.pass_context
def build_blocks_cmd(ctx, chroms, **kw):
    """Write per-chromosome model input blocks."""
    from rega_perturb.preprocess.blocks import build_blocks

    with _handle_errors(ctx.obj["debug"]):
        build_blocks(chroms=list(chroms) or None, **kw)


# ---------------------------------------------------------------- model


@main.command("train")
@click.option("--block-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False),
              help="YAML file with training parameters (overridden by options below).")
@click.option("--device", default=None, help="cuda / cpu (default: cuda if available).")
@click.option("--chroms", multiple=True, help="Train on these chromosomes only (repeatable).")
@click.option("--K", "K", type=int, default=None, help="Number of latent programs.")
@click.option("--batch-size", type=int, default=None)
@click.option("--p-update-every", type=int, default=None)
@click.option("--rebalance-every", type=int, default=None)
@click.option("--lam-W", "lam_W", type=float, default=None)
@click.option("--lam-B", "lam_B", type=float, default=None)
@click.option("--lam-V", "lam_V", type=float, default=None)
@click.option("--lam-H", "lam_H", type=float, default=None)
@click.option("--lam-P", "lam_P", type=float, default=None)
@click.option("--seed", type=int, default=None, help="Seed for mini-batch shuffling.")
@click.option("--V-seed", "V_seed", type=int, default=None, help="Seed for initialisation.")
@click.option("--H-seed", "H_seed", type=int, default=None, help="Seed for initialisation.")
@click.option("--tf32/--no-tf32", "allow_tf32", default=None,
              help="Allow TF32 matrix multiplication on GPUs.")
@click.option("--progress/--no-progress", "show_progress", default=None)
@click.option("--plot/--no-plot", "make_plot", default=None)
@click.pass_context
def train_cmd(ctx, block_dir, output_dir, config_file, device, chroms, **overrides):
    """Train the model on input blocks."""
    with _handle_errors(ctx.obj["debug"]):
        from rega_perturb.model.config import make_training_config
        from rega_perturb.model.train import load_perturb_blocks, train_rega_perturb

        config = make_training_config(config_file, overrides)
        block_data, p_init = load_perturb_blocks(block_dir=Path(block_dir),
                                                 chroms=list(chroms) or None)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        params = dataclasses.asdict(config)
        params.update(device=device, chroms=list(chroms) or None)
        write_run_record(output_dir, "train", params=params, inputs={"block_dir": block_dir})
        train_rega_perturb(block_data=block_data, p_init=p_init, output_dir=Path(output_dir),
                           config=config, device=device)


@main.command("assemble")
@click.option("--graph-h5ad", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out-h5ad", required=True, type=click.Path(dir_okay=False))
@click.option("--cell-embeddings", is_flag=True)
@click.option("--max-cells-dense", default=200000, show_default=True, type=int)
@click.option("--peak-activity", is_flag=True,
              help="Also store the RE x target activity matrix (large for genome-scale screens).")
@click.option("--motif-tf", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Motif-to-TF table; also store both TF-target gene scores "
                   "(obsm TF_TG_influence, TF_TG_connectivity).")
@click.pass_context
def collect_cmd(ctx, graph_h5ad, checkpoint, out_h5ad, cell_embeddings, max_cells_dense,
                peak_activity, motif_tf):
    """Collect a trained model into a result AnnData."""
    from rega_perturb.results import collect_results

    with _handle_errors(ctx.obj["debug"]):
        collect_results(graph_h5ad, checkpoint, out_h5ad, cell_embeddings=cell_embeddings,
                        max_cells_dense=max_cells_dense, peak_activity=peak_activity,
                        motif_tf=motif_tf)


# ---------------------------------------------------------------- analyses


def _model_options(f):
    f = click.option("--motif-tf", required=True, type=click.Path(exists=True, dir_okay=False),
                     help="Motif-to-TF table (tab-separated: motif, TF symbol; no header).")(f)
    f = click.option("--graph-h5ad", required=True,
                     type=click.Path(exists=True, dir_okay=False))(f)
    f = click.option("--checkpoint", required=True,
                     type=click.Path(exists=True, dir_okay=False))(f)
    return f


@main.command("tf-tg")
@_model_options
@click.option("-o", "--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--score", "scores", multiple=True, type=click.Choice(["influence", "connectivity"]),
              help="Score(s) to compute (repeatable; default: both).")
@click.option("--top-per-tf", type=int, default=None, help="Keep the top N target genes per TF.")
@click.option("--min-score", type=float, default=None, help="Drop rows with score below this.")
@click.pass_context
def tf_tg_cmd(ctx, checkpoint, graph_h5ad, motif_tf, out_dir, scores, top_per_tf, min_score):
    """TF-target gene scores (tf_tg_<score>.tsv.gz: TF, TG, score)."""
    with _handle_errors(ctx.obj["debug"]):
        from rega_perturb.analysis.model_io import load_trained_model
        from rega_perturb.analysis.tf_tg_scores import SCORES, write_tf_tg_scores

        scores = tuple(scores) or SCORES
        model = load_trained_model(checkpoint, graph_h5ad)
        write_tf_tg_scores(model, motif_tf, out_dir, scores=scores, top_per_tf=top_per_tf,
                           min_score=min_score)
        write_run_record(out_dir, "tf_tg", params=dict(scores=list(scores), top_per_tf=top_per_tf,
                                                        min_score=min_score),
                         inputs=dict(checkpoint=checkpoint, graph_h5ad=graph_h5ad, motif_tf=motif_tf))


@main.command("tf-activity")
@_model_options
@click.option("-o", "--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--control", default=None, help="Control target label (default: auto-detect).")
@click.option("--threshold", type=float, default=None,
              help="Add a 'changed' column: |log2 fold change| >= threshold.")
@click.pass_context
def tf_activity_cmd(ctx, checkpoint, graph_h5ad, motif_tf, out_dir, control, threshold):
    """Perturbation-induced TF activity changes (target x TF log2 fold change)."""
    with _handle_errors(ctx.obj["debug"]):
        from rega_perturb.analysis.model_io import load_trained_model
        from rega_perturb.analysis.tf_activity import write_tf_activity_changes

        model = load_trained_model(checkpoint, graph_h5ad)
        write_tf_activity_changes(model, motif_tf, out_dir, control=control, threshold=threshold)
        write_run_record(out_dir, "tf_activity", params=dict(control=control, threshold=threshold),
                         inputs=dict(checkpoint=checkpoint, graph_h5ad=graph_h5ad, motif_tf=motif_tf))


@main.command("diff-network")
@_model_options
@click.option("-o", "--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--control", default=None, help="Control target label (default: auto-detect).")
@click.option("--target", "targets", multiple=True,
              help="Write the edge-level network of these targets (repeatable).")
@click.option("--all-targets", is_flag=True, help="Write edge-level networks of all targets.")
@click.option("--kind", default="delta", show_default=True, type=click.Choice(["delta", "specific"]),
              help="delta: change versus control; specific: the perturbation-specific network.")
@click.option("--format", "fmt", default="tsv", show_default=True, type=click.Choice(["tsv", "h5"]))
@click.option("--min-abs", type=float, default=None, help="tsv only: keep edges with |value| >= this.")
@click.option("--top-k", type=int, default=None, help="tsv only: keep the k largest |value| edges.")
@click.option("--no-scores", is_flag=True, help="Skip the RRS / NRS summary (rewiring_scores.h5).")
@click.pass_context
def diff_network_cmd(ctx, checkpoint, graph_h5ad, motif_tf, out_dir, control, targets, all_targets,
                     kind, fmt, min_abs, top_k, no_scores):
    """Differential regulatory networks and per-gene rewiring scores (RRS, NRS)."""
    with _handle_errors(ctx.obj["debug"]):
        from rega_perturb.analysis.diff_network import (
            network_factors,
            write_networks,
            write_rewiring_scores,
        )
        from rega_perturb.analysis.model_io import load_trained_model

        model = load_trained_model(checkpoint, graph_h5ad)
        f = network_factors(model, motif_tf, control=control)
        if not no_scores:
            write_rewiring_scores(f, Path(out_dir) / "rewiring_scores.h5")
        sel = [t for t in f.targets if t != f.control] if all_targets else list(targets)
        for t in sel:
            model.target_index(t)
        if sel:
            write_networks(f, sel, out_dir, kind=kind, min_abs=min_abs, top_k=top_k, fmt=fmt)
        write_run_record(out_dir, "diff_network",
                         params=dict(control=f.control, targets=sel if len(sel) <= 50 else len(sel),
                                     kind=kind, format=fmt, min_abs=min_abs, top_k=top_k,
                                     scores=not no_scores),
                         inputs=dict(checkpoint=checkpoint, graph_h5ad=graph_h5ad, motif_tf=motif_tf))


if __name__ == "__main__":  # pragma: no cover
    main()
