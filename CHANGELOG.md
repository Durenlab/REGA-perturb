# Changelog

All notable changes are documented here. Versions follow [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-11-03

First public version.

- Preprocessing: cell / gene QC of raw counts or already-normalized data (streamed),
  peak naming, HOMER motif scanning, gene - regulatory element pairing, graph inputs,
  per-chromosome model input blocks (`logcp10k` default; `logcpm`, `as_is`, `counts`).
- Model training (CPU or GPU) with YAML configuration and recorded seeds; fixed
  training schedule that runs until the full-data reconstruction error has converged;
  result AnnData (optionally with both TF-target gene scores).
- Analyses: TF-target gene scores (`influence`, `connectivity`), perturbation-induced
  TF activity changes, perturbation-specific and differential networks with per-gene
  RRS / NRS scores.
- Command-line interface `rega-perturb` (`qc`, `peaks`, `motif`, `retg`, `build-graph`,
  `build-blocks`, `train`, `assemble`, `tf-tg`, `tf-activity`, `diff-network`); every
  command writes a `<step>.run.yaml` run record.
- Human reference files (`reference/`: HOMER-format motif collection, motif-to-TF table, hg38 TSS table).
- `qc --tss-reference`: gene coordinates from a TSS table for data without gene annotation.
- K562 example with bundled mini data and the K562 peak list; unit / end-to-end tests.
