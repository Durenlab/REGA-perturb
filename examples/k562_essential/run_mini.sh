#!/usr/bin/env bash
# Mini K562 essential example: raw counts -> model -> TF-TG scores, TF activity changes and
# differential networks. Requires rega-perturb (with PyTorch) and bedtools.
#
#   bash run_mini.sh [data_dir] [out_dir]
#
# data_dir defaults to the mini/ folder next to this script; out_dir defaults to ./output.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${1:-$HERE/mini}
OUT=${2:-output}
MOTIF_TF=$HERE/../../reference/motif_tf_human.txt
mkdir -p "$OUT"
RP="rega-perturb --log-file $OUT/run.log"

$RP qc -i "$DATA/k562_essential_mini_raw.h5ad" -o "$OUT/qc" --prefix k562_mini \
    --min-genes 100 --max-genes 5000
$RP peaks -i "$DATA/k562_mini_peaks.bed" -o "$OUT/reference/peaks_named.bed"
$RP retg --tss-file "$OUT/qc/k562_mini_REGA_qc_gene_TSS_300kb.txt" \
    --peaks-bed "$OUT/reference/peaks_named.bed" -o "$OUT/reference/re_tg.tsv"
$RP build-graph --gene-metadata "$OUT/qc/k562_mini_REGA_qc_gene_metadata.csv" \
    --re-tg "$OUT/reference/re_tg.tsv" --peak-motif "$DATA/k562_mini_peak_motif.txt.gz" \
    -o "$OUT/graph"
$RP build-blocks --graph-h5ad "$OUT/graph/REGA_graph_without_GEX.h5ad" \
    --raw-h5ad "$DATA/k562_essential_mini_raw.h5ad" \
    --cell-metadata "$OUT/qc/k562_mini_REGA_qc_cell_metadata.csv" \
    --gene-metadata "$OUT/qc/k562_mini_REGA_qc_gene_metadata.csv" -o "$OUT/blocks"
$RP train --block-dir "$OUT/blocks" -o "$OUT/model" --config "$HERE/train_mini.yaml" --no-progress
$RP assemble --graph-h5ad "$OUT/graph/REGA_graph_without_GEX.h5ad" \
    --checkpoint "$OUT/model/final_perturb.pt" -o "$OUT/model/rega_perturb_result.h5ad" \
    --motif-tf "$MOTIF_TF"

MODEL="--checkpoint $OUT/model/final_perturb.pt --graph-h5ad $OUT/graph/REGA_graph_without_GEX.h5ad --motif-tf $MOTIF_TF"
$RP tf-tg $MODEL -o "$OUT/tf_tg"
$RP tf-activity $MODEL -o "$OUT/tf_activity"
$RP diff-network $MODEL -o "$OUT/diff_network" --all-targets --format h5
echo "done: $OUT"
