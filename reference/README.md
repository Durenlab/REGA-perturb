# Reference files (human)

| File | Format | Used by |
|---|---|---|
| `motifs_homer.txt` | motif collection in HOMER motif format (1,465 motifs) | `rega-perturb motif --motif-file` |
| `tss_hg38.txt` | hg38 transcription start sites; four tab-separated columns, no header: chrom, TSS position, gene symbol, strand | `rega-perturb qc --tss-reference` (only for data without gene coordinates) |
| `motif_tf_human.txt` | two tab-separated columns, no header: motif name, TF gene symbol | `--motif-tf` of `tf-tg`, `tf-activity`, `diff-network` |

