# tf-atlas-cherimoya

Cherimoya peak-wise teachers and distillation for the tf-atlas TF ChIP-seq models, as AnVIL/Terra WDL workflows. The layout, inputs, output files and AnVIL table columns follow [tf-atlas-pipeline](https://github.com/viramalingam/tf-atlas-pipeline) (`anvil/modeling/`), so the cherimoya runs sit next to the BPNet `run_1` outputs in the same tables. First use: the 176 ENCODE4 Drosophila experiments (`fly_experiment`, dm6, 1102 bp in / 500 bp out).

The recipe is the one that won the human `bpnet-arch-benchmark` (52 ENCODE TF ChIP experiments): cherimoya v0.2.0 (+ the small `g2` patch below), the full 100-epoch budget with the checkpoint chosen by validation count Pearson ("es100"), 20 peak-wise teachers, and a distilled student trained on the teacher-ensemble soft targets.

## Workflows

| WDL | what it does | GPU |
|---|---|---|
| `anvil/modeling/create_peak_wise_splits.wdl` | tf-atlas `peak_wise_splits.py` (blocks of regions closer than input length + 2 x jitter, chunked by counts) plus an optional common held-out test set T: `test_chroms` (e.g. `chrX`), `test_frac` (fraction of all peaks and of all background regions), `seed`. T is the `test` split of every fold; the other blocks, on all chromosomes, form 20 chunks, fold k validates on chunk k and trains on the rest. Leakage gates fail the task on any violation. | no |
| `anvil/modeling/run_cherimoya_modelling.wdl` | one cherimoya teacher per fold (scatter). Same inputs as `run_modelling.wdl` (`training_input` / `testing_input` json templates, `indices_files`, ...), same output folders (`model`, `predictions_and_metrics_*`) and metrics (`pearson`, `spearman`, `jsd`, `*_all_peaks`, `auprc`, `auroc`, `*_wo_bias`), as arrays over folds. | yes |
| `anvil/modeling/run_cherimoya_distillation.wdl` | soft targets from all teachers on the pool (everything except T; original, reverse complement, 2 jittered copies with ~4% substitutions), the distilled student, and its outputs in the `run_modelling` layout. The same task scores the teacher ensemble (`*_ensemble`) and, if its prediction files are given, the released model (`*_released`) on the same T. | yes |

All "test" metrics are on T. For the released fly models (`run_1_fold0`, trained on chr2R/3L/3R/4, validated on chr2L), T is unseen data, because it is carved from chrX.

## Scripts (`anvil/modeling/`)

| file | origin |
|---|---|
| `peak_wise_splits.py`, `peak_wise_splits_pipeline.sh` | tf-atlas-pipeline `main` @ 124f049, plus the T options (default behaviour unchanged) |
| `auprc_auroc_calculations.py` | tf-atlas-pipeline v2.1.0-rc.5, unchanged |
| `bpnet_metrics.py` | the metric functions of `bpnet-predict` and `write_bigwig`, copied verbatim from kundajelab/bpnet-refactor @ 0b11a7c |
| `cherimoya_train.py` | `bpnet-train` stand-in: bpnet-format input json + splits json -> `cherimoya fit` |
| `cherimoya_predict.py` | `bpnet-predict` stand-in: same flags, same output files (`<exp>_split000_predictions.h5`, `pearson.txt`, ..., bigWigs) |
| `score_predictions_h5.py` | re-scores existing prediction h5 files (released model; teacher ensemble) on a region subset |
| `cherimoya_distill_targets.py`, `cherimoya_distill_student.py` | ports of the bpnet-arch-benchmark `cheri_distill_*` scripts |
| `cherimoya_modelling_pipeline.sh`, `cherimoya_distillation_pipeline.sh` | task drivers in the layout of `modelling_pipeline.sh` |
| `params/cherimoya_params_n8_f64_jit128_ipt_1102bp_out_500bp.json` | fly architecture: 8 layers (forced by 1102 -> 500 bp), 64 filters, jitter 128 |
| `docker/` | the task image (`vivekramalingam/tf-atlas:gcp-cherimoya_v0.2.0-g2`): PyTorch 2.10 + cherimoya v0.2.0 from GitHub + `cherimoya_v0.2.0_g2.patch` |

The `g2` patch adds `validation_loci` (needed for peak-wise folds) and two opt-in options (fixed count-loss weight, validation-loss checkpoint selection); with the defaults, training is the same as stock v0.2.0.

## Versions

Each WDL clones this repo at the tag in its `git clone --branch` line, like tf-atlas-pipeline. Releasing a change: bump that tag in all three WDLs, commit, tag, push, then point the Terra method configs at the new Dockstore version.
