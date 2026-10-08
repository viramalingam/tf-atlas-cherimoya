#!/usr/bin/env python3
"""`bpnet-train` for cherimoya: same inputs as the tf-atlas modelling pipeline.

Reads the bpnet-format training input json (signal / bias bigWigs, loci,
background_loci + ratio) and splits json, builds a cherimoya v0.2.0+g2 fit
config, and runs `python -m cherimoya_cli fit`.

splits json, either form used by tf-atlas-pipeline:
  chromosome-wise  {"0": {"train": [...], "val": [...], "test": [...]}}
  peak-wise        {"0": {"loci_train_indices_file": ..., "loci_val_indices_file": ...,
                          "background_train_indices_file": ..., ...}}
In the peak-wise form the indices are row numbers of the loci / background
files (as written by peak_wise_splits.py); train peaks + train background are
fit on, validation peaks select the checkpoint, and every chromosome in
--chroms is allowed.

Recipe = the bpnet-arch-benchmark cherimoya es100 recipe: cherimoya defaults,
full training budget (early stopping patience = epochs), checkpoint chosen by
validation count Pearson. Architecture/optimizer knobs come from --params
(json, keys of cherimoya's fit parameters, e.g. n_filters, n_layers,
max_jitter). Outputs in --output-dir, named after --model-output-filename
like bpnet-train's <name>_split000:
  <name>_split000.torch         best checkpoint (by validation count pearson)
  <name>_split000.final.torch   last epoch
  <name>_split000.log / .detailed.log   per-epoch training log
  config_split000.json          the cherimoya fit config
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

PEAK_COLS = ['chrom', 'st', 'end', 'name', 'weight', 'strand',
             'signal', 'p', 'q', 'summit']


def read_indices(path):
    return [int(l.strip()) for l in open(path) if l.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-data', required=True, help="training input json (bpnet format)")
    p.add_argument('--output-dir', required=True)
    p.add_argument('--reference-genome', required=True)
    p.add_argument('--chrom-sizes', required=True)
    p.add_argument('--chroms', nargs='+', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--params', required=True, help="cherimoya params json")
    p.add_argument('--model-output-filename', required=True)
    p.add_argument('--input-seq-len', type=int, required=True)
    p.add_argument('--output-len', type=int, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=None, help="AdamW lr (cherimoya adam_lr)")
    p.add_argument('--early-stopping-patience', type=int, default=None,
                   help="default: = --epochs (es100: never stop early)")
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args(argv)

    os.makedirs(a.output_dir, exist_ok=True)
    task = json.load(open(a.input_data))["0"]
    split = json.load(open(a.splits))["0"]
    params = json.load(open(a.params))

    (loci_bed,) = task['loci']['source']
    (bg_bed,) = task['background_loci']['source']
    ratio = task['background_loci']['ratio'][0]
    name = os.path.join(a.output_dir, f"{a.model_output_filename}_split000")

    cfg = {
        "sequences": a.reference_genome,
        "signals": [list(task['signal']['source'])],     # nested = one stranded group
        "controls": [list(task['bias']['source'])],
        "in_window": a.input_seq_len,
        "out_window": a.output_len,
        "max_epochs": a.epochs,
        "early_stopping": a.early_stopping_patience if a.early_stopping_patience is not None else a.epochs,
        "batch_size": a.batch_size,
        "negative_ratio": ratio,
        "summits": True,
        "random_state": a.seed,
        "verbose": True,
        "name": name,
        "performance_filename": name + ".performance.tsv",
    }
    if a.learning_rate is not None:
        cfg["adam_lr"] = a.learning_rate

    if 'loci_train_indices_file' in split:
        peaks = pd.read_csv(loci_bed, sep='\t', header=None, names=PEAK_COLS)
        bg = pd.read_csv(bg_bed, sep='\t', header=None, names=PEAK_COLS)
        tr = read_indices(split['loci_train_indices_file'])
        va = read_indices(split['loci_val_indices_file'])
        bg_tr = read_indices(split['background_train_indices_file'])
        te = read_indices(split['loci_test_indices_file']) if 'loci_test_indices_file' in split else []
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), \
            "peak indices overlap between train/valid/test"
        out = {}
        for key, df, idx in (("loci", peaks, tr), ("validation_loci", peaks, va),
                             ("negatives", bg, bg_tr)):
            path = os.path.join(a.output_dir, f"pw_{key}.bed")
            df.iloc[idx].to_csv(path, sep='\t', header=False, index=False)
            out[key] = path
        cfg.update(out)
        cfg["training_chroms"] = list(a.chroms)
        cfg["validation_chroms"] = list(a.chroms)
        print(f"peak-wise split: {len(tr)} train / {len(va)} valid / {len(te)} test peaks, "
              f"{len(bg_tr)} train background", flush=True)
    else:
        leak = (set(split['train']) | set(split['val'])) & set(split.get('test', []))
        assert not leak, f"chromosome leak between train/val and test: {leak}"
        cfg.update({"loci": loci_bed, "negatives": bg_bed,
                    "training_chroms": split['train'], "validation_chroms": split['val']})

    for k, v in params.items():
        if not k.startswith('_'):
            cfg[k] = v

    cfg_path = os.path.join(a.output_dir, "config_split000.json")
    with open(cfg_path, 'w') as fp:
        json.dump(cfg, fp, indent=2)
    print(json.dumps(cfg, indent=2), flush=True)

    r = subprocess.run([sys.executable, "-m", "cherimoya_cli", "fit", "-p", cfg_path])
    if r.returncode != 0:
        raise SystemExit(f"cherimoya fit failed rc={r.returncode}")
    if not os.path.exists(name + ".torch"):
        raise SystemExit(f"no checkpoint at {name}.torch")
    print(f"model: {name}.torch", flush=True)


if __name__ == '__main__':
    main()
