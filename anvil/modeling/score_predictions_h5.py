#!/usr/bin/env python3
"""Re-score existing BPNet-layout prediction h5 files on a region subset.

Used for the arms that are not a fresh cherimoya prediction:
  - the released BPNet model (run_1_fold0): its chrX predictions h5 already
    exists; restrict it to the held-out test set T;
  - a teacher ensemble: several h5 files over the same regions are averaged
    first (mean predicted profile; log of the mean predicted counts per strand,
    as in tfatlas/eval/src/pw_eval.py average_h5).

Rows are matched on (chrom, summit - output_len//2), the bpnet-predict
coordinate convention. Writes the same files as bpnet-predict
(cherimoya_predict.write_outputs), without bigWigs.

Usage:
  score_predictions_h5.py --h5 released/..._predictions.h5 \
      --peaks peaks.bed --test-indices-file loci_test_indices_fold0.txt \
      [--background bg.bed --background-indices-file background_test_indices_fold0.txt] \
      --output-len 500 --output-dir out --model-tag EXP_split000
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cherimoya_predict import PEAK_COLS, write_outputs  # noqa: E402


def read_indices(path):
    return [int(l.strip()) for l in open(path) if l.strip()]


def load_h5(path):
    with h5py.File(path, 'r') as f:
        chrom = np.array([c.decode() if isinstance(c, bytes) else str(c)
                          for c in f['coords/coords_chrom'][:]])
        return {'chrom': chrom, 'start': f['coords/coords_start'][:], 'end': f['coords/coords_end'][:],
                'pred_profs': f['predictions/pred_profs'][:], 'pred_logcounts': f['predictions/pred_logcounts'][:],
                'true_profs': f['predictions/true_profs'][:], 'true_logcounts': f['predictions/true_logcounts'][:]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5', nargs='+', required=True, help="one h5, or several to ensemble")
    p.add_argument('--peaks', required=True)
    p.add_argument('--test-indices-file', required=True, help="row numbers of --peaks to keep")
    p.add_argument('--background', default=None)
    p.add_argument('--background-indices-file', default=None)
    p.add_argument('--output-len', type=int, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-tag', required=True)
    a = p.parse_args(argv)

    keys = []
    peaks = pd.read_csv(a.peaks, sep='\t', header=None, names=PEAK_COLS)
    sel = peaks.iloc[read_indices(a.test_indices_file)]
    keys += list(zip(sel['chrom'], sel['st'] + sel['summit'] - a.output_len // 2))
    if a.background is not None:
        bg = pd.read_csv(a.background, sep='\t', header=None, names=PEAK_COLS)
        selb = bg.iloc[read_indices(a.background_indices_file)]
        keys += list(zip(selb['chrom'], selb['st'] + selb['summit'] - a.output_len // 2))
    keys = set((c, int(s)) for c, s in keys)

    ref = None
    prof_sum = lin_sum = None
    for path in a.h5:
        d = load_h5(path)
        rowkey = list(zip(d['chrom'], d['start'].astype(int)))
        keep = np.array([k in keys for k in rowkey])
        if ref is None:
            ref_rows = np.nonzero(keep)[0]
            ref = {rowkey[i]: j for j, i in enumerate(ref_rows)}
            out = {k: d[k][ref_rows] for k in ('chrom', 'start', 'end', 'true_profs', 'true_logcounts')}
            prof_sum = d['pred_profs'][ref_rows].astype(np.float64)
            lin_sum = np.exp(d['pred_logcounts'][ref_rows].astype(np.float64))
            continue
        sub = np.nonzero(keep)[0]
        order = np.argsort([ref[tuple(rowkey[i])] for i in sub])
        assert len(sub) == len(ref_rows), f"region set mismatch in {path}"
        rows = sub[order]
        prof_sum += d['pred_profs'][rows]
        lin_sum += np.exp(d['pred_logcounts'][rows].astype(np.float64))
    K = len(a.h5)
    n_missing = len(keys) - len(ref_rows)
    print(f"{len(ref_rows)} of {len(keys)} requested regions found in the h5 "
          f"({n_missing} missing); {K} model(s)", flush=True)
    res = write_outputs(a.output_dir, a.model_tag, out['chrom'], out['start'], out['end'],
                        prof_sum / K, np.log(lin_sum / K), out['true_profs'], out['true_logcounts'],
                        config={'h5': a.h5, 'peaks': a.peaks, 'test_indices_file': a.test_indices_file,
                                'background': a.background,
                                'background_indices_file': a.background_indices_file,
                                'n_models': K, 'n_regions': int(len(ref_rows)),
                                'n_missing': int(n_missing)})
    print(f"pearson {res['pearson']:.4f} spearman {res['spearman']:.4f} jsd {res['jsd']:.4f} n={res['n']}")


if __name__ == '__main__':
    main()
