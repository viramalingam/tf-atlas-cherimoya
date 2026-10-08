#!/usr/bin/env python3
"""`bpnet-predict` for cherimoya models.

Takes the same arguments as `bpnet-predict` (bpnet-refactor 0b11a7c,
bpnet/cli/predict.py) and writes the same files into --output-dir, so the
tf-atlas pipeline outputs (and AnVIL table columns) keep their meaning:

  <model_tag>_predictions.h5   coords/{coords_chrom,coords_start,coords_end},
                               predictions/{pred_profs,pred_logcounts,
                                            true_profs,true_logcounts}
  pearson.txt spearman.txt jsd.txt   (counts pearson/spearman = mean over the
                                      two strands; jsd = median profile JSD)
  mnll/cross_entropy/jsd/mse/pearson/spearman/counts_pearson/counts_spearman .npz
  true_logcounts_<i>.npy pred_logcounts_<i>.npy
  <model_tag>_predictions_track_<i>.bw (+ _stats.txt)   with
                               --generate-predicted-profile-bigWigs
  config.json

Region selection follows bpnet's generator (sequtils.getPeakPositions): loci
from the input json's "loci" sources; --chroms (if given) takes precedence
over --test-indices-file (row positions in the loci file); windows running off
the chromosome are dropped; sorted by chrom and position; duplicates of the
same (chrom, output start) are predicted once.

Prediction follows bpnet-predict's single-count-head branch: with
--reverse-complement-average, logits and log counts are averaged with the
reverse-complement prediction (sequence and stranded control flipped, output
flipped back); the profile is the joint (2 x output_len) softmax times
exp(log counts), and each strand's log count is the log of its profile sum.
The metric functions are bpnet's own (bpnet_metrics.py, copied verbatim).
"""

import argparse
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpnet_metrics import metrics_update, write_bigwig  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

PEAK_COLS = ['chrom', 'st', 'end', 'name', 'weight', 'strand',
             'signal', 'p', 'q', 'summit']


def none_or_str(v):
    return None if v == "None" else v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help="cherimoya checkpoint (.torch)")
    p.add_argument('--chrom-sizes', required=True)
    p.add_argument('--chroms', nargs='+', default=None)
    p.add_argument('--test-indices-file', type=none_or_str, default=None)
    p.add_argument('--reference-genome', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--input-data', required=True, help="testing input json (bpnet format)")
    p.add_argument('--sequence-generator-name', default='BPNet', help="ignored; kept for CLI parity")
    p.add_argument('--input-seq-len', type=int, required=True)
    p.add_argument('--output-len', type=int, required=True)
    p.add_argument('--output-window-size', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=1, help="ignored; kept for CLI parity")
    p.add_argument('--generate-predicted-profile-bigWigs', action='store_true')
    p.add_argument('--reverse-complement-average', action='store_true')
    p.add_argument('--set-bias-as-zero', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    if a.chroms == ['None']:
        a.chroms = None
    if a.output_window_size is None:
        a.output_window_size = a.output_len
    return a


def select_loci(input_data, chroms, test_indices, input_seq_len, output_len, chrom_sizes):
    """bpnet getPeakPositions(mode='test', jitter 0) for the 'loci' sources."""
    task = json.load(open(input_data))["0"]
    sizes = pd.read_csv(chrom_sizes, sep='\t', header=None, names=['chrom', 'size'])
    size_of = dict(zip(sizes['chrom'], sizes['size']))
    flank = input_seq_len // 2
    frames = []
    for src in task['loci']['source']:
        df = pd.read_csv(src, sep='\t', header=None, names=PEAK_COLS)
        if chroms is not None:
            df = df[df['chrom'].isin(chroms)]
        elif test_indices is not None:
            df = df.loc[df.index[test_indices]]
        df = df.copy()
        df['pos'] = df['st'] + df['summit']
        df['start_coord'] = (df['pos'] - flank).astype(int)
        df['end_coord'] = (df['pos'] + flank).astype(int)
        df = df[df['start_coord'] >= 0]
        df = df[df['end_coord'] <= df['chrom'].map(size_of)]
        frames.append(df.sort_values(['chrom', 'end_coord']).reset_index(drop=True))
    loci = pd.concat(frames).reset_index(drop=True)
    loci['output_start'] = loci['pos'] - output_len // 2
    loci['output_end'] = loci['pos'] + output_len // 2
    # bpnet-predict skips repeated (chrom, start) keys
    loci = loci.drop_duplicates(subset=['chrom', 'output_start']).reset_index(drop=True)
    return task, loci


def extract(task, loci, reference_genome, input_seq_len, output_len, set_bias_as_zero):
    """One-hot (N, 4, L_in), control (N, 2, L_in) and true counts (N, L_out, 2)."""
    import pyBigWig
    import pyfaidx
    fasta = pyfaidx.Fasta(reference_genome)
    sig = [pyBigWig.open(s) for s in task['signal']['source']]
    ctl = [pyBigWig.open(s) for s in task['bias']['source']] if 'bias' in task else []
    n = len(loci)
    X = np.zeros((n, 4, input_seq_len), dtype=np.float32)
    C = np.zeros((n, max(len(ctl), 1), input_seq_len), dtype=np.float32)
    Y = np.zeros((n, output_len, len(sig)), dtype=np.float64)
    lut = np.full(256, -1, dtype=np.int64)
    for k, b in enumerate(b"ACGT"):
        lut[b] = k
        lut[ord(chr(b).lower())] = k
    for i, r in enumerate(loci.itertuples()):
        s, e = int(r.start_coord), int(r.end_coord)
        seq = np.frombuffer(str(fasta[r.chrom][s:e]).encode(), dtype=np.uint8)
        idx = lut[seq]
        ok = idx >= 0
        X[i, idx[ok], np.nonzero(ok)[0]] = 1.0
        if not set_bias_as_zero:
            for j, bw in enumerate(ctl):
                C[i, j] = np.nan_to_num(bw.values(r.chrom, s, e, numpy=True))
        os_, oe = int(r.output_start), int(r.output_end)
        for j, bw in enumerate(sig):
            Y[i, :, j] = np.nan_to_num(bw.values(r.chrom, os_, oe, numpy=True))
    return X, C, Y


def inference_mode():
    """Grad mode for cherimoya inference. Under no_grad cherimoya dispatches its
    inference megakernel, which casts the MLP weights to bf16 for tl.dot; triton
    cannot compile that below compute capability 8.0 (e.g. T4). There, keep grad
    mode on so the fp32 training Triton path runs instead (same outputs to fp32
    precision; outputs are detached)."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return torch.enable_grad()
    return torch.no_grad()


def predict_logits(model_path, X, C, batch_size, rc_average, device):
    """Cherimoya forward -> (logits (N, 2, L), logcounts (N,)), RC-averaged."""
    import torch
    from cherimoya import Cherimoya
    model = Cherimoya.load(model_path, device=device)
    model.eval()
    n = X.shape[0]
    logits, logcounts = [], []
    with inference_mode():
        for s in range(0, n, batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).to(device)
            cb = torch.from_numpy(C[s:s + batch_size]).to(device)
            lg, lc = model(xb, cb)
            lg, lc = lg.detach().float(), lc.detach().float().reshape(-1)
            if rc_average:
                lg_rc, lc_rc = model(torch.flip(xb, dims=(1, 2)), torch.flip(cb, dims=(1, 2)))
                lg = (lg + torch.flip(lg_rc.detach().float(), dims=(1, 2))) / 2
                lc = (lc + lc_rc.detach().float().reshape(-1)) / 2
            logits.append(lg.cpu().numpy().astype(np.float64))
            logcounts.append(lc.cpu().numpy().astype(np.float64))
    return np.concatenate(logits), np.concatenate(logcounts)


def profiles_from_logits(logits, logcounts):
    """bpnet-predict, single count head: joint softmax over (strands x positions)
    times exp(logcounts); per-strand log counts = log of the strand's sum."""
    from scipy.special import logsumexp
    n, t, L = logits.shape
    flat = logits.reshape(n, t * L)
    probs = np.exp(flat - logsumexp(flat, axis=1, keepdims=True)).reshape(n, t, L)
    prof = probs * np.exp(logcounts)[:, None, None]          # (n, t, L)
    pred_profs = prof.transpose(0, 2, 1)                       # (n, L, t)
    pred_logcounts = np.log(pred_profs.sum(axis=1))            # (n, t)
    return pred_profs, pred_logcounts


def write_outputs(output_dir, model_tag, chroms_, starts, ends, pred_profs, pred_logcounts,
                  true_profs, true_logcounts, chrom_sizes=None, bigwig_chroms=None,
                  generate_bigwigs=False, config=None):
    """The tail of bpnet-predict: metrics, h5, npz/txt, optional bigWigs."""
    os.makedirs(output_dir, exist_ok=True)
    n, L, T = pred_profs.shape
    mt = {k: np.zeros((n, T)) for k in (
        'profile_mnlls', 'profile_cross_entropys', 'profile_jsds', 'profile_mses',
        'profile_pearsonrs', 'profile_spearmanrs', 'all_true_logcounts', 'all_pred_logcounts')}
    for i in range(n):
        for j in range(T):
            metrics_update(mt, i, j, true_profs[i, :, j], true_logcounts[i, j],
                           pred_profs[i, :, j], pred_logcounts[i, j])

    with h5py.File(os.path.join(output_dir, f"{model_tag}_predictions.h5"), "w") as h:
        c = h.create_group("coords")
        c.create_dataset("coords_chrom", data=np.array(list(chroms_), dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")
        c.create_dataset("coords_start", data=np.asarray(starts, dtype=int), compression="gzip")
        c.create_dataset("coords_end", data=np.asarray(ends, dtype=int), compression="gzip")
        p = h.create_group("predictions")
        p.create_dataset("pred_profs", data=pred_profs.astype(float), compression="gzip")
        p.create_dataset("pred_logcounts", data=pred_logcounts.astype(float), compression="gzip")
        p.create_dataset("true_profs", data=true_profs.astype(float), compression="gzip")
        p.create_dataset("true_logcounts", data=true_logcounts.astype(float), compression="gzip")

    counts_pearson = np.zeros(T)
    counts_spearman = np.zeros(T)
    for i in range(T):
        np.save(f'{output_dir}/true_logcounts_{i}.npy', mt['all_true_logcounts'][:, i])
        np.save(f'{output_dir}/pred_logcounts_{i}.npy', mt['all_pred_logcounts'][:, i])
        counts_pearson[i] = pearsonr(mt['all_true_logcounts'][:, i], mt['all_pred_logcounts'][:, i])[0]
        counts_spearman[i] = spearmanr(mt['all_true_logcounts'][:, i], mt['all_pred_logcounts'][:, i])[0]
    logging.info("counts pearson: {}".format(counts_pearson))
    logging.info("counts spearman: {}".format(counts_spearman))
    with open(f'{output_dir}/spearman.txt', "w+") as f:
        f.write(str(round(np.nan_to_num(np.mean(counts_spearman)), 3)))
    with open(f'{output_dir}/pearson.txt', "w+") as f:
        f.write(str(round(np.nan_to_num(np.mean(counts_pearson)), 3)))
    with open(f'{output_dir}/jsd.txt', "w+") as f:
        f.write(str(round(np.nan_to_num(np.median(mt['profile_jsds'])), 3)))
    np.savez_compressed(f'{output_dir}/mnll', mnll=mt['profile_mnlls'])
    np.savez_compressed(f'{output_dir}/cross_entropy', cross_entropy=mt['profile_cross_entropys'])
    np.savez_compressed(f'{output_dir}/jsd', jsd=mt['profile_jsds'])
    np.savez_compressed(f'{output_dir}/mse', mse=mt['profile_mses'])
    np.savez_compressed(f'{output_dir}/pearson', pearson=mt['profile_pearsonrs'])
    np.savez_compressed(f'{output_dir}/spearman', spearman=mt['profile_spearmanrs'])
    np.savez_compressed(f'{output_dir}/counts_pearson', counts_pearson=counts_pearson)
    np.savez_compressed(f'{output_dir}/counts_spearman', counts_spearman=counts_spearman)

    if generate_bigwigs:
        sizes = pd.read_csv(chrom_sizes, sep='\t', header=None, names=['chrom', 'size']).set_index('chrom')
        bw_chroms = sorted(bigwig_chroms[:] if bigwig_chroms is not None else set(chroms_))
        header = [(c_, int(sizes.at[c_, 'size'])) for c_ in bw_chroms]
        mids = list((np.asarray(starts) + np.asarray(ends)) // 2)
        regions = list(zip(list(chroms_), list(starts), list(ends), mids))
        for i in range(T):
            write_bigwig(pred_profs[:, :, i], regions, header,
                         f'{output_dir}/{model_tag}_predictions_track_{i}.bw',
                         f'{output_dir}/{model_tag}_predictions_track_{i}_stats.txt')
    if config is not None:
        with open(f'{output_dir}/config.json', 'w') as fp:
            json.dump(config, fp)
    return {'pearson': float(np.mean(counts_pearson)), 'spearman': float(np.mean(counts_spearman)),
            'jsd': float(np.median(mt['profile_jsds'])), 'n': int(n)}


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(f'{a.output_dir}/predict.log'),
                                  logging.StreamHandler()])
    t0 = time.time()
    test_indices = None
    if a.test_indices_file is not None:
        test_indices = [int(l.strip()) for l in open(a.test_indices_file) if l.strip()]
    task, loci = select_loci(a.input_data, a.chroms, test_indices, a.input_seq_len,
                             a.output_len, a.chrom_sizes)
    logging.info(f"{len(loci)} regions (chroms={a.chroms}, indices={a.test_indices_file})")
    X, C, Y = extract(task, loci, a.reference_genome, a.input_seq_len, a.output_len,
                      a.set_bias_as_zero)
    logits, logcounts = predict_logits(a.model, X, C, a.batch_size,
                                       a.reverse_complement_average, a.device)
    pred_profs, pred_logcounts = profiles_from_logits(logits, logcounts)
    s = a.output_len // 2 - a.output_window_size // 2
    e = s + a.output_window_size
    pred_profs = pred_profs[:, s:e, :]
    true_profs = Y[:, s:e, :]
    # bpnet's generator: log(1 + total true counts) per strand over the output window
    true_logcounts = np.log1p(Y.sum(axis=1))
    off = (a.output_len - a.output_window_size) // 2
    model_tag = os.path.basename(a.model).split('.')[0]
    res = write_outputs(a.output_dir, model_tag, loci['chrom'].values,
                        (loci['output_start'] + off).values, (loci['output_end'] - off).values,
                        pred_profs, pred_logcounts, true_profs, true_logcounts,
                        chrom_sizes=a.chrom_sizes, bigwig_chroms=a.chroms,
                        generate_bigwigs=a.generate_predicted_profile_bigWigs,
                        config=vars(a))
    logging.info(f"pearson {res['pearson']:.4f} spearman {res['spearman']:.4f} "
                 f"jsd {res['jsd']:.4f} n={res['n']} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
