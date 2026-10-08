#!/usr/bin/env python3
"""Distillation targets from the peak-wise cherimoya teachers.

Port of tfatlas/eval/src/cheri_distill_targets.py (bpnet-arch-benchmark) to the
tf-atlas pipeline inputs: regions come from the bpnet-format training input
json (loci / background_loci / ratio, signal + bias bigWigs) and the
peak_wise_splits.py indices.

  pool   = every peak and background region that is in some fold's valid
           split (i.e. everything except the common test set T); chunk = the
           fold whose valid split holds it. Background is subsampled to
           ratio x pool peaks (seeded), as in the human recipe.
  windows: original + reverse complement + n_mut copies with a random jitter
           (+-jitter, peaks only) and ~frac random substitutions; the control
           track follows the window (reverse complement flips strands).
  targets: mean over teachers of the logits (2 x output_len), of the joint
           softmax probabilities, and of the log-count scalar.

Teachers never see T; T rows are not in the pool, so they are never targets.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cherimoya_predict import PEAK_COLS, inference_mode  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

BASES = np.frombuffer(b"ACGT", dtype="S1")


def read_indices(path):
    return np.array([int(l.strip()) for l in open(path) if l.strip()], dtype=int)


def chunk_map(indices_dir, prefix, n_rows, n_folds):
    chunk = np.full(n_rows, -1, dtype=np.int16)
    for f in range(n_folds):
        idx = read_indices(os.path.join(indices_dir, f"{prefix}_valid_indices_fold{f}.txt"))
        assert (chunk[idx] == -1).all(), f"{prefix} rows valid in more than one fold"
        chunk[idx] = f
    return chunk


def load_pool(task, indices_dir, n_folds, seed, limit=None):
    (loci_bed,) = task['loci']['source']
    (bg_bed,) = task['background_loci']['source']
    ratio = task['background_loci']['ratio'][0]
    peaks = pd.read_csv(loci_bed, sep='\t', header=None, names=PEAK_COLS)
    negs = pd.read_csv(bg_bed, sep='\t', header=None, names=PEAK_COLS)
    peaks['chunk'] = chunk_map(indices_dir, 'loci', len(peaks), n_folds)
    negs['chunk'] = chunk_map(indices_dir, 'background', len(negs), n_folds)
    t_peaks = read_indices(os.path.join(indices_dir, "loci_test_indices_fold0.txt"))
    t_negs = read_indices(os.path.join(indices_dir, "background_test_indices_fold0.txt"))
    # every row is either in exactly one valid chunk or in T
    assert (peaks.loc[peaks['chunk'] < 0].index.sort_values() == np.sort(t_peaks)).all()
    assert (negs.loc[negs['chunk'] < 0].index.sort_values() == np.sort(t_negs)).all()
    peaks = peaks[peaks['chunk'] >= 0]
    negs = negs[negs['chunk'] >= 0]
    rng = np.random.RandomState(seed)
    n_bg = min(len(negs), int(ratio * len(peaks)))
    negs = negs.iloc[np.sort(rng.choice(len(negs), n_bg, replace=False))]
    if limit:
        peaks = peaks.iloc[:limit]
        negs = negs.iloc[:max(1, int(ratio * limit))]
    loci = pd.concat([peaks.assign(label=1), negs.assign(label=0)], ignore_index=True)
    loci['pos'] = loci['st'] + loci['summit']
    return loci, ratio


def extract_wide(loci, task, reference_genome, in_len, jit):
    """Wide sequences + wide control arrays (jitter margin around the input window)."""
    import pyBigWig
    import pyfaidx
    fasta = pyfaidx.Fasta(reference_genome)
    bws = [pyBigWig.open(s) for s in task['bias']['source']]
    seqs, ctls, keep = [], [], []
    W = in_len + 2 * jit
    for i, row in enumerate(loci.itertuples()):
        pos = int(row.pos)
        s, e = pos - W // 2, pos + W // 2
        if s < 0:
            continue
        try:
            seq = str(fasta[row.chrom][s:e]).upper()
            ctl = np.stack([np.nan_to_num(bw.values(row.chrom, s, e, numpy=True)) for bw in bws])
        except (KeyError, RuntimeError, ValueError):
            continue
        if len(seq) != W:
            continue
        seqs.append(seq)
        ctls.append(ctl.astype(np.float32))
        keep.append(i)
    for bw in bws:
        bw.close()
    return seqs, np.stack(ctls), loci.iloc[keep].reset_index(drop=True)


def one_hot(seqs):
    """list of str -> (N, 4, L) int8, ACGT rows, anything else all-zero."""
    arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
    out = np.zeros((len(seqs), 4, arr.shape[1]), dtype=np.int8)
    for k, b in enumerate(BASES):
        out[:, k, :] = arr == b
    return out


def mutate_seq(seq, frac, rng):
    n = max(1, int(frac * len(seq)))
    pos = rng.sample(range(len(seq)), n)
    others = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    out = list(seq)
    for p in pos:
        out[p] = rng.choice(others.get(out[p], "ACGT"))
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teachers", nargs='+', required=True, help="teacher checkpoints (.torch)")
    ap.add_argument("--input-data", required=True, help="training input json (bpnet format)")
    ap.add_argument("--indices-dir", required=True, help="peak_wise_splits.py indices files")
    ap.add_argument("--number-of-folds", type=int, default=20)
    ap.add_argument("--reference-genome", required=True)
    ap.add_argument("--input-seq-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--jitter", type=int, default=128)
    ap.add_argument("--n-mut", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    IN_LEN, OUT_LEN, JIT = args.input_seq_len, args.output_len, args.jitter
    teachers = sorted(args.teachers)
    print(f"{len(teachers)} teachers", flush=True)
    task = json.load(open(args.input_data))["0"]

    t0 = time.time()
    loci, ratio = load_pool(task, args.indices_dir, args.number_of_folds, args.seed, args.limit)
    seqs, ctls_wide, loci = extract_wide(loci, task, args.reference_genome, IN_LEN, JIT)
    print(f"pool: {int((loci.label == 1).sum())} peaks + {int((loci.label == 0).sum())} bg "
          f"(ratio {ratio}), extracted ({time.time() - t0:.0f}s)", flush=True)

    N = len(seqs)
    is_peak = loci["label"].values == 1
    c = JIT
    var_seqs, var_ctls, vtags = [], [], []
    var_seqs += [s[c:c + IN_LEN] for s in seqs]
    var_ctls.append(ctls_wide[:, :, c:c + IN_LEN])
    vtags += ["orig"] * N
    rc_tab = str.maketrans("ACGT", "TGCA")
    var_seqs += [s[c:c + IN_LEN].translate(rc_tab)[::-1] for s in seqs]
    var_ctls.append(ctls_wide[:, ::-1, ::-1][:, :, c:c + IN_LEN].copy())
    vtags += ["revcomp"] * N
    for k in range(args.n_mut):
        rng = random.Random(args.seed + 17 * k)
        nprng = np.random.RandomState(args.seed + 17 * k)
        shifts = nprng.randint(-JIT, JIT + 1, N)
        shifts[~is_peak] = 0
        sl_ctl = np.empty((N, ctls_wide.shape[1], IN_LEN), dtype=np.float32)
        for i in range(N):
            o = c + int(shifts[i])
            var_seqs.append(mutate_seq(seqs[i][o:o + IN_LEN], args.frac, rng))
            sl_ctl[i] = ctls_wide[i, :, o:o + IN_LEN]
        var_ctls.append(sl_ctl)
        vtags += [f"jitmut{k}"] * N
    del seqs, ctls_wide

    NV = len(var_seqs)
    X = np.empty((NV, 4, IN_LEN), dtype=np.int8)
    for i in range(0, NV, 50000):
        X[i:i + 50000] = one_hot(var_seqs[i:i + 50000])
    del var_seqs
    ctl = np.concatenate(var_ctls)
    del var_ctls
    print(f"{NV} windows ready ({time.time() - t0:.0f}s)", flush=True)

    import h5py
    import torch
    from cherimoya import Cherimoya

    logits_sum = np.zeros((NV, 2, OUT_LEN), dtype=np.float32)
    probs_sum = np.zeros((NV, 2, OUT_LEN), dtype=np.float32)
    t_sum = np.zeros(NV, dtype=np.float64)
    for ti, ck in enumerate(teachers):
        m = Cherimoya.load(ck, device="cuda")
        m.eval()
        with inference_mode():
            for s in range(0, NV, args.batch_size):
                e = min(s + args.batch_size, NV)
                xb = torch.from_numpy(X[s:e]).cuda().float()
                cb = torch.from_numpy(ctl[s:e]).cuda().float()
                lg, lc = m(xb, cb)
                lg = lg.detach().float()
                pr = torch.softmax(lg.reshape(e - s, -1), dim=1).reshape(e - s, 2, OUT_LEN)
                logits_sum[s:e] += lg.cpu().numpy()
                probs_sum[s:e] += pr.cpu().numpy()
                t_sum[s:e] += lc.detach().float().reshape(-1).cpu().numpy()
        del m
        torch.cuda.empty_cache()
        print(f"  teacher {ti + 1}/{len(teachers)} done ({time.time() - t0:.0f}s)", flush=True)

    K = len(teachers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("ohe", data=X, compression="gzip", chunks=(256, 4, IN_LEN))
        f.create_dataset("ctl", data=ctl.astype(np.float16), compression="gzip",
                         chunks=(256, ctl.shape[1], IN_LEN))
        f.create_dataset("logits_mean", data=(logits_sum / K).astype(np.float16),
                         compression="gzip", chunks=(256, 2, OUT_LEN))
        f.create_dataset("probs_mean", data=(probs_sum / K).astype(np.float16),
                         compression="gzip", chunks=(256, 2, OUT_LEN))
        f.create_dataset("t_mean", data=(t_sum / K).astype(np.float32))
        n_var = NV // len(loci)
        f.create_dataset("label", data=np.tile(loci["label"].values.astype(np.int8), n_var))
        f.create_dataset("chunk", data=np.tile(loci["chunk"].values.astype(np.int16), n_var))
        f.create_dataset("variant", data=np.array(vtags, dtype="S8"))
        f.attrs.update({"n_teachers": K, "n_mut": args.n_mut, "frac": args.frac, "jitter": JIT,
                        "seed": args.seed, "input_seq_len": IN_LEN, "output_len": OUT_LEN,
                        "teachers": json.dumps(teachers)})
    os.replace(tmp, args.out)
    print(f"saved {args.out}: {NV} windows, {K} teachers ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
