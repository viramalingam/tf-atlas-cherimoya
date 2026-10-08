#!/usr/bin/env python3
"""Distilled cherimoya student trained on the teacher-ensemble targets.

Port of tfatlas/eval/src/cheri_distill_student.py (bpnet-arch-benchmark, the
CHD1_student arm) with the experiment-specific options dropped. One fresh
Cherimoya with the teachers' config is fit to the ensemble-mean soft targets
with cherimoya's own losses: soft-target multinomial cross entropy over the
joint (2 x output_len) profile + count_weight * MSE on the teacher log-count
scalar. Optimizer recipe = cherimoya v0.2.0's: Muon over the CheriBlock
projection weights, AdamW over the rest, the loss-weight scalars frozen,
LinearLR warmup -> cosine, stepped per iteration.

Validation = chunk --val-chunk windows (never trained on); early stopping on
the validation loss (patience --patience). The best checkpoint is saved with
model.save() as <name>_split000.torch, so cherimoya_predict.py can load it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cherimoya_predict import inference_mode  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

MUON_LR, MUON_WD = 0.025, 0.03
ADAM_LR, ADAM_WD = 0.001, 0.0
WARMUP_EPOCHS = 5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="cherimoya_distill_targets.py h5")
    ap.add_argument("--teacher", required=True, help="one teacher checkpoint (supplies the config)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-output-filename", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--count-weight", type=float, default=10.0)
    ap.add_argument("--val-chunk", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--prob-avg", action="store_true", help="use the averaged probabilities instead of logits")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.join(args.output_dir, f"{args.model_output_filename}_split000")

    import h5py
    import torch
    from torch.nn.functional import log_softmax
    from torch.optim import Muon
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from cherimoya import Cherimoya

    t0 = time.time()
    with h5py.File(args.targets, "r") as f:
        sl = slice(0, args.limit) if args.limit else slice(None)
        X = torch.from_numpy(f["ohe"][sl])
        C = torch.from_numpy(f["ctl"][sl])
        PT = torch.from_numpy(f["probs_mean" if args.prob_avg else "logits_mean"][sl])
        TM = torch.from_numpy(f["t_mean"][sl]).float()
        chunk = f["chunk"][sl]
        label = f["label"][sl]
        variant = f["variant"][sl]
        n_teachers = int(f.attrs["n_teachers"])

    N = len(X)
    is_val = chunk == args.val_chunk
    tr_idx = torch.from_numpy(np.where(~is_val)[0])
    va_idx = torch.from_numpy(np.where(is_val)[0])
    print(f"targets: N={N} train={len(tr_idx)} val(chunk{args.val_chunk})={len(va_idx)} "
          f"teachers={n_teachers} ({time.time() - t0:.0f}s load)", flush=True)

    payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    config["name"] = name
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    student = Cherimoya(**config).cuda()
    print(f"student config: n_filters={config.get('n_filters')} n_layers={config.get('n_layers')}",
          flush=True)

    muon_params, adam_params = [], []
    for pname, p in student.named_parameters():
        if pname in ("lw0", "lw1"):
            p.requires_grad = False
        elif (p.ndim == 2 and "weight" in pname and pname != "linear.weight"
              and "conv_weight" not in pname):
            muon_params.append(p)
        else:
            adam_params.append(p)
    bs = args.batch_size
    nbatches = (len(tr_idx) + bs - 1) // bs
    n_warm = nbatches * WARMUP_EPOCHS
    n_decay = nbatches * max(1, args.epochs - WARMUP_EPOCHS)

    def sched(opt):
        return SequentialLR(opt, schedulers=[
            LinearLR(opt, start_factor=0.01, total_iters=n_warm),
            CosineAnnealingLR(opt, T_max=n_decay, eta_min=1e-5)], milestones=[n_warm])

    muon_opt = Muon(muon_params, lr=MUON_LR, weight_decay=MUON_WD)
    adam_opt = torch.optim.AdamW(adam_params, lr=ADAM_LR, weight_decay=ADAM_WD)
    muon_sched, adam_sched = sched(muon_opt), sched(adam_opt)
    print(f"Muon over {len(muon_params)} weights, AdamW over {len(adam_params)}; warmup "
          f"{WARMUP_EPOCHS}ep -> cosine {args.epochs}ep; count_weight={args.count_weight}", flush=True)

    def q_of(idx):
        flat = PT[idx].cuda().float().reshape(len(idx), -1)
        if args.prob_avg:
            return flat / flat.sum(dim=1, keepdim=True)
        return torch.softmax(flat, dim=1)

    def loss_on(idx):
        lg, lc = student(X[idx].cuda().float(), C[idx].cuda().float())
        sl_ = log_softmax(lg.float().reshape(len(idx), -1), dim=1)
        prof = -(q_of(idx) * sl_).sum(dim=1).mean()
        cnt = ((lc.float().reshape(-1) - TM[idx].cuda()) ** 2).mean()
        return prof + args.count_weight * cnt, prof.item(), cnt.item()

    best, bad = float("inf"), 0
    hist = []
    for ep in range(args.epochs):
        student.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tl = tp = tc = nb = 0
        for i in range(0, len(perm), bs):
            loss, p_, c_ = loss_on(perm[i:i + bs])
            muon_opt.zero_grad()
            adam_opt.zero_grad()
            loss.backward()
            muon_opt.step()
            adam_opt.step()
            muon_sched.step()
            adam_sched.step()
            tl += loss.item(); tp += p_; tc += c_; nb += 1
        student.eval()
        vl = vnb = 0
        with inference_mode():
            for i in range(0, len(va_idx), bs):
                loss, _, _ = loss_on(va_idx[i:i + bs])
                vl += loss.item(); vnb += 1
        vloss = vl / max(vnb, 1)
        hist.append({"epoch": ep, "train": tl / nb, "train_profile": tp / nb,
                     "train_count": tc / nb, "val": vloss})
        print(f"ep{ep:03d} train {tl / nb:.4f} (prof {tp / nb:.4f} cnt {tc / nb:.4f}) "
              f"val {vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            student.save(name + ".torch")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break

    # T-free selection statistic: validation-chunk peaks, original windows,
    # student vs teacher-ensemble log counts
    best_sd = torch.load(name + ".torch", map_location="cpu", weights_only=False)
    student.load_state_dict(best_sd["state_dict"])
    student.cuda().eval()
    sel = np.where(is_val & (label > 0) & (variant == b"orig"))[0]
    preds = []
    with inference_mode():
        for i in range(0, len(sel), bs):
            b = torch.from_numpy(sel[i:i + bs])
            _, lc = student(X[b].cuda().float(), C[b].cuda().float())
            preds.append(lc.detach().float().reshape(-1).cpu())
    pv = torch.cat(preds).numpy() if preds else np.array([])
    tv = TM[torch.from_numpy(sel)].numpy()
    sel_r = float(np.corrcoef(pv, tv)[0, 1]) if len(sel) > 2 else None

    with open(name + ".history.json", "w") as fp:
        json.dump(hist, fp)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as fp:
        json.dump({"seed": args.seed, "best_val_loss": best, "n_epochs_run": len(hist),
                   "targets_h5": os.path.basename(args.targets), "n_teachers": n_teachers,
                   "count_weight": args.count_weight, "prob_avg": bool(args.prob_avg),
                   "val_chunk": args.val_chunk, "patience": args.patience,
                   "valchunk_peak_count_pearson_vs_teachers": sel_r,
                   "elapsed_sec": time.time() - t0, "model": name + ".torch",
                   "teacher_config_from": args.teacher}, fp, indent=2)
    print(f"DONE: best val {best:.4f}, {len(hist)} epochs, val-chunk count r vs teachers "
          f"{sel_r}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
