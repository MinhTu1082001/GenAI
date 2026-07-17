#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py — sinh toàn bộ hình cho báo cáo từ log đã ghi sẵn.

    python plots.py runs/a_seq2seq runs/b_cvae1 runs/c_cvae2

Mỗi run → runs/<run>/plots/ : fig1_loss, fig2_gap, fig3_elbo, fig4_kl_dims, fig7_tsne
Chung    → plots_compare/   : fig0_compare (chrF+F1), fig5_rd (rate–distortion),
                              fig6_iwae (bound theo K, cần chạy evaluate.py trước)

Map hình → lý thuyết (caption gợi ý nằm trong tiêu đề mỗi hình):
  fig1/2: tối ưu & generalization gap (GF §8.1, §7.8; Mohri Ch.2–3)
  fig3  : phân rã ELBO + lịch β(t) (GF §19.1; Bowman 16)
  fig4  : KL từng chiều + active units (Kingma 16; Burda 15)
  fig5  : mặt phẳng rate–distortion (β-VAE; Alemi 18)
  fig6  : IWAE bound tăng theo K (Burda 15; GF §17.2)
  fig7  : t-SNE của μ tô màu theo level JLPT
"""

from __future__ import annotations
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def read_metrics(run: Path):
    tr, de = [], []
    with open(run / "metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {k: fnum(v) for k, v in r.items() if k != "phase"}
            (tr if r["phase"] == "train" else de).append(row)
    return tr, de


def series(rows, x, y):
    xs, ys = [], []
    for r in rows:
        if r.get(x) is not None and r.get(y) is not None and not np.isnan(r[y]):
            xs.append(r[x]); ys.append(r[y])
    return xs, ys


def newest_eval(run: Path, name: str) -> Path | None:
    cands = sorted(run.glob(f"eval_*/{name}"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


# ------------------------------------------------------------ hình theo run
def per_run(run: Path):
    out = run / "plots"; out.mkdir(exist_ok=True)
    tr, de = read_metrics(run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))

    # fig1 — loss theo bước (tối ưu hóa)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, lab in (("ce_trans", "CE dịch/token"), ("ce_gram", "CE ngữ pháp/token")):
        xs, ys = series(tr, "step", key)
        if ys: ax.plot(xs, ys, label=lab)
    ax.set_xlabel("bước"); ax.set_ylabel("CE/token"); ax.legend(loc="upper right")
    xs, ys = series(tr, "step", "kl")
    if ys:
        ax2 = ax.twinx(); ax2.plot(xs, ys, color="tab:red", alpha=0.5, label="KL")
        ax2.set_ylabel("KL (nats)", color="tab:red")
    ax.set_title(f"{run.name} — loss theo bước train")
    fig.tight_layout(); fig.savefig(out / "fig1_loss.png", dpi=150); plt.close(fig)

    # fig2 — train vs dev (generalization gap)
    ep_ce = {}
    for r in tr:
        if r.get("epoch") is not None and r.get("ce_trans") is not None:
            ep_ce.setdefault(int(r["epoch"]), []).append(r["ce_trans"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if ep_ce:
        eps = sorted(ep_ce)
        ax.plot(eps, [np.mean(ep_ce[e]) for e in eps], label="train CE dịch")
    xs, ys = series(de, "epoch", "ce_trans"); ax.plot(xs, ys, label="dev CE dịch")
    xs, ys = series(de, "epoch", "ce_gram")
    if ys: ax.plot(xs, ys, "--", label="dev CE ngữ pháp")
    ax.set_xlabel("epoch"); ax.set_ylabel("CE/token"); ax.legend()
    ax.set_title(f"{run.name} — generalization gap")
    fig.tight_layout(); fig.savefig(out / "fig2_gap.png", dpi=150); plt.close(fig)

    # fig3 — phân rã ELBO + β(t)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, lab in (("ce_trans", "CE dịch"), ("ce_gram", "CE ngữ pháp"), ("kl", "KL")):
        xs, ys = series(tr, "step", key)
        if ys: ax.plot(xs, ys, label=lab)
    ax.set_xlabel("bước"); ax.set_ylabel("nats"); ax.legend(loc="upper right")
    xs, ys = series(tr, "step", "beta")
    if ys:
        ax2 = ax.twinx(); ax2.plot(xs, ys, ":", color="gray"); ax2.set_ylabel("β(t)")
        ax2.set_ylim(0, 1.05)
    ax.set_title(f"{run.name} — phân rã ELBO (KL khỏe: plateau vài nats, không về 0)")
    fig.tight_layout(); fig.savefig(out / "fig3_elbo.png", dpi=150); plt.close(fig)

    # fig4 — heatmap KL từng chiều + AU
    kfile = run / "kl_dims.csv"
    if kfile.exists():
        rows = list(csv.reader(open(kfile, encoding="utf-8")))[1:]
        if rows:
            arr = np.array([[float(v) for v in r[1:]] for r in rows])
            eps = [int(float(r[0])) for r in rows]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4),
                                         gridspec_kw={"width_ratios": [3, 1]})
            im = a1.imshow(arr, aspect="auto", origin="lower", cmap="viridis",
                           extent=[0, arr.shape[1], eps[0], eps[-1]])
            a1.set_xlabel("chiều z"); a1.set_ylabel("epoch")
            a1.set_title(f"KL từng chiều (free bits λ={cfg.get('free_bits')})")
            fig.colorbar(im, ax=a1, label="nats")
            xs, ys = series(de, "epoch", "au")
            a2.plot(xs, ys); a2.set_xlabel("epoch"); a2.set_ylabel("active units")
            a2.set_title(f"AU / {cfg.get('d_z')} chiều")
            fig.tight_layout(); fig.savefig(out / "fig4_kl_dims.png", dpi=150); plt.close(fig)

    # fig7 — t-SNE của μ theo level
    mfile = newest_eval(run, "mu_dump.csv")
    if mfile:
        recs = list(csv.DictReader(open(mfile, encoding="utf-8")))
        if len(recs) >= 10:
            X = np.array([[float(r[k]) for k in r if k.startswith("m")] for r in recs])
            lv = [r["level"] for r in recs]
            try:
                from sklearn.manifold import TSNE
                P = TSNE(n_components=2, perplexity=min(30, max(5, (len(X) - 1) // 3)),
                         init="pca", random_state=13).fit_transform(X)
                method = "t-SNE"
            except Exception:
                Xc = X - X.mean(0)
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                P = Xc @ Vt[:2].T; method = "PCA"
            fig, ax = plt.subplots(figsize=(6, 5))
            for name, color in (("N5", "tab:blue"), ("N4", "tab:orange"), ("none", "gray")):
                m = [i for i, l in enumerate(lv) if l == name]
                if m: ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.7, label=name, c=color)
            ax.legend(); ax.set_title(f"{run.name} — {method} của μ theo level ({mfile.parent.name})")
            fig.tight_layout(); fig.savefig(out / "fig7_tsne.png", dpi=150); plt.close(fig)

    print(f"✓ {out}/")


# ------------------------------------------------------------ hình so sánh
def compare(runs: list[Path]):
    out = Path("plots_compare"); out.mkdir(exist_ok=True)

    # fig0 — dev chrF + F1 theo epoch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        _, de = read_metrics(run)
        xs, ys = series(de, "epoch", "chrf"); a1.plot(xs, ys, label=run.name)
        xs, ys = series(de, "epoch", "f1_micro")
        if ys: a2.plot(xs, ys, label=run.name)
    a1.set_xlabel("epoch"); a1.set_ylabel("dev chrF"); a1.legend(); a1.set_title("chrF")
    a2.set_xlabel("epoch"); a2.set_ylabel("dev F1 micro"); a2.legend(); a2.set_title("F1 ngữ pháp")
    fig.tight_layout(); fig.savefig(out / "fig0_compare.png", dpi=150); plt.close(fig)

    # fig5 — mặt phẳng rate–distortion (quỹ đạo dev qua các epoch)
    fig, ax = plt.subplots(figsize=(6, 5))
    for run in runs:
        _, de = read_metrics(run)
        pts = [(r["kl"], r["ce_trans"]) for r in de
               if r.get("kl") is not None and r.get("ce_trans") is not None]
        if pts:
            ks, cs = zip(*pts)
            ax.plot(ks, cs, "-o", ms=3, alpha=0.8, label=run.name)
            ax.annotate("cuối", (ks[-1], cs[-1]), fontsize=8)
    ax.set_xlabel("Rate — KL (nats)"); ax.set_ylabel("Distortion — CE dịch/token")
    ax.legend(); ax.set_title("Mặt phẳng rate–distortion (quỹ đạo huấn luyện)")
    fig.tight_layout(); fig.savefig(out / "fig5_rd.png", dpi=150); plt.close(fig)

    # fig6 — IWAE bound theo K (cần evaluate.py chạy trước)
    fig, ax = plt.subplots(figsize=(6, 4)); has = False
    for run in runs:
        f = newest_eval(run, "iwae_by_k.csv")
        if f:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            ax.plot([int(r["K"]) for r in rows],
                    [float(r["bound_per_token"]) for r in rows], "-o", label=run.name)
            has = True
    if has:
        ax.set_xscale("log", base=2); ax.set_xlabel("K"); ax.set_ylabel("bound/token (nats)")
        ax.legend(); ax.set_title("IWAE bound tăng đơn điệu theo K (Burda 15)")
        fig.tight_layout(); fig.savefig(out / "fig6_iwae.png", dpi=150)
    plt.close(fig)
    print(f"✓ {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Dùng: python plots.py runs/<run1> [runs/<run2> ...]")
    run_dirs = [Path(p) for p in sys.argv[1:]]
    for r in run_dirs:
        per_run(r)
    compare(run_dirs)
