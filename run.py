#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — chạy trọn pipeline model → eval → plots → demo theo config.ini.

    python run.py                 # đọc config.ini
    python run.py khac.ini        # đọc file config khác

Nó chỉ DỊCH config.ini thành các lệnh train.py/evaluate.py/plots.py/demo.py
rồi gọi lần lượt — dừng ngay nếu một bước lỗi. Không đụng bước dữ liệu
(download/prepare/tagger/split/synthetic): chạy tay theo huong_dan.MD.
"""
from __future__ import annotations
import configparser, os, shutil, subprocess, sys
from pathlib import Path

if sys.platform == "win32":                      # tránh bẫy cp1252 của Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_cfg(path: Path) -> configparser.ConfigParser:
    if not path.exists():
        sys.exit(f"Không thấy {path} — đặt config cạnh run.py hoặc truyền đường dẫn.")
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read(path, encoding="utf-8")
    return cp


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONUTF8="1")       # ép UTF-8 cho tiến trình con
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"‼ Bước lỗi (mã {r.returncode}) — dừng pipeline.")


def opt(args: list[str], flag: str, val) -> None:
    """Thêm cờ nếu giá trị không rỗng."""
    if val is not None and str(val).strip() != "":
        args += [flag, str(val).strip()]


def main() -> None:
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.ini")
    cp = load_cfg(cfg_path)
    py = sys.executable                          # python trong venv hiện tại
    pipe = cp["pipeline"]

    variants = [v.strip() for v in cp["train"].get("variants", "").split(",") if v.strip()]
    if not variants:
        sys.exit("[train] variants trống — cần ít nhất một biến thể.")
    run_dirs = [f"runs/{v}" for v in variants]
    print(f"Config: {cfg_path} | biến thể: {', '.join(variants)}")

    # ---- 0. sanity overfit (tùy chọn) ----
    if pipe.getboolean("sanity", fallback=False):
        s = cp["sanity"]
        run([py, "train.py", "--overfit", s.get("n", "50"),
             "--epochs", s.get("epochs", "300"), "--run", "sanity"])

    # ---- 1. train tam giác ablation ----
    if pipe.getboolean("train", fallback=True):
        t = cp["train"]
        if t.getboolean("fresh_artifacts", fallback=True):
            shutil.rmtree("artifacts", ignore_errors=True)
            print("→ đã xóa artifacts/ — vocab build lại trên run đầu, chung cho cả 3.")
        for v in variants:
            args = [py, "train.py", "--variant", v, "--run", v]
            if t.getint("limit", fallback=0):        # 0 = dùng toàn bộ train
                opt(args, "--limit", t.get("limit"))
            for flag, key in (("--epochs", "epochs"), ("--dev-eval-n", "dev_eval_n"),
                              ("--batch", "batch"), ("--lr", "lr"), ("--seed", "seed"),
                              ("--lam", "lam"), ("--free-bits", "free_bits"),
                              ("--p-wd", "p_wd"), ("--beta-max", "beta_max"),
                              ("--warmup-frac", "warmup_frac"), ("--ls-trans", "ls_trans"),
                              ("--patience", "patience"), ("--device", "device")):
                opt(args, flag, t.get(key, ""))
            run(args)

    # ---- 2. evaluate ----
    if pipe.getboolean("evaluate", fallback=True):
        e = cp["evaluate"]
        args = [py, "evaluate.py"] + run_dirs
        opt(args, "--split", e.get("split", "dev"))
        opt(args, "--iwae-k", e.get("iwae_k", "64"))
        run(args)

    # ---- 3. plots ----
    if pipe.getboolean("plots", fallback=True):
        run([py, "plots.py"] + run_dirs)

    # ---- 4. demo ----
    if pipe.getboolean("demo", fallback=True):
        d = cp["demo"]
        rname = d.get("run", "cvae2").strip()
        run([py, "demo.py", "--run", f"runs/{rname}", d.get("sentence", "").strip()])

    print("\n✅ Xong toàn bộ pipeline theo config.ini.")


if __name__ == "__main__":
    main()
