"""r1 迁移矩阵：单源域（及 3 源域 LODO）训练 -> 在每个网络上测试；含类别互换分解。

用法（项目根目录，01_preprocess.py 之后）:
    python scripts/r1_transfer.py                              # 主协议 dos_v1
    python scripts/r1_transfer.py --config configs/binary.yaml # 对照协议 binary_v1
输出: results/<run_name>/reversal/transfer.csv（可中断续跑）
  test_kind: in_domain（训练网络 30% 留出） / cross / swap_target_attack / swap_target_benign
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from src.common import load_config, path_of, resolve_n_jobs
from src.reversal import TRANSFER_FIELDS, run_transfer_job, transfer_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = path_of(cfg, "results") / "reversal" / "transfer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out, usecols=["job_id"]).job_id) if out.exists() else set()
    jobs = [j for j in transfer_jobs(cfg) if j["job_id"] not in done]
    order = {"mlp": 0, "xgboost": 1, "lr": 2}
    jobs.sort(key=lambda j: (j["kind"] != "lodo", order.get(j["classifier"], 9)))
    nj = resolve_n_jobs(cfg, args.n_jobs)
    print(f"[{cfg['run_name']}] {len(done)} 个已完成，本次 {len(jobs)} 个模型；并行 {nj}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=nj, return_as="generator_unordered")(
        delayed(run_transfer_job)(cfg, j) for j in jobs)
    for i, rows in enumerate(res, 1):
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRANSFER_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        if i % 10 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] +{el:.0f}s，剩余约 {el / i * (len(jobs) - i) / 60:.0f} 分钟",
                  flush=True)
    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
