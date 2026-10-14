"""r5 方向对齐诊断（oracle，不是方法）：特征级方向反转能否解释模型级反转？

对每个单源域模型（与 r1 相同的划分与模型），在目标域上比较 5 种输入：
  base            原始目标域特征
  flip            只把“方向相反”的特征在目标域自身分布内反序（u -> 1-u）
  qmap            无标签边缘分布对齐：目标特征按自身 ECDF 映射到源域边缘分布
  qmap_flip       对齐 + 反序（oracle：反序哪些特征由 r2 的目标域标签决定）
  qmap_randflip   对照：反序同样数量、但方向并未相反的特征（5 次随机平均）
解读：若 qmap 不能恢复而 qmap_flip 能恢复、且 randflip 不能恢复，
      说明模型级反转主要由特征–标签方向反转造成，而不是边缘分布偏移。
同时记录源域/目标域正常与攻击流量的模型得分中位数（得分位置）。

用法（项目根目录，r1、r2 之后）:
    python scripts/r5_alignment.py
    python scripts/r5_alignment.py --config configs/binary.yaml
    python scripts/r5_alignment.py --sens        # 方向阈值敏感性（config: reversal.sensitivity）
输出: results/<run_name>/reversal/alignment.csv、alignment_sens.csv（可中断续跑）
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
from src.reversal import (ALIGN_FIELDS, SENS_FIELDS, alignment_jobs, robust_directions,
                          run_alignment_job, run_alignment_sens_job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--sens", action="store_true", help="threshold sensitivity run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rv = cfg["reversal"]
    d = path_of(cfg, "results") / "reversal"
    fa = pd.read_csv(d / "feature_auc.csv")
    _, Dir = robust_directions(fa, rv["flip_hi"], rv["flip_lo"])
    if args.sens:
        dirs = {(float(h), float(l)): robust_directions(fa, h, l)[1] for h, l in rv["sensitivity"]}
        fields, worker, out = SENS_FIELDS, run_alignment_sens_job, d / "alignment_sens.csv"
        payload = dirs
    else:
        fields, worker, out = ALIGN_FIELDS, run_alignment_job, d / "alignment.csv"
        payload = Dir
    done = set(pd.read_csv(out, usecols=["job_id"]).job_id) if out.exists() else set()
    jobs = [j for j in alignment_jobs(cfg) if j["job_id"] not in done]
    jobs.sort(key=lambda j: {"mlp": 0, "xgboost": 1}.get(j["classifier"], 2))
    nj = resolve_n_jobs(cfg, args.n_jobs)
    print(f"[{cfg['run_name']}] 已完成 {len(done)}，本次 {len(jobs)} 个模型；并行 {nj}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=nj, return_as="generator_unordered")(
        delayed(worker)(cfg, j, payload) for j in jobs)
    for i, rows in enumerate(res, 1):
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerows(rows)
        if i % 10 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)
    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
