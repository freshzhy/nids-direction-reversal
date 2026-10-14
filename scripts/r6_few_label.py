"""r6 少量目标域标签估计方向（实用性补充，不是新方法）。

问题：不知道目标域全部标签时，用 n 条有标签的目标流量估计各特征的判别方向，
      再做“边缘对齐 + 反序”，能恢复多少？与 oracle（全部目标标签）相比差多少？
设计（预先固定）：
  - 每个 n ∈ reversal.few_n、每个种子抽 reversal.few_draws 次校准集（每类 n/2）
  - 方向只在校准集上估计；评估集 = 目标样本（最多 few_eval_max 行）去掉校准集
  - 源模型与 r1/r5 完全相同；阈值与主分析相同（flip_hi / flip_lo）
  - 必须的基线：直接用同样 n 条目标标签训练模型（few_target_only），在同一评估集上测
用法（项目根目录，r1、r2 之后）:
    python scripts/r6_few_label.py
    python scripts/r6_few_label.py --config configs/binary.yaml
输出: results/<run_name>/reversal/fewlabel.csv（按任务续跑：已完成的任务整体跳过）
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

from src.common import dataset_codes, load_config, path_of, resolve_n_jobs
from src.reversal import (FEW_FIELDS, alignment_jobs, run_fewlabel_target_only_job,
                          run_fewlabel_transfer_job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = path_of(cfg, "results") / "reversal" / "fewlabel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out, usecols=["kind", "train", "target", "classifier", "seed"])
        tr = o[o.kind == "direction_correction"]
        done |= {("T", a, c, int(s)) for a, c, s in zip(tr.train, tr.classifier, tr.seed)}
        to = o[o.kind == "target_only"]
        done |= {("B", t, int(s)) for t, s in zip(to.target, to.seed)}
    jobs = [("T", j) for j in alignment_jobs(cfg)
            if ("T", j["train"], j["classifier"], j["seed"]) not in done]
    jobs += [("B", {"target": t, "seed": int(s)}) for t in dataset_codes(cfg) for s in cfg["seeds"]
             if ("B", t, int(s)) not in done]
    nj = resolve_n_jobs(cfg, args.n_jobs)
    print(f"[{cfg['run_name']}] 本次 {len(jobs)} 个任务；并行 {nj}", flush=True)
    t0 = time.time()
    fn = {"T": run_fewlabel_transfer_job, "B": run_fewlabel_target_only_job}
    res = Parallel(n_jobs=nj, return_as="generator_unordered")(
        delayed(fn[k])(cfg, j) for k, j in jobs)
    for i, rows in enumerate(res, 1):
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FEW_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        if i % 10 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)
    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
