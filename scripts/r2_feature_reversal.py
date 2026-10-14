"""r2 特征级分析：每个网络中每个特征的单变量 AUC，以及每对网络在正常/攻击流量上的分布偏移。

用法（项目根目录）:
    python scripts/r2_feature_reversal.py
    python scripts/r2_feature_reversal.py --config configs/binary.yaml
输出: results/<run_name>/reversal/feature_auc.csv、feature_shift.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from src.common import load_config, path_of, resolve_n_jobs
from src.reversal import feature_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = path_of(cfg, "results") / "reversal"
    out.mkdir(parents=True, exist_ok=True)
    nj = min(resolve_n_jobs(cfg, args.n_jobs), len(cfg["seeds"]))
    res = Parallel(n_jobs=nj)(delayed(feature_level)(cfg, s) for s in cfg["seeds"])
    pd.DataFrame([r for a, _ in res for r in a]).to_csv(out / "feature_auc.csv", index=False)
    pd.DataFrame([r for _, b in res for r in b]).to_csv(out / "feature_shift.csv", index=False)
    print(f"[{cfg['run_name']}] 完成 -> {out}")


if __name__ == "__main__":
    main()
