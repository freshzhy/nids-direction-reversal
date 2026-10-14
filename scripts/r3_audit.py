"""r3 数据审计：原始记录数 vs 特征完全相同的记录数（三种列集合）。

用法（项目根目录）: python scripts/r3_audit.py
输出: results/audit.csv（与协议无关，只需跑一次）
  unique_all_columns      : 含 IP/端口/DNS_QUERY_ID 在内全部列都相同才算重复（真正的重复记录）
  unique_no_identifiers   : 去掉 IP/端口后
  unique_no_identifiers_dns: 再去掉 DNS_QUERY_ID 后（= 主协议使用的特征空间 + Label + Attack）
每个数据集读一遍 CSV，4 个并行。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common import load_config, path_of

IDS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]


def audit_one(cfg, code):
    name = cfg["datasets"][code]
    csv_path = path_of(cfg, "raw_root") / name / "data" / f"{name}.csv"
    sets = {"all_columns": None, "no_identifiers": IDS, "no_identifiers_dns": IDS + ["DNS_QUERY_ID"]}
    hashes = {k: [] for k in sets}
    labels = []
    n = 0
    t0 = time.time()
    for ch in pd.read_csv(csv_path, chunksize=cfg["preprocess"]["chunksize"], low_memory=False):
        n += len(ch)
        labels.append(ch["Label"].to_numpy())
        for k, drop in sets.items():
            sub = ch if drop is None else ch.drop(columns=drop)
            hashes[k].append(pd.util.hash_pandas_object(sub, index=False).to_numpy(np.uint64))
    lab = np.concatenate(labels)
    row = {"dataset": code, "name": name, "rows_raw": n, "attack_ratio_raw": float(lab.mean())}
    for k in sets:
        u = len(np.unique(np.concatenate(hashes[k])))
        row[f"unique_{k}"] = u
        row[f"duplicate_rate_{k}"] = 1 - u / n
    row["seconds"] = round(time.time() - t0, 1)
    print(row, flush=True)
    return row


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = Parallel(n_jobs=4)(delayed(audit_one)(cfg, c) for c in cfg["datasets"])
    base = path_of(cfg, "results")
    out = (base.parent if cfg.get("run_name") else base) / "audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
