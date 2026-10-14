"""01 预处理：4 个 NFv2 CSV -> data/processed/<run_name>/{代号}_pool.parquet

去重后按 config 的 task 映射标签（dos_family：只有 DoS/DDoS 为正类，其他攻击剔除；
binary：所有攻击为正类），每类最多保留 pool_per_class 条去重样本作为“样本池”。
每个种子的 1:1 样本与原始比例样本在加载时从池中抽取（src/common.py: load_dataset）。

用法（项目根目录）:
    python scripts/01_preprocess.py                                   # 主配置
    python scripts/01_preprocess.py --config configs/sens_dnsid.yaml  # 保留 DNS_QUERY_ID 的敏感性版本
    python scripts/01_preprocess.py --only UN
输出（data/processed/<run_name>/）:
    {UN,CIC,TON,BOT}_pool.parquet
    meta.json           每个数据集的原始正类比例（去重后）等，用于原始比例抽样
    features.csv        特征清单（名称、变换类型、说明）
    stats.csv           每步删除行数等（论文数据表）
    attack_types.csv    去重后与池中的攻击类型分布
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from src.common import (find_csv, library_versions, load_config, path_of, read_json,
                        task_family, write_json)
from src import preprocess as P


def header_features(csv_path: Path, cfg: dict) -> list[str]:
    """Header minus identifiers, label/attack, and anything matching drop_patterns
    (e.g. absolute timestamps such as FLOW_START_MILLISECONDS in NetFlow v3)."""
    import re
    pp = cfg["preprocess"]
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    drop = set(pp["drop_identifiers"]) | {pp["label_col"], pp["attack_col"]}
    pats = [re.compile(x) for x in pp.get("drop_patterns", [])]
    return [c for c in cols if c not in drop and not any(r.search(c) for r in pats)]


def log1p_list(cfg: dict, feats: list[str]) -> list[str]:
    """Explicit list, or (log1p_auto) every feature that is not categorical / kept raw."""
    pp = cfg["preprocess"]
    if pp.get("log1p_auto"):
        keep = set(pp["categorical_features"]) | set(pp.get("raw_features", []))
        return [f for f in feats if f not in keep]
    return list(pp["log1p_features"])


def process_one(code: str, cfg: dict) -> dict:
    pp = cfg["preprocess"]
    name = cfg["datasets"][code]
    csv_path = find_csv(path_of(cfg, "raw_root"), name)
    out_dir = path_of(cfg, "processed")
    feats = header_features(csv_path, cfg)
    log = lambda m: print(f"[{code}] {m}", flush=True)
    t0 = time.time()
    log(f"pass 1: {csv_path}")
    sc = P.scan(csv_path, feats, pp["label_col"], pp["attack_col"], pp["chunksize"], log=log)
    fam = task_family(cfg, code, available=list(sc["attack_names"]))
    log(f"positive attack names: {'ALL' if fam is None else fam}")
    ds = P.dedup_and_pool(sc, fam, pp["pool_per_class"], pp["sample_seed"])
    neg = sc["neg_counts"]
    del sc
    log(f"pass 2: extracting pool of {len(ds['pool_idx']):,} rows")
    parts = P.extract(csv_path, feats, pp["label_col"], pp["attack_col"], pp["chunksize"],
                      {"pool": ds["pool_idx"]}, log=log)
    out = P.transform(parts["pool"], feats, log1p_list(cfg, feats), pp["label_col"],
                      pp["attack_col"], label=ds["pool_label"])
    out.to_parquet(out_dir / f"{code}_pool.parquet", index=False)
    attack_rows = [{"dataset": code, "set": "pool", "attack": a, "n": int(n)}
                   for a, n in out[pp["attack_col"]].value_counts().items()]
    attack_rows += [{"dataset": code, "set": "unique_in_task", "attack": a, "n": int(n)}
                    for a, n in ds["u_attack_counts"].items()]
    st = {"dataset": code, "name": name, **ds["stats"],
          "n_features": len(feats), "seconds": round(time.time() - t0, 1)}
    neg_feats = {f: int(v) for f, v in neg.items() if v > 0}
    st["negative_values_clipped"] = ";".join(f"{f}:{v}" for f, v in neg_feats.items())
    log(f"done in {st['seconds']}s: {ds['stats']}")
    st["positive_attack_names"] = "ALL" if fam is None else ";".join(fam)
    return {"stats": st, "attacks": attack_rows, "features": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="dataset codes, e.g. UN CIC")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    codes = args.only or list(cfg["datasets"])
    out_dir = path_of(cfg, "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = Parallel(n_jobs=min(args.n_jobs, len(codes)))(delayed(process_one)(c, cfg) for c in codes)

    feats = res[0]["features"]
    for r in res[1:]:
        assert set(r["features"]) == set(feats), (
            "datasets do not share an identical feature set: "
            f"{sorted(set(r['features']) ^ set(feats))}")

    # feature list (identical across datasets)
    name0 = cfg["datasets"][codes[0]]
    cands = sorted((path_of(cfg, "raw_root") / name0).rglob("*eature*.csv"))
    desc_path = cands[0] if cands else Path("/nonexistent")
    desc = {}
    if desc_path.exists():
        d = pd.read_csv(desc_path, encoding="latin-1")
        desc = dict(zip(d.iloc[:, 0].str.strip(), d.iloc[:, 1]))
    pp = cfg["preprocess"]
    lf = set(log1p_list(cfg, feats))
    ftype = lambda f: ("log1p" if f in lf
                       else "categorical" if f in pp["categorical_features"] else "numeric_raw")
    pd.DataFrame({"feature": feats, "transform": [ftype(f) for f in feats],
                  "description": [desc.get(f, "") for f in feats]}).to_csv(
        out_dir / "features.csv", index=False)

    # merge stats with any existing rows for datasets not re-run
    meta_p = out_dir / "meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {}
    for r in res:
        st = r["stats"]
        meta[st["dataset"]] = {"raw_positive_ratio": st["raw_positive_ratio"],
                               "benign_unique": st["benign_unique"],
                               "positive_unique": st["positive_unique"],
                               "task": cfg.get("task", "binary")}
    write_json(meta_p, meta)
    stats = pd.DataFrame([r["stats"] for r in res])
    sp = out_dir / "stats.csv"
    if sp.exists():
        old = pd.read_csv(sp)
        stats = pd.concat([old[~old["dataset"].isin(stats["dataset"])], stats])
    stats.to_csv(sp, index=False)
    at = pd.DataFrame([a for r in res for a in r["attacks"]])
    ap_ = out_dir / "attack_types.csv"
    if ap_.exists():
        old = pd.read_csv(ap_)
        at = pd.concat([old[~old["dataset"].isin(codes)], at])
    at.to_csv(ap_, index=False)
    res_dir = path_of(cfg, "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    write_json(res_dir / "env.json", library_versions())
    print(stats.drop(columns=["negative_values_clipped"]).to_string(index=False))
    print(f"\n写入 {out_dir}")


if __name__ == "__main__":
    main()
