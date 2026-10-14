"""Streaming preprocessing of the NFv2 CSVs (two passes, bounded memory).

Pass 1: read chunks, drop identifiers, flag inf/NaN rows, hash every valid row
        (features+Label+Attack, and features only) -> global dedup + label-conflict stats.
Pass 2: re-read chunks and keep only the sampled row indices, apply log1p.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


def _read_chunks(csv_path: Path, usecols: list[str], dtypes: dict, chunksize: int):
    return pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, chunksize=chunksize,
                       low_memory=False)


def _hash_rows(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)


def scan(csv_path: Path, feats: list[str], label_col: str, attack_col: str,
         chunksize: int, log=print) -> dict:
    """Pass 1. Returns arrays over valid (finite) rows + counts."""
    usecols = feats + [label_col, attack_col]
    dtypes = {f: "float64" for f in feats}
    dtypes[label_col] = "float64"  # tolerate missing labels; cast after filtering
    dtypes[attack_col] = "string"
    idx_l, hfull_l, hfeat_l, lab_l, att_l = [], [], [], [], []
    att_map: dict[str, int] = {}
    n_total = n_bad = 0
    neg_counts = pd.Series(0, index=feats, dtype="int64")
    offset = 0
    t0 = time.time()
    for ch in _read_chunks(csv_path, usecols, dtypes, chunksize):
        n = len(ch)
        vals = ch[feats].to_numpy()
        finite = np.isfinite(vals).all(axis=1) & ch[label_col].notna().to_numpy() \
            & ch[attack_col].notna().to_numpy()
        n_total += n
        n_bad += int((~finite).sum())
        good = ch.loc[finite]
        neg_counts += (good[feats] < 0).sum()
        hfeat = _hash_rows(good[feats])
        hfull = _hash_rows(good[feats + [label_col, attack_col]])
        idx_l.append(np.flatnonzero(finite) + offset)
        hfeat_l.append(hfeat)
        hfull_l.append(hfull)
        lab_l.append(good[label_col].to_numpy(dtype=np.int8))
        a = good[attack_col].astype(str).to_numpy()
        for name in pd.unique(a):
            att_map.setdefault(name, len(att_map))
        att_l.append(pd.Series(a).map(att_map).to_numpy(dtype=np.int16))
        offset += n
        log(f"  [{csv_path.name}] scanned {offset:,} rows ({time.time() - t0:.0f}s)")
    return {
        "idx": np.concatenate(idx_l), "hfull": np.concatenate(hfull_l),
        "hfeat": np.concatenate(hfeat_l), "label": np.concatenate(lab_l),
        "attack": np.concatenate(att_l),
        "attack_names": np.array(sorted(att_map, key=att_map.get), dtype=object), "n_total": n_total, "n_bad": n_bad,
        "neg_counts": neg_counts,
    }


def dedup_and_pool(sc: dict, family: list[str] | None, pool_cap: int, seed: int) -> dict:
    """Global exact dedup on features+Label+Attack (first occurrence kept), map labels to
    the task (family=None: every attack is positive; otherwise only the listed Attack
    names are positive and other attacks are excluded), then keep a random pool of at
    most pool_cap unique rows per class. Per-seed samples are drawn from this pool at
    load time (src.common.load_dataset)."""
    _, first = np.unique(sc["hfull"], return_index=True)
    first = np.sort(first)
    u_idx, u_lab, u_hfeat = sc["idx"][first], sc["label"][first], sc["hfeat"][first]
    u_att = sc["attack_names"][sc["attack"][first]]

    # label conflicts: identical feature vectors that occur with both labels
    pair = np.unique(np.stack([u_hfeat, u_lab.astype(np.uint64)], 1), axis=0)
    hv, cnt = np.unique(pair[:, 0], return_counts=True)
    conflict_groups = int((cnt > 1).sum())
    conflict_rows = int(np.isin(u_hfeat, hv[cnt > 1]).sum())

    benign = u_lab == 0
    if family is None:
        pos = u_lab == 1
    else:
        present = set(np.unique(u_att))
        missing = [f for f in family if f not in present]
        if missing:
            raise ValueError(f"family names not found in Attack column: {missing}; "
                             f"available: {sorted(present)}")
        pos = (u_lab == 1) & np.isin(u_att, list(family))
    excluded = ~(benign | pos)

    rng = np.random.default_rng(seed)
    ben, att = np.flatnonzero(benign), np.flatnonzero(pos)
    pb = rng.choice(ben, min(pool_cap, len(ben)), replace=False)
    pa = rng.choice(att, min(pool_cap, len(att)), replace=False)
    pool_pos = np.concatenate([pb, pa])
    order = np.argsort(pool_pos)
    pool_pos = pool_pos[order]
    pool_lab = np.concatenate([np.zeros(len(pb), np.int8), np.ones(len(pa), np.int8)])[order]

    n_u = len(u_lab)
    n_task = int(benign.sum() + pos.sum())
    stats = {
        "rows_total": sc["n_total"],
        "rows_dropped_inf_nan": sc["n_bad"],
        "rows_after_clean": int(len(sc["idx"])),
        "rows_dropped_duplicate": int(len(sc["idx"]) - n_u),
        "rows_unique": int(n_u),
        "benign_unique": int(len(ben)),
        "attack_unique_all": int((u_lab == 1).sum()),
        "positive_unique": int(len(att)),
        "attack_excluded_not_in_task": int(excluded.sum()),
        "attack_ratio_before_dedup": float(sc["label"].mean()) if len(sc["label"]) else np.nan,
        "attack_ratio_after_dedup": float((u_lab == 1).mean()),
        "raw_positive_ratio": float(len(att) / n_task) if n_task else np.nan,
        "label_conflict_groups": conflict_groups,
        "label_conflict_rows": conflict_rows,
        "pool_benign": int(len(pb)),
        "pool_positive": int(len(pa)),
    }
    return {"pool_idx": u_idx[pool_pos], "pool_label": pool_lab, "stats": stats,
            "u_attack_counts": pd.Series(u_att[~excluded]).value_counts()}


def extract(csv_path: Path, feats: list[str], label_col: str, attack_col: str,
            chunksize: int, wanted: dict[str, np.ndarray], log=print) -> dict[str, pd.DataFrame]:
    """Pass 2: collect rows whose global index is in each wanted[name] (sorted arrays)."""
    usecols = feats + [label_col, attack_col]
    dtypes = {f: "float64" for f in feats}
    dtypes[label_col] = "float64"  # tolerate missing labels; cast after filtering
    dtypes[attack_col] = "string"
    parts = {k: [] for k in wanted}
    offset = 0
    for ch in _read_chunks(csv_path, usecols, dtypes, chunksize):
        n = len(ch)
        for k, w in wanted.items():
            lo, hi = np.searchsorted(w, [offset, offset + n])
            if hi > lo:
                parts[k].append(ch.iloc[w[lo:hi] - offset])
        offset += n
    out = {}
    for k, p in parts.items():
        df = pd.concat(p, ignore_index=True)[usecols]
        assert len(df) == len(wanted[k]), (k, len(df), len(wanted[k]))
        out[k] = df
    return out


def transform(df: pd.DataFrame, feats: list[str], log1p_feats: list[str],
              label_col: str, attack_col: str, label: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    lf = set(log1p_feats)
    for f in feats:
        v = df[f].to_numpy(dtype=np.float64)
        if f in lf:
            v = np.log1p(np.clip(v, 0, None))
        out[f] = v.astype(np.float32)
    out[label_col] = (df[label_col].to_numpy() if label is None else label).astype(np.int8)
    out[attack_col] = df[attack_col].astype(str).to_numpy()
    return out
