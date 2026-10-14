"""Config loading, paths and processed-data access shared by all scripts."""
from __future__ import annotations

import copy
import functools
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:                       # overlay config: base: ../config.yaml
        base = _read_yaml((path.parent / cfg.pop("base")).resolve())
        cfg = _deep_merge(base, cfg)
    return cfg


@functools.lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    path = Path(path) if path else Path(os.environ.get("SSC_CONFIG", ROOT / "config.yaml"))
    cfg = _read_yaml(path.resolve())
    cfg.pop("_root", None)
    # paths are relative to the project root (the main config.yaml location)
    cfg["_root"] = str(ROOT)
    return cfg


def path_of(cfg: dict, key: str) -> Path:
    """processed/results paths get the run_name sub-directory (one folder per protocol)."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else Path(cfg["_root"]) / p
    if key in ("processed", "results") and cfg.get("run_name"):
        p = p / cfg["run_name"]
    return p


def task_family(cfg: dict, code: str, available: list[str] | None = None) -> list[str] | None:
    """Attack names counted as positive for this dataset (None = every attack).

    A family entry is either an explicit list of Attack names, or {"regex": "..."} which is
    resolved against the Attack names actually present (`available`, Benign excluded)."""
    import re
    t = cfg.get("task", "binary")
    if t == "binary":
        return None
    fam = cfg["task_families"][t]
    entry = fam.get(code, fam.get("_all"))
    if isinstance(entry, dict) and "regex" in entry:
        if available is None:
            raise ValueError("regex family needs the list of available attack names")
        rx = re.compile(entry["regex"])
        names = [a for a in available if rx.search(a) and a.lower() != "benign"]
        if not names:
            raise ValueError(f"[{code}] regex {entry['regex']!r} matched no attack in {sorted(available)}")
        return names
    return list(entry)


def find_csv(root: Path, name: str) -> Path:
    """<root>/<name>/data/<name>.csv (NFv2 bagit layout); otherwise the largest CSV
    under <root>/<name>/ that is not a feature-description file."""
    p = root / name / "data" / f"{name}.csv"
    if p.exists():
        return p
    cands = [c for c in (root / name).rglob("*.csv")
             if "feature" not in c.name.lower() and not c.name.startswith(".")]
    if not cands:
        raise FileNotFoundError(f"no data CSV found under {root / name}")
    return max(cands, key=lambda c: c.stat().st_size)


def dataset_codes(cfg: dict) -> list[str]:
    return list(cfg["datasets"].keys())


def feature_names(cfg: dict) -> list[str]:
    fpath = path_of(cfg, "processed") / "features.csv"
    return pd.read_csv(fpath)["feature"].tolist()


@functools.lru_cache(maxsize=8)
def _load_pool(fpath: str, feats: tuple, label_col: str, attack_col: str):
    df = pd.read_parquet(fpath)
    X = df[list(feats)].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int8)
    attack = df[attack_col].astype(str).to_numpy()
    return X, y, attack


def sample_positions(y: np.ndarray, prior: str, n_per_class: int, n_raw: int,
                     raw_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """bal: min(n_per_class, pool sizes) per class.  raw: original positive ratio
    (from the full de-duplicated data), total <= n_raw, shrunk if a class pool is short."""
    ben, att = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if prior == "bal":
        m = min(n_per_class, len(ben), len(att))
        nb = na = m
    else:
        r = float(raw_ratio)
        nr = n_raw
        if r > 0:
            nr = min(nr, len(att) / r)
        if r < 1:
            nr = min(nr, len(ben) / (1 - r))
        na = min(max(int(round(nr * r)), 1), len(att))
        nb = min(int(round(nr)) - na, len(ben))
    return np.sort(np.concatenate([rng.choice(ben, nb, replace=False),
                                   rng.choice(att, na, replace=False)]))


@functools.lru_cache(maxsize=48)
def _load_sample(fpath, feats, label_col, attack_col, code, prior, seed, n_per_class, n_raw,
                 raw_ratio):
    X, y, att = _load_pool(fpath, feats, label_col, attack_col)
    rng = np.random.default_rng([int(seed), zlib.crc32(code.encode()), 0 if prior == "bal" else 1])
    pos = sample_positions(y, prior, n_per_class, n_raw, raw_ratio, rng)
    out = (X[pos], y[pos], att[pos])
    for a in out:
        a.setflags(write=False)
    return out


def dataset_meta(cfg: dict) -> dict:
    return read_json(path_of(cfg, "processed") / "meta.json")


def load_dataset(cfg: dict, code: str, prior: str = "bal", seed: int = 0):
    """Per-seed sample drawn from the de-duplicated pool.
    Returns (X float32 [n,d], y int8 [n], attack str [n]) — read-only, cached per process."""
    pp = cfg["preprocess"]
    meta = dataset_meta(cfg)[code]
    return _load_sample(str(path_of(cfg, "processed") / f"{code}_pool.parquet"),
                        tuple(feature_names(cfg)), pp["label_col"], pp["attack_col"], code,
                        prior, int(seed), int(pp["n_per_class"]), int(pp["n_raw"]),
                        float(meta["raw_positive_ratio"]))


def categorical_mask(cfg: dict, feats: list[str]) -> np.ndarray:
    cats = set(cfg["preprocess"]["categorical_features"])
    return np.array([f in cats for f in feats], dtype=bool)


def resolve_n_jobs(cfg: dict, override: int | None = None) -> int:
    n = override if override is not None else cfg.get("n_jobs", -1)
    if n is None or n < 1:
        n = os.cpu_count() or 1
    return int(n)


def single_thread_env():
    """Call inside worker processes: avoid BLAS/OpenMP oversubscription."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_np_default)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def stratified_subsample(y: np.ndarray, n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Indices of a stratified subsample of size ~n (all indices if n is None or >= len)."""
    N = len(y)
    if n is None or n >= N:
        return np.arange(N)
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        m = max(1, int(round(n * len(idx) / N)))
        out.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def library_versions() -> dict:
    import platform
    import sklearn
    import scipy
    import xgboost
    try:  # optional; only recorded if installed
        import shap
        shap_version = shap.__version__
    except ImportError:
        shap_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap_version,
    }
