"""Synthetic multi-domain data with a planted 'shortcut' feature, used by all tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D = 12
FEATS = ["STABLE_A", "STABLE_B", "SHORTCUT"] + [f"NOISE_{i}" for i in range(D - 3)]


def make_domain(rng, n, shortcut: bool, shift: float, attack_ratio=0.5, attack_name="DoS",
                flip: bool = False):
    y = (rng.random(n) < attack_ratio).astype(np.int8)
    X = rng.normal(size=(n, D)).astype(np.float32)
    X[:, 0] += (-1.2 if flip else 1.2) * y   # informative everywhere; direction reversed if flip
    X[:, 1] += 0.6 * y                       # weaker, stable
    X[:, 2] += shift                         # domain shift on benign + attack
    if shortcut:
        X[:, 2] += 6.0 * y                   # near-perfect separator in ONE domain only
    X[:, 3:] += rng.normal(scale=0.3, size=(1, D - 3)) * shift   # mild random shifts
    att = np.where(y == 1, attack_name, "Benign")
    return X, y, att


def small_cfg(root: Path) -> dict:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {"raw_root": str(root / "raw"), "processed": str(root / "processed"),
                    "results": str(root / "results")}
    cfg["run_name"] = "synth"
    cfg["preprocess"].update({"n_per_class": 1000, "n_raw": 1500})
    cfg["seeds"] = [0, 1]
    cfg["k_values"] = [2, 4]
    cfg["k_main"] = 2
    cfg["n_jobs"] = 2
    cfg["classifiers"] = {"xgboost": {"n_estimators": 40, "max_depth": 4, "learning_rate": 0.2},
                          "rf": {"n_estimators": 30},
                          "mlp": {"hidden_layer_sizes": [16, 8], "max_iter": 50}}
    cfg["selector_xgb"] = {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.2}
    cfg["ssc"].update({"B": 6, "B_max": 8, "k0": 3, "shap_max_rows": 400, "shift_max_rows": 2000})
    cfg["baselines"].update({"shap_max_rows": 500, "mi_max_rows": 2000, "rfe_max_rows": 2000,
                             "l1_max_rows": 2000, "l1_n_C": 10, "stab_pool_rows": 2000,
                             "stab_n_subsamples": 8, "stab_n_C": 5})
    cfg["e4"] = {"lambdas": [0.0, 1.0], "Bs": [4, 8], "k0s": [3]}
    cfg["reversal"].update({"compare_runs": ["synth"], "few_n": [20, 100], "few_draws": 3,
                            "few_eval_max": 2000})
    cfg["_root"] = str(root)
    return cfg


@pytest.fixture(scope="session")
def synth_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    cfg = small_cfg(root)
    pdir = root / "processed" / "synth"
    pdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    specs = {"UN": (True, 0.0), "CIC": (False, 1.5), "TON": (False, -1.0), "BOT": (False, 0.5)}
    meta = {}
    for code, (sc, sh) in specs.items():
        X, y, att = make_domain(rng, 3000, sc, sh, 0.5, attack_name=f"atk_{code}",
                                flip=(code == "BOT"))
        df = pd.DataFrame(X, columns=FEATS)
        df["Label"], df["Attack"] = y, att
        df.to_parquet(pdir / f"{code}_pool.parquet", index=False)
        meta[code] = {"raw_positive_ratio": 0.8 if code == "BOT" else 0.2}
    json.dump(meta, open(pdir / "meta.json", "w"))
    pd.DataFrame({"feature": FEATS, "transform": "numeric_raw", "description": ""}).to_csv(
        pdir / "features.csv", index=False)
    with open(root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_root"}, f, allow_unicode=True)
    return root, cfg


@pytest.fixture(scope="session")
def shortcut_sources():
    """3 sources: shortcut feature strong only in source 0 (and shifted there)."""
    rng = np.random.default_rng(1)
    doms = [make_domain(rng, 3000, True, 2.0), make_domain(rng, 3000, False, 0.0),
            make_domain(rng, 3000, False, -0.3)]
    return [d[0] for d in doms], [d[1] for d in doms]
