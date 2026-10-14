"""All numbers quoted in the manuscript, computed directly from the result CSVs.

Every value that appears in the text is produced here and written to
paper/generated/numbers.tex as \\val{key}; nothing is typed by hand.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .common import path_of
from .reversal import reversal_tables, robust_directions

RUNS = {"dos_v1": "v2dos", "binary_v1": "v2bin", "dos_v3": "v3dos"}
RUN_LABEL = {"dos_v1": "NFv2, DoS/DDoS-aligned", "binary_v1": "NFv2, heterogeneous binary",
             "dos_v3": "NFv3, DoS/DDoS-aligned"}
CLFS = ["lr", "xgboost", "mlp"]
CLF_LABEL = {"lr": "LR", "xgboost": "XGBoost", "mlp": "MLP"}
CODES = ["UN", "CIC", "TON", "BOT"]
NET_LABEL = {"UN": "UNSW-NB15", "CIC": "CSE-CIC-IDS2018", "TON": "ToN-IoT", "BOT": "BoT-IoT"}
CONDS = ["auc_base", "auc_flip", "auc_qmap", "auc_qmap_flip", "auc_qmap_randflip"]


def run_cfg(cfg, run):
    c = dict(cfg)
    c["run_name"] = run
    return c


def load(cfg, run, name):
    return pd.read_csv(path_of(run_cfg(cfg, run), "results") / "reversal" / name)


def stats_csv(cfg, run):
    return pd.read_csv(path_of(run_cfg(cfg, run), "processed") / "stats.csv").set_index("dataset")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


# ---------------------------------------------------------------------------
def model_level(cfg, run):
    rv = cfg["reversal"]
    tr = load(cfg, run, "transfer.csv")
    s = tr[tr.train_kind == "single"]
    out = {"per_clf": {}}
    ind_all, cross_all = [], []
    for clf in CLFS:
        d = s[s.classifier == clf]
        ind = d[d.test_kind == "in_domain"].groupby("train").auc.mean()
        cr = d[d.test_kind == "cross"].groupby(["train", "target"]).auc
        cm = cr.mean()
        out["per_clf"][clf] = {
            "in_mean": ind.mean(), "in_min": ind.min(), "cross_mean": cm.mean(),
            "rev": int((cm < rv["rev_auc"]).sum()), "below": int((cm < 0.5).sum()),
            "robust": int((cr.max() < 0.5).sum()), "min": cm.min(),
            "min_cell": "$\\rightarrow$".join(cm.idxmin()), "matrix": d[d.test_kind.isin(
                ["in_domain", "cross"])].groupby(["train", "target"]).auc.mean().unstack()
            .reindex(index=CODES, columns=CODES)}
        ind_all += list(ind.values)
        cross_all += list(cm.values)
    pc = out["per_clf"]
    out["rev_total"] = sum(v["rev"] for v in pc.values())
    out["robust_total"] = sum(v["robust"] for v in pc.values())
    out["below_total"] = sum(v["below"] for v in pc.values())
    out["n_cells"] = 12 * len(CLFS)
    out["in_min"] = min(ind_all)
    out["in_max"] = max(ind_all)
    out["cross_min"] = min(cross_all)
    # LODO
    lo = tr[(tr.train_kind == "lodo") & (tr.test_kind == "cross")]
    L = lo.groupby(["target", "classifier"]).auc.mean()
    out["lodo"] = L
    out["lodo_below"] = int((L < 0.5).sum())
    out["lodo_n"] = len(L)
    # class swap (cell = train x target x clf)
    sw = s[s.test_kind.isin(["cross", "swap_target_attack", "swap_target_benign"])]
    P = sw.groupby(["train", "target", "classifier", "test_kind"]).auc.mean().unstack()
    R = P[P.cross < rv["rev_auc"]]
    out["swap_rev_n"] = len(R)
    out["swap_attack_side"] = int((R.swap_target_attack < 0.5).sum())
    out["swap_benign_side"] = int((R.swap_target_benign < 0.5).sum())
    out["swap_both_above"] = int(((R.swap_target_attack >= 0.5) & (R.swap_target_benign >= 0.5)).sum())
    out["swap_table"] = sw.groupby(["train", "target", "test_kind"]).auc.mean().unstack()
    return out


def oracle(cfg, run):
    rv = cfg["reversal"]
    al = load(cfg, run, "alignment.csv")
    cell = al.groupby(["train", "target", "classifier"])[CONDS + ["n_flip"]].mean()
    rev = cell[cell.auc_base < rv["rev_auc"]]
    o = {"n_all": len(cell), "n_rev": len(rev), "mean_all": cell[CONDS].mean(),
         "mean_rev": rev[CONDS].mean(),
         "restored": {c: int((rev[c] > 0.5).sum()) for c in CONDS},
         "ci_rev": {c: boot_ci(rev[c]) for c in CONDS},
         "n_flip_mean": cell.n_flip.mean(), "n_flip_min": cell.n_flip.min(),
         "n_flip_max": cell.n_flip.max()}
    tests = []
    for scope, D in (("all", cell), ("rev", rev)):
        rows = []
        for a, b in [("auc_qmap_flip", "auc_qmap"), ("auc_qmap_flip", "auc_qmap_randflip"),
                     ("auc_qmap_flip", "auc_base"), ("auc_qmap", "auc_base")]:
            d = (D[a] - D[b]).values
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            rows.append({"scope": scope, "a": a, "b": b, "n": len(d), "med": float(np.median(d)),
                         "rrb": rank_biserial(d), "p": float(p)})
        ph = holm([r["p"] for r in rows])
        for r, h in zip(rows, ph):
            r["p_holm"] = float(h)
        tests += rows
    o["tests"] = pd.DataFrame(tests)
    # cluster-level robustness: unit = directed network pair (classifiers averaged)
    pair = al.groupby(["train", "target"])[CONDS].mean()
    crow = []
    for a, b in [("auc_qmap_flip", "auc_qmap"), ("auc_qmap_flip", "auc_qmap_randflip"),
                 ("auc_qmap_flip", "auc_base"), ("auc_qmap", "auc_base")]:
        d = (pair[a] - pair[b]).values
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        # cluster bootstrap over directed pairs, inverted cells only
        rv_ = rev.reset_index()
        rv_["diff"] = rv_[a] - rv_[b]
        keys = sorted(set(zip(rv_.train, rv_.target)))
        g = {k: rv_[(rv_.train == k[0]) & (rv_.target == k[1])]["diff"].values for k in keys}
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(10000):
            pick = rng.integers(0, len(keys), len(keys))
            bs.append(np.concatenate([g[keys[i]] for i in pick]).mean())
        crow.append({"a": a, "b": b, "n_pairs": len(d), "pair_med": float(np.median(d)),
                     "pair_pos": int((d > 0).sum()), "pair_rrb": rank_biserial(d), "pair_p": float(p),
                     "rev_mean": float(rv_["diff"].mean()), "rev_clusters": len(keys),
                     "rev_lo": float(np.percentile(bs, 2.5)), "rev_hi": float(np.percentile(bs, 97.5))})
    ph = holm([r["pair_p"] for r in crow])
    for r, h in zip(crow, ph):
        r["pair_p_holm"] = float(h)
    o["cluster"] = pd.DataFrame(crow)
    # score positions (medians over classifiers & seeds)
    pos = al.groupby(["train", "target"])[["med_src_benign", "med_src_attack", "med_tgt_benign",
                                           "med_tgt_attack"]].mean()
    o["pos"] = pos
    o["pos_inverted"] = int((pos.med_tgt_benign > pos.med_tgt_attack).sum())
    o["pos_collapsed"] = int(((pos.med_tgt_benign < 0.05) & (pos.med_tgt_attack < 0.05)).sum())
    # sensitivity
    se = load(cfg, run, "alignment_sens.csv")
    sc = se.groupby(["flip_hi", "flip_lo", "train", "target", "classifier"])[
        ["auc_base", "auc_qmap_flip", "auc_qmap_randflip", "n_flip"]].mean().reset_index()
    rows = []
    for (h, l), g in sc.groupby(["flip_hi", "flip_lo"]):
        r = g[g.auc_base < rv["rev_auc"]]
        rows.append({"hi": h, "lo": l, "n_flip": g.n_flip.mean(), "n_rev": len(r),
                     "rev_qf": r.auc_qmap_flip.mean(), "rev_rf": r.auc_qmap_randflip.mean(),
                     "rest_qf": int((r.auc_qmap_flip > 0.5).sum()),
                     "rest_rf": int((r.auc_qmap_randflip > 0.5).sum())})
    o["sens"] = pd.DataFrame(rows)
    return o


def features(cfg, run):
    rv = cfg["reversal"]
    fa = load(cfg, run, "feature_auc.csv")
    F, PR = reversal_tables(fa, rv["flip_hi"], rv["flip_lo"])
    mean, Dir = robust_directions(fa, rv["flip_hi"], rv["flip_lo"])
    tr = load(cfg, run, "transfer.csv")
    s = tr[(tr.train_kind == "single") & (tr.test_kind == "cross")]
    mA = s.groupby(["train", "target"]).auc.mean()
    rr = {frozenset(p.split("-")): v for p, v in zip(PR.pair, PR.reversal_rate)}
    rho, p = spearmanr([rr[frozenset(k)] for k in mA.index], mA.values)
    return {"F": F, "PR": PR, "mean": mean, "Dir": Dir, "n_feat": len(F),
            "n_rev_any": int((F.reversed_pairs > 0).sum()),
            "n_rev_all4": int((F.n_informative_networks == 4).sum()),
            "pair_rate_min": PR.reversal_rate.min(), "pair_rate_max": PR.reversal_rate.max(),
            "rho": float(rho), "rho_p": float(p)}


def few_label(cfg, run):
    rv = cfg["reversal"]
    fl = load(cfg, run, "fewlabel.csv")
    dc = fl[fl.kind == "direction_correction"]
    to = fl[fl.kind == "target_only"]
    cb = dc.groupby(["train", "target", "classifier"]).auc_base.mean()
    rk = set(cb[cb < rv["rev_auc"]].index)
    d = dc[[k in rk for k in zip(dc.train, dc.target, dc.classifier)]]
    pc = d.groupby(["n_labels", "train", "target", "classifier"])[
        ["auc_qmap_flip_est", "auc_qmap_flip_oracle", "auc_qmap", "auc_base", "flip_precision",
         "flip_recall"]].mean()
    T = pc.groupby("n_labels").mean()
    T["ci_lo"] = [boot_ci(g.auc_qmap_flip_est)[0] for _, g in pc.groupby("n_labels")]
    T["ci_hi"] = [boot_ci(g.auc_qmap_flip_est)[1] for _, g in pc.groupby("n_labels")]
    tw = pd.Series([k[1] for k in rk]).value_counts()
    tob = to.groupby(["n_labels", "classifier", "target"]).auc.mean()
    for clf in ["lr", "xgboost"]:
        T[f"to_{clf}"] = [float(np.average([tob.loc[(n, clf, t)] for t in tw.index], weights=tw.values))
                          for n in T.index]
    return {"T": T, "n_rev": len(rk)}


def audit(cfg):
    base = path_of(cfg, "results")
    p = (base.parent if cfg.get("run_name") else base) / "audit.csv"
    return pd.read_csv(p).set_index("dataset") if p.exists() else None


def collect(cfg) -> dict:
    out = {}
    for run in RUNS:
        out[run] = {"model": model_level(cfg, run), "oracle": oracle(cfg, run),
                    "feat": features(cfg, run), "few": few_label(cfg, run),
                    "stats": stats_csv(cfg, run)}
    out["audit"] = audit(cfg)
    return out
