"""Cross-network discriminative-reversal analysis (Letter main line).

r1  transfer matrix   : train on one source (or 3 sources, LODO), test on every network;
                        in-domain cells use a 30% hold-out of the training network.
                        Class-swap tests decompose a cross cell into benign-side vs attack-side shift.
r2  feature level     : univariate AUC(X_j, y) per network; benign-side / attack-side shift per pair.
r3  audit             : raw vs feature-identical record counts (with / without identifiers).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .common import dataset_codes, feature_names, load_dataset, single_thread_env
from .models import make_model

TRANSFER_FIELDS = ["job_id", "run", "train", "train_kind", "target", "test_kind", "classifier",
                   "seed", "n_test", "n_pos", "auc", "macro_f1", "dr", "fpr", "t_train_s"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def split_stratified(y: np.ndarray, frac: float, rng: np.random.Generator):
    tr = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        tr[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return np.flatnonzero(tr), np.flatnonzero(~tr)


def scores(y, p, thr=0.5) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_test": int(len(y)), "n_pos": int(y.sum()),
            "auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
            "dr": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan}


def balance_pair(Xa, Xb, rng):
    """Equal-size benign (Xa) and attack (Xb) sets -> (X, y)."""
    m = min(len(Xa), len(Xb))
    ia = rng.choice(len(Xa), m, replace=False)
    ib = rng.choice(len(Xb), m, replace=False)
    X = np.vstack([Xa[ia], Xb[ib]])
    y = np.r_[np.zeros(m, np.int8), np.ones(m, np.int8)]
    return X, y


def transfer_jobs(cfg: dict) -> list[dict]:
    codes = dataset_codes(cfg)
    trains = [(c,) for c in codes] + [tuple(x for x in codes if x != t) for t in codes]
    out = []
    for tr, clf, seed in itertools.product(trains, cfg["reversal"]["classifiers"], cfg["seeds"]):
        kind = "single" if len(tr) == 1 else "lodo"
        name = tr[0] if kind == "single" else "LODO:" + "+".join(tr)
        out.append({"train": tr, "train_name": name, "kind": kind, "classifier": clf,
                    "seed": int(seed),
                    "job_id": f"{cfg['run_name']}|{name}|{clf}|s{seed}"})
    return out


# ---------------------------------------------------------------------------
# r1: one model, evaluated everywhere
# ---------------------------------------------------------------------------
def run_transfer_job(cfg: dict, job: dict) -> list[dict]:
    single_thread_env()
    rv = cfg["reversal"]
    seed = job["seed"]
    codes = dataset_codes(cfg)
    data = {c: load_dataset(cfg, c, "bal", seed) for c in codes}
    rng = np.random.default_rng([seed, 7])
    parts = {}
    for s in job["train"]:
        X, y, _ = data[s]
        tr, ho = split_stratified(y, rv["train_frac"], rng)
        parts[s] = (X[tr], y[tr], X[ho], y[ho])
    Xtr = np.vstack([p[0] for p in parts.values()])
    ytr = np.concatenate([p[1] for p in parts.values()])
    model = make_model(job["classifier"], cfg, seed)
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    t_train = time.perf_counter() - t
    thr = cfg.get("threshold", 0.5)
    base = {"job_id": job["job_id"], "run": cfg["run_name"], "train": job["train_name"],
            "train_kind": job["kind"], "classifier": job["classifier"], "seed": seed,
            "t_train_s": t_train}
    rows = []

    def add(target, kind, X, y):
        rows.append({**base, "target": target, "test_kind": kind,
                     **scores(y, model.predict_proba(X)[:, 1], thr)})

    for t in codes:
        if t in parts:
            add(t, "in_domain", parts[t][2], parts[t][3])
        else:
            X, y, _ = data[t]
            add(t, "cross", X, y)
            if job["kind"] == "single":
                s = job["train"][0]
                Xh, yh = parts[s][2], parts[s][3]
                # swapA: target attacks vs source benign  -> isolates attack-side shift
                Xa, ya = balance_pair(Xh[yh == 0], X[y == 1], rng)
                add(t, "swap_target_attack", Xa, ya)
                # swapB: source attacks vs target benign  -> isolates benign-side shift
                Xb, yb = balance_pair(X[y == 0], Xh[yh == 1], rng)
                add(t, "swap_target_benign", Xb, yb)
    return rows


# ---------------------------------------------------------------------------
# r2: feature level
# ---------------------------------------------------------------------------
def _auc_safe(y, v):
    if np.all(v == v[0]) or y.min() == y.max():
        return 0.5
    return roc_auc_score(y, v)


def feature_level(cfg: dict, seed: int, max_rows: int = 20000) -> tuple[list[dict], list[dict]]:
    """Univariate AUC per (network, feature) and class-conditional shift per (pair, feature).

    shift_benign = |AUC(benign_s vs benign_t) - 0.5| * 2  (0 = identical, 1 = disjoint)
    shift_attack = same for attack flows.
    """
    single_thread_env()
    codes = dataset_codes(cfg)
    feats = feature_names(cfg)
    rng = np.random.default_rng([seed, 11])
    data = {c: load_dataset(cfg, c, "bal", seed) for c in codes}
    rows_auc, rows_shift = [], []
    for c in codes:
        X, y, _ = data[c]
        for j, f in enumerate(feats):
            rows_auc.append({"run": cfg["run_name"], "seed": seed, "dataset": c, "feature": f,
                             "auc": _auc_safe(y, X[:, j])})

    def cls(c, k):
        X, y, _ = data[c]
        Z = X[y == k]
        return Z[rng.choice(len(Z), min(max_rows, len(Z)), replace=False)]

    for s, t in itertools.combinations(codes, 2):
        Bs, Bt, As, At = cls(s, 0), cls(t, 0), cls(s, 1), cls(t, 1)
        for j, f in enumerate(feats):
            yb = np.r_[np.zeros(len(Bs)), np.ones(len(Bt))]
            ya = np.r_[np.zeros(len(As)), np.ones(len(At))]
            sb = abs(_auc_safe(yb, np.r_[Bs[:, j], Bt[:, j]]) - 0.5) * 2
            sa = abs(_auc_safe(ya, np.r_[As[:, j], At[:, j]]) - 0.5) * 2
            rows_shift.append({"run": cfg["run_name"], "seed": seed, "pair": f"{s}-{t}",
                               "feature": f, "shift_benign": sb, "shift_attack": sa})
    return rows_auc, rows_shift


def direction(auc: np.ndarray, hi: float, lo: float) -> np.ndarray:
    return np.where(auc > hi, 1, np.where(auc < lo, -1, 0))


def robust_directions(fa: pd.DataFrame, hi: float, lo: float):
    """-> (mean AUC [feature x dataset], direction in {-1,0,+1} [feature x dataset]).
    A direction counts only if every seed agrees; otherwise 0."""
    mean = fa.groupby(["dataset", "feature"])["auc"].mean().unstack(0)
    dmin = fa.assign(d=direction(fa.auc.values, hi, lo)).groupby(["dataset", "feature"])["d"]
    robust = (dmin.min() == dmin.max()).unstack(0)          # same sign in every seed
    dirs = direction(mean.values, hi, lo) * robust.reindex_like(mean).values
    return mean, pd.DataFrame(dirs, index=mean.index, columns=mean.columns)


def reversal_tables(fa: pd.DataFrame, hi: float, lo: float):
    """fa: feature_auc rows of ONE run. Returns (per-feature table, per-pair table).

    A feature is 'robustly' directional in a network only if every seed agrees.
    Reversal for pair (s,t): directions are +1 and -1 (uninformative networks don't count).
    """
    mean, D = robust_directions(fa, hi, lo)
    codes = list(mean.columns)
    pairs = list(itertools.combinations(codes, 2))
    feat_rows = []
    for f in mean.index:
        rev = [int(D.loc[f, a] * D.loc[f, b] == -1) for a, b in pairs]
        inf = [int(D.loc[f, a] != 0 and D.loc[f, b] != 0) for a, b in pairs]
        feat_rows.append({"feature": f, **{f"auc_{c}": mean.loc[f, c] for c in codes},
                          "n_informative_networks": int((D.loc[f] != 0).sum()),
                          "reversed_pairs": sum(rev), "informative_pairs": sum(inf),
                          "R": sum(rev) / len(pairs),
                          "max_strength": float((mean.loc[f] - 0.5).abs().max())})
    pair_rows = []
    for a, b in pairs:
        both = (D[a] != 0) & (D[b] != 0)
        rev = (D[a] * D[b] == -1)
        pair_rows.append({"pair": f"{a}-{b}", "informative_both": int(both.sum()),
                          "reversed": int(rev.sum()),
                          "reversal_rate": rev.sum() / both.sum() if both.sum() else np.nan})
    return (pd.DataFrame(feat_rows).sort_values(["R", "max_strength"], ascending=False),
            pd.DataFrame(pair_rows))


# ---------------------------------------------------------------------------
# r5: oracle direction-alignment diagnostic (NOT a method — uses target labels for directions)
# ---------------------------------------------------------------------------
ALIGN_FIELDS = ["job_id", "run", "train", "target", "classifier", "seed", "n_flip",
                "auc_base", "auc_flip", "auc_qmap", "auc_qmap_flip", "auc_qmap_randflip",
                "med_src_benign", "med_src_attack", "med_tgt_benign", "med_tgt_attack"]


def _midrank_u(v: np.ndarray) -> np.ndarray:
    srt = np.sort(v)
    return (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / (2.0 * len(v))


def quantile_transform(Xt: np.ndarray, ref: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """Map each target feature through its own ECDF onto the reference marginal `ref`
    (rows = reference samples). flip[j] reverses the order (u -> 1-u) for feature j.
    Uses target features only (no labels); `flip` is where labels enter (oracle)."""
    out = np.empty(Xt.shape, dtype=np.float64)
    for j in range(Xt.shape[1]):
        u = _midrank_u(Xt[:, j])
        if flip[j]:
            u = 1.0 - u
        r = np.sort(ref[:, j])
        q = (np.arange(len(r)) + 0.5) / len(r)
        out[:, j] = np.interp(u, q, r)
    return out


def alignment_jobs(cfg: dict) -> list[dict]:
    return [{"train": c, "classifier": clf, "seed": int(seed),
             "job_id": f"{cfg['run_name']}|{c}|{clf}|s{seed}"}
            for c in dataset_codes(cfg) for clf in cfg["reversal"]["classifiers"]
            for seed in cfg["seeds"]]


def run_alignment_job(cfg: dict, job: dict, Dir: pd.DataFrame, n_rand: int = 5) -> list[dict]:
    """Same single-source model as r1 (same split RNG), evaluated on the target under:
    base | flip (reverse reversed features within the target's own marginal) |
    qmap (label-free marginal alignment to the source) | qmap_flip (both) |
    qmap_randflip (control: flip the same number of NON-reversed informative features)."""
    single_thread_env()
    rv = cfg["reversal"]
    seed, s = job["seed"], job["train"]
    codes = dataset_codes(cfg)
    feats = feature_names(cfg)
    X, y, _ = load_dataset(cfg, s, "bal", seed)
    rng = np.random.default_rng([seed, 7])
    tr, ho = split_stratified(y, rv["train_frac"], rng)
    model = make_model(job["classifier"], cfg, seed).fit(X[tr], y[tr])
    ref = X[tr][np.random.default_rng([seed, 13]).choice(len(tr), min(20000, len(tr)), replace=False)]
    p_ho = model.predict_proba(X[ho])[:, 1]
    rows = []
    rr = np.random.default_rng([seed, 17])
    for t in codes:
        if t == s:
            continue
        Xt, yt, _ = load_dataset(cfg, t, "bal", seed)
        ds_, dt_ = Dir.loc[feats, s].values, Dir.loc[feats, t].values
        flip = (ds_ * dt_) == -1
        cand = np.flatnonzero((~flip) & (ds_ != 0) & (dt_ != 0))
        auc = lambda Z: roc_auc_score(yt, model.predict_proba(Z)[:, 1])
        p_base = model.predict_proba(Xt)[:, 1]
        rand = []
        for _ in range(n_rand):
            m = np.zeros(len(feats), bool)
            if len(cand):
                m[rr.choice(cand, min(int(flip.sum()), len(cand)), replace=False)] = True
            rand.append(auc(quantile_transform(Xt, ref, m)))
        rows.append({
            "job_id": job["job_id"], "run": cfg["run_name"], "train": s, "target": t,
            "classifier": job["classifier"], "seed": seed, "n_flip": int(flip.sum()),
            "auc_base": roc_auc_score(yt, p_base),
            "auc_flip": auc(quantile_transform(Xt, Xt, flip)),
            "auc_qmap": auc(quantile_transform(Xt, ref, np.zeros(len(feats), bool))),
            "auc_qmap_flip": auc(quantile_transform(Xt, ref, flip)),
            "auc_qmap_randflip": float(np.mean(rand)),
            "med_src_benign": float(np.median(p_ho[y[ho] == 0])),
            "med_src_attack": float(np.median(p_ho[y[ho] == 1])),
            "med_tgt_benign": float(np.median(p_base[yt == 0])),
            "med_tgt_attack": float(np.median(p_base[yt == 1])),
        })
    return rows


SENS_FIELDS = ["job_id", "run", "train", "target", "classifier", "seed", "flip_hi", "flip_lo",
               "n_flip", "auc_base", "auc_qmap_flip", "auc_qmap_randflip"]


def run_alignment_sens_job(cfg: dict, job: dict, dirs: dict, n_rand: int = 5) -> list[dict]:
    """Threshold sensitivity of the oracle test: same model as r1/r5, flip sets defined by
    alternative (flip_hi, flip_lo) thresholds. dirs: {(hi, lo): Dir DataFrame}."""
    single_thread_env()
    rv = cfg["reversal"]
    seed, s = job["seed"], job["train"]
    feats = feature_names(cfg)
    X, y, _ = load_dataset(cfg, s, "bal", seed)
    rng = np.random.default_rng([seed, 7])
    tr, _ = split_stratified(y, rv["train_frac"], rng)
    model = make_model(job["classifier"], cfg, seed).fit(X[tr], y[tr])
    ref = X[tr][np.random.default_rng([seed, 13]).choice(len(tr), min(20000, len(tr)), replace=False)]
    rr = np.random.default_rng([seed, 19])
    rows = []
    for t in dataset_codes(cfg):
        if t == s:
            continue
        Xt, yt, _ = load_dataset(cfg, t, "bal", seed)
        auc = lambda Z: roc_auc_score(yt, model.predict_proba(Z)[:, 1])
        base = auc(Xt)
        for (hi, lo), Dir in dirs.items():
            ds_, dt_ = Dir.loc[feats, s].values, Dir.loc[feats, t].values
            flip = (ds_ * dt_) == -1
            cand = np.flatnonzero((~flip) & (ds_ != 0) & (dt_ != 0))
            rand = []
            for _ in range(n_rand):
                m = np.zeros(len(feats), bool)
                if len(cand):
                    m[rr.choice(cand, min(int(flip.sum()), len(cand)), replace=False)] = True
                rand.append(auc(quantile_transform(Xt, ref, m)))
            rows.append({"job_id": job["job_id"], "run": cfg["run_name"], "train": s, "target": t,
                         "classifier": job["classifier"], "seed": seed, "flip_hi": hi,
                         "flip_lo": lo, "n_flip": int(flip.sum()), "auc_base": base,
                         "auc_qmap_flip": auc(quantile_transform(Xt, ref, flip)),
                         "auc_qmap_randflip": float(np.mean(rand))})
    return rows


# ---------------------------------------------------------------------------
# r6: few-label direction estimation (practical implication, NOT a new method)
# ---------------------------------------------------------------------------
FEW_FIELDS = ["run", "kind", "train", "target", "classifier", "seed", "n_labels", "draw",
              "n_eval", "n_flip_est", "n_flip_oracle", "flip_precision", "flip_recall",
              "auc_base", "auc_qmap", "auc_qmap_flip_est", "auc_qmap_flip_oracle", "auc"]


def univariate_dirs(X: np.ndarray, y: np.ndarray, hi: float, lo: float) -> np.ndarray:
    return direction(np.array([_auc_safe(y, X[:, j]) for j in range(X.shape[1])]), hi, lo)


def few_label_draws(yt: np.ndarray, n: int, draws: int, seed: int, t: str):
    """Fixed calibration draws (n/2 per class) shared by every method for a (target, seed, n)."""
    rng = np.random.default_rng([seed, zlib_crc(t), n, 23])
    ben, att = np.flatnonzero(yt == 0), np.flatnonzero(yt == 1)
    for d in range(draws):
        yield d, np.r_[rng.choice(ben, n // 2, replace=False), rng.choice(att, n - n // 2, replace=False)]


def zlib_crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


def _eval_subset(cfg, t, seed):
    """Target sample (per seed) capped at eval_max rows; same subset for every method."""
    Xt, yt, _ = load_dataset(cfg, t, "bal", seed)
    m = cfg["reversal"].get("few_eval_max", 20000)
    rng = np.random.default_rng([seed, zlib_crc(t), 29])
    idx = np.sort(rng.choice(len(yt), min(m, len(yt)), replace=False))
    return Xt[idx], yt[idx]


def run_fewlabel_transfer_job(cfg: dict, job: dict) -> list[dict]:
    """Source model (same as r1/r5) + target directions estimated from n labelled target flows.
    Calibration flows are removed from the evaluation set."""
    single_thread_env()
    rv = cfg["reversal"]
    hi, lo = rv["flip_hi"], rv["flip_lo"]
    seed, s = job["seed"], job["train"]
    X, y, _ = load_dataset(cfg, s, "bal", seed)
    rng = np.random.default_rng([seed, 7])
    tr, _ = split_stratified(y, rv["train_frac"], rng)
    model = make_model(job["classifier"], cfg, seed).fit(X[tr], y[tr])
    ref = X[tr][np.random.default_rng([seed, 13]).choice(len(tr), min(20000, len(tr)), replace=False)]
    ds_ = univariate_dirs(X[tr], y[tr], hi, lo)          # source directions: source training data
    rows = []
    for t in dataset_codes(cfg):
        if t == s:
            continue
        Xt, yt = _eval_subset(cfg, t, seed)
        p_base = model.predict_proba(Xt)[:, 1]
        p_q = model.predict_proba(quantile_transform(Xt, ref, np.zeros(Xt.shape[1], bool)))[:, 1]
        flip_or = (ds_ * univariate_dirs(Xt, yt, hi, lo)) == -1
        p_or = model.predict_proba(quantile_transform(Xt, ref, flip_or))[:, 1]
        cache = {}
        for n in rv["few_n"]:
            for d, cal in few_label_draws(yt, n, rv["few_draws"], seed, t):
                ev = np.setdiff1d(np.arange(len(yt)), cal)
                flip = (ds_ * univariate_dirs(Xt[cal], yt[cal], hi, lo)) == -1
                key = flip.tobytes()
                if key not in cache:
                    cache[key] = model.predict_proba(quantile_transform(Xt, ref, flip))[:, 1]
                p_f = cache[key]
                tp = int((flip & flip_or).sum())
                rows.append({
                    "run": cfg["run_name"], "kind": "direction_correction", "train": s,
                    "target": t, "classifier": job["classifier"], "seed": seed, "n_labels": n,
                    "draw": d, "n_eval": len(ev), "n_flip_est": int(flip.sum()),
                    "n_flip_oracle": int(flip_or.sum()),
                    "flip_precision": tp / flip.sum() if flip.sum() else np.nan,
                    "flip_recall": tp / flip_or.sum() if flip_or.sum() else np.nan,
                    "auc_base": roc_auc_score(yt[ev], p_base[ev]),
                    "auc_qmap": roc_auc_score(yt[ev], p_q[ev]),
                    "auc_qmap_flip_est": roc_auc_score(yt[ev], p_f[ev]),
                    "auc_qmap_flip_oracle": roc_auc_score(yt[ev], p_or[ev]), "auc": np.nan})
    return rows


def run_fewlabel_target_only_job(cfg: dict, job: dict) -> list[dict]:
    """Baseline a reviewer will ask for: train directly on the same n labelled target flows."""
    single_thread_env()
    rv = cfg["reversal"]
    t, seed = job["target"], job["seed"]
    Xt, yt = _eval_subset(cfg, t, seed)
    rows = []
    for n in rv["few_n"]:
        for d, cal in few_label_draws(yt, n, rv["few_draws"], seed, t):
            ev = np.setdiff1d(np.arange(len(yt)), cal)
            for clf in rv["few_target_only"]:
                if clf == "xgboost":
                    from .models import make_xgb
                    m = make_xgb(cfg["selector_xgb"], seed)
                else:
                    m = make_model(clf, cfg, seed)
                m.fit(Xt[cal], yt[cal])
                rows.append({"run": cfg["run_name"], "kind": "target_only", "train": "",
                             "target": t, "classifier": clf, "seed": seed, "n_labels": n,
                             "draw": d, "n_eval": len(ev),
                             "auc": roc_auc_score(yt[ev], m.predict_proba(Xt[ev])[:, 1])})
    return rows
