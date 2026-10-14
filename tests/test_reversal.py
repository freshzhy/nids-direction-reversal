"""Reversal analysis on synthetic data: BOT has STABLE_A with the opposite direction."""
import subprocess
import sys

import numpy as np
import pandas as pd

from conftest import ROOT
from src.reversal import balance_pair, direction, reversal_tables, split_stratified


def _run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_helpers():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(70), np.ones(30)].astype(int)
    tr, ho = split_stratified(y, 0.7, rng)
    assert len(set(tr) & set(ho)) == 0 and y[tr].sum() == 21
    X, yy = balance_pair(np.zeros((10, 2)), np.ones((4, 2)), rng)
    assert yy.sum() == 4 and len(yy) == 8
    assert direction(np.array([0.6, 0.5, 0.4]), 0.55, 0.45).tolist() == [1, 0, -1]


def test_reversal_tables_robust_rule():
    rows = []
    for seed in range(3):
        rows += [{"seed": seed, "dataset": "A", "feature": "f", "auc": 0.8},
                 {"seed": seed, "dataset": "B", "feature": "f", "auc": 0.2},
                 {"seed": seed, "dataset": "C", "feature": "f", "auc": 0.5},
                 {"seed": seed, "dataset": "A", "feature": "g", "auc": 0.8},
                 # g in B: mean < 0.45 but one seed disagrees -> not robust -> not counted
                 {"seed": seed, "dataset": "B", "feature": "g", "auc": [0.3, 0.3, 0.6][seed]},
                 {"seed": seed, "dataset": "C", "feature": "g", "auc": 0.7}]
    F, P = reversal_tables(pd.DataFrame(rows), 0.55, 0.45)
    F = F.set_index("feature")
    assert F.loc["f", "reversed_pairs"] == 1 and np.isclose(F.loc["f", "R"], 1 / 3)
    assert F.loc["g", "reversed_pairs"] == 0
    assert P.set_index("pair").loc["A-B", "reversed"] == 1


def test_scripts(synth_project):
    root, cfg = synth_project
    c = str(root / "config.yaml")
    _run("scripts/r1_transfer.py", "--config", c, "--n-jobs", "2")
    out = _run("scripts/r1_transfer.py", "--config", c, "--n-jobs", "2")
    assert "本次 0 个模型" in out                                    # resume
    _run("scripts/r2_feature_reversal.py", "--config", c)
    _run("scripts/r5_alignment.py", "--config", c, "--n-jobs", "2")
    _run("scripts/r5_alignment.py", "--config", c, "--n-jobs", "2", "--sens")
    _run("scripts/r6_few_label.py", "--config", c, "--n-jobs", "2")
    assert "本次 0 个任务" in _run("scripts/r6_few_label.py", "--config", c, "--n-jobs", "2")
    rep = _run("scripts/r4_report.py", c)
    res = root / "results"
    tr = pd.read_csv(res / "synth" / "reversal" / "transfer.csv")
    # 8 training sets x 3 classifiers x 2 seeds
    assert tr.job_id.nunique() == 48
    ind = tr[(tr.test_kind == "in_domain") & (tr.train_kind == "single")].auc
    assert ind.min() > 0.75
    # the planted reversal: any model trained on BOT ranks the other networks inversely
    lr = tr[(tr.classifier == "lr") & (tr.train == "BOT") & (tr.test_kind == "cross")]
    assert (lr.auc < 0.5).all()
    F = pd.read_csv(res / "reversal_summary" / "feature_reversal_synth.csv").set_index("feature")
    assert F.loc["STABLE_A", "reversed_pairs"] == 3                 # BOT vs the other three
    assert (res / "reversal_summary" / "fig1_transfer_synth.pdf").exists()
    assert (res / "reversal_summary" / "fig2_feature_auc_synth_top15.png").exists()
    assert "Spearman" in rep

    al = pd.read_csv(res / "synth" / "reversal" / "alignment.csv")
    b = al[(al.train == "BOT") & (al.classifier == "lr")]
    assert (b.n_flip >= 1).all()
    # planted reversal: flipping the reversed feature restores ranking, marginal alignment does not
    assert (b.auc_qmap_flip > 0.7).all() and (b.auc_qmap < 0.5).all()
    assert (b.auc_qmap_randflip < 0.5).all()
    assert "方向对齐诊断" in rep and "方向阈值敏感性" in rep and "Wilcoxon" in rep
    assert (res / "reversal_summary" / "fig4_alignment_synth.png").exists()
    fl = pd.read_csv(res / "synth" / "reversal" / "fewlabel.csv")
    assert set(fl.kind) == {"direction_correction", "target_only"}
    d = fl[(fl.kind == "direction_correction") & (fl.train == "BOT") & (fl.n_labels == 100)]
    assert d.auc_qmap_flip_est.mean() > 0.65 and d.flip_recall.mean() > 0.9
    assert (fl[fl.kind == "direction_correction"].n_eval < 2000).all()   # calibration removed
    assert (res / "reversal_summary" / "fig5_fewlabel_synth.png").exists()
    se = pd.read_csv(res / "synth" / "reversal" / "alignment_sens.csv")
    assert se[["flip_hi", "flip_lo"]].drop_duplicates().shape[0] == 3


def test_quantile_transform():
    from src.reversal import quantile_transform
    rng = np.random.default_rng(0)
    Xt = rng.normal(5, 1, size=(500, 2))
    ref = rng.normal(0, 1, size=(800, 2))
    Z = quantile_transform(Xt, ref, np.array([False, True]))
    assert abs(Z[:, 0].mean()) < 0.1                                # mapped onto reference
    assert np.corrcoef(Xt[:, 0], Z[:, 0])[0, 1] > 0.95
    assert np.corrcoef(Xt[:, 1], Z[:, 1])[0, 1] < -0.95             # order reversed
