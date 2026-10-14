"""r4 汇总：判别关系反转的全部表格、图和 claims.json（论文中的数字只从这里取）。

用法（项目根目录，r1/r2 对 compare_runs 中每个协议都跑完后；r3 可选）:
    python scripts/r4_report.py
输出: results/reversal_summary/
    report.md                    全部表格 + 关键数字
    claims.json                  论文引用的数字
    fig1_transfer_<run>.pdf/.png 单源域 4×4 AUC 热力图（每个分类器一幅）
    fig2_feature_auc_<run>.pdf   特征 × 网络 单变量 AUC 热力图（前 15 个 / 全部）
    fig3_swap_<run>.pdf          类别互换分解
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, wilcoxon

from src.common import dataset_codes, load_config, path_of
from src.reversal import reversal_tables, robust_directions

# diverging blue (AUC > 0.5) <-> red (AUC < 0.5), neutral gray midpoint
CMAP = LinearSegmentedColormap.from_list(
    "rev", ["#a3221f", "#e34948", "#f3b3ad", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"])
INK, MUTED = "#1f1f1e", "#6b6a66"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
MARK = ["o", "s", "^"]                            # secondary encoding
CLF_NAME = {"lr": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP", "rf": "Random forest"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def run_cfg(base_cfg, run):
    cfg = dict(base_cfg)
    cfg["run_name"] = run
    return cfg


def load_run(cfg):
    d = path_of(cfg, "results") / "reversal"
    need = ["transfer.csv", "feature_auc.csv", "feature_shift.csv"]
    if not all((d / f).exists() for f in need):
        return None
    return {f.split(".")[0]: pd.read_csv(d / f) for f in need}


def text_color(v):
    return "white" if abs(v - 0.5) > 0.3 else INK


def heat(ax, M, rows, cols, title, fmt="{:.2f}", diag=False):
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, j in itertools.product(range(len(rows)), range(len(cols))):
        v = M[i, j]
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color=text_color(v), fontweight="bold" if (diag and i == j) else "normal")
    ax.set_title(title, fontsize=8, color=INK)
    return im


def main():
    cfg0 = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    rv = cfg0["reversal"]
    codes = dataset_codes(cfg0)
    out = path_of(cfg0, "results").parent / "reversal_summary"
    out.mkdir(parents=True, exist_ok=True)
    L, C = [], {}
    L += ["# 跨网络判别关系反转：汇总报告", "",
          f"阈值：单变量方向 AUC > {rv['flip_hi']} / < {rv['flip_lo']}（之间为无信息，且所有种子方向一致才计）；"
          f"模型级反转 = 跨域 AUC < {rv['rev_auc']}。所有 AUC 为各种子的均值。", ""]

    runs = {}
    for run in rv["compare_runs"]:
        r = load_run(run_cfg(cfg0, run))
        if r is None:
            L += [f"> {run}: 缺少 r1/r2 结果，跳过", ""]
        else:
            runs[run] = r

    # ------------------------------------------------------------------ model level
    summary_rows = []
    for run, R in runs.items():
        tr = R["transfer"]
        C[run] = {}
        L += [f"## {run}", "", "### 1. 单源域迁移矩阵（行=训练，列=测试；对角线=域内 30% 留出）", ""]
        clfs = [c for c in rv["classifiers"] if c in set(tr.classifier)]
        fig, axes = plt.subplots(1, len(clfs), figsize=(2.3 * len(clfs) + 0.6, 2.4),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, clf in zip(axes, clfs):
            s = tr[(tr.train_kind == "single") & (tr.classifier == clf)
                   & tr.test_kind.isin(["in_domain", "cross"])]
            M = s.groupby(["train", "target"]).auc.mean().unstack().reindex(index=codes, columns=codes)
            L += [f"**{CLF_NAME.get(clf, clf)}**（AUC）", "", M.round(3).to_markdown(), ""]
            im = heat(ax, M.values, codes, codes, CLF_NAME.get(clf, clf), diag=True)
            ax.set_xlabel("Test network")
            if ax is axes[0]:
                ax.set_ylabel("Training network")
            cross = s[s.test_kind == "cross"]
            cell = cross.groupby(["train", "target"]).auc
            cm = cell.mean()
            robust = cell.max() < 0.5                                # every seed below 0.5
            ind = s[s.test_kind == "in_domain"].auc
            row = {"run": run, "classifier": clf, "in_domain_auc_mean": ind.mean(),
                   "in_domain_auc_min": ind.min(), "cross_cells": len(cm),
                   "cross_auc_mean": cm.mean(),
                   "cells_auc_lt_rev": int((cm < rv["rev_auc"]).sum()),
                   "cells_auc_lt_0.5": int((cm < 0.5).sum()),
                   "cells_all_seeds_lt_0.5": int(robust.sum()),
                   "min_cross_auc": cm.min(),
                   "min_cell": "→".join(cm.idxmin())}
            summary_rows.append(row)
            C[run][clf] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                           for k, v in row.items() if k not in ("run", "classifier")}
        cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1])
        cb.set_label("ROC-AUC  (< 0.5 = reversed ranking)")
        cb.outline.set_visible(False)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig1_transfer_{run}.{ext}", dpi=300)
        plt.close(fig)

        # LODO
        lo = tr[(tr.train_kind == "lodo") & (tr.test_kind == "cross")]
        if len(lo):
            T = lo.groupby(["target", "classifier"]).auc.agg(["mean", "std", "max"]).round(3)
            L += ["### 2. 三源域 LODO（目标域 AUC）", "", T.to_markdown(), ""]
            C[run]["lodo_auc"] = {f"{t}|{c}": float(v) for (t, c), v in T["mean"].items()}

        # class swap
        sw = tr[(tr.train_kind == "single") & tr.test_kind.isin(
            ["cross", "swap_target_attack", "swap_target_benign"])]
        P = sw.groupby(["train", "target", "test_kind"]).auc.mean().unstack()
        P = P[["cross", "swap_target_attack", "swap_target_benign"]]
        rev_cells = P[P.cross < rv["rev_auc"]]
        benign_side = int((rev_cells.swap_target_benign < 0.5).sum())
        attack_side = int((rev_cells.swap_target_attack < 0.5).sum())
        L += ["### 3. 类别互换分解（各分类器平均）", "",
              "swap_target_attack = 源域正常流量 vs 目标域攻击流量（只有攻击侧换成目标网络）；"
              "swap_target_benign = 目标域正常流量 vs 源域攻击流量（只有正常侧换成目标网络）。"
              "若某一侧互换后 AUC 已 < 0.5，说明反转来自这一侧的漂移。", "",
              P.round(3).to_markdown(), "",
              f"跨域 AUC < {rv['rev_auc']} 的单元共 {len(rev_cells)} 个："
              f"仅换正常侧就 < 0.5 的有 {benign_side} 个，仅换攻击侧就 < 0.5 的有 {attack_side} 个。", ""]
        C[run]["swap"] = {"reversed_cells": len(rev_cells), "benign_side_lt_0.5": benign_side,
                          "attack_side_lt_0.5": attack_side,
                          "mean_cross": float(P.cross.mean()),
                          "mean_swap_target_attack": float(P.swap_target_attack.mean()),
                          "mean_swap_target_benign": float(P.swap_target_benign.mean())}
        fig, ax = plt.subplots(figsize=(6.6, 2.4), constrained_layout=True)
        labels = [f"{a}→{b}" for a, b in P.index]
        x = np.arange(len(P))
        names = {"cross": "Cross-network test", "swap_target_attack": "Only attacks from target",
                 "swap_target_benign": "Only benign from target"}
        for k, (col, nm) in enumerate(names.items()):
            ax.scatter(x + (k - 1) * 0.22, P[col].values, s=22, color=CAT[k], marker=MARK[k],
                       label=nm, zorder=3, edgecolors="white", linewidths=0.6)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("ROC-AUC")
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig3_swap_{run}.{ext}", dpi=300)
        plt.close(fig)

        # ------------------------------------------------------------- oracle alignment (r5)
        apath = path_of(run_cfg(cfg0, run), "results") / "reversal" / "alignment.csv"
        if apath.exists():
            al = pd.read_csv(apath)
            conds = ["auc_base", "auc_flip", "auc_qmap", "auc_qmap_flip", "auc_qmap_randflip"]
            cell = al.groupby(["train", "target", "classifier"])[conds + ["n_flip"]].mean()
            revc = cell[cell.auc_base < rv["rev_auc"]]
            summ = pd.DataFrame({"全部跨域单元": cell[conds].mean(),
                                 f"反转单元 (base < {rv['rev_auc']})": revc[conds].mean()}).T
            summ["n"] = [len(cell), len(revc)]
            restored = {c: int((revc[c] > 0.5).sum()) for c in conds}
            pos = al.groupby(["train", "target"])[["med_src_benign", "med_src_attack",
                                                   "med_tgt_benign", "med_tgt_attack"]].mean()
            inverted_pos = int((pos.med_tgt_benign > pos.med_tgt_attack).sum())
            L += ["### 3b. 方向对齐诊断（oracle，r5）", "",
                  "base 原始 / flip 只在目标域内反序方向相反的特征 / qmap 无标签边缘对齐到源域 / "
                  "qmap_flip 对齐+反序 / qmap_randflip 对照（反序同样数量的非反转特征）。"
                  "单元 = 训练网络×目标网络×分类器，种子平均。", "",
                  summ.round(3).to_markdown(), "",
                  f"反转单元中 AUC 恢复到 > 0.5 的个数：" +
                  "，".join(f"{c.replace('auc_', '')} {v}/{len(revc)}" for c, v in restored.items()), "",
                  "按网络对（分类器、种子平均）：", "",
                  cell.groupby(["train", "target"])[conds + ["n_flip"]].mean().round(3).to_markdown(), "",
                  f"模型得分位置（中位数）：目标域正常流量得分高于目标域攻击流量的网络对 {inverted_pos}/{len(pos)}", "",
                  pos.round(3).to_markdown(), ""]
            # paired tests over cross cells (train x target x classifier; seeds averaged)
            tests = [("auc_qmap_flip", "auc_base"), ("auc_qmap_flip", "auc_qmap"),
                     ("auc_qmap_flip", "auc_qmap_randflip"), ("auc_qmap", "auc_base")]
            trows = []
            for scope, D_ in (("全部跨域单元", cell), ("反转单元", revc)):
                ps = []
                for a_, b_ in tests:
                    dlt = (D_[a_] - D_[b_]).values
                    st_ = wilcoxon(dlt) if len(dlt) > 1 and np.any(dlt != 0) else None
                    n_ = len(dlt)
                    # matched-pairs rank-biserial correlation
                    rk = pd.Series(np.abs(dlt)).rank().values
                    rbc = (rk[dlt > 0].sum() - rk[dlt < 0].sum()) / rk.sum() if rk.sum() else np.nan
                    ps.append(st_.pvalue if st_ is not None else np.nan)
                    trows.append({"scope": scope, "comparison": f"{a_[4:]} − {b_[4:]}", "n": n_,
                                  "median_diff": float(np.median(dlt)), "rank_biserial": rbc,
                                  "p": ps[-1]})
                # Holm within scope
                order_ = np.argsort(ps)
                m_ = len(ps)
                adj, run_max = [np.nan] * m_, 0.0
                for rank_, i_ in enumerate(order_):
                    v_ = min(1.0, (m_ - rank_) * ps[i_]) if np.isfinite(ps[i_]) else np.nan
                    run_max = max(run_max, v_) if np.isfinite(v_) else run_max
                    adj[i_] = run_max if np.isfinite(v_) else np.nan
                for k_, i_ in enumerate(range(len(trows) - m_, len(trows))):
                    trows[i_]["p_holm"] = adj[k_]
            TT = pd.DataFrame(trows)
            L += ["配对 Wilcoxon 符号秩检验（单元 = 训练×目标×分类器；同一模型的单元并不完全独立，p 值仅作参考，以效应量为主）：", "",
                  TT.to_markdown(index=False, floatfmt=".4g"), ""]
            # sensitivity to direction thresholds
            spath = apath.with_name("alignment_sens.csv")
            if spath.exists():
                se = pd.read_csv(spath)
                sc_ = se.groupby(["flip_hi", "flip_lo", "train", "target", "classifier"])[
                    ["auc_base", "auc_qmap_flip", "auc_qmap_randflip", "n_flip"]].mean().reset_index()
                rows_ = []
                for (h_, l_), g_ in sc_.groupby(["flip_hi", "flip_lo"]):
                    r_ = g_[g_.auc_base < rv["rev_auc"]]
                    rows_.append({"flip_hi": h_, "flip_lo": l_, "mean_n_flip": g_.n_flip.mean(),
                                  "all_qmap_flip": g_.auc_qmap_flip.mean(),
                                  "all_randflip": g_.auc_qmap_randflip.mean(),
                                  "rev_n": len(r_), "rev_qmap_flip": r_.auc_qmap_flip.mean(),
                                  "rev_restored": int((r_.auc_qmap_flip > 0.5).sum()),
                                  "rev_randflip_restored": int((r_.auc_qmap_randflip > 0.5).sum())})
                SS = pd.DataFrame(rows_)
                L += ["方向阈值敏感性：", "", SS.round(3).to_markdown(index=False), ""]
                C[run]["alignment_sensitivity"] = SS.to_dict(orient="records")
            # fig4: per reversed cell, base -> qmap -> qmap_flip, with random-flip control
            if len(revc):
                fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
                xs_ = [0, 1, 2, 3]
                lab_ = ["Original", "Marginal\nalignment", "Alignment +\nreversed\nfeatures flipped",
                        "Alignment +\nrandom flip\n(control)"]
                for _, r_ in revc.iterrows():
                    ax.plot(xs_[:3], [r_.auc_base, r_.auc_qmap, r_.auc_qmap_flip], color="#b7d3f6",
                            lw=0.8, zorder=1)
                for k_, c_ in enumerate(["auc_base", "auc_qmap", "auc_qmap_flip", "auc_qmap_randflip"]):
                    v_ = revc[c_].values
                    ax.scatter(np.full(len(v_), xs_[k_]) + np.linspace(-0.08, 0.08, len(v_)), v_, s=12,
                               color=CAT[0] if k_ < 3 else MUTED, zorder=2, edgecolors="white",
                               linewidths=0.4)
                    ax.plot([xs_[k_] - 0.22, xs_[k_] + 0.22], [np.median(v_)] * 2, color=INK, lw=1.6,
                            zorder=3)
                ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", zorder=0)
                ax.set_xticks(xs_, lab_, fontsize=6.5)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Target ROC-AUC")
                ax.set_title(f"Reversed cells (n = {len(revc)}); bar = median", fontsize=7.5)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                for ext in ("pdf", "png"):
                    fig.savefig(out / f"fig4_alignment_{run}.{ext}", dpi=300)
                plt.close(fig)
            C[run]["alignment_tests"] = TT.to_dict(orient="records")
            C[run]["alignment"] = {"all": summ.iloc[0][conds].astype(float).to_dict(),
                                   "reversed": summ.iloc[1][conds].astype(float).to_dict(),
                                   "n_all": len(cell), "n_reversed": len(revc),
                                   "restored_gt_0.5": restored,
                                   "pairs_target_benign_scored_above_attack": inverted_pos}

        # ------------------------------------------------------------- few-label (r6)
        fpath = path_of(run_cfg(cfg0, run), "results") / "reversal" / "fewlabel.csv"
        if fpath.exists():
            fl = pd.read_csv(fpath)
            dc = fl[fl.kind == "direction_correction"]
            to = fl[fl.kind == "target_only"]
            cellbase = dc.groupby(["train", "target", "classifier"]).auc_base.mean()
            revkeys = set(cellbase[cellbase < rv["rev_auc"]].index)
            dcr = dc[[k in revkeys for k in zip(dc.train, dc.target, dc.classifier)]]
            per_cell = dcr.groupby(["n_labels", "train", "target", "classifier"])[
                ["auc_base", "auc_qmap", "auc_qmap_flip_est", "auc_qmap_flip_oracle",
                 "flip_precision", "flip_recall", "n_flip_est"]].mean()
            g = per_cell.groupby("n_labels")
            FT = g.mean()
            FT["ci95_flip_est"] = 1.96 * g.auc_qmap_flip_est.std() / np.sqrt(g.size())
            FT["cells"] = g.size()
            # target-only baselines on the targets of the reversed cells (cell-weighted)
            tw = pd.Series([k[1] for k in revkeys]).value_counts()
            tob = to.groupby(["n_labels", "classifier", "target"]).auc.mean().unstack("target")
            for clf in sorted(set(to.classifier)):
                FT[f"target_only_{clf}"] = [
                    float(np.average(tob.loc[(n_, clf), tw.index], weights=tw.values))
                    if (n_, clf) in tob.index else np.nan for n_ in FT.index]
            L += ["### 3c. 少量目标标签估计方向（r6，反转单元）", "",
                  f"反转单元 = 原始 AUC < {rv['rev_auc']} 的 训练×目标×分类器（{len(revkeys)} 个）；"
                  f"每个 n 每个种子 {rv['few_draws']} 次抽样；评估集不含校准样本。"
                  "target_only_* = 只用这 n 条目标标签训练的模型（同一评估集）。", "",
                  FT.round(3).to_markdown(), "",
                  "所有目标域的 target-only 基线（AUC）：", "",
                  to.groupby(["n_labels", "classifier"]).auc.mean().unstack().round(3).to_markdown(), ""]
            C[run]["few_label"] = {"reversed_cells": len(revkeys),
                                   "by_n": FT.reset_index().to_dict(orient="records")}
            if len(FT):
                fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
                xn = FT.index.values
                series = [("auc_qmap_flip_oracle", "Oracle direction correction", CAT[0], "o", "--"),
                          ("auc_qmap_flip_est", "Directions from n target labels", CAT[0], "o", "-"),
                          ("auc_qmap", "Marginal alignment only", MUTED, "s", "-")]
                for col, nm, colr, mk, ls in series:
                    ax.plot(xn, FT[col], color=colr, marker=mk, ms=4, lw=1.4, ls=ls, label=nm)
                ax.fill_between(xn, FT.auc_qmap_flip_est - FT.ci95_flip_est,
                                FT.auc_qmap_flip_est + FT.ci95_flip_est, color=CAT[0], alpha=0.15, lw=0)
                for k_, clf in enumerate(sorted(set(to.classifier))):
                    ax.plot(xn, FT[f"target_only_{clf}"], color=CAT[1 + k_], marker=MARK[1 + k_],
                            ms=4, lw=1.2, label=f"Target-only {CLF_NAME.get(clf, clf)} (n labels)")
                ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
                ax.set_xscale("log")
                ax.set_xticks(xn, [str(v) for v in xn])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_xlabel("Labelled target flows (n)")
                ax.set_ylabel("Target ROC-AUC")
                ax.set_ylim(0, 1.02)
                ax.grid(axis="y", color="#e6e5e1", lw=0.6)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                ax.legend(frameon=False, fontsize=5.8, loc="lower right")
                for ext in ("pdf", "png"):
                    fig.savefig(out / f"fig5_fewlabel_{run}.{ext}", dpi=300)
                plt.close(fig)

        # ------------------------------------------------------------- feature level
        F, PR = reversal_tables(R["feature_auc"], rv["flip_hi"], rv["flip_lo"])
        F.to_csv(out / f"feature_reversal_{run}.csv", index=False)
        L += ["### 4. 特征级方向反转", "",
              f"在至少一对网络间方向相反（且各自所有种子一致）的特征：{int((F.reversed_pairs > 0).sum())} / {len(F)}", "",
              F.head(15).round(3).to_markdown(index=False), "",
              "每对网络中，两边都有判别力的特征里方向相反的比例：", "", PR.round(3).to_markdown(index=False), ""]
        # link: pair feature reversal rate vs model cross AUC (ordered pairs, mean over clfs)
        s = tr[(tr.train_kind == "single") & (tr.test_kind == "cross")]
        mA = s.groupby(["train", "target"]).auc.mean()
        rr = {frozenset(p.split("-")): v for p, v in zip(PR.pair, PR.reversal_rate)}
        xs = [rr[frozenset((a, b))] for a, b in mA.index]
        rho, p = spearmanr(xs, mA.values)
        # class-conditional shift: benign vs attack side, all features and reversed ones
        fs = R["feature_shift"].copy()
        fs["pkey"] = fs.pair.map(lambda p_: "-".join(sorted(p_.split("-"))))
        sh = fs.groupby(["pkey", "feature"])[["shift_benign", "shift_attack"]].mean()
        _, Dir = robust_directions(R["feature_auc"], rv["flip_hi"], rv["flip_lo"])
        revset = {("-".join(sorted((a, b))), f) for a, b in itertools.combinations(Dir.columns, 2)
                  for f in Dir.index if Dir.loc[f, a] * Dir.loc[f, b] == -1}
        shr = sh.loc[[k for k in sh.index if k in revset]] if revset else sh.iloc[:0]
        L += [f"网络对的特征反转率与模型跨域 AUC（12 个有向单元，分类器平均）的 Spearman ρ = {rho:.3f}（p = {p:.3g}）", "",
              f"类别条件偏移（|AUC−0.5|×2，0 = 分布相同，1 = 完全分开）：全部特征 正常侧 {sh.shift_benign.mean():.3f} / 攻击侧 {sh.shift_attack.mean():.3f}；"
              f"发生反转的 {len(shr)} 个（网络对, 特征）中 正常侧 {shr.shift_benign.mean():.3f} / 攻击侧 {shr.shift_attack.mean():.3f}", ""]
        C[run]["feature"] = {
            "n_features": len(F), "n_features_reversed_any_pair": int((F.reversed_pairs > 0).sum()),
            "top_reversed": F.head(8).feature.tolist(),
            "pair_reversal_rate": dict(zip(PR.pair, PR.reversal_rate.astype(float))),
            "spearman_pairrev_vs_auc": {"rho": float(rho), "p": float(p), "n": len(xs)},
            "shift_all": {"benign": float(sh.shift_benign.mean()), "attack": float(sh.shift_attack.mean())},
            "shift_reversed": {"benign": float(shr.shift_benign.mean()) if len(shr) else None,
                               "attack": float(shr.shift_attack.mean()) if len(shr) else None,
                               "n": len(shr)}}
        for tag, FF in (("top15", F.head(15)), ("all", F)):
            M = FF[[f"auc_{c}" for c in codes]].values
            fig, ax = plt.subplots(figsize=(3.3, 0.18 * len(FF) + 0.9), constrained_layout=True)
            im = heat(ax, M, FF.feature.tolist(), codes, "Univariate AUC (feature vs attack label)")
            ax.set_xlabel("Network")
            cb = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[0, 0.5, 1])
            cb.outline.set_visible(False)
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig2_feature_auc_{run}_{tag}.{ext}", dpi=300)
            plt.close(fig)

    if summary_rows:
        S = pd.DataFrame(summary_rows)
        L.insert(3, "")
        L.insert(3, S.round(3).to_markdown(index=False))
        L.insert(3, "")
        L.insert(3, "## 0. 总表：域内 vs 跨域（单源域，12 个跨域单元）")
        S.to_csv(out / "summary_model_level.csv", index=False)

    # audit
    ap = path_of(cfg0, "results").parent / "audit.csv"
    if ap.exists():
        A = pd.read_csv(ap)
        L += ["## 数据审计", "", A.round(4).to_markdown(index=False), ""]
        C["audit"] = A.to_dict(orient="records")
    for run in runs:
        sp = path_of(run_cfg(cfg0, run), "processed") / "stats.csv"
        if sp.exists():
            L += [f"### {run} 预处理统计", "", pd.read_csv(sp).drop(
                columns=["negative_values_clipped"], errors="ignore").to_markdown(index=False), ""]

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "claims.json").write_text(json.dumps(C, ensure_ascii=False, indent=1, default=str),
                                     encoding="utf-8")
    print("\n".join(L))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
