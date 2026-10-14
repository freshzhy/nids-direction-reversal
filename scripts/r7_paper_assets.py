"""r7 论文素材：从结果 CSV 生成 paper/generated/ 下的全部数字宏、表格和图。

正文中的每个数字都通过 \\val{key} 引用 numbers.tex，不手工抄写。
用法（项目根目录，r1–r6 跑完之后）:
    python scripts/r7_paper_assets.py
然后编译论文:
    cd paper && latexmk -pdf main.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.common import ROOT, load_config
from src.paper import (CLF_LABEL, CLFS, CODES, CONDS, NET_LABEL, RUN_LABEL, RUNS, collect)

OUT = ROOT / "paper" / "generated"
CMAP = LinearSegmentedColormap.from_list(
    "rev", ["#a3221f", "#e34948", "#f3b3ad", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"])
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
SHORT = {"auc_base": "base", "auc_flip": "flip", "auc_qmap": "qmap",
         "auc_qmap_flip": "qmapflip", "auc_qmap_randflip": "randflip"}

V: dict[str, str] = {}


def put(key, val, fmt="{:.3f}"):
    if isinstance(val, (int, np.integer)):
        s = f"{int(val):,}".replace(",", "{,}")
    elif isinstance(val, (float, np.floating)):
        s = fmt.format(val)
    else:
        s = str(val)
    V[key] = s


def ptex(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def pct(x):
    return f"{100 * x:.1f}\\%"


# ---------------------------------------------------------------------------
def numbers(R):
    allin = []
    for run, p in RUNS.items():
        M, O, Fe, Fw = R[run]["model"], R[run]["oracle"], R[run]["feat"], R[run]["few"]
        put(f"{p}.in.min", M["in_min"]); put(f"{p}.in.max", M["in_max"])
        allin += [M["in_min"], M["in_max"]]
        put(f"{p}.ncells", M["n_cells"]); put(f"{p}.rev.total", M["rev_total"])
        put(f"{p}.robust.total", M["robust_total"]); put(f"{p}.below.total", M["below_total"])
        put(f"{p}.cross.min", M["cross_min"])
        put(f"{p}.lodo.below", M["lodo_below"]); put(f"{p}.lodo.n", M["lodo_n"])
        put(f"{p}.lodo.min", M["lodo"].min())
        for clf in CLFS:
            c = M["per_clf"][clf]
            for k in ("rev", "robust", "below"):
                put(f"{p}.{clf}.{k}", c[k])
            put(f"{p}.{clf}.min", c["min"]); put(f"{p}.{clf}.mincell", c["min_cell"])
            put(f"{p}.{clf}.crossmean", c["cross_mean"]); put(f"{p}.{clf}.inmean", c["in_mean"])
            mat = c["matrix"]
            for a in CODES:
                for b in CODES:
                    put(f"{p}.{clf}.{a}.{b}", mat.loc[a, b])
        for k in ("swap_rev_n", "swap_attack_side", "swap_benign_side", "swap_both_above"):
            put(f"{p}.{k.replace('_', '')}", M[k])
        put(f"{p}.or.nall", O["n_all"]); put(f"{p}.or.nrev", O["n_rev"])
        for c in CONDS:
            s = SHORT[c]
            put(f"{p}.or.rev.{s}", O["mean_rev"][c]); put(f"{p}.or.all.{s}", O["mean_all"][c])
            put(f"{p}.or.rev.{s}.lo", O["ci_rev"][c][0]); put(f"{p}.or.rev.{s}.hi", O["ci_rev"][c][1])
            put(f"{p}.or.rest.{s}", O["restored"][c])
        put(f"{p}.or.nflip.mean", O["n_flip_mean"], "{:.1f}")
        put(f"{p}.or.nflip.min", int(round(O["n_flip_min"]))); put(f"{p}.or.nflip.max", int(round(O["n_flip_max"])))
        for _, t in O["tests"].iterrows():
            k = f"{p}.t.{t.scope}.{SHORT[t.a]}-{SHORT[t.b]}"
            put(k + ".med", t.med); put(k + ".rrb", t.rrb, "{:.2f}")
            V[k + ".p"] = ptex(t.p_holm); put(k + ".n", int(t.n))
        for _, c in O["cluster"].iterrows():
            k = f"{p}.cl.{SHORT[c.a]}-{SHORT[c.b]}"
            put(k + ".pmed", c.pair_med); put(k + ".ppos", int(c.pair_pos)); put(k + ".npairs", int(c.n_pairs))
            put(k + ".prrb", c.pair_rrb, "{:.2f}"); V[k + ".pp"] = ptex(c.pair_p_holm)
            put(k + ".rmean", c.rev_mean); put(k + ".rlo", c.rev_lo); put(k + ".rhi", c.rev_hi)
            put(k + ".nclus", int(c.rev_clusters))
        put(f"{p}.pos.inv", O["pos_inverted"]); put(f"{p}.pos.coll", O["pos_collapsed"])
        put(f"{p}.pos.n", len(O["pos"]))
        for (a, b), r in O["pos"].iterrows():
            put(f"{p}.pos.{a}.{b}.tb", r.med_tgt_benign); put(f"{p}.pos.{a}.{b}.ta", r.med_tgt_attack)
        se = O["sens"]
        put(f"{p}.sens.restmin", int(se.rest_qf.min())); put(f"{p}.sens.restmax", int(se.rest_qf.max()))
        put(f"{p}.sens.rfmax", int(se.rest_rf.max()))
        for f_, row in Fe["mean"].iterrows():
            for c in CODES:
                put(f"{p}.fa.{f_}.{c}", row[c])
        for f_, row in Fe["F"].set_index("feature").iterrows():
            put(f"{p}.fr.{f_}", int(row.reversed_pairs))
        put(f"{p}.f.nfeat", Fe["n_feat"]); put(f"{p}.f.nrevany", Fe["n_rev_any"])
        put(f"{p}.f.prmin", pct(Fe["pair_rate_min"])); put(f"{p}.f.prmax", pct(Fe["pair_rate_max"]))
        put(f"{p}.f.rho", Fe["rho"], "{:.2f}"); V[f"{p}.f.rhop"] = ptex(Fe["rho_p"])
        T = Fw["T"]
        put(f"{p}.few.nrev", Fw["n_rev"])
        for n, r in T.iterrows():
            q = f"{p}.few.{int(n)}"
            put(q + ".est", r.auc_qmap_flip_est); put(q + ".oracle", r.auc_qmap_flip_oracle)
            put(q + ".prec", r.flip_precision, "{:.2f}"); put(q + ".rec", r.flip_recall, "{:.2f}")
            put(q + ".tolr", r.to_lr); put(q + ".toxgb", r.to_xgboost)
            put(q + ".lo", r.ci_lo); put(q + ".hi", r.ci_hi)
        st = R[run]["stats"]
        for c in CODES:
            put(f"{p}.st.{c}.pos", int(st.loc[c, "positive_unique"]))
            put(f"{p}.st.{c}.ben", int(st.loc[c, "benign_unique"]))
            put(f"{p}.st.{c}.nan", int(st.loc[c, "rows_dropped_inf_nan"]))
            put(f"{p}.st.{c}.raw", int(st.loc[c, "rows_total"]))
            put(f"{p}.st.{c}.uniq", int(st.loc[c, "rows_unique"]))
        put(f"{p}.nfeat", int(st["n_features"].iloc[0]))
    put("all.in.min", min(allin)); put("all.in.max", max(allin))
    A = R["audit"]
    if A is not None:
        for c in CODES:
            put(f"audit.{c}.raw", int(A.loc[c, "rows_raw"]))
            put(f"audit.{c}.dupall", pct(A.loc[c, "duplicate_rate_all_columns"]))
            put(f"audit.{c}.dupnoid", pct(A.loc[c, "duplicate_rate_no_identifiers"]))
            put(f"audit.{c}.dupnoiddns", pct(A.loc[c, "duplicate_rate_no_identifiers_dns"]))
            put(f"audit.{c}.unoid", int(A.loc[c, "unique_no_identifiers"]))
            put(f"audit.{c}.unoiddns", int(A.loc[c, "unique_no_identifiers_dns"]))
        put("audit.dupall.max", pct(A.duplicate_rate_all_columns.max()))
    lines = ["% generated by scripts/r7_paper_assets.py -- do not edit",
             "\\makeatletter"]
    for k, v in sorted(V.items()):
        lines.append(f"\\@namedef{{val@{k}}}{{{v}}}")
    lines.append("\\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def tables(R):
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    f3 = lambda x: f"{x:.3f}"
    # datasets
    A = R["audit"]
    rows = []
    for c in CODES:
        s2, s3 = R["dos_v1"]["stats"].loc[c], R["dos_v3"]["stats"].loc[c]
        b2 = R["binary_v1"]["stats"].loc[c]
        rows.append(" & ".join([
            NET_LABEL[c], f"{int(s2.rows_total):,}", f"{100 * A.loc[c, 'duplicate_rate_all_columns']:.1f}",
            f"{int(s2.rows_unique):,}", f"{int(s2.benign_unique):,}", f"{int(b2.positive_unique):,}",
            f"{int(s2.positive_unique):,}", f"{int(s3.rows_total):,}", f"{int(s3.rows_dropped_inf_nan):,}",
            f"{int(s3.rows_unique):,}", f"{int(s3.positive_unique):,}"]) + r" \\")
    (T / "datasets.tex").write_text("\n".join(rows) + "\n")
    # model level
    rows = []
    for run, p in RUNS.items():
        M = R[run]["model"]
        for i, clf in enumerate(CLFS):
            c = M["per_clf"][clf]
            lab = f"\\multirow{{3}}{{*}}{{{RUN_LABEL[run]}}}" if i == 0 else ""
            rows.append(" & ".join([lab, CLF_LABEL[clf], f3(c["in_mean"]), f3(c["in_min"]),
                                    f3(c["cross_mean"]), f"{c['rev']}/12", f"{c['robust']}/12",
                                    f"{f3(c['min'])} ({c['min_cell']})"]) + r" \\")
        rows.append(r"\midrule" if run != "dos_v3" else "")
    (T / "model_level.tex").write_text("\n".join(rows) + "\n")
    # oracle
    rows = []
    for run, p in RUNS.items():
        O = R[run]["oracle"]
        cells = []
        for c in ["auc_base", "auc_qmap", "auc_flip", "auc_qmap_flip", "auc_qmap_randflip"]:
            lo, hi = O["ci_rev"][c]
            cells.append(f"{O['mean_rev'][c]:.3f} [{lo:.2f}, {hi:.2f}] & {O['restored'][c]}")
        rows.append(f"{RUN_LABEL[run]} & {O['n_rev']} & " + " & ".join(cells) + r" \\")
    (T / "oracle.tex").write_text("\n".join(rows) + "\n")
    # tests
    rows = []
    names = {("auc_qmap_flip", "auc_qmap"): "Align+flip vs. align only",
             ("auc_qmap_flip", "auc_qmap_randflip"): "Align+flip vs. align+random flip",
             ("auc_qmap_flip", "auc_base"): "Align+flip vs. original",
             ("auc_qmap", "auc_base"): "Align only vs. original"}
    for run, p in RUNS.items():
        tt = R[run]["oracle"]["tests"]
        tt = tt[tt.scope == "rev"]
        for i, (_, t) in enumerate(tt.iterrows()):
            lab = f"\\multirow{{4}}{{*}}{{{RUN_LABEL[run]}}}" if i == 0 else ""
            rows.append(" & ".join([lab, names[(t.a, t.b)], f"{t.n}", f"{t.med:+.3f}",
                                    f"{t.rrb:.2f}", f"${ptex(t.p_holm)}$"]) + r" \\")
        rows.append(r"\midrule" if run != "dos_v3" else "")
    (T / "tests.tex").write_text("\n".join(rows) + "\n")
    # cluster-level robustness
    rows = []
    for run, p in RUNS.items():
        cl = R[run]["oracle"]["cluster"]
        for i, (_, c) in enumerate(cl.iterrows()):
            lab = f"\\multirow{{4}}{{*}}{{{RUN_LABEL[run]}}}" if i == 0 else ""
            rows.append(" & ".join([lab, names[(c.a, c.b)], f"{c.pair_pos}/{c.n_pairs}",
                                    f"{c.pair_med:+.3f}", f"{c.pair_rrb:.2f}", f"${ptex(c.pair_p_holm)}$",
                                    f"{c.rev_mean:+.3f} [{c.rev_lo:+.2f}, {c.rev_hi:+.2f}]"]) + r" \\")
        rows.append(r"\midrule" if run != "dos_v3" else "")
    (T / "cluster.tex").write_text("\n".join(rows) + "\n")
    # sensitivity
    rows = []
    for run, p in RUNS.items():
        se = R[run]["oracle"]["sens"]
        for i, (_, s) in enumerate(se.iterrows()):
            lab = f"\\multirow{{3}}{{*}}{{{RUN_LABEL[run]}}}" if i == 0 else ""
            rows.append(" & ".join([lab, f"{s.hi:.2f}/{s.lo:.2f}", f"{s.n_flip:.1f}",
                                    f"{s.rev_qf:.3f}", f"{int(s.rest_qf)}/{int(s.n_rev)}",
                                    f"{s.rev_rf:.3f}", f"{int(s.rest_rf)}/{int(s.n_rev)}"]) + r" \\")
        rows.append(r"\midrule" if run != "dos_v3" else "")
    (T / "sensitivity.tex").write_text("\n".join(rows) + "\n")
    # few label
    rows = []
    for run, p in RUNS.items():
        Tf = R[run]["few"]["T"]
        for i, (n, r) in enumerate(Tf.iterrows()):
            lab = f"\\multirow{{{len(Tf)}}}{{*}}{{{RUN_LABEL[run]}}}" if i == 0 else ""
            rows.append(" & ".join([lab, f"{int(n)}", f"{r.flip_precision:.2f}", f"{r.flip_recall:.2f}",
                                    f"{r.auc_qmap_flip_est:.3f}", f"{r.auc_qmap_flip_oracle:.3f}",
                                    f"{r.to_lr:.3f}", f"{r.to_xgboost:.3f}"]) + r" \\")
        rows.append(r"\midrule" if run != "dos_v3" else "")
    (T / "fewlabel.tex").write_text("\n".join(rows) + "\n")
    # appendix: LODO and swap
    rows = []
    for c in CODES:
        cells = []
        for run in RUNS:
            L = R[run]["model"]["lodo"]
            cells += [f3(L.loc[(c, clf)]) for clf in CLFS]
        rows.append(NET_LABEL[c] + " & " + " & ".join(cells) + r" \\")
    (T / "lodo.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for a, b in [(a, b) for a in CODES for b in CODES if a != b]:
        cells = []
        for run in RUNS:
            S = R[run]["model"]["swap_table"]
            r = S.loc[(a, b)]
            cells += [f3(r.cross), f3(r.swap_target_attack), f3(r.swap_target_benign)]
        rows.append(f"{a}$\\rightarrow${b} & " + " & ".join(cells) + r" \\")
    (T / "swap.tex").write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
def heat(ax, M, rows, cols, diag=False, fs=6.5):
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if abs(v - 0.5) > 0.3 else INK,
                        fontweight="bold" if diag and i == j else "normal")
    return im


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figs" / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def figures(R):
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    # Fig: transfer matrices, rows = protocol, cols = classifier
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.3), constrained_layout=True)
    for i, run in enumerate(RUNS):
        for j, clf in enumerate(CLFS):
            ax = axes[i, j]
            M = R[run]["model"]["per_clf"][clf]["matrix"]
            im = heat(ax, M.values, CODES, CODES, diag=True)
            if i == 0:
                ax.set_title(CLF_LABEL[clf], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{RUN_LABEL[run]}\nTraining network", fontsize=7)
            if i == 2:
                ax.set_xlabel("Test network")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("ROC-AUC (< 0.5: reversed ranking)")
    cb.outline.set_visible(False)
    save(fig, "fig_transfer")

    # Fig: feature directions (top features by reversal, NFv2 DoS and NFv3 DoS)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), constrained_layout=True)
    for ax, run in zip(axes, ["dos_v1", "dos_v3"]):
        F = R[run]["feat"]["F"].head(14)
        M = F[[f"auc_{c}" for c in CODES]].values
        im = heat(ax, M, F.feature.str.replace("_", r"\_", regex=False).tolist()
                  if False else F.feature.tolist(), CODES, fs=6)
        ax.set_title(RUN_LABEL[run], fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, ticks=[0, 0.5, 1])
    cb.set_label("Univariate AUC (feature vs. label)")
    cb.outline.set_visible(False)
    save(fig, "fig_features")

    # Fig: oracle diagnostic (reversed cells)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True, constrained_layout=True)
    labels = ["Original", "Marginal\nalign.", "Align. +\nflip\nreversed", "Align. +\nrandom\nflip"]
    for ax, run in zip(axes, RUNS):
        from src.paper import load
        al = load(cfg, run, "alignment.csv")
        cell = al.groupby(["train", "target", "classifier"])[CONDS].mean()
        rev = cell[cell.auc_base < cfg["reversal"]["rev_auc"]]
        cols = ["auc_base", "auc_qmap", "auc_qmap_flip", "auc_qmap_randflip"]
        for _, r in rev.iterrows():
            ax.plot([0, 1, 2], [r.auc_base, r.auc_qmap, r.auc_qmap_flip], color="#b7d3f6", lw=0.7, zorder=1)
        for k, c in enumerate(cols):
            v = rev[c].values
            ax.scatter(np.full(len(v), k) + np.linspace(-0.09, 0.09, len(v)), v, s=9,
                       color=CAT[0] if k < 3 else MUTED, zorder=2, edgecolors="white", linewidths=0.3)
            ax.plot([k - 0.24, k + 0.24], [np.median(v)] * 2, color=INK, lw=1.5, zorder=3)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls="--", zorder=0)
        ax.set_xticks(range(4), labels, fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(f"{RUN_LABEL[run]} (n = {len(rev)})", fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Target ROC-AUC")
    save(fig, "fig_oracle")

    # Fig: few-label
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), sharey=True, layout="constrained")
    for ax, run in zip(axes, RUNS):
        T = R[run]["few"]["T"]
        x = T.index.values
        ax.plot(x, T.auc_qmap_flip_oracle, color=CAT[0], ls="--", lw=1.2, label="Oracle directions")
        ax.plot(x, T.auc_qmap_flip_est, color=CAT[0], marker="o", ms=3.5, lw=1.4,
                label="Directions from n labels")
        ax.fill_between(x, T.ci_lo, T.ci_hi, color=CAT[0], alpha=0.15, lw=0)
        ax.plot(x, T.auc_qmap, color=MUTED, marker="s", ms=3, lw=1.0, label="Marginal alignment only")
        ax.plot(x, T.to_lr, color=CAT[1], marker="^", ms=3.5, lw=1.2, label="Target-only LR (n labels)")
        ax.plot(x, T.to_xgboost, color=CAT[2], marker="D", ms=3, lw=1.2, label="Target-only XGBoost (n labels)")
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(x, [str(int(v)) for v in x])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0.2, 1.02)
        ax.set_xlabel("Labelled target flows (n)")
        ax.set_title(RUN_LABEL[run], fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Target ROC-AUC (inverted cells)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=6.2)
    save(fig, "fig_fewlabel")


if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    OUT.mkdir(parents=True, exist_ok=True)
    R = collect(cfg)
    numbers(R)
    tables(R)
    figures(R)
    print(f"写入 {OUT}（{len(V)} 个数字宏）")
