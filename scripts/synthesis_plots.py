# -*- coding: utf-8 -*-
"""Cross-experiment synthesis plots — all labels in English for font compatibility.

Output: data/results/_synthesis/
"""
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Fix stdout encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/results/_synthesis"
os.makedirs(OUT, exist_ok=True)

# Try to find a CJK-capable font
_font = None
for _fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    try:
        from matplotlib.font_manager import findfont, FontProperties
        _fp = FontProperties(family=_fname)
        _candidate = findfont(_fp, fallback_to_default=False)
        if _candidate:
            _font = _fname
            break
    except Exception:
        continue

if _font:
    plt.rcParams["font.family"] = _font
    plt.rcParams["axes.unicode_minus"] = False
    USE_CJK = True
else:
    USE_CJK = False

def _t(key):
    """Translate key to display label. English always works; CJK if font available."""
    en = {
        # Exp 0
        "Full": "Full",
        "Abl-1": "Abl-1 (-Encoding)",
        "Abl-2": "Abl-2 (-Penalty)",
        "Abl-3": "Abl-3 (-Zoom)",
        "Abl-4": "Abl-4 (Baseline)",
        # Exp 2
        "PenaltyFlex": "PenaltyFlex",
        "Grid-Search": "Grid-Search",
        "Linear": "Linear",
        "Fixed(λ=1)": "Fixed(λ=1)",
        "Fixed(λ=10)": "Fixed(λ=10)",
        "Fixed(λ=100)": "Fixed(λ=100)",
        # Exp 3
        "ParetoZoom": "ParetoZoom",
        "Uniform Grid": "Uniform Grid",
        "NSGA-II": "NSGA-II",
        "Random": "Random",
    }
    cn = {
        "Full": "Full (全开)",
        "Abl-1": "Abl-1 (−编码)",
        "Abl-2": "Abl-2 (−惩罚)",
        "Abl-3": "Abl-3 (−探索)",
        "Abl-4": "Abl-4 (基线)",
        "PenaltyFlex": "PenaltyFlex (自适应)",
        "Grid-Search": "Grid-Search (网格)",
        "Linear": "Linear (线性)",
        "Fixed(λ=1)": "Fixed(λ=1)",
        "Fixed(λ=10)": "Fixed(λ=10)",
        "Fixed(λ=100)": "Fixed(λ=100)",
        "ParetoZoom": "ParetoZoom (五阶段)",
        "Uniform Grid": "Uniform Grid (均匀)",
        "NSGA-II": "NSGA-II (遗传)",
        "Random": "Random (随机)",
    }
    return cn.get(key, key) if USE_CJK else en.get(key, key)

# Load data
exp0 = json.load(open(ROOT / "data/results/formal_exp0_reps20_reform/summary.json", "r", encoding="utf-8"))
exp2 = json.load(open(ROOT / "data/results/formal_exp2_reps20/summary.json", "r", encoding="utf-8"))
exp3 = json.load(open(ROOT / "data/results/formal_exp3_reps20/summary.json", "r", encoding="utf-8"))

e0 = exp0["experiments"]["experiment_0"]
e2 = exp2["experiments"]["experiment_2"]
e3 = exp3["experiments"]["experiment_3"]

def hv_means(exp_data):
    m = exp_data["metrics"]["HV"]
    keys = exp_data.get("groups", exp_data.get("configs", []))
    return {k: (m[k]["mean"], m[k]["std"]) for k in keys}

abl_hv = hv_means(e0)
pen_hv = hv_means(e2)
exp_hv = hv_means(e3)
abl_order = ["Full", "Abl-1", "Abl-2", "Abl-3", "Abl-4"]
pen_order = e2["groups"]
exp_order = e3["groups"]

# Colors
C = {
    "full": "#2c7bb6", "abl": "#d7191c", "best": "#1a9850",
    "other": "#abd9e9", "pen": "#fdae61", "zoom": "#5e4fa2",
    "nsga": "#d73027", "rand": "#a6d96a", "grid": "#ffffbf",
}

# ═══════════════════════════════════════════════
# Fig 1: 3-panel dashboard
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
title_suffix = " (CN font)" if USE_CJK else ""
fig.suptitle(f"AP3-QUBO Synthesis Dashboard{title_suffix}", fontsize=16, fontweight="bold", y=0.98)

# 1A: Ablation
ax = axes[0, 0]
labels_a = [_t(k) for k in abl_order]
means_a = [abl_hv[k][0] for k in abl_order]
stds_a = [abl_hv[k][1] for k in abl_order]
colors_a = [C["full"] if k == "Full" else C["abl"] for k in abl_order]
bars = ax.bar(range(len(labels_a)), means_a, yerr=stds_a, color=colors_a,
              edgecolor="white", linewidth=0.8, capsize=5)
ax.set_xticks(range(len(labels_a)))
ax.set_xticklabels(labels_a, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Hypervolume", fontsize=11)
title0 = "Exp 0: Ablation" if not USE_CJK else "Exp 0: 消融实验"
ax.set_title(title0, fontsize=13, fontweight="bold")
# Annotate significance (from Wilcoxon Bonferroni)
contrib = e0.get("contributions_AR_Synergy", {})
wilcoxon = contrib.get("Wilcoxon_Bonferroni", {}).get("comparisons", {})
for i, k in enumerate(abl_order):
    if k != "Full" and f"Full_vs_{k}" in wilcoxon:
        if wilcoxon[f"Full_vs_{k}"].get("significant"):
            ax.annotate("***", (i, means_a[i] + stds_a[i] + 200), ha="center",
                        fontsize=14, color="red", fontweight="bold")
for bar, v in zip(bars, means_a):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{v:.0f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(0, max(means_a) * 1.25)
ax.grid(axis="y", alpha=0.3)

# 1B: PenaltyFlex
ax = axes[0, 1]
pen_sorted = sorted([(k, pen_hv[k][0], pen_hv[k][1]) for k in pen_order],
                    key=lambda x: x[1], reverse=True)
labels_p = [_t(k) for k, _, _ in pen_sorted]
means_p = [m for _, m, _ in pen_sorted]
stds_p = [s for _, _, s in pen_sorted]
colors_p = [C["best"] if i == 0 else C["other"] for i in range(len(labels_p))]
bars = ax.bar(range(len(labels_p)), means_p, yerr=stds_p, color=colors_p,
              edgecolor="white", linewidth=0.8, capsize=5)
ax.set_xticks(range(len(labels_p)))
ax.set_xticklabels(labels_p, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Hypervolume", fontsize=11)
title2 = "Exp 2: PenaltyFlex" if not USE_CJK else "Exp 2: PenaltyFlex 对比"
ax.set_title(title2, fontsize=13, fontweight="bold")
for bar, v in zip(bars, means_p):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60, f"{v:.0f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(0, max(means_p) * 1.2)
ax.grid(axis="y", alpha=0.3)

# 1C: ParetoZoom
ax = axes[1, 0]
exp_sorted = sorted([(k, exp_hv[k][0], exp_hv[k][1]) for k in exp_order],
                    key=lambda x: x[1], reverse=True)
labels_e = [_t(k) for k, _, _ in exp_sorted]
means_e = [m for _, m, _ in exp_sorted]
stds_e = [s for _, _, s in exp_sorted]
colors_e = [C["best"] if i == 0 else C["other"] for i in range(len(labels_e))]
bars = ax.bar(range(len(labels_e)), means_e, yerr=stds_e, color=colors_e,
              edgecolor="white", linewidth=0.8, capsize=5)
ax.set_xticks(range(len(labels_e)))
ax.set_xticklabels(labels_e, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Hypervolume", fontsize=11)
title3 = "Exp 3: ParetoZoom" if not USE_CJK else "Exp 3: ParetoZoom 对比"
ax.set_title(title3, fontsize=13, fontweight="bold")
for bar, v in zip(bars, means_e):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{v:.0f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(0, max(means_e) * 1.2)
ax.grid(axis="y", alpha=0.3)

# 1D: Contribution rates
ax = axes[1, 1]
ar_items = [
    ("PrecisionSplit", contrib.get("AR_PrecisionSplit", 0)),
    ("PenaltyFlex",   contrib.get("AR_PenaltyFlex", 0)),
    ("ParetoZoom",    contrib.get("AR_ParetoZoom", 0)),
    ("Synergy",       contrib.get("Synergy", 0)),
]
ar_labels, ar_vals = zip(*ar_items)
ar_colors = [C["abl"] if v < 0 else C["best"] for v in ar_vals]
bars = ax.barh(range(len(ar_labels)), ar_vals, color=ar_colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(ar_labels)))
ax.set_yticklabels(ar_labels, fontsize=11)
ax.set_xlabel("Attribution Rate AR (%)", fontsize=11)
title_ar = "Innovation Contribution (vs Full)" if not USE_CJK else "各创新贡献率 (相对 Full)"
ax.set_title(title_ar, fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8)
for bar, v in zip(bars, ar_vals):
    xpos = bar.get_width() + 0.3 if v >= 0 else bar.get_width() - 0.3
    ha = "left" if v >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f"{v:.1f}%",
            va="center", ha=ha, fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] summary_dashboard.png")

# ═══════════════════════════════════════════════
# Fig 2: Ablation waterfall
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
vals = [abl_hv[k][0] for k in abl_order]
errs = [abl_hv[k][1] for k in abl_order]
deltas = [0] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
x_pos = np.arange(len(abl_order))
colors_wf = [C["full"]] + [C["abl"]] * (len(abl_order) - 1)

bars = ax.bar(x_pos, vals, yerr=errs, color=colors_wf, edgecolor="white",
              linewidth=1, capsize=6, width=0.6)

# Delta arrows
for i in range(1, len(vals)):
    y_top = max(vals[i], vals[i-1]) + 250
    ax.annotate("", xy=(i-1, y_top), xytext=(i, y_top),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    sign = "+" if deltas[i] > 0 else ""
    clr = C["best"] if deltas[i] > 0 else C["abl"]
    ax.text(i - 0.5, y_top + 80, f"{sign}{deltas[i]:.0f} HV", ha="center", fontsize=9,
            color=clr, fontweight="bold")

for bar, v, e in zip(bars, vals, errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 80,
            f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels([_t(k) for k in abl_order], fontsize=10)
ax.set_ylabel("Hypervolume", fontsize=12)
title_wf = "Ablation Waterfall" if not USE_CJK else "消融实验 HV 瀑布图"
ax.set_title(title_wf, fontsize=14, fontweight="bold")
ax.set_ylim(min(vals) * 0.85, max(vals) * 1.15 + 600)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "ablation_waterfall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] ablation_waterfall.png")

# ═══════════════════════════════════════════════
# Fig 3: Front size comparison
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fs = e0["metrics"]["Front Size"]
fs_vals = [fs[k]["mean"] for k in abl_order]
fs_errs = [fs[k]["std"] for k in abl_order]
colors_fs = [C["full"] if k == "Full" else C["abl"] for k in abl_order]

bars = ax.bar(range(len(abl_order)), fs_vals, yerr=fs_errs, color=colors_fs,
              edgecolor="white", linewidth=1, capsize=6)
ax.set_xticks(range(len(abl_order)))
ax.set_xticklabels([_t(k) for k in abl_order], fontsize=10)
ax.set_ylabel("Pareto Front Size", fontsize=12)
title_fs = "Front Size: PenaltyFlex Removed → 40% Drop" if not USE_CJK \
    else "前沿规模: 移除 PenaltyFlex → 暴跌 40%"
ax.set_title(title_fs, fontsize=13, fontweight="bold")
for bar, v, e in zip(bars, fs_vals, fs_errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.5,
            f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
# Arrow for the big drop
ax.annotate("-40%" if not USE_CJK else "-40%",
            xy=(2, fs_vals[2]), xytext=(3, fs_vals[0] + 2),
            arrowprops=dict(arrowstyle="->", color="red", lw=2), fontsize=12,
            color="red", fontweight="bold")
ax.set_ylim(0, max(fs_vals) * 1.3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "ablation_front_size.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] ablation_front_size.png")

# ═══════════════════════════════════════════════
# Fig 4: Cross-experiment HV ranking
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
all_labels = []
all_means = []
all_stds = []
all_colors = []

# PenaltyFlex (Exp 2)
for k, m, s in pen_sorted:
    all_labels.append(_t(k))
    all_means.append(m)
    all_stds.append(s)
    all_colors.append(C["pen"])

# Ablation (Exp 0)
for k in abl_order:
    all_labels.append(_t(k))
    all_means.append(abl_hv[k][0])
    all_stds.append(abl_hv[k][1])
    all_colors.append(C["full"] if k == "Full" else C["abl"])

# ParetoZoom (Exp 3)
for k, m, s in exp_sorted:
    all_labels.append(_t(k))
    all_means.append(m)
    all_stds.append(s)
    all_colors.append(C["zoom"])

y_pos = range(len(all_labels))
bars = ax.barh(y_pos, all_means, xerr=all_stds, color=all_colors,
               edgecolor="white", height=0.7, capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(all_labels, fontsize=9)
ax.set_xlabel("Hypervolume", fontsize=12)
title_all = "Cross-Experiment HV Ranking" if not USE_CJK else "跨实验 HV 总排名"
ax.set_title(f"{title_all} (n=20, mean +/- std)", fontsize=14, fontweight="bold")

legend_elements = [
    Patch(facecolor=C["full"], label="Exp 0: Ablation"),
    Patch(facecolor=C["pen"], label="Exp 2: PenaltyFlex"),
    Patch(facecolor=C["zoom"], label="Exp 3: ParetoZoom"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
for bar, v in zip(bars, all_means):
    ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
            f"{v:.0f}", va="center", fontsize=7.5)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "cross_exp_hv_ranking.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] cross_exp_hv_ranking.png")

print(f"\nAll plots saved to: {OUT}")
for f in sorted(OUT.glob("*.png")):
    kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({kb} KB)")
print("Done.")
