"""
MST Experiment — Comprehensive Visualization & Extended Analysis
================================================================
Reads participant_summary.csv from the output folder and generates:

 1. REC by boundary position (grouped bar, per group)
 2. LDI by boundary position (grouped bar, per group)
 3. Encoding RT by boundary position (grouped bar, per group)
 4. LDI by lure bin (line plot, per group)
 5. Response proportion breakdown (stacked bar: old/similar/new per stim type × position)
 6. Test-phase RT by position and stim type (aggregated across all participants)
 7. Individual-participant REC & LDI scatter (strip/swarm overlay)
 8. Correlation: encoding accuracy → REC / LDI
 9. Summary dashboard of all key metrics
"""

import os, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.join(_SCRIPT_DIR, "MST_Data")
OUT_DIR = os.path.join(BASE, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DIR, "participant_summary.csv"))

GROUP_ORDER  = ["item_only", "both", "task_only"]
GROUP_LABELS = {"item_only": "Item Only", "both": "Both", "task_only": "Task Only"}
POS_ORDER    = ["pre", "mid", "post"]
PALETTE      = {"item_only": "#4C72B0", "both": "#55A868", "task_only": "#C44E52"}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def grouped_bar(metric_cols, col_labels, ylabel, title, filename,
                group_order=GROUP_ORDER, add_individual=False):
    """
    Generic grouped-bar plot.
    metric_cols: list of column names in df (one per bar cluster member)
    col_labels:  corresponding display labels
    """
    n_groups = len(group_order)
    n_bars   = len(metric_cols)
    x = np.arange(n_groups)
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (col, lab) in enumerate(zip(metric_cols, col_labels)):
        means = [df[df["group"] == g][col].mean() for g in group_order]
        sems  = [df[df["group"] == g][col].sem()  for g in group_order]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=sems,
                       label=lab, capsize=3, edgecolor="white", linewidth=0.5)
        if add_individual:
            for j, g in enumerate(group_order):
                vals = df[df["group"] == g][col].dropna()
                jitter = np.random.normal(0, width * 0.15, len(vals))
                ax.scatter(x[j] + offset + jitter, vals,
                           s=8, alpha=0.25, color=bars[0].get_facecolor(), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in group_order])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    save(fig, filename)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 1: REC by Boundary Position                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n── Generating plots ──")
grouped_bar(
    ["REC_pre", "REC_mid", "REC_post"],
    ["Pre-boundary", "Mid-event", "Post-boundary"],
    "REC  (bias-corrected hit rate)",
    "Recognition Memory (REC) by Event Boundary Position",
    "01_REC_by_position.png",
    add_individual=True,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 2: LDI by Boundary Position                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
grouped_bar(
    ["LDI_pre", "LDI_mid", "LDI_post"],
    ["Pre-boundary", "Mid-event", "Post-boundary"],
    "LDI  (lure discrimination index)",
    "Lure Discrimination (LDI) by Event Boundary Position",
    "02_LDI_by_position.png",
    add_individual=True,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 3: Encoding RT by Boundary Position                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
grouped_bar(
    ["encoding_rt_pre", "encoding_rt_mid", "encoding_rt_post"],
    ["Pre-boundary", "Mid-event", "Post-boundary"],
    "Mean Encoding RT (seconds)",
    "Encoding Response Time by Event Boundary Position",
    "03_encoding_RT_by_position.png",
    add_individual=True,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 4: LDI by Lure Bin (line plot)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
ldi_bin = pd.read_csv(os.path.join(OUT_DIR, "ldi_by_bin.csv"))
fig, ax = plt.subplots(figsize=(7, 5))
for g in GROUP_ORDER:
    sub = ldi_bin[ldi_bin["group"] == g].sort_values("lure_bin")
    ax.errorbar(sub["lure_bin"], sub["mean_LDI"],
                yerr=sub["sd_LDI"] / np.sqrt(sub["n"]),
                label=GROUP_LABELS[g], color=PALETTE[g],
                marker="o", capsize=4, lw=2, markersize=7)
ax.set_xlabel("Lure Bin  (1 = most similar → 5 = least similar)")
ax.set_ylabel("LDI  (lure discrimination index)")
ax.set_title("LDI by Lure Similarity Bin", fontweight="bold")
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(frameon=True)
ax.axhline(0, color="grey", lw=0.5, ls="--")
fig.tight_layout()
save(fig, "04_LDI_by_lure_bin.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 5: Full response proportion breakdown                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Aggregate raw test data across all participants per group
def aggregate_test_responses(group_name, data_dir):
    """Returns a DataFrame with columns: stim_type, position, response, rt.
    For participants with multiple test files (e.g. 00033), uses only the latest."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*_MST_test_*.csv")))
    # Keep only the latest file per PID (matching mst_analysis.py behaviour)
    latest = {}
    for f in all_files:
        pid = os.path.basename(f)[:5]
        latest[pid] = f  # sorted order means last assignment = latest timestamp
    files = list(latest.values())
    rows = []
    for f in files:
        try:
            tdf = pd.read_csv(f, low_memory=False)
        except:
            continue
        t = tdf[tdf["image_path"].notna()].copy()
        if len(t) == 0:
            continue
        pid = os.path.basename(f)[:5]
        def _classify_stim(p):
            p = str(p).replace("\\", "/").strip()
            if p.lower().startswith("foils"):
                return "foil"
            if p.endswith("a.jpg"):
                return "target"
            if p.endswith("b.jpg"):
                return "lure"
            return "other"
        t["stim_type"] = t["image_path"].apply(_classify_stim)
        t["response"] = t["key_resp_3.keys"].astype(str).str.strip().str.lower()
        t["rt"] = pd.to_numeric(t["key_resp_3.rt"], errors="coerce")
        t["position"] = t["position_of_stimuli"].fillna("none").str.strip().str.lower()
        t["group"] = group_name
        t["pid"] = pid
        rows.append(t[["group", "pid", "stim_type", "position", "response", "rt"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

DATA_DIRS = {
    "item_only": os.path.join(BASE, "item_only", "item_only_data"),
    "both":      os.path.join(BASE, "Both_item_task", "both_data"),
    "task_only": os.path.join(BASE, "task_only", "task_only_data"),
}
all_test = pd.concat([aggregate_test_responses(g, d) for g, d in DATA_DIRS.items()], ignore_index=True)

# Response proportion stacked bars: for each group, stim_type × position → old/similar/new
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, g in zip(axes, GROUP_ORDER):
    sub = all_test[all_test["group"] == g].copy()
    categories = []
    for st in ["target", "lure", "foil"]:
        for pos in (["pre", "mid", "post"] if st != "foil" else ["none"]):
            mask = (sub["stim_type"] == st) & (sub["position"] == pos)
            n = mask.sum()
            if n == 0:
                continue
            p_o = (sub.loc[mask, "response"] == "o").sum() / n
            p_s = (sub.loc[mask, "response"] == "s").sum() / n
            p_n = (sub.loc[mask, "response"] == "n").sum() / n
            label = f"{st[:3].title()}\n{pos}" if st != "foil" else "Foil"
            categories.append({"label": label, "old": p_o, "similar": p_s, "new": p_n})

    cdf = pd.DataFrame(categories)
    x = np.arange(len(cdf))
    ax.bar(x, cdf["old"],     0.65, label="Old",     color="#4C72B0")
    ax.bar(x, cdf["similar"], 0.65, bottom=cdf["old"], label="Similar", color="#55A868")
    ax.bar(x, cdf["new"],     0.65, bottom=cdf["old"] + cdf["similar"], label="New", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(cdf["label"], fontsize=9)
    ax.set_title(GROUP_LABELS[g], fontweight="bold")
    ax.set_ylim(0, 1.05)
    if ax == axes[0]:
        ax.set_ylabel("Response Proportion")
    if ax == axes[2]:
        ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Response Distribution by Stimulus Type & Position", fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "05_response_proportions.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 6: Test-phase RT by position × stim type                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, g in zip(axes, GROUP_ORDER):
    sub = all_test[(all_test["group"] == g) & (all_test["stim_type"] != "foil")].copy()
    # Compute per-participant mean RT by position and stim_type, then group-level stats
    prt = sub.groupby(["pid", "position", "stim_type"])["rt"].mean().reset_index()
    for st, color, marker in [("target", "#4C72B0", "o"), ("lure", "#C44E52", "s")]:
        stdf = prt[prt["stim_type"] == st]
        means = [stdf[stdf["position"] == p]["rt"].mean() for p in POS_ORDER]
        sems  = [stdf[stdf["position"] == p]["rt"].sem()  for p in POS_ORDER]
        ax.errorbar(POS_ORDER, means, yerr=sems, label=st.title(),
                    color=color, marker=marker, capsize=4, lw=2, markersize=7)
    ax.set_title(GROUP_LABELS[g], fontweight="bold")
    ax.set_xlabel("Boundary Position")
    if ax == axes[0]:
        ax.set_ylabel("Mean Test RT (seconds)")
    ax.legend(frameon=True, fontsize=9)

fig.suptitle("Test-Phase Response Time by Position & Stimulus Type", fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "06_test_RT_by_position.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 7: Individual REC & LDI (strip + violin)                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, ylabel, title in zip(
    axes,
    ["REC_overall", "LDI_overall"],
    ["REC", "LDI"],
    ["Recognition Memory (REC)", "Lure Discrimination (LDI)"],
):
    plot_df = df[["group", metric]].copy()
    plot_df["group_label"] = plot_df["group"].map(GROUP_LABELS)
    order = [GROUP_LABELS[g] for g in GROUP_ORDER]
    palette_labels = {GROUP_LABELS[g]: PALETTE[g] for g in GROUP_ORDER}

    sns.violinplot(data=plot_df, x="group_label", y=metric, order=order,
                   palette=palette_labels, inner=None, alpha=0.25, ax=ax)
    sns.stripplot(data=plot_df, x="group_label", y=metric, order=order,
                  palette=palette_labels, size=5, alpha=0.6, jitter=True, ax=ax)
    # Add mean line
    for i, g in enumerate(GROUP_ORDER):
        m = df[df["group"] == g][metric].mean()
        ax.hlines(m, i - 0.3, i + 0.3, color="black", lw=2, zorder=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

fig.tight_layout()
save(fig, "07_individual_REC_LDI.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 8: Correlation — encoding accuracy vs REC / LDI                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, mlabel in zip(axes, ["REC_overall", "LDI_overall"], ["REC", "LDI"]):
    for g in GROUP_ORDER:
        gdf = df[df["group"] == g].dropna(subset=["csv_encoding_accuracy", metric])
        ax.scatter(gdf["csv_encoding_accuracy"], gdf[metric],
                   label=GROUP_LABELS[g], color=PALETTE[g], s=30, alpha=0.6)
        # Regression line per group
        if len(gdf) > 2:
            slope, intercept, r, p, _ = stats.linregress(gdf["csv_encoding_accuracy"], gdf[metric])
            xr = np.linspace(gdf["csv_encoding_accuracy"].min(), gdf["csv_encoding_accuracy"].max(), 50)
            ax.plot(xr, slope * xr + intercept, color=PALETTE[g], lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Encoding Task Accuracy")
    ax.set_ylabel(mlabel)
    ax.set_title(f"Encoding Accuracy vs {mlabel}", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)

fig.tight_layout()
save(fig, "08_accuracy_vs_REC_LDI.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 9: Summary dashboard                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 9a – REC by position (simplified)
ax = fig.add_subplot(gs[0, 0])
for g in GROUP_ORDER:
    means = [df[df["group"] == g][f"REC_{p}"].mean() for p in POS_ORDER]
    ax.plot(POS_ORDER, means, marker="o", label=GROUP_LABELS[g], color=PALETTE[g], lw=2)
ax.set_title("REC by Position", fontweight="bold")
ax.set_ylabel("REC")
ax.legend(fontsize=8)
ax.axhline(0, color="grey", lw=0.3, ls="--")

# 9b – LDI by position
ax = fig.add_subplot(gs[0, 1])
for g in GROUP_ORDER:
    means = [df[df["group"] == g][f"LDI_{p}"].mean() for p in POS_ORDER]
    ax.plot(POS_ORDER, means, marker="o", label=GROUP_LABELS[g], color=PALETTE[g], lw=2)
ax.set_title("LDI by Position", fontweight="bold")
ax.set_ylabel("LDI")
ax.legend(fontsize=8)
ax.axhline(0, color="grey", lw=0.3, ls="--")

# 9c – Encoding RT by position
ax = fig.add_subplot(gs[0, 2])
for g in GROUP_ORDER:
    means = [df[df["group"] == g][f"encoding_rt_{p}"].mean() for p in POS_ORDER]
    ax.plot(POS_ORDER, means, marker="o", label=GROUP_LABELS[g], color=PALETTE[g], lw=2)
ax.set_title("Encoding RT by Position", fontweight="bold")
ax.set_ylabel("RT (s)")
ax.legend(fontsize=8)

# 9d – LDI by lure bin
ax = fig.add_subplot(gs[1, 0])
for g in GROUP_ORDER:
    sub = ldi_bin[ldi_bin["group"] == g].sort_values("lure_bin")
    ax.plot(sub["lure_bin"], sub["mean_LDI"], marker="o",
            label=GROUP_LABELS[g], color=PALETTE[g], lw=2)
ax.set_title("LDI by Lure Bin", fontweight="bold")
ax.set_xlabel("Bin (1=hard → 5=easy)")
ax.set_ylabel("LDI")
ax.legend(fontsize=8)
ax.axhline(0, color="grey", lw=0.3, ls="--")

# 9e – Overall REC & LDI bar comparison
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(GROUP_ORDER))
w = 0.3
rec_means = [df[df["group"] == g]["REC_overall"].mean() for g in GROUP_ORDER]
ldi_means = [df[df["group"] == g]["LDI_overall"].mean() for g in GROUP_ORDER]
rec_sems  = [df[df["group"] == g]["REC_overall"].sem()  for g in GROUP_ORDER]
ldi_sems  = [df[df["group"] == g]["LDI_overall"].sem()  for g in GROUP_ORDER]
ax.bar(x - w/2, rec_means, w, yerr=rec_sems, label="REC", color="#4C72B0", capsize=3)
ax.bar(x + w/2, ldi_means, w, yerr=ldi_sems, label="LDI", color="#55A868", capsize=3)
ax.set_xticks(x)
ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
ax.set_title("Overall REC & LDI", fontweight="bold")
ax.legend(fontsize=8)
ax.axhline(0, color="grey", lw=0.3, ls="--")

# 9f – Participant counts & key numbers table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
table_data = []
for g in GROUP_ORDER:
    gdf = df[df["group"] == g]
    table_data.append([
        GROUP_LABELS[g],
        f"{len(gdf)}",
        f"{gdf['REC_overall'].mean():.3f}",
        f"{gdf['LDI_overall'].mean():.3f}",
        f"{gdf['encoding_rt_post'].mean():.2f}s",
    ])
table = ax.table(
    cellText=table_data,
    colLabels=["Group", "n", "REC", "LDI", "RT (post)"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
ax.set_title("Summary Table", fontweight="bold", pad=20)

fig.suptitle("MST Experiment — Results Dashboard", fontsize=14, fontweight="bold", y=0.98)
save(fig, "09_dashboard.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOT 10: Encoding RT across all 40 events (time course)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Show RT as function of trial position 0-6 within event, averaged over events
def aggregate_encoding_rt(group_name, data_dir, scenes_map={}):
    files = sorted(glob.glob(os.path.join(data_dir, "*_MST_task_*.csv")))
    # Keep only the latest file per PID (matching mst_analysis behaviour)
    latest = {}
    for f in files:
        pid = os.path.basename(f)[:5]
        latest[pid] = f  # sorted order means last assignment = latest timestamp
    rows = []
    for pid, f in sorted(latest.items()):
        try:
            tdf = pd.read_csv(f, low_memory=False)
        except:
            continue
        trials = tdf[tdf["image_path"].notna() &
                      tdf["image_path"].astype(str).str.contains("Objects|Scenes", na=False)].copy()
        trials = trials.reset_index(drop=True)
        if len(trials) != 280:
            continue
        trials["pos_in_event"] = trials.index % 7
        for _, r in trials.iterrows():
            if pd.notna(r.get("key_resp_9.rt")):
                rt = float(r["key_resp_9.rt"])
            elif pd.notna(r.get("key_resp_8.rt")):
                rt = 3.0 + float(r["key_resp_8.rt"])
            else:
                rt = np.nan
            rows.append({"group": group_name, "pid": pid,
                         "pos_in_event": int(r["pos_in_event"]), "rt": rt})
    return pd.DataFrame(rows)

enc_all = pd.concat([
    aggregate_encoding_rt("item_only", DATA_DIRS["item_only"]),
    aggregate_encoding_rt("both",      DATA_DIRS["both"]),
    aggregate_encoding_rt("task_only", DATA_DIRS["task_only"]),
], ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 5))
for g in GROUP_ORDER:
    sub = enc_all[enc_all["group"] == g]
    # Mean RT per position within event (across all participants and events)
    per_pid = sub.groupby(["pid", "pos_in_event"])["rt"].mean().reset_index()
    means = per_pid.groupby("pos_in_event")["rt"].mean()
    sems  = per_pid.groupby("pos_in_event")["rt"].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values,
                label=GROUP_LABELS[g], color=PALETTE[g],
                marker="o", capsize=3, lw=2, markersize=6)

ax.set_xlabel("Position Within Event  (0 = post-boundary → 6 = pre-boundary)")
ax.set_ylabel("Mean Encoding RT (seconds)")
ax.set_title("Encoding RT by Within-Event Position", fontweight="bold")
ax.set_xticks(range(7))
ax.set_xticklabels(["0\n(post)", "1", "2", "3\n(mid)", "4", "5", "6\n(pre)"])
ax.legend(frameon=True)
fig.tight_layout()
save(fig, "10_encoding_RT_within_event.png")


print(f"\n✅ All {len(os.listdir(FIG_DIR))} figures saved to {FIG_DIR}/")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BETWEEN-GROUP STATISTICAL COMPARISONS  (the core scientific question)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*70)
print("BETWEEN-GROUP STATISTICAL COMPARISONS")
print("Independent-samples t-tests (two-tailed, uncorrected)")
print("="*70)

PAIRS = [("both", "item_only"), ("task_only", "item_only"), ("both", "task_only")]

def between(label, col_or_series_fn):
    print(f"\n  {label}:")
    for g1, g2 in PAIRS:
        s1 = col_or_series_fn(g1).dropna()
        s2 = col_or_series_fn(g2).dropna()
        t, p = stats.ttest_ind(s1, s2)
        d = (s1.mean() - s2.mean()) / np.sqrt(
            ((len(s1)-1)*s1.std()**2 + (len(s2)-1)*s2.std()**2) / (len(s1)+len(s2)-2)
        )
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {g1:12s} vs {g2:12s}: "
              f"M_diff={s1.mean()-s2.mean():+.4f}  t={t:+.3f}  p={p:.4f}  d={d:+.3f}  [{sig}]")

# --- REC boundary effects (between-group) ---
between("REC effect: post − pre",
        lambda g: df[df["group"]==g]["REC_post"] - df[df["group"]==g]["REC_pre"])

between("REC effect: post − mid",
        lambda g: df[df["group"]==g]["REC_post"] - df[df["group"]==g]["REC_mid"])

between("REC effect: pre − mid  (should be ~0 if post is selectively impaired)",
        lambda g: df[df["group"]==g]["REC_pre"] - df[df["group"]==g]["REC_mid"])

# --- LDI boundary effects (between-group) ---
between("LDI effect: pre − post",
        lambda g: df[df["group"]==g]["LDI_pre"] - df[df["group"]==g]["LDI_post"])

between("LDI effect: pre − mid",
        lambda g: df[df["group"]==g]["LDI_pre"] - df[df["group"]==g]["LDI_mid"])

between("LDI effect: post − mid  (should be ~0 if pre is selectively enhanced)",
        lambda g: df[df["group"]==g]["LDI_post"] - df[df["group"]==g]["LDI_mid"])

# --- RT boundary effects (between-group) ---
between("RT effect: post − mid",
        lambda g: df[df["group"]==g]["encoding_rt_post"] - df[df["group"]==g]["encoding_rt_mid"])

between("RT effect: post − pre",
        lambda g: df[df["group"]==g]["encoding_rt_post"] - df[df["group"]==g]["encoding_rt_pre"])

between("RT effect: pre − mid  (should be ~0 if post is selectively slowed)",
        lambda g: df[df["group"]==g]["encoding_rt_pre"] - df[df["group"]==g]["encoding_rt_mid"])

# --- Overall metrics (between-group) ---
between("Overall REC",
        lambda g: df[df["group"]==g]["REC_overall"])

between("Overall LDI",
        lambda g: df[df["group"]==g]["LDI_overall"])

between("Encoding accuracy (CSV)",
        lambda g: df[df["group"]==g]["csv_encoding_accuracy"])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WITHIN-GROUP PAIRWISE POSITION TESTS  (paired-samples t-tests)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def within_pairwise(label, metric_template):
    """
    For each group, run all 3 pairwise paired t-tests between positions.
    metric_template is a format string like 'REC_{}' or 'LDI_{}'.
    Tests: post vs pre, post vs mid, pre vs mid.
    """
    print(f"\n{'='*70}")
    print(f"WITHIN-GROUP PAIRWISE: {label}")
    print(f"  Paired-samples t-tests (two-tailed)")
    print(f"{'='*70}")
    pairs = [("post", "pre"), ("post", "mid"), ("pre", "mid")]
    for g in GROUP_ORDER:
        gdf = df[df["group"] == g]
        print(f"\n  {g.upper()}  (n={len(gdf)})")
        for p1, p2 in pairs:
            col1 = metric_template.format(p1)
            col2 = metric_template.format(p2)
            diff = (gdf[col1] - gdf[col2]).dropna()
            if len(diff) < 2:
                print(f"    {p1:5s} vs {p2:5s}: insufficient data")
                continue
            t_val, p_val = stats.ttest_1samp(diff, 0)
            d_val = diff.mean() / diff.std()
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"    {p1:5s} vs {p2:5s}: "
                  f"M_diff={diff.mean():+.4f}  t={t_val:+.3f}  p={p_val:.4f}  d={d_val:+.3f}  [{sig}]")

# REC: expect post < mid ≈ pre  (post-boundary impairment)
within_pairwise("REC by position  (expect: post < mid ≈ pre)", "REC_{}")

# LDI: expect pre > mid ≈ post  (pre-boundary enhancement)
within_pairwise("LDI by position  (expect: pre > mid ≈ post)", "LDI_{}")

# Encoding RT: expect post > mid ≈ pre  (boundary slowing)
within_pairwise("Encoding RT by position  (expect: post > mid ≈ pre)", "encoding_rt_{}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PER-GROUP MEAN BOUNDARY EFFECTS  (post − pre difference per group)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def per_group_boundary_effects():
    """
    For each metric (REC, LDI, Encoding RT), compute each group's mean
    (post − pre) difference and print a summary table.
    """
    metrics = [
        ("REC",         "REC_post",         "REC_pre"),
        ("LDI",         "LDI_post",         "LDI_pre"),
        ("Encoding RT", "encoding_rt_post", "encoding_rt_pre"),
    ]
    print("\n" + "="*70)
    print("PER-GROUP MEAN BOUNDARY EFFECTS  (post − pre)")
    print("="*70)
    for label, post_col, pre_col in metrics:
        print(f"\n  {label}:")
        for g in GROUP_ORDER:
            gdf = df[df["group"] == g]
            diff = (gdf[post_col] - gdf[pre_col]).dropna()
            m = diff.mean()
            se = diff.std() / np.sqrt(len(diff))
            print(f"    {g:12s}  (n={len(diff):3d}):  "
                  f"M(post−pre) = {m:+.4f}  (SE = {se:.4f})")

per_group_boundary_effects()
