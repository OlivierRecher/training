"""Plots comparing the Original vs Relaxed activity-cell definitions.

Reads the slim join produced by prepare_comparison_data.py and answers the
question: do the confronted per-day counts refer to the SAME cells?

Figures (saved to MyResults/plots/, weekends in red on every x-axis):
  G1  Original vs Relaxed counts per day, per merge mode  -> reproduces the table
  G2  "same cell?" decomposition per day, per merge mode  -> the actual answer
  G3  agreement rate on the shared population, per mode
  G4  composition of the Relaxed-only surplus (sedentary / third place / no home)
  +   a 2x2 summary figure (No-merge headline)

Run (from Stage_2A/):  python MyPython/plot_comparison_definitions.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from utils import DAYS, is_weekend

MAIN_DIR = Path(__file__).resolve().parent.parent
CACHE = MAIN_DIR / "MyResults" / "intermediate" / "comparison_activity.csv"
PLOT_DIR = MAIN_DIR / "MyResults" / "plots"

MODES = [("nm", "No merge"), ("sm", "Simple merge"), ("2g", "2G/3G merge")]

# colours
C_ORIG = "#E08214"      # orange (echoes the screenshot's Original block)
C_RLX = "#8C510A"       # brown  (echoes the Relaxed block)
C_SAME = "#2CA02C"      # same cell
C_DIFF = "#D62728"      # different cell
C_EXTRA = "#1F77B4"     # relaxed-only surplus
C_HOME = "#9467BD"      # surplus that is activity=home (sedentary)
C_DIST = "#17BECF"      # surplus on a distinct cell, home known
C_NOHOME = "#BCBD22"    # surplus on a distinct cell, no home identified
X = np.arange(len(DAYS))


def _redden_weekends(ax):
    ax.set_xticks(X)
    ax.set_xticklabels(DAYS, rotation=90, fontsize=8)
    for lbl in ax.get_xticklabels():
        if is_weekend(lbl.get_text()):
            lbl.set_color("#ff0000")
            lbl.set_fontweight("bold")


def load_aggregates():
    """Return {mode_key: per-day DataFrame of the derived counts}."""
    df = pd.read_csv(CACHE, sep=";", dtype=str, keep_default_na=False, na_filter=False)
    same_home = df["rlx_same_home"] == "True"
    home_known = df["rlx_home"] != ""
    out = {}
    for key, _ in MODES:
        oc = df[f"orig_{key}"]
        rc = df[f"rlx_{key}"]
        onb = df[f"orig_{key}_nb"].astype(int)
        orig_has = oc != ""
        rlx_has = rc != ""
        extra = ~orig_has & rlx_has
        agg = pd.DataFrame({
            "day": df["day"],
            "table_orig": (onb == 1),               # table rule: exactly 1 activity cell
            "table_rlx": rlx_has,                    # table rule: >=1
            "both_same": orig_has & rlx_has & (oc == rc),
            "both_diff": orig_has & rlx_has & (oc != rc),
            "orig_only": orig_has & ~rlx_has,
            "relax_only": extra,
            "extra_home": extra & same_home,                         # activity = home
            "extra_dist_home": extra & ~same_home & home_known,      # activity != home (home known)
            "extra_dist_nohome": extra & ~same_home & ~home_known,   # activity, no home identified
        })
        per_day = agg.groupby("day").sum(numeric_only=True).reindex(DAYS).fillna(0).astype(int)
        out[key] = per_day
    return out


# ------------------------------------------------------------------ G1 -------
def plot_counts(agg, ax, key, label):
    d = agg[key]
    w = 0.4
    ax.bar(X - w / 2, d["table_orig"], w, label="Original (continuous)", color=C_ORIG)
    ax.bar(X + w / 2, d["table_rlx"], w, label="Relaxed (discontinuous)", color=C_RLX)
    ax.set_title(f"Daily counts — {label}", fontsize=10)
    ax.set_ylabel("number of users")
    ax.legend(fontsize=8)
    _redden_weekends(ax)


# ------------------------------------------------------------------ G2 -------
def plot_decomposition(agg, ax, key, label):
    d = agg[key]
    ax.bar(X, d["both_same"], label="same cell", color=C_SAME)
    ax.bar(X, d["both_diff"], bottom=d["both_same"], label="different cell", color=C_DIFF)
    base = d["both_same"] + d["both_diff"]
    ax.bar(X, d["relax_only"], bottom=base, label="Relaxed only (surplus)", color=C_EXTRA)
    if d["orig_only"].sum() > 0:
        ax.bar(X, d["orig_only"], bottom=base + d["relax_only"],
               label="Original only", color="black")
    # dashed line = Original population (>=1 activity cell) sitting inside Relaxed
    ax.plot(X, base, "k--", lw=1, label="Original total (≥1)")
    ax.set_title(f"Same cell? — {label}", fontsize=10)
    ax.set_ylabel("number of users")
    ax.legend(fontsize=7)
    _redden_weekends(ax)


# ------------------------------------------------------------------ G3 -------
def plot_agreement(agg, ax):
    styles = {"nm": ("o-", C_SAME), "sm": ("s-", C_ORIG), "2g": ("^-", C_RLX)}
    for key, label in MODES:
        d = agg[key]
        both = d["both_same"] + d["both_diff"]
        pct = 100 * d["both_same"] / both.replace(0, np.nan)
        fmt, col = styles[key]
        ax.plot(X, pct, fmt, color=col, label=label, markersize=5)
    ax.set_title("Agreement rate on the shared population", fontsize=10)
    ax.set_ylabel("% same cell")
    ax.set_ylim(90, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _redden_weekends(ax)


# ------------------------------------------------------------------ G4 -------
def plot_surplus(agg, ax, key, label):
    d = agg[key]
    ax.bar(X, d["extra_home"], label="activity = home (sedentary)", color=C_HOME)
    b1 = d["extra_home"]
    ax.bar(X, d["extra_dist_home"], bottom=b1,
           label="activity ≠ home (home known)", color=C_DIST)
    b2 = b1 + d["extra_dist_home"]
    ax.bar(X, d["extra_dist_nohome"], bottom=b2,
           label="activity, no home identified", color=C_NOHOME)
    ax.set_title(f"Composition of the Relaxed-only surplus — {label}", fontsize=10)
    ax.set_ylabel("number of users (Relaxed only)")
    ax.legend(fontsize=8)
    _redden_weekends(ax)


NOTE = ("Note: G1 reproduces the table (Original = users with exactly 1 activity cell; "
        "Relaxed = ≥1). G2-G4 use a consistent ≥1 basis on both sides "
        "(Original ≥1 ≈ +0.7% vs the table).")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    agg = load_aggregates()

    # G1
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    for ax, (key, label) in zip(axes, MODES):
        plot_counts(agg, ax, key, label)
    fig.suptitle("G1 — Activity-cell counts: Original vs Relaxed", fontsize=13)
    fig.savefig(PLOT_DIR / "G1_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # G2
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    for ax, (key, label) in zip(axes, MODES):
        plot_decomposition(agg, ax, key, label)
    fig.suptitle("G2 — Do the confronted counts refer to the same cells?", fontsize=13)
    fig.text(0.5, -0.01, NOTE, ha="center", fontsize=8, style="italic")
    fig.savefig(PLOT_DIR / "G2_same_cell_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # G3
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    plot_agreement(agg, ax)
    fig.savefig(PLOT_DIR / "G3_agreement_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # G4
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    for ax, (key, label) in zip(axes, MODES):
        plot_surplus(agg, ax, key, label)
    fig.suptitle("G4 — Who are the extra users captured by the Relaxed definition?",
                 fontsize=13)
    fig.savefig(PLOT_DIR / "G4_surplus_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Combined summary (No-merge headline)
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    plot_counts(agg, axes[0, 0], "nm", "No merge")
    plot_decomposition(agg, axes[0, 1], "nm", "No merge")
    plot_agreement(agg, axes[1, 0])
    plot_surplus(agg, axes[1, 1], "nm", "No merge")
    fig.suptitle("Original vs Relaxed — summary (No-merge)", fontsize=14)
    fig.text(0.5, -0.01, NOTE, ha="center", fontsize=8, style="italic")
    fig.savefig(PLOT_DIR / "comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # console recap
    print("Saved to", PLOT_DIR)
    nm = agg["nm"]
    both = nm["both_same"].sum() + nm["both_diff"].sum()
    print(f"[No-merge] agreement on overlap : {100*nm['both_same'].sum()/both:.2f}%")
    print(f"[No-merge] orig_only total      : {nm['orig_only'].sum()}")
    print(f"[No-merge] relax_only surplus   : {nm['relax_only'].sum()}")
    print(f"           - activity=home             : {nm['extra_home'].sum()}")
    print(f"           - activity!=home (home known): {nm['extra_dist_home'].sum()}")
    print(f"           - no home identified         : {nm['extra_dist_nohome'].sum()}")


if __name__ == "__main__":
    main()
