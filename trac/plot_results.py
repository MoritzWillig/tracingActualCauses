#!/usr/bin/env python3
"""
plot_results.py  –  Visualize TrAC benchmark results.

Scans a results directory for per-cell CSV files (n*_d*.csv written by
benchmark_trac.py) and produces three figures plus a statistics table:

  <stem>.png          boxplot of all runs + individual dots, one panel per density
  <stem>_mean.png     mean ± stddev line plot across node counts
  <stem>_pruning.png  Remark-6 pruning rates (node-level + partition-level)
  <stem>_stats.csv    per-(n_nodes, density) summary incl. pruning counts

Timed-out rows are excluded from timing plots but counted in the stats table.
Pruning stats are only available from runs using the instrumented trac().

Usage
-----
    python trac/plot_results.py
    python trac/plot_results.py --results-dir results --output results/plot.png

Dependencies: matplotlib, numpy
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as stats_mod
import sys
from collections import defaultdict
from typing import Dict, List, Optional

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "data")

_FIG_FMT = "pdf"   # output format for all figures ("pdf", "png", "svg", …)

_FONT_SCALE = 2.6  # multiply every font size in all figures by this factor
_LOG_YTICK_SCALE = 1.3  # extra multiplier for y-axis tick labels in log-scale figures (trac, trac_compare, trac_compare_max)
_LOG_XTICK_SCALE = 1.3  # extra multiplier for x-axis tick labels in log-scale figures (trac, trac_compare, trac_compare_max)
_COMPARE_YLABEL_SCALE = 0.88  # scale for y-axis label in trac_compare and trac_compare_max
_PRUNING_YTICK_SCALE = _LOG_YTICK_SCALE * 0.8  # y-axis tick label scale for trac_pruning
_FIG_WIDTH_SCALE = 1.5  # width multiplier for trac, trac_compare, trac_compare_max


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_results(results_dir: str) -> List[Dict]:
    """
    Scan *results_dir* for per-cell CSV files matching n*_d*.csv.

    Returns completed (non-timeout, non-error) rows only, deduped by seed.
    Used by the timing plots.
    """
    rows: List[Dict] = []
    seen_seeds: set  = set()

    for path in sorted(glob.glob(os.path.join(results_dir, "n*_d*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                seed = row.get("seed", "")
                if seed in seen_seeds:
                    continue
                seen_seeds.add(seed)
                if row.get("timed_out", "").lower() in ("true", "1"):
                    continue
                if row.get("error", "").strip():
                    continue
                def _bf(key):
                    v = row.get(key, "")
                    return float(v) if v not in ("", None) else -1.0

                rows.append({
                    "n_nodes":     int(row["n_nodes"]),
                    "density":     float(row["density"]),
                    "elapsed_s":   float(row["elapsed_s"]),
                    "n_causes":    int(row["n_causes"]),
                    "rep":         int(row["rep"]),
                    # pruning counters – default 0 for old CSVs without them
                    "n_cands":     int(row.get("n_cands",     0) or 0),
                    "n_skip_sup":  int(row.get("n_skip_sup",  0) or 0),
                    "n_skip_up":   int(row.get("n_skip_up",   0) or 0),
                    "n_part_skip": int(row.get("n_part_skip", 0) or 0),
                    "n_part_test": int(row.get("n_part_test", 0) or 0),
                    # baseline timings (-1 = unavailable / error)
                    "t_beam_s":    _bf("t_beam_s"),
                    "t_isi_s":     _bf("t_isi_s"),
                    # HP2SAT strategies
                    "t_bf_s":      _bf("t_bf_s"),
                    "t_sat_s":     _bf("t_sat_s"),
                    "t_ms_s":      _bf("t_ms_s"),
                    "t_ilp_s":     _bf("t_ilp_s"),
                })

    return rows


def compute_stats(results_dir: str) -> List[Dict]:
    """
    Read all per-cell CSV files and compute per-(n_nodes, density) statistics.

    Aggregates timing stats (mean, std) and Remark-6 pruning counts.
    Timed-out rows are counted but excluded from timing/pruning averages.
    Deduplicates by seed.
    """
    cells: Dict = defaultdict(lambda: {
        "times": [], "n_timed_out": 0,
        "n_cands": [], "n_skip_sup": [], "n_skip_up": [],
        "n_part_skip": [], "n_part_test": [],
        "t_beam": [], "t_isi": [],
        "t_bf": [], "t_sat": [], "t_ms": [], "t_ilp": [],
    })
    seen_seeds: set = set()

    for path in sorted(glob.glob(os.path.join(results_dir, "n*_d*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                seed = row.get("seed", "")
                if seed in seen_seeds:
                    continue
                seen_seeds.add(seed)

                key = (int(row["n_nodes"]), float(row["density"]))
                if row.get("timed_out", "").lower() in ("true", "1"):
                    cells[key]["n_timed_out"] += 1
                elif not row.get("error", "").strip():
                    cells[key]["times"].append(float(row["elapsed_s"]))
                    # pruning counters: skip rows where they weren't recorded (-1)
                    for field in ("n_cands", "n_skip_sup", "n_skip_up",
                                  "n_part_skip", "n_part_test"):
                        val = int(row.get(field, -1) or -1)
                        if val >= 0:
                            cells[key][field].append(val)
                    # baseline timings: include only successful runs (>= 0)
                    for csv_col, cell_key in (
                        ("t_beam_s", "t_beam"),
                        ("t_isi_s",  "t_isi"),
                        ("t_bf_s",   "t_bf"),
                        ("t_sat_s",  "t_sat"),
                        ("t_ms_s",   "t_ms"),
                        ("t_ilp_s",  "t_ilp"),
                    ):
                        raw = row.get(csv_col, "")
                        if raw not in ("", None):
                            t = float(raw)
                            if t >= 0:
                                cells[key][cell_key].append(t)

    result = []
    for (n, d), data in sorted(cells.items()):
        times = data["times"]

        def _mean(lst):
            return round(stats_mod.mean(lst), 4) if lst else None

        mn_cands    = _mean(data["n_cands"])
        mn_skip_sup = _mean(data["n_skip_sup"])
        mn_skip_up  = _mean(data["n_skip_up"])
        mn_part_skip = _mean(data["n_part_skip"])
        mn_part_test = _mean(data["n_part_test"])

        # Node skip fraction: exact from collected counters.
        if mn_cands and mn_cands > 0 and mn_skip_sup is not None and mn_skip_up is not None:
            node_skip_frac = round((mn_skip_sup + mn_skip_up) / mn_cands, 4)
        else:
            node_skip_frac = None

        # Partition skip fraction: exact *within visited nodes only*.
        # Partitions inside fully-skipped nodes are never enumerated,
        # so the true denominator would require knowing their sizes.
        if mn_part_skip is not None and mn_part_test is not None:
            denom = mn_part_skip + mn_part_test
            part_skip_frac = round(mn_part_skip / denom, 4) if denom > 0 else None
        else:
            part_skip_frac = None

        result.append({
            "n_nodes":        n,
            "density":        d,
            "n_completed":    len(times),
            "mean_s":         round(stats_mod.mean(times),  6) if times else None,
            "std_s":          round(stats_mod.stdev(times), 6) if len(times) > 1
                              else (0.0 if times else None),
            "n_timed_out":    data["n_timed_out"],
            "mean_n_cands":   mn_cands,
            "mean_skip_sup":  mn_skip_sup,
            "mean_skip_up":   mn_skip_up,
            "mean_part_skip": mn_part_skip,
            "mean_part_test": mn_part_test,
            # derived fractions (no algorithm changes needed)
            "node_skip_frac": node_skip_frac,
            "part_skip_frac_visited": part_skip_frac,
            # baseline mean timings (None if no data)
            "mean_beam_s":    _mean(data["t_beam"]),
            "mean_isi_s":     _mean(data["t_isi"]),
            "mean_bf_s":      _mean(data["t_bf"]),
            "mean_sat_s":     _mean(data["t_sat"]),
            "mean_ms_s":      _mean(data["t_ms"]),
            "mean_ilp_s":     _mean(data["t_ilp"]),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

_STATS_FIELDS = [
    "n_nodes", "density",
    "n_completed", "mean_s", "std_s", "n_timed_out",
    "mean_n_cands", "mean_skip_sup", "mean_skip_up",
    "mean_part_skip", "mean_part_test",
    # derived fractions (exact from collected counters, no algorithm changes)
    "node_skip_frac",           # (skip_sup + skip_up) / n_cands
    "part_skip_frac_visited",   # part_skip / (part_skip + part_test), visited nodes only
    # baseline mean timings (None if unavailable / all errors)
    "mean_beam_s",
    "mean_isi_s",
    "mean_bf_s",
    "mean_sat_s",
    "mean_ms_s",
    "mean_ilp_s",
]


def _save_stats_csv(stats: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_STATS_FIELDS)
        w.writeheader()
        w.writerows(stats)
    print(f"Stats     → {path}")


def _save_titles_txt(titles: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for t in titles:
            f.write(t.replace("\n", " ") + "\n")
    print(f"Titles    → {path}")


def _plot_stem(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return base


def _mpl():
    """Import matplotlib in non-interactive mode; exit on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        return plt, np, mticker
    except ImportError as exc:
        print(f"Cannot plot – missing dependency: {exc}")
        sys.exit(1)


_COLORS = ["#1976D2", "#F57C00", "#C62828"]   # blue, orange, red


def _draw_log_minor_grid(ax, color="#e8e8e8", linewidth=0.3, alpha=0.8):
    """Draw manual minor gridlines for a log-scale y-axis (2–9 subdivisions per decade)."""
    import math
    ymin, ymax = ax.get_ylim()
    lo = math.floor(math.log10(max(ymin, 1e-12)))
    hi = math.ceil(math.log10(max(ymax, 1e-11)))
    for exp in range(lo, hi):
        for sub in range(2, 10):
            y = sub * 10.0 ** exp
            if ymin <= y <= ymax:
                ax.axhline(y, color=color, linewidth=linewidth, alpha=alpha,
                           zorder=0, linestyle="-")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – boxplot + individual dots
# ══════════════════════════════════════════════════════════════════════════════

def make_boxplot(rows: List[Dict], output_path: str, show_title: bool = True) -> str:
    """
    One subplot per edge density.  Boxplot of per-repetition elapsed times
    with individual run dots overlaid.  Y-axis is log-scaled.
    Timed-out runs are excluded.
    """
    plt, np, mticker = _mpl()

    densities = sorted(set(r["density"] for r in rows))
    n_dens    = len(densities)
    fig, axes = plt.subplots(1, n_dens, figsize=(_FIG_WIDTH_SCALE * (5 * n_dens + 1), 6), sharey=True)
    if n_dens == 1:
        axes = [axes]

    title = (
        "TrAC – wall-clock runtime on random Boolean SCMs"
    )
    if show_title:
        fig.suptitle(title, fontsize=_FONT_SCALE * 12)
    rng = np.random.default_rng(seed=0)

    for ax, density, color in zip(axes, densities, _COLORS):
        cell_data: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            if r["density"] == density:
                cell_data[r["n_nodes"]].append(r["elapsed_s"])

        ns = sorted(cell_data.keys())
        if not ns:
            ax.set_title(f"p = {density:.2f}  (no data)")
            continue

        positions  = list(range(1, len(ns) + 1))
        data_lists = [cell_data[n] for n in ns]

        ax.boxplot(
            data_lists, positions=positions, widths=0.45,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.22, linewidth=1.2),
            medianprops=dict(color=color, linewidth=2.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        for i, (pos, n) in enumerate(zip(positions, ns)):
            times  = np.array(cell_data[n])
            jitter = rng.uniform(-0.18, 0.18, len(times))
            ax.scatter(pos + jitter, times,
                       color=color, alpha=0.65, s=18, zorder=4, linewidths=0,
                       label=f"p = {density:.2f}" if i == 0 else "_nolegend_")

        ax.set_xticks(positions)
        ax.set_xticklabels(ns, fontsize=_FONT_SCALE * _LOG_XTICK_SCALE * 8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.tick_params(axis="y", labelsize=_FONT_SCALE * _LOG_YTICK_SCALE * 8)
        ax.set_title(f"Edge density  p = {density:.2f}", fontsize=_FONT_SCALE * 11)
        ax.set_xlabel("Number of nodes", fontsize=_FONT_SCALE * 10)
        ax.grid(True, axis="y", which="major", alpha=0.5, linewidth=0.8, color="gray")
        ax.grid(False, axis="x")

    for ax in axes:
        _draw_log_minor_grid(ax)

    axes[0].set_ylabel("Wall-clock time (s)  [log scale]", fontsize=_FONT_SCALE * 10)

    # Collect one legend entry per density from each subplot, then place outside.
    _h, _l = [], []
    seen_labels: set = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                _h.append(h)
                _l.append(l)
                seen_labels.add(l)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=fig.subplotpars.left + _FONT_SCALE * 10 / (72 * fig.get_figwidth()))
    if _h:
        fig.legend(_h, _l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_FONT_SCALE * 10, frameon=True, title="Edge density",
                   title_fontsize=_FONT_SCALE * 10, markerscale=3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Boxplot   → {output_path}")
    return title




# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – number of actual cause sets
# ══════════════════════════════════════════════════════════════════════════════

def make_n_causes_plot(rows: List[Dict], output_path: str, show_title: bool = True) -> str:
    """
    Distribution of n_causes per cell on a single axes.

    All densities are shown together, colour-coded, with node counts grouped
    on the x-axis.  Boxplot + individual dots per (n_nodes, density) group.
    If a large gap (>2) exists in the value distribution the y-axis is broken:
    a small top panel shows the outlier region; the main panel shows the bulk.
    """
    plt, np, mticker = _mpl()

    densities   = sorted(set(r["density"] for r in rows))
    node_counts = sorted(set(r["n_nodes"] for r in rows))
    n_dens      = len(densities)

    all_vals = [r["n_causes"] for r in rows]
    data_max = max(all_vals) if all_vals else 1

    # Detect a large gap between consecutive unique values → y-axis break.
    uniq = sorted(set(all_vals))
    break_lo = break_hi = None
    if len(uniq) > 1:
        gap_size, gap_lo, gap_hi = max(
            (uniq[i + 1] - uniq[i], uniq[i], uniq[i + 1])
            for i in range(len(uniq) - 1)
        )
        if gap_size > 2:
            break_lo, break_hi = gap_lo, gap_hi

    use_break = break_lo is not None
    fig_width = max(5, 2 * len(node_counts) - 2)

    title = (
        "TrAC – number of found actual cause sets\n"
        "on random Boolean SCMs"
    )

    if use_break:
        assert break_lo is not None and break_hi is not None
        bot_range = break_lo + 1
        top_ratio = max(bot_range * 0.25, data_max - break_hi + 1)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=(fig_width, 5),
            gridspec_kw={"height_ratios": [top_ratio, bot_range], "hspace": 0.06},
            sharex=True,
        )
        axes_all = [ax_top, ax_bot]
    else:
        fig, ax_bot = plt.subplots(figsize=(fig_width, 5))
        ax_top = None
        axes_all = [ax_bot]

    if show_title:
        fig.suptitle(title, fontsize=_FONT_SCALE * 12)

    rng = np.random.default_rng(seed=1)

    group_width = 0.5
    offsets   = np.linspace(-group_width / 2, group_width / 2, n_dens) if n_dens > 1 else [0.0]
    box_width = group_width / n_dens * 0.85

    for i, (density, color) in enumerate(zip(densities, _COLORS)):
        cell_data: Dict[int, List[int]] = defaultdict(list)
        for r in rows:
            if r["density"] == density:
                cell_data[r["n_nodes"]].append(r["n_causes"])

        ns = sorted(cell_data.keys())
        if not ns:
            continue

        positions  = [node_counts.index(n) + 1 + offsets[i] for n in ns]
        data_lists = [cell_data[n] for n in ns]

        bp_kwargs = dict(
            positions=positions, widths=box_width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.22, linewidth=1.2),
            medianprops=dict(color=color, linewidth=2.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        for ax in axes_all:
            ax.boxplot(data_lists, **bp_kwargs)

        # pre-compute jitter so dots are identical across both panels
        jitters = {
            n: rng.uniform(-box_width * 0.35, box_width * 0.35, len(cell_data[n]))
            for n in ns
        }
        for j, (pos, n) in enumerate(zip(positions, ns)):
            vals   = np.array(cell_data[n], dtype=float)
            jitter = jitters[n]
            for k, ax in enumerate(axes_all):
                ax.scatter(
                    pos + jitter, vals,
                    color=color, alpha=0.65, s=18, zorder=4, linewidths=0,
                    label=f"p = {density:.2f}" if (j == 0 and k == 0) else None,
                )

    ax_bot.set_xticks(range(1, len(node_counts) + 1))
    ax_bot.set_xticklabels(node_counts, fontsize=_FONT_SCALE * 10)
    ax_bot.set_xlabel("Number of nodes", fontsize=_FONT_SCALE * 12)

    if use_break:
        assert ax_top is not None and break_lo is not None and break_hi is not None
        ax_top.set_ylim(break_hi - 0.5, data_max + 0.5)
        ax_top.set_yticks(range(break_hi, data_max + 1))
        ax_top.tick_params(axis="y", labelsize=_FONT_SCALE * 10)
        ax_bot.set_ylim(-0.5, break_lo + 0.5)
        ax_bot.set_yticks(range(0, break_lo + 1))
        ax_bot.tick_params(axis="y", labelsize=_FONT_SCALE * 10)

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False)

        # small diagonal break marks using matplotlib marker approach
        _bkw = dict(
            marker=[(-1, -0.5), (1, 0.5)], markersize=7,
            linestyle="none", color="k", mec="k", mew=1.0, clip_on=False,
        )
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **_bkw)
        ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **_bkw)

        ax_bot.set_ylabel("# Actual Cause Sets", fontsize=_FONT_SCALE * 12, labelpad=4)
        for ax in axes_all:
            ax.grid(True, axis="y", alpha=0.25)
            ax.grid(False, axis="x")
    else:
        ax_bot.set_yticks(range(0, data_max + 1))
        ax_bot.tick_params(axis="y", labelsize=_FONT_SCALE * 10)
        ax_bot.set_ylabel("# Actual Cause Sets", fontsize=_FONT_SCALE * 12)
        ax_bot.grid(True, axis="y", alpha=0.25)
        ax_bot.grid(False, axis="x")

    _h, _l = [], []
    for ax in axes_all:
        h, l = ax.get_legend_handles_labels()
        if h and len(h) > len(_h):
            _h, _l = h, l
    plt.tight_layout(pad=1.5)
    if _h:
        fig.legend(_h, _l, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=_FONT_SCALE * 11, frameon=True, markerscale=3)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"n_causes  → {output_path}")
    return title


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Remark-6 pruning rates
# ══════════════════════════════════════════════════════════════════════════════

def make_pruning_plot(rows: List[Dict], output_path: str, show_title: bool = True) -> str:
    """
    2×2 grid of pruning-rate plots with mean + IQR [p25–p75] shading.

    Fractions are computed per run (not from pre-aggregated means) so the
    percentile bands reflect the actual distribution across repetitions.

      (top-left)    Superset pruning (iii) – skip_sup / n_cands per run
      (top-right)   U-projection (ii)      – skip_up  / n_cands per run
      (bottom-left) Combined node skip     – (skip_sup + skip_up) / n_cands
      (bottom-right) Partition skip        – part_skip / (part_skip + part_test)

    Runs where the denominator is zero are excluded.
    Requires runs produced with the instrumented trac() (n_cands column present).
    """
    plt, np, mticker = _mpl()

    has_data = any(r.get("n_cands", 0) > 0 for r in rows)
    if not has_data:
        print("Pruning plot → skipped (no pruning data; re-run benchmark to collect)")
        return ""

    densities   = sorted(set(r["density"] for r in rows))
    node_counts = sorted(set(r["n_nodes"] for r in rows))

    def _sup_rate(r):
        n = r.get("n_cands", 0)
        return r["n_skip_sup"] / n if n > 0 else None

    def _up_rate(r):
        n = r.get("n_cands", 0)
        return r["n_skip_up"] / n if n > 0 else None

    def _node_skip(r):
        n = r.get("n_cands", 0)
        return (r["n_skip_sup"] + r["n_skip_up"]) / n if n > 0 else None

    def _part_skip(r):
        denom = r.get("n_part_skip", 0) + r.get("n_part_test", 0)
        return r["n_part_skip"] / denom if denom > 0 else None

    panel_specs = [
        ("Node-level pre-check\n(Remark 6(ii))",                    _up_rate),
        ("Superset pruning\n(Remark 6(iii))",                       _sup_rate),
        ("Combined node pruning\n(Remark 6(ii)+(iii))",             _node_skip),
        ("Partition pruning, visited nodes\n(Remark 6(iii))",       _part_skip),
    ]

    pruning_title = "TrAC – pruning rates (mean + IQR)"
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 8), sharey=True)
    if show_title:
        fig.suptitle(pruning_title, fontsize=_FONT_SCALE * 12)

    left_col = {axes[0, 0], axes[1, 0]}
    for ax, (title, frac_fn) in zip(axes.flat, panel_specs):
        for color, density in zip(_COLORS, densities):
            cell_data: Dict[int, List[float]] = defaultdict(list)
            for r in rows:
                if r["density"] == density:
                    v = frac_fn(r)
                    if v is not None:
                        cell_data[r["n_nodes"]].append(v)

            ns = sorted(cell_data.keys())
            if not ns:
                continue

            xs    = np.array(ns)
            means = np.array([np.mean(cell_data[n])           for n in ns])
            lo    = np.array([np.percentile(cell_data[n], 25) for n in ns])
            hi    = np.array([np.percentile(cell_data[n], 75) for n in ns])

            ax.plot(xs, means, color=color, marker="o",
                    linewidth=1.8, markersize=4, label=f"p = {density:.2f}")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18)

        ax.set_xlabel("Number of nodes", fontsize=_FONT_SCALE * 10)
        if ax in left_col:
            ax.set_ylabel("Fraction pruned", fontsize=_FONT_SCALE * 10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=_FONT_SCALE * 10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(node_counts)
        ax.tick_params(axis="x", labelsize=_FONT_SCALE * _LOG_XTICK_SCALE * 8)
        ax.tick_params(axis="y", labelsize=_FONT_SCALE * _PRUNING_YTICK_SCALE * 8)

    # Single shared legend outside the grid
    _h, _l = axes.flat[0].get_legend_handles_labels()
    plt.tight_layout(pad=1.5)
    if _h:
        fig.legend(_h, _l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_FONT_SCALE * 8, frameon=True,
                   title="Edge density", title_fontsize=_FONT_SCALE * 8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Pruning   → {output_path}")
    return pruning_title


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – algorithm comparison (TrAC vs actualcauses beam / ISI)
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_plot(rows: List[Dict], output_path: str, show_title: bool = True) -> str:
    """
    Mean runtime comparison with IQR [p25–p75] shading per algorithm.

    One subplot per edge density.  Algorithms with no valid timing data are
    silently skipped.  Y-axis is log-scaled.
    """
    plt, np, mticker = _mpl()

    # (label, row-key, valid-predicate, color, marker, linewidth)
    algo_specs = [
        ("Reyd et al. (beam)",     "t_beam_s",  lambda v: v >  0, "#1976D2", "s", 1.6),
        ("Reyd et al. (ISI)",      "t_isi_s",   lambda v: v >  0, "#F57C00", "^", 1.6),
        ("Ibrahim et al. (BF)",    "t_bf_s",    lambda v: v >  0, "#7B1FA2", "D", 1.4),
        ("Ibrahim et al. (SAT)",   "t_sat_s",   lambda v: v >  0, "#388E3C", "v", 1.4),
        ("Ibrahim et al. (MaxSAT)","t_ms_s",    lambda v: v >  0, "#D32F2F", "P", 1.4),
        #("Ibrahim et al. (ILP)",   "t_ilp_s",   lambda v: v >  0, "#0097A7", "X", 1.4), # requires commerical solver...
        ("TrAC (ours)",            "elapsed_s", lambda v: v >= 0, "black",   "o", 2.0),
    ]

    densities   = sorted(set(r["density"] for r in rows))
    n_dens      = len(densities)
    node_counts = sorted(set(r["n_nodes"] for r in rows))

    fig, axes = plt.subplots(1, n_dens, figsize=(_FIG_WIDTH_SCALE * (5 * n_dens + 1), 6), sharey=True)
    if n_dens == 1:
        axes = [axes]

    compare_title = (
        "Runtime comparison – TrAC vs Reyd et al. (beam / ISI) vs Ibrahim et al. HP2SAT (BF / SAT / MaxSAT)\n"
        "(mean + IQR band [p25–p75])"
    )
    if show_title:
        fig.suptitle(compare_title, fontsize=_FONT_SCALE * 10)

    for ax, density in zip(axes, densities):
        for label, key, valid, color, marker, lw in algo_specs:
            cell_data: Dict[int, List[float]] = defaultdict(list)
            for r in rows:
                if r["density"] == density:
                    v = r.get(key)
                    if v is not None and valid(v):
                        cell_data[r["n_nodes"]].append(v)

            ns = sorted(cell_data.keys())
            if not ns:
                continue

            xs    = np.array(ns)
            means = np.array([np.mean(cell_data[n])           for n in ns])
            lo    = np.array([np.percentile(cell_data[n], 25) for n in ns])
            hi    = np.array([np.percentile(cell_data[n], 75) for n in ns])

            ax.plot(xs, means, label=label, color=color, marker=marker,
                    linewidth=lw, markersize=5)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

        ax.set_title(f"Edge density  p = {density:.2f}", fontsize=_FONT_SCALE * 11)
        ax.set_xlabel("Number of nodes", fontsize=_FONT_SCALE * 10)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.tick_params(axis="y", labelsize=_FONT_SCALE * _LOG_YTICK_SCALE * 8)
        ax.grid(True, axis="y", which="major", alpha=0.5, linewidth=0.8, color="gray")
        ax.grid(False, axis="x")
        ax.set_xticks(node_counts)
        ax.tick_params(axis="x", labelsize=_FONT_SCALE * _LOG_XTICK_SCALE * 8)

    for ax in axes:
        _draw_log_minor_grid(ax)

    axes[0].set_ylabel("Wall-clock time (s)  [log scale]", fontsize=_FONT_SCALE * _COMPARE_YLABEL_SCALE * 10)

    # Single legend outside to the right, collected from first subplot with data.
    _h, _l = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h and len(h) > len(_h):
            _h, _l = h, l
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=fig.subplotpars.left + _FONT_SCALE * 10 / (72 * fig.get_figwidth()))
    if _h:
        fig.legend(_h, _l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_FONT_SCALE * 9, frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Compare   → {output_path}")
    return compare_title


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – worst-case (max) runtime comparison
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_max_plot(rows: List[Dict], output_path: str, show_title: bool = True) -> str:
    """
    Maximum runtime per (n_nodes, density) for each algorithm.

    Complements the mean+IQR plot by showing worst-case behaviour.
    One subplot per edge density, log-scaled y-axis.
    Algorithms with no valid data in a cell are skipped.
    """
    plt, np, mticker = _mpl()

    algo_specs = [
        ("Reyd et al. (beam)",     "t_beam_s",  lambda v: v >  0, "#1976D2", "s", 1.6),
        ("Reyd et al. (ISI)",      "t_isi_s",   lambda v: v >  0, "#F57C00", "^", 1.6),
        ("Ibrahim et al. (BF)",    "t_bf_s",    lambda v: v >  0, "#7B1FA2", "D", 1.4),
        ("Ibrahim et al. (SAT)",   "t_sat_s",   lambda v: v >  0, "#388E3C", "v", 1.4),
        ("Ibrahim et al. (MaxSAT)","t_ms_s",    lambda v: v >  0, "#D32F2F", "P", 1.4),
        #("Ibrahim et al. (ILP)",   "t_ilp_s",   lambda v: v >  0, "#0097A7", "X", 1.4), # requires commercial solver
        ("TrAC (ours)",            "elapsed_s", lambda v: v >= 0, "black",   "o", 2.0),
    ]

    densities   = sorted(set(r["density"] for r in rows))
    n_dens      = len(densities)
    node_counts = sorted(set(r["n_nodes"] for r in rows))

    fig, axes = plt.subplots(1, n_dens, figsize=(_FIG_WIDTH_SCALE * (5 * n_dens + 1), 6), sharey=True)
    if n_dens == 1:
        axes = [axes]

    compare_max_title = (
        "Worst-case runtime – TrAC vs Reyd et al. (beam / ISI) vs Ibrahim et al. HP2SAT (BF / SAT / MaxSAT)"
    )
    if show_title:
        fig.suptitle(compare_max_title, fontsize=_FONT_SCALE * 10)

    for ax, density in zip(axes, densities):
        any_line = False
        for label, key, valid, color, marker, lw in algo_specs:
            cell_data: Dict[int, List[float]] = defaultdict(list)
            for r in rows:
                if r["density"] == density:
                    v = r.get(key)
                    if v is not None and valid(v):
                        cell_data[r["n_nodes"]].append(v)

            ns = sorted(cell_data.keys())
            if not ns:
                continue

            xs   = np.array(ns)
            maxs = np.array([max(cell_data[n]) for n in ns])

            ax.plot(xs, maxs, label=label, color=color, marker=marker,
                    linewidth=lw, markersize=5)
            any_line = True

        ax.set_title(f"Edge density  p = {density:.2f}", fontsize=_FONT_SCALE * 11)
        ax.set_xlabel("Number of nodes", fontsize=_FONT_SCALE * 10)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.tick_params(axis="y", labelsize=_FONT_SCALE * _LOG_YTICK_SCALE * 8)
        ax.grid(True, axis="y", which="major", alpha=0.5, linewidth=0.8, color="gray")
        ax.grid(False, axis="x")
        ax.set_xticks(node_counts)
        ax.tick_params(axis="x", labelsize=_FONT_SCALE * _LOG_XTICK_SCALE * 8)

    for ax in axes:
        _draw_log_minor_grid(ax)

    axes[0].set_ylabel("Max wall-clock time (s)  [log scale]", fontsize=_FONT_SCALE * _COMPARE_YLABEL_SCALE * 10)

    # Single legend outside to the right, collected from first subplot with data.
    _h, _l = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h and len(h) > len(_h):
            _h, _l = h, l
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=fig.subplotpars.left + _FONT_SCALE * 10 / (72 * fig.get_figwidth()))
    if _h:
        fig.legend(_h, _l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_FONT_SCALE * 9, frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Compare max → {output_path}")
    return compare_max_title


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TrAC benchmark results from per-cell CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default=_DEFAULT_RESULTS_DIR,
        help="Directory containing n*_d*.csv files",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Main figure path (default: <results-dir>/trac.{_FIG_FMT})",
    )
    parser.add_argument(
        "--include-titles", action="store_true",
        help="Add suptitles to all figures",
    )
    parser.add_argument(
        "--save-titles", action="store_true",
        help="Write figure titles to <stem>_titles.txt",
    )
    args = parser.parse_args()

    # Derive default output dir from parent of results_dir so that when
    # results_dir is results/data/, plots land in results/ (not results/data/).
    _cells_abs  = os.path.abspath(args.results_dir)
    _plots_dir  = os.path.dirname(_cells_abs)
    output = args.output or os.path.join(_plots_dir, f"trac.{_FIG_FMT}")
    stem   = _plot_stem(output)

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No completed data found in: {args.results_dir}")
        sys.exit(1)
    print(f"Loaded {len(rows)} completed data points from {args.results_dir}")

    stats = compute_stats(args.results_dir)
    show_title = args.include_titles

    titles = [
        make_boxplot(rows, output, show_title=show_title),
        make_n_causes_plot(rows, f"{stem}_n_causes.{_FIG_FMT}", show_title=show_title),
        make_pruning_plot(rows, f"{stem}_pruning.{_FIG_FMT}", show_title=show_title),
        make_comparison_plot(rows, f"{stem}_compare.{_FIG_FMT}", show_title=show_title),
        make_comparison_max_plot(rows, f"{stem}_compare_max.{_FIG_FMT}", show_title=show_title),
    ]
    _save_stats_csv(stats, stem + "_stats.csv")
    if args.save_titles:
        _save_titles_txt([t for t in titles if t], stem + "_titles.txt")


if __name__ == "__main__":
    main()
