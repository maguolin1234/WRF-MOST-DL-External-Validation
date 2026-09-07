#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final comparison, statistics and plotting for Xilinhaote generalization results
=============================================================================

This script does NOT retrain or apply any model. It reads the outputs from:

    code3_xilinhaote_apply_code1_generalization.py
    code4_xilinhaote_apply_code2_generalization_same_time_only_fixed_v4_exact_model_root.py

and creates a compact, auditable comparison package for later scientific
interpretation.

Main goals
----------
1. Clean and audit Code1/Code2 generalization metric tables.
2. Avoid repeated Code1 WD/TKE records caused by multiple WS-height tasks.
3. Summarize mean, spread, positive-skill rate and worst-case behavior.
4. Rank cases within each exact generalization task.
5. Quantify the added effect of:
       MOST input,
       MOST loss,
       MOST input + loss,
   relative to the data-driven C2 baseline.
6. Compare Code1 and Code2 at the common 10 m cross-tower task.
7. Produce two clearly separated output layers:
       A) a comprehensive result library containing all statistics and plots;
       B) three compact multi-panel figures designed for the SCI manuscript.
8. Export the exact data used by every SCI figure for reproducibility.
9. Recompute plotting/statistical comparisons on a strict common-time intersection
   so Code1/Code2, networks and cases share the same OBS/Raw-WRF baseline whenever
   they are directly compared.

Important statistical conventions
---------------------------------
- Raw WRF is represented by Raw_RMSE stored in each corrected-case metric row.
- WD uses circular-error metrics already calculated by the generalization code.
- Arithmetic means across runs are descriptive only. The script also reports
  median, standard deviation, minimum, maximum, positive-skill fraction and
  the number of valid runs.
- Code1 WD/TKE rows are deduplicated before aggregation because the same native
  WD/TKE application may be repeated for several WS-height tasks.
- No user-defined composite score is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

CODE1_GENERALIZATION_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\xilinhaote\result\bias_correction\1km\correction\c1_generalization_xilinhaote_same_and_all_target_time"
)

CODE2_GENERALIZATION_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\xilinhaote\result\bias_correction\1km\correction\c2_generalization_xilinhaote_same_and_all_target_time"
)

OUTPUT_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\xilinhaote\result\bias_correction\1km\correction\xlh_v8_fair_obs_corrected"
)

# OBS / cross-code baseline policy --------------------------------------------
# The checked OBS reader supplied by the user reads raw tower files directly,
# removes logger missing flags / nonphysical wind speeds, and then plots the
# surviving observations.  This postprocessor still evaluates saved model
# outputs, but every direct Code1-vs-Code2 comparison must use ONE OBS/WRF
# lineage rather than averaging/splicing two preprocessing pipelines.
# Code2 is preferred because it is derived from the original 10-min
# four_sites_full target data and is aggregated to hourly here; Code1 uses the
# separately prepared D1H_1hour dataset.  Raw tower OBS is an independent audit
# reference only and never silently replaces model-evaluation data.
CROSS_CODE_CANONICAL_BASELINE = "Code2"
RAW_OBS_DIR_CANDIDATES = [
    Path(r"I:\wake\WRF_RANS_wake\xilinhaote\result\obs\obs"),
    Path(r"I:\wake\WRF_RANS_wake\xilinhaote\result\obs"),
    Path(r"D:\wake\WRF_RANS_wake\xilinhaote\result\obs\obs"),
    Path(r"D:\wake\WRF_RANS_wake\xilinhaote\result\obs"),
]
RAW_OBS_SUFFIXES = (".csv", ".txt", ".dat", ".xls", ".xlsx")
RAW_OBS_MISSING_FLAGS = [999, 9999, 99999, -999, -9999, -99999, 32766, 32767]
RAW_OBS_INVALID_ABS_THRESHOLD = 1.0e10
RAW_OBS_WS_RANGE = (0.0, 60.0)

TIME_COL = "北京时间"

CODE1_CASES = ["C2", "C4", "C6", "C8"]
CODE2_CASES = ["C2", "C4b", "C6m", "C8d"]

CODE1_ROLE = {
    "C2": "Data-driven",
    "C4": "MOST input",
    "C6": "MOST loss",
    "C8": "MOST input + loss",
}
CODE2_ROLE = {
    "C2": "Data-driven",
    "C4b": "MOST input",
    "C6m": "MOST loss",
    "C8d": "MOST input + loss",
}
ROLE_ORDER = ["Data-driven", "MOST input", "MOST loss", "MOST input + loss"]

CODE1_ABLATIONS = {
    "C4_minus_C2": ("C4", "C2"),
    "C6_minus_C2": ("C6", "C2"),
    "C8_minus_C2": ("C8", "C2"),
}
CODE2_ABLATIONS = {
    "C4b_minus_C2": ("C4b", "C2"),
    "C6m_minus_C2": ("C6m", "C2"),
    "C8d_minus_C2": ("C8d", "C2"),
}

GENERALIZATION_LABELS = {
    "same_tower_other_height": "Same tower, other height",
    "cross_tower_same_height": "Cross tower, same height",
    "cross_tower_other_height": "Cross tower, other height",
    "cross_tower_same_time": "Cross tower, same time",
    "cross_tower_other_time": "Cross tower, all target times",
}

VARIABLE_ORDER = ["WS", "WD", "TKE", "T"]
PRIMARY_METRIC = {"WS": "RMSE", "WD": "MAE", "TKE": "RMSE", "T": "RMSE"}
VARIABLE_LABEL = {
    "WS": "WS RMSE (m s$^{-1}$)",
    "WD": "WD MAE (°)",
    "TKE": "TKE RMSE (m$^2$ s$^{-2}$)",
    "T": "Temperature RMSE (K)",
}

REFERENCE_WS_HEIGHT_M = 10

PNG_DPI = 600
SAVE_PDF = True
SAVE_SVG = True
MAX_TIME_POINTS = 2500

# Curves from different ablation cases may be exactly identical, especially when
# the selected physics-loss weight is zero. Plotting them independently makes
# later curves completely cover earlier curves. The comparison script therefore
# detects exact numerical overlap and draws one combined curve with a label such
# as "C2 = C6". Statistical tables still retain every case separately.
CURVE_EQUAL_ATOL = 1.0e-10
CURVE_EQUAL_RTOL = 1.0e-10
CURVE_OVERLAP_RECORDS: List[Dict[str, object]] = []

CASE_STYLE = {
    "C2":  {"linestyle": "-",  "marker": "o"},
    "C4":  {"linestyle": "--", "marker": "s"},
    "C6":  {"linestyle": "-.", "marker": "^"},
    "C8":  {"linestyle": ":",  "marker": "D"},
    "C4b": {"linestyle": "--", "marker": "s"},
    "C6m": {"linestyle": "-.", "marker": "^"},
    "C8d": {"linestyle": ":",  "marker": "D"},
}


# =============================================================================
# SCI MAIN-FIGURE CONFIGURATION
# =============================================================================
# The full-results layer always retains both CNN_LSTM and TCN. The SCI profile
# and time-series panels use one fixed representative architecture so that the
# manuscript does not become overloaded. TCN is the default because it showed
# the strongest overall MOST-informed WS generalization in the current results.
# Change this to "CNN_LSTM" to regenerate the main figures with that architecture.
SCI_NETWORKS = ["CNN_LSTM", "TCN"]

# For Xilinhaote, all 12 directed transfers enter the statistics and rankings.
# The SCI 3×3 figures use NO1↔NO2 as a fixed representative pair.
# Fixed direction used for the representative time-series panel. This is selected
# a priori rather than chosen from the best-looking realization.
SCI_TS_SOURCE_SITE = "NO1_1159"
SCI_TS_TARGET_SITE = "NO2_1107"
SCI_TS_HEIGHT_M = 10

# Best overall WS cases used only in the direct Code1-vs-Code2 figure.
SCI_CODE1_BEST_WS_CASE = "C8"
SCI_CODE2_BEST_WS_CASE = "C8d"

# Main-paper figure layout.
# Three rows: metrics, CNN-LSTM panels, and TCN panels.
SCI_FIGSIZE = (17.8, 15.2)
SCI_SINGLE_PANEL_FIGSIZE = (7.0, 5.3)
SCI_PROFILE_DIRECTIONS = [
    ("NO1_1159", "NO2_1107"),
    ("NO2_1107", "NO1_1159"),
]

# Set explicit dates to crop the manuscript time-series panels. None keeps the
# complete common test period.
SCI_TIME_START = None
SCI_TIME_END = None

# Generate a parallel set of SCI candidate figures for the new all-target-time
# validation. The original G1-G3 figures remain unchanged and represent the
# source-test-time comparison; G4-G6 represent all valid target times.
GENERATE_ALL_TARGET_TIME_SCI_CANDIDATES = True

# Distribution diagnostics for G4-G6. The underlying all-target-time prediction
# tables come from the WRF-period ML-ready files; only their valid matched rows
# are used. No observation outside the WRF/postprocess period is loaded.
GENERATE_WRF_PERIOD_DISTRIBUTION_DIAGNOSTICS = True
DISTRIBUTION_WS_HEIGHT_M = 10
WINDROSE_WD_HEIGHT_BY_TARGET = {
    "NO1_1159": 10,
    "NO2_1107": 10,
    "NO3_1071": 10,
    "NOT_1166": 10,
}
WEIBULL_MIN_SAMPLES = 20
WEIBULL_BIN_WIDTH = 0.5
WINDROSE_NUM_SECTORS = 16
WINDROSE_SPEED_BINS = (
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 75.0
)

# Automatic best-model selection and compact best-result figures.
# One global network-case pair is selected for each Code/time scope using the
# mean WS RMSE across all available directed cross-tower transfers. Tie-breaking order:
# lower worst-direction RMSE, higher mean RMSE skill, higher mean Pearson r.
GENERATE_BEST_MODEL_SUMMARIES = True
BEST_BASELINE_CASE_CODE1 = "C2"
BEST_BASELINE_CASE_CODE2 = "C2"
BEST_SELECTION_SCOPES = ("test_period", "all_wrf_period")
BEST_TIME_SCOPE_LABELS = {
    "test_period": "Source-test/common-time period",
    "all_wrf_period": "All valid target times within the WRF period",
}

# Main-figure line widths.
SCI_LINEWIDTH_OBS = 1.7
SCI_LINEWIDTH_WRF = 1.25
SCI_LINEWIDTH_CASE = 1.15

# Hatch patterns improve readability in grayscale printing.
CASE_HATCH = {
    "Raw WRF": "",
    "C2": "",
    "C4": "//",
    "C6": "..",
    "C8": "xx",
    "C4b": "//",
    "C6m": "..",
    "C8d": "xx",
    "Code1": "",
    "Code2": "//",
}

# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

FONT_SMALL = 8.5
FONT_NORMAL = 10.0
FONT_LARGE = 11.5
FONT_TITLE = 13.0
GRID_ALPHA = 0.22

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_NORMAL,
    "axes.titlesize": FONT_LARGE,
    "axes.labelsize": FONT_NORMAL,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "figure.titlesize": FONT_TITLE,
    "axes.linewidth": 0.8,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

# =============================================================================
# 3. GENERIC HELPERS
# =============================================================================

def _windows_long_path(path: Path) -> str:
    """Return a Windows extended-length path when the normal path is long."""
    value = str(Path(path).resolve())
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value.lstrip("\\")
        return "\\\\?\\" + value
    return value


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, OSError):
        if os.name != "nt":
            raise
        os.makedirs(_windows_long_path(path), exist_ok=True)
    return path


def safe_name(value: object, max_len: int = 100) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(value))[:max_len]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, base: Path) -> None:
    """Save figures safely on Windows, including long nested output paths."""
    base = Path(base)
    ensure_dir(base.parent)

    outputs = [
        (base.with_suffix(".png"), {"dpi": PNG_DPI, "bbox_inches": "tight"}),
    ]
    if SAVE_PDF:
        outputs.append(
            (base.with_suffix(".pdf"), {"bbox_inches": "tight"})
        )
    if SAVE_SVG:
        outputs.append(
            (base.with_suffix(".svg"), {"bbox_inches": "tight"})
        )

    try:
        for path, kwargs in outputs:
            fig.savefig(path, **kwargs)
    except (FileNotFoundError, OSError):
        if os.name != "nt":
            plt.close(fig)
            raise
        for path, kwargs in outputs:
            fig.savefig(_windows_long_path(path), **kwargs)
    finally:
        plt.close(fig)


def add_panel_labels(axes: Iterable[plt.Axes]) -> None:
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(list(axes)):
        ax.text(
            0.012, 0.985, f"({letters[i]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_LARGE,
            fontweight="bold",
        )


def collect_legend(axes: Iterable[plt.Axes]) -> Tuple[List[object], List[str]]:
    handles: List[object] = []
    labels: List[str] = []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in seen:
                handles.append(hh)
                labels.append(ll)
                seen.add(ll)
    return handles, labels


def label_bars(ax: plt.Axes, bars, decimals: int = 2) -> None:
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        ax.annotate(
            f"{value:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SMALL,
        )


def configure_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=GRID_ALPHA)



# =============================================================================
# GAP-SAFE TIME-SERIES PLOTTING
# =============================================================================
# GLOBAL-COMMON timestamps can contain genuine gaps after complete-case filtering.
# Matplotlib would otherwise connect the last valid point before a gap directly to
# the first valid point after it.  The helper below breaks the curve at unusually
# large time gaps while reserving only ONE matplotlib color per physical series.
# Therefore the original OBS / Raw WRF / case color order is preserved.

def _plot_gap_safe(
    ax: plt.Axes,
    times: pd.Series,
    values: pd.Series,
    *,
    label: str,
    linewidth: float = 1.0,
    linestyle: str = "-",
    marker=None,
    color=None,
    gap_multiplier: float = 1.5,
    **kwargs,
):
    d = pd.DataFrame({"_time": pd.to_datetime(times, errors="coerce"), "_value": numeric(values)})
    d = d.dropna(subset=["_time", "_value"]).sort_values("_time")
    if d.empty:
        return []

    # Reserve exactly one color for this series, then reuse it for every segment.
    if color is None:
        color = ax._get_lines.get_next_color()

    dt = d["_time"].diff()
    positive = dt[dt > pd.Timedelta(0)]
    if positive.empty:
        segment_id = pd.Series(0, index=d.index)
    else:
        nominal = positive.median()
        threshold = nominal * float(gap_multiplier)
        segment_id = dt.gt(threshold).cumsum()

    artists = []
    first = True
    for _, seg in d.groupby(segment_id, sort=False):
        artists.extend(ax.plot(
            seg["_time"], seg["_value"],
            label=label if first else None,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            color=color,
            **kwargs,
        ))
        first = False
    return artists


def find_required_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    hits = list(root.rglob(filename)) if root.exists() else []
    if not hits:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")
    hits.sort(key=lambda p: (len(p.parts), len(str(p))))
    return hits[0]


# =============================================================================
# 4. LOAD AND CLEAN METRICS
# =============================================================================

REQUIRED_METRIC_COLS = {
    "source_site", "target_site", "network", "case",
    "generalization_type", "task_name", "variable", "height_m",
    "N", "RMSE", "MAE", "MBE", "Pearson_r", "Raw_RMSE",
    "Skill_RMSE_pct",
}


def load_metrics(root: Path, code: str) -> pd.DataFrame:
    filename = (
        "ALL_code1_generalization_metrics.csv"
        if code == "Code1"
        else "ALL_code2_generalization_metrics.csv"
    )
    path = find_required_file(root, filename)
    df = pd.read_csv(path)
    missing = REQUIRED_METRIC_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{code} metrics missing columns: {sorted(missing)}")

    out = df.copy()
    out["code"] = code
    out["source_site"] = out["source_site"].astype(str)
    out["target_site"] = out["target_site"].astype(str)
    out["network"] = out["network"].astype(str)
    out["case"] = out["case"].astype(str)
    out["variable"] = out["variable"].astype(str)
    out["generalization_type"] = out["generalization_type"].astype(str)
    out["height_num"] = pd.to_numeric(out["height_m"], errors="coerce")
    for c in ["N", "RMSE", "MAE", "MBE", "Pearson_r", "Raw_RMSE", "Skill_RMSE_pct"]:
        out[c] = numeric(out[c])

    role_map = CODE1_ROLE if code == "Code1" else CODE2_ROLE
    out["role"] = out["case"].map(role_map)
    out["direction"] = out["source_site"] + "→" + out["target_site"]
    out["primary_metric_name"] = out["variable"].map(PRIMARY_METRIC)
    out["primary_error"] = np.where(
        out["variable"].eq("WD"),
        out["MAE"],
        out["RMSE"],
    )
    return out


def deduplicate_code1_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove repeated native-height WD/TKE/T rows across different WS tasks.

    Code1 creates one task per application WS height. For a given source/target,
    generalization type, network and case, native WD/TKE/T may be repeated for
    several WS-height tasks. Those duplicates must not receive extra weight.
    """
    key = [
        "source_site", "target_site", "network", "case",
        "generalization_type", "variable", "height_m",
    ]
    d = df.copy()
    d["_dup_count"] = d.groupby(key, dropna=False)["task_name"].transform("count")
    dup_report = d[d["_dup_count"] > 1].copy()

    # Preserve every WS row because distinct task names normally correspond to
    # distinct application heights. For non-WS variables, retain one row per
    # unique physical configuration.
    ws = d[d["variable"].eq("WS")].copy()
    non_ws = d[~d["variable"].eq("WS")].copy()
    non_ws = non_ws.sort_values("task_name").drop_duplicates(key, keep="first")
    cleaned = pd.concat([ws, non_ws], ignore_index=True)
    cleaned = cleaned.drop(columns=["_dup_count"], errors="ignore")
    return cleaned, dup_report


def validate_case_coverage(df: pd.DataFrame, cases: Sequence[str], code: str) -> pd.DataFrame:
    rows = []
    grouping = ["source_site", "target_site", "network", "generalization_type"]
    for keys, g in df.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for case in cases:
            rows.append({
                "code": code,
                **meta,
                "case": case,
                "present": bool(g["case"].eq(case).any()),
                "n_metric_rows": int(g["case"].eq(case).sum()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

def summarize_group(g: pd.DataFrame) -> Dict[str, float]:
    x = numeric(g["primary_error"]).dropna()
    skill = numeric(g["Skill_RMSE_pct"]).dropna()
    n = numeric(g["N"]).dropna()
    return {
        "n_runs": int(len(x)),
        "N_sum": int(n.sum()) if not n.empty else 0,
        "error_mean": float(x.mean()) if not x.empty else np.nan,
        "error_median": float(x.median()) if not x.empty else np.nan,
        "error_std": float(x.std(ddof=1)) if len(x) >= 2 else 0.0 if len(x) == 1 else np.nan,
        "error_min": float(x.min()) if not x.empty else np.nan,
        "error_max": float(x.max()) if not x.empty else np.nan,
        "skill_mean_pct": float(skill.mean()) if not skill.empty else np.nan,
        "skill_median_pct": float(skill.median()) if not skill.empty else np.nan,
        "skill_min_pct": float(skill.min()) if not skill.empty else np.nan,
        "skill_max_pct": float(skill.max()) if not skill.empty else np.nan,
        "positive_skill_fraction": float((skill > 0).mean()) if not skill.empty else np.nan,
        "negative_skill_fraction": float((skill < 0).mean()) if not skill.empty else np.nan,
        "raw_rmse_mean": float(numeric(g["Raw_RMSE"]).mean()),
        "mbe_mean": float(numeric(g["MBE"]).mean()),
        "pearson_r_mean": float(numeric(g["Pearson_r"]).mean()),
    }


def grouped_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_tuple))
        row.update(summarize_group(g))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Rank cases within exact task/network/variable/height comparisons."""
    keys = [
        "code", "source_site", "target_site", "network",
        "generalization_type", "task_name", "variable", "height_m",
    ]
    rows = []
    for group_key, g in df.groupby(keys, dropna=False):
        gg = g[np.isfinite(g["primary_error"])].copy()
        if gg.empty:
            continue
        gg = gg.sort_values(["primary_error", "case"])
        gg["rank"] = np.arange(1, len(gg) + 1)
        gg["best_case"] = gg.iloc[0]["case"]
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_ablation_effects(df: pd.DataFrame, ablations: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Calculate paired differences for cases evaluated on the same task.

    Negative delta_error means the physics case improves on C2.
    Positive delta_skill means the physics case adds RMSE skill.
    """
    keys = [
        "code", "source_site", "target_site", "network",
        "generalization_type", "task_name", "variable", "height_m",
    ]
    rows = []
    indexed = df.set_index(keys + ["case"], drop=False)
    for effect_name, (physics_case, baseline_case) in ablations.items():
        physics = df[df["case"].eq(physics_case)].copy()
        for _, p in physics.iterrows():
            lookup = tuple(p[k] for k in keys) + (baseline_case,)
            try:
                b = indexed.loc[lookup]
            except KeyError:
                continue
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            delta_error = p["primary_error"] - b["primary_error"]
            delta_skill = p["Skill_RMSE_pct"] - b["Skill_RMSE_pct"]
            rows.append({
                **{k: p[k] for k in keys},
                "effect": effect_name,
                "physics_case": physics_case,
                "baseline_case": baseline_case,
                "baseline_error": b["primary_error"],
                "physics_error": p["primary_error"],
                "delta_error_physics_minus_baseline": delta_error,
                "relative_error_change_pct": (
                    delta_error / b["primary_error"] * 100.0
                    if np.isfinite(b["primary_error"]) and abs(b["primary_error"]) > 1e-12
                    else np.nan
                ),
                "baseline_skill_pct": b["Skill_RMSE_pct"],
                "physics_skill_pct": p["Skill_RMSE_pct"],
                "delta_skill_percentage_points": delta_skill,
                "physics_improved_error": bool(np.isfinite(delta_error) and delta_error < 0),
            })
    return pd.DataFrame(rows)


def summarize_ablation_effects(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["code", "generalization_type", "variable", "effect"]
    for keys, g in effects.groupby(group_cols, dropna=False):
        vals = numeric(g["delta_error_physics_minus_baseline"]).dropna()
        rel = numeric(g["relative_error_change_pct"]).dropna()
        skill = numeric(g["delta_skill_percentage_points"]).dropna()
        rows.append({
            **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
            "n_pairs": int(len(vals)),
            "mean_delta_error": float(vals.mean()) if not vals.empty else np.nan,
            "median_delta_error": float(vals.median()) if not vals.empty else np.nan,
            "mean_relative_error_change_pct": float(rel.mean()) if not rel.empty else np.nan,
            "median_relative_error_change_pct": float(rel.median()) if not rel.empty else np.nan,
            "mean_delta_skill_percentage_points": float(skill.mean()) if not skill.empty else np.nan,
            "improvement_fraction": float((vals < 0).mean()) if not vals.empty else np.nan,
            "deterioration_fraction": float((vals > 0).mean()) if not vals.empty else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. LOAD PREDICTIONS
# =============================================================================

def load_predictions(root: Path, code: str) -> pd.DataFrame:
    filename = (
        "ALL_code1_generalization_predictions.csv"
        if code == "Code1"
        else "ALL_code2_generalization_predictions.csv"
    )
    try:
        path = find_required_file(root, filename)
    except FileNotFoundError:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["code"] = code
    return df



# =============================================================================
# FAIR-TIME EVALUATION LAYER
# =============================================================================
# All statistics/plots below are recomputed from prediction tables after temporal
# alignment.  The policy is deliberately strict for WS cross-code comparisons:
# for each source->target direction and physical height, Code1 and Code2, both
# networks, and all compared cases use the SAME valid timestamps.  For variables
# or tasks that are not compared across codes, all cases/networks within the exact
# task still use one common timestamp intersection.
FAIR_TIME_ALIGNMENT = True
FAIR_BASELINE_ATOL = 1.0e-8
FAIR_BASELINE_RTOL = 1.0e-8
FAIR_STRICT_BASELINE_MATCH = True
FAIR_TIME_INDEX: Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex] = {}



# -----------------------------------------------------------------------------
# XILINHAOTE FAIR COMMON-TIME POLICY
# -----------------------------------------------------------------------------
# IMPORTANT:
# Xilinhaote contains four source towers and therefore twelve directed transfers.
# Each saved source model supplies its own source-test calendar.  A single
# intersection across *all* towers/directions is generally empty and is not a
# scientifically meaningful fairness unit.
#
# This script therefore defines the strict common-time unit as:
#
#     time scope + source -> target + physical variable
#
# Within that unit, BOTH codes (when available), BOTH networks, ALL compared
# cases, and ALL available heights use the exact same timestamps.  Thus every
# profile height and every model in one transfer/variable comparison is evaluated
# on one complete-case calendar.  Different directed transfer experiments are
# allowed to have different calendars because they are different experimental
# tasks with different source-model test periods and target-data availability.
#
# A second Xilinhaote-specific issue is temporal resolution:
# Code1 generalization is based on the D1H 1-hour data, whereas Code2 uses the
# original 10-minute ML-ready data.  For fair comparison in these postprocessors,
# Code2 prediction rows are aggregated to an hourly grid before common-time
# alignment.  WS/TKE/T use arithmetic hourly means; WD uses a circular hourly
# mean.  Code1 rows are also normalized to the same hourly labels.
#
# Final OBS and Raw WRF baselines are canonicalized per timestamp AND height.
# Code1 is preferred as the canonical baseline when Code1 is present at that
# physical point.  The original Code1-Code2 baseline spread is retained in the
# audit, so any resolution/preprocessing difference remains visible.
GLOBAL_COMMON_TIME_ALIGNMENT = True
# DISTFIX: optional Weibull/wind-rose panels may legitimately have zero common
# WS+WD rows for a direction/network. Empty distribution tables retain TIME_COL
# and safe merges skip those panels instead of terminating the full v8 workflow.
GLOBAL_COMMON_TIME_INDEX: Dict[Tuple[str, str, str, str], pd.DatetimeIndex] = {}
GLOBAL_TIME_SERIES_AUDIT = pd.DataFrame()
GLOBAL_TIME_SCOPE_AUDIT = pd.DataFrame()


def _global_scope_from_gtype(gtype: object) -> str:
    return "ALL" if str(gtype) == "cross_tower_other_time" else "TEST"


def _time_stats(values: pd.Series) -> Tuple[int, pd.Timestamp, pd.Timestamp]:
    t = pd.DatetimeIndex(
        pd.to_datetime(values, errors="coerce").dropna().unique()
    ).sort_values()
    if len(t) == 0:
        return 0, pd.NaT, pd.NaT
    return int(len(t)), pd.Timestamp(t.min()), pd.Timestamp(t.max())


def _circular_mean_deg(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    rad = np.deg2rad(x % 360.0)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    if not np.isfinite(s) or not np.isfinite(c):
        return np.nan
    return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)


def _hourly_harmonize_long(d: pd.DataFrame) -> pd.DataFrame:
    """Put Code1 and Code2 prediction rows on one 1-hour evaluation grid."""
    if d.empty:
        return d.copy()
    work = d.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    work = work.dropna(subset=[TIME_COL]).copy()
    work["_hour"] = work[TIME_COL].dt.floor("h")

    group_cols = [
        "code", "source_site", "target_site", "network", "case",
        "generalization_type", "task_name", "variable", "height_m", "_hour",
    ]
    rows = []
    for key, g in work.groupby(group_cols, dropna=False, sort=False):
        meta = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        var = str(meta["variable"])
        if var == "WD":
            obs = _circular_mean_deg(g["OBS"])
            wrf = _circular_mean_deg(g["WRF"])
            corr = _circular_mean_deg(g["CORR"])
        else:
            obs = float(pd.to_numeric(g["OBS"], errors="coerce").mean())
            wrf = float(pd.to_numeric(g["WRF"], errors="coerce").mean())
            corr = float(pd.to_numeric(g["CORR"], errors="coerce").mean())
        meta[TIME_COL] = meta.pop("_hour")
        rows.append({**meta, "OBS": obs, "WRF": wrf, "CORR": corr})
    return pd.DataFrame(rows)


def _fair_cross_scope(code: str, gtype: str) -> Optional[str]:
    g = str(gtype)
    if code == "Code1" and g in {"cross_tower_same_height", "cross_tower_other_height"}:
        return "same_time"
    if code == "Code2" and g == "cross_tower_same_time":
        return "same_time"
    if g == "cross_tower_other_time":
        return "all_target_time"
    return None


def _fair_code1_long(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    rows = []
    meta = ["source_site", "target_site", "network", "case", "generalization_type", "task_name"]
    if any(c not in pred.columns for c in meta[:-1]):
        return pd.DataFrame()
    p = pred.copy()
    p[TIME_COL] = pd.to_datetime(p[TIME_COL], errors="coerce")
    if "task_name" not in p.columns:
        p["task_name"] = ""
    for var in ["WS", "WD", "TKE", "T"]:
        obs, wrf, corr = f"OBS_{var}", f"WRF_{var}", f"CORR_{var}"
        hcol = f"{var}_height_m"
        if not {obs, wrf, corr, hcol}.issubset(p.columns):
            continue
        d = p[meta + [TIME_COL, hcol, obs, wrf, corr]].copy()
        d = d.rename(columns={hcol: "height_m", obs: "OBS", wrf: "WRF", corr: "CORR"})
        d["height_m"] = pd.to_numeric(d["height_m"], errors="coerce")
        for c in ["OBS", "WRF", "CORR"]:
            d[c] = numeric(d[c])
        d = d.dropna(subset=[TIME_COL, "height_m"])
        d["variable"] = var
        d["code"] = "Code1"
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fair_code2_long(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    meta = ["source_site", "target_site", "network", "case", "generalization_type", "task_name"]
    if any(c not in pred.columns for c in meta[:-1]):
        return pd.DataFrame()
    p = pred.copy()
    p[TIME_COL] = pd.to_datetime(p[TIME_COL], errors="coerce")
    if "task_name" not in p.columns:
        p["task_name"] = ""
    rows = []
    pattern = re.compile(r"^OBS_(WS|WD|TKE|T)(\d+)$")
    for col in p.columns:
        m = pattern.fullmatch(str(col))
        if not m:
            continue
        var, htxt = m.group(1), m.group(2)
        h = int(htxt)
        wrf, corr = f"WRF_{var}{h}", f"CORR_{var}{h}"
        if wrf not in p.columns or corr not in p.columns:
            continue
        d = p[meta + [TIME_COL, col, wrf, corr]].copy()
        d = d.rename(columns={col: "OBS", wrf: "WRF", corr: "CORR"})
        for c in ["OBS", "WRF", "CORR"]:
            d[c] = numeric(d[c])
        d["height_m"] = float(h)
        d["variable"] = var
        d["code"] = "Code2"
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fair_group_id(row: pd.Series) -> Tuple[object, ...]:
    code = str(row["code"])
    gtype = str(row["generalization_type"])
    var = str(row["variable"])
    source = str(row["source_site"])
    target = str(row["target_site"])
    cross_scope = _fair_cross_scope(code, gtype)

    # Cross-tower: one exact timestamp set across Code1+Code2, both networks,
    # all cases, and every available height for the same direction+variable.
    if cross_scope is not None:
        return ("CROSS_DIRECTION_VARIABLE", cross_scope, source, target, var)

    # Code1-only tasks (e.g. same-tower other-height): one exact timestamp set
    # across all cases/networks/heights for the same physical experiment.
    return (
        "WITHIN_CODE_DIRECTION_VARIABLE",
        _global_scope_from_gtype(gtype), code, gtype, source, target, var,
    )


def _fair_group_audit_fields(fair_group: Tuple[object, ...]) -> Dict[str, object]:
    if fair_group and fair_group[0] == "CROSS_DIRECTION_VARIABLE":
        _, fair_scope, source, target, variable = fair_group
        return {
            "group_type": "CROSS_DIRECTION_VARIABLE",
            "scope": "ALL" if str(fair_scope) == "all_target_time" else "TEST",
            "fair_scope": str(fair_scope),
            "source_site": str(source),
            "target_site": str(target),
            "variable": str(variable),
            "code": "Code1+Code2",
            "generalization_type": "cross-code aligned",
        }
    if fair_group and fair_group[0] == "WITHIN_CODE_DIRECTION_VARIABLE":
        _, scope, code, gtype, source, target, variable = fair_group
        return {
            "group_type": "WITHIN_CODE_DIRECTION_VARIABLE",
            "scope": str(scope),
            "fair_scope": str(gtype),
            "source_site": str(source),
            "target_site": str(target),
            "variable": str(variable),
            "code": str(code),
            "generalization_type": str(gtype),
        }
    return {"group_type": "UNKNOWN", "scope": "UNKNOWN"}


def _build_direction_common_audits(
    combined: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str, str, str], pd.DatetimeIndex], pd.DataFrame, pd.DataFrame]:
    """Build strict common time per scope+direction+variable instead of all towers at once."""
    if combined.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    series_rows = []
    group_rows = []
    index_map: Dict[Tuple[str, str, str, str], pd.DatetimeIndex] = {}

    combined = combined.copy()
    combined["_fair_group"] = combined.apply(_fair_group_id, axis=1)

    series_cols = [
        "code", "network", "case", "generalization_type", "task_name", "height_m"
    ]

    for fair_group, g in combined.groupby("_fair_group", dropna=False, sort=False):
        fields = _fair_group_audit_fields(fair_group)
        valid_sets = []
        n_nonempty = 0
        n_zero = 0

        for sid, gs in g.groupby(series_cols, dropna=False, sort=False):
            smeta = dict(zip(series_cols, sid if isinstance(sid, tuple) else (sid,)))
            gs = gs.copy()
            obs_mask = gs[TIME_COL].notna() & gs["OBS"].notna()
            wrf_mask = gs[TIME_COL].notna() & gs["WRF"].notna()
            corr_mask = gs[TIME_COL].notna() & gs["CORR"].notna()
            complete_mask = obs_mask & wrf_mask & corr_mask

            obs_n, obs_start, obs_end = _time_stats(gs.loc[obs_mask, TIME_COL])
            wrf_n, wrf_start, wrf_end = _time_stats(gs.loc[wrf_mask, TIME_COL])
            corr_n, corr_start, corr_end = _time_stats(gs.loc[corr_mask, TIME_COL])
            complete_n, complete_start, complete_end = _time_stats(gs.loc[complete_mask, TIME_COL])

            if complete_n > 0:
                t = pd.DatetimeIndex(
                    gs.loc[complete_mask, TIME_COL].drop_duplicates().sort_values()
                )
                valid_sets.append(set(t.asi8.tolist()))
                n_nonempty += 1
                status = "INCLUDED_IN_DIRECTION_VARIABLE_INTERSECTION"
                include = True
            else:
                n_zero += 1
                status = "NO_COMPLETE_VALID_TIMES_EXCLUDED"
                include = False

            series_rows.append({
                **fields, **smeta,
                "OBS_N": obs_n, "OBS_start": obs_start, "OBS_end": obs_end,
                "WRF_N": wrf_n, "WRF_start": wrf_start, "WRF_end": wrf_end,
                "MODEL_N": corr_n, "MODEL_start": corr_start, "MODEL_end": corr_end,
                "COMPLETE_N": complete_n,
                "COMPLETE_start": complete_start,
                "COMPLETE_end": complete_end,
                "included_in_common_intersection": include,
                "series_status": status,
            })

        common_ns = set.intersection(*valid_sets) if valid_sets else set()
        common = (
            pd.DatetimeIndex(pd.to_datetime(sorted(common_ns)))
            if common_ns else pd.DatetimeIndex([])
        )

        if len(common):
            start = pd.Timestamp(common.min())
            end = pd.Timestamp(common.max())
            expected_hourly = int((end - start) / pd.Timedelta(hours=1)) + 1
            missing_inside = int(expected_hourly - len(common))
        else:
            start = end = pd.NaT
            expected_hourly = 0
            missing_inside = 0

        group_rows.append({
            **fields,
            "N_global_common": int(len(common)),
            "time_start": start,
            "time_end": end,
            "n_nonempty_series_in_intersection": int(n_nonempty),
            "n_zero_complete_series_excluded": int(n_zero),
            "hourly_slots_between_start_end": expected_hourly,
            "missing_hourly_slots_inside_span": missing_inside,
            "status": "PASS" if len(common) else "NO_DIRECTION_VARIABLE_COMMON_TIME",
        })

        if fair_group and fair_group[0] == "CROSS_DIRECTION_VARIABLE":
            _, fair_scope, source, target, variable = fair_group
            key = (
                "ALL" if str(fair_scope) == "all_target_time" else "TEST",
                str(source), str(target), str(variable),
            )
            index_map[key] = common

    series_audit = pd.DataFrame(series_rows)
    group_audit = pd.DataFrame(group_rows)

    if not series_audit.empty and not group_audit.empty:
        lookup = group_audit[
            ["group_type", "scope", "source_site", "target_site", "variable",
             "code", "generalization_type", "N_global_common", "time_start", "time_end"]
        ].copy()
        # A direct merge can duplicate rows when multiple within-code groups share
        # metadata, so use a compact textual key.
        def kfun(df):
            return (
                df["group_type"].astype(str) + "|" +
                df["scope"].astype(str) + "|" +
                df["source_site"].astype(str) + "|" +
                df["target_site"].astype(str) + "|" +
                df["variable"].astype(str) + "|" +
                df["code"].astype(str) + "|" +
                df["generalization_type"].astype(str)
            )
        lookup["_k"] = kfun(lookup)
        lookup = lookup.drop_duplicates("_k").set_index("_k")
        series_audit["_k"] = kfun(series_audit)
        series_audit["USED_N"] = series_audit["_k"].map(lookup["N_global_common"])
        series_audit["USED_start"] = series_audit["_k"].map(lookup["time_start"])
        series_audit["USED_end"] = series_audit["_k"].map(lookup["time_end"])
        series_audit = series_audit.drop(columns=["_k"])

    return index_map, series_audit, group_audit


def _safe_token(x: object) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(x))


def _write_global_time_audit_files(*roots: Path) -> None:
    """Write transfer-specific common-time CSV/TXT audits."""
    roots = tuple(Path(r) for r in roots if r is not None)
    if not roots:
        return

    lines = [
        "Xilinhaote FAIR DIRECTION-COMMON TIME AUDIT",
        "=" * 100,
        "Why there is not one universal timestamp set:",
        "  Xilinhaote has four source towers and 12 directed transfers. Each source",
        "  model has its own source-test calendar, so intersecting every transfer at",
        "  once can legitimately produce N=0.",
        "",
        "Fairness unit used here:",
        "  scope + source->target + physical variable.",
        "  Within each unit, both codes (when available), both networks, all cases,",
        "  and all available heights use the exact same hourly timestamps.",
        "",
        "Temporal-resolution harmonization:",
        "  Code1 is already 1-hour. Code2 native 10-minute predictions are averaged",
        "  to the same hourly grid before fair-time statistics and profile means.",
        "  WD uses a circular hourly mean.",
        "",
    ]

    if GLOBAL_TIME_SCOPE_AUDIT.empty:
        lines.append("No common-time groups were produced.")
    else:
        for _, r in GLOBAL_TIME_SCOPE_AUDIT.iterrows():
            lines.extend([
                f"[{r.get('scope','?')}] {r.get('source_site','?')} -> {r.get('target_site','?')} "
                f"| {r.get('variable','?')} | {r.get('group_type','?')}",
                f"  start = {r.get('time_start', pd.NaT)}",
                f"  end   = {r.get('time_end', pd.NaT)}",
                f"  N     = {int(r.get('N_global_common', 0))}",
                f"  included non-empty series = {int(r.get('n_nonempty_series_in_intersection', 0))}",
                f"  structurally empty series excluded = {int(r.get('n_zero_complete_series_excluded', 0))}",
                f"  status = {r.get('status','')}",
                "",
            ])

    lines.extend([
        "global_time_series_audit.csv gives per-series availability.",
        "global_time_scope_audit.csv gives every direction-variable common set.",
        "Exact cross-code timestamps are written as:",
        "COMMON_TIMESTAMPS_<TEST|ALL>_<SOURCE>_to_<TARGET>_<VARIABLE>.csv",
        "",
        "For backward compatibility, GLOBAL_COMMON_TIMESTAMPS_TEST.csv and",
        "GLOBAL_COMMON_TIMESTAMPS_ALL.csv are aliases ONLY for the configured SCI",
        f"representative direction {SCI_TS_SOURCE_SITE}->{SCI_TS_TARGET_SITE}, WS.",
    ])
    txt = "\n".join(lines)

    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        GLOBAL_TIME_SCOPE_AUDIT.to_csv(
            root / "global_time_scope_audit.csv", index=False, encoding="utf-8-sig"
        )
        GLOBAL_TIME_SERIES_AUDIT.to_csv(
            root / "global_time_series_audit.csv", index=False, encoding="utf-8-sig"
        )
        (root / "TIME_RANGE_AUDIT.txt").write_text(txt, encoding="utf-8")

        for key, idx in GLOBAL_COMMON_TIME_INDEX.items():
            if len(idx) == 0:
                continue
            scope, source, target, variable = key
            name = (
                f"COMMON_TIMESTAMPS_{_safe_token(scope)}_"
                f"{_safe_token(source)}_to_{_safe_token(target)}_"
                f"{_safe_token(variable)}.csv"
            )
            pd.DataFrame({TIME_COL: pd.DatetimeIndex(idx).sort_values()}).to_csv(
                root / name, index=False, encoding="utf-8-sig"
            )

            # Representative-direction alias for the selected-case postprocessor.
            if (
                str(source) == str(SCI_TS_SOURCE_SITE)
                and str(target) == str(SCI_TS_TARGET_SITE)
                and str(variable) == "WS"
            ):
                pd.DataFrame({TIME_COL: pd.DatetimeIndex(idx).sort_values()}).to_csv(
                    root / f"GLOBAL_COMMON_TIMESTAMPS_{scope}.csv",
                    index=False, encoding="utf-8-sig"
                )


def build_fair_time_aligned_long(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex]]:
    global GLOBAL_COMMON_TIME_INDEX, GLOBAL_TIME_SERIES_AUDIT, GLOBAL_TIME_SCOPE_AUDIT

    c1 = _fair_code1_long(c1_pred)
    c2 = _fair_code2_long(c2_pred)
    parts = [x for x in [c1, c2] if not x.empty]
    if not parts:
        raise ValueError("No prediction rows were available for fair-time recomputation.")

    # Harmonize the temporal resolution BEFORE constructing any common mask.
    combined = _hourly_harmonize_long(pd.concat(parts, ignore_index=True))
    if combined.empty:
        raise ValueError("Prediction rows became empty after hourly harmonization.")

    combined["_fair_group"] = combined.apply(_fair_group_id, axis=1)
    GLOBAL_COMMON_TIME_INDEX, GLOBAL_TIME_SERIES_AUDIT, GLOBAL_TIME_SCOPE_AUDIT = \
        _build_direction_common_audits(combined)

    aligned_parts = []
    audit_rows = []
    time_index: Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex] = {}
    series_cols = [
        "code", "network", "case", "generalization_type", "task_name", "height_m"
    ]

    for fair_group, g in combined.groupby("_fair_group", dropna=False, sort=False):
        fields = _fair_group_audit_fields(fair_group)

        # Reconstruct this group's exact common set from all non-empty model series.
        valid_sets = []
        n_series_with_valid = 0
        for sid, gs in g.groupby(series_cols, dropna=False, sort=False):
            valid = gs[
                gs[TIME_COL].notna()
                & gs["OBS"].notna()
                & gs["WRF"].notna()
                & gs["CORR"].notna()
            ]
            times = pd.DatetimeIndex(valid[TIME_COL].drop_duplicates().sort_values())
            if len(times):
                valid_sets.append(set(times.asi8.tolist()))
                n_series_with_valid += 1

        common_ns = set.intersection(*valid_sets) if valid_sets else set()
        common = (
            pd.DatetimeIndex(pd.to_datetime(sorted(common_ns)))
            if common_ns else pd.DatetimeIndex([])
        )

        if len(common) == 0:
            audit_rows.append({
                "fair_group": repr(fair_group), **fields,
                "n_series_with_valid": n_series_with_valid,
                "N_common": 0,
                "time_start": pd.NaT,
                "time_end": pd.NaT,
                "baseline_obs_max_spread": np.nan,
                "baseline_wrf_max_spread": np.nan,
                "status": "NO_DIRECTION_VARIABLE_COMMON_TIME",
            })
            continue

        gg = g[g[TIME_COL].isin(common)].copy()
        gg = gg[
            gg["OBS"].notna() & gg["WRF"].notna() & gg["CORR"].notna()
        ].copy()
        if gg.empty:
            continue

        # Baseline comparison must be done at the same time AND physical height.
        spread = gg.groupby([TIME_COL, "height_m"], dropna=False).agg(
            obs_min=("OBS", "min"),
            obs_max=("OBS", "max"),
            wrf_min=("WRF", "min"),
            wrf_max=("WRF", "max"),
        )
        obs_spread = float((spread["obs_max"] - spread["obs_min"]).abs().max())
        wrf_spread = float((spread["wrf_max"] - spread["wrf_min"]).abs().max())

        # Prefer the Code1 hourly baseline when it exists.  This creates one
        # identical OBS/Raw-WRF reference for Code1 and the hourly-aggregated
        # Code2 predictions, while preserving original spreads in the audit.
        baseline_rows = []
        for (t, h), gb in gg.groupby([TIME_COL, "height_m"], dropna=False, sort=False):
            g1 = gb[gb["code"].astype(str).eq("Code1")]
            src = g1 if not g1.empty else gb
            baseline_rows.append({
                TIME_COL: t,
                "height_m": h,
                "OBS_canonical": float(pd.to_numeric(src["OBS"], errors="coerce").dropna().iloc[0])
                    if pd.to_numeric(src["OBS"], errors="coerce").notna().any() else np.nan,
                "WRF_canonical": float(pd.to_numeric(src["WRF"], errors="coerce").dropna().iloc[0])
                    if pd.to_numeric(src["WRF"], errors="coerce").notna().any() else np.nan,
                "baseline_source": "Code1" if not g1.empty else str(src.iloc[0]["code"]),
            })
        baseline = pd.DataFrame(baseline_rows)

        gg = gg.merge(
            baseline,
            on=[TIME_COL, "height_m"],
            how="left",
        )
        gg["OBS"] = gg["OBS_canonical"]
        gg["WRF"] = gg["WRF_canonical"]
        gg = gg.drop(columns=["OBS_canonical", "WRF_canonical"])
        gg["fair_time_aligned"] = True
        gg["global_common_time_aligned"] = True
        gg["global_scope"] = fields.get("scope")
        gg["fair_N_common"] = int(len(common))
        aligned_parts.append(gg)

        audit_rows.append({
            "fair_group": repr(fair_group), **fields,
            "n_series_with_valid": n_series_with_valid,
            "N_common": int(len(common)),
            "time_start": common.min(),
            "time_end": common.max(),
            "baseline_obs_max_spread": obs_spread,
            "baseline_wrf_max_spread": wrf_spread,
            "baseline_policy": "Code1 preferred canonical hourly baseline",
            "status": (
                "PASS_CANONICALIZED_BASELINE"
                if (obs_spread > FAIR_BASELINE_ATOL or wrf_spread > FAIR_BASELINE_ATOL)
                else "PASS_IDENTICAL_BASELINE"
            ),
        })

        # Downstream WS/WD time-series and distribution filters use a
        # height-specific key, but every height in this direction-variable group
        # receives the SAME common timestamps.
        fair_scope = fields.get("fair_scope")
        if str(fields.get("group_type")) == "CROSS_DIRECTION_VARIABLE":
            source = str(fields["source_site"])
            target = str(fields["target_site"])
            variable = str(fields["variable"])
            for h in sorted(pd.to_numeric(gg["height_m"], errors="coerce").dropna().unique()):
                time_index[
                    (str(fair_scope), source, target, variable, float(h))
                ] = common

    aligned = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    return aligned, audit, time_index



# =============================================================================
# MEMORY-SAFE FAIR-TIME ENGINE FOR XILINHAOTE
# =============================================================================
# The Code2 prediction table is wide (many height-specific OBS/WRF/CORR heads)
# and contains native 10-minute rows. Expanding the *entire* table to long form
# before hourly aggregation can create tens of millions of object rows and use
# several GB of RAM. The engine below processes one direction/variable at a time,
# aggregates to hourly resolution first, and retains only aggregated metrics,
# profiles and audit/time-index products.


def _ms_hourly_aggregate_long(d: pd.DataFrame, var: str) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    x = d.copy()
    x[TIME_COL] = pd.to_datetime(x[TIME_COL], errors="coerce")
    x = x.dropna(subset=[TIME_COL, "height_m"]).copy()
    if x.empty:
        return x
    x[TIME_COL] = x[TIME_COL].dt.floor("h")
    group_cols = [
        "code", "source_site", "target_site", "network", "case",
        "generalization_type", "task_name", "variable", "height_m", TIME_COL,
    ]
    for c in ["OBS", "WRF", "CORR"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if str(var) != "WD":
        return (
            x.groupby(group_cols, dropna=False, sort=False, as_index=False)[["OBS", "WRF", "CORR"]]
             .mean()
        )

    # Circular hourly mean for WD, vectorized to avoid Python group loops.
    tmp = x[group_cols].copy()
    for c in ["OBS", "WRF", "CORR"]:
        rad = np.deg2rad(x[c] % 360.0)
        tmp[f"_{c}_sin"] = np.sin(rad)
        tmp[f"_{c}_cos"] = np.cos(rad)
    agg_cols = [c for c in tmp.columns if c.startswith("_")]
    g = tmp.groupby(group_cols, dropna=False, sort=False, as_index=False)[agg_cols].mean()
    for c in ["OBS", "WRF", "CORR"]:
        g[c] = (np.rad2deg(np.arctan2(g[f"_{c}_sin"], g[f"_{c}_cos"])) + 360.0) % 360.0
    return g[group_cols + ["OBS", "WRF", "CORR"]]


def _ms_code1_group_long(
    pred: pd.DataFrame,
    source: str,
    target: str,
    gtypes: Sequence[str],
    var: str,
) -> pd.DataFrame:
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    required_meta = ["source_site", "target_site", "network", "case", "generalization_type"]
    if not set(required_meta).issubset(pred.columns):
        return pd.DataFrame()
    obs, wrf, corr = f"OBS_{var}", f"WRF_{var}", f"CORR_{var}"
    hcol = f"{var}_height_m"
    if not {obs, wrf, corr, hcol}.issubset(pred.columns):
        return pd.DataFrame()

    mask = (
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["generalization_type"].astype(str).isin([str(x) for x in gtypes])
    )
    if not bool(mask.any()):
        return pd.DataFrame()
    cols = required_meta + (["task_name"] if "task_name" in pred.columns else []) + [TIME_COL, hcol, obs, wrf, corr]
    d = pred.loc[mask, cols].copy()
    if "task_name" not in d.columns:
        d["task_name"] = ""
    d = d.rename(columns={hcol: "height_m", obs: "OBS", wrf: "WRF", corr: "CORR"})
    d["height_m"] = pd.to_numeric(d["height_m"], errors="coerce")
    d["variable"] = str(var)
    d["code"] = "Code1"
    return _ms_hourly_aggregate_long(d, var)


def _ms_code2_group_long(
    pred: pd.DataFrame,
    source: str,
    target: str,
    gtypes: Sequence[str],
    var: str,
) -> pd.DataFrame:
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    required_meta = ["source_site", "target_site", "network", "case", "generalization_type"]
    if not set(required_meta).issubset(pred.columns):
        return pd.DataFrame()

    mask = (
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["generalization_type"].astype(str).isin([str(x) for x in gtypes])
    )
    if not bool(mask.any()):
        return pd.DataFrame()

    # IMPORTANT: do NOT copy/filter the full wide Code2 table. Select only one
    # OBS/WRF/CORR height triplet at a time, then aggregate that small slice to
    # hourly resolution before appending it to long form.
    rows = []
    pat = re.compile(rf"^OBS_{re.escape(str(var))}(\d+)$")
    base_cols = required_meta + (["task_name"] if "task_name" in pred.columns else []) + [TIME_COL]
    for col in pred.columns:
        m = pat.fullmatch(str(col))
        if not m:
            continue
        h = int(m.group(1))
        wrf, corr = f"WRF_{var}{h}", f"CORR_{var}{h}"
        if wrf not in pred.columns or corr not in pred.columns:
            continue
        d = pred.loc[mask, base_cols + [col, wrf, corr]].copy()
        if "task_name" not in d.columns:
            d["task_name"] = ""
        d = d.rename(columns={col: "OBS", wrf: "WRF", corr: "CORR"})
        d["height_m"] = float(h)
        d["variable"] = str(var)
        d["code"] = "Code2"
        hly = _ms_hourly_aggregate_long(d, var)
        if not hly.empty:
            rows.append(hly)
        del d, hly
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _ms_metric_row(meta: Dict[str, object], g: pd.DataFrame) -> Dict[str, object]:
    obs = pd.to_numeric(g["OBS"], errors="coerce").to_numpy(dtype=float)
    wrf = pd.to_numeric(g["WRF"], errors="coerce").to_numpy(dtype=float)
    corr = pd.to_numeric(g["CORR"], errors="coerce").to_numpy(dtype=float)
    stats = _fair_direction_metrics(obs, corr, wrf) if str(meta["variable"]) == "WD" else _fair_scalar_metrics(obs, corr, wrf)
    return {
        **meta, **stats,
        "fair_time_aligned": True,
        "fair_N_common": int(pd.DatetimeIndex(g[TIME_COL].dropna().unique()).size),
    }


def _ms_finalize_metrics(rows: List[Dict[str, object]], code: str) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["code"] = code
    for c in ["source_site", "target_site", "network", "case", "variable", "generalization_type"]:
        out[c] = out[c].astype(str)
    out["height_num"] = pd.to_numeric(out["height_m"], errors="coerce")
    out["role"] = out["case"].map(CODE1_ROLE if code == "Code1" else CODE2_ROLE)
    out["direction"] = out["source_site"] + "→" + out["target_site"]
    out["primary_metric_name"] = out["variable"].map(PRIMARY_METRIC)
    out["primary_error"] = np.where(out["variable"].eq("WD"), out["MAE"], out["RMSE"])
    return out


def _ms_specs(c1_pred: pd.DataFrame, c2_pred: pd.DataFrame) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    seen = set()

    def add(group_type, scope, source, target, c1_gtypes, c2_gtypes):
        key = (group_type, scope, str(source), str(target), tuple(c1_gtypes), tuple(c2_gtypes))
        if key in seen:
            return
        seen.add(key)
        specs.append({
            "group_type": group_type, "scope": scope,
            "source": str(source), "target": str(target),
            "c1_gtypes": list(c1_gtypes), "c2_gtypes": list(c2_gtypes),
        })

    # TEST cross-code direction groups.
    pairs = set()
    if not c1_pred.empty:
        mask = c1_pred["generalization_type"].astype(str).isin(["cross_tower_same_height", "cross_tower_other_height"])
        d = c1_pred.loc[mask, ["source_site", "target_site"]]
        pairs |= set(d.astype(str).itertuples(index=False, name=None))
    if not c2_pred.empty:
        mask = c2_pred["generalization_type"].astype(str).eq("cross_tower_same_time")
        d = c2_pred.loc[mask, ["source_site", "target_site"]]
        pairs |= set(d.astype(str).itertuples(index=False, name=None))
    for s, t in sorted(pairs):
        add("CROSS_DIRECTION_VARIABLE", "TEST", s, t,
            ["cross_tower_same_height", "cross_tower_other_height"], ["cross_tower_same_time"])

    # ALL target-time cross-code direction groups (v8; absent naturally in v5).
    pairs = set()
    if not c1_pred.empty:
        mask = c1_pred["generalization_type"].astype(str).eq("cross_tower_other_time")
        d = c1_pred.loc[mask, ["source_site", "target_site"]]
        pairs |= set(d.astype(str).itertuples(index=False, name=None))
    if not c2_pred.empty:
        mask = c2_pred["generalization_type"].astype(str).eq("cross_tower_other_time")
        d = c2_pred.loc[mask, ["source_site", "target_site"]]
        pairs |= set(d.astype(str).itertuples(index=False, name=None))
    for s, t in sorted(pairs):
        add("CROSS_DIRECTION_VARIABLE", "ALL", s, t,
            ["cross_tower_other_time"], ["cross_tower_other_time"])

    # Code1-only same-tower height-transfer groups.
    if not c1_pred.empty:
        mask = c1_pred["generalization_type"].astype(str).eq("same_tower_other_height")
        d = c1_pred.loc[mask, ["source_site", "target_site"]]
        for s, t in sorted(set(d.astype(str).itertuples(index=False, name=None))):
            add("WITHIN_CODE_DIRECTION_VARIABLE", "TEST", s, t,
                ["same_tower_other_height"], [])
    return specs


def build_fair_products_memory_safe(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex]]:
    """Memory-safe fair-time recomputation without constructing a global long table."""
    global GLOBAL_COMMON_TIME_INDEX, GLOBAL_TIME_SERIES_AUDIT, GLOBAL_TIME_SCOPE_AUDIT

    metric_rows = {"Code1": [], "Code2": []}
    profile_rows = {"Code1": [], "Code2": []}
    audit_rows: List[Dict[str, object]] = []
    series_audit_rows: List[Dict[str, object]] = []
    scope_audit_rows: List[Dict[str, object]] = []
    time_index: Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex] = {}
    GLOBAL_COMMON_TIME_INDEX = {}

    specs = _ms_specs(c1_pred, c2_pred)
    if not specs:
        raise ValueError("No Xilinhaote fair-time task specifications were found in prediction tables.")

    for spec in specs:
        source, target = str(spec["source"]), str(spec["target"])
        scope = str(spec["scope"])
        group_type = str(spec["group_type"])
        for var in ["WS", "WD", "TKE", "T"]:
            parts = []
            if spec["c1_gtypes"]:
                x1 = _ms_code1_group_long(c1_pred, source, target, spec["c1_gtypes"], var)
                if not x1.empty:
                    parts.append(x1)
            if spec["c2_gtypes"]:
                x2 = _ms_code2_group_long(c2_pred, source, target, spec["c2_gtypes"], var)
                if not x2.empty:
                    parts.append(x2)
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True)
            del parts

            series_cols = ["code", "network", "case", "generalization_type", "task_name", "height_m"]
            valid_sets = []
            n_nonempty = 0
            n_zero = 0
            for sid, gs in combined.groupby(series_cols, dropna=False, sort=False):
                smeta = dict(zip(series_cols, sid if isinstance(sid, tuple) else (sid,)))
                obs_mask = gs[TIME_COL].notna() & gs["OBS"].notna()
                wrf_mask = gs[TIME_COL].notna() & gs["WRF"].notna()
                model_mask = gs[TIME_COL].notna() & gs["CORR"].notna()
                complete = obs_mask & wrf_mask & model_mask
                obs_n, obs_start, obs_end = _time_stats(gs.loc[obs_mask, TIME_COL])
                wrf_n, wrf_start, wrf_end = _time_stats(gs.loc[wrf_mask, TIME_COL])
                model_n, model_start, model_end = _time_stats(gs.loc[model_mask, TIME_COL])
                complete_n, complete_start, complete_end = _time_stats(gs.loc[complete, TIME_COL])
                include = complete_n > 0
                if include:
                    idx = pd.DatetimeIndex(gs.loc[complete, TIME_COL].drop_duplicates().sort_values())
                    valid_sets.append(set(idx.asi8.tolist()))
                    n_nonempty += 1
                else:
                    n_zero += 1
                series_audit_rows.append({
                    "group_type": group_type, "scope": scope,
                    "source_site": source, "target_site": target, "variable": var,
                    **smeta,
                    "OBS_N": obs_n, "OBS_start": obs_start, "OBS_end": obs_end,
                    "WRF_N": wrf_n, "WRF_start": wrf_start, "WRF_end": wrf_end,
                    "MODEL_N": model_n, "MODEL_start": model_start, "MODEL_end": model_end,
                    "COMPLETE_N": complete_n, "COMPLETE_start": complete_start, "COMPLETE_end": complete_end,
                    "included_in_common_intersection": include,
                    "series_status": "INCLUDED_IN_DIRECTION_VARIABLE_INTERSECTION" if include else "NO_COMPLETE_VALID_TIMES_EXCLUDED",
                })

            common_ns = set.intersection(*valid_sets) if valid_sets else set()
            common = pd.DatetimeIndex(pd.to_datetime(sorted(common_ns))) if common_ns else pd.DatetimeIndex([])
            group_code_label = "Code1+Code2" if group_type == "CROSS_DIRECTION_VARIABLE" else "Code1"
            group_gtype_label = "cross-code aligned" if group_type == "CROSS_DIRECTION_VARIABLE" else "same_tower_other_height"
            if len(common):
                st, en = pd.Timestamp(common.min()), pd.Timestamp(common.max())
                slots = int((en - st) / pd.Timedelta(hours=1)) + 1
                missing = int(slots - len(common))
            else:
                st = en = pd.NaT; slots = missing = 0
            scope_audit_rows.append({
                "group_type": group_type, "scope": scope,
                "fair_scope": "all_target_time" if scope == "ALL" else ("same_time" if group_type == "CROSS_DIRECTION_VARIABLE" else "same_tower_other_height"),
                "source_site": source, "target_site": target, "variable": var,
                "code": group_code_label, "generalization_type": group_gtype_label,
                "N_global_common": int(len(common)), "time_start": st, "time_end": en,
                "n_nonempty_series_in_intersection": int(n_nonempty),
                "n_zero_complete_series_excluded": int(n_zero),
                "hourly_slots_between_start_end": slots,
                "missing_hourly_slots_inside_span": missing,
                "status": "PASS" if len(common) else "NO_DIRECTION_VARIABLE_COMMON_TIME",
            })

            if group_type == "CROSS_DIRECTION_VARIABLE":
                GLOBAL_COMMON_TIME_INDEX[(scope, source, target, var)] = common

            if len(common) == 0:
                audit_rows.append({
                    "group_type": group_type, "scope": scope,
                    "source_site": source, "target_site": target, "variable": var,
                    "N_common": 0, "time_start": pd.NaT, "time_end": pd.NaT,
                    "baseline_obs_max_spread": np.nan, "baseline_wrf_max_spread": np.nan,
                    "status": "NO_DIRECTION_VARIABLE_COMMON_TIME",
                })
                del combined
                continue

            gg = combined[combined[TIME_COL].isin(common)].copy()
            gg = gg[gg[["OBS", "WRF", "CORR"]].notna().all(axis=1)].copy()
            del combined
            if gg.empty:
                continue

            # Audit original baseline disagreement at identical time+height.
            spread = gg.groupby([TIME_COL, "height_m"], dropna=False).agg(
                obs_min=("OBS", "min"), obs_max=("OBS", "max"),
                wrf_min=("WRF", "min"), wrf_max=("WRF", "max"),
            )
            obs_spread = float((spread["obs_max"] - spread["obs_min"]).abs().max()) if not spread.empty else np.nan
            wrf_spread = float((spread["wrf_max"] - spread["wrf_min"]).abs().max()) if not spread.empty else np.nan

            # One canonical OBS/WRF per timestamp+height. Prefer Code1 when present.
            b = gg[[TIME_COL, "height_m", "code", "OBS", "WRF"]].copy()
            b["_priority"] = np.where(b["code"].astype(str).eq("Code1"), 0, 1)
            b = b.sort_values([TIME_COL, "height_m", "_priority"]).drop_duplicates([TIME_COL, "height_m"], keep="first")
            b = b.rename(columns={"OBS": "_OBS_CAN", "WRF": "_WRF_CAN"})[[TIME_COL, "height_m", "_OBS_CAN", "_WRF_CAN"]]
            gg = gg.merge(b, on=[TIME_COL, "height_m"], how="left")
            gg["OBS"] = gg["_OBS_CAN"]
            gg["WRF"] = gg["_WRF_CAN"]
            gg = gg.drop(columns=["_OBS_CAN", "_WRF_CAN"])

            audit_rows.append({
                "group_type": group_type, "scope": scope,
                "source_site": source, "target_site": target, "variable": var,
                "n_series_with_valid": int(n_nonempty), "N_common": int(len(common)),
                "time_start": common.min(), "time_end": common.max(),
                "baseline_obs_max_spread": obs_spread,
                "baseline_wrf_max_spread": wrf_spread,
                "baseline_policy": "Code1 preferred canonical hourly baseline",
                "status": "PASS_CANONICALIZED_BASELINE" if ((np.isfinite(obs_spread) and obs_spread > FAIR_BASELINE_ATOL) or (np.isfinite(wrf_spread) and wrf_spread > FAIR_BASELINE_ATOL)) else "PASS_IDENTICAL_BASELINE",
            })

            fair_scope = "all_target_time" if scope == "ALL" else ("same_time" if group_type == "CROSS_DIRECTION_VARIABLE" else "same_tower_other_height")
            if group_type == "CROSS_DIRECTION_VARIABLE":
                for h in sorted(pd.to_numeric(gg["height_m"], errors="coerce").dropna().unique()):
                    time_index[(fair_scope, source, target, var, float(h))] = common

            # Exact per-height/per-task metrics.
            keys = ["source_site", "target_site", "network", "case", "generalization_type", "task_name", "variable", "height_m"]
            for code in ["Code1", "Code2"]:
                dc = gg[gg["code"].astype(str).eq(code)]
                if dc.empty:
                    continue
                for key, gm in dc.groupby(keys, dropna=False, sort=False):
                    meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
                    metric_rows[code].append(_ms_metric_row(meta, gm))

                # Code2 pooled ALL_PROFILE statistics use exactly the same hourly
                # direction-variable calendar as every constituent height.
                if code == "Code2":
                    pkeys = ["source_site", "target_site", "network", "case", "generalization_type", "variable"]
                    for key, gp in dc.groupby(pkeys, dropna=False, sort=False):
                        meta = dict(zip(pkeys, key if isinstance(key, tuple) else (key,)))
                        row = _ms_metric_row({**meta, "task_name": f"FAIR_{source}_to_{target}_{meta['generalization_type']}", "height_m": "ALL_PROFILE"}, gp)
                        metric_rows[code].append(row)

                if var == "WS":
                    pkeys = ["source_site", "target_site", "network", "case", "generalization_type", "height_m"]
                    for key, gp in dc.groupby(pkeys, dropna=False, sort=False):
                        meta = dict(zip(pkeys, key if isinstance(key, tuple) else (key,)))
                        vv = gp[["OBS", "WRF", "CORR"]].dropna()
                        if vv.empty:
                            continue
                        profile_rows[code].append({
                            **meta,
                            "OBS_mean": float(pd.to_numeric(vv["OBS"], errors="coerce").mean()),
                            "WRF_mean": float(pd.to_numeric(vv["WRF"], errors="coerce").mean()),
                            "CORR_mean": float(pd.to_numeric(vv["CORR"], errors="coerce").mean()),
                            "N": int(len(vv)), "fair_time_aligned": True,
                        })
            del gg

    GLOBAL_TIME_SERIES_AUDIT = pd.DataFrame(series_audit_rows)
    GLOBAL_TIME_SCOPE_AUDIT = pd.DataFrame(scope_audit_rows)
    if not GLOBAL_TIME_SERIES_AUDIT.empty and not GLOBAL_TIME_SCOPE_AUDIT.empty:
        # Fill USED_* columns from scope-direction-variable common calendar.
        lookup = {}
        for _, r in GLOBAL_TIME_SCOPE_AUDIT.iterrows():
            lookup[(str(r["group_type"]), str(r["scope"]), str(r["source_site"]), str(r["target_site"]), str(r["variable"]))] = (r["N_global_common"], r["time_start"], r["time_end"])
        used = [lookup.get((str(r["group_type"]), str(r["scope"]), str(r["source_site"]), str(r["target_site"]), str(r["variable"])), (np.nan, pd.NaT, pd.NaT)) for _, r in GLOBAL_TIME_SERIES_AUDIT.iterrows()]
        GLOBAL_TIME_SERIES_AUDIT["USED_N"] = [u[0] for u in used]
        GLOBAL_TIME_SERIES_AUDIT["USED_start"] = [u[1] for u in used]
        GLOBAL_TIME_SERIES_AUDIT["USED_end"] = [u[2] for u in used]

    c1m = _ms_finalize_metrics(metric_rows["Code1"], "Code1")
    c2m = _ms_finalize_metrics(metric_rows["Code2"], "Code2")
    c1p = pd.DataFrame(profile_rows["Code1"])
    c2p = pd.DataFrame(profile_rows["Code2"])
    return c1m, c2m, c1p, c2p, pd.DataFrame(audit_rows), time_index

def _fair_scalar_metrics(obs: np.ndarray, pred: np.ndarray, wrf: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wrf)
    if int(m.sum()) < 1:
        return {"N": int(m.sum()), "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan,
                "Pearson_r": np.nan, "Raw_RMSE": np.nan, "Skill_RMSE_pct": np.nan}
    o, p, b = obs[m], pred[m], wrf[m]
    e, eb = p - o, b - o
    rmse = float(np.sqrt(np.mean(e ** 2)))
    raw = float(np.sqrt(np.mean(eb ** 2)))
    corr = np.nan
    if len(o) >= 2 and np.nanstd(o) > 0 and np.nanstd(p) > 0:
        corr = float(np.corrcoef(o, p)[0, 1])
    return {
        "N": int(len(o)), "RMSE": rmse, "MAE": float(np.mean(np.abs(e))),
        "MBE": float(np.mean(e)), "Pearson_r": corr, "Raw_RMSE": raw,
        "Skill_RMSE_pct": float((raw - rmse) / raw * 100.0) if raw > 1.0e-12 else np.nan,
    }


def _fair_direction_metrics(obs: np.ndarray, pred: np.ndarray, wrf: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wrf)
    if int(m.sum()) < 1:
        return {"N": int(m.sum()), "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan,
                "Pearson_r": np.nan, "Raw_RMSE": np.nan, "Skill_RMSE_pct": np.nan}
    o, p, b = obs[m], pred[m], wrf[m]
    e = ((p - o + 180.0) % 360.0) - 180.0
    eb = ((b - o + 180.0) % 360.0) - 180.0
    rmse = float(np.sqrt(np.mean(e ** 2)))
    raw = float(np.sqrt(np.mean(eb ** 2)))
    return {
        "N": int(len(o)), "RMSE": rmse, "MAE": float(np.mean(np.abs(e))),
        "MBE": float(np.mean(e)), "Pearson_r": np.nan, "Raw_RMSE": raw,
        "Skill_RMSE_pct": float((raw - rmse) / raw * 100.0) if raw > 1.0e-12 else np.nan,
    }


def fair_metrics_from_long(aligned: pd.DataFrame, code: str) -> pd.DataFrame:
    d = aligned[aligned["code"].astype(str).eq(code)].copy()
    if d.empty:
        return pd.DataFrame()
    keys = [
        "source_site", "target_site", "network", "case",
        "generalization_type", "task_name", "variable", "height_m",
    ]
    rows = []
    for key, g in d.groupby(keys, dropna=False, sort=False):
        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        obs = numeric(g["OBS"]).to_numpy(dtype=float)
        wrf = numeric(g["WRF"]).to_numpy(dtype=float)
        corr = numeric(g["CORR"]).to_numpy(dtype=float)
        stats = (
            _fair_direction_metrics(obs, corr, wrf)
            if str(meta["variable"]) == "WD"
            else _fair_scalar_metrics(obs, corr, wrf)
        )
        rows.append({
            **meta, **stats,
            "fair_time_aligned": True,
            "fair_N_common": int(len(pd.DatetimeIndex(g[TIME_COL].dropna().unique()))),
        })

    out = pd.DataFrame(rows)
    if code == "Code2" and not d.empty:
        pool_keys = [
            "source_site", "target_site", "network", "case",
            "generalization_type", "variable",
        ]
        pool_rows = []
        for key, g in d.groupby(pool_keys, dropna=False, sort=False):
            meta = dict(zip(pool_keys, key if isinstance(key, tuple) else (key,)))
            obs = numeric(g["OBS"]).to_numpy(dtype=float)
            wrf = numeric(g["WRF"]).to_numpy(dtype=float)
            corr = numeric(g["CORR"]).to_numpy(dtype=float)
            stats = (
                _fair_direction_metrics(obs, corr, wrf)
                if str(meta["variable"]) == "WD"
                else _fair_scalar_metrics(obs, corr, wrf)
            )
            pool_rows.append({
                **meta,
                "task_name": (
                    f"FAIR_{meta['source_site']}_to_{meta['target_site']}_"
                    f"{meta['generalization_type']}"
                ),
                "height_m": "ALL_PROFILE",
                **stats,
                "fair_time_aligned": True,
                "fair_N_common": int(stats["N"]),
            })
        out = pd.concat([out, pd.DataFrame(pool_rows)], ignore_index=True)

    out["code"] = code
    out["source_site"] = out["source_site"].astype(str)
    out["target_site"] = out["target_site"].astype(str)
    out["network"] = out["network"].astype(str)
    out["case"] = out["case"].astype(str)
    out["variable"] = out["variable"].astype(str)
    out["generalization_type"] = out["generalization_type"].astype(str)
    out["height_num"] = pd.to_numeric(out["height_m"], errors="coerce")
    role_map = CODE1_ROLE if code == "Code1" else CODE2_ROLE
    out["role"] = out["case"].map(role_map)
    out["direction"] = out["source_site"] + "→" + out["target_site"]
    out["primary_metric_name"] = out["variable"].map(PRIMARY_METRIC)
    out["primary_error"] = np.where(
        out["variable"].eq("WD"), out["MAE"], out["RMSE"]
    )
    return out


def fair_profile_table(aligned: pd.DataFrame, code: str) -> pd.DataFrame:
    d = aligned[
        aligned["code"].astype(str).eq(code)
        & aligned["variable"].astype(str).eq("WS")
    ].copy()
    if d.empty:
        return pd.DataFrame()
    keys = [
        "source_site", "target_site", "network", "case",
        "generalization_type", "height_m",
    ]
    rows = []
    for key, g in d.groupby(keys, dropna=False, sort=False):
        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        valid = g[[TIME_COL, "OBS", "WRF", "CORR"]].dropna().copy()
        if valid.empty:
            continue
        rows.append({
            **meta,
            "OBS_mean": float(numeric(valid["OBS"]).mean()),
            "WRF_mean": float(numeric(valid["WRF"]).mean()),
            "CORR_mean": float(numeric(valid["CORR"]).mean()),
            "N": int(len(valid)),
            "fair_time_aligned": True,
        })
    return pd.DataFrame(rows)


def _hourly_mean_timeseries(d: pd.DataFrame) -> pd.DataFrame:
    """Hourly arithmetic mean for WS time-series tables; preserves only numeric series."""
    if d.empty or TIME_COL not in d.columns:
        return d.copy()
    out = d.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL]).copy()
    out["_hour"] = out[TIME_COL].dt.floor("h")
    value_cols = [c for c in out.columns if c not in {TIME_COL, "_hour"}]
    numeric_cols = []
    for c in value_cols:
        x = pd.to_numeric(out[c], errors="coerce")
        if x.notna().any():
            out[c] = x
            numeric_cols.append(c)
    if not numeric_cols:
        return out[[TIME_COL]].drop_duplicates().sort_values(TIME_COL)
    out = (
        out.groupby("_hour", as_index=False)[numeric_cols]
        .mean()
        .rename(columns={"_hour": TIME_COL})
    )
    return out


def _fair_filter_timeseries(
    d: pd.DataFrame,
    source: str,
    target: str,
    scope: str,
    height_m: int,
) -> pd.DataFrame:
    if d.empty or TIME_COL not in d.columns:
        return d
    key_scope = "ALL" if str(scope) == "all_target_time" else "TEST"
    idx = GLOBAL_COMMON_TIME_INDEX.get(
        (key_scope, str(source), str(target), "WS"),
        pd.DatetimeIndex([]),
    )
    if len(idx) == 0:
        return d.iloc[0:0].copy()
    out = _hourly_mean_timeseries(d)
    return out[out[TIME_COL].isin(idx)].copy()


def _fair_distribution_time_index(
    source: str,
    target: str,
    scope: str,
    ws_height_m: int,
    wd_height_m: int,
) -> pd.DatetimeIndex:
    key_scope = "ALL" if str(scope) == "all_target_time" else "TEST"
    ws = GLOBAL_COMMON_TIME_INDEX.get(
        (key_scope, str(source), str(target), "WS")
    )
    wd = GLOBAL_COMMON_TIME_INDEX.get(
        (key_scope, str(source), str(target), "WD")
    )
    if ws is None or wd is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(ws.intersection(wd).sort_values())


def _hourly_wind_distribution(d: pd.DataFrame) -> pd.DataFrame:
    """Hourly WS arithmetic means and WD circular means for distribution diagnostics."""
    if d.empty or TIME_COL not in d.columns:
        return d.copy()
    work = d.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    work = work.dropna(subset=[TIME_COL]).copy()
    work["_hour"] = work[TIME_COL].dt.floor("h")
    rows = []
    for t, g in work.groupby("_hour", sort=True):
        row = {TIME_COL: t}
        for c in ["OBS_WS", "WRF_WS", "CORR_WS"]:
            if c in g.columns:
                row[c] = float(pd.to_numeric(g[c], errors="coerce").mean())
        for c in ["OBS_WD", "WRF_WD", "CORR_WD"]:
            if c in g.columns:
                row[c] = _circular_mean_deg(g[c])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6B. CURVE OVERLAP AUDIT AND PLOTTING GROUPS
# =============================================================================

def _same_index_and_values(
    a: pd.Series,
    b: pd.Series,
    atol: float = CURVE_EQUAL_ATOL,
    rtol: float = CURVE_EQUAL_RTOL,
) -> Tuple[bool, float, int]:
    """Compare two labelled numeric series on exactly the same coordinates."""
    aa = numeric(a).dropna().sort_index()
    bb = numeric(b).dropna().sort_index()
    common = aa.index.intersection(bb.index)
    if len(common) == 0:
        return False, np.nan, 0
    aa = aa.loc[common]
    bb = bb.loc[common]
    same_grid = len(aa) == len(numeric(a).dropna()) and len(bb) == len(numeric(b).dropna())
    diff = np.abs(aa.to_numpy(dtype=float) - bb.to_numpy(dtype=float))
    max_diff = float(np.nanmax(diff)) if diff.size else np.nan
    equal = bool(
        same_grid
        and np.allclose(
            aa.to_numpy(dtype=float),
            bb.to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
    )
    return equal, max_diff, int(len(common))


def group_identical_profile_cases(
    g: pd.DataFrame,
    cases: Sequence[str],
    coordinate_col: str,
    value_col: str,
    context: Dict[str, object],
) -> List[List[str]]:
    """Group cases whose profile/height-metric curves are numerically identical."""
    series_map: Dict[str, pd.Series] = {}
    for case in cases:
        gc = g[g["case"].eq(case)][[coordinate_col, value_col]].copy()
        gc[coordinate_col] = pd.to_numeric(gc[coordinate_col], errors="coerce")
        gc[value_col] = numeric(gc[value_col])
        gc = gc.dropna(subset=[coordinate_col, value_col])
        if gc.empty:
            continue
        gc = gc.groupby(coordinate_col, as_index=False)[value_col].mean().sort_values(coordinate_col)
        series_map[case] = gc.set_index(coordinate_col)[value_col]

    available = [c for c in cases if c in series_map]
    for i, case_a in enumerate(available):
        for case_b in available[i + 1:]:
            equal, max_diff, n_common = _same_index_and_values(series_map[case_a], series_map[case_b])
            CURVE_OVERLAP_RECORDS.append({
                **context,
                "case_a": case_a,
                "case_b": case_b,
                "n_common_points": n_common,
                "max_abs_difference": max_diff,
                "numerically_identical": equal,
                "atol": CURVE_EQUAL_ATOL,
                "rtol": CURVE_EQUAL_RTOL,
            })

    groups: List[List[str]] = []
    used = set()
    for case in available:
        if case in used:
            continue
        group = [case]
        used.add(case)
        for other in available:
            if other in used:
                continue
            equal, _, _ = _same_index_and_values(series_map[case], series_map[other])
            if equal:
                group.append(other)
                used.add(other)
        groups.append(group)
    return groups


def group_identical_time_series_cases(
    merged: pd.DataFrame,
    cases: Sequence[str],
    context: Dict[str, object],
) -> List[List[str]]:
    """Group cases with identical point-by-point time-series predictions."""
    available = [c for c in cases if c in merged.columns]
    for i, case_a in enumerate(available):
        for case_b in available[i + 1:]:
            aa = pd.Series(numeric(merged[case_a]).to_numpy(), index=np.arange(len(merged)))
            bb = pd.Series(numeric(merged[case_b]).to_numpy(), index=np.arange(len(merged)))
            equal, max_diff, n_common = _same_index_and_values(aa, bb)
            CURVE_OVERLAP_RECORDS.append({
                **context,
                "case_a": case_a,
                "case_b": case_b,
                "n_common_points": n_common,
                "max_abs_difference": max_diff,
                "numerically_identical": equal,
                "atol": CURVE_EQUAL_ATOL,
                "rtol": CURVE_EQUAL_RTOL,
            })

    groups: List[List[str]] = []
    used = set()
    for case in available:
        if case in used:
            continue
        group = [case]
        used.add(case)
        aa = pd.Series(numeric(merged[case]).to_numpy(), index=np.arange(len(merged)))
        for other in available:
            if other in used:
                continue
            bb = pd.Series(numeric(merged[other]).to_numpy(), index=np.arange(len(merged)))
            equal, _, _ = _same_index_and_values(aa, bb)
            if equal:
                group.append(other)
                used.add(other)
        groups.append(group)
    return groups


def combined_case_label(group: Sequence[str]) -> str:
    return " = ".join(group)


def combined_case_style(group: Sequence[str]) -> Dict[str, object]:
    first = group[0]
    return dict(CASE_STYLE.get(first, {"linestyle": "-", "marker": "o"}))


def annotate_exact_overlaps(ax: plt.Axes, groups: Sequence[Sequence[str]]) -> None:
    labels = [combined_case_label(g) for g in groups if len(g) > 1]
    if not labels:
        return
    ax.text(
        0.985,
        0.025,
        "Exact overlap: " + "; ".join(labels),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SMALL,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.65"},
    )


# =============================================================================
# 7. PLOTS: SUMMARY BARS
# =============================================================================

def metric_decimals(variable: str) -> int:
    return 1 if variable == "WD" else 2


def plot_case_summary_bars(
    df: pd.DataFrame,
    cases: Sequence[str],
    role_map: Dict[str, str],
    out_dir: Path,
    code: str,
    generalization_type: Optional[str] = None,
    all_profile_only: bool = False,
) -> None:
    d = df.copy()
    if generalization_type is not None:
        d = d[d["generalization_type"].eq(generalization_type)]
    if all_profile_only:
        d = d[d["height_m"].astype(str).str.upper().eq("ALL_PROFILE")]
    if d.empty:
        return

    variables = [
        v for v in ["WS", "WD", "TKE"]
        if (
            d["variable"].eq(v)
            & numeric(d["primary_error"]).notna()
            & numeric(d["N"]).fillna(0).gt(0)
        ).any()
    ]
    if not variables:
        return
    fig, axes = plt.subplots(1, len(variables), figsize=(5.2 * len(variables), 5.2), layout="constrained")
    if len(variables) == 1:
        axes = np.asarray([axes])

    for ax, variable in zip(axes, variables):
        gvar = d[d["variable"].eq(variable)]
        means, mins, maxs, labels = [], [], [], []
        # Raw_RMSE is available for every variable, but raw circular MAE is not
        # stored for WD. Therefore do not mislabel raw WD RMSE as WD MAE.
        raw_values = numeric(gvar["Raw_RMSE"]).dropna()
        if variable != "WD" and not raw_values.empty:
            labels.append("Raw WRF")
            means.append(float(raw_values.mean()))
            mins.append(float(raw_values.min()))
            maxs.append(float(raw_values.max()))

        for case in cases:
            vals = numeric(gvar.loc[gvar["case"].eq(case), "primary_error"]).dropna()
            if vals.empty:
                continue
            labels.append(case)
            means.append(float(vals.mean()))
            mins.append(float(vals.min()))
            maxs.append(float(vals.max()))

        if not means:
            ax.set_axis_off()
            continue
        x = np.arange(len(means))
        bars = ax.bar(x, means)
        yerr = np.vstack([np.asarray(means) - np.asarray(mins), np.asarray(maxs) - np.asarray(means)])
        ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=3, linewidth=0.9)
        label_bars(ax, bars, metric_decimals(variable))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(VARIABLE_LABEL[variable])
        ax.set_title(variable)
        ax.grid(axis="y", alpha=GRID_ALPHA)

    add_panel_labels(axes)
    title_part = GENERALIZATION_LABELS.get(generalization_type, generalization_type) if generalization_type else "All tasks"
    suffix = safe_name(generalization_type or "all")
    fig.suptitle(
        f"{code} generalization case comparison — {title_part}\n"
        "Bars show the mean; error bars show the observed run range"
    )
    if code == "Code2" and not (
        (
            d["variable"].eq("WD")
            & numeric(d["primary_error"]).notna()
            & numeric(d["N"]).fillna(0).gt(0)
        ).any()
    ):
        fig.text(
            0.5,
            0.005,
            "WD is not shown because the two towers have no common observed WD height "
            "(NO1_1159: 80 m; NO2_1107: 10 m).",
            ha="center",
            va="bottom",
            fontsize=FONT_SMALL,
        )
    save_figure(fig, out_dir / f"Fig_{code}_case_metrics_{suffix}")


def plot_positive_skill_fraction(
    summary: pd.DataFrame,
    out_dir: Path,
    code: str,
    cases: Sequence[str],
) -> None:
    if summary.empty:
        return
    d = summary[summary["case"].isin(cases) & summary["variable"].isin(["WS", "WD", "TKE"])].copy()
    if d.empty:
        return

    gen_types = [x for x in d["generalization_type"].dropna().unique()]
    for gtype in gen_types:
        dg = d[d["generalization_type"].eq(gtype)]
        variables = [
            v for v in ["WS", "WD", "TKE"]
            if (
                dg["variable"].eq(v)
                & numeric(dg["positive_skill_fraction"]).notna()
                & numeric(dg["n_runs"]).fillna(0).gt(0)
            ).any()
        ]
        if not variables:
            continue
        fig, axes = plt.subplots(1, len(variables), figsize=(5.2 * len(variables), 4.8), layout="constrained")
        if len(variables) == 1:
            axes = np.asarray([axes])
        for ax, var in zip(axes, variables):
            gv = dg[dg["variable"].eq(var)].copy()
            vals, labels = [], []
            for case in cases:
                hit = gv[gv["case"].eq(case)]
                if hit.empty:
                    continue
                vals.append(float(hit.iloc[0]["positive_skill_fraction"]) * 100.0)
                labels.append(case)
            x = np.arange(len(vals))
            bars = ax.bar(x, vals)
            label_bars(ax, bars, 0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Runs with positive RMSE skill (%)")
            ax.set_title(var)
            ax.grid(axis="y", alpha=GRID_ALPHA)
        add_panel_labels(axes)
        fig.suptitle(f"{code}: positive-skill robustness — {GENERALIZATION_LABELS.get(gtype, gtype)}")
        save_figure(fig, out_dir / f"Fig_{code}_positive_skill_{safe_name(gtype)}")


# =============================================================================
# 8. PLOTS: HEIGHT PROFILES
# =============================================================================

def plot_code1_ws_rmse_by_height(df: pd.DataFrame, out_dir: Path) -> None:
    d = df[
        df["variable"].eq("WS")
        & df["height_num"].notna()
        & df["case"].isin(CODE1_CASES)
    ].copy()
    if d.empty:
        return

    group_cols = ["generalization_type", "source_site", "target_site", "network"]
    for keys, g in d.groupby(group_cols, dropna=False):
        gtype, source, target, network = keys
        heights = sorted(g["height_num"].dropna().unique())
        if len(heights) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        raw = (
            g.groupby("height_num", as_index=False)["Raw_RMSE"]
            .mean()
            .sort_values("height_num")
        )
        ax.plot(raw["height_num"], raw["Raw_RMSE"], marker="s", linestyle="--", label="Raw WRF")
        overlap_groups = group_identical_profile_cases(
            g,
            CODE1_CASES,
            "height_num",
            "RMSE",
            {
                "plot_family": "height_metric_profile",
                "code": "Code1",
                "source_site": source,
                "target_site": target,
                "network": network,
                "generalization_type": gtype,
                "variable": "WS_RMSE",
            },
        )
        for case_group in overlap_groups:
            case = case_group[0]
            gc = (
                g[g["case"].eq(case)]
                .groupby("height_num", as_index=False)["RMSE"]
                .mean()
                .sort_values("height_num")
            )
            if not gc.empty:
                style = combined_case_style(case_group)
                ax.plot(
                    gc["height_num"],
                    gc["RMSE"],
                    linewidth=1.35,
                    markersize=5.5,
                    label=combined_case_label(case_group),
                    **style,
                )
        annotate_exact_overlaps(ax, overlap_groups)
        ax.set_xlabel("Application height (m)")
        ax.set_ylabel("WS RMSE (m s$^{-1}$)")
        ax.set_title(
            f"Code1 {source}→{target} — {network}\n"
            f"{GENERALIZATION_LABELS.get(gtype, gtype)}"
        )
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            f"Fig_Code1_WS_RMSE_height_{source}_to_{target}_{network}",
        )


def plot_code2_ws_rmse_profiles(df: pd.DataFrame, out_dir: Path) -> None:
    d = df[
        df["variable"].eq("WS")
        & df["height_num"].notna()
        & df["case"].isin(CODE2_CASES)
        & numeric(df["N"]).fillna(0).gt(0)
    ].copy()
    if d.empty:
        return

    for gtype, dg in d.groupby("generalization_type", dropna=False):
        combinations = list(
            dg[["source_site", "target_site", "network"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not combinations:
            continue
        ncols = 2
        nrows = int(math.ceil(len(combinations) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(12.8, 4.6 * nrows),
            layout="constrained",
        )
        axes_flat = np.asarray(axes).reshape(-1)

        for ax, (source, target, network) in zip(axes_flat, combinations):
            g = dg[
                dg["source_site"].eq(source)
                & dg["target_site"].eq(target)
                & dg["network"].eq(network)
            ]
            raw = (
                g.groupby("height_num", as_index=False)["Raw_RMSE"]
                .mean()
                .sort_values("height_num")
            )
            ax.plot(
                raw["Raw_RMSE"], raw["height_num"],
                marker="s", linestyle="--", label="Raw WRF",
            )
            overlap_groups = group_identical_profile_cases(
                g,
                CODE2_CASES,
                "height_num",
                "RMSE",
                {
                    "plot_family": "height_metric_profile",
                    "code": "Code2",
                    "source_site": source,
                    "target_site": target,
                    "network": network,
                    "generalization_type": gtype,
                    "variable": "WS_RMSE",
                },
            )
            for case_group in overlap_groups:
                case = case_group[0]
                gc = (
                    g[g["case"].eq(case)]
                    .groupby("height_num", as_index=False)["RMSE"]
                    .mean()
                    .sort_values("height_num")
                )
                if not gc.empty:
                    style = combined_case_style(case_group)
                    ax.plot(
                        gc["RMSE"], gc["height_num"],
                        linewidth=1.35, markersize=5.5,
                        label=combined_case_label(case_group),
                        **style,
                    )
            annotate_exact_overlaps(ax, overlap_groups)
            ax.set_xlabel("WS RMSE (m s$^{-1}$)")
            ax.set_ylabel("Height (m)")
            ax.set_title(f"{source}→{target} — {network}")
            ax.grid(alpha=GRID_ALPHA)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

        for ax in axes_flat[len(combinations):]:
            ax.set_axis_off()
        handles, labels = collect_legend(
            axes_flat[:len(combinations)]
        )
        if handles:
            fig.legend(
                handles, labels, frameon=False,
                loc="lower center", ncol=min(6, len(labels)),
                bbox_to_anchor=(0.5, -0.01),
            )
        add_panel_labels(axes_flat[:len(combinations)])
        fig.suptitle(
            "Code2 WS RMSE profiles — "
            f"{GENERALIZATION_LABELS.get(gtype, gtype)}"
        )
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            "Fig_Code2_WS_RMSE_profiles",
        )



def plot_code2_exact_run_all_profile_ws(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Exact-run Code2 WS comparisons, separated by time definition."""
    d = df[
        df["variable"].eq("WS")
        & df["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & df["case"].isin(CODE2_CASES)
        & numeric(df["N"]).fillna(0).gt(0)
        & numeric(df["RMSE"]).notna()
    ].copy()

    for gtype, dg in d.groupby("generalization_type", dropna=False):
        combos = list(
            dg[["source_site", "target_site", "network"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not combos:
            continue
        ncols = 2
        nrows = int(math.ceil(len(combos) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(11.8, 4.4 * nrows),
            layout="constrained",
        )
        axes_flat = np.asarray(axes).reshape(-1)
        for ax, (source, target, network) in zip(axes_flat, combos):
            g = dg[
                dg["source_site"].eq(source)
                & dg["target_site"].eq(target)
                & dg["network"].eq(network)
            ].copy()
            values, labels = [], []
            for case in CODE2_CASES:
                hit = g[g["case"].eq(case)]
                if hit.empty:
                    continue
                values.append(float(hit.iloc[0]["RMSE"]))
                labels.append(case)
            x = np.arange(len(values))
            bars = ax.bar(x, values)
            label_bars(ax, bars, 2)
            if values:
                best_i = int(np.argmin(values))
                bars[best_i].set_hatch("///")
                bars[best_i].set_linewidth(1.4)
                bars[best_i].set_edgecolor("black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Pooled WS RMSE (m s$^{-1}$)")
            ax.set_title(f"{source}→{target} — {network}")
            ax.grid(axis="y", alpha=GRID_ALPHA)
        for ax in axes_flat[len(combos):]:
            ax.set_axis_off()
        add_panel_labels(axes_flat[:len(combos)])
        fig.suptitle(
            "Code2 exact-run WS generalization — "
            f"{GENERALIZATION_LABELS.get(gtype, gtype)}\n"
            "Hatched bar marks the lowest ALL_PROFILE RMSE"
        )
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            "Fig_Code2_exact_run_ALL_PROFILE_WS_RMSE",
        )


# =============================================================================
# 9. PLOTS: COMBINED MEAN PROFILES FROM PREDICTIONS
# =============================================================================

def code1_profile_table(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    required = {
        "source_site", "target_site", "network", "case",
        "generalization_type", "WS_height_m", "OBS_WS", "WRF_WS", "CORR_WS",
    }
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    rows = []
    keys = ["source_site", "target_site", "network", "case", "generalization_type", "WS_height_m"]
    for key, g in pred.groupby(keys, dropna=False):
        source, target, network, case, gtype, height = key
        rows.append({
            "source_site": source,
            "target_site": target,
            "network": network,
            "case": case,
            "generalization_type": gtype,
            "height_m": float(height),
            "OBS_mean": numeric(g["OBS_WS"]).mean(),
            "WRF_mean": numeric(g["WRF_WS"]).mean(),
            "CORR_mean": numeric(g["CORR_WS"]).mean(),
            "N": int(numeric(g["OBS_WS"]).notna().sum()),
        })
    return pd.DataFrame(rows)


def code2_profile_table(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    meta_cols = ["source_site", "target_site", "network", "case", "generalization_type"]
    if not set(meta_cols).issubset(pred.columns):
        return pd.DataFrame()
    rows = []
    for key, g in pred.groupby(meta_cols, dropna=False):
        meta = dict(zip(meta_cols, key if isinstance(key, tuple) else (key,)))
        heights = []
        for col in g.columns:
            m = re.fullmatch(r"OBS_WS(\d+)", str(col))
            if m:
                heights.append(int(m.group(1)))
        for h in sorted(set(heights)):
            obs, wrf, corr = f"OBS_WS{h}", f"WRF_WS{h}", f"CORR_WS{h}"
            if not all(c in g.columns for c in [obs, wrf, corr]):
                continue
            rows.append({
                **meta,
                "height_m": h,
                "OBS_mean": numeric(g[obs]).mean(),
                "WRF_mean": numeric(g[wrf]).mean(),
                "CORR_mean": numeric(g[corr]).mean(),
                "N": int(numeric(g[obs]).notna().sum()),
            })
    return pd.DataFrame(rows)



def exact_all_profile_rmse_map(
    metrics: Optional[pd.DataFrame],
    code: str,
    source: str,
    target: str,
    network: str,
    gtype: str,
    cases: Sequence[str],
) -> Dict[str, float]:
    """Return exact-run pooled WS RMSE for the current transfer/network."""
    if metrics is None or metrics.empty or code != "Code2":
        return {}
    d = metrics[
        metrics["source_site"].astype(str).eq(str(source))
        & metrics["target_site"].astype(str).eq(str(target))
        & metrics["network"].astype(str).eq(str(network))
        & metrics["generalization_type"].astype(str).eq(str(gtype))
        & metrics["variable"].astype(str).eq("WS")
        & metrics["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & metrics["case"].astype(str).isin(list(cases))
        & numeric(metrics["N"]).fillna(0).gt(0)
    ].copy()
    result: Dict[str, float] = {}
    for case in cases:
        hit = d[d["case"].astype(str).eq(str(case))]
        if not hit.empty:
            value = pd.to_numeric(hit.iloc[0]["RMSE"], errors="coerce")
            if np.isfinite(value):
                result[str(case)] = float(value)
    return result


def case_group_label_with_rmse(
    case_group: Sequence[str],
    rmse_map: Dict[str, float],
    best_case: Optional[str],
) -> str:
    base = combined_case_label(case_group)
    available = [rmse_map[c] for c in case_group if c in rmse_map]
    if not available:
        return base
    value = float(available[0])
    is_best = best_case is not None and best_case in case_group
    tag = " [best]" if is_best else ""
    return f"{base}{tag} (pooled RMSE={value:.2f})"


def plot_combined_mean_profiles(
    profile: pd.DataFrame,
    cases: Sequence[str],
    out_dir: Path,
    code: str,
    metrics: Optional[pd.DataFrame] = None,
) -> None:
    if profile.empty:
        return
    group_cols = ["source_site", "target_site", "network", "generalization_type"]
    for keys, g in profile.groupby(group_cols, dropna=False):
        source, target, network, gtype = keys
        if g["height_m"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 6.7))
        baseline = (
            g.groupby("height_m", as_index=False)
            .agg(OBS_mean=("OBS_mean", "mean"), WRF_mean=("WRF_mean", "mean"))
            .sort_values("height_m")
        )
        ax.plot(baseline["OBS_mean"], baseline["height_m"], marker="o", linewidth=1.8, label="OBS")
        ax.plot(baseline["WRF_mean"], baseline["height_m"], marker="s", linestyle="--", linewidth=1.5, label="Raw WRF")
        rmse_map = exact_all_profile_rmse_map(
            metrics, code, source, target, network, gtype, cases
        )
        best_case = min(rmse_map, key=rmse_map.get) if rmse_map else None

        overlap_groups = group_identical_profile_cases(
            g,
            cases,
            "height_m",
            "CORR_mean",
            {
                "plot_family": "mean_WS_profile",
                "code": code,
                "source_site": source,
                "target_site": target,
                "network": network,
                "generalization_type": gtype,
                "variable": "Mean_WS",
            },
        )
        for case_group in overlap_groups:
            case = case_group[0]
            gc = g[g["case"].eq(case)].sort_values("height_m")
            if not gc.empty:
                style = combined_case_style(case_group)
                ax.plot(
                    gc["CORR_mean"],
                    gc["height_m"],
                    linewidth=1.45,
                    markersize=5.8,
                    label=case_group_label_with_rmse(case_group, rmse_map, best_case),
                    **style,
                )
        annotate_exact_overlaps(ax, overlap_groups)
        ax.set_xlabel("Mean WS (m s$^{-1}$)")
        ax.set_ylabel("Height (m)")
        ax.set_title(
            f"{code} {source}→{target} — {network}\n"
            f"{GENERALIZATION_LABELS.get(gtype, gtype)}"
        )
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(frameon=False, ncol=1 if code == "Code2" else 2)
        if code == "Code2" and rmse_map:
            ax.text(
                0.015,
                0.015,
                "Curves show time-mean profiles.\n"
                "Legend RMSE ranks all hourly-height pairs, not only the six means.",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=FONT_SMALL,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.84,
                    "edgecolor": "0.65",
                },
            )
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            f"Fig_{code}_mean_profile_{source}_to_{target}_{network}",
        )


# =============================================================================
# 10. CROSS-CODE 100 M COMPARISON
# =============================================================================

ROLE_CASE_CODE1 = {v: k for k, v in CODE1_ROLE.items()}
ROLE_CASE_CODE2 = {v: k for k, v in CODE2_ROLE.items()}


def build_cross_code_10m_table(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Code1 and Code2 at 10 m for both time definitions."""
    scope_specs = [
        (
            "same_time",
            "cross_tower_same_height",
            "cross_tower_same_time",
        ),
        (
            "all_target_time",
            "cross_tower_other_time",
            "cross_tower_other_time",
        ),
    ]
    rows = []
    common_keys = ["source_site", "target_site", "network"]

    for time_scope, c1_gtype, c2_gtype in scope_specs:
        c1s = c1[
            c1["generalization_type"].eq(c1_gtype)
            & c1["variable"].eq("WS")
            & c1["height_num"].eq(float(REFERENCE_WS_HEIGHT_M))
        ].copy()
        c2s = c2[
            c2["generalization_type"].eq(c2_gtype)
            & c2["variable"].eq("WS")
            & c2["height_num"].eq(float(REFERENCE_WS_HEIGHT_M))
        ].copy()

        for key, g1 in c1s.groupby(common_keys, dropna=False):
            source, target, network = key
            g2 = c2s[
                c2s["source_site"].eq(source)
                & c2s["target_site"].eq(target)
                & c2s["network"].eq(network)
            ]
            for role in ROLE_ORDER:
                c1_case = ROLE_CASE_CODE1[role]
                c2_case = ROLE_CASE_CODE2[role]
                h1 = g1[g1["case"].eq(c1_case)]
                h2 = g2[g2["case"].eq(c2_case)]
                if h1.empty or h2.empty:
                    continue
                r1, r2 = h1.iloc[0], h2.iloc[0]
                rows.append({
                    "time_scope": time_scope,
                    "Code1_generalization_type": c1_gtype,
                    "Code2_generalization_type": c2_gtype,
                    "source_site": source,
                    "target_site": target,
                    "direction": f"{source}→{target}",
                    "network": network,
                    "height_m": REFERENCE_WS_HEIGHT_M,
                    "role": role,
                    "Code1_case": c1_case,
                    "Code2_case": c2_case,
                    "Code1_RMSE": r1["RMSE"],
                    "Code2_RMSE": r2["RMSE"],
                    "Code1_skill_pct": r1["Skill_RMSE_pct"],
                    "Code2_skill_pct": r2["Skill_RMSE_pct"],
                    "Code2_minus_Code1_RMSE": (
                        r2["RMSE"] - r1["RMSE"]
                    ),
                    "Raw_RMSE_Code1": r1["Raw_RMSE"],
                    "Raw_RMSE_Code2": r2["Raw_RMSE"],
                    "Code1_N": r1["N"],
                    "Code2_N": r2["N"],
                })
    return pd.DataFrame(rows)


def plot_cross_code_10m(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        return

    for time_scope, dt in table.groupby("time_scope", dropna=False):
        combinations = list(
            dt[["direction", "network"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        ncols = 2
        nrows = int(math.ceil(len(combinations) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(13.0, 4.8 * nrows),
            layout="constrained",
        )
        axes_flat = np.asarray(axes).reshape(-1)

        for ax, (direction, network) in zip(
            axes_flat, combinations
        ):
            g = dt[
                dt["direction"].eq(direction)
                & dt["network"].eq(network)
            ].copy()
            g["role"] = pd.Categorical(
                g["role"], categories=ROLE_ORDER, ordered=True
            )
            g = g.sort_values("role")
            x = np.arange(len(g))
            width = 0.35
            b1 = ax.bar(
                x - width / 2, g["Code1_RMSE"],
                width, label="Code1",
            )
            b2 = ax.bar(
                x + width / 2, g["Code2_RMSE"],
                width, label="Code2",
            )
            label_bars(ax, b1, 2)
            label_bars(ax, b2, 2)
            ax.set_xticks(x)
            ax.set_xticklabels(
                g["role"].astype(str), rotation=18, ha="right"
            )
            ax.set_ylabel("WS RMSE at 10 m (m s$^{-1}$)")
            ax.set_title(f"{direction} — {network}")
            ax.grid(axis="y", alpha=GRID_ALPHA)
            ax.legend(frameon=False)

        for ax in axes_flat[len(combinations):]:
            ax.set_axis_off()
        add_panel_labels(axes_flat[:len(combinations)])
        scope_title = (
            "source-test common times"
            if time_scope == "same_time"
            else "all valid target times"
        )
        fig.suptitle(
            "Code1 and Code2 cross-tower generalization at 10 m — "
            f"{scope_title}"
        )
        save_figure(
            fig,
            out_dir / safe_name(time_scope) /
            "Fig_cross_code_10m_WS_RMSE",
        )


# =============================================================================
# 11. SELECTED COMBINED TIME SERIES
# =============================================================================

def downsample(d: pd.DataFrame) -> pd.DataFrame:
    if len(d) <= MAX_TIME_POINTS:
        return d
    step = int(math.ceil(len(d) / MAX_TIME_POINTS))
    return d.iloc[::step].copy()


def plot_code1_cross_tower_10m_timeseries(
    pred: pd.DataFrame,
    out_dir: Path,
) -> None:
    if pred.empty or TIME_COL not in pred.columns:
        return
    required = {
        "source_site", "target_site", "network", "case",
        "generalization_type", "WS_height_m",
        "OBS_WS", "WRF_WS", "CORR_WS",
    }
    if not required.issubset(pred.columns):
        return

    valid_gtypes = [
        "cross_tower_same_height",
        "cross_tower_other_time",
    ]
    d = pred[
        pred["generalization_type"].isin(valid_gtypes)
        & pd.to_numeric(
            pred["WS_height_m"], errors="coerce"
        ).eq(float(REFERENCE_WS_HEIGHT_M))
        & pred["case"].isin(CODE1_CASES)
    ].copy()

    for keys, g in d.groupby(
        ["generalization_type", "source_site", "target_site", "network"],
        dropna=False,
    ):
        gtype, source, target, network = keys
        cases_present = [
            c for c in CODE1_CASES if g["case"].eq(c).any()
        ]
        if not cases_present:
            continue
        base_case = cases_present[0]
        base = g[g["case"].eq(base_case)][
            [TIME_COL, "OBS_WS", "WRF_WS"]
        ].dropna(subset=[TIME_COL]).copy()
        merged = base
        for case in cases_present:
            gc = g[g["case"].eq(case)][
                [TIME_COL, "CORR_WS"]
            ].rename(columns={"CORR_WS": case})
            merged = merged.merge(gc, on=TIME_COL, how="inner")
        fair_scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
        merged = _fair_filter_timeseries(merged, source, target, fair_scope, REFERENCE_WS_HEIGHT_M)
        merged = downsample(merged.sort_values(TIME_COL))
        if merged.empty:
            continue
        fig, ax = plt.subplots(figsize=(13.0, 4.8))
        _plot_gap_safe(
            ax, merged[TIME_COL], merged["OBS_WS"],
            label="OBS", linewidth=1.4,
        )
        _plot_gap_safe(
            ax, merged[TIME_COL], merged["WRF_WS"],
            label="Raw WRF", linestyle="--", linewidth=1.1,
        )
        overlap_groups = group_identical_time_series_cases(
            merged,
            cases_present,
            {
                "plot_family": "time_series",
                "code": "Code1",
                "source_site": source,
                "target_site": target,
                "network": network,
                "generalization_type": gtype,
                "variable": "WS10",
            },
        )
        for case_group in overlap_groups:
            case = case_group[0]
            style = combined_case_style(case_group)
            _plot_gap_safe(
                ax, merged[TIME_COL], merged[case],
                label=combined_case_label(case_group),
                linewidth=1.05, marker=None,
                linestyle=style.get("linestyle", "-"),
            )
        annotate_exact_overlaps(ax, overlap_groups)
        ax.set_ylabel("WS at 10 m (m s$^{-1}$)")
        ax.set_xlabel("Time")
        ax.set_title(
            f"Code1 {GENERALIZATION_LABELS.get(gtype, gtype)}: "
            f"{source}→{target} — {network}"
        )
        configure_time_axis(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            f"Fig_Code1_timeseries_{source}_to_{target}_{network}_WS10",
        )


def plot_code2_cross_tower_10m_timeseries(
    pred: pd.DataFrame,
    out_dir: Path,
) -> None:
    if pred.empty or TIME_COL not in pred.columns:
        return
    meta = {
        "source_site", "target_site", "network",
        "case", "generalization_type",
    }
    cols = {"OBS_WS10", "WRF_WS10", "CORR_WS10"}
    if not meta.issubset(pred.columns) or not cols.issubset(pred.columns):
        return
    d = pred[
        pred["generalization_type"].isin([
            "cross_tower_same_time",
            "cross_tower_other_time",
        ])
        & pred["case"].isin(CODE2_CASES)
    ].copy()

    for keys, g in d.groupby(
        ["generalization_type", "source_site", "target_site", "network"],
        dropna=False,
    ):
        gtype, source, target, network = keys
        cases_present = [
            c for c in CODE2_CASES if g["case"].eq(c).any()
        ]
        if not cases_present:
            continue
        base_case = cases_present[0]
        base = g[g["case"].eq(base_case)][
            [TIME_COL, "OBS_WS10", "WRF_WS10"]
        ].dropna(subset=[TIME_COL]).copy()
        merged = base
        for case in cases_present:
            gc = g[g["case"].eq(case)][
                [TIME_COL, "CORR_WS10"]
            ].rename(columns={"CORR_WS10": case})
            merged = merged.merge(gc, on=TIME_COL, how="inner")
        fair_scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
        merged = _fair_filter_timeseries(merged, source, target, fair_scope, REFERENCE_WS_HEIGHT_M)
        merged = downsample(merged.sort_values(TIME_COL))
        if merged.empty:
            continue
        fig, ax = plt.subplots(figsize=(13.0, 4.8))
        _plot_gap_safe(
            ax, merged[TIME_COL], merged["OBS_WS10"],
            label="OBS", linewidth=1.4,
        )
        _plot_gap_safe(
            ax, merged[TIME_COL], merged["WRF_WS10"],
            label="Raw WRF", linestyle="--", linewidth=1.1,
        )
        overlap_groups = group_identical_time_series_cases(
            merged,
            cases_present,
            {
                "plot_family": "time_series",
                "code": "Code2",
                "source_site": source,
                "target_site": target,
                "network": network,
                "generalization_type": gtype,
                "variable": "WS10",
            },
        )
        for case_group in overlap_groups:
            case = case_group[0]
            style = combined_case_style(case_group)
            _plot_gap_safe(
                ax, merged[TIME_COL], merged[case],
                label=combined_case_label(case_group),
                linewidth=1.05, marker=None,
                linestyle=style.get("linestyle", "-"),
            )
        annotate_exact_overlaps(ax, overlap_groups)
        ax.set_ylabel("WS at 10 m (m s$^{-1}$)")
        ax.set_xlabel("Time")
        ax.set_title(
            f"Code2 {GENERALIZATION_LABELS.get(gtype, gtype)}: "
            f"{source}→{target} — {network}"
        )
        configure_time_axis(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / safe_name(gtype) /
            f"Fig_Code2_timeseries_{source}_to_{target}_{network}_WS10",
        )



# =============================================================================
# 12. SCI MANUSCRIPT FIGURES
# =============================================================================

def _sci_apply_time_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIME_COL not in out.columns:
        return out
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    if SCI_TIME_START is not None:
        out = out[out[TIME_COL] >= pd.to_datetime(SCI_TIME_START)]
    if SCI_TIME_END is not None:
        out = out[out[TIME_COL] <= pd.to_datetime(SCI_TIME_END)]
    return out


def _sci_style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=GRID_ALPHA)
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def _sci_add_grouped_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    series_data: Dict[str, Sequence[float]],
    series_min: Optional[Dict[str, Sequence[float]]] = None,
    series_max: Optional[Dict[str, Sequence[float]]] = None,
    ylabel: str = "",
    title: str = "",
    zero_line: bool = False,
    decimals: int = 2,
    show_legend: bool = True,
    legend_ncol: Optional[int] = None,
    legend_loc: str = "best",
) -> None:
    names = list(series_data)
    if not names:
        ax.set_axis_off()
        return
    x = np.arange(len(categories), dtype=float)
    total_width = 0.82
    width = total_width / max(len(names), 1)
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2.0) * width

    for i, name in enumerate(names):
        values = np.asarray(series_data[name], dtype=float)
        bars = ax.bar(
            x + offsets[i],
            values,
            width * 0.92,
            label=name,
            hatch=CASE_HATCH.get(name, ""),
            linewidth=0.8,
            edgecolor="black",
        )
        if series_min is not None and series_max is not None:
            lo = np.asarray(series_min.get(name, values), dtype=float)
            hi = np.asarray(series_max.get(name, values), dtype=float)
            lower = np.maximum(values - lo, 0.0)
            upper = np.maximum(hi - values, 0.0)
            if np.isfinite(np.r_[lower, upper]).any():
                ax.errorbar(
                    x + offsets[i],
                    values,
                    yerr=np.vstack([lower, upper]),
                    fmt="none",
                    capsize=2.5,
                    linewidth=0.8,
                )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.annotate(
                f"{value:.{decimals}f}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 3 if value >= 0 else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.0,
                rotation=90 if len(categories) >= 4 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=14, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if zero_line:
        ax.axhline(0.0, linewidth=0.9, linestyle="--")
    if show_legend:
        ncol = legend_ncol if legend_ncol is not None else min(3, max(len(names), 1))
        ax.legend(
            frameon=False,
            fontsize=7.0,
            ncol=ncol,
            loc=legend_loc,
            handlelength=1.8,
            columnspacing=0.9,
            handletextpad=0.45,
        )
    _sci_style_axis(ax)



def _save_single_panel(
    fig: plt.Figure,
    out_path: Path,
) -> None:
    """Save one SCI panel in PNG/PDF/SVG using the common figure writer."""
    save_figure(fig, out_path)


def _individual_axis() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(
        1, 1,
        figsize=SCI_SINGLE_PANEL_FIGSIZE,
        layout="constrained",
    )
    return fig, ax


def _safe_name(value: object) -> str:
    return (
        str(value)
        .replace("→", "_to_")
        .replace("–", "_")
        .replace("—", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _code1_metric_panel_data(
    c1: pd.DataFrame,
    variable: str,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], pd.DataFrame]:
    gtypes = [
        "same_tower_other_height",
        "cross_tower_same_height",
        "cross_tower_other_height",
    ]
    labels = [
        "Same tower,\nother height",
        "Cross tower,\nsame height",
        "Cross tower,\nother height",
    ]
    rows = []
    series_names = ([] if variable == "WD" else ["Raw WRF"]) + CODE1_CASES
    means = {s: [] for s in series_names}
    mins = {s: [] for s in series_names}
    maxs = {s: [] for s in series_names}

    for gtype, label in zip(gtypes, labels):
        d = c1[
            c1["generalization_type"].eq(gtype)
            & c1["variable"].eq(variable)
            & numeric(c1["N"]).fillna(0).gt(0)
        ].copy()

        if variable != "WD":
            raw = numeric(d["Raw_RMSE"]).dropna()
            means["Raw WRF"].append(float(raw.mean()) if not raw.empty else np.nan)
            mins["Raw WRF"].append(float(raw.min()) if not raw.empty else np.nan)
            maxs["Raw WRF"].append(float(raw.max()) if not raw.empty else np.nan)
            rows.append({
                "panel_variable": variable,
                "generalization_type": gtype,
                "generalization_label": label.replace("\n", " "),
                "series": "Raw WRF",
                "mean": means["Raw WRF"][-1],
                "min": mins["Raw WRF"][-1],
                "max": maxs["Raw WRF"][-1],
            })

        for case in CODE1_CASES:
            vals = numeric(d.loc[d["case"].eq(case), "primary_error"]).dropna()
            means[case].append(float(vals.mean()) if not vals.empty else np.nan)
            mins[case].append(float(vals.min()) if not vals.empty else np.nan)
            maxs[case].append(float(vals.max()) if not vals.empty else np.nan)
            rows.append({
                "panel_variable": variable,
                "generalization_type": gtype,
                "generalization_label": label.replace("\n", " "),
                "series": case,
                "mean": means[case][-1],
                "min": mins[case][-1],
                "max": maxs[case][-1],
            })

    return labels, means, mins, maxs, pd.DataFrame(rows)


def _profile_panel(
    ax: plt.Axes,
    profile: pd.DataFrame,
    cases: Sequence[str],
    code: str,
    source: str,
    target: str,
    network: str,
    generalization_type: str,
    title: str,
    metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    g = profile[
        profile["source_site"].astype(str).eq(str(source))
        & profile["target_site"].astype(str).eq(str(target))
        & profile["network"].astype(str).eq(str(network))
        & profile["generalization_type"].astype(str).eq(str(generalization_type))
        & profile["case"].astype(str).isin(list(cases))
    ].copy()
    if g.empty:
        ax.set_axis_off()
        return pd.DataFrame()

    baseline = (
        g.groupby("height_m", as_index=False)
        .agg(OBS_mean=("OBS_mean", "mean"), WRF_mean=("WRF_mean", "mean"))
        .sort_values("height_m")
    )
    ax.plot(
        baseline["OBS_mean"], baseline["height_m"],
        marker="o", linewidth=SCI_LINEWIDTH_OBS, label="OBS",
    )
    ax.plot(
        baseline["WRF_mean"], baseline["height_m"],
        marker="s", linestyle="--", linewidth=SCI_LINEWIDTH_WRF, label="Raw WRF",
    )

    overlap_groups = group_identical_profile_cases(
        g, cases, "height_m", "CORR_mean",
        {
            "plot_family": "SCI_mean_profile",
            "code": code,
            "source_site": source,
            "target_site": target,
            "network": network,
            "generalization_type": generalization_type,
            "variable": "Mean_WS",
        },
    )

    rmse_map = exact_all_profile_rmse_map(
        metrics, code, source, target, network, generalization_type, cases
    )
    best_case = min(rmse_map, key=rmse_map.get) if rmse_map else None

    for case_group in overlap_groups:
        case = case_group[0]
        gc = g[g["case"].eq(case)].sort_values("height_m")
        style = combined_case_style(case_group)
        label = (
            case_group_label_with_rmse(case_group, rmse_map, best_case)
            if code == "Code2"
            else combined_case_label(case_group)
        )
        ax.plot(
            gc["CORR_mean"], gc["height_m"],
            linewidth=SCI_LINEWIDTH_CASE,
            markersize=5.0,
            label=label,
            **style,
        )

    annotate_exact_overlaps(ax, overlap_groups)
    ax.set_xlabel("Mean WS (m s$^{-1}$)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    _sci_style_axis(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=1)

    out = g.copy()
    out["panel_source"] = source
    out["panel_target"] = target
    out["panel_network"] = network
    return out


def _merge_code1_timeseries(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
) -> pd.DataFrame:
    if pred.empty:
        return _empty_wind_distribution()
    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq("cross_tower_same_height")
        & pd.to_numeric(pred["WS_height_m"], errors="coerce").eq(float(height_m))
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty:
        return pd.DataFrame()

    cases_present = [c for c in cases if d["case"].astype(str).eq(c).any()]
    base_case = cases_present[0]
    merged = d[d["case"].astype(str).eq(base_case)][
        [TIME_COL, "OBS_WS", "WRF_WS"]
    ].copy()
    for case in cases_present:
        gc = d[d["case"].astype(str).eq(case)][
            [TIME_COL, "CORR_WS"]
        ].rename(columns={"CORR_WS": case})
        merged = merged.merge(gc, on=TIME_COL, how="inner")
    merged = _fair_filter_timeseries(merged, source, target, "same_time", height_m)
    return _sci_apply_time_window(merged)


def _merge_code2_timeseries(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    obs_col = f"OBS_WS{height_m}"
    wrf_col = f"WRF_WS{height_m}"
    corr_col = f"CORR_WS{height_m}"
    required = {obs_col, wrf_col, corr_col}
    if not required.issubset(pred.columns):
        return pd.DataFrame()

    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq("cross_tower_same_time")
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty:
        return pd.DataFrame()

    cases_present = [c for c in cases if d["case"].astype(str).eq(c).any()]
    base_case = cases_present[0]
    merged = d[d["case"].astype(str).eq(base_case)][
        [TIME_COL, obs_col, wrf_col]
    ].rename(columns={obs_col: "OBS", wrf_col: "Raw WRF"})
    for case in cases_present:
        gc = d[d["case"].astype(str).eq(case)][
            [TIME_COL, corr_col]
        ].rename(columns={corr_col: case})
        merged = merged.merge(gc, on=TIME_COL, how="inner")
    merged = _fair_filter_timeseries(merged, source, target, "same_time", height_m)
    return _sci_apply_time_window(merged)


def _timeseries_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    cases: Sequence[str],
    code: str,
    source: str,
    target: str,
    network: str,
    title: str,
) -> None:
    if merged.empty:
        ax.set_axis_off()
        return

    if code == "Code1":
        obs_col, raw_col = "OBS_WS", "WRF_WS"
    else:
        obs_col, raw_col = "OBS", "Raw WRF"

    _plot_gap_safe(
        ax, merged[TIME_COL], merged[obs_col],
        label="OBS", linewidth=SCI_LINEWIDTH_OBS,
    )
    _plot_gap_safe(
        ax, merged[TIME_COL], merged[raw_col],
        label="Raw WRF", linestyle="--", linewidth=SCI_LINEWIDTH_WRF,
    )

    cases_present = [c for c in cases if c in merged.columns]
    overlap_groups = group_identical_time_series_cases(
        merged,
        cases_present,
        {
            "plot_family": "SCI_time_series",
            "code": code,
            "source_site": source,
            "target_site": target,
            "network": network,
            "generalization_type": (
                "cross_tower_same_height" if code == "Code1"
                else "cross_tower_same_time"
            ),
            "variable": f"WS{SCI_TS_HEIGHT_M}",
        },
    )
    for case_group in overlap_groups:
        case = case_group[0]
        style = combined_case_style(case_group)
        _plot_gap_safe(
            ax, merged[TIME_COL], merged[case],
            label=combined_case_label(case_group),
            linewidth=SCI_LINEWIDTH_CASE,
            linestyle=style.get("linestyle", "-"),
        )
    annotate_exact_overlaps(ax, overlap_groups)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"WS at {SCI_TS_HEIGHT_M} m (m s$^{{-1}}$)")
    ax.set_title(title)
    configure_time_axis(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=3)



def plot_sci_individual_code1_panels(
    c1: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    panel_root: Path,
) -> None:
    out_dir = ensure_dir(panel_root / "G1_Code1")

    metric_specs = [
        ("WS", "WS RMSE (m s$^{-1}$)", "Code1 wind-speed generalization"),
        ("WD", "Circular WD MAE (°)", "Code1 wind-direction generalization"),
        ("TKE", "TKE RMSE (m$^2$ s$^{-2}$)", "Code1 TKE generalization"),
    ]
    for panel_letter, (variable, ylabel, title) in zip("abc", metric_specs):
        categories, means, mins, maxs, _ = _code1_metric_panel_data(c1, variable)
        fig, ax = _individual_axis()
        _sci_add_grouped_bars(
            ax,
            categories,
            means,
            mins,
            maxs,
            ylabel=ylabel,
            title=title,
            decimals=1 if variable == "WD" else 2,
            show_legend=True,
            legend_ncol=min(3, len(means)),
            legend_loc="best",
        )
        _save_single_panel(
            fig,
            out_dir / f"Fig_G1{panel_letter}_Code1_{variable}_metrics",
        )

    panel_index = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _profile_panel(
                ax,
                c1_profiles,
                CODE1_CASES,
                "Code1",
                source,
                target,
                network,
                "cross_tower_other_height",
                f"Code1 {source}→{target}, {network}\n"
                "Cross-tower, other-height pseudo-profile",
                metrics=c1,
            )
            _save_single_panel(
                fig,
                out_dir
                / f"Fig_G1_profile_{panel_index:02d}_{source}_to_{target}_{network}",
            )
            panel_index += 1

        ts = _merge_code1_timeseries(
            c1_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            CODE1_CASES,
            SCI_TS_HEIGHT_M,
        )
        fig, ax = _individual_axis()
        _timeseries_panel(
            ax,
            ts,
            CODE1_CASES,
            "Code1",
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            f"Code1 {SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            f"Cross-tower {SCI_TS_HEIGHT_M} m time series",
        )
        _save_single_panel(
            fig,
            out_dir / f"Fig_G1_timeseries_{network}_{SCI_TS_HEIGHT_M}m",
        )


def plot_sci_individual_code2_panels(
    c2: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    panel_root: Path,
) -> None:
    out_dir = ensure_dir(panel_root / "G2_Code2")

    categories, ws_series, _ = _code2_exact_metric_data(c2, "WS")
    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax,
        categories,
        ws_series,
        ylabel="Pooled WS RMSE (m s$^{-1}$)",
        title="Code2 exact cross-tower WS performance",
        decimals=2,
        show_legend=True,
        legend_ncol=3,
        legend_loc="best",
    )
    _save_single_panel(fig, out_dir / "Fig_G2a_Code2_WS_ALL_PROFILE_RMSE")

    categories_tke, tke_series, _ = _code2_exact_metric_data(c2, "TKE")
    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax,
        categories_tke,
        tke_series,
        ylabel="TKE RMSE (m$^2$ s$^{-2}$)",
        title="Code2 exact cross-tower TKE performance",
        decimals=2,
        show_legend=True,
        legend_ncol=3,
        legend_loc="best",
    )
    _save_single_panel(fig, out_dir / "Fig_G2b_Code2_TKE_RMSE")

    _, curves = _code2_heightwise_mean_range(c2)
    fig, ax = _individual_axis()
    for name, gc in curves.items():
        if gc.empty:
            continue
        style = (
            {"linestyle": "--", "marker": "s"}
            if name == "Raw WRF"
            else combined_case_style([name])
        )
        ax.plot(
            gc["height_m"],
            gc["mean"],
            linewidth=1.35,
            markersize=5.2,
            label=name,
            **style,
        )
        ax.fill_between(
            gc["height_m"].to_numpy(dtype=float),
            gc["min"].to_numpy(dtype=float),
            gc["max"].to_numpy(dtype=float),
            alpha=0.10,
        )
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("WS RMSE (m s$^{-1}$)")
    ax.set_title("Code2 height-wise WS robustness")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    _sci_style_axis(ax)
    _save_single_panel(fig, out_dir / "Fig_G2c_Code2_heightwise_WS_RMSE")

    panel_index = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _profile_panel(
                ax,
                c2_profiles,
                CODE2_CASES,
                "Code2",
                source,
                target,
                network,
                "cross_tower_same_time",
                f"Code2 {source}→{target}, {network}\n"
                "Full multi-height profile",
                metrics=c2,
            )
            _save_single_panel(
                fig,
                out_dir
                / f"Fig_G2_profile_{panel_index:02d}_{source}_to_{target}_{network}",
            )
            panel_index += 1

        ts = _merge_code2_timeseries(
            c2_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            CODE2_CASES,
            SCI_TS_HEIGHT_M,
        )
        fig, ax = _individual_axis()
        _timeseries_panel(
            ax,
            ts,
            CODE2_CASES,
            "Code2",
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            f"Code2 {SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            f"Cross-tower {SCI_TS_HEIGHT_M} m time series",
        )
        _save_single_panel(
            fig,
            out_dir / f"Fig_G2_timeseries_{network}_{SCI_TS_HEIGHT_M}m",
        )


def _plot_cross_code_time_axis(
    ax: plt.Axes,
    ts: pd.DataFrame,
    network: str,
    title: str,
) -> None:
    if ts.empty:
        ax.set_axis_off()
        return
    obs = np.nanmean(
        np.vstack(
            [
                numeric(ts["OBS_Code1"]),
                numeric(ts["OBS_Code2"]),
            ]
        ),
        axis=0,
    )
    wrf = np.nanmean(
        np.vstack(
            [
                numeric(ts["Raw_WRF_Code1"]),
                numeric(ts["Raw_WRF_Code2"]),
            ]
        ),
        axis=0,
    )
    _plot_gap_safe(ax, ts[TIME_COL], pd.Series(obs, index=ts.index), label="OBS", linewidth=SCI_LINEWIDTH_OBS)
    _plot_gap_safe(
        ax, ts[TIME_COL], pd.Series(wrf, index=ts.index),
        label="Raw WRF",
        linestyle="--",
        linewidth=SCI_LINEWIDTH_WRF,
    )
    _plot_gap_safe(
        ax, ts[TIME_COL], ts[f"Code1_{SCI_CODE1_BEST_WS_CASE}"],
        label=f"Code1 {SCI_CODE1_BEST_WS_CASE}",
        linewidth=SCI_LINEWIDTH_CASE,
    )
    _plot_gap_safe(
        ax, ts[TIME_COL], ts[f"Code2_{SCI_CODE2_BEST_WS_CASE}"],
        label=f"Code2 {SCI_CODE2_BEST_WS_CASE}",
        linestyle=":",
        linewidth=SCI_LINEWIDTH_CASE,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(f"WS at {SCI_TS_HEIGHT_M} m (m s$^{{-1}}$)")
    ax.set_title(title)
    configure_time_axis(ax)
    ax.legend(frameon=False, fontsize=7.4, ncol=2)


def plot_sci_individual_cross_code_panels(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    cross_100: pd.DataFrame,
    panel_root: Path,
) -> None:
    out_dir = ensure_dir(panel_root / "G3_Code1_vs_Code2")

    categories, means, mins, maxs, _ = _cross_code_role_summary(
        cross_100, "RMSE"
    )
    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax,
        categories,
        means,
        mins,
        maxs,
        ylabel="WS RMSE at 10 m (m s$^{-1}$)",
        title="Code1 versus Code2: common-height error",
        decimals=2,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )
    _save_single_panel(fig, out_dir / "Fig_G3a_cross_code_10m_RMSE")

    categories_s, means_s, mins_s, maxs_s, _ = _cross_code_role_summary(
        cross_100, "Skill"
    )
    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax,
        categories_s,
        means_s,
        mins_s,
        maxs_s,
        ylabel="RMSE skill at 10 m (%)",
        title="Code1 versus Code2: common-height skill",
        decimals=1,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )
    _save_single_panel(fig, out_dir / "Fig_G3b_cross_code_10m_skill")

    delta_series = {}
    combo_order = [
        ("NO1_1159→NO2_1107", "CNN_LSTM"),
        ("NO1_1159→NO2_1107", "TCN"),
        ("NO2_1107→NO1_1159", "CNN_LSTM"),
        ("NO2_1107→NO1_1159", "TCN"),
    ]
    combo_labels = [
        "A1→A2\nCNN–LSTM",
        "A1→A2\nTCN",
        "A2→A1\nCNN–LSTM",
        "A2→A1\nTCN",
    ]
    for role in ROLE_ORDER:
        values = []
        for direction, network in combo_order:
            hit = cross_100[
                cross_100["direction"].eq(direction)
                & cross_100["network"].eq(network)
                & cross_100["role"].eq(role)
            ]
            values.append(
                float(hit.iloc[0]["Code2_minus_Code1_RMSE"])
                if not hit.empty
                else np.nan
            )
        delta_series[role] = values

    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax,
        combo_labels,
        delta_series,
        ylabel="Code2 − Code1 RMSE (m s$^{-1}$)",
        title="Exact Code2 advantage\nNegative values favor Code2",
        zero_line=True,
        decimals=2,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )
    _save_single_panel(fig, out_dir / "Fig_G3c_exact_Code2_minus_Code1")

    panel_index = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _cross_code_profile_panel(
                ax,
                c1_profiles,
                c2_profiles,
                source,
                target,
                network,
                f"{source}→{target}, {network}\n"
                "Code1 C8 versus Code2 C8d profiles",
            )
            _save_single_panel(
                fig,
                out_dir
                / f"Fig_G3_profile_{panel_index:02d}_{source}_to_{target}_{network}",
            )
            panel_index += 1

        ts = _cross_code_timeseries(
            c1_pred,
            c2_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
        )
        fig, ax = _individual_axis()
        _plot_cross_code_time_axis(
            ax,
            ts,
            network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "Direct Code1–Code2 time-series comparison",
        )
        _save_single_panel(
            fig,
            out_dir / f"Fig_G3_timeseries_{network}_{SCI_TS_HEIGHT_M}m",
        )



def plot_sci_figure_code1(
    c1: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    out_dir: Path,
    data_dir: Path,
) -> None:
    composite_dir = ensure_dir(out_dir / "00_COMPOSITE")
    panel_root = ensure_dir(out_dir / "01_INDIVIDUAL_PANELS")

    fig, axes = plt.subplots(3, 3, figsize=SCI_FIGSIZE, layout="constrained")
    axes = np.asarray(axes)

    all_metric_data = []
    metric_specs = [
        ("WS", "WS RMSE (m s$^{-1}$)", "Wind speed"),
        ("WD", "Circular WD MAE (°)", "Wind direction"),
        ("TKE", "TKE RMSE (m$^2$ s$^{-2}$)", "TKE"),
    ]
    for ax, (variable, ylabel, title) in zip(axes[0], metric_specs):
        categories, means, mins, maxs, table = _code1_metric_panel_data(
            c1, variable
        )
        _sci_add_grouped_bars(
            ax,
            categories,
            means,
            mins,
            maxs,
            ylabel=ylabel,
            title=title,
            decimals=1 if variable == "WD" else 2,
            show_legend=True,
            legend_ncol=min(3, len(means)),
            legend_loc="best",
        )
        all_metric_data.append(table)

    profile_tables = []
    timeseries_tables = []

    for row, network in enumerate(SCI_NETWORKS, start=1):
        for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
            table = _profile_panel(
                axes[row, col],
                c1_profiles,
                CODE1_CASES,
                "Code1",
                source,
                target,
                network,
                "cross_tower_other_height",
                f"{source}→{target}, {network}\n"
                "Cross-tower, other-height pseudo-profile",
                metrics=c1,
            )
            profile_tables.append(table)

        ts = _merge_code1_timeseries(
            c1_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            CODE1_CASES,
            SCI_TS_HEIGHT_M,
        )
        _timeseries_panel(
            axes[row, 2],
            ts,
            CODE1_CASES,
            "Code1",
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            f"Cross-tower {SCI_TS_HEIGHT_M} m time series",
        )
        if not ts.empty:
            timeseries_tables.append(ts.assign(network=network))

    add_panel_labels(axes.reshape(-1))
    fig.suptitle(
        "Code1 single-height model generalization: metrics, "
        "CNN–LSTM/TCN pseudo-profiles, and time series"
    )
    save_figure(fig, composite_dir / "Fig_G1_Code1_generalization_COMPOSITE")

    pd.concat(all_metric_data, ignore_index=True).to_csv(
        data_dir / "Fig_G1a_c_Code1_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_profiles = [x for x in profile_tables if not x.empty]
    if valid_profiles:
        pd.concat(valid_profiles, ignore_index=True).to_csv(
            data_dir / "Fig_G1_profiles_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if timeseries_tables:
        pd.concat(timeseries_tables, ignore_index=True).to_csv(
            data_dir / "Fig_G1_timeseries_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_sci_individual_code1_panels(
        c1,
        c1_pred,
        c1_profiles,
        panel_root,
    )

def _code2_exact_metric_data(
    c2: pd.DataFrame,
    variable: str,
) -> Tuple[List[str], Dict[str, List[float]], pd.DataFrame]:
    d = c2[
        c2["generalization_type"].eq("cross_tower_same_time")
        & c2["variable"].eq(variable)
        & c2["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & c2["case"].isin(CODE2_CASES)
        & numeric(c2["N"]).fillna(0).gt(0)
    ].copy()

    combos = [
        ("NO1_1159", "NO2_1107", "CNN_LSTM", "A1→A2\nCNN–LSTM"),
        ("NO1_1159", "NO2_1107", "TCN", "A1→A2\nTCN"),
        ("NO2_1107", "NO1_1159", "CNN_LSTM", "A2→A1\nCNN–LSTM"),
        ("NO2_1107", "NO1_1159", "TCN", "A2→A1\nTCN"),
    ]
    categories = [x[3] for x in combos]
    series = {"Raw WRF": []}
    series.update({case: [] for case in CODE2_CASES})
    rows = []

    for source, target, network, label in combos:
        g = d[
            d["source_site"].eq(source)
            & d["target_site"].eq(target)
            & d["network"].eq(network)
        ].copy()
        raw = numeric(g["Raw_RMSE"]).dropna()
        raw_value = float(raw.mean()) if not raw.empty else np.nan
        series["Raw WRF"].append(raw_value)
        rows.append({
            "variable": variable, "source_site": source, "target_site": target,
            "network": network, "category": label.replace("\n", " "),
            "series": "Raw WRF", "value": raw_value,
        })
        for case in CODE2_CASES:
            hit = g[g["case"].eq(case)]
            value = float(hit.iloc[0]["primary_error"]) if not hit.empty else np.nan
            series[case].append(value)
            rows.append({
                "variable": variable, "source_site": source, "target_site": target,
                "network": network, "category": label.replace("\n", " "),
                "series": case, "value": value,
            })
    return categories, series, pd.DataFrame(rows)


def _code2_heightwise_mean_range(
    c2: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    d = c2[
        c2["generalization_type"].eq("cross_tower_same_time")
        & c2["variable"].eq("WS")
        & c2["height_num"].notna()
        & c2["case"].isin(CODE2_CASES)
        & numeric(c2["N"]).fillna(0).gt(0)
    ].copy()

    rows = []
    curves: Dict[str, pd.DataFrame] = {}
    raw = (
        d.groupby("height_num", as_index=False)["Raw_RMSE"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"height_num": "height_m"})
    )
    curves["Raw WRF"] = raw
    for _, r in raw.iterrows():
        rows.append({
            "series": "Raw WRF", "height_m": r["height_m"],
            "mean": r["mean"], "min": r["min"], "max": r["max"],
        })

    for case in CODE2_CASES:
        gc = (
            d[d["case"].eq(case)]
            .groupby("height_num", as_index=False)["RMSE"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .rename(columns={"height_num": "height_m"})
        )
        curves[case] = gc
        for _, r in gc.iterrows():
            rows.append({
                "series": case, "height_m": r["height_m"],
                "mean": r["mean"], "min": r["min"], "max": r["max"],
            })
    return pd.DataFrame(rows), curves



def plot_sci_figure_code2(
    c2: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    out_dir: Path,
    data_dir: Path,
) -> None:
    composite_dir = ensure_dir(out_dir / "00_COMPOSITE")
    panel_root = ensure_dir(out_dir / "01_INDIVIDUAL_PANELS")

    fig, axes = plt.subplots(3, 3, figsize=SCI_FIGSIZE, layout="constrained")
    axes = np.asarray(axes)

    categories, ws_series, ws_table = _code2_exact_metric_data(c2, "WS")
    _sci_add_grouped_bars(
        axes[0, 0],
        categories,
        ws_series,
        ylabel="Pooled WS RMSE (m s$^{-1}$)",
        title="Exact cross-tower WS performance",
        decimals=2,
        show_legend=True,
        legend_ncol=3,
        legend_loc="best",
    )

    categories_tke, tke_series, tke_table = _code2_exact_metric_data(
        c2, "TKE"
    )
    _sci_add_grouped_bars(
        axes[0, 1],
        categories_tke,
        tke_series,
        ylabel="TKE RMSE (m$^2$ s$^{-2}$)",
        title="Exact cross-tower TKE performance",
        decimals=2,
        show_legend=True,
        legend_ncol=3,
        legend_loc="best",
    )

    height_table, curves = _code2_heightwise_mean_range(c2)
    for name, gc in curves.items():
        if gc.empty:
            continue
        style = (
            {"linestyle": "--", "marker": "s"}
            if name == "Raw WRF"
            else combined_case_style([name])
        )
        axes[0, 2].plot(
            gc["height_m"],
            gc["mean"],
            linewidth=1.3,
            markersize=5.0,
            label=name,
            **style,
        )
        axes[0, 2].fill_between(
            gc["height_m"].to_numpy(dtype=float),
            gc["min"].to_numpy(dtype=float),
            gc["max"].to_numpy(dtype=float),
            alpha=0.10,
        )
    axes[0, 2].set_xlabel("Height (m)")
    axes[0, 2].set_ylabel("WS RMSE (m s$^{-1}$)")
    axes[0, 2].set_title(
        "Height-wise robustness\n"
        "Mean and range across all transfers/networks"
    )
    axes[0, 2].legend(frameon=False, fontsize=7.2, ncol=2)
    _sci_style_axis(axes[0, 2])

    profile_tables = []
    timeseries_tables = []

    for row, network in enumerate(SCI_NETWORKS, start=1):
        for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
            table = _profile_panel(
                axes[row, col],
                c2_profiles,
                CODE2_CASES,
                "Code2",
                source,
                target,
                network,
                "cross_tower_same_time",
                f"{source}→{target}, {network}\n"
                "Full multi-height profile",
                metrics=c2,
            )
            profile_tables.append(table)

        ts = _merge_code2_timeseries(
            c2_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            CODE2_CASES,
            SCI_TS_HEIGHT_M,
        )
        _timeseries_panel(
            axes[row, 2],
            ts,
            CODE2_CASES,
            "Code2",
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            f"Cross-tower {SCI_TS_HEIGHT_M} m time series",
        )
        if not ts.empty:
            timeseries_tables.append(ts.assign(network=network))

    add_panel_labels(axes.reshape(-1))
    fig.suptitle(
        "Code2 multi-height model generalization: exact metrics, "
        "CNN–LSTM/TCN profiles, and time series"
    )
    save_figure(fig, composite_dir / "Fig_G2_Code2_generalization_COMPOSITE")

    pd.concat([ws_table, tke_table], ignore_index=True).to_csv(
        data_dir / "Fig_G2a_b_Code2_exact_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    height_table.to_csv(
        data_dir / "Fig_G2c_Code2_heightwise_WS.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_profiles = [x for x in profile_tables if not x.empty]
    if valid_profiles:
        pd.concat(valid_profiles, ignore_index=True).to_csv(
            data_dir / "Fig_G2_profiles_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if timeseries_tables:
        pd.concat(timeseries_tables, ignore_index=True).to_csv(
            data_dir / "Fig_G2_timeseries_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_sci_individual_code2_panels(
        c2,
        c2_pred,
        c2_profiles,
        panel_root,
    )

def _cross_code_role_summary(
    cross_100: pd.DataFrame,
    value_col: str,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], pd.DataFrame]:
    categories = ROLE_ORDER
    means = {"Code1": [], "Code2": []}
    mins = {"Code1": [], "Code2": []}
    maxs = {"Code1": [], "Code2": []}
    rows = []
    col_map = {
        "RMSE": ("Code1_RMSE", "Code2_RMSE"),
        "Skill": ("Code1_skill_pct", "Code2_skill_pct"),
    }
    c1_col, c2_col = col_map[value_col]
    for role in categories:
        g = cross_100[cross_100["role"].eq(role)]
        for code, col in [("Code1", c1_col), ("Code2", c2_col)]:
            vals = numeric(g[col]).dropna()
            means[code].append(float(vals.mean()) if not vals.empty else np.nan)
            mins[code].append(float(vals.min()) if not vals.empty else np.nan)
            maxs[code].append(float(vals.max()) if not vals.empty else np.nan)
            rows.append({
                "metric": value_col, "role": role, "code": code,
                "mean": means[code][-1], "min": mins[code][-1], "max": maxs[code][-1],
            })
    return categories, means, mins, maxs, pd.DataFrame(rows)


def _code1_full_cross_tower_pseudo_profile(
    c1_profiles: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    case: str,
) -> pd.DataFrame:
    d = c1_profiles[
        c1_profiles["source_site"].eq(source)
        & c1_profiles["target_site"].eq(target)
        & c1_profiles["network"].eq(network)
        & c1_profiles["case"].eq(case)
        & c1_profiles["generalization_type"].isin(
            ["cross_tower_same_height", "cross_tower_other_height"]
        )
    ].copy()
    if d.empty:
        return d
    d = (
        d.groupby("height_m", as_index=False)
        .agg(
            OBS_mean=("OBS_mean", "mean"),
            WRF_mean=("WRF_mean", "mean"),
            CORR_mean=("CORR_mean", "mean"),
            N=("N", "max"),
        )
        .sort_values("height_m")
    )
    return d


def _code2_single_profile(
    c2_profiles: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    case: str,
) -> pd.DataFrame:
    return c2_profiles[
        c2_profiles["source_site"].eq(source)
        & c2_profiles["target_site"].eq(target)
        & c2_profiles["network"].eq(network)
        & c2_profiles["case"].eq(case)
        & c2_profiles["generalization_type"].eq("cross_tower_same_time")
    ].sort_values("height_m").copy()


def _cross_code_profile_panel(
    ax: plt.Axes,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    title: str,
) -> pd.DataFrame:
    p1 = _code1_full_cross_tower_pseudo_profile(
        c1_profiles, source, target, network, SCI_CODE1_BEST_WS_CASE
    )
    p2 = _code2_single_profile(
        c2_profiles, source, target, network, SCI_CODE2_BEST_WS_CASE
    )
    if p1.empty or p2.empty:
        ax.set_axis_off()
        return pd.DataFrame()

    # Direct Code1-Code2 profile comparison uses only shared physical heights.
    # Fair-time profile means at each retained height are based on the same timestamps.
    common_heights = sorted(set(pd.to_numeric(p1["height_m"], errors="coerce").dropna())
                            & set(pd.to_numeric(p2["height_m"], errors="coerce").dropna()))
    p1 = p1[pd.to_numeric(p1["height_m"], errors="coerce").isin(common_heights)].copy()
    p2 = p2[pd.to_numeric(p2["height_m"], errors="coerce").isin(common_heights)].copy()
    if not common_heights:
        ax.set_axis_off()
        return pd.DataFrame()

    baseline = p2[["height_m", "OBS_mean", "WRF_mean"]].copy()
    ax.plot(
        baseline["OBS_mean"], baseline["height_m"],
        marker="o", linewidth=SCI_LINEWIDTH_OBS, label="OBS",
    )
    ax.plot(
        baseline["WRF_mean"], baseline["height_m"],
        marker="s", linestyle="--", linewidth=SCI_LINEWIDTH_WRF, label="Raw WRF",
    )
    ax.plot(
        p1["CORR_mean"], p1["height_m"],
        marker="^", linewidth=SCI_LINEWIDTH_CASE,
        label=f"Code1 {SCI_CODE1_BEST_WS_CASE} pseudo-profile",
    )
    ax.plot(
        p2["CORR_mean"], p2["height_m"],
        marker="D", linestyle=":", linewidth=SCI_LINEWIDTH_CASE,
        label=f"Code2 {SCI_CODE2_BEST_WS_CASE} full profile",
    )
    ax.set_xlabel("Mean WS (m s$^{-1}$)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.2)
    _sci_style_axis(ax)

    out1 = p1.assign(code="Code1", case=SCI_CODE1_BEST_WS_CASE)
    out2 = p2.assign(code="Code2", case=SCI_CODE2_BEST_WS_CASE)
    return pd.concat([out1, out2], ignore_index=True)


def _cross_code_timeseries(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
) -> pd.DataFrame:
    t1 = _merge_code1_timeseries(
        c1_pred, source, target, network, [SCI_CODE1_BEST_WS_CASE], SCI_TS_HEIGHT_M
    )
    t2 = _merge_code2_timeseries(
        c2_pred, source, target, network, [SCI_CODE2_BEST_WS_CASE], SCI_TS_HEIGHT_M
    )
    if t1.empty or t2.empty:
        return pd.DataFrame()

    t1 = t1.rename(columns={
        "OBS_WS": "OBS_Code1",
        "WRF_WS": "Raw_WRF_Code1",
        SCI_CODE1_BEST_WS_CASE: f"Code1_{SCI_CODE1_BEST_WS_CASE}",
    })
    t2 = t2.rename(columns={
        "OBS": "OBS_Code2",
        "Raw WRF": "Raw_WRF_Code2",
        SCI_CODE2_BEST_WS_CASE: f"Code2_{SCI_CODE2_BEST_WS_CASE}",
    })
    keep1 = [TIME_COL, "OBS_Code1", "Raw_WRF_Code1", f"Code1_{SCI_CODE1_BEST_WS_CASE}"]
    keep2 = [TIME_COL, "OBS_Code2", "Raw_WRF_Code2", f"Code2_{SCI_CODE2_BEST_WS_CASE}"]
    return t1[keep1].merge(t2[keep2], on=TIME_COL, how="inner")



def plot_sci_figure_cross_code(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    cross_100: pd.DataFrame,
    out_dir: Path,
    data_dir: Path,
) -> None:
    composite_dir = ensure_dir(out_dir / "00_COMPOSITE")
    panel_root = ensure_dir(out_dir / "01_INDIVIDUAL_PANELS")

    fig, axes = plt.subplots(3, 3, figsize=SCI_FIGSIZE, layout="constrained")
    axes = np.asarray(axes)

    categories, means, mins, maxs, rmse_table = _cross_code_role_summary(
        cross_100, "RMSE"
    )
    _sci_add_grouped_bars(
        axes[0, 0],
        categories,
        means,
        mins,
        maxs,
        ylabel="WS RMSE at 10 m (m s$^{-1}$)",
        title="Common-height error",
        decimals=2,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )

    categories_s, means_s, mins_s, maxs_s, skill_table = (
        _cross_code_role_summary(cross_100, "Skill")
    )
    _sci_add_grouped_bars(
        axes[0, 1],
        categories_s,
        means_s,
        mins_s,
        maxs_s,
        ylabel="RMSE skill at 10 m (%)",
        title="Common-height skill",
        decimals=1,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )

    delta_series = {}
    delta_rows = []
    combo_order = [
        ("NO1_1159→NO2_1107", "CNN_LSTM"),
        ("NO1_1159→NO2_1107", "TCN"),
        ("NO2_1107→NO1_1159", "CNN_LSTM"),
        ("NO2_1107→NO1_1159", "TCN"),
    ]
    combo_labels = [
        "A1→A2\nCNN–LSTM",
        "A1→A2\nTCN",
        "A2→A1\nCNN–LSTM",
        "A2→A1\nTCN",
    ]
    for role in ROLE_ORDER:
        vals = []
        for (direction, network), label in zip(combo_order, combo_labels):
            hit = cross_100[
                cross_100["direction"].eq(direction)
                & cross_100["network"].eq(network)
                & cross_100["role"].eq(role)
            ]
            value = (
                float(hit.iloc[0]["Code2_minus_Code1_RMSE"])
                if not hit.empty
                else np.nan
            )
            vals.append(value)
            delta_rows.append(
                {
                    "direction": direction,
                    "network": network,
                    "category": label.replace("\n", " "),
                    "role": role,
                    "Code2_minus_Code1_RMSE": value,
                }
            )
        delta_series[role] = vals

    _sci_add_grouped_bars(
        axes[0, 2],
        combo_labels,
        delta_series,
        ylabel="Code2 − Code1 RMSE (m s$^{-1}$)",
        title="Exact Code2 advantage\nNegative values favor Code2",
        zero_line=True,
        decimals=2,
        show_legend=True,
        legend_ncol=2,
        legend_loc="best",
    )

    profile_tables = []
    timeseries_tables = []

    for row, network in enumerate(SCI_NETWORKS, start=1):
        for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
            table = _cross_code_profile_panel(
                axes[row, col],
                c1_profiles,
                c2_profiles,
                source,
                target,
                network,
                f"{source}→{target}, {network}\n"
                "Code1 C8 versus Code2 C8d profiles",
            )
            profile_tables.append(table)

        ts = _cross_code_timeseries(
            c1_pred,
            c2_pred,
            SCI_TS_SOURCE_SITE,
            SCI_TS_TARGET_SITE,
            network,
        )
        _plot_cross_code_time_axis(
            axes[row, 2],
            ts,
            network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "Direct Code1–Code2 time-series comparison",
        )
        if not ts.empty:
            timeseries_tables.append(ts.assign(network=network))

    add_panel_labels(axes.reshape(-1))
    fig.suptitle(
        "Direct comparison of Code1 single-height and Code2 multi-height "
        "generalization for CNN–LSTM and TCN"
    )
    save_figure(
        fig,
        composite_dir / "Fig_G3_Code1_vs_Code2_generalization_COMPOSITE",
    )

    pd.concat([rmse_table, skill_table], ignore_index=True).to_csv(
        data_dir / "Fig_G3a_b_cross_code_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(delta_rows).to_csv(
        data_dir / "Fig_G3c_cross_code_exact_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_profiles = [x for x in profile_tables if not x.empty]
    if valid_profiles:
        pd.concat(valid_profiles, ignore_index=True).to_csv(
            data_dir / "Fig_G3_profiles_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if timeseries_tables:
        pd.concat(timeseries_tables, ignore_index=True).to_csv(
            data_dir / "Fig_G3_timeseries_both_networks.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_sci_individual_cross_code_panels(
        c1_pred,
        c2_pred,
        c1_profiles,
        c2_profiles,
        cross_100,
        panel_root,
    )


# =============================================================================
# SCI CANDIDATE FIGURES: ALL VALID TARGET TIMES
# =============================================================================

def _code1_metric_panel_data_for_gtype(
    c1: pd.DataFrame,
    variable: str,
    gtype: str,
) -> Tuple[
    List[str],
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    pd.DataFrame,
]:
    label = "All valid\ntarget times"
    d = c1[
        c1["generalization_type"].eq(gtype)
        & c1["variable"].eq(variable)
        & numeric(c1["N"]).fillna(0).gt(0)
    ].copy()
    series_names = ([] if variable == "WD" else ["Raw WRF"]) + CODE1_CASES
    means = {s: [] for s in series_names}
    mins = {s: [] for s in series_names}
    maxs = {s: [] for s in series_names}
    rows = []

    if variable != "WD":
        raw = numeric(d["Raw_RMSE"]).dropna()
        means["Raw WRF"].append(float(raw.mean()) if not raw.empty else np.nan)
        mins["Raw WRF"].append(float(raw.min()) if not raw.empty else np.nan)
        maxs["Raw WRF"].append(float(raw.max()) if not raw.empty else np.nan)
        rows.append({
            "panel_variable": variable,
            "generalization_type": gtype,
            "generalization_label": label.replace("\n", " "),
            "series": "Raw WRF",
            "mean": means["Raw WRF"][-1],
            "min": mins["Raw WRF"][-1],
            "max": maxs["Raw WRF"][-1],
        })

    for case in CODE1_CASES:
        vals = numeric(
            d.loc[d["case"].eq(case), "primary_error"]
        ).dropna()
        means[case].append(float(vals.mean()) if not vals.empty else np.nan)
        mins[case].append(float(vals.min()) if not vals.empty else np.nan)
        maxs[case].append(float(vals.max()) if not vals.empty else np.nan)
        rows.append({
            "panel_variable": variable,
            "generalization_type": gtype,
            "generalization_label": label.replace("\n", " "),
            "series": case,
            "mean": means[case][-1],
            "min": mins[case][-1],
            "max": maxs[case][-1],
        })

    return [label], means, mins, maxs, pd.DataFrame(rows)


def _code2_exact_metric_data_for_gtype(
    c2: pd.DataFrame,
    variable: str,
    gtype: str,
) -> Tuple[List[str], Dict[str, List[float]], pd.DataFrame]:
    d = c2[
        c2["generalization_type"].eq(gtype)
        & c2["variable"].eq(variable)
        & c2["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & c2["case"].isin(CODE2_CASES)
        & numeric(c2["N"]).fillna(0).gt(0)
    ].copy()

    combos = [
        ("NO1_1159", "NO2_1107", "CNN_LSTM", "A1→A2\nCNN–LSTM"),
        ("NO1_1159", "NO2_1107", "TCN", "A1→A2\nTCN"),
        ("NO2_1107", "NO1_1159", "CNN_LSTM", "A2→A1\nCNN–LSTM"),
        ("NO2_1107", "NO1_1159", "TCN", "A2→A1\nTCN"),
    ]
    categories = [x[3] for x in combos]
    series = {"Raw WRF": []}
    series.update({case: [] for case in CODE2_CASES})
    rows = []

    for source, target, network, label in combos:
        g = d[
            d["source_site"].eq(source)
            & d["target_site"].eq(target)
            & d["network"].eq(network)
        ].copy()
        raw = numeric(g["Raw_RMSE"]).dropna()
        raw_value = float(raw.mean()) if not raw.empty else np.nan
        series["Raw WRF"].append(raw_value)
        rows.append({
            "generalization_type": gtype,
            "variable": variable,
            "source_site": source,
            "target_site": target,
            "network": network,
            "category": label.replace("\n", " "),
            "series": "Raw WRF",
            "value": raw_value,
        })
        for case in CODE2_CASES:
            hit = g[g["case"].eq(case)]
            value = (
                float(hit.iloc[0]["primary_error"])
                if not hit.empty else np.nan
            )
            series[case].append(value)
            rows.append({
                "generalization_type": gtype,
                "variable": variable,
                "source_site": source,
                "target_site": target,
                "network": network,
                "category": label.replace("\n", " "),
                "series": case,
                "value": value,
            })
    return categories, series, pd.DataFrame(rows)


def _code2_heightwise_mean_range_for_gtype(
    c2: pd.DataFrame,
    gtype: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    d = c2[
        c2["generalization_type"].eq(gtype)
        & c2["variable"].eq("WS")
        & c2["height_num"].notna()
        & c2["case"].isin(CODE2_CASES)
        & numeric(c2["N"]).fillna(0).gt(0)
    ].copy()

    rows = []
    curves: Dict[str, pd.DataFrame] = {}
    raw = (
        d.groupby("height_num", as_index=False)["Raw_RMSE"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"height_num": "height_m"})
    )
    curves["Raw WRF"] = raw
    for _, r in raw.iterrows():
        rows.append({
            "generalization_type": gtype,
            "series": "Raw WRF",
            "height_m": r["height_m"],
            "mean": r["mean"],
            "min": r["min"],
            "max": r["max"],
        })

    for case in CODE2_CASES:
        gc = (
            d[d["case"].eq(case)]
            .groupby("height_num", as_index=False)["RMSE"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .rename(columns={"height_num": "height_m"})
        )
        curves[case] = gc
        for _, r in gc.iterrows():
            rows.append({
                "generalization_type": gtype,
                "series": case,
                "height_m": r["height_m"],
                "mean": r["mean"],
                "min": r["min"],
                "max": r["max"],
            })
    return pd.DataFrame(rows), curves


def _merge_code1_timeseries_for_gtype(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
    gtype: str,
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq(str(gtype))
        & pd.to_numeric(
            pred["WS_height_m"], errors="coerce"
        ).eq(float(height_m))
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty:
        return pd.DataFrame()

    cases_present = [
        c for c in cases if d["case"].astype(str).eq(c).any()
    ]
    if not cases_present:
        return pd.DataFrame()
    base_case = cases_present[0]
    merged = d[d["case"].astype(str).eq(base_case)][
        [TIME_COL, "OBS_WS", "WRF_WS"]
    ].copy()
    for case in cases_present:
        gc = d[d["case"].astype(str).eq(case)][
            [TIME_COL, "CORR_WS"]
        ].rename(columns={"CORR_WS": case})
        merged = merged.merge(gc, on=TIME_COL, how="inner")
    fair_scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
    merged = _fair_filter_timeseries(merged, source, target, fair_scope, height_m)
    return _sci_apply_time_window(merged)


def _merge_code2_timeseries_for_gtype(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
    gtype: str,
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    obs_col = f"OBS_WS{height_m}"
    wrf_col = f"WRF_WS{height_m}"
    corr_col = f"CORR_WS{height_m}"
    required = {obs_col, wrf_col, corr_col}
    if not required.issubset(pred.columns):
        return pd.DataFrame()

    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq(str(gtype))
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty:
        return pd.DataFrame()

    cases_present = [
        c for c in cases if d["case"].astype(str).eq(c).any()
    ]
    if not cases_present:
        return pd.DataFrame()
    base_case = cases_present[0]
    merged = d[d["case"].astype(str).eq(base_case)][
        [TIME_COL, obs_col, wrf_col]
    ].rename(columns={obs_col: "OBS", wrf_col: "Raw WRF"})
    for case in cases_present:
        gc = d[d["case"].astype(str).eq(case)][
            [TIME_COL, corr_col]
        ].rename(columns={corr_col: case})
        merged = merged.merge(gc, on=TIME_COL, how="inner")
    fair_scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
    merged = _fair_filter_timeseries(merged, source, target, fair_scope, height_m)
    return _sci_apply_time_window(merged)


def _draw_code2_heightwise_panel(
    ax: plt.Axes,
    curves: Dict[str, pd.DataFrame],
    title: str,
) -> None:
    for name, gc in curves.items():
        if gc.empty:
            continue
        style = (
            {"linestyle": "--", "marker": "s"}
            if name == "Raw WRF"
            else combined_case_style([name])
        )
        ax.plot(
            gc["height_m"], gc["mean"],
            linewidth=1.3, markersize=5.0,
            label=name, **style,
        )
        ax.fill_between(
            gc["height_m"].to_numpy(dtype=float),
            gc["min"].to_numpy(dtype=float),
            gc["max"].to_numpy(dtype=float),
            alpha=0.10,
        )
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("WS RMSE (m s$^{-1}$)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.2, ncol=2)
    _sci_style_axis(ax)


def _cross_code_profile_panel_for_gtype(
    ax: plt.Axes,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    gtype: str,
    title: str,
) -> pd.DataFrame:
    p1 = c1_profiles[
        c1_profiles["source_site"].eq(source)
        & c1_profiles["target_site"].eq(target)
        & c1_profiles["network"].eq(network)
        & c1_profiles["case"].eq(SCI_CODE1_BEST_WS_CASE)
        & c1_profiles["generalization_type"].eq(gtype)
    ].sort_values("height_m").copy()
    p2 = c2_profiles[
        c2_profiles["source_site"].eq(source)
        & c2_profiles["target_site"].eq(target)
        & c2_profiles["network"].eq(network)
        & c2_profiles["case"].eq(SCI_CODE2_BEST_WS_CASE)
        & c2_profiles["generalization_type"].eq(gtype)
    ].sort_values("height_m").copy()

    if p1.empty or p2.empty:
        ax.set_axis_off()
        return pd.DataFrame()

    # Direct Code1-Code2 profile comparison uses only shared physical heights.
    # Fair-time profile means at each retained height are based on the same timestamps.
    common_heights = sorted(set(pd.to_numeric(p1["height_m"], errors="coerce").dropna())
                            & set(pd.to_numeric(p2["height_m"], errors="coerce").dropna()))
    p1 = p1[pd.to_numeric(p1["height_m"], errors="coerce").isin(common_heights)].copy()
    p2 = p2[pd.to_numeric(p2["height_m"], errors="coerce").isin(common_heights)].copy()
    if not common_heights:
        ax.set_axis_off()
        return pd.DataFrame()

    baseline = p2[["height_m", "OBS_mean", "WRF_mean"]].copy()
    ax.plot(
        baseline["OBS_mean"], baseline["height_m"],
        marker="o", linewidth=SCI_LINEWIDTH_OBS, label="OBS",
    )
    ax.plot(
        baseline["WRF_mean"], baseline["height_m"],
        marker="s", linestyle="--",
        linewidth=SCI_LINEWIDTH_WRF, label="Raw WRF",
    )
    ax.plot(
        p1["CORR_mean"], p1["height_m"],
        marker="^", linewidth=SCI_LINEWIDTH_CASE,
        label=f"Code1 {SCI_CODE1_BEST_WS_CASE} pseudo-profile",
    )
    ax.plot(
        p2["CORR_mean"], p2["height_m"],
        marker="D", linestyle=":",
        linewidth=SCI_LINEWIDTH_CASE,
        label=f"Code2 {SCI_CODE2_BEST_WS_CASE} full profile",
    )
    ax.set_xlabel("Mean WS (m s$^{-1}$)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.2)
    _sci_style_axis(ax)

    out1 = p1.assign(
        code="Code1",
        case=SCI_CODE1_BEST_WS_CASE,
        source_site=source,
        target_site=target,
        network=network,
    )
    out2 = p2.assign(
        code="Code2",
        case=SCI_CODE2_BEST_WS_CASE,
        source_site=source,
        target_site=target,
        network=network,
    )
    return pd.concat([out1, out2], ignore_index=True)


def _cross_code_timeseries_for_gtype(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    gtype: str,
) -> pd.DataFrame:
    t1 = _merge_code1_timeseries_for_gtype(
        c1_pred, source, target, network,
        [SCI_CODE1_BEST_WS_CASE],
        SCI_TS_HEIGHT_M, gtype,
    )
    t2 = _merge_code2_timeseries_for_gtype(
        c2_pred, source, target, network,
        [SCI_CODE2_BEST_WS_CASE],
        SCI_TS_HEIGHT_M, gtype,
    )
    if t1.empty or t2.empty:
        return pd.DataFrame()

    t1 = t1.rename(columns={
        "OBS_WS": "OBS_Code1",
        "WRF_WS": "Raw_WRF_Code1",
        SCI_CODE1_BEST_WS_CASE: f"Code1_{SCI_CODE1_BEST_WS_CASE}",
    })
    t2 = t2.rename(columns={
        "OBS": "OBS_Code2",
        "Raw WRF": "Raw_WRF_Code2",
        SCI_CODE2_BEST_WS_CASE: f"Code2_{SCI_CODE2_BEST_WS_CASE}",
    })
    keep1 = [
        TIME_COL, "OBS_Code1", "Raw_WRF_Code1",
        f"Code1_{SCI_CODE1_BEST_WS_CASE}",
    ]
    keep2 = [
        TIME_COL, "OBS_Code2", "Raw_WRF_Code2",
        f"Code2_{SCI_CODE2_BEST_WS_CASE}",
    ]
    return t1[keep1].merge(t2[keep2], on=TIME_COL, how="inner")


def _save_all_time_code1_individuals(
    c1: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    panel_root: Path,
    gtype: str,
) -> None:
    out_dir = ensure_dir(panel_root / "G4_Code1_all_target_time")
    metric_specs = [
        ("WS", "WS RMSE (m s$^{-1}$)", "Code1 WS: all target times"),
        ("WD", "Circular WD MAE (°)", "Code1 WD: all target times"),
        ("TKE", "TKE RMSE (m$^2$ s$^{-2}$)", "Code1 TKE: all target times"),
    ]
    for letter, (variable, ylabel, title) in zip("abc", metric_specs):
        cats, means, mins, maxs, _ = _code1_metric_panel_data_for_gtype(
            c1, variable, gtype
        )
        fig, ax = _individual_axis()
        _sci_add_grouped_bars(
            ax, cats, means, mins, maxs,
            ylabel=ylabel, title=title,
            decimals=1 if variable == "WD" else 2,
            show_legend=True,
            legend_ncol=min(3, len(means)),
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G4{letter}_{variable}_all_target_time"
        )

    k = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _profile_panel(
                ax, c1_profiles, CODE1_CASES, "Code1",
                source, target, network, gtype,
                f"Code1 {source}→{target}, {network}\nAll valid target times",
                metrics=c1,
            )
            _save_single_panel(
                fig,
                out_dir / f"Fig_G4_profile_{k:02d}_{source}_to_{target}_{network}",
            )
            k += 1
        ts = _merge_code1_timeseries_for_gtype(
            c1_pred, SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
            network, CODE1_CASES, SCI_TS_HEIGHT_M, gtype,
        )
        fig, ax = _individual_axis()
        _timeseries_panel(
            ax, ts, CODE1_CASES, "Code1",
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE, network,
            f"Code1 {SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "All valid target times",
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G4_timeseries_{network}_{SCI_TS_HEIGHT_M}m"
        )


def _save_all_time_code2_individuals(
    c2: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    panel_root: Path,
    gtype: str,
) -> None:
    out_dir = ensure_dir(panel_root / "G5_Code2_all_target_time")
    for letter, variable, ylabel in [
        ("a", "WS", "Pooled WS RMSE (m s$^{-1}$)"),
        ("b", "TKE", "TKE RMSE (m$^2$ s$^{-2}$)"),
    ]:
        cats, series, _ = _code2_exact_metric_data_for_gtype(
            c2, variable, gtype
        )
        fig, ax = _individual_axis()
        _sci_add_grouped_bars(
            ax, cats, series, ylabel=ylabel,
            title=f"Code2 {variable}: all target times",
            decimals=2, show_legend=True, legend_ncol=3,
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G5{letter}_{variable}_all_target_time"
        )

    _, curves = _code2_heightwise_mean_range_for_gtype(c2, gtype)
    fig, ax = _individual_axis()
    _draw_code2_heightwise_panel(
        ax, curves, "Code2 height-wise WS RMSE: all target times"
    )
    _save_single_panel(fig, out_dir / "Fig_G5c_heightwise_WS_all_target_time")

    k = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _profile_panel(
                ax, c2_profiles, CODE2_CASES, "Code2",
                source, target, network, gtype,
                f"Code2 {source}→{target}, {network}\nAll valid target times",
                metrics=c2,
            )
            _save_single_panel(
                fig,
                out_dir / f"Fig_G5_profile_{k:02d}_{source}_to_{target}_{network}",
            )
            k += 1
        ts = _merge_code2_timeseries_for_gtype(
            c2_pred, SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
            network, CODE2_CASES, SCI_TS_HEIGHT_M, gtype,
        )
        fig, ax = _individual_axis()
        _timeseries_panel(
            ax, ts, CODE2_CASES, "Code2",
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE, network,
            f"Code2 {SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "All valid target times",
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G5_timeseries_{network}_{SCI_TS_HEIGHT_M}m"
        )


def _save_all_time_cross_individuals(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    cross_all: pd.DataFrame,
    panel_root: Path,
    gtype: str,
) -> None:
    out_dir = ensure_dir(panel_root / "G6_Code1_vs_Code2_all_target_time")

    for letter, metric, ylabel in [
        ("a", "RMSE", "WS RMSE at 10 m (m s$^{-1}$)"),
        ("b", "Skill", "RMSE skill at 10 m (%)"),
    ]:
        cats, means, mins, maxs, _ = _cross_code_role_summary(
            cross_all, metric
        )
        fig, ax = _individual_axis()
        _sci_add_grouped_bars(
            ax, cats, means, mins, maxs,
            ylabel=ylabel,
            title=f"Code1 versus Code2: all target times ({metric})",
            decimals=2 if metric == "RMSE" else 1,
            show_legend=True, legend_ncol=2,
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G6{letter}_{metric}_all_target_time"
        )

    combo_order = [
        ("NO1_1159→NO2_1107", "CNN_LSTM"),
        ("NO1_1159→NO2_1107", "TCN"),
        ("NO2_1107→NO1_1159", "CNN_LSTM"),
        ("NO2_1107→NO1_1159", "TCN"),
    ]
    combo_labels = [
        "A1→A2\nCNN–LSTM", "A1→A2\nTCN",
        "A2→A1\nCNN–LSTM", "A2→A1\nTCN",
    ]
    delta_series = {}
    for role in ROLE_ORDER:
        values = []
        for direction, network in combo_order:
            hit = cross_all[
                cross_all["direction"].eq(direction)
                & cross_all["network"].eq(network)
                & cross_all["role"].eq(role)
            ]
            values.append(
                float(hit.iloc[0]["Code2_minus_Code1_RMSE"])
                if not hit.empty else np.nan
            )
        delta_series[role] = values
    fig, ax = _individual_axis()
    _sci_add_grouped_bars(
        ax, combo_labels, delta_series,
        ylabel="Code2 − Code1 RMSE (m s$^{-1}$)",
        title="Exact Code2 advantage: all target times",
        zero_line=True, decimals=2,
        show_legend=True, legend_ncol=2,
    )
    _save_single_panel(fig, out_dir / "Fig_G6c_exact_difference_all_target_time")

    k = 1
    for network in SCI_NETWORKS:
        for source, target in SCI_PROFILE_DIRECTIONS:
            fig, ax = _individual_axis()
            _cross_code_profile_panel_for_gtype(
                ax, c1_profiles, c2_profiles,
                source, target, network, gtype,
                f"{source}→{target}, {network}\nAll valid target times",
            )
            _save_single_panel(
                fig,
                out_dir / f"Fig_G6_profile_{k:02d}_{source}_to_{target}_{network}",
            )
            k += 1
        ts = _cross_code_timeseries_for_gtype(
            c1_pred, c2_pred,
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
            network, gtype,
        )
        fig, ax = _individual_axis()
        _plot_cross_code_time_axis(
            ax, ts, network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "All valid target times",
        )
        _save_single_panel(
            fig, out_dir / f"Fig_G6_timeseries_{network}_{SCI_TS_HEIGHT_M}m"
        )


def plot_sci_all_target_time_candidates(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    cross_100: pd.DataFrame,
    out_dir: Path,
    data_dir: Path,
) -> None:
    """Generate G4-G6 candidates for all valid target times."""
    ensure_dir(out_dir)
    ensure_dir(data_dir)
    gtype = "cross_tower_other_time"
    composite_dir = ensure_dir(out_dir / "02_ALL_TARGET_TIME_COMPOSITES")
    panel_root = ensure_dir(out_dir / "03_ALL_TARGET_TIME_INDIVIDUAL_PANELS")
    cross_all = cross_100[
        cross_100["time_scope"].eq("all_target_time")
    ].copy()

    # ---------------- G4: Code1 ----------------
    fig, axes = plt.subplots(3, 3, figsize=SCI_FIGSIZE, layout="constrained")
    axes = np.asarray(axes)
    metric_tables = []
    for ax, (variable, ylabel, title) in zip(
        axes[0],
        [
            ("WS", "WS RMSE (m s$^{-1}$)", "Wind speed"),
            ("WD", "Circular WD MAE (°)", "Wind direction"),
            ("TKE", "TKE RMSE (m$^2$ s$^{-2}$)", "TKE"),
        ],
    ):
        cats, means, mins, maxs, table = _code1_metric_panel_data_for_gtype(
            c1, variable, gtype
        )
        _sci_add_grouped_bars(
            ax, cats, means, mins, maxs,
            ylabel=ylabel, title=title,
            decimals=1 if variable == "WD" else 2,
            show_legend=True, legend_ncol=min(3, len(means)),
        )
        metric_tables.append(table)

    g4_profiles = []
    g4_ts = []
    for row, network in enumerate(SCI_NETWORKS, start=1):
        for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
            table = _profile_panel(
                axes[row, col], c1_profiles, CODE1_CASES, "Code1",
                source, target, network, gtype,
                f"{source}→{target}, {network}\nAll valid target times",
                metrics=c1,
            )
            g4_profiles.append(table)
        ts = _merge_code1_timeseries_for_gtype(
            c1_pred, SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
            network, CODE1_CASES, SCI_TS_HEIGHT_M, gtype,
        )
        _timeseries_panel(
            axes[row, 2], ts, CODE1_CASES, "Code1",
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE, network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "All valid target times",
        )
        if not ts.empty:
            g4_ts.append(ts.assign(network=network))
    add_panel_labels(axes.reshape(-1))
    fig.suptitle(
        "Code1 cross-tower generalization over all valid target times"
    )
    save_figure(
        fig, composite_dir / "Fig_G4_Code1_all_target_time_COMPOSITE"
    )
    pd.concat(metric_tables, ignore_index=True).to_csv(
        data_dir / "Fig_G4_Code1_all_target_time_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    valid = [x for x in g4_profiles if not x.empty]
    if valid:
        pd.concat(valid, ignore_index=True).to_csv(
            data_dir / "Fig_G4_Code1_all_target_time_profiles.csv",
            index=False, encoding="utf-8-sig",
        )
    if g4_ts:
        pd.concat(g4_ts, ignore_index=True).to_csv(
            data_dir / "Fig_G4_Code1_all_target_time_timeseries.csv",
            index=False, encoding="utf-8-sig",
        )

    # ---------------- G5: Code2 ----------------
    fig, axes = plt.subplots(3, 3, figsize=SCI_FIGSIZE, layout="constrained")
    axes = np.asarray(axes)
    cats, ws_series, ws_table = _code2_exact_metric_data_for_gtype(
        c2, "WS", gtype
    )
    _sci_add_grouped_bars(
        axes[0, 0], cats, ws_series,
        ylabel="Pooled WS RMSE (m s$^{-1}$)",
        title="Exact cross-tower WS performance",
        decimals=2, show_legend=True, legend_ncol=3,
    )
    cats_t, tke_series, tke_table = _code2_exact_metric_data_for_gtype(
        c2, "TKE", gtype
    )
    _sci_add_grouped_bars(
        axes[0, 1], cats_t, tke_series,
        ylabel="TKE RMSE (m$^2$ s$^{-2}$)",
        title="Exact cross-tower TKE performance",
        decimals=2, show_legend=True, legend_ncol=3,
    )
    height_table, curves = _code2_heightwise_mean_range_for_gtype(
        c2, gtype
    )
    _draw_code2_heightwise_panel(
        axes[0, 2], curves,
        "Height-wise robustness\nAll valid target times",
    )

    g5_profiles = []
    g5_ts = []
    for row, network in enumerate(SCI_NETWORKS, start=1):
        for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
            table = _profile_panel(
                axes[row, col], c2_profiles, CODE2_CASES, "Code2",
                source, target, network, gtype,
                f"{source}→{target}, {network}\nAll valid target times",
                metrics=c2,
            )
            g5_profiles.append(table)
        ts = _merge_code2_timeseries_for_gtype(
            c2_pred, SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
            network, CODE2_CASES, SCI_TS_HEIGHT_M, gtype,
        )
        _timeseries_panel(
            axes[row, 2], ts, CODE2_CASES, "Code2",
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE, network,
            f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
            "All valid target times",
        )
        if not ts.empty:
            g5_ts.append(ts.assign(network=network))
    add_panel_labels(axes.reshape(-1))
    fig.suptitle(
        "Code2 cross-tower generalization over all valid target times"
    )
    save_figure(
        fig, composite_dir / "Fig_G5_Code2_all_target_time_COMPOSITE"
    )
    pd.concat([ws_table, tke_table], ignore_index=True).to_csv(
        data_dir / "Fig_G5_Code2_all_target_time_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    height_table.to_csv(
        data_dir / "Fig_G5_Code2_all_target_time_heightwise.csv",
        index=False, encoding="utf-8-sig",
    )
    valid = [x for x in g5_profiles if not x.empty]
    if valid:
        pd.concat(valid, ignore_index=True).to_csv(
            data_dir / "Fig_G5_Code2_all_target_time_profiles.csv",
            index=False, encoding="utf-8-sig",
        )
    if g5_ts:
        pd.concat(g5_ts, ignore_index=True).to_csv(
            data_dir / "Fig_G5_Code2_all_target_time_timeseries.csv",
            index=False, encoding="utf-8-sig",
        )

    # ---------------- G6: direct Code1-Code2 comparison ----------------
    if not cross_all.empty:
        fig, axes = plt.subplots(
            3, 3, figsize=SCI_FIGSIZE, layout="constrained"
        )
        axes = np.asarray(axes)
        cats, means, mins, maxs, rmse_table = _cross_code_role_summary(
            cross_all, "RMSE"
        )
        _sci_add_grouped_bars(
            axes[0, 0], cats, means, mins, maxs,
            ylabel="WS RMSE at 10 m (m s$^{-1}$)",
            title="Common-height error",
            decimals=2, show_legend=True, legend_ncol=2,
        )
        cats_s, means_s, mins_s, maxs_s, skill_table = (
            _cross_code_role_summary(cross_all, "Skill")
        )
        _sci_add_grouped_bars(
            axes[0, 1], cats_s, means_s, mins_s, maxs_s,
            ylabel="RMSE skill at 10 m (%)",
            title="Common-height skill",
            decimals=1, show_legend=True, legend_ncol=2,
        )

        combo_order = [
            ("NO1_1159→NO2_1107", "CNN_LSTM"),
            ("NO1_1159→NO2_1107", "TCN"),
            ("NO2_1107→NO1_1159", "CNN_LSTM"),
            ("NO2_1107→NO1_1159", "TCN"),
        ]
        combo_labels = [
            "A1→A2\nCNN–LSTM", "A1→A2\nTCN",
            "A2→A1\nCNN–LSTM", "A2→A1\nTCN",
        ]
        delta_series = {}
        delta_rows = []
        for role in ROLE_ORDER:
            vals = []
            for (direction, network), label in zip(
                combo_order, combo_labels
            ):
                hit = cross_all[
                    cross_all["direction"].eq(direction)
                    & cross_all["network"].eq(network)
                    & cross_all["role"].eq(role)
                ]
                value = (
                    float(hit.iloc[0]["Code2_minus_Code1_RMSE"])
                    if not hit.empty else np.nan
                )
                vals.append(value)
                delta_rows.append({
                    "time_scope": "all_target_time",
                    "direction": direction,
                    "network": network,
                    "category": label.replace("\n", " "),
                    "role": role,
                    "Code2_minus_Code1_RMSE": value,
                })
            delta_series[role] = vals
        _sci_add_grouped_bars(
            axes[0, 2], combo_labels, delta_series,
            ylabel="Code2 − Code1 RMSE (m s$^{-1}$)",
            title="Exact Code2 advantage\nNegative values favor Code2",
            zero_line=True, decimals=2,
            show_legend=True, legend_ncol=2,
        )

        g6_profiles = []
        g6_ts = []
        for row, network in enumerate(SCI_NETWORKS, start=1):
            for col, (source, target) in enumerate(
                SCI_PROFILE_DIRECTIONS
            ):
                table = _cross_code_profile_panel_for_gtype(
                    axes[row, col], c1_profiles, c2_profiles,
                    source, target, network, gtype,
                    f"{source}→{target}, {network}\nAll valid target times",
                )
                g6_profiles.append(table)
            ts = _cross_code_timeseries_for_gtype(
                c1_pred, c2_pred,
                SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE,
                network, gtype,
            )
            _plot_cross_code_time_axis(
                axes[row, 2], ts, network,
                f"{SCI_TS_SOURCE_SITE}→{SCI_TS_TARGET_SITE}, {network}\n"
                "All valid target times",
            )
            if not ts.empty:
                g6_ts.append(ts.assign(network=network))
        add_panel_labels(axes.reshape(-1))
        fig.suptitle(
            "Direct Code1-Code2 comparison over all valid target times"
        )
        save_figure(
            fig,
            composite_dir /
            "Fig_G6_Code1_vs_Code2_all_target_time_COMPOSITE",
        )
        pd.concat([rmse_table, skill_table], ignore_index=True).to_csv(
            data_dir / "Fig_G6_all_target_time_cross_code_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        pd.DataFrame(delta_rows).to_csv(
            data_dir / "Fig_G6_all_target_time_exact_differences.csv",
            index=False, encoding="utf-8-sig",
        )
        valid = [x for x in g6_profiles if not x.empty]
        if valid:
            pd.concat(valid, ignore_index=True).to_csv(
                data_dir / "Fig_G6_all_target_time_profiles.csv",
                index=False, encoding="utf-8-sig",
            )
        if g6_ts:
            pd.concat(g6_ts, ignore_index=True).to_csv(
                data_dir / "Fig_G6_all_target_time_timeseries.csv",
                index=False, encoding="utf-8-sig",
            )

    # Every G4-G6 panel is also exported independently.
    _save_all_time_code1_individuals(
        c1, c1_pred, c1_profiles, panel_root, gtype
    )
    _save_all_time_code2_individuals(
        c2, c2_pred, c2_profiles, panel_root, gtype
    )
    if not cross_all.empty:
        _save_all_time_cross_individuals(
            c1_pred, c2_pred, c1_profiles, c2_profiles,
            cross_all, panel_root, gtype,
        )



# =============================================================================
# 11D. WRF-PERIOD WEIBULL AND WIND-ROSE DIAGNOSTICS FOR G4-G6
# =============================================================================

def _finite_series(values: pd.Series) -> np.ndarray:
    x = numeric(values).to_numpy(dtype=float)
    return x[np.isfinite(x)]


def _fit_weibull_2p(values: pd.Series) -> Tuple[float, float, str]:
    """Two-parameter Weibull fit with location fixed at zero."""
    x = _finite_series(values)
    x = x[x > 0.0]
    if x.size < WEIBULL_MIN_SAMPLES:
        return np.nan, np.nan, "insufficient_positive_samples"
    try:
        from scipy.stats import weibull_min  # type: ignore
        k, _, c = weibull_min.fit(x, floc=0.0)
        return float(k), float(c), "scipy_mle_floc0"
    except Exception:
        pass

    logx = np.log(x)
    mean_logx = float(np.mean(logx))
    k = float(np.clip(1.2 / max(float(np.std(logx)), 1.0e-6), 0.3, 10.0))

    def equation(kk: float) -> float:
        kk = float(np.clip(kk, 0.2, 20.0))
        xk = x ** kk
        return 1.0 / kk - (
            float(np.sum(xk * logx) / np.sum(xk)) - mean_logx
        )

    for _ in range(80):
        f = equation(k)
        eps = max(1.0e-4, 1.0e-4 * abs(k))
        fp = (equation(k + eps) - equation(k - eps)) / (2.0 * eps)
        if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1.0e-10:
            break
        new_k = float(np.clip(k - f / fp, 0.2, 20.0))
        if abs(new_k - k) < 1.0e-7:
            k = new_k
            break
        k = new_k
    c = float((np.mean(x ** k)) ** (1.0 / k))
    return k, c, "fallback_mle_floc0"


def _weibull_pdf(x: np.ndarray, shape_k: float, scale_c: float) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    out = np.zeros_like(xx)
    good = (
        (xx >= 0.0)
        & np.isfinite(xx)
        & np.isfinite(shape_k)
        & np.isfinite(scale_c)
        & (shape_k > 0.0)
        & (scale_c > 0.0)
    )
    if np.any(good):
        z = xx[good] / scale_c
        out[good] = (
            shape_k / scale_c
            * z ** (shape_k - 1.0)
            * np.exp(-(z ** shape_k))
        )
    return out


WIND_DISTRIBUTION_COLUMNS = [
    TIME_COL, "OBS_WS", "WRF_WS", "CORR_WS",
    "OBS_WD", "WRF_WD", "CORR_WD",
]


def _empty_wind_distribution() -> pd.DataFrame:
    """Return an empty distribution table that still preserves the merge key/schema."""
    return pd.DataFrame(columns=WIND_DISTRIBUTION_COLUMNS)


def _safe_time_inner_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Inner-merge two time tables without crashing when one side has no rows/schema.

    Distribution diagnostics are optional for some direction/network combinations.
    An empty extractor must therefore mean 'no panel data', not a KeyError on TIME_COL.
    """
    if left is None or right is None:
        return pd.DataFrame(columns=[TIME_COL])
    if TIME_COL not in left.columns or TIME_COL not in right.columns:
        return pd.DataFrame(columns=[TIME_COL])
    if left.empty or right.empty:
        # Preserve the merged schema as far as practical; this keeps downstream
        # column-existence checks and metadata code deterministic.
        try:
            return left.head(0).merge(
                right.head(0), on=TIME_COL, how="inner", suffixes=suffixes
            )
        except Exception:
            return pd.DataFrame(columns=[TIME_COL])
    return left.merge(right, on=TIME_COL, how="inner", suffixes=suffixes)


def _extract_code1_wind_distribution(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    case: str,
    gtype: str = "cross_tower_other_time",
) -> pd.DataFrame:
    """Matched Code1 WS10 + target-native WD over valid WRF-period endpoints."""
    if pred.empty:
        return _empty_wind_distribution()
    d = pred[
        pred["source_site"].astype(str).eq(source)
        & pred["target_site"].astype(str).eq(target)
        & pred["network"].astype(str).eq(network)
        & pred["case"].astype(str).eq(case)
        & pred["generalization_type"].astype(str).eq(gtype)
        & pd.to_numeric(pred.get("WS_height_m"), errors="coerce").eq(
            float(DISTRIBUTION_WS_HEIGHT_M)
        )
    ].copy()
    need = [
        TIME_COL, "OBS_WS", "WRF_WS", "CORR_WS",
        "OBS_WD", "WRF_WD", "CORR_WD",
    ]
    if d.empty or any(c not in d.columns for c in need):
        return _empty_wind_distribution()
    out = d[need].copy()
    out = out.rename(columns={
        "OBS_WS": "OBS_WS",
        "WRF_WS": "WRF_WS",
        "CORR_WS": "CORR_WS",
        "OBS_WD": "OBS_WD",
        "WRF_WD": "WRF_WD",
        "CORR_WD": "CORR_WD",
    })
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    for c in need[1:]:
        out[c] = numeric(out[c])
    out = _hourly_wind_distribution(out)

    # Distribution diagnostics must use the same physical timestamps as every
    # other Code1/Code2 comparison.  Intersect the fair WS10 and target-native
    # WD masks, then require a complete row for OBS, Raw WRF and Corrected.
    wd_h = int(WINDROSE_WD_HEIGHT_BY_TARGET[target])
    scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
    fair_idx = _fair_distribution_time_index(
        source, target, scope, DISTRIBUTION_WS_HEIGHT_M, wd_h
    )
    if len(fair_idx) > 0:
        out = out[out[TIME_COL].isin(fair_idx)].copy()
    complete = ["OBS_WS", "WRF_WS", "CORR_WS", "OBS_WD", "WRF_WD", "CORR_WD"]
    return (
        out.dropna(subset=[TIME_COL] + complete)
        .sort_values(TIME_COL)
        .drop_duplicates(TIME_COL)
    )


def _extract_code2_wind_distribution(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    case: str,
    gtype: str = "cross_tower_other_time",
) -> pd.DataFrame:
    """Matched Code2 WS10 + target-native WD over valid WRF-period endpoints."""
    if pred.empty:
        return _empty_wind_distribution()
    wd_h = int(WINDROSE_WD_HEIGHT_BY_TARGET[target])
    cols = {
        f"OBS_WS{DISTRIBUTION_WS_HEIGHT_M}": "OBS_WS",
        f"WRF_WS{DISTRIBUTION_WS_HEIGHT_M}": "WRF_WS",
        f"CORR_WS{DISTRIBUTION_WS_HEIGHT_M}": "CORR_WS",
        f"OBS_WD{wd_h}": "OBS_WD",
        f"WRF_WD{wd_h}": "WRF_WD",
        f"CORR_WD{wd_h}": "CORR_WD",
    }
    d = pred[
        pred["source_site"].astype(str).eq(source)
        & pred["target_site"].astype(str).eq(target)
        & pred["network"].astype(str).eq(network)
        & pred["case"].astype(str).eq(case)
        & pred["generalization_type"].astype(str).eq(gtype)
    ].copy()
    if d.empty or any(c not in d.columns for c in cols):
        return _empty_wind_distribution()
    out = d[[TIME_COL] + list(cols)].rename(columns=cols).copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    for c in cols.values():
        out[c] = numeric(out[c])
    out = _hourly_wind_distribution(out)

    # Keep exactly the same fair WS/WD timestamps used by Code1 so that Weibull
    # and wind-rose comparisons cannot be shifted by different validity masks.
    scope = "all_target_time" if str(gtype) == "cross_tower_other_time" else "same_time"
    fair_idx = _fair_distribution_time_index(
        source, target, scope, DISTRIBUTION_WS_HEIGHT_M, wd_h
    )
    if len(fair_idx) > 0:
        out = out[out[TIME_COL].isin(fair_idx)].copy()
    complete = ["OBS_WS", "WRF_WS", "CORR_WS", "OBS_WD", "WRF_WD", "CORR_WD"]
    return (
        out.dropna(subset=[TIME_COL] + complete)
        .sort_values(TIME_COL)
        .drop_duplicates(TIME_COL)
    )


def _distribution_period_metadata(
    data: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    code: str,
    case: str,
) -> Dict[str, object]:
    return {
        "code": code,
        "case": case,
        "source_site": source,
        "target_site": target,
        "network": network,
        "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
        "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET.get(target, np.nan),
        "period_definition": "WRF/postprocess period; all valid target endpoints",
        "period_start": (
            pd.to_datetime(data[TIME_COL], errors="coerce").min()
            if not data.empty else pd.NaT
        ),
        "period_end": (
            pd.to_datetime(data[TIME_COL], errors="coerce").max()
            if not data.empty else pd.NaT
        ),
        "n_time_rows": int(len(data)),
    }


def _plot_weibull_comparison_axis(
    ax: plt.Axes,
    series: Dict[str, pd.Series],
    title: str,
    metadata: Dict[str, object],
) -> pd.DataFrame:
    cleaned: Dict[str, np.ndarray] = {}
    for name, values in series.items():
        arr = _finite_series(values)
        arr = arr[(arr > 0.0) & (arr <= 75.0)]
        if arr.size >= WEIBULL_MIN_SAMPLES:
            cleaned[name] = arr

    if not cleaned:
        ax.text(0.5, 0.5, "Insufficient valid WRF-period WS samples",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return pd.DataFrame()

    vmax = max(float(np.quantile(x, 0.995)) for x in cleaned.values())
    vmax = max(2.0, vmax * 1.15)
    bins = np.arange(0.0, vmax + WEIBULL_BIN_WIDTH, WEIBULL_BIN_WIDTH)
    if len(bins) < 4:
        bins = np.linspace(0.0, vmax, 20)
    grid = np.linspace(0.0, vmax, 400)

    rows = []
    for name, arr in cleaned.items():
        density, edges = np.histogram(arr, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        line = ax.step(
            centers, density, where="mid", alpha=0.42, linewidth=0.9,
            label=f"{name} empirical",
        )[0]
        k, c, method = _fit_weibull_2p(pd.Series(arr))
        if np.isfinite(k) and np.isfinite(c):
            fit_line, = ax.plot(
                grid, _weibull_pdf(grid, k, c),
                linewidth=1.55,
                linestyle=line.get_linestyle(),
                label=f"{name} Weibull (k={k:.2f}, c={c:.2f})",
            )
            fitted = _weibull_pdf(centers, k, c)
            density_rmse = float(np.sqrt(np.mean((density - fitted) ** 2)))
        else:
            density_rmse = np.nan
        rows.append({
            **metadata,
            "dataset": name,
            "N_positive": int(arr.size),
            "mean_WS": float(np.mean(arr)),
            "std_WS": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "weibull_shape_k": k,
            "weibull_scale_c": c,
            "fit_method": method,
            "empirical_density_RMSE": density_rmse,
        })

    ax.set_xlabel("Wind speed (m s$^{-1}$)")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=6.8, ncol=1)
    return pd.DataFrame(rows)


def _windrose_frequency(
    ws: pd.Series,
    wd: pd.Series,
) -> Tuple[np.ndarray, int]:
    speed = numeric(ws).to_numpy(dtype=float)
    direction = numeric(wd).to_numpy(dtype=float)
    valid = (
        np.isfinite(speed)
        & np.isfinite(direction)
        & (speed >= 0.0)
        & (speed <= WINDROSE_SPEED_BINS[-1])
        & (direction >= 0.0)
        & (direction <= 360.0)
    )
    speed = speed[valid]
    direction = direction[valid] % 360.0
    n = int(speed.size)
    matrix = np.zeros(
        (WINDROSE_NUM_SECTORS, len(WINDROSE_SPEED_BINS) - 1),
        dtype=float,
    )
    if n == 0:
        return matrix, 0
    sector_width = 360.0 / WINDROSE_NUM_SECTORS
    sector = np.floor((direction + sector_width / 2.0) / sector_width).astype(int)
    sector %= WINDROSE_NUM_SECTORS
    speed_bin = np.digitize(speed, WINDROSE_SPEED_BINS, right=False) - 1
    speed_bin = np.clip(speed_bin, 0, matrix.shape[1] - 1)
    for s, b in zip(sector, speed_bin):
        matrix[int(s), int(b)] += 1.0
    matrix = matrix / float(n) * 100.0
    return matrix, n


def _plot_windrose_axis(
    ax: plt.Axes,
    ws: pd.Series,
    wd: pd.Series,
    title: str,
    metadata: Dict[str, object],
    dataset: str,
    show_legend: bool = False,
) -> pd.DataFrame:
    matrix, n = _windrose_frequency(ws, wd)
    if n == 0:
        ax.text(
            0.5, 0.5,
            "No matched WS–WD samples\nat target-native WD height",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_axis_off()
        return pd.DataFrame([{
            **metadata,
            "dataset": dataset,
            "status": "not_evaluable",
            "N": 0,
        }])

    theta = np.deg2rad(
        np.arange(WINDROSE_NUM_SECTORS)
        * (360.0 / WINDROSE_NUM_SECTORS)
    )
    width = np.deg2rad(360.0 / WINDROSE_NUM_SECTORS) * 0.88
    bottom = np.zeros(WINDROSE_NUM_SECTORS)
    rows = []
    for j in range(matrix.shape[1]):
        lo = WINDROSE_SPEED_BINS[j]
        hi = WINDROSE_SPEED_BINS[j + 1]
        values = matrix[:, j]
        ax.bar(
            theta, values, width=width, bottom=bottom,
            align="center", edgecolor="black", linewidth=0.25,
            label=f"{lo:g}–{hi:g}",
        )
        for sector_index, freq in enumerate(values):
            rows.append({
                **metadata,
                "dataset": dataset,
                "status": "evaluated",
                "N": n,
                "sector_index": sector_index,
                "sector_center_deg": sector_index
                * (360.0 / WINDROSE_NUM_SECTORS),
                "speed_bin_low": lo,
                "speed_bin_high": hi,
                "joint_frequency_pct": float(freq),
            })
        bottom += values

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title(f"{title}\nN={n}", va="bottom")
    if show_legend:
        ax.legend(
            title="WS (m s$^{-1}$)",
            frameon=False,
            fontsize=6.0,
            title_fontsize=6.5,
            bbox_to_anchor=(1.20, 1.08),
            loc="upper left",
        )
    return pd.DataFrame(rows)


def _save_weibull_individual(
    data: pd.DataFrame,
    datasets: Dict[str, str],
    title: str,
    base: Path,
    metadata: Dict[str, object],
) -> pd.DataFrame:
    fig, ax = _individual_axis()
    table = _plot_weibull_comparison_axis(
        ax,
        {label: data[col] for label, col in datasets.items() if col in data},
        title,
        metadata,
    )
    save_figure(fig, base)
    return table


def _save_windrose_comparison(
    data: pd.DataFrame,
    datasets: Dict[str, Tuple[str, str]],
    title: str,
    base: Path,
    metadata: Dict[str, object],
) -> pd.DataFrame:
    names = list(datasets)
    fig, axes = plt.subplots(
        1, len(names),
        figsize=(5.2 * len(names), 5.4),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    rows = []
    for i, (ax, name) in enumerate(zip(axes, names)):
        ws_col, wd_col = datasets[name]
        table = _plot_windrose_axis(
            ax,
            data[ws_col] if ws_col in data else pd.Series(dtype=float),
            data[wd_col] if wd_col in data else pd.Series(dtype=float),
            name,
            metadata,
            name,
            show_legend=(i == len(names) - 1),
        )
        rows.append(table)
    fig.suptitle(title)
    save_figure(fig, base)
    valid = [x for x in rows if not x.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def plot_sci_wrf_period_distribution_diagnostics(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    sci_root: Path,
    data_dir: Path,
) -> None:
    """Generate G4-G6 Weibull and wind-rose companion figures.

    Every row is drawn only from cross_tower_other_time predictions, which are
    target-tower valid endpoints within the WRF/postprocess period.
    """
    ensure_dir(data_dir)
    composite_root = ensure_dir(
        sci_root / "02_ALL_TARGET_TIME_COMPOSITES" / "distribution_diagnostics"
    )
    individual_root = ensure_dir(
        sci_root / "03_ALL_TARGET_TIME_INDIVIDUAL_PANELS"
        / "distribution_diagnostics"
    )
    all_weibull = []
    all_windrose = []
    period_rows = []

    for source, target in SCI_PROFILE_DIRECTIONS:
        direction_tag = f"{source}_to_{target}"
        for network in SCI_NETWORKS:
            c1 = _extract_code1_wind_distribution(
                c1_pred, source, target, network,
                SCI_CODE1_BEST_WS_CASE,
            )
            c2 = _extract_code2_wind_distribution(
                c2_pred, source, target, network,
                SCI_CODE2_BEST_WS_CASE,
            )

            meta1 = _distribution_period_metadata(
                c1, source, target, network,
                "Code1", SCI_CODE1_BEST_WS_CASE,
            )
            meta2 = _distribution_period_metadata(
                c2, source, target, network,
                "Code2", SCI_CODE2_BEST_WS_CASE,
            )
            period_rows.extend([meta1, meta2])

            # G4: Code1 OBS/WRF/C8
            if not c1.empty:
                g4_dir = ensure_dir(individual_root / "G4_Code1")
                weib = _save_weibull_individual(
                    c1,
                    {
                        "OBS": "OBS_WS",
                        "Raw WRF": "WRF_WS",
                        f"Code1 {SCI_CODE1_BEST_WS_CASE}": "CORR_WS",
                    },
                    f"G4 Code1 {source}→{target}, {network}\n"
                    "WRF-period all valid target times",
                    g4_dir / f"Fig_G4_Weibull_{direction_tag}_{network}",
                    meta1,
                )
                all_weibull.append(weib.assign(figure_group="G4"))
                rose = _save_windrose_comparison(
                    c1,
                    {
                        "OBS": ("OBS_WS", "OBS_WD"),
                        "Raw WRF": ("WRF_WS", "WRF_WD"),
                        f"Code1 {SCI_CODE1_BEST_WS_CASE}": (
                            "CORR_WS", "CORR_WD"
                        ),
                    },
                    f"G4 Code1 wind roses: {source}→{target}, {network}\n"
                    f"WS {DISTRIBUTION_WS_HEIGHT_M} m + "
                    f"WD {WINDROSE_WD_HEIGHT_BY_TARGET[target]} m; WRF period",
                    g4_dir / f"Fig_G4_Windrose_{direction_tag}_{network}",
                    meta1,
                )
                all_windrose.append(rose.assign(figure_group="G4"))

            # G5: Code2 OBS/WRF/C8d
            if not c2.empty:
                g5_dir = ensure_dir(individual_root / "G5_Code2")
                weib = _save_weibull_individual(
                    c2,
                    {
                        "OBS": "OBS_WS",
                        "Raw WRF": "WRF_WS",
                        f"Code2 {SCI_CODE2_BEST_WS_CASE}": "CORR_WS",
                    },
                    f"G5 Code2 {source}→{target}, {network}\n"
                    "WRF-period all valid target times",
                    g5_dir / f"Fig_G5_Weibull_{direction_tag}_{network}",
                    meta2,
                )
                all_weibull.append(weib.assign(figure_group="G5"))
                rose = _save_windrose_comparison(
                    c2,
                    {
                        "OBS": ("OBS_WS", "OBS_WD"),
                        "Raw WRF": ("WRF_WS", "WRF_WD"),
                        f"Code2 {SCI_CODE2_BEST_WS_CASE}": (
                            "CORR_WS", "CORR_WD"
                        ),
                    },
                    f"G5 Code2 wind roses: {source}→{target}, {network}\n"
                    f"WS {DISTRIBUTION_WS_HEIGHT_M} m + "
                    f"WD {WINDROSE_WD_HEIGHT_BY_TARGET[target]} m; WRF period",
                    g5_dir / f"Fig_G5_Windrose_{direction_tag}_{network}",
                    meta2,
                )
                all_windrose.append(rose.assign(figure_group="G5"))

            # G6: direct Code1/Code2 distribution comparison on common times.
            if not c1.empty and not c2.empty:
                merged = _safe_time_inner_merge(
                    c1, c2, suffixes=("_Code1", "_Code2")
                )
                if not merged.empty:
                    g6_meta = {
                        "code": "Code1_vs_Code2",
                        "case": (
                            f"{SCI_CODE1_BEST_WS_CASE}_vs_"
                            f"{SCI_CODE2_BEST_WS_CASE}"
                        ),
                        "source_site": source,
                        "target_site": target,
                        "network": network,
                        "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
                        "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET[target],
                        "period_definition": (
                            "WRF/postprocess period; common valid target endpoints"
                        ),
                        "period_start": merged[TIME_COL].min(),
                        "period_end": merged[TIME_COL].max(),
                        "n_time_rows": int(len(merged)),
                    }
                    # OBS/WRF should be the same across both codes; use the mean
                    # only after retaining both originals in the exported table.
                    # The two codes are evaluated against the same target
                    # observations and WRF baseline. Use the Code1 copy directly
                    # rather than averaging circular directions across 0/360°.
                    merged["OBS_WS_common"] = numeric(
                        merged["OBS_WS_Code1"]
                    )
                    merged["WRF_WS_common"] = numeric(
                        merged["WRF_WS_Code1"]
                    )
                    merged["OBS_WD_common"] = numeric(
                        merged["OBS_WD_Code1"]
                    )
                    merged["WRF_WD_common"] = numeric(
                        merged["WRF_WD_Code1"]
                    )
                    g6_dir = ensure_dir(individual_root / "G6_Code1_vs_Code2")
                    weib = _save_weibull_individual(
                        merged,
                        {
                            "OBS": "OBS_WS_common",
                            "Raw WRF": "WRF_WS_common",
                            f"Code1 {SCI_CODE1_BEST_WS_CASE}": "CORR_WS_Code1",
                            f"Code2 {SCI_CODE2_BEST_WS_CASE}": "CORR_WS_Code2",
                        },
                        f"G6 Code1–Code2 {source}→{target}, {network}\n"
                        "Common WRF-period target times",
                        g6_dir / f"Fig_G6_Weibull_{direction_tag}_{network}",
                        g6_meta,
                    )
                    all_weibull.append(weib.assign(figure_group="G6"))
                    rose = _save_windrose_comparison(
                        merged,
                        {
                            "OBS": ("OBS_WS_common", "OBS_WD_common"),
                            "Raw WRF": ("WRF_WS_common", "WRF_WD_common"),
                            f"Code1 {SCI_CODE1_BEST_WS_CASE}": (
                                "CORR_WS_Code1", "CORR_WD_Code1"
                            ),
                            f"Code2 {SCI_CODE2_BEST_WS_CASE}": (
                                "CORR_WS_Code2", "CORR_WD_Code2"
                            ),
                        },
                        f"G6 Code1–Code2 wind roses: "
                        f"{source}→{target}, {network}\n"
                        f"WS {DISTRIBUTION_WS_HEIGHT_M} m + "
                        f"WD {WINDROSE_WD_HEIGHT_BY_TARGET[target]} m; "
                        "common WRF-period times",
                        g6_dir / f"Fig_G6_Windrose_{direction_tag}_{network}",
                        g6_meta,
                    )
                    all_windrose.append(rose.assign(figure_group="G6"))

    # Compact Weibull overview figures: two directions × two networks.
    for group, code_label, case_label in [
        ("G4", "Code1", SCI_CODE1_BEST_WS_CASE),
        ("G5", "Code2", SCI_CODE2_BEST_WS_CASE),
        ("G6", "Code1 vs Code2", ""),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), layout="constrained")
        axes = np.asarray(axes)
        for row, network in enumerate(SCI_NETWORKS):
            for col, (source, target) in enumerate(SCI_PROFILE_DIRECTIONS):
                if group == "G4":
                    data = _extract_code1_wind_distribution(
                        c1_pred, source, target, network,
                        SCI_CODE1_BEST_WS_CASE,
                    )
                    datasets = {
                        "OBS": data.get("OBS_WS", pd.Series(dtype=float)),
                        "Raw WRF": data.get("WRF_WS", pd.Series(dtype=float)),
                        f"Code1 {SCI_CODE1_BEST_WS_CASE}": data.get(
                            "CORR_WS", pd.Series(dtype=float)
                        ),
                    }
                    meta = _distribution_period_metadata(
                        data, source, target, network,
                        "Code1", SCI_CODE1_BEST_WS_CASE,
                    )
                elif group == "G5":
                    data = _extract_code2_wind_distribution(
                        c2_pred, source, target, network,
                        SCI_CODE2_BEST_WS_CASE,
                    )
                    datasets = {
                        "OBS": data.get("OBS_WS", pd.Series(dtype=float)),
                        "Raw WRF": data.get("WRF_WS", pd.Series(dtype=float)),
                        f"Code2 {SCI_CODE2_BEST_WS_CASE}": data.get(
                            "CORR_WS", pd.Series(dtype=float)
                        ),
                    }
                    meta = _distribution_period_metadata(
                        data, source, target, network,
                        "Code2", SCI_CODE2_BEST_WS_CASE,
                    )
                else:
                    d1 = _extract_code1_wind_distribution(
                        c1_pred, source, target, network,
                        SCI_CODE1_BEST_WS_CASE,
                    )
                    d2 = _extract_code2_wind_distribution(
                        c2_pred, source, target, network,
                        SCI_CODE2_BEST_WS_CASE,
                    )
                    data = _safe_time_inner_merge(
                        d1, d2, suffixes=("_Code1", "_Code2")
                    )
                    if data.empty:
                        datasets = {}
                    else:
                        data["OBS_common"] = numeric(
                            data["OBS_WS_Code1"]
                        )
                        data["WRF_common"] = numeric(
                            data["WRF_WS_Code1"]
                        )
                        datasets = {
                            "OBS": data["OBS_common"],
                            "Raw WRF": data["WRF_common"],
                            f"Code1 {SCI_CODE1_BEST_WS_CASE}": data[
                                "CORR_WS_Code1"
                            ],
                            f"Code2 {SCI_CODE2_BEST_WS_CASE}": data[
                                "CORR_WS_Code2"
                            ],
                        }
                    meta = {
                        "code": "Code1_vs_Code2",
                        "case": (
                            f"{SCI_CODE1_BEST_WS_CASE}_vs_"
                            f"{SCI_CODE2_BEST_WS_CASE}"
                        ),
                        "source_site": source,
                        "target_site": target,
                        "network": network,
                        "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
                        "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET[target],
                        "period_definition": "common WRF-period target endpoints",
                        "period_start": (
                            data[TIME_COL].min() if not data.empty else pd.NaT
                        ),
                        "period_end": (
                            data[TIME_COL].max() if not data.empty else pd.NaT
                        ),
                        "n_time_rows": int(len(data)),
                    }
                _plot_weibull_comparison_axis(
                    axes[row, col],
                    datasets,
                    f"{source}→{target}, {network}",
                    meta,
                )
        add_panel_labels(axes.reshape(-1))
        fig.suptitle(
            f"{group} {code_label}: Weibull comparison over WRF-period "
            "all-valid target times"
        )
        save_figure(
            fig,
            composite_root / f"Fig_{group}_Weibull_WRFperiod_COMPOSITE",
        )

    weibull_table = (
        pd.concat([x for x in all_weibull if not x.empty], ignore_index=True)
        if any(not x.empty for x in all_weibull)
        else pd.DataFrame()
    )
    windrose_table = (
        pd.concat([x for x in all_windrose if not x.empty], ignore_index=True)
        if any(not x.empty for x in all_windrose)
        else pd.DataFrame()
    )
    period_table = pd.DataFrame(period_rows)

    weibull_table.to_csv(
        data_dir / "Fig_G4_G6_WRFperiod_Weibull_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    windrose_table.to_csv(
        data_dir / "Fig_G4_G6_WRFperiod_windrose_frequency.csv",
        index=False, encoding="utf-8-sig",
    )
    period_table.to_csv(
        data_dir / "Fig_G4_G6_WRFperiod_distribution_time_audit.csv",
        index=False, encoding="utf-8-sig",
    )


# =============================================================================
# 11E. BEST NETWORK-CASE SELECTION AND BEST-RESULT FIGURES
# =============================================================================

def _scope_gtype(code: str, scope: str) -> str:
    if code == "Code1":
        return (
            "cross_tower_same_height"
            if scope == "test_period"
            else "cross_tower_other_time"
        )
    return (
        "cross_tower_same_time"
        if scope == "test_period"
        else "cross_tower_other_time"
    )


def _selection_metric_rows(
    metrics: pd.DataFrame,
    code: str,
    scope: str,
) -> pd.DataFrame:
    gtype = _scope_gtype(code, scope)
    d = metrics[
        metrics["generalization_type"].astype(str).eq(gtype)
        & metrics["variable"].astype(str).eq("WS")
        & metrics["case"].astype(str).isin(
            CODE1_CASES if code == "Code1" else CODE2_CASES
        )
        & numeric(metrics["N"]).fillna(0).gt(0)
    ].copy()
    if code == "Code1":
        d = d[d["height_num"].eq(float(REFERENCE_WS_HEIGHT_M))].copy()
    else:
        d = d[
            d["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        ].copy()
    return d


def select_best_network_case(
    metrics: pd.DataFrame,
    code: str,
    scope: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Select one global network-case pair without direction-wise cherry-picking."""
    d = _selection_metric_rows(metrics, code, scope)
    if d.empty:
        return pd.Series(dtype=object), pd.DataFrame()

    rows = []
    for (network, case), g in d.groupby(["network", "case"], dropna=False):
        rmse = numeric(g["RMSE"]).dropna()
        skill = numeric(g["Skill_RMSE_pct"]).dropna()
        corr = numeric(g["Pearson_r"]).dropna()
        mae = numeric(g["MAE"]).dropna()
        rows.append({
            "code": code,
            "scope": scope,
            "selection_generalization_type": _scope_gtype(code, scope),
            "selection_metric": (
                "mean 10 m WS RMSE across directions"
                if code == "Code1"
                else "mean ALL_PROFILE WS RMSE across directions"
            ),
            "network": str(network),
            "case": str(case),
            "n_direction_rows": int(len(rmse)),
            "mean_RMSE": float(rmse.mean()) if not rmse.empty else np.nan,
            "worst_direction_RMSE": (
                float(rmse.max()) if not rmse.empty else np.nan
            ),
            "mean_MAE": float(mae.mean()) if not mae.empty else np.nan,
            "mean_Skill_RMSE_pct": (
                float(skill.mean()) if not skill.empty else np.nan
            ),
            "mean_Pearson_r": (
                float(corr.mean()) if not corr.empty else np.nan
            ),
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        [
            "mean_RMSE",
            "worst_direction_RMSE",
            "mean_Skill_RMSE_pct",
            "mean_Pearson_r",
            "network",
            "case",
        ],
        ascending=[True, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking["rank"].eq(1)
    return ranking.iloc[0], ranking


def _best_single_code_timeseries(
    pred: pd.DataFrame,
    code: str,
    scope: str,
    source: str,
    target: str,
    network: str,
    best_case: str,
    baseline_case: str,
) -> pd.DataFrame:
    gtype = _scope_gtype(code, scope)
    cases = list(dict.fromkeys([baseline_case, best_case]))
    if code == "Code1":
        d = _merge_code1_timeseries_for_gtype(
            pred,
            source,
            target,
            network,
            cases,
            REFERENCE_WS_HEIGHT_M,
            gtype,
        )
        if d.empty:
            return d
        rename = {
            "OBS_WS": "OBS",
            "WRF_WS": "Raw WRF",
            baseline_case: f"Baseline {baseline_case}",
            best_case: f"Best {network} {best_case}",
        }
    else:
        d = _merge_code2_timeseries_for_gtype(
            pred,
            source,
            target,
            network,
            cases,
            REFERENCE_WS_HEIGHT_M,
            gtype,
        )
        if d.empty:
            return d
        rename = {
            baseline_case: f"Baseline {baseline_case}",
            best_case: f"Best {network} {best_case}",
        }
    out = d.rename(columns=rename).copy()
    keep = [
        TIME_COL,
        "OBS",
        "Raw WRF",
        f"Baseline {baseline_case}",
        f"Best {network} {best_case}",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].drop_duplicates(TIME_COL).sort_values(TIME_COL)
    value_cols = [c for c in keep if c != TIME_COL]
    return out.dropna(subset=value_cols).reset_index(drop=True)


def _series_metrics(
    table: pd.DataFrame,
    series_names: Sequence[str],
    metadata: Dict[str, object],
) -> pd.DataFrame:
    if table.empty or "OBS" not in table.columns:
        return pd.DataFrame()
    obs = numeric(table["OBS"]).to_numpy(dtype=float)
    raw_rmse = np.nan
    if "Raw WRF" in table.columns:
        raw = numeric(table["Raw WRF"]).to_numpy(dtype=float)
        valid_raw = np.isfinite(obs) & np.isfinite(raw)
        if np.any(valid_raw):
            raw_rmse = float(
                np.sqrt(np.mean((raw[valid_raw] - obs[valid_raw]) ** 2))
            )

    rows = []
    for name in series_names:
        if name not in table.columns:
            continue
        y = numeric(table[name]).to_numpy(dtype=float)
        valid = np.isfinite(obs) & np.isfinite(y)
        if not np.any(valid):
            continue
        o = obs[valid]
        p = y[valid]
        err = p - o
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        mbe = float(np.mean(err))
        if len(o) >= 2 and np.nanstd(o) > 0 and np.nanstd(p) > 0:
            corr = float(np.corrcoef(o, p)[0, 1])
        else:
            corr = np.nan
        skill = (
            float((1.0 - rmse / raw_rmse) * 100.0)
            if np.isfinite(raw_rmse) and raw_rmse > 1.0e-12
            else np.nan
        )
        rows.append({
            **metadata,
            "series": name,
            "N": int(valid.sum()),
            "RMSE": rmse,
            "MAE": mae,
            "MBE": mbe,
            "Pearson_r": corr,
            "Skill_RMSE_pct": skill,
        })
    return pd.DataFrame(rows)


def _best_profile_single_code(
    profiles: pd.DataFrame,
    code: str,
    scope: str,
    source: str,
    target: str,
    network: str,
    best_case: str,
    baseline_case: str,
) -> pd.DataFrame:
    gtype = _scope_gtype(code, scope)
    if code == "Code1" and scope == "test_period":
        allowed = ["cross_tower_same_height", "cross_tower_other_height"]
        d = profiles[
            profiles["source_site"].eq(source)
            & profiles["target_site"].eq(target)
            & profiles["network"].eq(network)
            & profiles["case"].isin([baseline_case, best_case])
            & profiles["generalization_type"].isin(allowed)
        ].copy()
    else:
        d = profiles[
            profiles["source_site"].eq(source)
            & profiles["target_site"].eq(target)
            & profiles["network"].eq(network)
            & profiles["case"].isin([baseline_case, best_case])
            & profiles["generalization_type"].eq(gtype)
        ].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    baseline = (
        d.groupby("height_m", as_index=False)
        .agg(
            OBS_mean=("OBS_mean", "mean"),
            WRF_mean=("WRF_mean", "mean"),
        )
        .sort_values("height_m")
    )
    for _, r in baseline.iterrows():
        rows.extend([
            {
                "code": code,
                "scope": scope,
                "source_site": source,
                "target_site": target,
                "network": network,
                "case": "OBS",
                "height_m": r["height_m"],
                "mean_WS": r["OBS_mean"],
            },
            {
                "code": code,
                "scope": scope,
                "source_site": source,
                "target_site": target,
                "network": network,
                "case": "Raw WRF",
                "height_m": r["height_m"],
                "mean_WS": r["WRF_mean"],
            },
        ])
    for case, label in [
        (baseline_case, f"Baseline {baseline_case}"),
        (best_case, f"Best {network} {best_case}"),
    ]:
        gc = (
            d[d["case"].eq(case)]
            .groupby("height_m", as_index=False)
            .agg(CORR_mean=("CORR_mean", "mean"))
            .sort_values("height_m")
        )
        for _, r in gc.iterrows():
            rows.append({
                "code": code,
                "scope": scope,
                "source_site": source,
                "target_site": target,
                "network": network,
                "case": label,
                "height_m": r["height_m"],
                "mean_WS": r["CORR_mean"],
            })
    return pd.DataFrame(rows)


def _plot_best_metrics_figure(
    metric_table: pd.DataFrame,
    title: str,
    base: Path,
) -> None:
    if metric_table.empty:
        return
    metrics = [
        ("RMSE", "RMSE (m s$^{-1}$)"),
        ("MAE", "MAE (m s$^{-1}$)"),
        ("Skill_RMSE_pct", "RMSE skill (%)"),
        ("Pearson_r", "Pearson r"),
    ]
    directions = list(metric_table["direction"].drop_duplicates())
    series_order = list(metric_table["series"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), layout="constrained")
    for ax, (metric, ylabel) in zip(axes.reshape(-1), metrics):
        series = {}
        for name in series_order:
            vals = []
            for direction in directions:
                hit = metric_table[
                    metric_table["direction"].eq(direction)
                    & metric_table["series"].eq(name)
                ]
                vals.append(
                    float(hit.iloc[0][metric]) if not hit.empty else np.nan
                )
            series[name] = vals
        _sci_add_grouped_bars(
            ax,
            directions,
            series,
            ylabel=ylabel,
            title=metric,
            decimals=2 if metric != "Skill_RMSE_pct" else 1,
            show_legend=True,
            legend_ncol=min(3, len(series_order)),
            zero_line=metric in {"Skill_RMSE_pct"},
        )
    add_panel_labels(axes.reshape(-1))
    fig.suptitle(title)
    save_figure(fig, base)


def _plot_best_profile_timeseries_figure(
    profiles_by_direction: Dict[str, pd.DataFrame],
    timeseries_by_direction: Dict[str, pd.DataFrame],
    title: str,
    base: Path,
) -> None:
    directions = [f"{s}→{t}" for s, t in SCI_PROFILE_DIRECTIONS]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0), layout="constrained")
    for col, direction in enumerate(directions):
        p = profiles_by_direction.get(direction, pd.DataFrame())
        ax = axes[0, col]
        if p.empty:
            ax.set_axis_off()
        else:
            for name, g in p.groupby("case", sort=False):
                style = {}
                if name == "OBS":
                    style = {"marker": "o", "linewidth": SCI_LINEWIDTH_OBS}
                elif name == "Raw WRF":
                    style = {
                        "marker": "s",
                        "linestyle": "--",
                        "linewidth": SCI_LINEWIDTH_WRF,
                    }
                elif name.startswith("Baseline"):
                    style = {"marker": "^", "linestyle": "-."}
                else:
                    style = {"marker": "D", "linestyle": ":"}
                ax.plot(
                    g["mean_WS"],
                    g["height_m"],
                    label=name,
                    **style,
                )
            ax.set_xlabel("Mean WS (m s$^{-1}$)")
            ax.set_ylabel("Height (m)")
            ax.set_title(f"{direction}: mean profile")
            ax.grid(alpha=GRID_ALPHA)
            ax.legend(frameon=False, fontsize=7.3)

        ts = timeseries_by_direction.get(direction, pd.DataFrame())
        ax = axes[1, col]
        if ts.empty:
            ax.set_axis_off()
        else:
            for name in [c for c in ts.columns if c != TIME_COL]:
                linewidth = 1.45 if name == "OBS" else 1.05
                linestyle = (
                    "--" if name == "Raw WRF"
                    else "-." if name.startswith("Baseline")
                    else ":"
                )
                if name == "OBS":
                    linestyle = "-"
                ax.plot(
                    ts[TIME_COL],
                    ts[name],
                    label=name,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
            ax.set_ylabel("WS (m s$^{-1}$)")
            ax.set_title(f"{direction}: 10 m time series")
            configure_time_axis(ax)
            ax.legend(frameon=False, fontsize=7.0, ncol=2)
    add_panel_labels(axes.reshape(-1))
    fig.suptitle(title)
    save_figure(fig, base)


def _plot_best_weibull_figure(
    data_by_direction: Dict[str, pd.DataFrame],
    datasets_by_direction: Dict[str, Dict[str, str]],
    metadata_by_direction: Dict[str, Dict[str, object]],
    title: str,
    base: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), layout="constrained")
    rows = []
    directions = [f"{s}→{t}" for s, t in SCI_PROFILE_DIRECTIONS]
    for ax, direction in zip(axes, directions):
        d = data_by_direction.get(direction, pd.DataFrame())
        datasets = datasets_by_direction.get(direction, {})
        meta = metadata_by_direction.get(direction, {})
        table = _plot_weibull_comparison_axis(
            ax,
            {
                label: d[col]
                for label, col in datasets.items()
                if col in d.columns
            },
            direction,
            meta,
        )
        rows.append(table)
    add_panel_labels(axes)
    fig.suptitle(title)
    save_figure(fig, base)
    valid = [x for x in rows if not x.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def _plot_best_windrose_figure(
    data_by_direction: Dict[str, pd.DataFrame],
    datasets_by_direction: Dict[str, Dict[str, Tuple[str, str]]],
    metadata_by_direction: Dict[str, Dict[str, object]],
    title: str,
    base: Path,
) -> pd.DataFrame:
    directions = [f"{s}→{t}" for s, t in SCI_PROFILE_DIRECTIONS]
    max_cols = max(
        [len(datasets_by_direction.get(d, {})) for d in directions] or [1]
    )
    fig, axes = plt.subplots(
        2,
        max_cols,
        figsize=(5.0 * max_cols, 10.4),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(2, max_cols)
    rows = []
    for row, direction in enumerate(directions):
        d = data_by_direction.get(direction, pd.DataFrame())
        datasets = datasets_by_direction.get(direction, {})
        meta = metadata_by_direction.get(direction, {})
        for col in range(max_cols):
            ax = axes[row, col]
            if col >= len(datasets):
                ax.set_axis_off()
                continue
            name = list(datasets)[col]
            ws_col, wd_col = datasets[name]
            table = _plot_windrose_axis(
                ax,
                d[ws_col] if ws_col in d.columns else pd.Series(dtype=float),
                d[wd_col] if wd_col in d.columns else pd.Series(dtype=float),
                f"{direction}\n{name}",
                meta,
                name,
                show_legend=(row == 0 and col == max_cols - 1),
            )
            rows.append(table)
    fig.suptitle(title)
    save_figure(fig, base)
    valid = [x for x in rows if not x.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def _best_all_period_distribution_data_single_code(
    pred: pd.DataFrame,
    code: str,
    source: str,
    target: str,
    network: str,
    best_case: str,
    baseline_case: str,
) -> pd.DataFrame:
    extractor = (
        _extract_code1_wind_distribution
        if code == "Code1"
        else _extract_code2_wind_distribution
    )
    base = extractor(
        pred, source, target, network, baseline_case,
        "cross_tower_other_time",
    )
    best = extractor(
        pred, source, target, network, best_case,
        "cross_tower_other_time",
    )
    if base.empty or best.empty:
        return pd.DataFrame(columns=[TIME_COL])
    merged = _safe_time_inner_merge(
        base, best, suffixes=("_baseline", "_best")
    )
    value_cols = [c for c in merged.columns if c != TIME_COL]
    return merged.dropna(subset=value_cols).sort_values(TIME_COL).reset_index(drop=True)


def _make_direct_windrose_composites(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    sci_root: Path,
) -> None:
    """Place G4-G6 wind-rose composites directly in the requested folder."""
    composite_dir = ensure_dir(sci_root / "02_ALL_TARGET_TIME_COMPOSITES")

    # G4 and G5: four direction-network rows, three data columns.
    for group, code, pred, case in [
        ("G4", "Code1", c1_pred, SCI_CODE1_BEST_WS_CASE),
        ("G5", "Code2", c2_pred, SCI_CODE2_BEST_WS_CASE),
    ]:
        combos = [
            (source, target, network)
            for network in SCI_NETWORKS
            for source, target in SCI_PROFILE_DIRECTIONS
        ]
        fig, axes = plt.subplots(
            len(combos),
            3,
            figsize=(15.6, 5.0 * len(combos)),
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )
        axes = np.asarray(axes).reshape(len(combos), 3)
        for row, (source, target, network) in enumerate(combos):
            extractor = (
                _extract_code1_wind_distribution
                if code == "Code1"
                else _extract_code2_wind_distribution
            )
            d = extractor(
                pred, source, target, network, case,
                "cross_tower_other_time",
            )
            meta = _distribution_period_metadata(
                d, source, target, network, code, case
            )
            datasets = {
                "OBS": ("OBS_WS", "OBS_WD"),
                "Raw WRF": ("WRF_WS", "WRF_WD"),
                f"{code} {case}": ("CORR_WS", "CORR_WD"),
            }
            for col, (name, (ws_col, wd_col)) in enumerate(datasets.items()):
                _plot_windrose_axis(
                    axes[row, col],
                    d[ws_col] if ws_col in d else pd.Series(dtype=float),
                    d[wd_col] if wd_col in d else pd.Series(dtype=float),
                    f"{source}→{target}, {network}\n{name}",
                    meta,
                    name,
                    show_legend=(row == 0 and col == 2),
                )
        fig.suptitle(
            f"{group} {code}: WRF-period wind-rose comparison "
            f"(OBS, Raw WRF, {case})"
        )
        save_figure(
            fig,
            composite_dir / f"Fig_{group}_Windrose_WRFperiod_COMPOSITE",
        )

    # G6: four rows, four data columns.
    combos = [
        (source, target, network)
        for network in SCI_NETWORKS
        for source, target in SCI_PROFILE_DIRECTIONS
    ]
    fig, axes = plt.subplots(
        len(combos),
        4,
        figsize=(20.4, 5.0 * len(combos)),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(len(combos), 4)
    for row, (source, target, network) in enumerate(combos):
        d1 = _extract_code1_wind_distribution(
            c1_pred, source, target, network,
            SCI_CODE1_BEST_WS_CASE, "cross_tower_other_time",
        )
        d2 = _extract_code2_wind_distribution(
            c2_pred, source, target, network,
            SCI_CODE2_BEST_WS_CASE, "cross_tower_other_time",
        )
        merged = _safe_time_inner_merge(
            d1, d2, suffixes=("_Code1", "_Code2")
        )
        meta = {
            "code": "Code1_vs_Code2",
            "case": (
                f"{SCI_CODE1_BEST_WS_CASE}_vs_{SCI_CODE2_BEST_WS_CASE}"
            ),
            "source_site": source,
            "target_site": target,
            "network": network,
            "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
            "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET[target],
            "period_definition": "common WRF-period target endpoints",
            "period_start": (
                merged[TIME_COL].min() if not merged.empty else pd.NaT
            ),
            "period_end": (
                merged[TIME_COL].max() if not merged.empty else pd.NaT
            ),
            "n_time_rows": int(len(merged)),
        }
        datasets = {
            "OBS": ("OBS_WS_Code1", "OBS_WD_Code1"),
            "Raw WRF": ("WRF_WS_Code1", "WRF_WD_Code1"),
            f"Code1 {SCI_CODE1_BEST_WS_CASE}": (
                "CORR_WS_Code1", "CORR_WD_Code1"
            ),
            f"Code2 {SCI_CODE2_BEST_WS_CASE}": (
                "CORR_WS_Code2", "CORR_WD_Code2"
            ),
        }
        for col, (name, (ws_col, wd_col)) in enumerate(datasets.items()):
            _plot_windrose_axis(
                axes[row, col],
                merged[ws_col] if ws_col in merged else pd.Series(dtype=float),
                merged[wd_col] if wd_col in merged else pd.Series(dtype=float),
                f"{source}→{target}, {network}\n{name}",
                meta,
                name,
                show_legend=(row == 0 and col == 3),
            )
    fig.suptitle(
        "G6 Code1-Code2: WRF-period wind-rose comparison"
    )
    save_figure(
        fig,
        composite_dir / "Fig_G6_Windrose_WRFperiod_COMPOSITE",
    )


def generate_best_model_summaries(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    sci_root: Path,
    data_dir: Path,
) -> None:
    best_root = ensure_dir(sci_root / "04_BEST_MODEL_SUMMARY")
    ensure_dir(data_dir)

    selections = {}
    ranking_tables = []
    for scope in BEST_SELECTION_SCOPES:
        for code, metrics in [("Code1", c1), ("Code2", c2)]:
            selected, ranking = select_best_network_case(
                metrics, code, scope
            )
            if selected.empty:
                continue
            selections[(scope, code)] = selected
            ranking_tables.append(ranking)

    if ranking_tables:
        ranking_all = pd.concat(ranking_tables, ignore_index=True)
        ranking_all.to_csv(
            data_dir / "best_model_network_case_ranking.csv",
            index=False, encoding="utf-8-sig",
        )
        ranking_all[ranking_all["selected"]].to_csv(
            data_dir / "best_model_selection_summary.csv",
            index=False, encoding="utf-8-sig",
        )

    all_metric_rows = []
    all_profile_rows = []
    all_timeseries_rows = []
    all_weibull_rows = []
    all_windrose_rows = []

    scope_specs = [
        ("test_period", "01_TEST_PERIOD", "B1", "B2", "B3"),
        ("all_wrf_period", "02_ALL_WRF_PERIOD", "B4", "B5", "B6"),
    ]
    for scope, folder, b_code1, b_code2, b_cross in scope_specs:
        scope_dir = ensure_dir(best_root / folder)
        scope_label = BEST_TIME_SCOPE_LABELS[scope]

        selected_code = {}
        for code, metrics, pred, profiles, fig_id, baseline_case in [
            (
                "Code1", c1, c1_pred, c1_profiles,
                b_code1, BEST_BASELINE_CASE_CODE1,
            ),
            (
                "Code2", c2, c2_pred, c2_profiles,
                b_code2, BEST_BASELINE_CASE_CODE2,
            ),
        ]:
            selected = selections.get((scope, code))
            if selected is None or selected.empty:
                continue
            network = str(selected["network"])
            best_case = str(selected["case"])
            selected_code[code] = {
                "network": network,
                "best_case": best_case,
                "baseline_case": baseline_case,
            }
            out_dir = ensure_dir(
                scope_dir / f"{fig_id}_{code}_best_{network}_{best_case}"
            )

            metrics_parts = []
            profiles_by_direction = {}
            timeseries_by_direction = {}
            for source, target in SCI_PROFILE_DIRECTIONS:
                direction = f"{source}→{target}"
                ts = _best_single_code_timeseries(
                    pred, code, scope, source, target,
                    network, best_case, baseline_case,
                )
                timeseries_by_direction[direction] = ts
                if not ts.empty:
                    ts_export = ts.copy()
                    ts_export["code"] = code
                    ts_export["scope"] = scope
                    ts_export["source_site"] = source
                    ts_export["target_site"] = target
                    ts_export["network"] = network
                    ts_export["best_case"] = best_case
                    ts_export["baseline_case"] = baseline_case
                    all_timeseries_rows.append(ts_export)
                    metric = _series_metrics(
                        ts,
                        [c for c in ts.columns if c not in {TIME_COL, "OBS"}],
                        {
                            "code": code,
                            "scope": scope,
                            "direction": direction,
                            "source_site": source,
                            "target_site": target,
                            "network": network,
                            "best_case": best_case,
                            "baseline_case": baseline_case,
                        },
                    )
                    metrics_parts.append(metric)
                    all_metric_rows.append(metric)

                profile = _best_profile_single_code(
                    profiles, code, scope, source, target,
                    network, best_case, baseline_case,
                )
                profiles_by_direction[direction] = profile
                if not profile.empty:
                    all_profile_rows.append(profile)

            metric_table = (
                pd.concat(metrics_parts, ignore_index=True)
                if metrics_parts else pd.DataFrame()
            )
            _plot_best_metrics_figure(
                metric_table,
                f"{fig_id} {code}: selected {network} {best_case} "
                f"versus matched {baseline_case}\n{scope_label}",
                out_dir / f"Fig_{fig_id}_{code}_best_metrics",
            )
            _plot_best_profile_timeseries_figure(
                profiles_by_direction,
                timeseries_by_direction,
                f"{fig_id} {code}: OBS, Raw WRF, matched baseline, "
                f"and selected best model\n{scope_label}",
                out_dir / f"Fig_{fig_id}_{code}_best_profiles_timeseries",
            )

            # Weibull and wind roses are intentionally restricted to all WRF time.
            if scope == "all_wrf_period":
                data_by_direction = {}
                weibull_sets = {}
                windrose_sets = {}
                meta_by_direction = {}
                for source, target in SCI_PROFILE_DIRECTIONS:
                    direction = f"{source}→{target}"
                    d = _best_all_period_distribution_data_single_code(
                        pred, code, source, target,
                        network, best_case, baseline_case,
                    )
                    data_by_direction[direction] = d
                    weibull_sets[direction] = {
                        "OBS": "OBS_WS_baseline",
                        "Raw WRF": "WRF_WS_baseline",
                        f"Baseline {baseline_case}": "CORR_WS_baseline",
                        f"Best {network} {best_case}": "CORR_WS_best",
                    }
                    windrose_sets[direction] = {
                        "OBS": ("OBS_WS_baseline", "OBS_WD_baseline"),
                        "Raw WRF": (
                            "WRF_WS_baseline", "WRF_WD_baseline"
                        ),
                        f"Baseline {baseline_case}": (
                            "CORR_WS_baseline", "CORR_WD_baseline"
                        ),
                        f"Best {network} {best_case}": (
                            "CORR_WS_best", "CORR_WD_best"
                        ),
                    }
                    meta_by_direction[direction] = {
                        "code": code,
                        "case": best_case,
                        "baseline_case": baseline_case,
                        "source_site": source,
                        "target_site": target,
                        "network": network,
                        "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
                        "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET[target],
                        "period_definition": (
                            "WRF/postprocess period; all valid target endpoints"
                        ),
                        "period_start": (
                            d[TIME_COL].min() if not d.empty else pd.NaT
                        ),
                        "period_end": (
                            d[TIME_COL].max() if not d.empty else pd.NaT
                        ),
                        "n_time_rows": int(len(d)),
                    }
                weib = _plot_best_weibull_figure(
                    data_by_direction,
                    weibull_sets,
                    meta_by_direction,
                    f"{fig_id} {code}: Weibull comparison over all valid "
                    "WRF-period target times",
                    out_dir / f"Fig_{fig_id}_{code}_best_Weibull_WRFperiod",
                )
                if not weib.empty:
                    weib["figure_id"] = fig_id
                    all_weibull_rows.append(weib)
                rose = _plot_best_windrose_figure(
                    data_by_direction,
                    windrose_sets,
                    meta_by_direction,
                    f"{fig_id} {code}: wind-rose comparison over all valid "
                    "WRF-period target times",
                    out_dir / f"Fig_{fig_id}_{code}_best_Windrose_WRFperiod",
                )
                if not rose.empty:
                    rose["figure_id"] = fig_id
                    all_windrose_rows.append(rose)

        # Direct Code1-Code2 best-model comparison.
        if {"Code1", "Code2"}.issubset(selected_code):
            s1 = selected_code["Code1"]
            s2 = selected_code["Code2"]
            out_dir = ensure_dir(
                scope_dir /
                f"{b_cross}_Code1_{s1['network']}_{s1['best_case']}"
                f"_vs_Code2_{s2['network']}_{s2['best_case']}"
            )
            metric_parts = []
            profiles_by_direction = {}
            timeseries_by_direction = {}
            for source, target in SCI_PROFILE_DIRECTIONS:
                direction = f"{source}→{target}"
                t1 = _best_single_code_timeseries(
                    c1_pred, "Code1", scope, source, target,
                    s1["network"], s1["best_case"], s1["baseline_case"],
                )
                t2 = _best_single_code_timeseries(
                    c2_pred, "Code2", scope, source, target,
                    s2["network"], s2["best_case"], s2["baseline_case"],
                )
                if t1.empty or t2.empty:
                    timeseries_by_direction[direction] = pd.DataFrame()
                    profiles_by_direction[direction] = pd.DataFrame()
                    continue
                t1 = t1.rename(columns={
                    "OBS": "OBS_Code1",
                    "Raw WRF": "Raw_WRF_Code1",
                    f"Baseline {s1['baseline_case']}": "Code1 baseline",
                    f"Best {s1['network']} {s1['best_case']}": "Code1 best",
                })
                t2 = t2.rename(columns={
                    "OBS": "OBS_Code2",
                    "Raw WRF": "Raw_WRF_Code2",
                    f"Baseline {s2['baseline_case']}": "Code2 baseline",
                    f"Best {s2['network']} {s2['best_case']}": "Code2 best",
                })
                merged = t1.merge(t2, on=TIME_COL, how="inner")
                merged["OBS"] = numeric(merged["OBS_Code1"])
                merged["Raw WRF"] = numeric(merged["Raw_WRF_Code1"])
                ts = merged[
                    [
                        TIME_COL, "OBS", "Raw WRF",
                        "Code1 baseline", "Code1 best",
                        "Code2 baseline", "Code2 best",
                    ]
                ].copy()
                timeseries_by_direction[direction] = ts
                metric = _series_metrics(
                    ts,
                    [
                        "Raw WRF", "Code1 baseline", "Code1 best",
                        "Code2 baseline", "Code2 best",
                    ],
                    {
                        "code": "Code1_vs_Code2",
                        "scope": scope,
                        "direction": direction,
                        "source_site": source,
                        "target_site": target,
                        "Code1_network": s1["network"],
                        "Code1_best_case": s1["best_case"],
                        "Code2_network": s2["network"],
                        "Code2_best_case": s2["best_case"],
                    },
                )
                metric_parts.append(metric)
                all_metric_rows.append(metric)

                p1 = _best_profile_single_code(
                    c1_profiles, "Code1", scope, source, target,
                    s1["network"], s1["best_case"], s1["baseline_case"],
                )
                p2 = _best_profile_single_code(
                    c2_profiles, "Code2", scope, source, target,
                    s2["network"], s2["best_case"], s2["baseline_case"],
                )
                if p1.empty or p2.empty:
                    profiles_by_direction[direction] = pd.DataFrame()
                else:
                    p1 = p1[p1["case"].isin([
                        "OBS", "Raw WRF",
                        f"Baseline {s1['baseline_case']}",
                        f"Best {s1['network']} {s1['best_case']}",
                    ])].copy()
                    p1["case"] = p1["case"].replace({
                        f"Baseline {s1['baseline_case']}": "Code1 baseline",
                        f"Best {s1['network']} {s1['best_case']}": "Code1 best",
                    })
                    p2 = p2[~p2["case"].isin(["OBS", "Raw WRF"])].copy()
                    p2["case"] = p2["case"].replace({
                        f"Baseline {s2['baseline_case']}": "Code2 baseline",
                        f"Best {s2['network']} {s2['best_case']}": "Code2 best",
                    })
                    profiles_by_direction[direction] = pd.concat(
                        [p1, p2], ignore_index=True
                    )

            metric_table = (
                pd.concat(metric_parts, ignore_index=True)
                if metric_parts else pd.DataFrame()
            )
            _plot_best_metrics_figure(
                metric_table,
                f"{b_cross}: independently selected Code1 and Code2 best "
                f"models\n{scope_label}",
                out_dir / f"Fig_{b_cross}_best_cross_code_metrics",
            )
            _plot_best_profile_timeseries_figure(
                profiles_by_direction,
                timeseries_by_direction,
                f"{b_cross}: OBS, Raw WRF, matched baselines, and "
                f"independently selected best models\n{scope_label}",
                out_dir / f"Fig_{b_cross}_best_cross_code_profiles_timeseries",
            )

            if scope == "all_wrf_period":
                data_by_direction = {}
                weibull_sets = {}
                windrose_sets = {}
                meta_by_direction = {}
                for source, target in SCI_PROFILE_DIRECTIONS:
                    direction = f"{source}→{target}"
                    d1 = _best_all_period_distribution_data_single_code(
                        c1_pred, "Code1", source, target,
                        s1["network"], s1["best_case"], s1["baseline_case"],
                    )
                    d2 = _best_all_period_distribution_data_single_code(
                        c2_pred, "Code2", source, target,
                        s2["network"], s2["best_case"], s2["baseline_case"],
                    )
                    d = _safe_time_inner_merge(
                        d1, d2, suffixes=("_Code1", "_Code2")
                    )
                    data_by_direction[direction] = d
                    weibull_sets[direction] = {
                        "OBS": "OBS_WS_baseline_Code1",
                        "Raw WRF": "WRF_WS_baseline_Code1",
                        "Code1 baseline": "CORR_WS_baseline_Code1",
                        "Code1 best": "CORR_WS_best_Code1",
                        "Code2 baseline": "CORR_WS_baseline_Code2",
                        "Code2 best": "CORR_WS_best_Code2",
                    }
                    windrose_sets[direction] = {
                        "OBS": (
                            "OBS_WS_baseline_Code1",
                            "OBS_WD_baseline_Code1",
                        ),
                        "Raw WRF": (
                            "WRF_WS_baseline_Code1",
                            "WRF_WD_baseline_Code1",
                        ),
                        "Code1 baseline": (
                            "CORR_WS_baseline_Code1",
                            "CORR_WD_baseline_Code1",
                        ),
                        "Code1 best": (
                            "CORR_WS_best_Code1",
                            "CORR_WD_best_Code1",
                        ),
                        "Code2 baseline": (
                            "CORR_WS_baseline_Code2",
                            "CORR_WD_baseline_Code2",
                        ),
                        "Code2 best": (
                            "CORR_WS_best_Code2",
                            "CORR_WD_best_Code2",
                        ),
                    }
                    meta_by_direction[direction] = {
                        "code": "Code1_vs_Code2",
                        "case": (
                            f"{s1['network']} {s1['best_case']} vs "
                            f"{s2['network']} {s2['best_case']}"
                        ),
                        "source_site": source,
                        "target_site": target,
                        "ws_height_m": DISTRIBUTION_WS_HEIGHT_M,
                        "wd_height_m": WINDROSE_WD_HEIGHT_BY_TARGET[target],
                        "period_definition": (
                            "common all-valid target endpoints in WRF period"
                        ),
                        "period_start": (
                            d[TIME_COL].min() if not d.empty else pd.NaT
                        ),
                        "period_end": (
                            d[TIME_COL].max() if not d.empty else pd.NaT
                        ),
                        "n_time_rows": int(len(d)),
                    }
                weib = _plot_best_weibull_figure(
                    data_by_direction,
                    weibull_sets,
                    meta_by_direction,
                    f"{b_cross}: Weibull comparison for independently "
                    "selected best models over the WRF period",
                    out_dir / f"Fig_{b_cross}_best_cross_code_Weibull_WRFperiod",
                )
                if not weib.empty:
                    weib["figure_id"] = b_cross
                    all_weibull_rows.append(weib)
                rose = _plot_best_windrose_figure(
                    data_by_direction,
                    windrose_sets,
                    meta_by_direction,
                    f"{b_cross}: wind-rose comparison for independently "
                    "selected best models over the WRF period",
                    out_dir / f"Fig_{b_cross}_best_cross_code_Windrose_WRFperiod",
                )
                if not rose.empty:
                    rose["figure_id"] = b_cross
                    all_windrose_rows.append(rose)

    if all_metric_rows:
        pd.concat(all_metric_rows, ignore_index=True).to_csv(
            data_dir / "best_model_direct_WS_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
    if all_profile_rows:
        pd.concat(all_profile_rows, ignore_index=True).to_csv(
            data_dir / "best_model_profiles.csv",
            index=False, encoding="utf-8-sig",
        )
    if all_timeseries_rows:
        pd.concat(all_timeseries_rows, ignore_index=True).to_csv(
            data_dir / "best_model_timeseries.csv",
            index=False, encoding="utf-8-sig",
        )
    if all_weibull_rows:
        pd.concat(all_weibull_rows, ignore_index=True).to_csv(
            data_dir / "best_model_WRFperiod_Weibull_summary.csv",
            index=False, encoding="utf-8-sig",
        )
    if all_windrose_rows:
        pd.concat(all_windrose_rows, ignore_index=True).to_csv(
            data_dir / "best_model_WRFperiod_windrose_frequency.csv",
            index=False, encoding="utf-8-sig",
        )

def write_sci_figure_guide(out_dir: Path) -> None:
    guide = f"""SCI main-figure guide
=====================

Two time definitions are reported separately.

A. Source-test-time candidates
------------------------------
00_COMPOSITE and 01_INDIVIDUAL_PANELS contain:
- G1: Code1 generalization at source-test endpoints.
- G2: Code2 cross-tower same-time generalization.
- G3: direct Code1-Code2 comparison at common 10 m and common test times.

B. All-valid-target-time candidates
-----------------------------------
02_ALL_TARGET_TIME_COMPOSITES and
03_ALL_TARGET_TIME_INDIVIDUAL_PANELS contain:
- G4: Code1 cross-tower generalization over all valid target times.
- G5: Code2 cross-tower generalization over all valid target times.
- G6: direct Code1-Code2 comparison over all valid target times.

WRF-period distribution companions
----------------------------------
For G4, G5 and G6, additional Weibull and wind-rose figures are generated from
the same cross_tower_other_time prediction tables. Therefore, all samples lie
within the WRF/postprocess period and pass the target-time validity filtering.

- Weibull: compares positive WS samples at 10 m.
- Wind rose: combines WS at 10 m with the configured target-tower WD height.
  Current configuration uses 10 m WD for NO1_1159, NO2_1107, NO3_1071 and NOT_1166.
- G4 compares OBS, Raw WRF and Code1 C8.
- G5 compares OBS, Raw WRF and Code2 C8d.
- G6 compares OBS, Raw WRF, Code1 C8 and Code2 C8d on common timestamps.
- Code2 wind roses require the revised generalization runner with audited
  target-native WD-head rebasing; no target OBS is interpolated.

Direct composite wind roses
----------------------------
The following wind-rose composites are written directly under
02_ALL_TARGET_TIME_COMPOSITES:
- Fig_G4_Windrose_WRFperiod_COMPOSITE
- Fig_G5_Windrose_WRFperiod_COMPOSITE
- Fig_G6_Windrose_WRFperiod_COMPOSITE

Best-model summaries
--------------------
One global network-case pair is selected independently for Code1 and Code2 in:
- the source-test/common-time period;
- all valid target times within the WRF period.

Selection uses mean WS RMSE across all available transfer directions. It does not choose
a different model for each direction. The matched baseline is C2 using the same
selected network. Output 04_BEST_MODEL_SUMMARY contains:
- B1-B3 for the test/common-time period: metrics, profiles and time series;
- B4-B6 for the all-WRF-period evaluation: metrics, profiles, time series,
  Weibull distributions and wind roses.

Weibull and wind-rose best-model comparisons are generated only for the all
valid target times within the WRF/postprocess period.

"All valid target times" means every target-site endpoint retained after:
1. the existing ML-ready obs_valid_flag, when present;
2. finite/non-negative target WS observation and WRF checks;
3. finite model-input windows.
Source-period overlap is intentionally retained because this follows the user's
definition of other-time validation; it is not a strict temporal non-overlap test.

Composite layout
----------------
Row 1: metric comparisons.
Row 2: CNN-LSTM profiles in both directions and the CNN-LSTM time series.
Row 3: TCN profiles in both directions and the TCN time series.

Networks included: {", ".join(SCI_NETWORKS)}
Time-series direction: {SCI_TS_SOURCE_SITE}->{SCI_TS_TARGET_SITE}
Time-series height: {SCI_TS_HEIGHT_M} m

Legend policy
-------------
- Every grouped bar panel has a complete legend.
- Every profile and time-series panel has a complete legend.
- Exact-overlap cases are combined in the legend, e.g. C2 = C6.
- Code2 profile legends include pooled ALL_PROFILE RMSE and mark the best case.

Interpretation
--------------
- Mean profiles assess long-term vertical structure.
- Pooled and height-wise RMSE assess time-varying predictive performance.
- Code2 WD is omitted when no common observed WD height exists.
"""
    write_text(out_dir / "SCI_FIGURE_GUIDE.txt", guide)


# =============================================================================
# 12. ANALYSIS BUNDLE
# =============================================================================

def build_key_metrics_bundle(c1: pd.DataFrame, c2: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "code", "source_site", "target_site", "direction", "network", "case",
        "role", "generalization_type", "task_name", "variable", "height_m",
        "N", "RMSE", "MAE", "MBE", "Pearson_r", "Raw_RMSE",
        "Skill_RMSE_pct", "primary_error",
    ]
    return pd.concat([c1[keep], c2[keep]], ignore_index=True)


def copy_audit_files(source_root: Path, target_dir: Path, prefix: str) -> None:
    audit = source_root / "00_audit"
    if not audit.exists():
        return
    ensure_dir(target_dir)
    for filename in [
        "run_summary.json",
        "model_discovery.csv",
        "target_data_discovery.csv",
        "application_audit.json",
        "failures.csv",
        "generated_tasks.json",
    ]:
        src = audit / filename
        if src.exists():
            shutil.copy2(src, target_dir / f"{prefix}_{filename}")


# =============================================================================
# 13. MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Xilinhaote Code1/Code2 generalization results.")
    parser.add_argument("--code1-root", type=Path, default=CODE1_GENERALIZATION_ROOT)
    parser.add_argument("--code2-root", type=Path, default=CODE2_GENERALIZATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=PNG_DPI)
    return parser.parse_args()



# =============================================================================
# FINAL COMPLETE-COVERAGE OVERRIDES
# =============================================================================
# Fairness policy used by this final version:
#   * Formal metrics: common time is enforced PER physical height, not across all
#     heights.  This prevents a missing 70/100-m value from deleting an otherwise
#     valid 10-m sample.
#   * At one height, all available compared codes/networks/cases share exactly the
#     same hourly timestamps.
#   * Mean-profile points use the corresponding height-specific common calendar;
#     N is exported for every height.
#   * Time-series figures are an availability view: each series is drawn wherever
#     that series + OBS/Raw WRF are valid. Formal fair metrics remain in the strict
#     height-common metric tables. This avoids visually empty time series while
#     preserving auditable fair statistics.
#   * No plotting downsampling is used after hourly harmonisation.

MAX_TIME_POINTS = 10**9


def _complete_scope_name(scope: str) -> str:
    s = str(scope)
    if s in {"ALL", "all_target_time", "cross_tower_other_time"}:
        return "all_target_time"
    if s == "same_tower_other_height":
        return "same_tower_other_height"
    return "same_time"


def _complete_scope_label(fair_scope: str) -> str:
    return "ALL" if str(fair_scope) == "all_target_time" else "TEST"


def build_fair_products_memory_safe(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex]]:
    """Height-specific, memory-safe fair-time recomputation.

    The former direction-variable intersection forced every height to be valid at
    once.  With four Xilinhaote towers that can unnecessarily collapse a 10-m
    series because another height is missing.  This final engine uses

        scope + source->target + variable + physical height

    as the formal fairness unit.  Within that unit every non-empty compared
    code/network/case series uses exactly the same hourly timestamps.
    """
    global GLOBAL_COMMON_TIME_INDEX, GLOBAL_TIME_SERIES_AUDIT, GLOBAL_TIME_SCOPE_AUDIT

    metric_rows = {"Code1": [], "Code2": []}
    profile_rows = {"Code1": [], "Code2": []}
    audit_rows: List[Dict[str, object]] = []
    series_audit_rows: List[Dict[str, object]] = []
    scope_audit_rows: List[Dict[str, object]] = []
    time_index: Dict[Tuple[str, str, str, str, float], pd.DatetimeIndex] = {}
    GLOBAL_COMMON_TIME_INDEX = {}

    specs = _ms_specs(c1_pred, c2_pred)
    if not specs:
        raise ValueError("No Xilinhaote fair-time task specifications were found in prediction tables.")

    for spec in specs:
        source, target = str(spec["source"]), str(spec["target"])
        scope = str(spec["scope"])
        group_type = str(spec["group_type"])
        fair_scope = (
            "all_target_time" if scope == "ALL"
            else ("same_time" if group_type == "CROSS_DIRECTION_VARIABLE" else "same_tower_other_height")
        )

        for var in ["WS", "WD", "TKE", "T"]:
            parts = []
            if spec["c1_gtypes"]:
                x1 = _ms_code1_group_long(c1_pred, source, target, spec["c1_gtypes"], var)
                if not x1.empty:
                    parts.append(x1)
            if spec["c2_gtypes"]:
                x2 = _ms_code2_group_long(c2_pred, source, target, spec["c2_gtypes"], var)
                if not x2.empty:
                    parts.append(x2)
            if not parts:
                continue

            combined = pd.concat(parts, ignore_index=True)
            del parts
            heights = sorted(pd.to_numeric(combined["height_m"], errors="coerce").dropna().unique())
            code2_pool_chunks = []

            for h in heights:
                gh = combined[pd.to_numeric(combined["height_m"], errors="coerce").eq(float(h))].copy()
                if gh.empty:
                    continue

                series_cols = ["code", "network", "case", "generalization_type", "task_name"]
                valid_sets = []
                n_nonempty = 0
                n_zero = 0
                for sid, gs in gh.groupby(series_cols, dropna=False, sort=False):
                    smeta = dict(zip(series_cols, sid if isinstance(sid, tuple) else (sid,)))
                    obs_mask = gs[TIME_COL].notna() & gs["OBS"].notna()
                    wrf_mask = gs[TIME_COL].notna() & gs["WRF"].notna()
                    model_mask = gs[TIME_COL].notna() & gs["CORR"].notna()
                    complete = obs_mask & wrf_mask & model_mask
                    obs_n, obs_start, obs_end = _time_stats(gs.loc[obs_mask, TIME_COL])
                    wrf_n, wrf_start, wrf_end = _time_stats(gs.loc[wrf_mask, TIME_COL])
                    model_n, model_start, model_end = _time_stats(gs.loc[model_mask, TIME_COL])
                    complete_n, complete_start, complete_end = _time_stats(gs.loc[complete, TIME_COL])
                    include = complete_n > 0
                    if include:
                        idx = pd.DatetimeIndex(gs.loc[complete, TIME_COL].drop_duplicates().sort_values())
                        valid_sets.append(set(idx.asi8.tolist()))
                        n_nonempty += 1
                    else:
                        n_zero += 1
                    series_audit_rows.append({
                        "group_type": group_type,
                        "scope": scope,
                        "fair_scope": fair_scope,
                        "source_site": source,
                        "target_site": target,
                        "variable": var,
                        "height_m": float(h),
                        **smeta,
                        "OBS_N": obs_n, "OBS_start": obs_start, "OBS_end": obs_end,
                        "WRF_N": wrf_n, "WRF_start": wrf_start, "WRF_end": wrf_end,
                        "MODEL_N": model_n, "MODEL_start": model_start, "MODEL_end": model_end,
                        "COMPLETE_N": complete_n,
                        "COMPLETE_start": complete_start,
                        "COMPLETE_end": complete_end,
                        "included_in_common_intersection": include,
                        "series_status": (
                            "INCLUDED_IN_HEIGHT_INTERSECTION" if include
                            else "NO_COMPLETE_VALID_TIMES_EXCLUDED"
                        ),
                    })

                common_ns = set.intersection(*valid_sets) if valid_sets else set()
                common = (
                    pd.DatetimeIndex(pd.to_datetime(sorted(common_ns)))
                    if common_ns else pd.DatetimeIndex([])
                )
                if len(common):
                    st, en = pd.Timestamp(common.min()), pd.Timestamp(common.max())
                    slots = int((en - st) / pd.Timedelta(hours=1)) + 1
                    missing = int(slots - len(common))
                else:
                    st = en = pd.NaT
                    slots = missing = 0

                group_code_label = "Code1+Code2" if group_type == "CROSS_DIRECTION_VARIABLE" else "Code1"
                group_gtype_label = "cross-code height-aligned" if group_type == "CROSS_DIRECTION_VARIABLE" else "same_tower_other_height"
                scope_audit_rows.append({
                    "group_type": group_type,
                    "scope": scope,
                    "fair_scope": fair_scope,
                    "source_site": source,
                    "target_site": target,
                    "variable": var,
                    "height_m": float(h),
                    "code": group_code_label,
                    "generalization_type": group_gtype_label,
                    "N_global_common": int(len(common)),
                    "time_start": st,
                    "time_end": en,
                    "n_nonempty_series_in_intersection": int(n_nonempty),
                    "n_zero_complete_series_excluded": int(n_zero),
                    "hourly_slots_between_start_end": slots,
                    "missing_hourly_slots_inside_span": missing,
                    "fairness_unit": "scope+direction+variable+height",
                    "status": "PASS" if len(common) else "NO_HEIGHT_COMMON_TIME",
                })

                time_index[(fair_scope, source, target, var, float(h))] = common
                # Backward-compatible direction-level alias is the physical 10-m
                # calendar only; it is no longer an all-height intersection.
                if abs(float(h) - float(REFERENCE_WS_HEIGHT_M)) < 1.0e-9:
                    GLOBAL_COMMON_TIME_INDEX[(scope, source, target, var)] = common

                if len(common) == 0:
                    audit_rows.append({
                        "group_type": group_type, "scope": scope, "fair_scope": fair_scope,
                        "source_site": source, "target_site": target,
                        "variable": var, "height_m": float(h),
                        "N_common": 0, "time_start": pd.NaT, "time_end": pd.NaT,
                        "baseline_obs_max_spread": np.nan,
                        "baseline_wrf_max_spread": np.nan,
                        "status": "NO_HEIGHT_COMMON_TIME",
                    })
                    continue

                gg = gh[gh[TIME_COL].isin(common)].copy()
                gg = gg[gg[["OBS", "WRF", "CORR"]].notna().all(axis=1)].copy()
                if gg.empty:
                    continue

                spread = gg.groupby(TIME_COL, dropna=False).agg(
                    obs_min=("OBS", "min"), obs_max=("OBS", "max"),
                    wrf_min=("WRF", "min"), wrf_max=("WRF", "max"),
                )
                obs_spread = float((spread["obs_max"] - spread["obs_min"]).abs().max()) if not spread.empty else np.nan
                wrf_spread = float((spread["wrf_max"] - spread["wrf_min"]).abs().max()) if not spread.empty else np.nan

                # One canonical OBS/WRF pair at this exact height/time.  Never
                # average or splice Code1/Code2 observation lineages row by row.
                # For direct cross-code transfer groups, prefer Code2 because it
                # comes from original 10-min target data and is hourly-aggregated
                # by this postprocessor.  Code1-only same-tower groups retain their
                # native Code1 baseline.
                b = gg[[TIME_COL, "code", "OBS", "WRF"]].copy()
                if group_type == "CROSS_DIRECTION_VARIABLE":
                    b["_priority"] = np.where(
                        b["code"].astype(str).eq(CROSS_CODE_CANONICAL_BASELINE), 0, 1
                    )
                    baseline_policy = (
                        "Code2 preferred original-10min-derived hourly baseline at same height; "
                        "Code1 fallback only if Code2 is structurally unavailable"
                    )
                else:
                    b["_priority"] = np.where(b["code"].astype(str).eq("Code1"), 0, 1)
                    baseline_policy = "Code1 baseline for Code1-only task"
                b = b.sort_values([TIME_COL, "_priority"]).drop_duplicates(TIME_COL, keep="first")
                b = b.rename(columns={"OBS": "_OBS_CAN", "WRF": "_WRF_CAN"})[[TIME_COL, "_OBS_CAN", "_WRF_CAN"]]
                gg = gg.merge(b, on=TIME_COL, how="left")
                gg["OBS"] = gg["_OBS_CAN"]
                gg["WRF"] = gg["_WRF_CAN"]
                gg = gg.drop(columns=["_OBS_CAN", "_WRF_CAN"])

                audit_rows.append({
                    "group_type": group_type, "scope": scope, "fair_scope": fair_scope,
                    "source_site": source, "target_site": target,
                    "variable": var, "height_m": float(h),
                    "n_series_with_valid": int(n_nonempty),
                    "N_common": int(len(common)),
                    "time_start": common.min(), "time_end": common.max(),
                    "baseline_obs_max_spread": obs_spread,
                    "baseline_wrf_max_spread": wrf_spread,
                    "baseline_policy": baseline_policy,
                    "status": (
                        "PASS_CANONICALIZED_BASELINE"
                        if ((np.isfinite(obs_spread) and obs_spread > FAIR_BASELINE_ATOL)
                            or (np.isfinite(wrf_spread) and wrf_spread > FAIR_BASELINE_ATOL))
                        else "PASS_IDENTICAL_BASELINE"
                    ),
                })

                keys = ["source_site", "target_site", "network", "case", "generalization_type", "task_name", "variable", "height_m"]
                for code in ["Code1", "Code2"]:
                    dc = gg[gg["code"].astype(str).eq(code)].copy()
                    if dc.empty:
                        continue
                    for key, gm in dc.groupby(keys, dropna=False, sort=False):
                        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
                        row = _ms_metric_row(meta, gm)
                        row["fair_time_policy"] = "HEIGHT_SPECIFIC_COMMON"
                        metric_rows[code].append(row)

                    if code == "Code2":
                        code2_pool_chunks.append(dc.copy())

                    if var == "WS":
                        pkeys = ["source_site", "target_site", "network", "case", "generalization_type", "height_m"]
                        for key, gp in dc.groupby(pkeys, dropna=False, sort=False):
                            meta = dict(zip(pkeys, key if isinstance(key, tuple) else (key,)))
                            vv = gp[["OBS", "WRF", "CORR"]].dropna()
                            if vv.empty:
                                continue
                            profile_rows[code].append({
                                **meta,
                                "OBS_mean": float(pd.to_numeric(vv["OBS"], errors="coerce").mean()),
                                "WRF_mean": float(pd.to_numeric(vv["WRF"], errors="coerce").mean()),
                                "CORR_mean": float(pd.to_numeric(vv["CORR"], errors="coerce").mean()),
                                "N": int(len(vv)),
                                "fair_time_aligned": True,
                                "fair_time_policy": "HEIGHT_SPECIFIC_COMMON",
                            })
                del gg, gh

            # Pool Code2 across all height-specific fair samples. Each height is
            # internally fair; no unrelated height can delete another height.
            if code2_pool_chunks:
                pool = pd.concat(code2_pool_chunks, ignore_index=True)
                pkeys = ["source_site", "target_site", "network", "case", "generalization_type", "variable"]
                for key, gp in pool.groupby(pkeys, dropna=False, sort=False):
                    meta = dict(zip(pkeys, key if isinstance(key, tuple) else (key,)))
                    row = _ms_metric_row({
                        **meta,
                        "task_name": f"FAIR_HEIGHTWISE_POOLED_{source}_to_{target}_{meta['generalization_type']}",
                        "height_m": "ALL_PROFILE",
                    }, gp)
                    row["fair_time_policy"] = "HEIGHT_SPECIFIC_COMMON_POOLED"
                    metric_rows["Code2"].append(row)
                del pool, code2_pool_chunks
            del combined

    GLOBAL_TIME_SERIES_AUDIT = pd.DataFrame(series_audit_rows)
    GLOBAL_TIME_SCOPE_AUDIT = pd.DataFrame(scope_audit_rows)
    if not GLOBAL_TIME_SERIES_AUDIT.empty:
        lookup = {
            (str(r["group_type"]), str(r["scope"]), str(r["source_site"]), str(r["target_site"]),
             str(r["variable"]), float(r["height_m"])):
            (r["N_global_common"], r["time_start"], r["time_end"])
            for _, r in GLOBAL_TIME_SCOPE_AUDIT.iterrows()
            if pd.notna(r.get("height_m", np.nan))
        }
        used = []
        for _, r in GLOBAL_TIME_SERIES_AUDIT.iterrows():
            key = (str(r["group_type"]), str(r["scope"]), str(r["source_site"]), str(r["target_site"]),
                   str(r["variable"]), float(r["height_m"]))
            used.append(lookup.get(key, (np.nan, pd.NaT, pd.NaT)))
        GLOBAL_TIME_SERIES_AUDIT["USED_N"] = [u[0] for u in used]
        GLOBAL_TIME_SERIES_AUDIT["USED_start"] = [u[1] for u in used]
        GLOBAL_TIME_SERIES_AUDIT["USED_end"] = [u[2] for u in used]

    c1m = _ms_finalize_metrics(metric_rows["Code1"], "Code1")
    c2m = _ms_finalize_metrics(metric_rows["Code2"], "Code2")
    c1p = pd.DataFrame(profile_rows["Code1"])
    c2p = pd.DataFrame(profile_rows["Code2"])
    return c1m, c2m, c1p, c2p, pd.DataFrame(audit_rows), time_index


def _write_global_time_audit_files(*roots: Path) -> None:
    """Write exact HEIGHT-SPECIFIC common-time audit files."""
    roots = tuple(Path(r) for r in roots if r is not None)
    if not roots:
        return
    lines = [
        "Xilinhaote FINAL FAIR TIME AUDIT",
        "=" * 100,
        "Formal fairness unit: scope + source->target + physical variable + physical height.",
        "Code2 native 10-min rows are aggregated to the hourly grid before alignment.",
        "This avoids deleting valid 10-m samples merely because another profile height is missing.",
        "Each height point remains strictly fair across all non-empty compared codes/networks/cases.",
        "",
    ]
    if GLOBAL_TIME_SCOPE_AUDIT.empty:
        lines.append("No height-specific common-time groups were produced.")
    else:
        for _, r in GLOBAL_TIME_SCOPE_AUDIT.iterrows():
            h = r.get("height_m", np.nan)
            hs = f"{float(h):g} m" if pd.notna(h) else "NA"
            lines.extend([
                f"[{r.get('scope','?')}] {r.get('source_site','?')} -> {r.get('target_site','?')} "
                f"| {r.get('variable','?')} | {hs}",
                f"  start = {r.get('time_start', pd.NaT)}",
                f"  end   = {r.get('time_end', pd.NaT)}",
                f"  N     = {int(r.get('N_global_common', 0))}",
                f"  non-empty series = {int(r.get('n_nonempty_series_in_intersection', 0))}",
                f"  empty series excluded = {int(r.get('n_zero_complete_series_excluded', 0))}",
                f"  status = {r.get('status','')}",
                "",
            ])
    lines.extend([
        "Exact timestamp files use:",
        "COMMON_TIMESTAMPS_<TEST|ALL>_<SOURCE>_to_<TARGET>_<VARIABLE>_<HEIGHT>m.csv",
        "GLOBAL_COMMON_TIMESTAMPS_TEST.csv / ALL.csv are backward-compatible aliases",
        f"for {SCI_TS_SOURCE_SITE}->{SCI_TS_TARGET_SITE}, WS, {REFERENCE_WS_HEIGHT_M:g} m only.",
    ])
    txt = "\n".join(lines)

    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        GLOBAL_TIME_SCOPE_AUDIT.to_csv(root / "global_time_scope_audit.csv", index=False, encoding="utf-8-sig")
        GLOBAL_TIME_SERIES_AUDIT.to_csv(root / "global_time_series_audit.csv", index=False, encoding="utf-8-sig")
        (root / "TIME_RANGE_AUDIT.txt").write_text(txt, encoding="utf-8")

        for (fair_scope, source, target, variable, h), idx in FAIR_TIME_INDEX.items():
            if len(idx) == 0:
                continue
            scope = _complete_scope_label(fair_scope)
            htoken = f"{float(h):g}".replace(".", "p")
            name = (
                f"COMMON_TIMESTAMPS_{_safe_token(scope)}_"
                f"{_safe_token(source)}_to_{_safe_token(target)}_"
                f"{_safe_token(variable)}_{htoken}m.csv"
            )
            pd.DataFrame({TIME_COL: pd.DatetimeIndex(idx).sort_values()}).to_csv(
                root / name, index=False, encoding="utf-8-sig"
            )
            if (
                str(source) == str(SCI_TS_SOURCE_SITE)
                and str(target) == str(SCI_TS_TARGET_SITE)
                and str(variable) == "WS"
                and abs(float(h) - float(REFERENCE_WS_HEIGHT_M)) < 1.0e-9
                and scope in {"TEST", "ALL"}
            ):
                pd.DataFrame({TIME_COL: pd.DatetimeIndex(idx).sort_values()}).to_csv(
                    root / f"GLOBAL_COMMON_TIMESTAMPS_{scope}.csv",
                    index=False, encoding="utf-8-sig",
                )


def _fair_filter_timeseries(
    d: pd.DataFrame,
    source: str,
    target: str,
    scope: str,
    height_m: int,
) -> pd.DataFrame:
    """Strict per-height common filter used where direct fair comparison is required."""
    if d.empty or TIME_COL not in d.columns:
        return d.copy()
    fair_scope = _complete_scope_name(scope)
    idx = FAIR_TIME_INDEX.get(
        (fair_scope, str(source), str(target), "WS", float(height_m)),
        pd.DatetimeIndex([]),
    )
    out = _hourly_mean_timeseries(d)
    if len(idx):
        out = out[out[TIME_COL].isin(idx)].copy()
    # If an upstream group has no formal index, retain the displayed-series
    # complete cases instead of returning a blank figure.
    value_cols = [c for c in out.columns if c != TIME_COL]
    if value_cols:
        out = out.dropna(subset=value_cols, how="any").copy()
    return out.sort_values(TIME_COL)


def _fair_distribution_time_index(
    source: str,
    target: str,
    scope: str,
    ws_height_m: int,
    wd_height_m: int,
) -> pd.DatetimeIndex:
    fair_scope = _complete_scope_name(scope)
    ws = FAIR_TIME_INDEX.get((fair_scope, str(source), str(target), "WS", float(ws_height_m)))
    wd = FAIR_TIME_INDEX.get((fair_scope, str(source), str(target), "WD", float(wd_height_m)))
    if ws is None or wd is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(ws.intersection(wd).sort_values())


def _complete_available_timeseries_frame(
    base: pd.DataFrame,
    case_frames: Dict[str, pd.DataFrame],
    obs_col: str,
    wrf_col: str,
) -> pd.DataFrame:
    """Outer-merge every available hourly series without erasing sparse curves.

    Important robustness rule: an all-NaN OBS/WRF/case column is kept in the
    returned schema.  `_hourly_mean_timeseries` intentionally drops columns that
    contain no numeric values, so without this restoration a structurally
    unavailable height (for example OBS_WS4 at one target tower) can raise a
    KeyError later even though another curve at that height is available.
    """
    required_cols = [obs_col, wrf_col] + list(case_frames.keys())
    if base.empty:
        return pd.DataFrame(columns=[TIME_COL] + required_cols)

    # Aggregate the baseline to hourly resolution FIRST.  Do not drop duplicate
    # timestamps before aggregation: baseline rows can be repeated across cases,
    # and one case may contain NaN where another carries the valid OBS/WRF value.
    out = base[[TIME_COL, obs_col, wrf_col]].copy()
    out = _hourly_mean_timeseries(out)

    # Each corrected case is hourly-aggregated independently before the outer
    # merge, preventing duplicate-timestamp Cartesian products for native 10-min
    # Code2 outputs.
    for case, frame in case_frames.items():
        if frame.empty or TIME_COL not in frame.columns:
            continue
        value_candidates = [c for c in frame.columns if c != TIME_COL]
        if not value_candidates:
            continue
        value_col = value_candidates[-1]
        f = frame[[TIME_COL, value_col]].copy().rename(columns={value_col: case})
        f = _hourly_mean_timeseries(f)
        if TIME_COL not in f.columns:
            continue
        out = out.merge(f, on=TIME_COL, how="outer")

    # Preserve a stable schema even when a particular physical height has no OBS
    # at this tower. Plotting code can then skip the unavailable curve while still
    # drawing the WRF/model series that really exist.
    if TIME_COL not in out.columns:
        out[TIME_COL] = pd.Series(dtype="datetime64[ns]")
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    ordered = [TIME_COL] + required_cols
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras].sort_values(TIME_COL).reset_index(drop=True)


def _complete_ts_metric_rows(
    frame: pd.DataFrame,
    code: str,
    gtype: str,
    source: str,
    target: str,
    network: str,
    height: float,
    obs_col: str,
    wrf_col: str,
    cases: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    if frame.empty:
        return rows
    obs = pd.to_numeric(frame[obs_col], errors="coerce")
    wrf = pd.to_numeric(frame[wrf_col], errors="coerce")
    strict_idx = FAIR_TIME_INDEX.get(
        (_complete_scope_name(gtype), str(source), str(target), "WS", float(height)),
        pd.DatetimeIndex([]),
    )
    # Raw WRF availability metric.
    mask = frame[TIME_COL].notna() & obs.notna() & wrf.notna()
    if mask.any():
        o = obs[mask].to_numpy(dtype=float)
        b = wrf[mask].to_numpy(dtype=float)
        stats = _fair_scalar_metrics(o, b, b)
        rows.append({
            "code": code, "generalization_type": gtype,
            "source_site": source, "target_site": target, "network": network,
            "height_m": float(height), "series": "Raw WRF",
            "display_metric_basis": "PAIRWISE_AVAILABLE",
            "formal_strict_height_common_N": int(len(strict_idx)),
            "time_start": frame.loc[mask, TIME_COL].min(),
            "time_end": frame.loc[mask, TIME_COL].max(),
            **stats,
        })
    for case in cases:
        if case not in frame.columns:
            continue
        pred = pd.to_numeric(frame[case], errors="coerce")
        mask = frame[TIME_COL].notna() & obs.notna() & wrf.notna() & pred.notna()
        if not mask.any():
            continue
        stats = _fair_scalar_metrics(
            obs[mask].to_numpy(dtype=float),
            pred[mask].to_numpy(dtype=float),
            wrf[mask].to_numpy(dtype=float),
        )
        rows.append({
            "code": code, "generalization_type": gtype,
            "source_site": source, "target_site": target, "network": network,
            "height_m": float(height), "series": case,
            "display_metric_basis": "PAIRWISE_AVAILABLE",
            "formal_strict_height_common_N": int(len(strict_idx)),
            "time_start": frame.loc[mask, TIME_COL].min(),
            "time_end": frame.loc[mask, TIME_COL].max(),
            **stats,
        })
    return rows


def _complete_short_site(x: object) -> str:
    return {"NO1_1159": "N1", "NO2_1107": "N2", "NO3_1071": "N3", "NOT_1166": "NT"}.get(str(x), safe_name(x, 12))


def _complete_short_network(x: object) -> str:
    return {"CNN_LSTM": "CL", "TCN": "TCN"}.get(str(x), safe_name(x, 10))


def _complete_short_gtype(x: object) -> str:
    return {
        "same_tower_other_height": "STOH",
        "cross_tower_same_height": "CTSH",
        "cross_tower_other_height": "CTOH",
        "cross_tower_same_time": "CTST",
        "cross_tower_other_time": "CTOT",
    }.get(str(x), safe_name(x, 16))


def plot_all_available_ws_timeseries(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    out_dir: Path,
    metrics_csv: Path,
) -> pd.DataFrame:
    """Plot every available WS time series for every tower/direction/network/height.

    No empty panels are created.  Each physical series is drawn wherever it is
    available; formal strict fair statistics remain in the main fair metric tables.
    This function also exports pairwise-available display metrics and the strict-N
    used by the formal height-common evaluation.
    """
    out_dir = ensure_dir(out_dir)
    metric_rows: List[Dict[str, object]] = []

    # ---------------- Code1 ----------------
    if not c1_pred.empty and TIME_COL in c1_pred.columns:
        req = {"source_site", "target_site", "network", "case", "generalization_type", "WS_height_m", "OBS_WS", "WRF_WS", "CORR_WS"}
        if req.issubset(c1_pred.columns):
            d = c1_pred[c1_pred["case"].astype(str).isin(CODE1_CASES)].copy()
            d["_h"] = pd.to_numeric(d["WS_height_m"], errors="coerce")
            for (gtype, source, target, network, h), g in d.dropna(subset=["_h"]).groupby(
                ["generalization_type", "source_site", "target_site", "network", "_h"], dropna=False, sort=True
            ):
                cases_present = [c for c in CODE1_CASES if g["case"].astype(str).eq(c).any()]
                if not cases_present:
                    continue
                base = g[[TIME_COL, "OBS_WS", "WRF_WS"]].copy()
                cframes = {
                    c: g[g["case"].astype(str).eq(c)][[TIME_COL, "CORR_WS"]].copy()
                    for c in cases_present
                }
                ts = _complete_available_timeseries_frame(base, cframes, "OBS_WS", "WRF_WS")
                if ts.empty:
                    continue
                # Keep only hours with at least OBS or Raw WRF or one corrected curve.
                value_cols = [c for c in ts.columns if c != TIME_COL]
                ts = ts.dropna(subset=value_cols, how="all")
                if ts.empty:
                    continue
                fig, ax = plt.subplots(figsize=(13.2, 4.8))
                _plot_gap_safe(ax, ts[TIME_COL], ts["OBS_WS"], label="OBS", linewidth=1.35)
                _plot_gap_safe(ax, ts[TIME_COL], ts["WRF_WS"], label="Raw WRF", linestyle="--", linewidth=1.05)
                for case in cases_present:
                    if case in ts.columns and pd.to_numeric(ts[case], errors="coerce").notna().any():
                        style = CASE_STYLE.get(case, {})
                        _plot_gap_safe(ax, ts[TIME_COL], ts[case], label=case, linewidth=1.0, **style)
                n_obs = int(pd.to_numeric(ts["OBS_WS"], errors="coerce").notna().sum())
                ax.set_title(f"Code1 {source}→{target} | {network} | {gtype} | {float(h):g} m | OBS N={n_obs}")
                ax.set_ylabel("WS (m s$^{-1}$)")
                ax.set_xlabel("Time")
                configure_time_axis(ax)
                ax.legend(frameon=False, ncol=min(6, 2 + len(cases_present)))
                fig.tight_layout()
                save_figure(fig, ensure_dir(out_dir / "C1" / _complete_short_gtype(gtype) / f"{_complete_short_site(source)}_{_complete_short_site(target)}" / _complete_short_network(network)) /
                            f"TS_C1_{_complete_short_site(source)}_{_complete_short_site(target)}_{_complete_short_network(network)}_{float(h):g}m")
                metric_rows.extend(_complete_ts_metric_rows(
                    ts, "Code1", str(gtype), str(source), str(target), str(network), float(h),
                    "OBS_WS", "WRF_WS", cases_present,
                ))

    # ---------------- Code2 ----------------
    if not c2_pred.empty and TIME_COL in c2_pred.columns:
        meta = {"source_site", "target_site", "network", "case", "generalization_type"}
        if meta.issubset(c2_pred.columns):
            d = c2_pred[c2_pred["case"].astype(str).isin(CODE2_CASES)].copy()
            heights = sorted({int(m.group(1)) for c in d.columns if (m := re.fullmatch(r"OBS_WS(\d+)", str(c)))})
            for (gtype, source, target, network), g in d.groupby(
                ["generalization_type", "source_site", "target_site", "network"], dropna=False, sort=True
            ):
                cases_present = [c for c in CODE2_CASES if g["case"].astype(str).eq(c).any()]
                if not cases_present:
                    continue
                for h in heights:
                    obs_col, wrf_col, corr_col = f"OBS_WS{h}", f"WRF_WS{h}", f"CORR_WS{h}"
                    if not {obs_col, wrf_col, corr_col}.issubset(g.columns):
                        continue
                    # The wide Code2 table exposes the union of height columns from
                    # all towers. A column can therefore exist but be completely NaN
                    # for the current target tower. Such a height is not an evaluable
                    # target-observation level and should not generate an empty plot.
                    obs_available = pd.to_numeric(g[obs_col], errors="coerce").notna().any()
                    if not bool(obs_available):
                        continue
                    base = g[[TIME_COL, obs_col, wrf_col]].copy()
                    cframes = {
                        c: g[g["case"].astype(str).eq(c)][[TIME_COL, corr_col]].copy()
                        for c in cases_present
                    }
                    ts = _complete_available_timeseries_frame(base, cframes, obs_col, wrf_col)
                    if ts.empty:
                        continue
                    value_cols = [c for c in ts.columns if c != TIME_COL]
                    ts = ts.dropna(subset=value_cols, how="all")
                    if ts.empty:
                        continue
                    obs_ok = obs_col in ts.columns and pd.to_numeric(ts[obs_col], errors="coerce").notna().any()
                    wrf_ok = wrf_col in ts.columns and pd.to_numeric(ts[wrf_col], errors="coerce").notna().any()
                    valid_cases = [
                        case for case in cases_present
                        if case in ts.columns and pd.to_numeric(ts[case], errors="coerce").notna().any()
                    ]
                    if not obs_ok and not wrf_ok and not valid_cases:
                        continue
                    fig, ax = plt.subplots(figsize=(13.2, 4.8))
                    if obs_ok:
                        _plot_gap_safe(ax, ts[TIME_COL], ts[obs_col], label="OBS", linewidth=1.35)
                    if wrf_ok:
                        _plot_gap_safe(ax, ts[TIME_COL], ts[wrf_col], label="Raw WRF", linestyle="--", linewidth=1.05)
                    for case in valid_cases:
                        style = CASE_STYLE.get(case, {})
                        _plot_gap_safe(ax, ts[TIME_COL], ts[case], label=case, linewidth=1.0, **style)
                    n_obs = int(pd.to_numeric(ts[obs_col], errors="coerce").notna().sum()) if obs_col in ts.columns else 0
                    ax.set_title(f"Code2 {source}→{target} | {network} | {gtype} | {h:g} m | OBS N={n_obs}")
                    ax.set_ylabel("WS (m s$^{-1}$)")
                    ax.set_xlabel("Time")
                    configure_time_axis(ax)
                    ax.legend(frameon=False, ncol=min(6, 2 + len(cases_present)))
                    fig.tight_layout()
                    save_figure(fig, ensure_dir(out_dir / "C2" / _complete_short_gtype(gtype) / f"{_complete_short_site(source)}_{_complete_short_site(target)}" / _complete_short_network(network)) /
                                f"TS_C2_{_complete_short_site(source)}_{_complete_short_site(target)}_{_complete_short_network(network)}_{h:g}m")
                    metric_rows.extend(_complete_ts_metric_rows(
                        ts, "Code2", str(gtype), str(source), str(target), str(network), float(h),
                        obs_col, wrf_col, cases_present,
                    ))

    metrics = pd.DataFrame(metric_rows)
    ensure_dir(metrics_csv.parent)
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    return metrics


def _complete_code1_full_profile_data(
    c1_profiles: pd.DataFrame,
    source: str,
    target: str,
    network: str,
) -> pd.DataFrame:
    if c1_profiles.empty:
        return pd.DataFrame()
    d = c1_profiles[
        c1_profiles["source_site"].astype(str).eq(str(source))
        & c1_profiles["target_site"].astype(str).eq(str(target))
        & c1_profiles["network"].astype(str).eq(str(network))
        & c1_profiles["generalization_type"].astype(str).isin(["cross_tower_same_height", "cross_tower_other_height"])
        & c1_profiles["case"].astype(str).isin(CODE1_CASES)
    ].copy()
    if d.empty:
        return d
    # Same-height contributes the 10-m point; other-height contributes the rest.
    return d.sort_values(["height_m", "case"])


def plot_code1_full_cross_tower_pseudo_profiles_all(
    c1_profiles: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """All directed Code1 pseudo-profiles, explicitly including the 10-m same-height point."""
    out_dir = ensure_dir(out_dir)
    rows = []
    if c1_profiles.empty:
        return pd.DataFrame()
    pairs = c1_profiles[
        c1_profiles["generalization_type"].astype(str).isin(["cross_tower_same_height", "cross_tower_other_height"])
    ][["source_site", "target_site", "network"]].drop_duplicates()
    for source, target, network in pairs.itertuples(index=False, name=None):
        g = _complete_code1_full_profile_data(c1_profiles, source, target, network)
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.3, 6.8))
        baseline = (
            g.groupby("height_m", as_index=False)
             .agg(OBS_mean=("OBS_mean", "mean"), WRF_mean=("WRF_mean", "mean"), N=("N", "max"))
             .sort_values("height_m")
        )
        ax.plot(baseline["OBS_mean"], baseline["height_m"], marker="o", linewidth=1.8, label="OBS")
        ax.plot(baseline["WRF_mean"], baseline["height_m"], marker="s", linestyle="--", linewidth=1.5, label="Raw WRF")
        for case in CODE1_CASES:
            gc = g[g["case"].astype(str).eq(case)].sort_values("height_m")
            if gc.empty:
                continue
            # There can be one row per physical height after fair recomputation.
            gc = gc.groupby("height_m", as_index=False).agg(CORR_mean=("CORR_mean", "mean"), N=("N", "max"))
            style = CASE_STYLE.get(case, {})
            ax.plot(gc["CORR_mean"], gc["height_m"], linewidth=1.35, markersize=5.0, label=case, **style)
        ax.set_xlabel("Mean WS (m s$^{-1}$)")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"Code1 full cross-tower pseudo-profile\n{source}→{target} — {network} (10 m included)")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        save_figure(fig, out_dir / f"Fig_Code1_FULL_profile_{source}_to_{target}_{network}")
        rows.append(g.assign(profile_family="cross_tower_full_pseudo_profile"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_code2_exact_run_all_profile_ws(df: pd.DataFrame, out_dir: Path) -> None:
    """Dynamic, complete exact-run Code2 case-ranking pages for every transfer/network."""
    d = df[
        df["variable"].eq("WS")
        & df["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & df["case"].isin(CODE2_CASES)
        & numeric(df["N"]).fillna(0).gt(0)
        & numeric(df["RMSE"]).notna()
    ].copy()
    if d.empty:
        return
    out_dir = ensure_dir(out_dir)
    for gtype, dg in d.groupby("generalization_type", dropna=False):
        combos = list(dg[["source_site", "target_site", "network"]].drop_duplicates().itertuples(index=False, name=None))
        if not combos:
            continue
        per_page = 6
        for page0 in range(0, len(combos), per_page):
            page = combos[page0:page0 + per_page]
            ncols = 2
            nrows = int(math.ceil(len(page) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12.4, 4.2 * nrows), layout="constrained")
            axes_flat = np.atleast_1d(axes).reshape(-1)
            for ax, (source, target, network) in zip(axes_flat, page):
                g = dg[
                    dg["source_site"].eq(source)
                    & dg["target_site"].eq(target)
                    & dg["network"].eq(network)
                ]
                values, labels = [], []
                for case in CODE2_CASES:
                    hit = g[g["case"].eq(case)]
                    if hit.empty:
                        continue
                    values.append(float(hit.iloc[0]["RMSE"]))
                    labels.append(case)
                x = np.arange(len(values))
                bars = ax.bar(x, values)
                label_bars(ax, bars, 2)
                if values:
                    best_i = int(np.nanargmin(values))
                    bars[best_i].set_hatch("///")
                    bars[best_i].set_linewidth(1.3)
                    bars[best_i].set_edgecolor("black")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylabel("Pooled WS RMSE (m s$^{-1}$)")
                ax.set_title(f"{source}→{target} — {network}")
                ax.grid(axis="y", alpha=GRID_ALPHA)
            for ax in axes_flat[len(page):]:
                ax.set_axis_off()
            fig.suptitle(f"Code2 exact ALL_PROFILE WS ranking — {gtype} — page {page0//per_page + 1}")
            save_figure(fig, out_dir / safe_name(gtype) / f"Fig_Code2_exact_ALL_PROFILE_page_{page0//per_page + 1:02d}")


def _complete_metric_pages(
    metrics: pd.DataFrame,
    code: str,
    cases: Sequence[str],
    out_dir: Path,
) -> None:
    """SCI-candidate exact metric pages covering all directions/networks/heights."""
    if metrics.empty:
        return
    out_dir = ensure_dir(out_dir)
    d = metrics[metrics["case"].astype(str).isin(list(cases))].copy()
    for (gtype, variable), dg in d.groupby(["generalization_type", "variable"], dropna=False):
        # One physical panel per direction/network/height (ALL_PROFILE included).
        combos = list(dg[["source_site", "target_site", "network", "height_m"]].drop_duplicates().itertuples(index=False, name=None))
        per_page = 6
        for page0 in range(0, len(combos), per_page):
            page = combos[page0:page0 + per_page]
            ncols = 2
            nrows = int(math.ceil(len(page) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12.4, 4.0*nrows), layout="constrained")
            axes_flat = np.atleast_1d(axes).reshape(-1)
            for ax, (source, target, network, height) in zip(axes_flat, page):
                g = dg[
                    dg["source_site"].eq(source)
                    & dg["target_site"].eq(target)
                    & dg["network"].eq(network)
                    & dg["height_m"].astype(str).eq(str(height))
                ]
                vals = []
                labs = []
                raw = numeric(g["Raw_RMSE"]).dropna()
                if str(variable) != "WD" and not raw.empty:
                    vals.append(float(raw.iloc[0])); labs.append("Raw WRF")
                for case in cases:
                    hit = g[g["case"].astype(str).eq(str(case))]
                    if hit.empty:
                        continue
                    v = hit.iloc[0]["MAE"] if str(variable) == "WD" else hit.iloc[0]["RMSE"]
                    if np.isfinite(pd.to_numeric(v, errors="coerce")):
                        vals.append(float(v)); labs.append(case)
                if not vals:
                    ax.set_axis_off(); continue
                x = np.arange(len(vals)); bars = ax.bar(x, vals); label_bars(ax, bars, 1 if str(variable)=="WD" else 2)
                ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, ha="right")
                ax.set_title(f"{source}→{target} | {network} | {height} m")
                ax.set_ylabel("Circular MAE (°)" if str(variable)=="WD" else f"{variable} RMSE")
                ax.grid(axis="y", alpha=GRID_ALPHA)
            for ax in axes_flat[len(page):]:
                ax.set_axis_off()
            fig.suptitle(f"{code} complete exact metrics — {gtype} — {variable} — page {page0//per_page+1}")
            save_figure(fig, out_dir / safe_name(gtype) / safe_name(variable) / f"Fig_{code}_{safe_name(gtype)}_{safe_name(variable)}_page_{page0//per_page+1:02d}")


def generate_complete_sci_library(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    sci_root: Path,
    sci_data: Path,
) -> None:
    """Create an exhaustive SCI-candidate library; no fixed NO1<->NO2 restriction."""
    root = ensure_dir(sci_root / "ALL")
    data_root = ensure_dir(sci_data / "ALL")

    _complete_metric_pages(c1, "Code1", CODE1_CASES, root / "M_C1")
    _complete_metric_pages(c2, "Code2", CODE2_CASES, root / "M_C2")

    # Code1 full cross-tower pseudo-profiles: merge 10-m same-height with all
    # other-height transfer points. This is the missing 10-m fix.
    c1_full = plot_code1_full_cross_tower_pseudo_profiles_all(
        c1_profiles, root / "P_C1_X"
    )
    if not c1_full.empty:
        c1_full.to_csv(data_root / "Code1_full_cross_tower_profiles_all.csv", index=False, encoding="utf-8-sig")

    # Code1 same-tower other-height profiles and Code2 profiles: all towers,
    # all directed transfers, both networks, all cases and all available heights.
    c1_same = c1_profiles[c1_profiles["generalization_type"].astype(str).eq("same_tower_other_height")].copy()
    plot_combined_mean_profiles(c1_same, CODE1_CASES, root / "P_C1_S", "Code1", metrics=c1)
    plot_combined_mean_profiles(c2_profiles, CODE2_CASES, root / "P_C2", "Code2", metrics=c2)

    # Complete availability-view time series and pairwise display statistics.
    plot_all_available_ws_timeseries(
        c1_pred, c2_pred,
        root / "TS",
        data_root / "ALL_time_series_available_metrics.csv",
    )

    # Coverage index makes omissions explicit rather than silently creating blank panels.
    coverage_rows = []
    for code, prof in [("Code1", c1_profiles), ("Code2", c2_profiles)]:
        if prof.empty:
            continue
        keys = ["source_site", "target_site", "network", "case", "generalization_type", "height_m"]
        for key, g in prof.groupby(keys, dropna=False):
            meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            coverage_rows.append({"code": code, **meta, "N": int(pd.to_numeric(g.get("N"), errors="coerce").max()), "profile_present": True})
    pd.DataFrame(coverage_rows).to_csv(data_root / "SCI_COMPLETE_PROFILE_COVERAGE.csv", index=False, encoding="utf-8-sig")

    guide = (
        "SCI COMPLETE LIBRARY\n"
        "====================\n"
        "This folder is exhaustive, unlike the compact 3x3 representative figures.\n"
        "Code1 cross-tower profiles combine cross_tower_same_height (10 m) with\n"
        "cross_tower_other_height (all other available heights).\n"
        "Code2 profiles include every available model height.\n"
        "Time-series figures are availability views: valid points are drawn instead\n"
        "of being removed by unrelated-height gaps. Formal metric tables remain\n"
        "strictly height-specific common-time fair.\n"
        "No empty panel is saved when a combination has no usable data.\n"
    )
    (root / "README_COMPLETE_LIBRARY.txt").write_text(guide, encoding="utf-8")




# =============================================================================
# FINAL V8 SCI / TIME-SERIES OVERRIDES (NO3 REPRESENTATIVE SOURCE)
# =============================================================================
# V8 policy:
#   * Formal TEST/ALL statistics remain HEIGHT_SPECIFIC_COMMON and unchanged.
#   * Final V8 SCI selection uses the ALL-target-time experiment
#     (cross_tower_other_time), because V8 is the all-WRF-period processor.
#   * The representative source is NO3_1071, transferred to NO1/NO2/NT.
#   * Metric panels summarize ALL available V8 transfer experiments; they are
#     not restricted to NO3.
#   * Time-series are display-only availability views: every series is drawn
#     wherever it has valid data, using continuous lines and NO point markers.
#   * Exhaustive full-results TS still include all towers/directions/heights.

SCI_V8_REP_SOURCE_SITE = "NO3_1071"
SCI_V8_DIRECTIONS = [
    ("NO3_1071", "NO1_1159"),
    ("NO3_1071", "NO2_1107"),
    ("NO3_1071", "NOT_1166"),
]
SCI_V8_GTYPE = "cross_tower_other_time"
SCI_V8_TS_HEIGHT_M = 10


def _plot_gap_safe(
    ax: plt.Axes,
    times: pd.Series,
    values: pd.Series,
    *,
    label: str,
    linewidth: float = 1.0,
    linestyle: str = "-",
    marker=None,
    color=None,
    gap_multiplier: float = 1.5,
    **kwargs,
):
    """Line-only availability display.

    Formal statistics are computed elsewhere from strict height-specific common
    timestamps.  Here each series uses all of its own valid points and those
    points are connected directly for a readable display.  No markers are used.
    """
    d = pd.DataFrame({
        "_time": pd.to_datetime(times, errors="coerce"),
        "_value": pd.to_numeric(values, errors="coerce"),
    }).dropna(subset=["_time", "_value"]).sort_values("_time")
    if d.empty:
        return []
    kwargs.pop("markersize", None)
    kwargs.pop("markeredgewidth", None)
    if color is None:
        color = ax._get_lines.get_next_color()
    return ax.plot(
        d["_time"], d["_value"],
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=None,
        color=color,
        **kwargs,
    )


def _merge_code1_timeseries_for_gtype(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
    gtype: str,
) -> pd.DataFrame:
    """Availability-view Code1 TS for any V8 generalization type.

    Cases are outer-merged after hourly aggregation; no formal fair-time clipping
    is applied to display curves.
    """
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    required = {
        "source_site", "target_site", "network", "case", "generalization_type",
        "WS_height_m", "OBS_WS", "WRF_WS", "CORR_WS",
    }
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq(str(gtype))
        & pd.to_numeric(pred["WS_height_m"], errors="coerce").eq(float(height_m))
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty:
        return pd.DataFrame()
    cases_present = [c for c in cases if d["case"].astype(str).eq(str(c)).any()]
    if not cases_present:
        return pd.DataFrame()
    base = d[[TIME_COL, "OBS_WS", "WRF_WS"]].copy()
    case_frames = {
        c: d[d["case"].astype(str).eq(str(c))][[TIME_COL, "CORR_WS"]].copy()
        for c in cases_present
    }
    merged = _complete_available_timeseries_frame(base, case_frames, "OBS_WS", "WRF_WS")
    merged = _sci_apply_time_window(merged)
    value_cols = [c for c in merged.columns if c != TIME_COL]
    if value_cols:
        merged = merged.dropna(subset=value_cols, how="all")
    return merged.sort_values(TIME_COL).reset_index(drop=True)


def _merge_code2_timeseries_for_gtype(
    pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    cases: Sequence[str],
    height_m: int,
    gtype: str,
) -> pd.DataFrame:
    """Availability-view Code2 TS for any V8 generalization type."""
    if pred.empty or TIME_COL not in pred.columns:
        return pd.DataFrame()
    obs_col = f"OBS_WS{height_m}"
    wrf_col = f"WRF_WS{height_m}"
    corr_col = f"CORR_WS{height_m}"
    required = {
        "source_site", "target_site", "network", "case", "generalization_type",
        obs_col, wrf_col, corr_col,
    }
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    d = pred[
        pred["source_site"].astype(str).eq(str(source))
        & pred["target_site"].astype(str).eq(str(target))
        & pred["network"].astype(str).eq(str(network))
        & pred["generalization_type"].astype(str).eq(str(gtype))
        & pred["case"].astype(str).isin(list(cases))
    ].copy()
    if d.empty or not pd.to_numeric(d[obs_col], errors="coerce").notna().any():
        return pd.DataFrame()
    cases_present = [c for c in cases if d["case"].astype(str).eq(str(c)).any()]
    if not cases_present:
        return pd.DataFrame()
    base = d[[TIME_COL, obs_col, wrf_col]].copy()
    case_frames = {
        c: d[d["case"].astype(str).eq(str(c))][[TIME_COL, corr_col]].copy()
        for c in cases_present
    }
    merged = _complete_available_timeseries_frame(base, case_frames, obs_col, wrf_col)
    merged = merged.rename(columns={obs_col: "OBS", wrf_col: "Raw WRF"})
    merged = _sci_apply_time_window(merged)
    value_cols = [c for c in merged.columns if c != TIME_COL]
    if value_cols:
        merged = merged.dropna(subset=value_cols, how="all")
    return merged.sort_values(TIME_COL).reset_index(drop=True)


def _cross_code_timeseries_for_gtype(
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    source: str,
    target: str,
    network: str,
    gtype: str,
) -> pd.DataFrame:
    """Availability-view Code1-vs-Code2 TS; union of hourly timestamps."""
    t1 = _merge_code1_timeseries_for_gtype(
        c1_pred, source, target, network,
        [SCI_CODE1_BEST_WS_CASE], SCI_V8_TS_HEIGHT_M, gtype,
    )
    t2 = _merge_code2_timeseries_for_gtype(
        c2_pred, source, target, network,
        [SCI_CODE2_BEST_WS_CASE], SCI_V8_TS_HEIGHT_M, gtype,
    )
    if t1.empty and t2.empty:
        return pd.DataFrame()
    if not t1.empty:
        t1 = t1.rename(columns={
            "OBS_WS": "OBS_Code1",
            "WRF_WS": "Raw_WRF_Code1",
            SCI_CODE1_BEST_WS_CASE: f"Code1_{SCI_CODE1_BEST_WS_CASE}",
        })
        keep1 = [c for c in [TIME_COL, "OBS_Code1", "Raw_WRF_Code1", f"Code1_{SCI_CODE1_BEST_WS_CASE}"] if c in t1.columns]
        t1 = t1[keep1]
    else:
        t1 = pd.DataFrame(columns=[TIME_COL])
    if not t2.empty:
        t2 = t2.rename(columns={
            "OBS": "OBS_Code2",
            "Raw WRF": "Raw_WRF_Code2",
            SCI_CODE2_BEST_WS_CASE: f"Code2_{SCI_CODE2_BEST_WS_CASE}",
        })
        keep2 = [c for c in [TIME_COL, "OBS_Code2", "Raw_WRF_Code2", f"Code2_{SCI_CODE2_BEST_WS_CASE}"] if c in t2.columns]
        t2 = t2[keep2]
    else:
        t2 = pd.DataFrame(columns=[TIME_COL])
    return t1.merge(t2, on=TIME_COL, how="outer").sort_values(TIME_COL).reset_index(drop=True)


def _save_if_axis_has_data(fig: plt.Figure, ax: plt.Axes, path: Path) -> None:
    if ax.axison and (len(ax.lines) + len(ax.patches) > 0):
        save_figure(fig, path)
    else:
        plt.close(fig)


def _v8_code2_metric_all_transfer_data(c2: pd.DataFrame, variable: str):
    """Code2 V8 metric panel: ALL available all-target-time transfers, by network."""
    d = c2[
        c2["generalization_type"].astype(str).eq(SCI_V8_GTYPE)
        & c2["variable"].astype(str).eq(str(variable))
        & c2["height_m"].astype(str).str.upper().eq("ALL_PROFILE")
        & c2["case"].astype(str).isin(CODE2_CASES)
        & numeric(c2["N"]).fillna(0).gt(0)
    ].copy()
    networks = [n for n in SCI_NETWORKS if d["network"].astype(str).eq(str(n)).any()]
    categories = ["CNN–LSTM" if n == "CNN_LSTM" else str(n) for n in networks]
    names = ([] if variable == "WD" else ["Raw WRF"]) + CODE2_CASES
    means = {n: [] for n in names}
    mins = {n: [] for n in names}
    maxs = {n: [] for n in names}
    rows = []
    for network, label in zip(networks, categories):
        g = d[d["network"].astype(str).eq(str(network))].copy()
        if variable != "WD":
            raw = g[["source_site", "target_site", "network", "Raw_RMSE"]].drop_duplicates()
            rv = numeric(raw["Raw_RMSE"]).dropna()
            means["Raw WRF"].append(float(rv.mean()) if not rv.empty else np.nan)
            mins["Raw WRF"].append(float(rv.min()) if not rv.empty else np.nan)
            maxs["Raw WRF"].append(float(rv.max()) if not rv.empty else np.nan)
            rows.append({"variable": variable, "network": network, "series": "Raw WRF",
                         "scope": "ALL_AVAILABLE_V8_TRANSFERS", "mean": means["Raw WRF"][-1],
                         "min": mins["Raw WRF"][-1], "max": maxs["Raw WRF"][-1], "n_transfers": int(len(rv))})
        for case in CODE2_CASES:
            vals = numeric(g.loc[g["case"].astype(str).eq(case), "primary_error"]).dropna()
            means[case].append(float(vals.mean()) if not vals.empty else np.nan)
            mins[case].append(float(vals.min()) if not vals.empty else np.nan)
            maxs[case].append(float(vals.max()) if not vals.empty else np.nan)
            rows.append({"variable": variable, "network": network, "series": case,
                         "scope": "ALL_AVAILABLE_V8_TRANSFERS", "mean": means[case][-1],
                         "min": mins[case][-1], "max": maxs[case][-1], "n_transfers": int(len(vals))})
    return categories, means, mins, maxs, pd.DataFrame(rows)


def _v8_cross_code_metric_all(cross_100: pd.DataFrame):
    d = cross_100[cross_100["time_scope"].astype(str).eq("all_target_time")].copy()
    categories = ROLE_ORDER
    means, mins, maxs, rows = [], [], [], []
    for role in categories:
        v = numeric(d.loc[d["role"].astype(str).eq(str(role)), "Code2_minus_Code1_RMSE"]).dropna()
        means.append(float(v.mean()) if not v.empty else np.nan)
        mins.append(float(v.min()) if not v.empty else np.nan)
        maxs.append(float(v.max()) if not v.empty else np.nan)
        rows.append({"role": role, "scope": "ALL_V8_10m_DIRECT_COMPARISONS",
                     "mean": means[-1], "min": mins[-1], "max": maxs[-1], "N": int(len(v))})
    return categories, {"Code2 − Code1": means}, {"Code2 − Code1": mins}, {"Code2 − Code1": maxs}, pd.DataFrame(rows)


def _v8_profile_panel(ax, profile, cases, code, source, target, network, title, metrics):
    return _profile_panel(
        ax, profile, cases, code,
        source, target, network, SCI_V8_GTYPE,
        title, metrics=metrics,
    )


def _v8_ts_panel(ax, merged, cases, code, source, target, network, title):
    _timeseries_panel(ax, merged, cases, code, source, target, network, title)


def generate_v8_no3_selected_sci(
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    c1_pred: pd.DataFrame,
    c2_pred: pd.DataFrame,
    c1_profiles: pd.DataFrame,
    c2_profiles: pd.DataFrame,
    cross_100: pd.DataFrame,
    sci_root: Path,
    sci_data: Path,
) -> None:
    """Final selected V8 SCI set: NO3 -> the other three towers, ALL target times."""
    root = ensure_dir(sci_root / "04_NO3_SELECTED_SCI")
    data_root = ensure_dir(sci_data / "04_NO3_SELECTED_SCI")

    # ---------------- G4 / Code1: ALL metrics ----------------
    metric_specs = [
        ("WS", "WS RMSE (m s$^{-1}$)", "Wind speed"),
        ("WD", "Circular WD MAE (°)", "Wind direction"),
        ("TKE", "TKE RMSE (m$^2$ s$^{-2}$)", "TKE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.3), layout="constrained")
    mt = []
    for ax, (variable, ylabel, title) in zip(np.atleast_1d(axes), metric_specs):
        cats, means, mins, maxs, table = _code1_metric_panel_data_for_gtype(c1, variable, SCI_V8_GTYPE)
        _sci_add_grouped_bars(ax, cats, means, mins, maxs, ylabel=ylabel,
                              title=f"{title}: all V8 transfer tasks",
                              decimals=1 if variable == "WD" else 2,
                              show_legend=True, legend_ncol=min(3, len(means)))
        mt.append(table)
    add_panel_labels(np.atleast_1d(axes))
    fig.suptitle("Code1 V8 all-target-time metrics — all towers/directions/heights")
    save_figure(fig, root / "Fig_V8_G4a_Code1_metrics_ALL")
    if mt:
        pd.concat(mt, ignore_index=True).to_csv(data_root / "Fig_V8_G4a_Code1_metrics_ALL.csv", index=False, encoding="utf-8-sig")

    # Code1 NO3 profiles (2 networks x 3 targets)
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(17.8, 10.2), layout="constrained", squeeze=False)
    prof_tabs = []
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            tab = _v8_profile_panel(axes[r, c], c1_profiles, CODE1_CASES, "Code1", source, target, network,
                                    f"{source}→{target}, {network}\nAll valid target times", c1)
            if not tab.empty:
                prof_tabs.append(tab)
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("Code1 V8: NO3-trained generalization to the other three towers")
    save_figure(fig, root / "Fig_V8_G4b_Code1_NO3_profiles_3targets")
    if prof_tabs:
        pd.concat(prof_tabs, ignore_index=True).to_csv(data_root / "Fig_V8_G4b_Code1_NO3_profiles_3targets.csv", index=False, encoding="utf-8-sig")

    # Code1 NO3 10-m TS (2 x 3), availability view, line only
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(18.8, 8.8), layout="constrained", squeeze=False)
    ts_tabs = []
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            ts = _merge_code1_timeseries_for_gtype(c1_pred, source, target, network, CODE1_CASES, SCI_V8_TS_HEIGHT_M, SCI_V8_GTYPE)
            _v8_ts_panel(axes[r, c], ts, CODE1_CASES, "Code1", source, target, network,
                         f"{source}→{target}, {network} | {SCI_V8_TS_HEIGHT_M} m")
            if not ts.empty:
                ts_tabs.append(ts.assign(source_site=source, target_site=target, network=network))
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("Code1 V8: NO3-trained 10-m time series — all available data")
    save_figure(fig, root / "Fig_V8_G4c_Code1_NO3_TS_3targets_10m")
    if ts_tabs:
        pd.concat(ts_tabs, ignore_index=True).to_csv(data_root / "Fig_V8_G4c_Code1_NO3_TS_3targets_10m.csv", index=False, encoding="utf-8-sig")

    # ---------------- G5 / Code2: ALL metrics ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.3), layout="constrained")
    code2_tabs = []
    for ax, (variable, ylabel, title) in zip(np.atleast_1d(axes), metric_specs):
        cats, means, mins, maxs, table = _v8_code2_metric_all_transfer_data(c2, variable)
        _sci_add_grouped_bars(ax, cats, means, mins, maxs, ylabel=ylabel,
                              title=f"{title}: all available V8 transfers",
                              decimals=1 if variable == "WD" else 2,
                              show_legend=True, legend_ncol=min(3, len(means)))
        code2_tabs.append(table)
    add_panel_labels(np.atleast_1d(axes))
    fig.suptitle("Code2 V8 all-target-time metrics — all available transfers")
    save_figure(fig, root / "Fig_V8_G5a_Code2_metrics_ALL")
    if code2_tabs:
        pd.concat(code2_tabs, ignore_index=True).to_csv(data_root / "Fig_V8_G5a_Code2_metrics_ALL.csv", index=False, encoding="utf-8-sig")

    # Code2 profiles
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(17.8, 10.2), layout="constrained", squeeze=False)
    prof_tabs = []
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            tab = _v8_profile_panel(axes[r, c], c2_profiles, CODE2_CASES, "Code2", source, target, network,
                                    f"{source}→{target}, {network}\nAll valid target times", c2)
            if not tab.empty:
                prof_tabs.append(tab)
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("Code2 V8: NO3-trained generalization to the other three towers")
    save_figure(fig, root / "Fig_V8_G5b_Code2_NO3_profiles_3targets")
    if prof_tabs:
        pd.concat(prof_tabs, ignore_index=True).to_csv(data_root / "Fig_V8_G5b_Code2_NO3_profiles_3targets.csv", index=False, encoding="utf-8-sig")

    # Code2 TS
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(18.8, 8.8), layout="constrained", squeeze=False)
    ts_tabs = []
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            ts = _merge_code2_timeseries_for_gtype(c2_pred, source, target, network, CODE2_CASES, SCI_V8_TS_HEIGHT_M, SCI_V8_GTYPE)
            _v8_ts_panel(axes[r, c], ts, CODE2_CASES, "Code2", source, target, network,
                         f"{source}→{target}, {network} | {SCI_V8_TS_HEIGHT_M} m")
            if not ts.empty:
                ts_tabs.append(ts.assign(source_site=source, target_site=target, network=network))
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("Code2 V8: NO3-trained 10-m time series — all available data")
    save_figure(fig, root / "Fig_V8_G5c_Code2_NO3_TS_3targets_10m")
    if ts_tabs:
        pd.concat(ts_tabs, ignore_index=True).to_csv(data_root / "Fig_V8_G5c_Code2_NO3_TS_3targets_10m.csv", index=False, encoding="utf-8-sig")

    # ---------------- G6 / Code1 vs Code2 ----------------
    cross_all = cross_100[cross_100["time_scope"].astype(str).eq("all_target_time")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.3), layout="constrained")
    # RMSE by role across every direct comparison
    cats, means, mins, maxs, _ = _cross_code_role_summary(cross_all, "RMSE")
    _sci_add_grouped_bars(axes[0], cats, means, mins, maxs,
                          ylabel="WS RMSE at 10 m (m s$^{-1}$)",
                          title="All 10-m direct comparisons", decimals=2,
                          show_legend=True, legend_ncol=2)
    cats, means, mins, maxs, _ = _cross_code_role_summary(cross_all, "Skill")
    _sci_add_grouped_bars(axes[1], cats, means, mins, maxs,
                          ylabel="RMSE skill at 10 m (%)", title="All 10-m skill comparisons",
                          decimals=1, show_legend=True, legend_ncol=2)
    cats, means, mins, maxs, delta_tab = _v8_cross_code_metric_all(cross_100)
    _sci_add_grouped_bars(axes[2], cats, means, mins, maxs,
                          ylabel="Code2 − Code1 RMSE (m s$^{-1}$)",
                          title="Code2 relative to Code1", zero_line=True, decimals=2,
                          show_legend=True, legend_ncol=1)
    add_panel_labels(np.atleast_1d(axes))
    fig.suptitle("V8 Code1 versus Code2 — all available 10-m direct comparisons")
    save_figure(fig, root / "Fig_V8_G6a_cross_code_metrics_ALL")
    delta_tab.to_csv(data_root / "Fig_V8_G6a_cross_code_delta_ALL.csv", index=False, encoding="utf-8-sig")

    # Cross-code profiles: 2 x 3
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(17.8, 10.2), layout="constrained", squeeze=False)
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            _cross_code_profile_panel_for_gtype(
                axes[r, c], c1_profiles, c2_profiles,
                source, target, network, SCI_V8_GTYPE,
                f"{source}→{target}, {network}\nAll valid target times",
            )
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("V8 Code1 versus Code2: NO3-trained profiles to the other three towers")
    save_figure(fig, root / "Fig_V8_G6b_cross_code_NO3_profiles_3targets")

    # Cross-code TS: 2 x 3, line only and availability union
    fig, axes = plt.subplots(len(SCI_NETWORKS), 3, figsize=(18.8, 8.8), layout="constrained", squeeze=False)
    for r, network in enumerate(SCI_NETWORKS):
        for c, (source, target) in enumerate(SCI_V8_DIRECTIONS):
            ts = _cross_code_timeseries_for_gtype(c1_pred, c2_pred, source, target, network, SCI_V8_GTYPE)
            _plot_cross_code_time_axis(
                axes[r, c], ts, network,
                f"{source}→{target}, {network} | {SCI_V8_TS_HEIGHT_M} m",
            )
    add_panel_labels(axes.reshape(-1))
    fig.suptitle("V8 Code1 versus Code2: NO3-trained 10-m time series — all available data")
    save_figure(fig, root / "Fig_V8_G6c_cross_code_NO3_TS_3targets_10m")

    guide = f"""FINAL V8 SCI selection\n======================\nRepresentative source: {SCI_V8_REP_SOURCE_SITE}\nTargets: NO1_1159, NO2_1107, NOT_1166\nScope: all valid target times / all WRF period ({SCI_V8_GTYPE})\nRepresentative TS height: {SCI_V8_TS_HEIGHT_M} m\n\nMetric panels use ALL available V8 tasks, not only NO3.\nProfiles use NO3 -> each of the other three towers.\nSCI TS uses the same three directions at 10 m.\nThe exhaustive TS library still contains every available height/direction.\nTS display is continuous line-only and uses each series wherever it has data.\nFormal metrics remain strict HEIGHT_SPECIFIC_COMMON.\n"""
    write_text(root / "README_NO3_SELECTED_SCI.txt", guide)



# =============================================================================
# FINAL OBS LINEAGE AUDIT (matches the checked raw-OBS reader logic)
# =============================================================================

def _obs_ref_detect_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _obs_ref_read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        last_err = None
        for header in range(0, 12):
            try:
                df = pd.read_excel(path, sheet_name=0, header=header, engine=None)
                df.columns = [str(c).strip() for c in df.columns]
                joined = " ".join(df.columns).lower()
                score = sum(k in joined for k in ["rec_time", "time", "时间", "wspd", "wdir", "tp", "rh", "风速", "风向"])
                if score >= 1 and df.shape[1] >= 2:
                    return df
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Failed to read OBS Excel file: {path}; last error: {last_err}")
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        for sep in [None, ",", "\t", ";", r"\s+"]:
            try:
                if sep is None:
                    df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception as exc:
                last_err = exc
    raise RuntimeError(f"Failed to read raw OBS file: {path}; last error: {last_err}")


def _obs_ref_parse_time(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    col = _obs_ref_detect_col(df, [
        "北京时间", "rec_time", "record_time", "Record Time", "time", "Time",
        "datetime", "Datetime", "DateTime", "日期时间", "时间", "Date/Time",
        "Timestamp", "timestamp", "TIMESTAMP", "记录时间",
    ])
    if col is not None:
        return pd.to_datetime(df[col], errors="coerce"), col
    date_col = _obs_ref_detect_col(df, ["日期", "Date", "date"])
    hour_col = _obs_ref_detect_col(df, ["小时", "Hour", "hour", "时"])
    minute_col = _obs_ref_detect_col(df, ["分钟", "Minute", "minute", "分"])
    if date_col is not None:
        text = df[date_col].astype(str)
        if hour_col is not None:
            text = text + " " + df[hour_col].astype(str).str.zfill(2)
            text = text + ((":" + df[minute_col].astype(str).str.zfill(2)) if minute_col is not None else ":00")
        return pd.to_datetime(text, errors="coerce"), date_col
    return pd.Series(pd.NaT, index=df.index), None


def _obs_ref_station_from_path(path: Path) -> str:
    low = str(path).lower()
    for st in ["NO1_1159", "NO2_1107", "NO3_1071", "NOT_1166"]:
        if st.lower() in low:
            return st
    if "no1" in low and "1159" in low:
        return "NO1_1159"
    if "no2" in low and "1107" in low:
        return "NO2_1107"
    if "no3" in low and "1071" in low:
        return "NO3_1071"
    if "not" in low:
        return "NOT_1166"
    if "tower1" in low:
        return "NO1_1159"
    if "tower2" in low:
        return "NO2_1107"
    if "tower3" in low:
        return "NO3_1071"
    if "tower4" in low:
        return "NOT_1166"
    return "unknown"


def _obs_ref_height_from_col(col: str) -> Optional[float]:
    m = re.search(r"(?P<h>\d+(?:\.\d+|p\d+)?)\s*m", str(col), flags=re.I)
    if not m:
        return None
    try:
        return float(m.group("h").replace("p", "."))
    except Exception:
        return None


def _obs_ref_is_ws_col(col: str) -> bool:
    s = str(col).strip()
    low = s.lower()
    if any(k in low for k in ["wrf", "model", "sim", "target_delta", "error", "bias", "u_mo"]):
        return False
    return low.startswith("wspd") or low.startswith("ws") or "风速" in s


def _obs_ref_qc_ws(v: pd.Series) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = x.isna() | x.isin(RAW_OBS_MISSING_FLAGS) | (x.abs() >= RAW_OBS_INVALID_ABS_THRESHOLD)
    x = x.mask(missing)
    lo, hi = RAW_OBS_WS_RANGE
    x = x.mask(x.notna() & ~((x >= lo) & (x <= hi)))
    return x


def _obs_ref_existing_dirs() -> List[Path]:
    out, seen = [], set()
    for p in RAW_OBS_DIR_CANDIDATES:
        p = Path(p)
        if p.exists() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def _obs_ref_hourly_ws(station: str, height_m: float) -> Tuple[pd.DataFrame, List[str]]:
    """Raw-QC hourly OBS WS using the checked OBS-script conventions."""
    pieces: List[pd.DataFrame] = []
    used_files: List[str] = []
    seen_paths = set()
    for root in _obs_ref_existing_dirs():
        for suffix in RAW_OBS_SUFFIXES:
            for path in sorted(root.rglob(f"*{suffix}")):
                if str(path) in seen_paths:
                    continue
                seen_paths.add(str(path))
                low = str(path).lower()
                if "obs_basic_summary" in low or "figures" in low:
                    continue
                if _obs_ref_station_from_path(path) != str(station):
                    continue
                try:
                    df = _obs_ref_read_table_auto(path)
                    times, _ = _obs_ref_parse_time(df)
                    for col in df.columns:
                        if not _obs_ref_is_ws_col(col):
                            continue
                        h = _obs_ref_height_from_col(col)
                        if h is None or not np.isclose(float(h), float(height_m), atol=1.0e-9):
                            continue
                        x = _obs_ref_qc_ws(df[col])
                        tmp = pd.DataFrame({TIME_COL: pd.to_datetime(times, errors="coerce"), "RAW_OBS_WS": x})
                        tmp = tmp.dropna(subset=[TIME_COL, "RAW_OBS_WS"])
                        if not tmp.empty:
                            pieces.append(tmp)
                            used_files.append(str(path))
                except Exception:
                    continue
    if not pieces:
        return pd.DataFrame(columns=[TIME_COL, "RAW_OBS_WS"]), []
    d = pd.concat(pieces, ignore_index=True)
    d["_hour"] = pd.to_datetime(d[TIME_COL], errors="coerce").dt.floor("h")
    d = d.groupby("_hour", as_index=False)["RAW_OBS_WS"].mean().rename(columns={"_hour": TIME_COL})
    return d.sort_values(TIME_COL).reset_index(drop=True), sorted(set(used_files))


def _series_difference_stats(a: pd.DataFrame, a_col: str, b: pd.DataFrame, b_col: str) -> Dict[str, object]:
    if a.empty or b.empty or TIME_COL not in a.columns or TIME_COL not in b.columns:
        return {"N": 0, "MAE": np.nan, "bias_a_minus_b": np.nan, "max_abs_diff": np.nan}
    if a_col not in a.columns or b_col not in b.columns:
        return {"N": 0, "MAE": np.nan, "bias_a_minus_b": np.nan, "max_abs_diff": np.nan}
    aa = a[[TIME_COL, a_col]].copy(); bb = b[[TIME_COL, b_col]].copy()
    aa[TIME_COL] = pd.to_datetime(aa[TIME_COL], errors="coerce")
    bb[TIME_COL] = pd.to_datetime(bb[TIME_COL], errors="coerce")
    aa[a_col] = pd.to_numeric(aa[a_col], errors="coerce")
    bb[b_col] = pd.to_numeric(bb[b_col], errors="coerce")
    aa = aa.dropna().groupby(TIME_COL, as_index=False)[a_col].mean()
    bb = bb.dropna().groupby(TIME_COL, as_index=False)[b_col].mean()
    m = aa.merge(bb, on=TIME_COL, how="inner").dropna()
    if m.empty:
        return {"N": 0, "MAE": np.nan, "bias_a_minus_b": np.nan, "max_abs_diff": np.nan}
    diff = pd.to_numeric(m[a_col], errors="coerce") - pd.to_numeric(m[b_col], errors="coerce")
    return {
        "N": int(diff.notna().sum()),
        "MAE": float(diff.abs().mean()),
        "bias_a_minus_b": float(diff.mean()),
        "max_abs_diff": float(diff.abs().max()),
    }


def build_obs_baseline_audit_v8(c1_pred: pd.DataFrame, c2_pred: pd.DataFrame) -> pd.DataFrame:
    """Audit the NO3->three-target V8 all-target-time 10-m OBS lineages."""
    rows: List[Dict[str, object]] = []
    raw_cache: Dict[Tuple[str, float], Tuple[pd.DataFrame, List[str]]] = {}
    for source, target in SCI_V8_DIRECTIONS:
        raw_key = (str(target), float(SCI_V8_TS_HEIGHT_M))
        if raw_key not in raw_cache:
            raw_cache[raw_key] = _obs_ref_hourly_ws(target, SCI_V8_TS_HEIGHT_M)
        raw, raw_files = raw_cache[raw_key]
        for network in SCI_NETWORKS:
            t1 = _merge_code1_timeseries_for_gtype(
                c1_pred, source, target, network,
                [SCI_CODE1_BEST_WS_CASE], SCI_V8_TS_HEIGHT_M, SCI_V8_GTYPE,
            )
            t2 = _merge_code2_timeseries_for_gtype(
                c2_pred, source, target, network,
                [SCI_CODE2_BEST_WS_CASE], SCI_V8_TS_HEIGHT_M, SCI_V8_GTYPE,
            )
            c12 = _series_difference_stats(t1, "OBS_WS", t2, "OBS")
            w12 = _series_difference_stats(t1, "WRF_WS", t2, "Raw WRF")
            r1 = _series_difference_stats(raw, "RAW_OBS_WS", t1, "OBS_WS")
            r2 = _series_difference_stats(raw, "RAW_OBS_WS", t2, "OBS")
            rows.append({
                "time_scope": "all_target_time",
                "generalization_type": SCI_V8_GTYPE,
                "source_site": source,
                "target_site": target,
                "network": network,
                "height_m": SCI_V8_TS_HEIGHT_M,
                "Code1_OBS_N": int(pd.to_numeric(t1.get("OBS_WS", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not t1.empty else 0,
                "Code2_OBS_N": int(pd.to_numeric(t2.get("OBS", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not t2.empty else 0,
                "Code1_vs_Code2_OBS_common_N": c12["N"],
                "Code1_vs_Code2_OBS_MAE": c12["MAE"],
                "Code1_minus_Code2_OBS_bias": c12["bias_a_minus_b"],
                "Code1_vs_Code2_OBS_max_abs_diff": c12["max_abs_diff"],
                "Code1_vs_Code2_WRF_common_N": w12["N"],
                "Code1_vs_Code2_WRF_MAE": w12["MAE"],
                "Code1_vs_Code2_WRF_max_abs_diff": w12["max_abs_diff"],
                "raw_OBS_found": bool(not raw.empty),
                "raw_OBS_hourly_N": int(len(raw)),
                "raw_OBS_source_files": " | ".join(raw_files),
                "raw_vs_Code1_common_N": r1["N"],
                "raw_vs_Code1_MAE": r1["MAE"],
                "raw_vs_Code1_max_abs_diff": r1["max_abs_diff"],
                "raw_vs_Code2_common_N": r2["N"],
                "raw_vs_Code2_MAE": r2["MAE"],
                "raw_vs_Code2_max_abs_diff": r2["max_abs_diff"],
                "cross_code_canonical_baseline": CROSS_CODE_CANONICAL_BASELINE,
                "policy_note": (
                    "V8 cross-code panels/formal direct comparisons use one Code2-derived hourly OBS/WRF baseline; "
                    "Code1 fallback only if Code2 is structurally unavailable. Raw OBS is audit-only."
                ),
            })
    return pd.DataFrame(rows)

# =============================================================================
# FINAL FIX: robust cross-code time-series plotting for partially available V8 data
# =============================================================================
# Why this override is needed:
# _cross_code_timeseries_for_gtype() intentionally uses an OUTER union of Code1 and
# Code2 timestamps so that every available series can be displayed.  Therefore, for
# some direction/network/gtype combinations, the returned frame can legitimately
# contain only Code1 columns or only Code2 columns.  The older plotting helper
# indexed OBS_Code1/OBS_Code2 unconditionally and also averaged the two baselines,
# which caused KeyError (e.g. missing OBS_Code2) and mixed two preprocessing
# lineages into one displayed OBS curve.
#
# Final policy:
#   1) Never require both codes to exist just to draw the panel.
#   2) Use ONE baseline lineage only: prefer Code2 (original-10min-derived hourly)
#      when both its OBS and Raw WRF are available; otherwise fall back as a pair to
#      Code1.  Do NOT average/combine/fill between Code1 and Code2 baselines.
#   3) Plot Code1 C8 and Code2 C8d independently whenever each is available.
#   4) Display is line-only and uses all available values; formal metrics remain
#      controlled by the strict fair-time tables elsewhere in this script.

def _plot_cross_code_time_axis(
    ax: plt.Axes,
    ts: pd.DataFrame,
    network: str,
    title: str,
) -> None:
    if ts is None or ts.empty or TIME_COL not in ts.columns:
        ax.set_axis_off()
        return

    def _series(name: str) -> pd.Series:
        if name in ts.columns:
            return pd.to_numeric(ts[name], errors="coerce")
        return pd.Series(np.nan, index=ts.index, dtype=float)

    obs1 = _series("OBS_Code1")
    wrf1 = _series("Raw_WRF_Code1")
    obs2 = _series("OBS_Code2")
    wrf2 = _series("Raw_WRF_Code2")

    # Select a whole OBS/WRF baseline pair from one lineage; never splice lineages.
    if (CROSS_CODE_CANONICAL_BASELINE == "Code2"
            and obs2.notna().any() and wrf2.notna().any()):
        obs = obs2
        wrf = wrf2
        baseline_source = "Code2"
    elif obs1.notna().any() and wrf1.notna().any():
        obs = obs1
        wrf = wrf1
        baseline_source = "Code1"
    else:
        # If neither code has a complete baseline pair, keep whatever individual
        # baseline series exists, but still never combine values across codes.
        if obs2.notna().any():
            obs = obs2
            baseline_source = "Code2_partial"
        else:
            obs = obs1
            baseline_source = "Code1_partial"
        if baseline_source.startswith("Code2") and wrf2.notna().any():
            wrf = wrf2
        elif baseline_source.startswith("Code1") and wrf1.notna().any():
            wrf = wrf1
        else:
            wrf = pd.Series(np.nan, index=ts.index, dtype=float)

    plotted = False
    if obs.notna().any():
        _plot_gap_safe(
            ax, ts[TIME_COL], obs,
            label="OBS", linewidth=SCI_LINEWIDTH_OBS,
        )
        plotted = True
    if wrf.notna().any():
        _plot_gap_safe(
            ax, ts[TIME_COL], wrf,
            label="Raw WRF", linestyle="--", linewidth=SCI_LINEWIDTH_WRF,
        )
        plotted = True

    c1c = f"Code1_{SCI_CODE1_BEST_WS_CASE}"
    c2c = f"Code2_{SCI_CODE2_BEST_WS_CASE}"
    if c1c in ts.columns and pd.to_numeric(ts[c1c], errors="coerce").notna().any():
        _plot_gap_safe(
            ax, ts[TIME_COL], ts[c1c],
            label=f"Code1 {SCI_CODE1_BEST_WS_CASE}",
            linewidth=SCI_LINEWIDTH_CASE,
        )
        plotted = True
    if c2c in ts.columns and pd.to_numeric(ts[c2c], errors="coerce").notna().any():
        _plot_gap_safe(
            ax, ts[TIME_COL], ts[c2c],
            label=f"Code2 {SCI_CODE2_BEST_WS_CASE}",
            linestyle=":", linewidth=SCI_LINEWIDTH_CASE,
        )
        plotted = True

    if not plotted:
        ax.set_axis_off()
        return

    h = globals().get("SCI_V8_TS_HEIGHT_M", globals().get("SCI_TS_HEIGHT_M", 10))
    ax.set_xlabel("Time")
    ax.set_ylabel(f"WS at {h} m (m s$^{{-1}}$)")
    ax.set_title(title)
    configure_time_axis(ax)
    ax.legend(frameon=False, fontsize=7.0, ncol=2)
    # Small internal attribute for debugging/audit; does not alter the figure.
    setattr(ax, "_xlh_baseline_source", baseline_source)


def main() -> None:
    global CODE1_GENERALIZATION_ROOT, CODE2_GENERALIZATION_ROOT, OUTPUT_ROOT, PNG_DPI, CURVE_OVERLAP_RECORDS, FAIR_TIME_INDEX
    CURVE_OVERLAP_RECORDS = []
    args = parse_args()
    CODE1_GENERALIZATION_ROOT = args.code1_root
    CODE2_GENERALIZATION_ROOT = args.code2_root
    OUTPUT_ROOT = args.output_root
    PNG_DPI = args.dpi

    out_audit = ensure_dir(OUTPUT_ROOT / "00_audit")

    # Layer 1: comprehensive result library.
    full_root = ensure_dir(OUTPUT_ROOT / "10_FULL_RESULTS")
    out_tables = ensure_dir(full_root / "00_tables")
    out_c1 = ensure_dir(full_root / "01_Code1_all")
    out_c2 = ensure_dir(full_root / "02_Code2_all")
    out_cross = ensure_dir(full_root / "03_cross_code_all")
    out_timeseries = ensure_dir(full_root / "04_all_selected_timeseries")

    # Layer 2: three manuscript-oriented figures and their source data.
    sci_root = ensure_dir(OUTPUT_ROOT / "90_SCI_MAIN_FIGURES")
    sci_data = ensure_dir(OUTPUT_ROOT / "91_SCI_FIGURE_DATA")

    bundle = ensure_dir(OUTPUT_ROOT / "FOR_ANALYSIS")

    # Keep the source metric files for audit only.  All statistics and plots are
    # recomputed from prediction tables after fair-time alignment.
    c1_source_metrics = load_metrics(CODE1_GENERALIZATION_ROOT, "Code1")
    c2_source_metrics = load_metrics(CODE2_GENERALIZATION_ROOT, "Code2")

    c1_pred = load_predictions(CODE1_GENERALIZATION_ROOT, "Code1")
    c2_pred = load_predictions(CODE2_GENERALIZATION_ROOT, "Code2")
    # Memory-safe fair-time recomputation: process one direction/variable at a
    # time and aggregate native Code2 10-min rows to hourly BEFORE long expansion.
    c1_raw, c2, c1_profiles, c2_profiles, fair_time_audit, FAIR_TIME_INDEX = \
        build_fair_products_memory_safe(c1_pred, c2_pred)
    fair_time_audit.to_csv(out_audit / "fair_time_alignment_audit.csv", index=False, encoding="utf-8-sig")

    # Independent V8 OBS-lineage audit for the NO3-selected all-target-time 10-m
    # panels.  Raw tower OBS is compared when reachable but is never silently
    # substituted into the saved model outputs.
    obs_baseline_audit = build_obs_baseline_audit_v8(c1_pred, c2_pred)
    obs_baseline_audit.to_csv(out_audit / "OBS_BASELINE_LINEAGE_AUDIT_V8.csv", index=False, encoding="utf-8-sig")
    obs_baseline_audit.to_csv(sci_data / "OBS_BASELINE_LINEAGE_AUDIT_V8.csv", index=False, encoding="utf-8-sig")
    obs_baseline_audit.to_csv(bundle / "OBS_BASELINE_LINEAGE_AUDIT_V8.csv", index=False, encoding="utf-8-sig")

    _write_global_time_audit_files(out_audit, sci_data, bundle)
    if c1_raw.empty and c2.empty:
        raise RuntimeError(
            "DIRECTION-COMMON TIME produced no valid fair metrics. "
            "Inspect 00_audit/TIME_RANGE_AUDIT.txt and global_time_series_audit.csv."
        )

    # All formal metrics below use height-specific common hourly calendars;
    # unrelated heights no longer delete valid samples at the current height.
    c1, c1_duplicates = deduplicate_code1_metrics(c1_raw)

    c1_coverage = validate_case_coverage(c1, CODE1_CASES, "Code1")
    c2_coverage = validate_case_coverage(c2, CODE2_CASES, "Code2")
    coverage = pd.concat([c1_coverage, c2_coverage], ignore_index=True)

    c1_summary = grouped_summary(
        c1,
        ["code", "generalization_type", "case", "role", "variable"],
    )
    c2_all_profile = c2[c2["height_m"].astype(str).str.upper().eq("ALL_PROFILE")].copy()
    c2_summary_all = grouped_summary(
        c2_all_profile,
        ["code", "generalization_type", "case", "role", "variable"],
    )
    c2_summary_height = grouped_summary(
        c2[c2["height_num"].notna()],
        ["code", "generalization_type", "case", "role", "variable", "height_m"],
    )

    ranks = pd.concat([rank_cases(c1), rank_cases(c2)], ignore_index=True)
    c1_effects = paired_ablation_effects(c1, CODE1_ABLATIONS)
    c2_effects = paired_ablation_effects(c2, CODE2_ABLATIONS)
    effects = pd.concat([c1_effects, c2_effects], ignore_index=True)
    effect_summary = summarize_ablation_effects(effects)

    # Profiles were computed during the same memory-safe aligned pass.

    cross_100 = build_cross_code_10m_table(c1, c2)
    key_metrics = build_key_metrics_bundle(c1, c2)

    # Full tables
    c1_source_metrics.to_csv(out_tables / "Code1_metrics_SOURCE_ORIGINAL.csv", index=False, encoding="utf-8-sig")
    c2_source_metrics.to_csv(out_tables / "Code2_metrics_SOURCE_ORIGINAL.csv", index=False, encoding="utf-8-sig")
    fair_time_audit.to_csv(out_audit / "fair_time_alignment_audit.csv", index=False, encoding="utf-8-sig")
    fair_time_audit.to_csv(sci_data / "fair_time_alignment_audit.csv", index=False, encoding="utf-8-sig")
    _write_global_time_audit_files(out_audit, sci_data, bundle)
    c1_raw.to_csv(out_tables / "Code1_metrics_raw.csv", index=False, encoding="utf-8-sig")
    c1.to_csv(out_tables / "Code1_metrics_clean_deduplicated.csv", index=False, encoding="utf-8-sig")
    c1_duplicates.to_csv(out_audit / "Code1_repeated_native_variable_rows.csv", index=False, encoding="utf-8-sig")
    c2.to_csv(out_tables / "Code2_metrics_clean.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_audit / "case_coverage.csv", index=False, encoding="utf-8-sig")
    c1_summary.to_csv(out_tables / "Code1_summary_by_generalization_case_variable.csv", index=False, encoding="utf-8-sig")
    c2_summary_all.to_csv(out_tables / "Code2_ALL_PROFILE_summary_by_case_variable.csv", index=False, encoding="utf-8-sig")
    c2_summary_height.to_csv(out_tables / "Code2_heightwise_summary_by_case_variable.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(out_tables / "case_ranking_within_exact_tasks.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(out_tables / "paired_physics_effects_vs_C2.csv", index=False, encoding="utf-8-sig")
    effect_summary.to_csv(out_tables / "paired_physics_effects_summary.csv", index=False, encoding="utf-8-sig")
    cross_100.to_csv(
        out_tables / "cross_code_10m_WS_comparison_both_time_scopes.csv",
        index=False, encoding="utf-8-sig",
    )
    c1_profiles.to_csv(out_tables / "Code1_pseudo_profile_means_all_cases.csv", index=False, encoding="utf-8-sig")
    c2_profiles.to_csv(out_tables / "Code2_profile_means_all_cases.csv", index=False, encoding="utf-8-sig")

    # Figures
    for gtype in c1["generalization_type"].dropna().unique():
        plot_case_summary_bars(c1, CODE1_CASES, CODE1_ROLE, out_c1 / "case_bars", "Code1", gtype)
    plot_positive_skill_fraction(c1_summary, out_c1 / "skill_robustness", "Code1", CODE1_CASES)
    plot_code1_ws_rmse_by_height(c1, out_c1 / "WS_RMSE_by_height")
    plot_combined_mean_profiles(c1_profiles, CODE1_CASES, out_c1 / "mean_profiles", "Code1", metrics=c1)

    for gtype in c2["generalization_type"].dropna().unique():
        plot_case_summary_bars(
            c2, CODE2_CASES, CODE2_ROLE,
            out_c2 / "case_bars", "Code2",
            generalization_type=gtype,
            all_profile_only=True,
        )
    plot_positive_skill_fraction(c2_summary_all, out_c2 / "skill_robustness", "Code2", CODE2_CASES)
    plot_code2_ws_rmse_profiles(c2, out_c2 / "WS_RMSE_profiles")
    plot_code2_exact_run_all_profile_ws(c2, out_c2 / "exact_run_case_ranking")
    plot_combined_mean_profiles(c2_profiles, CODE2_CASES, out_c2 / "mean_profiles", "Code2", metrics=c2)

    plot_cross_code_10m(cross_100, out_cross)
    plot_code1_cross_tower_10m_timeseries(c1_pred, out_timeseries / "Code1")
    plot_code2_cross_tower_10m_timeseries(c2_pred, out_timeseries / "Code2")

    # SCI manuscript figures: G1-G3 for source-test times and optional
    # parallel G4-G6 candidates for all valid target times.
    plot_sci_figure_code1(
        c1, c1_pred, c1_profiles, sci_root, sci_data
    )
    plot_sci_figure_code2(
        c2, c2_pred, c2_profiles, sci_root, sci_data
    )
    cross_100_same = cross_100[
        cross_100["time_scope"].eq("same_time")
    ].copy()
    plot_sci_figure_cross_code(
        c1_pred, c2_pred, c1_profiles, c2_profiles,
        cross_100_same, sci_root, sci_data
    )
    if GENERATE_ALL_TARGET_TIME_SCI_CANDIDATES:
        plot_sci_all_target_time_candidates(
            c1, c2, c1_pred, c2_pred,
            c1_profiles, c2_profiles, cross_100,
            sci_root, sci_data,
        )
        # Final selected V8 SCI set: NO3 source -> NO1/NO2/NT.
        generate_v8_no3_selected_sci(
            c1, c2, c1_pred, c2_pred,
            c1_profiles, c2_profiles, cross_100,
            sci_root, sci_data,
        )
        if GENERATE_WRF_PERIOD_DISTRIBUTION_DIAGNOSTICS:
            plot_sci_wrf_period_distribution_diagnostics(
                c1_pred, c2_pred, sci_root, sci_data
            )
            _make_direct_windrose_composites(
                c1_pred, c2_pred, sci_root
            )
    if GENERATE_BEST_MODEL_SUMMARIES:
        generate_best_model_summaries(
            c1, c2, c1_pred, c2_pred,
            c1_profiles, c2_profiles,
            sci_root, sci_data,
        )
    write_sci_figure_guide(sci_root)


    # FINAL COMPLETE-COVERAGE outputs -------------------------------------------------
    # In addition to the compact representative G1/G2/G3 figures, generate an
    # exhaustive library covering all towers, all directed transfers, both networks,
    # all cases and all available heights. Time-series figures draw every available
    # point and never save intentionally blank panels.
    complete_ts_metrics = plot_all_available_ws_timeseries(
        c1_pred, c2_pred,
        ensure_dir(full_root / "05_TS_ALL"),
        out_tables / "ALL_TIMESERIES_AVAILABLE_DISPLAY_METRICS.csv",
    )
    c1_full_profile_table = plot_code1_full_cross_tower_pseudo_profiles_all(
        c1_profiles, ensure_dir(out_c1 / "FULL_X_PROFILE_10m")
    )
    if not c1_full_profile_table.empty:
        c1_full_profile_table.to_csv(
            out_tables / "Code1_FULL_cross_tower_pseudo_profiles_with_10m.csv",
            index=False, encoding="utf-8-sig",
        )
    generate_complete_sci_library(
        c1, c2, c1_pred, c2_pred, c1_profiles, c2_profiles,
        sci_root, sci_data,
    )

    overlap_audit = pd.DataFrame(CURVE_OVERLAP_RECORDS)
    if not overlap_audit.empty:
        overlap_audit = overlap_audit.drop_duplicates().sort_values(
            [
                "plot_family", "code", "source_site", "target_site",
                "network", "generalization_type", "case_a", "case_b"
            ],
            na_position="last",
        )
    overlap_audit.to_csv(
        out_audit / "curve_overlap_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact package for later analysis/sharing
    key_metrics.to_csv(bundle / "generalization_key_metrics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(bundle / "case_coverage.csv", index=False, encoding="utf-8-sig")
    overlap_audit.to_csv(bundle / "curve_overlap_audit.csv", index=False, encoding="utf-8-sig")
    fair_time_audit.to_csv(bundle / "fair_time_alignment_audit.csv", index=False, encoding="utf-8-sig")
    c1_summary.to_csv(bundle / "Code1_summary.csv", index=False, encoding="utf-8-sig")
    c2_summary_all.to_csv(bundle / "Code2_ALL_PROFILE_summary.csv", index=False, encoding="utf-8-sig")
    c2_summary_height.to_csv(bundle / "Code2_heightwise_summary.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(bundle / "case_ranking.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(bundle / "paired_physics_effects_vs_C2.csv", index=False, encoding="utf-8-sig")
    effect_summary.to_csv(bundle / "paired_physics_effects_summary.csv", index=False, encoding="utf-8-sig")
    cross_100.to_csv(bundle / "cross_code_10m_WS_comparison.csv", index=False, encoding="utf-8-sig")
    c1_profiles.to_csv(bundle / "Code1_pseudo_profile_means.csv", index=False, encoding="utf-8-sig")
    c2_profiles.to_csv(bundle / "Code2_profile_means.csv", index=False, encoding="utf-8-sig")
    copy_audit_files(CODE1_GENERALIZATION_ROOT, bundle / "source_run_audits", "Code1")
    copy_audit_files(CODE2_GENERALIZATION_ROOT, bundle / "source_run_audits", "Code2")

    # Human-readable report
    n_missing = int((~coverage["present"]).sum()) if not coverage.empty else 0
    c1_fail = CODE1_GENERALIZATION_ROOT / "00_audit" / "failures.csv"
    c2_fail = CODE2_GENERALIZATION_ROOT / "00_audit" / "failures.csv"
    report = [
        "Xilinhaote generalization comparison report",
        "=" * 84,
        f"Code1 root: {CODE1_GENERALIZATION_ROOT}",
        f"Code2 root: {CODE2_GENERALIZATION_ROOT}",
        f"Output root: {OUTPUT_ROOT}",
        f"Comprehensive results: {full_root}",
        f"SCI main figures: {sci_root}",
        f"SCI figure data: {sci_data}",
        "",
        f"Code1 source metric rows (before fair-time recomputation): {len(c1_source_metrics)}",
        f"Code2 source metric rows (before fair-time recomputation): {len(c2_source_metrics)}",
        f"Fair-time groups: {len(fair_time_audit)}",
        "FAIR direction-variable common-time groups: " + "; ".join(
            f"{r.scope} {r.source_site}->{r.target_site} {r.variable}: "
            f"N={int(r.N_global_common)}, {r.time_start} -> {r.time_end}"
            for r in GLOBAL_TIME_SCOPE_AUDIT.itertuples(index=False)
        ) if not GLOBAL_TIME_SCOPE_AUDIT.empty else "FAIR common-time groups: none",
        f"Fair-time groups with zero common samples: {int((pd.to_numeric(fair_time_audit.get('N_common', pd.Series(dtype=float)), errors='coerce').fillna(0) == 0).sum()) if not fair_time_audit.empty else 0}",
        f"Code1 fair-time metric rows before deduplication: {len(c1_raw)}",
        f"Code1 cleaned metric rows: {len(c1)}",
        f"Code1 repeated native-variable rows reported: {len(c1_duplicates)}",
        f"Code2 metric rows: {len(c2)}",
        f"Missing required case groups: {n_missing}",
        f"Paired physics-vs-C2 comparisons: {len(effects)}",
        f"Cross-code 10 m comparisons: {len(cross_100)}",
        f"Cross-code canonical OBS/WRF baseline: {CROSS_CODE_CANONICAL_BASELINE} (single lineage; no averaging/splicing)",
        f"V8 raw OBS reference found for audit rows: {int(obs_baseline_audit['raw_OBS_found'].sum()) if not obs_baseline_audit.empty else 0}/{len(obs_baseline_audit)}",
        f"Numerically identical curve pairs detected: {int(overlap_audit['numerically_identical'].sum()) if not overlap_audit.empty else 0}",
        f"Code1 source failure file exists: {c1_fail.exists()}",
        f"Code2 source failure file exists: {c2_fail.exists()}",
        "",
        "Use FOR_ANALYSIS/ as the compact package for later interpretation.",
        "Do not rely only on mean errors; also inspect positive-skill fraction,",
        "worst-case error, paired effects and the exact-task case ranking.",
    ]
    write_text(out_audit / "RUN_REPORT.txt", "\n".join(report))
    write_text(bundle / "README.txt", "\n".join(report))

    manifest = {
        "code1_root": str(CODE1_GENERALIZATION_ROOT),
        "code2_root": str(CODE2_GENERALIZATION_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "code1_cases": CODE1_CASES,
        "code2_cases": CODE2_CASES,
        "reference_ws_height_m": REFERENCE_WS_HEIGHT_M,
        "full_results_root": str(full_root),
        "sci_main_figures_root": str(sci_root),
        "sci_figure_data_root": str(sci_data),
        "sci_networks": SCI_NETWORKS,
        "sci_representative_direction": [
            SCI_TS_SOURCE_SITE, SCI_TS_TARGET_SITE
        ],
        "v8_final_sci_representative_source": SCI_V8_REP_SOURCE_SITE,
        "v8_final_sci_representative_directions": [list(x) for x in SCI_V8_DIRECTIONS],
        "v8_final_sci_scope": "all_target_time",
        "important_files_for_analysis": [
            "generalization_key_metrics.csv",
            "Code1_summary.csv",
            "Code2_ALL_PROFILE_summary.csv",
            "Code2_heightwise_summary.csv",
            "case_ranking.csv",
            "paired_physics_effects_vs_C2.csv",
            "paired_physics_effects_summary.csv",
            "cross_code_10m_WS_comparison.csv",
            "Code1_pseudo_profile_means.csv",
            "Code2_profile_means.csv",
            "case_coverage.csv",
            "curve_overlap_audit.csv",
            "fair_time_alignment_audit.csv",
            "TIME_RANGE_AUDIT.txt",
            "global_time_scope_audit.csv",
            "global_time_series_audit.csv",
            "GLOBAL_COMMON_TIMESTAMPS_TEST.csv",
            "GLOBAL_COMMON_TIMESTAMPS_ALL.csv",
        ],
    }
    with open(bundle / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("Finished Xilinhaote generalization comparison")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Comprehensive result library: {full_root}")
    print(f"SCI main figures: {sci_root}")
    print(f"SCI figure data: {sci_data}")
    print(f"Compact analysis package: {bundle}")
    print(f"Code1 raw/clean rows: {len(c1_raw)}/{len(c1)}")
    print(f"Code2 rows: {len(c2)}")
    print(f"Missing required case groups: {n_missing}")
    print(f"Paired effects: {len(effects)}")
    print(f"Cross-code 10 m rows: {len(cross_100)}")
    if not GLOBAL_TIME_SCOPE_AUDIT.empty:
        for _r in GLOBAL_TIME_SCOPE_AUDIT.itertuples(index=False):
            print(
                f"COMMON {_r.scope} {_r.source_site}->{_r.target_site} {_r.variable}: "
                f"N={int(_r.N_global_common)}, {_r.time_start} -> {_r.time_end}"
            )
    print("=" * 100)


if __name__ == "__main__":
    main()
