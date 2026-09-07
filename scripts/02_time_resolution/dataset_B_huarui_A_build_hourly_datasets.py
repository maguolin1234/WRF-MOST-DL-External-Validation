#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 + Stage 2: time-resolution audit and fair 10-min / 1-hour ML-ready datasets
===================================================================================

Purpose
-------
1) Stage 1 audits the actual time resolution of each ML_ready_ref10_*.csv file:
   - full time-axis interval;
   - effective interval of each OBS / WRF / MOST / PBL variable;
   - non-missing sample counts and value-change counts.

2) Stage 2 builds two fair datasets from the same ML-ready source:
   - D10_10min: original file copied unchanged;
   - D1H_1hour: hourly aggregated file, with wind direction handled by vector averaging.

Recommended downstream use
--------------------------
For a fair 24-hour-history comparison:
    D10_10min  -> WINDOW_SIZE = 144   # 144 x 10 min = 24 h
    D1H_1hour  -> WINDOW_SIZE = 24    # 24 x 1 h    = 24 h

This script does not modify the original ML-ready files.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# User configuration
# =============================================================================

TIME_COL = "北京时间"

# Select one or more cases to run. Use ["huarui_A"] or ["huarui_C"] if you only want one case.
RUN_CASES = ["huarui_A", "huarui_C"]

# IMPORTANT:
# These default roots follow the WRF postprocess scripts on the Linux server.
# If you run this audit script on Windows after copying the data, replace each
# ml_ready_root/output_root with your local D:\... path.
CASE_CONFIGS = {
    "huarui_A": {
        "ml_ready_root": r"D:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\huarui_A_two_towers_full",
        "output_root": r"D:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\time_resolution_sensitivity_huarui_A",
        # huarui_A includes C039801_d03 and C039802_d02, so do not restrict to *_d03.csv.
        "glob": "**/ML_ready_ref10_*.csv",
        # huarui_A validation wind source can be hourly; one valid obs sample per hour should be accepted.
        "min_obs_records_per_hour": 1,
        "min_10min_records_per_hour": 3,
    },
    "huarui_C": {
        "ml_ready_root": r"D:\wake\WRF_RANS_wake\huarui\C\bias_correction\1km\input_data\data_post_new\huarui_C_three_towers_full",
        "output_root": r"D:\wake\WRF_RANS_wake\huarui\C\bias_correction\1km\input_data\data_post_new\time_resolution_sensitivity_huarui_C",
        # huarui_C 1-km regular towers are d03; using all ML_ready files is safer and also harmless.
        "glob": "**/ML_ready_ref10_*.csv",
        "min_obs_records_per_hour": 3,
        "min_10min_records_per_hour": 3,
    },
}

RUN_STAGE1_AUDIT = True
RUN_STAGE2_BUILD_DATASETS = True

# Hourly aggregation settings. Case-specific min sample thresholds are set in run_one_case().
HOURLY_RULE = "1H"
MIN_10MIN_RECORDS_PER_HOUR = 3       # keep hour if at least this many source rows exist
MIN_OBS_RECORDS_PER_HOUR = 3         # obs_valid_flag_h = 1 if >= this many valid obs samples
TIME_STAMP_FOR_HOURLY = "hour_start" # "hour_start" or "hour_end"

# Treat very tiny numerical differences as unchanged when estimating effective variable resolution.
CHANGE_TOL_DEFAULT = 1e-10
CHANGE_TOL_BY_KEYWORD = {
    "WD": 1e-6,
    "WS": 1e-8,
    "U": 1e-8,
    "V": 1e-8,
    "TKE": 1e-10,
    "T2": 1e-8,
    "TEMP": 1e-8,
    "PBLH": 1e-8,
    "UST": 1e-10,
    "HFX": 1e-8,
    "PSFC": 1e-6,
    "Z0": 1e-12,
    "L": 1e-8,
}

# =============================================================================
# Small utilities
# =============================================================================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str, max_len: int = 180) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(text))[:max_len]


def read_ml_ready(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{csv_path} lacks required time column: {TIME_COL}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    for c in df.columns:
        if c != TIME_COL and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def timedel_to_seconds(x) -> Optional[float]:
    if pd.isna(x):
        return None
    try:
        return float(pd.Timedelta(x).total_seconds())
    except Exception:
        return None


def seconds_to_label(sec: Optional[float]) -> str:
    if sec is None or not np.isfinite(sec):
        return ""
    sec = float(sec)
    if abs(sec - 600.0) < 1e-6:
        return "10min"
    if abs(sec - 1800.0) < 1e-6:
        return "30min"
    if abs(sec - 3600.0) < 1e-6:
        return "1h"
    if sec < 3600:
        return f"{sec/60:.3g}min"
    if sec < 86400:
        return f"{sec/3600:.3g}h"
    return f"{sec/86400:.3g}d"


def dominant_interval_seconds(times: pd.Series) -> Tuple[Optional[float], str, int, int]:
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    dt = t.diff().dropna()
    if len(dt) == 0:
        return None, "", 0, int(len(t))
    vc = dt.value_counts()
    dom_td = vc.index[0]
    dom_count = int(vc.iloc[0])
    sec = timedel_to_seconds(dom_td)
    return sec, str(dom_td), dom_count, int(len(dt))


def classify_column(col: str) -> str:
    c = str(col)
    if c == TIME_COL:
        return "time"
    if c.startswith("OBS_"):
        return "OBS"
    if c.startswith("target_delta"):
        return "target_residual"
    if c in {"hour_sin", "hour_cos", "month_sin", "month_cos"}:
        return "time_feature"
    if c.startswith("U_MO") or c.startswith("MO_minus_WRF") or c.startswith("MOST"):
        return "MOST_feature"
    if c in {"WRF_L (m)", "WRF_PBLH (m)", "WRF_Bulk_Ri", "UST (m/s)", "Z0 (m)", "HFX (W/m2)", "PSFC (Pa)", "T2 (K)"}:
        return "WRF_wrfout_surface_PBL"
    if c.startswith("WRF_"):
        if "interp" in c or any(k in c for k in ["WS", "WD", "U", "V", "TKE", "TEMP", "T10"]):
            return "WRF_tslist_or_interpolated_profile"
        return "WRF_other"
    if c.startswith("obs_valid_flag"):
        return "OBS_flag"
    return "other"


def change_tolerance(col: str) -> float:
    upper = col.upper()
    for key, tol in CHANGE_TOL_BY_KEYWORD.items():
        if key in upper:
            return float(tol)
    return float(CHANGE_TOL_DEFAULT)


def audit_time_axis(csv_path: Path, df: pd.DataFrame, rel_path: str) -> Dict[str, object]:
    t = pd.to_datetime(df[TIME_COL], errors="coerce").dropna().sort_values()
    dt = t.diff().dropna()
    dom_sec, dom_str, dom_count, n_dt = dominant_interval_seconds(t)
    row = {
        "file": str(csv_path),
        "relative_file": rel_path,
        "n_rows": int(len(df)),
        "n_valid_times": int(len(t)),
        "n_duplicate_times": int(t.duplicated().sum()),
        "time_start": str(t.min()) if len(t) else "",
        "time_end": str(t.max()) if len(t) else "",
        "median_dt": str(dt.median()) if len(dt) else "",
        "min_dt": str(dt.min()) if len(dt) else "",
        "max_dt": str(dt.max()) if len(dt) else "",
        "dominant_dt": dom_str,
        "dominant_dt_seconds": dom_sec,
        "dominant_dt_label": seconds_to_label(dom_sec),
        "dominant_dt_count": dom_count,
        "n_dt": n_dt,
        "dominant_dt_percent": 100.0 * dom_count / n_dt if n_dt else np.nan,
    }
    return row


def audit_variable_resolution(csv_path: Path, df: pd.DataFrame, rel_path: str) -> pd.DataFrame:
    rows = []
    t_all = pd.to_datetime(df[TIME_COL], errors="coerce")
    for col in df.columns:
        if col == TIME_COL:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.notna() & t_all.notna()
        tv = t_all[valid].sort_values()
        nonmissing_sec, nonmissing_str, nonmissing_dom_count, nonmissing_n_dt = dominant_interval_seconds(tv)

        # Effective value-change interval. This is useful for detecting hourly wrfout variables
        # that have been nearest-merged onto a 10-min time axis.
        d = s[valid].sort_index().diff().abs()
        tol = change_tolerance(col)
        changed_mask = d > tol
        changed_times = t_all[valid][changed_mask].dropna().sort_values()
        change_sec, change_str, change_dom_count, change_n_dt = dominant_interval_seconds(changed_times)

        rows.append({
            "file": str(csv_path),
            "relative_file": rel_path,
            "column": col,
            "column_class": classify_column(col),
            "nonmissing_n": int(valid.sum()),
            "nonmissing_dominant_dt": nonmissing_str,
            "nonmissing_dominant_dt_seconds": nonmissing_sec,
            "nonmissing_dominant_dt_label": seconds_to_label(nonmissing_sec),
            "nonmissing_dominant_dt_percent": 100.0 * nonmissing_dom_count / nonmissing_n_dt if nonmissing_n_dt else np.nan,
            "value_change_n": int(changed_mask.sum()),
            "value_change_dominant_dt": change_str,
            "value_change_dominant_dt_seconds": change_sec,
            "value_change_dominant_dt_label": seconds_to_label(change_sec),
            "value_change_dominant_dt_percent": 100.0 * change_dom_count / change_n_dt if change_n_dt else np.nan,
            "change_tolerance": tol,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Wind parsing and hourly aggregation
# =============================================================================


def height_label_to_float(label: str) -> float:
    return float(str(label).replace("p", "."))


def height_float_to_label(h: float) -> str:
    if abs(float(h) - round(float(h))) < 1e-6:
        return str(int(round(float(h))))
    return (f"{float(h):.3f}".rstrip("0").rstrip(".")).replace(".", "p")


def parse_wind_column(col: str) -> Optional[Tuple[str, str, str]]:
    """Return (source, variable, height_label) for OBS/WRF wind-like columns."""
    c = str(col)
    # Examples:
    # OBS_WS10 (m/s), OBS_WD70 (deg), WRF_WS10_interp (m/s), WRF_U100_interp (m/s)
    m = re.match(r"^(OBS|WRF)_(WS|WD|U|V)(\d+(?:p\d+)?)(?:_|\s|\(|$)", c)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def find_col(cols: Iterable[str], prefix: str) -> Optional[str]:
    for c in cols:
        if str(c).startswith(prefix):
            return c
    return None


def ws_wd_to_uv(ws, wd_deg):
    ws = np.asarray(ws, dtype=float)
    wd_rad = np.deg2rad(np.asarray(wd_deg, dtype=float))
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v


def uv_to_ws_wd(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ws = np.sqrt(u ** 2 + v ** 2)
    wd = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0
    return ws, wd


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[TIME_COL], errors="coerce")
    hour = t.dt.hour.astype(float) + t.dt.minute.astype(float) / 60.0
    month = t.dt.month.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    return out


def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    df["__hour"] = df[TIME_COL].dt.floor(HOURLY_RULE)
    group_sizes = df.groupby("__hour").size().rename("__n_records")
    good_hours = group_sizes[group_sizes >= MIN_10MIN_RECORDS_PER_HOUR].index
    df = df[df["__hour"].isin(good_hours)].copy()

    numeric_cols = [c for c in df.columns if c not in {TIME_COL, "__hour"}]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Start with simple means for all numeric columns, then overwrite wind-direction-sensitive fields.
    hourly = df.groupby("__hour", as_index=False)[numeric_cols].mean(numeric_only=True)
    hourly = hourly.rename(columns={"__hour": TIME_COL})

    if TIME_STAMP_FOR_HOURLY == "hour_end":
        hourly[TIME_COL] = pd.to_datetime(hourly[TIME_COL]) + pd.Timedelta(hours=1)

    cols = list(df.columns)
    families: Dict[Tuple[str, str], Dict[str, str]] = {}
    for col in cols:
        parsed = parse_wind_column(col)
        if not parsed:
            continue
        source, var, h_label = parsed
        families.setdefault((source, h_label), {})[var] = col

    hourly_indexed = hourly.set_index(TIME_COL)
    for (source, h_label), fmap in families.items():
        ws_col = fmap.get("WS")
        wd_col = fmap.get("WD")
        u_col = fmap.get("U")
        v_col = fmap.get("V")

        # Compute hourly vector mean from U/V if present; otherwise from WS/WD.
        vec_rows = []
        for hour, g in df.groupby("__hour"):
            u_mean = np.nan
            v_mean = np.nan
            if u_col in g.columns and v_col in g.columns:
                u = pd.to_numeric(g[u_col], errors="coerce")
                v = pd.to_numeric(g[v_col], errors="coerce")
                m = u.notna() & v.notna()
                if m.sum() > 0:
                    u_mean = float(u[m].mean())
                    v_mean = float(v[m].mean())
            elif ws_col in g.columns and wd_col in g.columns:
                ws = pd.to_numeric(g[ws_col], errors="coerce")
                wd = pd.to_numeric(g[wd_col], errors="coerce")
                m = ws.notna() & wd.notna()
                if m.sum() > 0:
                    u, v = ws_wd_to_uv(ws[m].values, wd[m].values)
                    u_mean = float(np.nanmean(u))
                    v_mean = float(np.nanmean(v))
            if np.isfinite(u_mean) and np.isfinite(v_mean):
                ws_mean, wd_mean = uv_to_ws_wd(u_mean, v_mean)
                vec_rows.append((hour, float(ws_mean), float(wd_mean), u_mean, v_mean))

        if not vec_rows:
            continue
        vec = pd.DataFrame(vec_rows, columns=[TIME_COL, "__ws", "__wd", "__u", "__v"]).set_index(TIME_COL)
        if ws_col in hourly_indexed.columns:
            hourly_indexed.loc[vec.index, ws_col] = vec["__ws"]
        if wd_col in hourly_indexed.columns:
            hourly_indexed.loc[vec.index, wd_col] = vec["__wd"]
        if u_col in hourly_indexed.columns:
            hourly_indexed.loc[vec.index, u_col] = vec["__u"]
        if v_col in hourly_indexed.columns:
            hourly_indexed.loc[vec.index, v_col] = vec["__v"]

        # Recompute existing sin/cos columns if present.
        sin_col = f"{source}_WD{h_label}_sin"
        cos_col = f"{source}_WD{h_label}_cos"
        if wd_col in hourly_indexed.columns:
            wd = pd.to_numeric(hourly_indexed[wd_col], errors="coerce")
            if sin_col in hourly_indexed.columns:
                hourly_indexed[sin_col] = np.sin(np.deg2rad(wd))
            if cos_col in hourly_indexed.columns:
                hourly_indexed[cos_col] = np.cos(np.deg2rad(wd))

    hourly = hourly_indexed.reset_index()
    hourly = add_cyclic_time_features(hourly)

    # Recompute residual and MOST-minus-WRF columns after aggregation.
    h_labels = sorted({h for (_src, h) in families.keys()}, key=lambda x: height_label_to_float(x))
    for h_label in h_labels:
        obs_ws = find_col(hourly.columns, f"OBS_WS{h_label}")
        wrf_ws = find_col(hourly.columns, f"WRF_WS{h_label}_") or find_col(hourly.columns, f"WRF_WS{h_label} ")
        if obs_ws is not None and wrf_ws is not None:
            residual = pd.to_numeric(hourly[obs_ws], errors="coerce") - pd.to_numeric(hourly[wrf_ws], errors="coerce")
            for c in [
                f"target_delta_WS{h_label}_obs_minus_wrf (m/s)",
                f"target_delta_U{h_label}_obs_minus_wrf (m/s)",  # legacy name kept for compatibility
            ]:
                if c in hourly.columns or h_label == "10":
                    hourly[c] = residual

        u_mo = find_col(hourly.columns, f"U_MO{h_label}")
        if u_mo is not None and wrf_ws is not None:
            hourly[f"MO_minus_WRF{h_label} (m/s)"] = pd.to_numeric(hourly[u_mo], errors="coerce") - pd.to_numeric(hourly[wrf_ws], errors="coerce")

        # obs_valid_flag_h based on valid OBS wind-speed samples in each hour.
        if obs_ws is not None:
            tmp = df.groupby("__hour")[obs_ws].apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum() >= MIN_OBS_RECORDS_PER_HOUR))
            col_flag = f"obs_valid_flag_{h_label}"
            flag_map = tmp.to_dict()
            hourly[col_flag] = pd.to_datetime(hourly[TIME_COL]).dt.floor(HOURLY_RULE).map(flag_map).fillna(0).astype(int)
            if h_label == "10":
                hourly["obs_valid_flag"] = hourly[col_flag]

    # Keep a clean column order: time first, then original order plus newly created columns.
    original_order = [c for c in df.columns if c not in {TIME_COL, "__hour"}]
    final_cols = [TIME_COL] + [c for c in original_order if c in hourly.columns]
    final_cols += [c for c in hourly.columns if c not in final_cols]
    return hourly[final_cols].sort_values(TIME_COL).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================


def run_one_case(case_name: str, cfg: Dict[str, object]) -> None:
    """Run Stage 1/2 audit and dataset build for one case."""
    global MIN_10MIN_RECORDS_PER_HOUR, MIN_OBS_RECORDS_PER_HOUR

    root = Path(str(cfg["ml_ready_root"]))
    glob_pattern = str(cfg.get("glob", "**/ML_ready_ref10_*.csv"))
    out_root = ensure_dir(Path(str(cfg["output_root"])))

    MIN_10MIN_RECORDS_PER_HOUR = int(cfg.get("min_10min_records_per_hour", MIN_10MIN_RECORDS_PER_HOUR))
    MIN_OBS_RECORDS_PER_HOUR = int(cfg.get("min_obs_records_per_hour", MIN_OBS_RECORDS_PER_HOUR))

    files = sorted(root.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"[{case_name}] No files found under {root} with glob {glob_pattern}")

    audit_root = ensure_dir(out_root / "00_audit")
    d10_root = ensure_dir(out_root / "D10_10min")
    d1h_root = ensure_dir(out_root / "D1H_1hour")

    axis_rows = []
    var_audit_frames = []
    build_rows = []

    print("=" * 100)
    print(f"Running case: {case_name}")
    print(f"ML-ready root: {root}")
    print(f"Glob pattern : {glob_pattern}")
    print(f"Output root  : {out_root}")
    print(f"Hourly keep  : MIN_10MIN_RECORDS_PER_HOUR={MIN_10MIN_RECORDS_PER_HOUR}; "
          f"MIN_OBS_RECORDS_PER_HOUR={MIN_OBS_RECORDS_PER_HOUR}")

    for csv_path in files:
        rel = str(csv_path.relative_to(root))
        print("=" * 100)
        print(f"[{case_name}] Processing: {csv_path}")
        df = read_ml_ready(csv_path)

        if RUN_STAGE1_AUDIT:
            axis_rows.append(audit_time_axis(csv_path, df, rel))
            var_audit_frames.append(audit_variable_resolution(csv_path, df, rel))

        if RUN_STAGE2_BUILD_DATASETS:
            d10_path = d10_root / rel
            d1h_path = d1h_root / rel
            ensure_dir(d10_path.parent)
            ensure_dir(d1h_path.parent)
            shutil.copy2(csv_path, d10_path)

            df_h = build_hourly_dataset(df)
            df_h.to_csv(d1h_path, index=False, encoding="utf-8-sig")
            build_rows.append({
                "case_name": case_name,
                "source_file": str(csv_path),
                "relative_file": rel,
                "D10_file": str(d10_path),
                "D1H_file": str(d1h_path),
                "source_rows": int(len(df)),
                "D1H_rows": int(len(df_h)),
                "min_10min_records_per_hour": MIN_10MIN_RECORDS_PER_HOUR,
                "min_obs_records_per_hour": MIN_OBS_RECORDS_PER_HOUR,
                "hourly_time_stamp": TIME_STAMP_FOR_HOURLY,
            })

    if RUN_STAGE1_AUDIT:
        axis_df = pd.DataFrame(axis_rows)
        axis_df.to_csv(audit_root / "01_time_axis_resolution_by_file.csv", index=False, encoding="utf-8-sig")
        if var_audit_frames:
            var_df = pd.concat(var_audit_frames, ignore_index=True)
            var_df.to_csv(audit_root / "02_variable_effective_resolution_by_file.csv", index=False, encoding="utf-8-sig")
            summary = (
                var_df.groupby(["column_class", "nonmissing_dominant_dt_label", "value_change_dominant_dt_label"], dropna=False)
                .size().reset_index(name="n_columns")
                .sort_values(["column_class", "n_columns"], ascending=[True, False])
            )
            summary.to_csv(audit_root / "03_variable_resolution_summary.csv", index=False, encoding="utf-8-sig")

    if RUN_STAGE2_BUILD_DATASETS:
        build_df = pd.DataFrame(build_rows)
        build_df.to_csv(audit_root / "04_dataset_build_manifest.csv", index=False, encoding="utf-8-sig")

    config = {
        "case_name": case_name,
        "ML_READY_ROOT": str(root),
        "ML_READY_GLOB": glob_pattern,
        "OUTPUT_ROOT": str(out_root),
        "D10_ROOT": str(d10_root),
        "D1H_ROOT": str(d1h_root),
        "recommended_downstream_window_size": {
            "D10_10min": 144,
            "D1H_1hour": 24,
        },
        "hourly_aggregation": {
            "rule": HOURLY_RULE,
            "min_10min_records_per_hour": MIN_10MIN_RECORDS_PER_HOUR,
            "min_obs_records_per_hour": MIN_OBS_RECORDS_PER_HOUR,
            "wind_direction_method": "vector average through U/V; direct angle mean is not used",
            "time_stamp_for_hourly": TIME_STAMP_FOR_HOURLY,
        },
    }
    with open(audit_root / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"[{case_name}] Stage 1/2 finished.")
    print(f"[{case_name}] Audit outputs: {audit_root}")
    print(f"[{case_name}] D10 dataset : {d10_root}")
    print(f"[{case_name}] D1H dataset : {d1h_root}")
    print(f"[{case_name}] Recommended: D10 WINDOW_SIZE=144; D1H WINDOW_SIZE=24 for equal 24-hour history.")


def main() -> None:
    for case_name in RUN_CASES:
        if case_name not in CASE_CONFIGS:
            raise KeyError(f"Unknown case_name={case_name}; available={list(CASE_CONFIGS)}")
        run_one_case(case_name, CASE_CONFIGS[case_name])

if __name__ == "__main__":
    main()
