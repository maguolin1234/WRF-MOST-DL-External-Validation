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

# Edit this to your current ML-ready root.
ML_READY_ROOT = Path(r"D:\wake\WRF_RANS_wake\xilinhaote\result\bias_correction\1km\inputdata\data_post_new\four_sites_full")
ML_READY_GLOB = "**/ML_ready_ref10_*_d03.csv"

# Output root. The script creates:
#   OUTPUT_ROOT / 00_audit
#   OUTPUT_ROOT / D10_10min
#   OUTPUT_ROOT / D1H_1hour
OUTPUT_ROOT = Path(r"D:\wake\WRF_RANS_wake\xilinhaote\result\bias_correction\1km\inputdata\data_post_new\most_diag_v7_filtered")

RUN_STAGE1_AUDIT = False
RUN_STAGE2_BUILD_DATASETS = False

# Hourly aggregation settings.
HOURLY_RULE = "1H"
MIN_10MIN_RECORDS_PER_HOUR = 3       # keep hour if at least 3 ten-min rows exist
MIN_OBS_RECORDS_PER_HOUR = 3         # obs_valid_flag_h = 1 if >= this many valid obs samples
TIME_STAMP_FOR_HOURLY = "hour_start" # "hour_start" or "hour_end"

# MOST diagnostic settings. These diagnostics check whether MOST-derived wind speed
# is actually closer to OBS wind speed than raw WRF wind speed. They are written
# under OUTPUT_ROOT / 00_audit and do not modify the ML-ready source files.
RUN_MOST_WS_DIAGNOSTICS = False
MOST_DIAG_MIN_SAMPLES = 20
MOST_DIAG_ABSDIFF_TOL = 1.0       # m/s, used for residual-consistency gate diagnostics
MOST_DIAG_MIN_ABS_OBS_RESID = 0.25 # m/s, ignore tiny OBS residuals for sign agreement
MOST_DIAG_SAVE_TIMESERIES = True
MOST_DIAG_SAVE_PLOTS = True

# MOST temperature diagnostic / augmentation settings.
# We use the MOST scalar-profile relation as the most promising next physical
# constraint candidate beyond wind speed. T_MO is diagnosed from T2, UST, HFX,
# PSFC, Z0 and L and then compared against OBS_T / WRF_T at each height.
RUN_MOST_TEMP_DIAGNOSTICS = False
MOST_TEMP_DIAG_MIN_SAMPLES = 20
MOST_TEMP_DIAG_SAVE_TIMESERIES = True
MOST_TEMP_DIAG_SAVE_PLOTS = True
MOST_TEMP_REF_HEIGHT_M = 2.0
MOST_TEMP_MIN_Z0 = 1.0e-4
MOST_TEMP_D_FACTOR = 0.65
MOST_TEMP_VALID_RANGE_K = (180.0, 380.0)

# Diagnostic metric plot settings.
RUN_METRIC_PLOTS = False
METRIC_PLOT_DPI = 180

# If True, the exported D10_10min file is written from the augmented in-memory
# table (including T_MO and MO_T_minus_WRF columns when available) instead of a
# byte-for-byte copy of the source CSV. Set to False if you want the old behavior.
D10_WRITE_AUGMENTED_DIAGNOSTIC_COLUMNS = True

# Generic MOST-related variable visualization. These plots cover both MOST-derived
# profile variables and the surface / stability drivers used to build MOST.
RUN_MOST_RELATED_VARIABLE_PLOTS = False
MOST_RELATED_PREFIXES = ['U_MO', 'MO_minus_WRF', 'T_MO', 'MO_T_minus_WRF', 'MOST']
MOST_RELATED_SCALAR_COLS = ['WRF_L (m)', 'UST (m/s)', 'Z0 (m)', 'HFX (W/m2)', 'PSFC (Pa)', 'T2 (K)']
# Physical / diagnostic plausibility filters used before mean-statistics plots.
MOST_RELATED_VALID_RANGES = {
    'U_MO': (0.0, 60.0),
    'MOST_WS': (0.0, 60.0),
    'MO_minus_WRF': (-40.0, 40.0),
    'T_MO': (180.0, 380.0),
    'MO_T_minus_WRF': (-40.0, 40.0),
    'WRF_L (m)': (-1.0e6, 1.0e6),
    'UST (m/s)': (0.0, 5.0),
    'Z0 (m)': (0.0, 10.0),
    'HFX (W/m2)': (-1000.0, 1500.0),
    'PSFC (Pa)': (50000.0, 120000.0),
    'T2 (K)': (180.0, 380.0),
}


# Gryning / Batchvarova extended wind-profile inversion settings.
# This diagnostic estimates tower-level aerodynamic roughness length z0 and
# friction velocity u_star from the available multi-height wind-speed profile.
# At each height/time, OBS wind speed is used first; WRF wind speed is used only
# as a fallback where OBS is missing or invalid.
RUN_GRYNING_PROFILE_INVERSION = True
GRYNING_DATASET_VERSIONS = ("D10_10min_original",)   # optionally add "D1H_1hour_aggregated"
GRYNING_MIN_HEIGHTS_TO_FIT = 2
GRYNING_MIN_WIND_SPEED = 0.05
GRYNING_MAX_WIND_SPEED = 75.0
GRYNING_Z0_MIN = 1.0e-4
GRYNING_Z0_MAX = 10.0
GRYNING_Z0_GRID_N = 180
GRYNING_USTAR_MIN = 0.001
GRYNING_USTAR_MAX = 10.0
GRYNING_ITER_MAX = 8
GRYNING_ITER_TOL = 1.0e-4

# Runtime control. Full 10-min annual files are expensive because each timestamp
# performs a z0/u* inversion. Use stride=6 for hourly diagnostics first.
# Full-resolution run: use every row/time step. Increase temporarily for quick tests.
GRYNING_ROW_STRIDE = 1
GRYNING_MAX_ROWS_PER_FILE = None   # e.g. 5000 for a quick test; None = no limit
GRYNING_PROGRESS_EVERY_ROWS = 2000

# Robust spike filtering for Gryning inversion/profile diagnostics.
# These filters are deliberately diagnostic: raw source files are not modified.
# They remove obvious spikes before inversion and before computing/plotting profile metrics.
GRYNING_USE_ROBUST_FILTERS = True
GRYNING_FILTER_INPUTS_BEFORE_INVERSION = True
GRYNING_FILTER_PROFILE_POINTS_BEFORE_METRICS = True
GRYNING_FILTER_OUTPUT_PARAMETERS_FOR_PLOTS = True
GRYNING_SPIKE_ROLLING_WINDOW = 9
GRYNING_SPIKE_HAMPEL_NSIGMA = 8.0
GRYNING_WS_METRIC_RANGE = (0.0, 45.0)
GRYNING_TEMP_METRIC_RANGE_K = (230.0, 330.0)
GRYNING_WS_JUMP_THRESHOLD = 8.0       # m/s between adjacent diagnostic samples
GRYNING_TEMP_JUMP_THRESHOLD_K = 10.0  # K between adjacent diagnostic samples
GRYNING_USTAR_JUMP_THRESHOLD = 1.0    # m/s between adjacent diagnostic samples
GRYNING_LOG10_Z0_JUMP_THRESHOLD = 1.0 # one order of magnitude between adjacent samples
GRYNING_PROFILE_RMSE_MAX_MPS = 8.0
GRYNING_T_PROFILE_RMSE_MAX_K = 10.0

# Use Gryning et al. (2007) Eq. (8)/(15)/(21) and L_MBL parameterization Eq. (31).
GRYNING_USE_LMBL_EQ31 = True
GRYNING_LMBL_FALLBACK_M = 150.0
GRYNING_USE_PBLH_ZI = True
GRYNING_PBLH_FALLBACK_M = 1000.0
GRYNING_D_FACTOR = 0.0       # d = factor * z0; 0 follows the original flat homogeneous terrain formula
GRYNING_NEUTRAL_ABS_L_THRESHOLD_M = 500.0
GRYNING_MIN_ABS_L_FOR_STABILITY_M = 10.0
GRYNING_STABLE_BETA = 4.7    # stable surface-layer coefficient b used in Gryning et al./Hogstrom form

# Coriolis parameter for Eq. (31). If a station-specific latitude cannot be inferred,
# this default is appropriate for the Xilinhaote / Inner Mongolia towers (~44 deg N).
GRYNING_LATITUDE_DEG_DEFAULT = 44.16
GRYNING_SITE_LATITUDE_DEG = {
    "NO1_1159": 44.124,
    "NO2_1107": 44.193,
    "NO3_1071": 44.16375,
    # NOT latitude is not encoded in the current file names; use the domain default unless edited here.
}

# Output and plotting.
GRYNING_SAVE_PROFILE_POINTS = True
GRYNING_SAVE_PARAMETER_PLOTS = True
GRYNING_SAVE_PROFILE_PLOTS = True
GRYNING_SAVE_METRIC_PLOTS = True
GRYNING_PLOT_DPI = 180

# Temperature-profile diagnostics. There is no direct "Gryning temperature" equivalent
# as established as the wind-profile Eq. (8)/(15)/(21), so the script diagnoses scalar
# profiles with the MOST heat-profile relation. A second T_MO_GRYNING uses the inverted
# z0/u_star in place of WRF Z0/UST while keeping WRF T2/HFX/PSFC/L.
GRYNING_RUN_T_PROFILE_DIAGNOSTICS = True
GRYNING_TEMP_REF_HEIGHT_M = 2.0
GRYNING_TEMP_Z0H_OVER_Z0M = 0.10
GRYNING_TEMP_MIN_Z0H = 1.0e-5
GRYNING_TEMP_VALID_RANGE_K = (180.0, 380.0)

# Inverted heat-profile diagnostics. These fit MOST scalar-temperature parameters
# from the available multi-height T profile, analogous to the wind-profile z0/u*
# inversion. The fitted parameters are z0h and theta_star; HFX_inv is then
# diagnosed as -rho * cp * u_star * theta_star, using inverted wind u_star when
# available and WRF UST as fallback.
GRYNING_TEMP_FIT_INVHEAT = True
GRYNING_TEMP_MIN_HEIGHTS_TO_FIT = 2
GRYNING_Z0H_MIN = 1.0e-6
GRYNING_Z0H_MAX = 2.0
GRYNING_Z0H_GRID_N = 160
GRYNING_THETA_STAR_MIN_K = -10.0
GRYNING_THETA_STAR_MAX_K = 10.0

# Physical constants used by MOST scalar-profile diagnostics.
KAPPA = 0.40
RD_AIR = 287.05
CP_AIR = 1004.67
G = 9.81

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



# =============================================================================
# MOST scalar-temperature utilities
# =============================================================================


def parse_temp_column(col: str) -> Optional[Tuple[str, str]]:
    """Return (source, height_label) for OBS/WRF temperature-like columns."""
    c = str(col)
    m = re.match(r'^(OBS|WRF)_(?:T|TEMP)(\d+(?:p\d+)?)(?:_|\s|\(|$)', c)
    if m:
        return m.group(1), m.group(2)
    return None


def psi_h_businger(zeta: np.ndarray) -> np.ndarray:
    """MOST scalar stability correction psi_h using Businger-Dyer form."""
    zeta = np.asarray(zeta, dtype=float)
    psi = np.zeros_like(zeta, dtype=float)
    stable = zeta > 0
    unstable = zeta < 0
    psi[stable] = -5.0 * zeta[stable]
    if np.any(unstable):
        x = np.maximum(1.0 - 16.0 * zeta[unstable], 1e-6) ** 0.25
        psi[unstable] = 2.0 * np.log((1.0 + x * x) / 2.0)
    return psi


def _first_existing(cols: Iterable[str], prefixes: List[str]) -> Optional[str]:
    cols = list(cols)
    for p in prefixes:
        c = find_col(cols, p)
        if c is not None:
            return c
    return None


def _surface_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _most_temperature_from_surface(
    z: float,
    t2_k: np.ndarray,
    ustar: np.ndarray,
    z0: np.ndarray,
    L: np.ndarray,
    hfx: np.ndarray,
    psfc: np.ndarray,
    z_ref: float = MOST_TEMP_REF_HEIGHT_M,
) -> np.ndarray:
    """Compute T_MO(z) from T2 using the MOST scalar-profile relation."""
    t2 = np.asarray(t2_k, dtype=float)
    ust = np.asarray(ustar, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    L = np.asarray(L, dtype=float)
    hfx = np.asarray(hfx, dtype=float)
    psfc = np.asarray(psfc, dtype=float)

    z0 = np.maximum(z0, MOST_TEMP_MIN_Z0)
    d = MOST_TEMP_D_FACTOR * z0
    zeff = np.maximum(float(z) - d, z0 * 1.01)
    zref_eff = np.maximum(float(z_ref) - d, z0 * 1.01)

    rho = psfc / (RD_AIR * np.maximum(t2, 150.0))
    theta_star = np.full_like(t2, np.nan, dtype=float)
    valid_flux = np.isfinite(hfx) & np.isfinite(rho) & np.isfinite(ust) & (np.abs(ust) > 1e-6)
    theta_star[valid_flux] = -hfx[valid_flux] / (rho[valid_flux] * CP_AIR * ust[valid_flux])

    neutral = (~np.isfinite(L)) | (np.abs(L) < 1e-6)
    zeta = np.where(neutral, 0.0, zeff / L)
    zeta_ref = np.where(neutral, 0.0, zref_eff / L)
    psi = psi_h_businger(zeta) - psi_h_businger(zeta_ref)
    prof = np.log(zeff / zref_eff) - psi

    t_mo = t2 + (theta_star / KAPPA) * prof
    tmin, tmax = MOST_TEMP_VALID_RANGE_K
    t_mo = np.where(np.isfinite(t_mo) & (t_mo >= tmin) & (t_mo <= tmax), t_mo, np.nan)
    return t_mo


def add_most_temperature_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add T_MO{h} and MO_T_minus_WRF{h} columns when the required surface vars exist."""
    if df is None or df.empty:
        return df
    out = df.copy()
    cols = list(out.columns)
    t2_col = _surface_col(out, ['T2 (K)', 'T2'])
    ust_col = _surface_col(out, ['UST (m/s)', 'UST'])
    z0_col = _surface_col(out, ['Z0 (m)', 'Z0'])
    L_col = _surface_col(out, ['WRF_L (m)', 'L (m)', 'WRF_L'])
    hfx_col = _surface_col(out, ['HFX (W/m2)', 'HFX'])
    psfc_col = _surface_col(out, ['PSFC (Pa)', 'PSFC'])
    if any(c is None for c in [t2_col, ust_col, z0_col, L_col, hfx_col, psfc_col]):
        return out

    height_labels = set()
    for c in cols:
        p = parse_temp_column(c)
        if p:
            height_labels.add(p[1])
        h = parse_height_from_prefixed_column(c, ('T_MO', 'MO_T_minus_WRF'))
        if h is not None:
            height_labels.add(h)
    for h_label in sorted(height_labels, key=lambda x: height_label_to_float(x)):
        try:
            z = height_label_to_float(h_label)
        except Exception:
            continue
        t_mo_col = f'T_MO{h_label} (K)'
        out[t_mo_col] = _most_temperature_from_surface(
            z=z,
            t2_k=pd.to_numeric(out[t2_col], errors='coerce').values,
            ustar=pd.to_numeric(out[ust_col], errors='coerce').values,
            z0=pd.to_numeric(out[z0_col], errors='coerce').values,
            L=pd.to_numeric(out[L_col], errors='coerce').values,
            hfx=pd.to_numeric(out[hfx_col], errors='coerce').values,
            psfc=pd.to_numeric(out[psfc_col], errors='coerce').values,
            z_ref=MOST_TEMP_REF_HEIGHT_M,
        )
        wrf_t_col = _first_existing(cols, [
            f'WRF_T{h_label}_', f'WRF_T{h_label} ', f'WRF_T{h_label}',
            f'WRF_TEMP{h_label}_', f'WRF_TEMP{h_label} ', f'WRF_TEMP{h_label}',
        ])
        if wrf_t_col is not None:
            out[f'MO_T_minus_WRF{h_label} (K)'] = pd.to_numeric(out[t_mo_col], errors='coerce') - pd.to_numeric(out[wrf_t_col], errors='coerce')
    return out


# =============================================================================
# MOST wind-speed diagnostic utilities
# =============================================================================


def parse_height_from_prefixed_column(col: str, prefixes: Tuple[str, ...]) -> Optional[str]:
    """Return height label from columns such as U_MO70 (m/s) or MO_minus_WRF70."""
    c = str(col)
    for prefix in prefixes:
        m = re.match(rf"^{re.escape(prefix)}(\d+(?:p\d+)?)(?:_|\s|\(|$)", c)
        if m:
            return m.group(1)
    return None


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    if np.nanstd(x[m]) <= 0.0 or np.nanstd(y[m]) <= 0.0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _basic_error_metrics(obs: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    err = pred - obs
    m = np.isfinite(obs) & np.isfinite(pred)
    if int(m.sum()) == 0:
        return {
            f"{prefix}_N": 0,
            f"{prefix}_RMSE": np.nan,
            f"{prefix}_MAE": np.nan,
            f"{prefix}_MBE_pred_minus_obs": np.nan,
            f"{prefix}_Pearson_r": np.nan,
        }
    e = err[m]
    return {
        f"{prefix}_N": int(m.sum()),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(e ** 2))),
        f"{prefix}_MAE": float(np.mean(np.abs(e))),
        f"{prefix}_MBE_pred_minus_obs": float(np.mean(e)),
        f"{prefix}_Pearson_r": safe_pearson(obs[m], pred[m]),
    }


def _skill_pct(reference_error: float, candidate_error: float) -> float:
    if not np.isfinite(reference_error) or not np.isfinite(candidate_error) or abs(reference_error) < 1e-12:
        return np.nan
    return float(100.0 * (reference_error - candidate_error) / reference_error)


def _format_height_for_output(h_label: str) -> float:
    try:
        return height_label_to_float(h_label)
    except Exception:
        return np.nan


def _plot_most_ws_diagnostic(ts: pd.DataFrame, out_png: Path, title: str) -> bool:
    """Save OBS/WRF/MOST wind-speed and residual comparison plot. Failure is non-fatal."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip MOST plot {out_png.name}: {exc}")
        return False

    try:
        t = pd.to_datetime(ts[TIME_COL], errors="coerce")
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(t, ts["OBS_WS (m/s)"], label="OBS_WS", linewidth=1.0)
        axes[0].plot(t, ts["WRF_WS (m/s)"], label="WRF_WS", linewidth=1.0, alpha=0.9)
        axes[0].plot(t, ts["MOST_U_MO (m/s)"], label="MOST_U_MO", linewidth=1.0, alpha=0.9)
        axes[0].set_ylabel("Wind speed (m/s)")
        axes[0].legend(loc="best", ncol=3)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, ts["OBS_minus_WRF (m/s)"], label="OBS - WRF", linewidth=1.0)
        axes[1].plot(t, ts["MO_minus_WRF (m/s)"], label="MOST - WRF", linewidth=1.0, alpha=0.9)
        axes[1].plot(t, ts["MOST_minus_OBS (m/s)"], label="MOST - OBS", linewidth=1.0, alpha=0.8)
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_ylabel("Residual / error (m/s)")
        axes[1].set_xlabel("Time")
        axes[1].legend(loc="best", ncol=3)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save MOST diagnostic plot {out_png}: {exc}")
        return False


def build_most_ws_diagnostics_for_dataset(
    df: pd.DataFrame,
    csv_path: Path,
    rel_path: str,
    dataset_version: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare OBS_WS, raw WRF_WS and MOST U_MO at all available heights.

    Outputs per-height time-series CSVs and optional plots, and returns one metrics
    row per source file / dataset version / height. The key question is whether
    U_MO is closer to OBS_WS than WRF_WS and whether U_MO-WRF has the same sign as
    OBS-WRF, which directly diagnoses whether MOST is a useful bias-correction guide.
    """
    if df is None or df.empty or TIME_COL not in df.columns:
        return pd.DataFrame()

    cols = list(df.columns)
    height_labels = set()
    for c in cols:
        parsed = parse_wind_column(c)
        if parsed and parsed[1] == "WS":
            height_labels.add(parsed[2])
        h_most = parse_height_from_prefixed_column(c, ("U_MO", "MOST_WS"))
        if h_most is not None:
            height_labels.add(h_most)

    rows = []
    ts_root = ensure_dir(out_dir / "05_most_ws_obs_wrf_timeseries")
    plot_root = ensure_dir(out_dir / "05_most_ws_obs_wrf_plots")

    for h_label in sorted(height_labels, key=lambda x: height_label_to_float(x)):
        obs_ws = find_col(cols, f"OBS_WS{h_label}")
        wrf_ws = find_col(cols, f"WRF_WS{h_label}_") or find_col(cols, f"WRF_WS{h_label} ") or find_col(cols, f"WRF_WS{h_label}")
        u_mo = find_col(cols, f"U_MO{h_label}") or find_col(cols, f"MOST_WS{h_label}")
        if obs_ws is None or wrf_ws is None or u_mo is None:
            continue

        tmp = pd.DataFrame({
            TIME_COL: pd.to_datetime(df[TIME_COL], errors="coerce"),
            "OBS_WS (m/s)": pd.to_numeric(df[obs_ws], errors="coerce"),
            "WRF_WS (m/s)": pd.to_numeric(df[wrf_ws], errors="coerce"),
            "MOST_U_MO (m/s)": pd.to_numeric(df[u_mo], errors="coerce"),
        }).dropna(subset=[TIME_COL])
        tmp = tmp.replace([np.inf, -np.inf], np.nan)
        mask = tmp[["OBS_WS (m/s)", "WRF_WS (m/s)", "MOST_U_MO (m/s)"]].notna().all(axis=1)
        paired = tmp.loc[mask].copy()
        if len(paired) == 0:
            continue

        paired["WRF_minus_OBS (m/s)"] = paired["WRF_WS (m/s)"] - paired["OBS_WS (m/s)"]
        paired["MOST_minus_OBS (m/s)"] = paired["MOST_U_MO (m/s)"] - paired["OBS_WS (m/s)"]
        paired["OBS_minus_WRF (m/s)"] = paired["OBS_WS (m/s)"] - paired["WRF_WS (m/s)"]
        paired["MO_minus_WRF (m/s)"] = paired["MOST_U_MO (m/s)"] - paired["WRF_WS (m/s)"]
        paired["abs_WRF_error (m/s)"] = paired["WRF_minus_OBS (m/s)"].abs()
        paired["abs_MOST_error (m/s)"] = paired["MOST_minus_OBS (m/s)"].abs()
        paired["MOST_improves_abs_error"] = (paired["abs_MOST_error (m/s)"] < paired["abs_WRF_error (m/s)"]).astype(int)

        obs_res = paired["OBS_minus_WRF (m/s)"].to_numpy(dtype=float)
        mo_res = paired["MO_minus_WRF (m/s)"].to_numpy(dtype=float)
        finite_res = np.isfinite(obs_res) & np.isfinite(mo_res)
        obs_large = np.abs(obs_res) >= float(MOST_DIAG_MIN_ABS_OBS_RESID)
        sign_ok = (obs_res * mo_res) >= 0.0
        absdiff_ok = np.abs(mo_res - obs_res) <= float(MOST_DIAG_ABSDIFF_TOL)
        gate = finite_res & ((sign_ok & obs_large) | absdiff_ok)
        paired["MOST_OBS_residual_sign_agree"] = (finite_res & sign_ok & obs_large).astype(int)
        paired["MOST_OBS_residual_absdiff_le_tol"] = (finite_res & absdiff_ok).astype(int)
        paired["MOST_OBS_consistency_gate"] = gate.astype(int)

        wrf_metrics = _basic_error_metrics(
            paired["OBS_WS (m/s)"].values, paired["WRF_WS (m/s)"].values, "WRF_vs_OBS"
        )
        most_metrics = _basic_error_metrics(
            paired["OBS_WS (m/s)"].values, paired["MOST_U_MO (m/s)"].values, "MOST_vs_OBS"
        )
        residual_metrics = _basic_error_metrics(obs_res, mo_res, "MOminusWRF_vs_OBSminusWRF")

        wrf_rmse = wrf_metrics.get("WRF_vs_OBS_RMSE", np.nan)
        most_rmse = most_metrics.get("MOST_vs_OBS_RMSE", np.nan)
        wrf_mae = wrf_metrics.get("WRF_vs_OBS_MAE", np.nan)
        most_mae = most_metrics.get("MOST_vs_OBS_MAE", np.nan)

        rel_safe = safe_name(rel_path.replace("/", "__").replace("\\", "__"))
        stem = f"{safe_name(dataset_version)}__{rel_safe}__H{h_label}"
        ts_file = ts_root / f"{stem}_MOST_WS_OBS_WRF_timeseries.csv"
        plot_file = plot_root / f"{stem}_MOST_WS_OBS_WRF_timeseries.png"

        if RUN_MOST_WS_DIAGNOSTICS and MOST_DIAG_SAVE_TIMESERIES:
            paired.to_csv(ts_file, index=False, encoding="utf-8-sig")
        plot_saved = False
        if RUN_MOST_WS_DIAGNOSTICS and MOST_DIAG_SAVE_PLOTS:
            title = f"{dataset_version} | {rel_path} | height={height_label_to_float(h_label):g} m"
            plot_saved = _plot_most_ws_diagnostic(paired, plot_file, title)

        row = {
            "source_file": str(csv_path),
            "relative_file": rel_path,
            "dataset_version": dataset_version,
            "height_label": h_label,
            "height_m": _format_height_for_output(h_label),
            "obs_ws_col": obs_ws,
            "wrf_ws_col": wrf_ws,
            "most_ws_col": u_mo,
            "n_common": int(len(paired)),
            "time_start": str(paired[TIME_COL].min()),
            "time_end": str(paired[TIME_COL].max()),
            "mean_OBS_WS": float(paired["OBS_WS (m/s)"].mean()),
            "mean_WRF_WS": float(paired["WRF_WS (m/s)"].mean()),
            "mean_MOST_U_MO": float(paired["MOST_U_MO (m/s)"].mean()),
            "MOST_skill_RMSE_pct_vs_WRF": _skill_pct(wrf_rmse, most_rmse),
            "MOST_skill_MAE_pct_vs_WRF": _skill_pct(wrf_mae, most_mae),
            "MOST_better_abs_error_ratio": float(paired["MOST_improves_abs_error"].mean()),
            "residual_corr_MOminusWRF_vs_OBSminusWRF": safe_pearson(obs_res, mo_res),
            "residual_sign_agreement_ratio": float(paired["MOST_OBS_residual_sign_agree"].mean()),
            "residual_absdiff_le_tol_ratio": float(paired["MOST_OBS_residual_absdiff_le_tol"].mean()),
            "consistency_gate_ratio_sign_or_absdiff": float(paired["MOST_OBS_consistency_gate"].mean()),
            "min_samples_required": int(MOST_DIAG_MIN_SAMPLES),
            "enough_samples": int(len(paired) >= int(MOST_DIAG_MIN_SAMPLES)),
            "timeseries_csv": str(ts_file) if MOST_DIAG_SAVE_TIMESERIES else "",
            "timeseries_plot": str(plot_file) if plot_saved else "",
        }
        row.update(wrf_metrics)
        row.update(most_metrics)
        row.update(residual_metrics)
        rows.append(row)

    return pd.DataFrame(rows)



def _plot_profile_timeseries_generic(ts: pd.DataFrame, out_png: Path, title: str, value_cols: Tuple[str, str, str], residual_cols: Tuple[str, str, str], y_label: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip plot {out_png.name}: {exc}")
        return False
    try:
        t = pd.to_datetime(ts[TIME_COL], errors='coerce')
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        labels1 = ['OBS', 'WRF', 'MOST']
        for c, lab in zip(value_cols, labels1):
            axes[0].plot(t, ts[c], label=lab, linewidth=1.0)
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc='best', ncol=3)
        axes[0].grid(True, alpha=0.3)
        labels2 = ['OBS - WRF', 'MOST - WRF', 'MOST - OBS']
        for c, lab in zip(residual_cols, labels2):
            axes[1].plot(t, ts[c], label=lab, linewidth=1.0)
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_ylabel('Residual / error')
        axes[1].set_xlabel('Time')
        axes[1].legend(loc='best', ncol=3)
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save plot {out_png}: {exc}")
        return False


def _plot_metric_bars_for_file(metrics_df: pd.DataFrame, out_png: Path, title: str, var_kind: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip metric plot {out_png.name}: {exc}")
        return False
    if metrics_df is None or metrics_df.empty:
        return False
    dfp = metrics_df.copy().sort_values('height_m')
    heights = dfp['height_m'].astype(float).values
    idx = np.arange(len(dfp))
    width = 0.35
    if var_kind == 'ws':
        rmse_wrf, rmse_most = 'WRF_vs_OBS_RMSE', 'MOST_vs_OBS_RMSE'
        mae_wrf, mae_most = 'WRF_vs_OBS_MAE', 'MOST_vs_OBS_MAE'
        yskill1, yskill2 = 'MOST_skill_RMSE_pct_vs_WRF', 'MOST_skill_MAE_pct_vs_WRF'
        ymean_wrf, ymean_most, ymean_obs = 'mean_WRF_WS', 'mean_MOST_U_MO', 'mean_OBS_WS'
        ylabel = 'Wind speed (m/s)'
    else:
        rmse_wrf, rmse_most = 'WRF_vs_OBS_RMSE', 'MOST_vs_OBS_RMSE'
        mae_wrf, mae_most = 'WRF_vs_OBS_MAE', 'MOST_vs_OBS_MAE'
        yskill1, yskill2 = 'MOST_skill_RMSE_pct_vs_WRF', 'MOST_skill_MAE_pct_vs_WRF'
        ymean_wrf, ymean_most, ymean_obs = 'mean_WRF_T', 'mean_MOST_T_MO', 'mean_OBS_T'
        ylabel = 'Temperature (K)'
    try:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        ax = axes[0,0]
        ax.bar(idx - width/2, dfp[rmse_wrf].values, width=width, label='WRF')
        ax.bar(idx + width/2, dfp[rmse_most].values, width=width, label='MOST')
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{h:g}' for h in heights])
        ax.set_title('RMSE by height')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[0,1]
        ax.bar(idx - width/2, dfp[mae_wrf].values, width=width, label='WRF')
        ax.bar(idx + width/2, dfp[mae_most].values, width=width, label='MOST')
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{h:g}' for h in heights])
        ax.set_title('MAE by height')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1,0]
        ax.bar(idx - width/2, dfp[yskill1].values, width=width, label='RMSE skill')
        ax.bar(idx + width/2, dfp[yskill2].values, width=width, label='MAE skill')
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{h:g}' for h in heights])
        ax.set_title('MOST skill vs WRF (%)')
        ax.set_ylabel('Skill (%)')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1,1]
        ax.plot(heights, dfp[ymean_obs].values, marker='o', label='OBS')
        ax.plot(heights, dfp[ymean_wrf].values, marker='o', label='WRF')
        ax.plot(heights, dfp[ymean_most].values, marker='o', label='MOST')
        ax.set_title('Mean value by height')
        ax.set_xlabel('Height (m)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save metric plot {out_png}: {exc}")
        return False


def _save_metric_plot_grouped(metrics_df: pd.DataFrame, plot_root: Path, prefix: str, var_kind: str) -> None:
    if not RUN_METRIC_PLOTS or metrics_df is None or metrics_df.empty:
        return
    ensure_dir(plot_root)
    for (dataset_version, rel_path), g in metrics_df.groupby(['dataset_version', 'relative_file'], dropna=False):
        rel_safe = safe_name(str(rel_path).replace('/', '__').replace('\\', '__'))
        stem = f"{safe_name(str(dataset_version))}__{rel_safe}__{prefix}_metrics_by_height.png"
        title = f"{prefix.upper()} metrics | {dataset_version} | {rel_path}"
        _plot_metric_bars_for_file(g, plot_root / stem, title, var_kind)




def _plot_most_temp_diagnostic(ts: pd.DataFrame, out_png: Path, title: str) -> bool:
    return _plot_profile_timeseries_generic(
        ts=ts,
        out_png=out_png,
        title=title,
        value_cols=('OBS_T (K)', 'WRF_T (K)', 'MOST_T_MO (K)'),
        residual_cols=('OBS_minus_WRF (K)', 'MO_minus_WRF (K)', 'MOST_minus_OBS (K)'),
        y_label='Temperature (K)',
    )


def build_most_temp_diagnostics_for_dataset(
    df: pd.DataFrame,
    csv_path: Path,
    rel_path: str,
    dataset_version: str,
    out_dir: Path,
) -> pd.DataFrame:
    if df is None or df.empty or TIME_COL not in df.columns:
        return pd.DataFrame()

    df = add_most_temperature_profile_columns(df)
    cols = list(df.columns)
    height_labels = set()
    for c in cols:
        parsed = parse_temp_column(c)
        if parsed:
            height_labels.add(parsed[1])
        h_most = parse_height_from_prefixed_column(c, ('T_MO', 'MO_T_minus_WRF'))
        if h_most is not None:
            height_labels.add(h_most)

    rows = []
    ts_root = ensure_dir(out_dir / '08_most_temp_obs_wrf_timeseries')
    plot_root = ensure_dir(out_dir / '08_most_temp_obs_wrf_plots')

    for h_label in sorted(height_labels, key=lambda x: height_label_to_float(x)):
        obs_t = _first_existing(cols, [f'OBS_T{h_label}', f'OBS_TEMP{h_label}'])
        wrf_t = _first_existing(cols, [f'WRF_T{h_label}_', f'WRF_T{h_label} ', f'WRF_T{h_label}', f'WRF_TEMP{h_label}_', f'WRF_TEMP{h_label} ', f'WRF_TEMP{h_label}'])
        t_mo = find_col(cols, f'T_MO{h_label}')
        if obs_t is None or wrf_t is None or t_mo is None:
            continue
        tmp = pd.DataFrame({
            TIME_COL: pd.to_datetime(df[TIME_COL], errors='coerce'),
            'OBS_T (K)': pd.to_numeric(df[obs_t], errors='coerce'),
            'WRF_T (K)': pd.to_numeric(df[wrf_t], errors='coerce'),
            'MOST_T_MO (K)': pd.to_numeric(df[t_mo], errors='coerce'),
        }).dropna(subset=[TIME_COL])
        tmp = tmp.replace([np.inf, -np.inf], np.nan)
        mask = tmp[['OBS_T (K)', 'WRF_T (K)', 'MOST_T_MO (K)']].notna().all(axis=1)
        paired = tmp.loc[mask].copy()
        if len(paired) < MOST_TEMP_DIAG_MIN_SAMPLES:
            continue

        paired['WRF_minus_OBS (K)'] = paired['WRF_T (K)'] - paired['OBS_T (K)']
        paired['MOST_minus_OBS (K)'] = paired['MOST_T_MO (K)'] - paired['OBS_T (K)']
        paired['OBS_minus_WRF (K)'] = paired['OBS_T (K)'] - paired['WRF_T (K)']
        paired['MO_minus_WRF (K)'] = paired['MOST_T_MO (K)'] - paired['WRF_T (K)']
        paired['abs_WRF_error (K)'] = paired['WRF_minus_OBS (K)'].abs()
        paired['abs_MOST_error (K)'] = paired['MOST_minus_OBS (K)'].abs()
        paired['MOST_improves_abs_error'] = (paired['abs_MOST_error (K)'] < paired['abs_WRF_error (K)']).astype(int)

        obs_res = paired['OBS_minus_WRF (K)'].to_numpy(dtype=float)
        mo_res = paired['MO_minus_WRF (K)'].to_numpy(dtype=float)
        finite_res = np.isfinite(obs_res) & np.isfinite(mo_res)
        obs_large = np.abs(obs_res) >= float(MOST_DIAG_MIN_ABS_OBS_RESID)
        sign_ok = (obs_res * mo_res) >= 0.0
        absdiff_ok = np.abs(mo_res - obs_res) <= float(MOST_DIAG_ABSDIFF_TOL)
        gate = finite_res & ((sign_ok & obs_large) | absdiff_ok)
        paired['MOST_OBS_residual_sign_agree'] = (finite_res & sign_ok & obs_large).astype(int)
        paired['MOST_OBS_residual_absdiff_le_tol'] = (finite_res & absdiff_ok).astype(int)
        paired['MOST_OBS_consistency_gate'] = gate.astype(int)

        wrf_metrics = _basic_error_metrics(paired['OBS_T (K)'].values, paired['WRF_T (K)'].values, 'WRF_vs_OBS')
        most_metrics = _basic_error_metrics(paired['OBS_T (K)'].values, paired['MOST_T_MO (K)'].values, 'MOST_vs_OBS')
        residual_metrics = _basic_error_metrics(obs_res, mo_res, 'MOminusWRF_vs_OBSminusWRF')
        wrf_rmse = wrf_metrics.get('WRF_vs_OBS_RMSE', np.nan)
        most_rmse = most_metrics.get('MOST_vs_OBS_RMSE', np.nan)
        wrf_mae = wrf_metrics.get('WRF_vs_OBS_MAE', np.nan)
        most_mae = most_metrics.get('MOST_vs_OBS_MAE', np.nan)

        rel_safe = safe_name(rel_path.replace('/', '__').replace('\\', '__'))
        stem = f"{safe_name(dataset_version)}__{rel_safe}__H{h_label}"
        ts_file = ts_root / f"{stem}_MOST_TEMP_OBS_WRF_timeseries.csv"
        plot_file = plot_root / f"{stem}_MOST_TEMP_OBS_WRF_timeseries.png"
        if RUN_MOST_TEMP_DIAGNOSTICS and MOST_TEMP_DIAG_SAVE_TIMESERIES:
            paired.to_csv(ts_file, index=False, encoding='utf-8-sig')
        plot_saved = False
        if RUN_MOST_TEMP_DIAGNOSTICS and MOST_TEMP_DIAG_SAVE_PLOTS:
            title = f"{dataset_version} | {rel_path} | T height={height_label_to_float(h_label):g} m"
            plot_saved = _plot_most_temp_diagnostic(paired, plot_file, title)

        row = {
            'source_file': str(csv_path),
            'relative_file': rel_path,
            'dataset_version': dataset_version,
            'height_label': h_label,
            'height_m': _format_height_for_output(h_label),
            'obs_t_col': obs_t,
            'wrf_t_col': wrf_t,
            'most_t_col': t_mo,
            'n_common': int(len(paired)),
            'time_start': str(paired[TIME_COL].min()),
            'time_end': str(paired[TIME_COL].max()),
            'mean_OBS_T': float(paired['OBS_T (K)'].mean()),
            'mean_WRF_T': float(paired['WRF_T (K)'].mean()),
            'mean_MOST_T_MO': float(paired['MOST_T_MO (K)'].mean()),
            'MOST_skill_RMSE_pct_vs_WRF': _skill_pct(wrf_rmse, most_rmse),
            'MOST_skill_MAE_pct_vs_WRF': _skill_pct(wrf_mae, most_mae),
            'MOST_better_abs_error_ratio': float(paired['MOST_improves_abs_error'].mean()),
            'residual_corr_MOminusWRF_vs_OBSminusWRF': safe_pearson(obs_res, mo_res),
            'residual_sign_agreement_ratio': float(paired['MOST_OBS_residual_sign_agree'].mean()),
            'residual_absdiff_le_tol_ratio': float(paired['MOST_OBS_residual_absdiff_le_tol'].mean()),
            'consistency_gate_ratio_sign_or_absdiff': float(paired['MOST_OBS_consistency_gate'].mean()),
            'timeseries_csv': str(ts_file) if (RUN_MOST_TEMP_DIAGNOSTICS and MOST_TEMP_DIAG_SAVE_TIMESERIES) else '',
            'timeseries_plot': str(plot_file) if plot_saved else '',
        }
        row.update(wrf_metrics)
        row.update(most_metrics)
        row.update(residual_metrics)
        rows.append(row)
    return pd.DataFrame(rows)




def _valid_range_for_col(col: str) -> Optional[Tuple[float, float]]:
    c = str(col)
    if c in MOST_RELATED_VALID_RANGES:
        return MOST_RELATED_VALID_RANGES[c]
    for key, vr in MOST_RELATED_VALID_RANGES.items():
        if c.startswith(key):
            return vr
    return None


def _clean_reasonable_series(s: pd.Series, col: str) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    vr = _valid_range_for_col(col)
    if vr is not None:
        lo, hi = vr
        x = x.where((x >= lo) & (x <= hi), np.nan)
    return x


def _collect_most_related_columns(cols: Iterable[str]) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    profile_groups: Dict[str, List[Tuple[str, str]]] = {}
    scalar_cols: List[str] = []
    for col in cols:
        c = str(col)
        matched = False
        for prefix in MOST_RELATED_PREFIXES:
            h = parse_height_from_prefixed_column(c, (prefix,))
            if h is not None:
                profile_groups.setdefault(prefix, []).append((h, c))
                matched = True
                break
            if c.startswith(prefix) and prefix == 'MOST':
                # fallback for unusual MOST-prefixed scalars
                scalar_cols.append(c)
                matched = True
                break
        if matched:
            continue
        if c in MOST_RELATED_SCALAR_COLS:
            scalar_cols.append(c)
    for prefix in list(profile_groups.keys()):
        profile_groups[prefix] = sorted(profile_groups[prefix], key=lambda t: height_label_to_float(t[0]))
    scalar_cols = [c for c in MOST_RELATED_SCALAR_COLS if c in scalar_cols] + [c for c in scalar_cols if c not in MOST_RELATED_SCALAR_COLS]
    return profile_groups, scalar_cols


def _plot_most_related_profile_timeseries(df: pd.DataFrame, time_col: str, cols_info: List[Tuple[str, str]], out_png: Path, title: str, ylabel: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip profile plot {out_png.name}: {exc}")
        return False
    try:
        t = pd.to_datetime(df[time_col], errors='coerce')
        fig, ax = plt.subplots(figsize=(13, 5.5))
        any_line = False
        for h, col in cols_info:
            y = _clean_reasonable_series(df[col], col)
            if y.notna().sum() == 0:
                continue
            ax.plot(t, y, linewidth=0.9, label=f'{h} m')
            any_line = True
        if not any_line:
            plt.close(fig)
            return False
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=min(4, max(1, len(cols_info))))
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save profile timeseries plot {out_png}: {exc}")
        return False


def _plot_most_related_profile_mean(df: pd.DataFrame, cols_info: List[Tuple[str, str]], out_png: Path, title: str, ylabel: str) -> Tuple[bool, pd.DataFrame]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip profile mean plot {out_png.name}: {exc}")
        return False, pd.DataFrame()
    rows = []
    for h, col in cols_info:
        y = _clean_reasonable_series(df[col], col)
        rows.append({
            'height_label': h,
            'height_m': height_label_to_float(h),
            'column': col,
            'n_valid_clean': int(y.notna().sum()),
            'mean_clean': float(y.mean()) if y.notna().sum() else np.nan,
            'median_clean': float(y.median()) if y.notna().sum() else np.nan,
            'std_clean': float(y.std()) if y.notna().sum() else np.nan,
            'min_clean': float(y.min()) if y.notna().sum() else np.nan,
            'max_clean': float(y.max()) if y.notna().sum() else np.nan,
        })
    stats = pd.DataFrame(rows).sort_values('height_m')
    if stats.empty or stats['n_valid_clean'].fillna(0).sum() <= 0:
        return False, stats
    try:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(stats['mean_clean'].values, stats['height_m'].values, marker='o', linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Height (m)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True, stats
    except Exception as exc:
        print(f"⚠ Failed to save profile mean plot {out_png}: {exc}")
        return False, stats


def _plot_most_related_scalar_timeseries(df: pd.DataFrame, time_col: str, scalar_cols: List[str], out_png: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip scalar plot {out_png.name}: {exc}")
        return False
    if not scalar_cols:
        return False
    n = len(scalar_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    try:
        t = pd.to_datetime(df[time_col], errors='coerce')
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(3.5, 2.7 * nrows)), sharex=True)
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        for ax, col in zip(axes.ravel(), scalar_cols):
            y = _clean_reasonable_series(df[col], col)
            ax.plot(t, y, linewidth=0.9)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[len(scalar_cols):]:
            ax.axis('off')
        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save scalar timeseries plot {out_png}: {exc}")
        return False


def _plot_most_related_scalar_means(df: pd.DataFrame, scalar_cols: List[str], out_png: Path, title: str) -> Tuple[bool, pd.DataFrame]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip scalar mean plot {out_png.name}: {exc}")
        return False, pd.DataFrame()
    rows = []
    for col in scalar_cols:
        y = _clean_reasonable_series(df[col], col)
        rows.append({
            'column': col,
            'n_valid_clean': int(y.notna().sum()),
            'mean_clean': float(y.mean()) if y.notna().sum() else np.nan,
            'median_clean': float(y.median()) if y.notna().sum() else np.nan,
            'std_clean': float(y.std()) if y.notna().sum() else np.nan,
            'min_clean': float(y.min()) if y.notna().sum() else np.nan,
            'max_clean': float(y.max()) if y.notna().sum() else np.nan,
        })
    stats = pd.DataFrame(rows)
    if stats.empty:
        return False, stats
    try:
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(stats)), 5.5))
        ax.bar(np.arange(len(stats)), stats['mean_clean'].values)
        ax.set_xticks(np.arange(len(stats)))
        ax.set_xticklabels(stats['column'].tolist(), rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel('Mean (cleaned values)')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=METRIC_PLOT_DPI)
        plt.close(fig)
        return True, stats
    except Exception as exc:
        print(f"⚠ Failed to save scalar mean plot {out_png}: {exc}")
        return False, stats


def build_most_related_variable_plots_for_dataset(
    df: pd.DataFrame,
    csv_path: Path,
    rel_path: str,
    dataset_version: str,
    out_dir: Path,
) -> pd.DataFrame:
    if df is None or df.empty or TIME_COL not in df.columns:
        return pd.DataFrame()
    profile_groups, scalar_cols = _collect_most_related_columns(df.columns)
    ts_root = ensure_dir(out_dir / '11_most_related_variable_timeseries')
    mean_root = ensure_dir(out_dir / '12_most_related_variable_mean_plots')
    rows = []
    rel_safe = safe_name(rel_path.replace('/', '__').replace('\\', '__'))

    for prefix, cols_info in profile_groups.items():
        if not cols_info:
            continue
        ylabel = 'Value'
        if prefix in {'U_MO', 'MOST', 'MO_minus_WRF'}:
            ylabel = 'Wind-related value (m/s)'
        elif prefix in {'T_MO', 'MO_T_minus_WRF'}:
            ylabel = 'Temperature-related value (K)'
        ts_png = ts_root / f"{safe_name(dataset_version)}__{rel_safe}__{prefix}_timeseries.png"
        mean_png = mean_root / f"{safe_name(dataset_version)}__{rel_safe}__{prefix}_mean_profile.png"
        title_ts = f"MOST-related profile time series | {dataset_version} | {rel_path} | {prefix}"
        title_mean = f"MOST-related profile mean | {dataset_version} | {rel_path} | {prefix}"
        ts_saved = _plot_most_related_profile_timeseries(df, TIME_COL, cols_info, ts_png, title_ts, ylabel)
        mean_saved, stats = _plot_most_related_profile_mean(df, cols_info, mean_png, title_mean, ylabel)
        if not stats.empty:
            stats['source_file'] = str(csv_path)
            stats['relative_file'] = rel_path
            stats['dataset_version'] = dataset_version
            stats['group_kind'] = 'profile'
            stats['group_name'] = prefix
            stats['timeseries_plot'] = str(ts_png) if ts_saved else ''
            stats['mean_plot'] = str(mean_png) if mean_saved else ''
            rows.append(stats)

    if scalar_cols:
        ts_png = ts_root / f"{safe_name(dataset_version)}__{rel_safe}__MOST_surface_scalar_timeseries.png"
        mean_png = mean_root / f"{safe_name(dataset_version)}__{rel_safe}__MOST_surface_scalar_means.png"
        title_ts = f"MOST-related surface/stability drivers | {dataset_version} | {rel_path}"
        title_mean = f"MOST-related surface/stability driver means | {dataset_version} | {rel_path}"
        ts_saved = _plot_most_related_scalar_timeseries(df, TIME_COL, scalar_cols, ts_png, title_ts)
        mean_saved, stats = _plot_most_related_scalar_means(df, scalar_cols, mean_png, title_mean)
        if not stats.empty:
            stats['source_file'] = str(csv_path)
            stats['relative_file'] = rel_path
            stats['dataset_version'] = dataset_version
            stats['group_kind'] = 'scalar'
            stats['group_name'] = 'MOST_surface_scalar'
            stats['height_label'] = ''
            stats['height_m'] = np.nan
            stats['timeseries_plot'] = str(ts_png) if ts_saved else ''
            stats['mean_plot'] = str(mean_png) if mean_saved else ''
            rows.append(stats)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()




# =============================================================================
# Gryning / Batchvarova extended wind-profile inversion
# =============================================================================


def _find_wrf_ws_col(cols: Iterable[str], h_label: str) -> Optional[str]:
    """Find WRF wind-speed column for a height label, keeping compatibility with ML-ready names."""
    return (
        find_col(cols, f"WRF_WS{h_label}_")
        or find_col(cols, f"WRF_WS{h_label} ")
        or find_col(cols, f"WRF_WS{h_label}")
    )


def _find_obs_ws_col(cols: Iterable[str], h_label: str) -> Optional[str]:
    """Find OBS wind-speed column for a height label."""
    return find_col(cols, f"OBS_WS{h_label}_") or find_col(cols, f"OBS_WS{h_label} ") or find_col(cols, f"OBS_WS{h_label}")


def _find_wrf_t_col(cols: Iterable[str], h_label: str) -> Optional[str]:
    return _first_existing(list(cols), [
        f"WRF_T{h_label}_", f"WRF_T{h_label} ", f"WRF_T{h_label}",
        f"WRF_TEMP{h_label}_", f"WRF_TEMP{h_label} ", f"WRF_TEMP{h_label}",
    ])


def _find_obs_t_col(cols: Iterable[str], h_label: str) -> Optional[str]:
    return _first_existing(list(cols), [
        f"OBS_T{h_label}_", f"OBS_T{h_label} ", f"OBS_T{h_label}",
        f"OBS_TEMP{h_label}_", f"OBS_TEMP{h_label} ", f"OBS_TEMP{h_label}",
    ])


def _find_scalar_col(cols: Iterable[str], names: List[str]) -> Optional[str]:
    """Find a scalar surface/PBL column by exact name first, then by prefix."""
    cols = list(cols)
    for n in names:
        if n in cols:
            return n
    for n in names:
        c = find_col(cols, n)
        if c is not None:
            return c
    return None


def _collect_ws_height_labels_for_gryning(cols: Iterable[str]) -> List[str]:
    """Collect all OBS/WRF WS heights available in a ML-ready wide table."""
    labels = set()
    for c in cols:
        parsed = parse_wind_column(c)
        if parsed and parsed[1] == "WS":
            labels.add(parsed[2])
    return sorted(labels, key=lambda x: height_label_to_float(x))


def _collect_temp_height_labels_for_gryning(cols: Iterable[str]) -> List[str]:
    """Collect all OBS/WRF temperature heights available in a ML-ready wide table."""
    labels = set()
    for c in cols:
        parsed = parse_temp_column(c)
        if parsed:
            labels.add(parsed[1])
    return sorted(labels, key=lambda x: height_label_to_float(x))


def _scalar_at_row(df: pd.DataFrame, row_i: int, col: Optional[str]) -> float:
    if col is None or col not in df.columns:
        return np.nan
    return pd.to_numeric(pd.Series([df.iloc[row_i][col]]), errors="coerce").iloc[0]


def _to_kelvin_value(v: float) -> float:
    """Keep K values as-is; convert plausible Celsius values to K."""
    if not np.isfinite(v):
        return np.nan
    return float(v + 273.15) if v < 150.0 else float(v)


def coriolis_parameter_from_lat(lat_deg: float) -> float:
    omega = 7.2921159e-5  # rad s-1
    if not np.isfinite(lat_deg):
        lat_deg = GRYNING_LATITUDE_DEG_DEFAULT
    return float(2.0 * omega * math.sin(math.radians(float(lat_deg))))


def infer_station_latitude(rel_path: str) -> float:
    text = str(rel_path)
    for key, lat in GRYNING_SITE_LATITUDE_DEG.items():
        if key in text:
            return float(lat)
    return float(GRYNING_LATITUDE_DEG_DEFAULT)


def classify_gryning_stability(L: float) -> str:
    if not np.isfinite(L) or abs(float(L)) >= float(GRYNING_NEUTRAL_ABS_L_THRESHOLD_M):
        return "neutral_or_L_missing"
    if abs(float(L)) < float(GRYNING_MIN_ABS_L_FOR_STABILITY_M):
        return "very_small_abs_L_clipped"
    return "stable" if float(L) > 0.0 else "unstable"


def _effective_L_for_formula(L: float) -> float:
    if not np.isfinite(L) or abs(float(L)) >= float(GRYNING_NEUTRAL_ABS_L_THRESHOLD_M):
        return np.nan
    if abs(float(L)) < float(GRYNING_MIN_ABS_L_FOR_STABILITY_M):
        return math.copysign(float(GRYNING_MIN_ABS_L_FOR_STABILITY_M), float(L) if L != 0 else 1.0)
    return float(L)


def psi_m_unstable_gryning(z_over_L: np.ndarray) -> np.ndarray:
    """Gryning et al. (2007) Eq. (20), with p=-1/3 and a=-12 for unstable conditions."""
    xarg = np.maximum(1.0 - 12.0 * np.asarray(z_over_L, dtype=float), 1.0e-12)
    x = np.power(xarg, 1.0 / 3.0)
    return 1.5 * np.log((1.0 + x + x * x) / 3.0) - np.sqrt(3.0) * np.arctan((1.0 + 2.0 * x) / np.sqrt(3.0)) + np.pi / np.sqrt(3.0)


def gryning_lmbl_eq31(ustar: float, z0: float, L: float, coriolis_f: float) -> float:
    """
    Gryning et al. (2007) applied parameterization of L_MBL, Eq. (31):

        u*/(f L_MBL) = [-2 ln(u*/(f z0)) + 55] * exp(-((u*/(f L))^2)/400)

    For missing/near-neutral L, the exponential term is set to 1. Returns NaN when
    the parameterization is numerically or physically invalid.
    """
    if not (np.isfinite(ustar) and np.isfinite(z0) and np.isfinite(coriolis_f)):
        return np.nan
    ustar = float(ustar)
    z0 = float(z0)
    f = abs(float(coriolis_f))
    if ustar <= 0.0 or z0 <= 0.0 or f <= 1.0e-8:
        return np.nan
    rough_arg = ustar / (f * z0)
    if rough_arg <= 1.0:
        return np.nan
    base = -2.0 * math.log(rough_arg) + 55.0
    if not np.isfinite(base) or base <= 0.0:
        return np.nan
    L_eff = _effective_L_for_formula(L)
    if np.isfinite(L_eff):
        stab_arg = ustar / (f * L_eff)
        expo = math.exp(-((stab_arg) ** 2) / 400.0)
    else:
        expo = 1.0
    denom = f * base * expo
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan
    lmbl = ustar / denom
    if not np.isfinite(lmbl) or lmbl <= 0.0:
        return np.nan
    return float(lmbl)


def gryning_profile_shape(
    z: np.ndarray,
    z0: float,
    zi: float = np.nan,
    lmbl: float = np.nan,
    L: float = np.nan,
    d_factor: float = GRYNING_D_FACTOR,
) -> np.ndarray:
    """
    Gryning/Batchvarova extended wind-profile shape G(z).

    U(z) = (u_star / kappa) * G(z)

    Neutral, Gryning et al. (2007) Eq. (8):
      G = ln(z/z0) + z/L_MBL - z^2/(2 L_MBL z_i)

    Stable, Eq. (15):
      G = Eq.(8) + b z/L * (1 - z/(2 z_i))

    Unstable, Eq. (21):
      G = Eq.(8) - psi(z/L)

    If L_MBL or z_i is unavailable, the extension term is skipped and the shape
    reduces to the corresponding surface-layer form.
    """
    z = np.asarray(z, dtype=float)
    z0 = float(z0)
    d = float(d_factor) * z0
    zeff = np.maximum(z - d, z0 * 1.01)
    g = np.log(zeff / z0)

    zi_use = float(zi) if np.isfinite(zi) and zi > 0 else np.nan
    lmbl_use = float(lmbl) if np.isfinite(lmbl) and lmbl > 0 else np.nan
    if GRYNING_USE_PBLH_ZI and np.isfinite(zi_use) and np.isfinite(lmbl_use):
        zclip = np.minimum(zeff, 0.98 * zi_use)
        g = g + zclip / lmbl_use - (zclip ** 2) / (2.0 * lmbl_use * zi_use)
    else:
        zclip = zeff

    L_eff = _effective_L_for_formula(L)
    if np.isfinite(L_eff):
        if L_eff > 0.0:
            zi_for_stab = zi_use if np.isfinite(zi_use) else np.inf
            g = g + float(GRYNING_STABLE_BETA) * zclip / L_eff * (1.0 - zclip / (2.0 * zi_for_stab))
        elif L_eff < 0.0:
            g = g - psi_m_unstable_gryning(zclip / L_eff)
    return g


def _initial_ustar_from_shape(z: np.ndarray, u: np.ndarray, z0: float, L: float) -> float:
    g = gryning_profile_shape(z, z0, zi=np.nan, lmbl=np.nan, L=L)
    finite = np.isfinite(g) & (g > 0) & np.isfinite(u)
    if int(finite.sum()) < 1:
        return np.nan
    denom = float(np.sum(g[finite] ** 2))
    if denom <= 1.0e-14:
        return np.nan
    beta = float(np.sum(g[finite] * u[finite]) / denom)
    return float(KAPPA * beta)


def fit_gryning_z0_ustar(
    z: np.ndarray,
    u: np.ndarray,
    pblh: float = np.nan,
    L: float = np.nan,
    coriolis_f: float = np.nan,
    z0_hint: float = np.nan,
    ustar_hint: float = np.nan,
) -> Dict[str, object]:
    """
    Estimate z0 and u_star from one multi-height wind-speed profile using
    Gryning et al. (2007) Eq. (8)/(15)/(21) and Eq. (31) for L_MBL.

    Because Eq. (31) makes L_MBL depend on z0 and u_star, the fitting is:
      1. bounded log-grid search over z0;
      2. fixed-point iteration for u_star at each z0;
      3. select the z0/u_star pair with minimum profile RMSE.
    """
    z = np.asarray(z, dtype=float)
    u = np.asarray(u, dtype=float)
    valid = (
        np.isfinite(z) & np.isfinite(u)
        & (z > 0.0)
        & (u >= float(GRYNING_MIN_WIND_SPEED))
        & (u <= float(GRYNING_MAX_WIND_SPEED))
    )
    z = z[valid]
    u = u[valid]
    order = np.argsort(z)
    z = z[order]
    u = u[order]

    result = {
        "z0_gryning_m": np.nan,
        "ustar_gryning_mps": np.nan,
        "profile_rmse_mps": np.nan,
        "profile_mae_mps": np.nan,
        "fit_status": "insufficient_profile_levels",
        "fit_method": "gryning2007_eq8_15_21_lmbl_eq31_z0grid_iterated_ustar",
        "pblh_used_m": np.nan,
        "zi_used_m": np.nan,
        "lmb_used_m": np.nan,
        "L_used_m": _effective_L_for_formula(L),
        "stability_class": classify_gryning_stability(L),
        "coriolis_f_s-1": coriolis_f,
        "n_heights_used_in_fit": 0,
    }

    if len(z) > 0:
        zi = float(pblh) if np.isfinite(pblh) and pblh > 0 else float(GRYNING_PBLH_FALLBACK_M)
        zi = max(zi, 1.10 * float(np.nanmax(z)))
    else:
        zi = float(pblh) if np.isfinite(pblh) and pblh > 0 else float(GRYNING_PBLH_FALLBACK_M)
    result["pblh_used_m"] = zi
    result["zi_used_m"] = zi

    if len(z) < int(GRYNING_MIN_HEIGHTS_TO_FIT):
        if np.isfinite(z0_hint) and z0_hint > 0 and np.isfinite(ustar_hint) and ustar_hint > 0:
            lmbl = gryning_lmbl_eq31(float(ustar_hint), float(z0_hint), L, coriolis_f)
            if not np.isfinite(lmbl):
                lmbl = float(GRYNING_LMBL_FALLBACK_M)
            result.update({
                "z0_gryning_m": float(np.clip(z0_hint, GRYNING_Z0_MIN, GRYNING_Z0_MAX)),
                "ustar_gryning_mps": float(np.clip(ustar_hint, GRYNING_USTAR_MIN, GRYNING_USTAR_MAX)),
                "lmb_used_m": float(lmbl),
                "fit_status": "insufficient_heights_used_wrf_surface_z0_ustar",
                "fit_method": "wrf_surface_fallback",
                "n_heights_used_in_fit": int(len(z)),
            })
        return result

    z0_hi = min(float(GRYNING_Z0_MAX), max(float(GRYNING_Z0_MIN) * 10.0, 0.95 * float(np.nanmin(z))))
    if z0_hi <= float(GRYNING_Z0_MIN):
        return result
    z0_grid = np.geomspace(float(GRYNING_Z0_MIN), z0_hi, int(GRYNING_Z0_GRID_N))

    best = None
    for z0 in z0_grid:
        if np.isfinite(ustar_hint) and GRYNING_USTAR_MIN <= ustar_hint <= GRYNING_USTAR_MAX:
            ustar = float(ustar_hint)
        else:
            ustar = _initial_ustar_from_shape(z, u, z0, L)
            if not np.isfinite(ustar):
                continue
            ustar = float(np.clip(ustar, GRYNING_USTAR_MIN, GRYNING_USTAR_MAX))

        lmbl = np.nan
        converged = False
        for _ in range(int(GRYNING_ITER_MAX)):
            if GRYNING_USE_LMBL_EQ31:
                lmbl = gryning_lmbl_eq31(ustar, z0, L, coriolis_f)
            if not np.isfinite(lmbl):
                lmbl = float(GRYNING_LMBL_FALLBACK_M)
            g = gryning_profile_shape(z, z0, zi=zi, lmbl=lmbl, L=L)
            finite = np.isfinite(g) & (g > 0) & np.isfinite(u)
            if int(finite.sum()) < int(GRYNING_MIN_HEIGHTS_TO_FIT):
                break
            denom = float(np.sum(g[finite] ** 2))
            if denom <= 1.0e-14:
                break
            beta = float(np.sum(g[finite] * u[finite]) / denom)
            new_ustar = float(np.clip(KAPPA * beta, GRYNING_USTAR_MIN, GRYNING_USTAR_MAX))
            if abs(new_ustar - ustar) <= float(GRYNING_ITER_TOL) * max(1.0, abs(ustar)):
                ustar = new_ustar
                converged = True
                break
            ustar = new_ustar

        if GRYNING_USE_LMBL_EQ31:
            lmbl = gryning_lmbl_eq31(ustar, z0, L, coriolis_f)
        if not np.isfinite(lmbl):
            lmbl = float(GRYNING_LMBL_FALLBACK_M)
        g = gryning_profile_shape(z, z0, zi=zi, lmbl=lmbl, L=L)
        finite = np.isfinite(g) & (g > 0) & np.isfinite(u)
        if int(finite.sum()) < int(GRYNING_MIN_HEIGHTS_TO_FIT):
            continue
        pred = (ustar / KAPPA) * g[finite]
        uu = u[finite]
        err = pred - uu
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        rec = (rmse, mae, z0, ustar, lmbl, int(finite.sum()), converged)
        if best is None or rec[:2] < best[:2]:
            best = rec

    if best is None:
        return result

    rmse, mae, z0, ustar, lmbl, nfit, converged = best
    result.update({
        "z0_gryning_m": float(z0),
        "ustar_gryning_mps": float(ustar),
        "profile_rmse_mps": float(rmse),
        "profile_mae_mps": float(mae),
        "fit_status": "ok" if converged else "ok_not_strictly_converged",
        "n_heights_used_in_fit": int(nfit),
        "lmb_used_m": float(lmbl),
    })
    return result


def gryning_temperature_from_surface(
    z: float,
    t2_k: float,
    ustar: float,
    z0m: float,
    L: float,
    hfx: float,
    psfc: float,
    z_ref: float = GRYNING_TEMP_REF_HEIGHT_M,
) -> float:
    """
    MOST scalar temperature-profile diagnostic using inverted z0/u_star.
    z0h is parameterized as GRYNING_TEMP_Z0H_OVER_Z0M * z0m.
    """
    vals = [z, t2_k, ustar, z0m, hfx, psfc]
    if not all(np.isfinite(v) for v in vals):
        return np.nan
    if ustar <= 1.0e-6 or z0m <= 0 or psfc <= 1000 or t2_k <= 150:
        return np.nan
    z0h = max(float(GRYNING_TEMP_MIN_Z0H), float(GRYNING_TEMP_Z0H_OVER_Z0M) * float(z0m))
    zeff = max(float(z), z0h * 1.01)
    zref_eff = max(float(z_ref), z0h * 1.01)
    rho = float(psfc) / (RD_AIR * max(float(t2_k), 150.0))
    theta_star = -float(hfx) / (rho * CP_AIR * float(ustar))
    L_eff = _effective_L_for_formula(L)
    if not np.isfinite(L_eff):
        zeta = 0.0
        zeta_ref = 0.0
    else:
        zeta = zeff / L_eff
        zeta_ref = zref_eff / L_eff
    prof = math.log(zeff / zref_eff) - float(psi_h_businger(np.asarray([zeta]))[0] - psi_h_businger(np.asarray([zeta_ref]))[0])
    t = float(t2_k) + (theta_star / KAPPA) * prof
    tmin, tmax = GRYNING_TEMP_VALID_RANGE_K
    return float(t) if np.isfinite(t) and tmin <= t <= tmax else np.nan




def gryning_wind_speed_at_height(
    z: float,
    z0m: float,
    ustar: float,
    L: float,
    zi: float = np.nan,
    lmbl: float = np.nan,
) -> float:
    """Evaluate the Gryning wind profile at one height for supplied z0/u*/L/zi/LMBL."""
    if not (np.isfinite(z) and np.isfinite(z0m) and np.isfinite(ustar)):
        return np.nan
    if z <= 0.0 or z0m <= 0.0 or ustar <= 0.0:
        return np.nan
    gshape = gryning_profile_shape(np.asarray([float(z)]), float(z0m), zi=zi, lmbl=lmbl, L=L)[0]
    if not np.isfinite(gshape):
        return np.nan
    out = (float(ustar) / KAPPA) * float(gshape)
    return float(out) if np.isfinite(out) and 0.0 <= out <= GRYNING_MAX_WIND_SPEED else np.nan


def log_most_wind_speed_at_height(z: float, z0m: float, ustar: float, L: float) -> float:
    """
    Traditional surface-layer MOST/log-law wind profile used only as a reference.
    It is equivalent to Gryning with the middle/upper boundary-layer terms disabled.
    """
    return gryning_wind_speed_at_height(z=z, z0m=z0m, ustar=ustar, L=L, zi=np.nan, lmbl=np.nan)


def most_temperature_profile_shape(
    z: np.ndarray,
    z0h: float,
    L: float,
    z_ref: float = GRYNING_TEMP_REF_HEIGHT_M,
) -> np.ndarray:
    """
    MOST scalar-temperature shape H(z) relative to T(z_ref):

        T(z) = T_ref + (theta_star / kappa) * H(z)
        H(z) = ln(z/z_ref) - [psi_h(z/L) - psi_h(z_ref/L)]

    z0h is used as the lower bound for effective height, but the reference is the
    actual near-surface T reference height (default 2 m), not z0h itself.
    """
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z0h) or float(z0h) <= 0.0:
        return np.full_like(z, np.nan, dtype=float)
    z0h = max(float(GRYNING_TEMP_MIN_Z0H), float(z0h))
    zeff = np.maximum(z, z0h * 1.01)
    zref_eff = max(float(z_ref), z0h * 1.01)
    L_eff = _effective_L_for_formula(L)
    if not np.isfinite(L_eff):
        zeta = np.zeros_like(zeff, dtype=float)
        zeta_ref = 0.0
    else:
        zeta = zeff / float(L_eff)
        zeta_ref = zref_eff / float(L_eff)
    return np.log(zeff / zref_eff) - (psi_h_businger(zeta) - psi_h_businger(np.asarray([zeta_ref]))[0])


def most_temperature_from_theta_star(
    z: float,
    t_ref_k: float,
    z0h: float,
    theta_star: float,
    L: float,
    z_ref: float = GRYNING_TEMP_REF_HEIGHT_M,
) -> float:
    """Evaluate a fitted MOST scalar-temperature profile at one height."""
    if not all(np.isfinite(v) for v in [z, t_ref_k, z0h, theta_star]):
        return np.nan
    hshape = most_temperature_profile_shape(np.asarray([float(z)]), float(z0h), L, z_ref=z_ref)[0]
    if not np.isfinite(hshape):
        return np.nan
    t = float(t_ref_k) + (float(theta_star) / KAPPA) * float(hshape)
    tmin, tmax = GRYNING_TEMP_VALID_RANGE_K
    return float(t) if np.isfinite(t) and tmin <= t <= tmax else np.nan


def fit_most_temperature_z0h_thetastar(
    z: np.ndarray,
    temp_k: np.ndarray,
    t_ref_k: float,
    L: float = np.nan,
    psfc: float = np.nan,
    ustar_for_hfx: float = np.nan,
    z_ref: float = GRYNING_TEMP_REF_HEIGHT_M,
) -> Dict[str, object]:
    """
    Fit MOST scalar-temperature parameters z0h and theta_star from one multi-height
    temperature profile. OBS_T is expected to be used first and WRF_T only as fallback
    before this function is called.
    """
    z = np.asarray(z, dtype=float)
    temp_k = np.asarray(temp_k, dtype=float)
    tmin, tmax = GRYNING_TEMP_VALID_RANGE_K
    valid = (
        np.isfinite(z) & np.isfinite(temp_k)
        & (z > 0.0)
        & (temp_k >= tmin) & (temp_k <= tmax)
        & np.isfinite(float(t_ref_k))
    )
    z = z[valid]
    temp_k = temp_k[valid]
    order = np.argsort(z)
    z = z[order]
    temp_k = temp_k[order]

    result = {
        "z0h_inv_m": np.nan,
        "theta_star_inv_K": np.nan,
        "HFX_inv_Wm2": np.nan,
        "T_profile_rmse_K": np.nan,
        "T_profile_mae_K": np.nan,
        "T_fit_status": "insufficient_T_profile_levels",
        "T_fit_method": "MOST_scalar_T_z0h_grid_linear_theta_star",
        "n_T_heights_used_in_fit": int(len(z)),
    }
    if len(z) < int(GRYNING_TEMP_MIN_HEIGHTS_TO_FIT):
        return result

    z0h_hi = min(float(GRYNING_Z0H_MAX), 0.95 * min(float(np.nanmin(z)), float(z_ref)))
    z0h_hi = max(z0h_hi, float(GRYNING_Z0H_MIN) * 10.0)
    if z0h_hi <= float(GRYNING_Z0H_MIN):
        return result
    z0h_grid = np.geomspace(float(GRYNING_Z0H_MIN), z0h_hi, int(GRYNING_Z0H_GRID_N))

    y = temp_k - float(t_ref_k)
    best = None
    for z0h in z0h_grid:
        hshape = most_temperature_profile_shape(z, float(z0h), L, z_ref=z_ref)
        finite = np.isfinite(hshape) & np.isfinite(y)
        if int(finite.sum()) < int(GRYNING_TEMP_MIN_HEIGHTS_TO_FIT):
            continue
        denom = float(np.sum(hshape[finite] ** 2))
        if denom <= 1.0e-14:
            continue
        beta = float(np.sum(hshape[finite] * y[finite]) / denom)
        theta_star = float(np.clip(KAPPA * beta, GRYNING_THETA_STAR_MIN_K, GRYNING_THETA_STAR_MAX_K))
        pred = float(t_ref_k) + (theta_star / KAPPA) * hshape[finite]
        err = pred - temp_k[finite]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        rec = (rmse, mae, float(z0h), theta_star, int(finite.sum()))
        if best is None or rec[:2] < best[:2]:
            best = rec

    if best is None:
        return result
    rmse, mae, z0h, theta_star, nfit = best
    hfx_inv = np.nan
    if np.isfinite(psfc) and np.isfinite(ustar_for_hfx) and psfc > 1000.0 and ustar_for_hfx > 1.0e-6 and np.isfinite(t_ref_k):
        rho = float(psfc) / (RD_AIR * max(float(t_ref_k), 150.0))
        hfx_inv = -rho * CP_AIR * float(ustar_for_hfx) * float(theta_star)
    result.update({
        "z0h_inv_m": float(z0h),
        "theta_star_inv_K": float(theta_star),
        "HFX_inv_Wm2": float(hfx_inv) if np.isfinite(hfx_inv) else np.nan,
        "T_profile_rmse_K": float(rmse),
        "T_profile_mae_K": float(mae),
        "T_fit_status": "ok",
        "n_T_heights_used_in_fit": int(nfit),
    })
    return result



def _hampel_outlier_mask_gryning(y: pd.Series,
                                 window: int = GRYNING_SPIKE_ROLLING_WINDOW,
                                 nsigma: float = GRYNING_SPIKE_HAMPEL_NSIGMA) -> pd.Series:
    """Return True where y is a local Hampel outlier."""
    x = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < max(5, int(window)):
        return pd.Series(False, index=x.index)
    w = int(max(3, window))
    med = x.rolling(w, center=True, min_periods=max(3, w // 2)).median()
    mad = (x - med).abs().rolling(w, center=True, min_periods=max(3, w // 2)).median()
    scale = 1.4826 * mad.replace(0.0, np.nan)
    flag = (x - med).abs() > float(nsigma) * scale
    return flag.fillna(False)


def _gryning_range_jump_hampel_keep(series: pd.Series, kind: str) -> pd.Series:
    """True means a time series value passes range + local spike/jump checks."""
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = x.notna()
    y_for_spike = x
    jump_thr = None

    if kind in {"ws", "u"}:
        lo, hi = GRYNING_WS_METRIC_RANGE
        ok = ok & (x >= lo) & (x <= hi)
        jump_thr = GRYNING_WS_JUMP_THRESHOLD
    elif kind in {"temp", "T"}:
        lo, hi = GRYNING_TEMP_METRIC_RANGE_K
        ok = ok & (x >= lo) & (x <= hi)
        jump_thr = GRYNING_TEMP_JUMP_THRESHOLD_K
    elif kind == "ustar":
        ok = ok & (x >= GRYNING_USTAR_MIN) & (x <= min(GRYNING_USTAR_MAX, 5.0))
        jump_thr = GRYNING_USTAR_JUMP_THRESHOLD
    elif kind == "z0":
        ok = ok & (x >= GRYNING_Z0_MIN) & (x <= GRYNING_Z0_MAX)
        y_for_spike = np.log10(x.where(x > 0.0))
        jump_thr = GRYNING_LOG10_Z0_JUMP_THRESHOLD
    elif kind == "z0h":
        ok = ok & (x >= GRYNING_Z0H_MIN) & (x <= GRYNING_Z0H_MAX)
        y_for_spike = np.log10(x.where(x > 0.0))
        jump_thr = GRYNING_LOG10_Z0_JUMP_THRESHOLD
    else:
        # Generic: only finite + Hampel, no hard range.
        jump_thr = None

    if jump_thr is not None:
        dy = pd.to_numeric(y_for_spike, errors="coerce").diff().abs()
        jump_flag = (dy > float(jump_thr)) | (dy.shift(-1) > float(jump_thr))
    else:
        jump_flag = pd.Series(False, index=x.index)
    hampel_flag = _hampel_outlier_mask_gryning(pd.Series(y_for_spike, index=x.index))
    return ok & (~jump_flag.fillna(False)) & (~hampel_flag.fillna(False))


def _gryning_clean_series(series: pd.Series, kind: str) -> pd.Series:
    """Return series with obvious spikes/out-of-range values replaced by NaN."""
    if not GRYNING_USE_ROBUST_FILTERS:
        return pd.to_numeric(series, errors="coerce")
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = _gryning_range_jump_hampel_keep(x, kind)
    return x.where(keep)


def _apply_gryning_input_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter OBS/WRF wind-speed and temperature profile columns before inversion."""
    if (not GRYNING_USE_ROBUST_FILTERS) or (not GRYNING_FILTER_INPUTS_BEFORE_INVERSION):
        return df
    out = df.copy()
    for c in list(out.columns):
        if c == TIME_COL:
            continue
        wind = parse_wind_column(c)
        temp = parse_temp_column(c)
        if wind and wind[1] == "WS":
            out[c] = _gryning_clean_series(out[c], "ws")
        elif temp:
            out[c] = _gryning_clean_series(out[c], "temp")
    return out


def _add_gryning_output_filter_columns(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Add filtered z0/u* columns and a final filter gate for plots/summaries."""
    if ts_df is None or ts_df.empty or (not GRYNING_USE_ROBUST_FILTERS):
        return ts_df
    out = ts_df.copy()
    out["GRYNING_output_filter_gate"] = 0
    out["z0_gryning_m_filtered"] = np.nan
    out["ustar_gryning_mps_filtered"] = np.nan
    out["z0_log10_diff_gryning_minus_wrf_filtered"] = np.nan
    out["ustar_diff_gryning_minus_wrf_mps_filtered"] = np.nan
    group_cols = [c for c in ["relative_file", "dataset_version"] if c in out.columns]
    groups = out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]
    ok_status = {"ok", "ok_not_strictly_converged", "single_height_used_wrf_z0", "insufficient_heights_used_wrf_surface_z0_ustar"}
    for _, g in groups:
        idx = g.index
        z0_keep = _gryning_range_jump_hampel_keep(g["z0_gryning_m"], "z0") if "z0_gryning_m" in g else pd.Series(False, index=idx)
        ust_keep = _gryning_range_jump_hampel_keep(g["ustar_gryning_mps"], "ustar") if "ustar_gryning_mps" in g else pd.Series(False, index=idx)
        fit_ok = g["fit_status"].astype(str).isin(ok_status) if "fit_status" in g else pd.Series(True, index=idx)
        n_ok = pd.to_numeric(g.get("n_heights_used", pd.Series(0, index=idx)), errors="coerce") >= GRYNING_MIN_HEIGHTS_TO_FIT
        rmse_ok = pd.to_numeric(g.get("profile_rmse_mps", pd.Series(np.nan, index=idx)), errors="coerce") <= GRYNING_PROFILE_RMSE_MAX_MPS
        # Allow WRF-surface fallback rows where profile_rmse is NaN but status is explicit fallback.
        fallback_ok = g["fit_status"].astype(str).str.contains("fallback|single_height", case=False, na=False) if "fit_status" in g else pd.Series(False, index=idx)
        final = z0_keep & ust_keep & fit_ok & (rmse_ok | fallback_ok) & (n_ok | fallback_ok)
        out.loc[idx, "GRYNING_output_filter_gate"] = final.astype(int).values
        out.loc[idx, "z0_gryning_m_filtered"] = pd.to_numeric(g["z0_gryning_m"], errors="coerce").where(final).values
        out.loc[idx, "ustar_gryning_mps_filtered"] = pd.to_numeric(g["ustar_gryning_mps"], errors="coerce").where(final).values
        if "z0_log10_diff_gryning_minus_wrf" in g:
            out.loc[idx, "z0_log10_diff_gryning_minus_wrf_filtered"] = pd.to_numeric(g["z0_log10_diff_gryning_minus_wrf"], errors="coerce").where(final).values
        if "ustar_diff_gryning_minus_wrf_mps" in g:
            out.loc[idx, "ustar_diff_gryning_minus_wrf_mps_filtered"] = pd.to_numeric(g["ustar_diff_gryning_minus_wrf_mps"], errors="coerce").where(final).values
    return out


def _filter_profile_point_table(points: pd.DataFrame, var_kind: str) -> pd.DataFrame:
    """Filter OBS/WRF/model profile point time series before mean-profile plots and metrics."""
    if points is None or points.empty or (not GRYNING_USE_ROBUST_FILTERS) or (not GRYNING_FILTER_PROFILE_POINTS_BEFORE_METRICS):
        return points
    out = points.copy()
    if var_kind == "ws":
        value_cols = [
            "obs_ws_mps", "wrf_ws_mps", "U_used_for_fit_mps",
            "U_GRYNING_WRFparam_mps", "U_GRYNING_INVzu_mps",
            "U_LOGMOST_WRFparam_mps", "U_LOGMOST_INVzu_mps",
        ]
        kind = "ws"
    else:
        value_cols = [
            "obs_T_K", "wrf_T_K", "T_used_K",
            "T_MOST_WRFparam_K", "T_MOST_INVzu_WRFhfx_K", "T_MOST_INVheat_K",
        ]
        kind = "temp"
    group_cols = [c for c in ["relative_file", "dataset_version", "height_label"] if c in out.columns]
    if not group_cols:
        group_cols = ["height_m"] if "height_m" in out.columns else []
    groups = out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]
    for _, g in groups:
        idx = g.index
        for c in value_cols:
            if c not in out.columns:
                continue
            s = pd.to_numeric(g[c], errors="coerce")
            keep = _gryning_range_jump_hampel_keep(s, kind)
            out.loc[idx, c] = s.where(keep).values
    return out

def _param_pair_metrics(a: pd.Series, b: pd.Series, prefix: str) -> Dict[str, float]:
    """Metrics where b-a is reported as candidate minus reference."""
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(m.sum()) == 0:
        return {f"{prefix}_N": 0, f"{prefix}_bias": np.nan, f"{prefix}_MAE": np.nan, f"{prefix}_RMSE": np.nan, f"{prefix}_Pearson_r": np.nan}
    e = bb[m] - aa[m]
    return {
        f"{prefix}_N": int(m.sum()),
        f"{prefix}_bias": float(np.mean(e)),
        f"{prefix}_MAE": float(np.mean(np.abs(e))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(e ** 2))),
        f"{prefix}_Pearson_r": safe_pearson(aa[m], bb[m]),
    }


def _plot_parameter_timeseries(ts: pd.DataFrame, out_png: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip {out_png.name}: {exc}")
        return False
    try:
        t = pd.to_datetime(ts[TIME_COL], errors="coerce")
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        z0_plot_col = "z0_gryning_m_filtered" if "z0_gryning_m_filtered" in ts.columns else "z0_gryning_m"
        ust_plot_col = "ustar_gryning_mps_filtered" if "ustar_gryning_mps_filtered" in ts.columns else "ustar_gryning_mps"
        z0diff_plot_col = "z0_log10_diff_gryning_minus_wrf_filtered" if "z0_log10_diff_gryning_minus_wrf_filtered" in ts.columns else "z0_log10_diff_gryning_minus_wrf"
        ustdiff_plot_col = "ustar_diff_gryning_minus_wrf_mps_filtered" if "ustar_diff_gryning_minus_wrf_mps_filtered" in ts.columns else "ustar_diff_gryning_minus_wrf_mps"
        axes[0].plot(t, pd.to_numeric(ts[z0_plot_col], errors="coerce"), label="Gryning-inverted z0 filtered", linewidth=0.9)
        axes[0].plot(t, pd.to_numeric(ts["wrf_z0_m"], errors="coerce"), label="WRF Z0", linewidth=0.9, alpha=0.85)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("z0 (m, log)")
        axes[0].legend(loc="best", ncol=2)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, pd.to_numeric(ts[z0diff_plot_col], errors="coerce"), label="log10(z0_G) - log10(z0_WRF) filtered", linewidth=0.9)
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_ylabel("log10 z0 diff")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, pd.to_numeric(ts[ust_plot_col], errors="coerce"), label="Gryning-inverted u* filtered", linewidth=0.9)
        axes[2].plot(t, pd.to_numeric(ts["wrf_ustar_mps"], errors="coerce"), label="WRF UST", linewidth=0.9, alpha=0.85)
        axes[2].set_ylabel("u* (m/s)")
        axes[2].legend(loc="best", ncol=2)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, pd.to_numeric(ts[ustdiff_plot_col], errors="coerce"), label="u*_G - u*_WRF filtered", linewidth=0.9)
        axes[3].axhline(0.0, linewidth=0.8)
        axes[3].set_ylabel("u* diff")
        axes[3].set_xlabel("Time")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=GRYNING_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save parameter timeseries plot {out_png}: {exc}")
        return False


def _plot_profile_mean(df: pd.DataFrame, out_png: Path, title: str, x_cols: List[Tuple[str, str]], y_col: str = "height_m", xlabel: str = "") -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip {out_png.name}: {exc}")
        return False
    try:
        if df.empty or y_col not in df.columns:
            return False
        g = df.copy()
        g[y_col] = pd.to_numeric(g[y_col], errors="coerce")
        mean_df = g.groupby(y_col, as_index=False).mean(numeric_only=True).sort_values(y_col)
        fig, ax = plt.subplots(figsize=(6.5, 8.0))
        plotted = False
        for c, lab in x_cols:
            if c in mean_df.columns and pd.to_numeric(mean_df[c], errors="coerce").notna().any():
                ax.plot(pd.to_numeric(mean_df[c], errors="coerce"), mean_df[y_col], marker="o", label=lab)
                plotted = True
        if not plotted:
            plt.close(fig)
            return False
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Height (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=GRYNING_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save profile plot {out_png}: {exc}")
        return False


def _plot_metric_summary(metrics_df: pd.DataFrame, out_png: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip {out_png.name}: {exc}")
        return False
    try:
        if metrics_df.empty:
            return False
        # Plot selected interpretable metrics if present. One plot per source file.
        row = metrics_df.iloc[0]
        names = []
        vals = []
        for c, lab in [
            ("ustar_gryning_vs_wrf_RMSE", "u* RMSE"),
            ("ustar_gryning_vs_wrf_MAE", "u* MAE"),
            ("log10_z0_gryning_vs_wrf_RMSE", "log10 z0 RMSE"),
            ("log10_z0_gryning_vs_wrf_MAE", "log10 z0 MAE"),
        ]:
            if c in row.index and np.isfinite(row[c]):
                names.append(lab)
                vals.append(float(row[c]))
        if not vals:
            return False
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(names, vals)
        ax.set_title(title)
        ax.set_ylabel("Metric value")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=GRYNING_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save metric plot {out_png}: {exc}")
        return False




def _plot_profile_metric_by_height(
    metrics_df: pd.DataFrame,
    out_png: Path,
    title: str,
    method_prefixes: List[Tuple[str, str]],
    metric_suffix: str = "RMSE",
) -> bool:
    """Plot profile metric values against height for several diagnostic methods."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠ matplotlib unavailable; skip {out_png.name}: {exc}")
        return False
    try:
        if metrics_df is None or metrics_df.empty or "height_m" not in metrics_df.columns:
            return False
        g = metrics_df.copy()
        g["height_m"] = pd.to_numeric(g["height_m"], errors="coerce")
        g = g.sort_values("height_m")
        fig, ax = plt.subplots(figsize=(7.0, 7.5))
        plotted = False
        for prefix, lab in method_prefixes:
            c = f"{prefix}_{metric_suffix}"
            if c in g.columns and pd.to_numeric(g[c], errors="coerce").notna().any():
                ax.plot(pd.to_numeric(g[c], errors="coerce"), g["height_m"], marker="o", label=lab)
                plotted = True
        if not plotted:
            plt.close(fig)
            return False
        ax.set_xlabel(metric_suffix)
        ax.set_ylabel("Height (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=GRYNING_PLOT_DPI)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"⚠ Failed to save profile metric plot {out_png}: {exc}")
        return False


def build_gryning_profile_inversion_for_dataset(
    df: pd.DataFrame,
    csv_path: Path,
    rel_path: str,
    dataset_version: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build Gryning z0/u_star inversion for one ML-ready wide CSV.

    Priority rule at each timestamp/height:
      1. use OBS_WS if available and valid;
      2. otherwise use WRF_WS at the same height.
    """
    if df is None or df.empty or TIME_COL not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Add existing WRF-surface MOST T columns when possible; this does not alter source files.
    df = add_most_temperature_profile_columns(df)
    df = _apply_gryning_input_filters(df)

    # Runtime throttle: diagnosing every 10-min row for a full year can take hours.
    # For a first-pass physical diagnosis, stride=6 gives hourly samples while keeping
    # all columns/heights and the same OBS-first/WRF-fallback logic.
    original_n_rows = len(df)
    stride = int(max(1, GRYNING_ROW_STRIDE))
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    if GRYNING_MAX_ROWS_PER_FILE is not None:
        df = df.iloc[:int(GRYNING_MAX_ROWS_PER_FILE)].reset_index(drop=True)
    if original_n_rows != len(df):
        print(f"  Runtime throttle: using {len(df)} / {original_n_rows} rows "
              f"(stride={stride}, max_rows={GRYNING_MAX_ROWS_PER_FILE})")

    cols = list(df.columns)
    height_labels = _collect_ws_height_labels_for_gryning(cols)
    if not height_labels:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pblh_col = _find_scalar_col(cols, ["WRF_PBLH (m)", "PBLH (m)", "WRF_PBLH"])
    z0_col = _find_scalar_col(cols, ["Z0 (m)", "Z0"])
    ust_col = _find_scalar_col(cols, ["UST (m/s)", "UST"])
    L_col = _find_scalar_col(cols, ["WRF_L (m)", "L (m)", "WRF_L"])
    hfx_col = _find_scalar_col(cols, ["HFX (W/m2)", "HFX"])
    psfc_col = _find_scalar_col(cols, ["PSFC (Pa)", "PSFC"])
    t2_col = _find_scalar_col(cols, ["T2 (K)", "T2"])

    lat_deg = infer_station_latitude(rel_path)
    coriolis_f = coriolis_parameter_from_lat(lat_deg)

    rows = []
    point_rows = []
    t_point_rows = []
    t_all = pd.to_datetime(df[TIME_COL], errors="coerce")

    ws_col_map = []
    for h_label in height_labels:
        obs_col = _find_obs_ws_col(cols, h_label)
        wrf_col = _find_wrf_ws_col(cols, h_label)
        if obs_col is None and wrf_col is None:
            continue
        ws_col_map.append((h_label, height_label_to_float(h_label), obs_col, wrf_col))

    temp_col_map = []
    if GRYNING_RUN_T_PROFILE_DIAGNOSTICS:
        for h_label in _collect_temp_height_labels_for_gryning(cols):
            obs_t_col = _find_obs_t_col(cols, h_label)
            wrf_t_col = _find_wrf_t_col(cols, h_label)
            t_mo_col = find_col(cols, f"T_MO{h_label}")
            if obs_t_col is None and wrf_t_col is None and t_mo_col is None:
                continue
            temp_col_map.append((h_label, height_label_to_float(h_label), obs_t_col, wrf_t_col, t_mo_col))

    for i in range(len(df)):
        if GRYNING_PROGRESS_EVERY_ROWS and i > 0 and i % int(GRYNING_PROGRESS_EVERY_ROWS) == 0:
            print(f"  Progress {dataset_version} | {rel_path}: {i}/{len(df)} rows")
        t = t_all.iloc[i]
        if pd.isna(t):
            continue
        z_vals = []
        u_vals = []
        src_vals = []
        h_used = []
        n_obs = 0
        n_wrf = 0
        n_missing = 0
        per_height_cache = []

        for h_label, h_m, obs_col, wrf_col in ws_col_map:
            obs_v = _scalar_at_row(df, i, obs_col)
            wrf_v = _scalar_at_row(df, i, wrf_col)
            val = np.nan
            src = "missing"
            if np.isfinite(obs_v) and GRYNING_MIN_WIND_SPEED <= obs_v <= GRYNING_MAX_WIND_SPEED:
                val = float(obs_v)
                src = "OBS"
            elif np.isfinite(wrf_v) and GRYNING_MIN_WIND_SPEED <= wrf_v <= GRYNING_MAX_WIND_SPEED:
                val = float(wrf_v)
                src = "WRF_fallback"
            if np.isfinite(val):
                z_vals.append(float(h_m))
                u_vals.append(float(val))
                src_vals.append(src)
                h_used.append(h_label)
                n_obs += int(src == "OBS")
                n_wrf += int(src == "WRF_fallback")
            else:
                n_missing += 1
            per_height_cache.append((h_label, h_m, obs_v, wrf_v, val, src))

        pblh = _scalar_at_row(df, i, pblh_col)
        z0_hint = _scalar_at_row(df, i, z0_col)
        ust_hint = _scalar_at_row(df, i, ust_col)
        L_raw = _scalar_at_row(df, i, L_col)

        fit = fit_gryning_z0_ustar(
            np.asarray(z_vals), np.asarray(u_vals),
            pblh=pblh, L=L_raw, coriolis_f=coriolis_f,
            z0_hint=z0_hint, ustar_hint=ust_hint,
        )
        z0_fit = fit.get("z0_gryning_m", np.nan)
        ust_fit = fit.get("ustar_gryning_mps", np.nan)
        zi_fit = fit.get("zi_used_m", np.nan)
        lmb_fit = fit.get("lmb_used_m", np.nan)
        L_eff = fit.get("L_used_m", np.nan)

        wrf_z0 = float(z0_hint) if np.isfinite(z0_hint) else np.nan
        wrf_ust = float(ust_hint) if np.isfinite(ust_hint) else np.nan
        z0_diff = z0_fit - wrf_z0 if np.isfinite(z0_fit) and np.isfinite(wrf_z0) else np.nan
        ust_diff = ust_fit - wrf_ust if np.isfinite(ust_fit) and np.isfinite(wrf_ust) else np.nan
        z0_ratio = z0_fit / wrf_z0 if np.isfinite(z0_fit) and np.isfinite(wrf_z0) and wrf_z0 > 0 else np.nan
        ust_ratio = ust_fit / wrf_ust if np.isfinite(ust_fit) and np.isfinite(wrf_ust) and wrf_ust > 0 else np.nan
        log10_z0_g = math.log10(z0_fit) if np.isfinite(z0_fit) and z0_fit > 0 else np.nan
        log10_z0_w = math.log10(wrf_z0) if np.isfinite(wrf_z0) and wrf_z0 > 0 else np.nan

        row = {
            "source_file": str(csv_path),
            "relative_file": rel_path,
            "dataset_version": dataset_version,
            TIME_COL: t,
            "latitude_deg_used": lat_deg,
            "coriolis_f_s-1": coriolis_f,
            "n_available_heights_total": len(ws_col_map),
            "n_heights_used": len(z_vals),
            "n_obs_heights_used": int(n_obs),
            "n_wrf_fallback_heights_used": int(n_wrf),
            "n_missing_heights": int(n_missing),
            "height_labels_used": ";".join(h_used),
            "source_sequence_by_height": ";".join(src_vals),
            "z0_gryning_m": z0_fit,
            "wrf_z0_m": wrf_z0,
            "z0_diff_gryning_minus_wrf_m": z0_diff,
            "z0_ratio_gryning_over_wrf": z0_ratio,
            "log10_z0_gryning": log10_z0_g,
            "log10_z0_wrf": log10_z0_w,
            "z0_log10_diff_gryning_minus_wrf": log10_z0_g - log10_z0_w if np.isfinite(log10_z0_g) and np.isfinite(log10_z0_w) else np.nan,
            "ustar_gryning_mps": ust_fit,
            "wrf_ustar_mps": wrf_ust,
            "ustar_diff_gryning_minus_wrf_mps": ust_diff,
            "ustar_ratio_gryning_over_wrf": ust_ratio,
            "profile_rmse_mps": fit.get("profile_rmse_mps", np.nan),
            "profile_mae_mps": fit.get("profile_mae_mps", np.nan),
            "fit_status": fit.get("fit_status", ""),
            "fit_method": fit.get("fit_method", ""),
            "stability_class": fit.get("stability_class", ""),
            "L_raw_m": L_raw,
            "L_used_m": L_eff,
            "pblh_used_m": fit.get("pblh_used_m", np.nan),
            "zi_used_m": zi_fit,
            "lmb_used_m": lmb_fit,
            "wrf_pblh_raw_m": pblh,
        }
        rows.append(row)

        if GRYNING_SAVE_PROFILE_POINTS:
            zi_wrf = float(pblh) if np.isfinite(pblh) and pblh > 0 else float(GRYNING_PBLH_FALLBACK_M)
            if per_height_cache:
                zi_wrf = max(zi_wrf, 1.10 * max(float(x[1]) for x in per_height_cache if np.isfinite(x[1])))
            lmbl_wrf = gryning_lmbl_eq31(wrf_ust, wrf_z0, L_raw, coriolis_f)
            if not np.isfinite(lmbl_wrf):
                lmbl_wrf = float(GRYNING_LMBL_FALLBACK_M)
            for h_label, z, obs_v, wrf_v, used_v, src in per_height_cache:
                u_gryning_invzu = gryning_wind_speed_at_height(
                    z=float(z), z0m=float(z0_fit) if np.isfinite(z0_fit) else np.nan,
                    ustar=float(ust_fit) if np.isfinite(ust_fit) else np.nan, L=L_raw,
                    zi=float(zi_fit) if np.isfinite(zi_fit) else np.nan,
                    lmbl=float(lmb_fit) if np.isfinite(lmb_fit) else np.nan,
                )
                u_gryning_wrfparam = gryning_wind_speed_at_height(
                    z=float(z), z0m=wrf_z0, ustar=wrf_ust, L=L_raw, zi=zi_wrf, lmbl=lmbl_wrf,
                )
                u_log_wrfparam = log_most_wind_speed_at_height(float(z), wrf_z0, wrf_ust, L_raw)
                u_log_invzu = log_most_wind_speed_at_height(
                    float(z), float(z0_fit) if np.isfinite(z0_fit) else np.nan,
                    float(ust_fit) if np.isfinite(ust_fit) else np.nan, L_raw,
                )
                point_rows.append({
                    "source_file": str(csv_path),
                    "relative_file": rel_path,
                    "dataset_version": dataset_version,
                    TIME_COL: t,
                    "height_label": h_label,
                    "height_m": float(z),
                    "obs_ws_mps": float(obs_v) if np.isfinite(obs_v) else np.nan,
                    "wrf_ws_mps": float(wrf_v) if np.isfinite(wrf_v) else np.nan,
                    "ws_used_mps": float(used_v) if np.isfinite(used_v) else np.nan,
                    "ws_source": src,
                    "z0_gryning_m": z0_fit,
                    "ustar_gryning_mps": ust_fit,
                    "wrf_z0_m": wrf_z0,
                    "wrf_ustar_mps": wrf_ust,
                    "lmbl_gryning_invzu_m": lmb_fit,
                    "lmbl_gryning_wrfparam_m": lmbl_wrf,
                    "zi_gryning_invzu_m": zi_fit,
                    "zi_gryning_wrfparam_m": zi_wrf,
                    "U_GRYNING_INVzu_mps": u_gryning_invzu,
                    "U_GRYNING_WRFparam_mps": u_gryning_wrfparam,
                    "U_LOGMOST_INVzu_mps": u_log_invzu,
                    "U_LOGMOST_WRFparam_mps": u_log_wrfparam,
                    # Backward-compatible alias used by earlier v6 metrics.
                    "ws_gryning_fit_mps": u_gryning_invzu,
                    "fit_residual_vs_used_mps": u_gryning_invzu - float(used_v) if np.isfinite(u_gryning_invzu) and np.isfinite(used_v) else np.nan,
                    "fit_status": fit.get("fit_status", ""),
                })

        if GRYNING_RUN_T_PROFILE_DIAGNOSTICS and temp_col_map:
            t2 = _to_kelvin_value(_scalar_at_row(df, i, t2_col))
            hfx = _scalar_at_row(df, i, hfx_col)
            psfc = _scalar_at_row(df, i, psfc_col)
            temp_fit_cache = []
            temp_z_vals = []
            temp_used_vals = []
            temp_src_vals = []
            for h_label, z, obs_t_col, wrf_t_col, t_mo_col in temp_col_map:
                obs_t = _to_kelvin_value(_scalar_at_row(df, i, obs_t_col))
                wrf_t = _to_kelvin_value(_scalar_at_row(df, i, wrf_t_col))
                used_t = np.nan
                t_src = "missing"
                tmin, tmax = GRYNING_TEMP_VALID_RANGE_K
                if np.isfinite(obs_t) and tmin <= obs_t <= tmax:
                    used_t = float(obs_t)
                    t_src = "OBS"
                elif np.isfinite(wrf_t) and tmin <= wrf_t <= tmax:
                    used_t = float(wrf_t)
                    t_src = "WRF_fallback"
                if np.isfinite(used_t):
                    temp_z_vals.append(float(z))
                    temp_used_vals.append(float(used_t))
                    temp_src_vals.append(t_src)
                temp_fit_cache.append((h_label, z, obs_t, wrf_t, used_t, t_src, t_mo_col))

            ustar_for_heat = float(ust_fit) if np.isfinite(ust_fit) else wrf_ust
            temp_fit = fit_most_temperature_z0h_thetastar(
                np.asarray(temp_z_vals), np.asarray(temp_used_vals),
                t_ref_k=t2, L=L_raw, psfc=psfc, ustar_for_hfx=ustar_for_heat,
                z_ref=GRYNING_TEMP_REF_HEIGHT_M,
            ) if GRYNING_TEMP_FIT_INVHEAT else {}
            row.update({
                "z0h_inv_m": temp_fit.get("z0h_inv_m", np.nan),
                "theta_star_inv_K": temp_fit.get("theta_star_inv_K", np.nan),
                "HFX_inv_Wm2": temp_fit.get("HFX_inv_Wm2", np.nan),
                "T_profile_rmse_K": temp_fit.get("T_profile_rmse_K", np.nan),
                "T_profile_mae_K": temp_fit.get("T_profile_mae_K", np.nan),
                "T_fit_status": temp_fit.get("T_fit_status", "disabled_or_no_T_profile"),
                "T_fit_method": temp_fit.get("T_fit_method", ""),
                "n_T_heights_used": len(temp_z_vals),
                "n_T_obs_heights_used": int(sum(1 for x in temp_src_vals if x == "OBS")),
                "n_T_wrf_fallback_heights_used": int(sum(1 for x in temp_src_vals if x == "WRF_fallback")),
            })

            z0h_inv = temp_fit.get("z0h_inv_m", np.nan)
            theta_star_inv = temp_fit.get("theta_star_inv_K", np.nan)
            hfx_inv = temp_fit.get("HFX_inv_Wm2", np.nan)
            for h_label, z, obs_t, wrf_t, used_t, t_src, t_mo_col in temp_fit_cache:
                t_mo_wrf = _to_kelvin_value(_scalar_at_row(df, i, t_mo_col))
                if not np.isfinite(t_mo_wrf):
                    t_mo_wrf = gryning_temperature_from_surface(
                        z=float(z), t2_k=t2, ustar=wrf_ust, z0m=wrf_z0, L=L_raw, hfx=hfx, psfc=psfc,
                    )
                t_mo_invzu = gryning_temperature_from_surface(
                    z=float(z), t2_k=t2, ustar=float(ust_fit) if np.isfinite(ust_fit) else np.nan,
                    z0m=float(z0_fit) if np.isfinite(z0_fit) else np.nan, L=L_raw,
                    hfx=hfx, psfc=psfc,
                )
                t_mo_invheat = most_temperature_from_theta_star(
                    z=float(z), t_ref_k=t2, z0h=z0h_inv, theta_star=theta_star_inv, L=L_raw,
                    z_ref=GRYNING_TEMP_REF_HEIGHT_M,
                )
                if not any(np.isfinite(x) for x in [obs_t, wrf_t, used_t, t_mo_wrf, t_mo_invzu, t_mo_invheat]):
                    continue
                t_point_rows.append({
                    "source_file": str(csv_path),
                    "relative_file": rel_path,
                    "dataset_version": dataset_version,
                    TIME_COL: t,
                    "height_label": h_label,
                    "height_m": float(z),
                    "obs_T_K": obs_t,
                    "wrf_T_K": wrf_t,
                    "T_used_K": used_t,
                    "T_source": t_src,
                    "T_MOST_WRFparam_K": t_mo_wrf,
                    "T_MOST_INVzu_WRFhfx_K": t_mo_invzu,
                    "T_MOST_INVheat_K": t_mo_invheat,
                    # Backward-compatible aliases from v6.
                    "T_MO_WRFsurface_K": t_mo_wrf,
                    "T_MO_GRYNINGsurface_K": t_mo_invzu,
                    "z0_gryning_m": z0_fit,
                    "ustar_gryning_mps": ust_fit,
                    "wrf_z0_m": wrf_z0,
                    "wrf_ustar_mps": wrf_ust,
                    "z0h_inv_m": z0h_inv,
                    "theta_star_inv_K": theta_star_inv,
                    "HFX_inv_Wm2": hfx_inv,
                    "T_profile_rmse_K": temp_fit.get("T_profile_rmse_K", np.nan),
                    "T_profile_mae_K": temp_fit.get("T_profile_mae_K", np.nan),
                    "T_fit_status": temp_fit.get("T_fit_status", ""),
                    "L_raw_m": L_raw,
                    "HFX_WRF_Wm2": hfx,
                    "HFX_Wm2": hfx,
                    "PSFC_Pa": psfc,
                    "T2_K": t2,
                })

    ts_df = pd.DataFrame(rows)
    pts_df = pd.DataFrame(point_rows)
    temp_pts_df = pd.DataFrame(t_point_rows)

    ts_df = _add_gryning_output_filter_columns(ts_df)
    pts_df = _filter_profile_point_table(pts_df, "ws")
    temp_pts_df = _filter_profile_point_table(temp_pts_df, "temp")

    if not ts_df.empty:
        rel_safe = safe_name(rel_path.replace("/", "__").replace("\\", "__"))
        stem = f"{safe_name(dataset_version)}__{rel_safe}"
        ts_root = ensure_dir(out_dir / "13_gryning_z0_ustar_timeseries")
        ts_path = ts_root / f"{stem}_gryning_z0_ustar_timeseries.csv"
        ts_df.to_csv(ts_path, index=False, encoding="utf-8-sig")

        if GRYNING_SAVE_PARAMETER_PLOTS:
            plot_root = ensure_dir(out_dir / "16_gryning_vs_wrf_parameter_timeseries_plots")
            _plot_parameter_timeseries(ts_df, plot_root / f"{stem}_z0_ustar_timeseries.png", f"{dataset_version} | {rel_path} | Gryning vs WRF z0/u*")

        if GRYNING_SAVE_PROFILE_POINTS and not pts_df.empty:
            pts_root = ensure_dir(out_dir / "14_gryning_profile_points")
            pts_df.to_csv(pts_root / f"{stem}_gryning_U_profile_points.csv", index=False, encoding="utf-8-sig")
            if GRYNING_SAVE_PROFILE_PLOTS:
                uprof_root = ensure_dir(out_dir / "18_gryning_U_profile_plots")
                _plot_profile_mean(
                    pts_df, uprof_root / f"{stem}_mean_U_profile.png",
                    f"{dataset_version} | {rel_path} | mean wind-speed profile",
                    [
                        ("obs_ws_mps", "OBS"),
                        ("wrf_ws_mps", "WRF"),
                        ("U_GRYNING_WRFparam_mps", "Gryning/MOST all-WRF params"),
                        ("U_GRYNING_INVzu_mps", "Gryning/MOST inv z0/u*"),
                        ("U_LOGMOST_WRFparam_mps", "log-MOST all-WRF params"),
                        ("U_LOGMOST_INVzu_mps", "log-MOST inv z0/u*"),
                    ],
                    xlabel="Wind speed U (m/s)",
                )

        if GRYNING_RUN_T_PROFILE_DIAGNOSTICS and not temp_pts_df.empty:
            t_root = ensure_dir(out_dir / "19_gryning_T_profile_points")
            temp_pts_df.to_csv(t_root / f"{stem}_T_profile_points.csv", index=False, encoding="utf-8-sig")
            if GRYNING_SAVE_PROFILE_PLOTS:
                tplot_root = ensure_dir(out_dir / "20_gryning_T_profile_plots")
                _plot_profile_mean(
                    temp_pts_df, tplot_root / f"{stem}_mean_T_profile.png",
                    f"{dataset_version} | {rel_path} | mean temperature profile",
                    [
                        ("obs_T_K", "OBS T"),
                        ("wrf_T_K", "WRF T"),
                        ("T_MOST_WRFparam_K", "MOST-T WRF z0/u*/HFX"),
                        ("T_MOST_INVzu_WRFhfx_K", "MOST-T inv z0/u*, WRF HFX"),
                        ("T_MOST_INVheat_K", "MOST-T inv z0h/theta*"),
                    ],
                    xlabel="Temperature (K)",
                )

    return ts_df, pts_df, temp_pts_df


def summarize_gryning_inversion(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Gryning z0/u_star inversion and WRF comparison by file/tower."""
    if ts_df is None or ts_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["relative_file", "dataset_version"]
    ok_status = {"ok", "ok_not_strictly_converged", "single_height_used_wrf_z0", "insufficient_heights_used_wrf_surface_z0_ustar"}
    for keys, g in ts_df.groupby(group_cols, dropna=False):
        rel_path, dataset_version = keys
        ok = g[g["fit_status"].astype(str).isin(ok_status)].copy()
        z0_col = "z0_gryning_m_filtered" if "z0_gryning_m_filtered" in ok.columns else "z0_gryning_m"
        ust_col = "ustar_gryning_mps_filtered" if "ustar_gryning_mps_filtered" in ok.columns else "ustar_gryning_mps"
        z0 = pd.to_numeric(ok[z0_col], errors="coerce") if not ok.empty else pd.Series(dtype=float)
        ust = pd.to_numeric(ok[ust_col], errors="coerce") if not ok.empty else pd.Series(dtype=float)
        row = {
            "relative_file": rel_path,
            "dataset_version": dataset_version,
            "n_times_total": int(len(g)),
            "n_times_ok_or_fallback": int(len(ok)),
            "ok_ratio": float(len(ok) / len(g)) if len(g) else np.nan,
            "filtered_keep_ratio": float(pd.to_numeric(g.get("GRYNING_output_filter_gate", pd.Series(np.nan, index=g.index)), errors="coerce").mean()) if "GRYNING_output_filter_gate" in g.columns else np.nan,
            "z0_mean_m": float(z0.mean()) if z0.notna().any() else np.nan,
            "z0_median_m": float(z0.median()) if z0.notna().any() else np.nan,
            "z0_p05_m": float(z0.quantile(0.05)) if z0.notna().any() else np.nan,
            "z0_p95_m": float(z0.quantile(0.95)) if z0.notna().any() else np.nan,
            "ustar_mean_mps": float(ust.mean()) if ust.notna().any() else np.nan,
            "ustar_median_mps": float(ust.median()) if ust.notna().any() else np.nan,
            "ustar_p05_mps": float(ust.quantile(0.05)) if ust.notna().any() else np.nan,
            "ustar_p95_mps": float(ust.quantile(0.95)) if ust.notna().any() else np.nan,
            "mean_n_heights_used": float(pd.to_numeric(g["n_heights_used"], errors="coerce").mean()),
            "mean_n_obs_heights_used": float(pd.to_numeric(g["n_obs_heights_used"], errors="coerce").mean()),
            "mean_n_wrf_fallback_heights_used": float(pd.to_numeric(g["n_wrf_fallback_heights_used"], errors="coerce").mean()),
            "mean_profile_rmse_mps": float(pd.to_numeric(ok["profile_rmse_mps"], errors="coerce").mean()) if not ok.empty else np.nan,
            "fit_status_counts": "; ".join(f"{k}:{v}" for k, v in g["fit_status"].astype(str).value_counts().items()),
        }
        if not ok.empty:
            row.update(_param_pair_metrics(ok["wrf_ustar_mps"], ok[ust_col], "ustar_gryning_vs_wrf"))
            row.update(_param_pair_metrics(ok["wrf_z0_m"], ok[z0_col], "z0_gryning_vs_wrf"))
            log_z0_col = "log10_z0_gryning"
            if z0_col == "z0_gryning_m_filtered":
                ok = ok.copy()
                ok["log10_z0_gryning_filtered"] = np.log10(pd.to_numeric(ok[z0_col], errors="coerce").where(pd.to_numeric(ok[z0_col], errors="coerce") > 0))
                log_z0_col = "log10_z0_gryning_filtered"
            row.update(_param_pair_metrics(ok["log10_z0_wrf"], ok[log_z0_col], "log10_z0_gryning_vs_wrf"))
            zr = pd.to_numeric(ok[z0_col], errors="coerce") / pd.to_numeric(ok["wrf_z0_m"], errors="coerce") if "wrf_z0_m" in ok else pd.Series(dtype=float)
            ur = pd.to_numeric(ok[ust_col], errors="coerce") / pd.to_numeric(ok["wrf_ustar_mps"], errors="coerce") if "wrf_ustar_mps" in ok else pd.Series(dtype=float)
            row["z0_ratio_median_gryning_over_wrf"] = float(zr.median()) if zr.notna().any() else np.nan
            row["ustar_ratio_median_gryning_over_wrf"] = float(ur.median()) if ur.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_profile_metric_tables(u_pts: pd.DataFrame, t_pts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    u_rows = []
    if u_pts is not None and not u_pts.empty:
        for keys, g in u_pts.groupby(["relative_file", "dataset_version", "height_m"], dropna=False):
            rel_path, dataset_version, h = keys
            row = {"relative_file": rel_path, "dataset_version": dataset_version, "height_m": h, "var": "U"}
            if "obs_ws_mps" in g.columns:
                row.update(_param_pair_metrics(g["obs_ws_mps"], g["wrf_ws_mps"], "WRF_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_ws_mps"], g.get("U_GRYNING_WRFparam_mps", pd.Series(index=g.index, dtype=float)), "GRYNING_WRFparam_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_ws_mps"], g.get("U_GRYNING_INVzu_mps", g.get("ws_gryning_fit_mps", pd.Series(index=g.index, dtype=float))), "GRYNING_INVzu_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_ws_mps"], g.get("U_LOGMOST_WRFparam_mps", pd.Series(index=g.index, dtype=float)), "LOGMOST_WRFparam_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_ws_mps"], g.get("U_LOGMOST_INVzu_mps", pd.Series(index=g.index, dtype=float)), "LOGMOST_INVzu_vs_OBS"))
            row.update(_param_pair_metrics(g["ws_used_mps"], g.get("U_GRYNING_INVzu_mps", g.get("ws_gryning_fit_mps", pd.Series(index=g.index, dtype=float))), "GRYNING_INVzu_vs_used_profile"))
            u_rows.append(row)
    t_rows = []
    if t_pts is not None and not t_pts.empty:
        for keys, g in t_pts.groupby(["relative_file", "dataset_version", "height_m"], dropna=False):
            rel_path, dataset_version, h = keys
            row = {"relative_file": rel_path, "dataset_version": dataset_version, "height_m": h, "var": "T"}
            if "obs_T_K" in g.columns:
                row.update(_param_pair_metrics(g["obs_T_K"], g["wrf_T_K"], "WRF_T_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_T_K"], g.get("T_MOST_WRFparam_K", g.get("T_MO_WRFsurface_K", pd.Series(index=g.index, dtype=float))), "MOST_T_WRFparam_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_T_K"], g.get("T_MOST_INVzu_WRFhfx_K", g.get("T_MO_GRYNINGsurface_K", pd.Series(index=g.index, dtype=float))), "MOST_T_INVzu_WRFhfx_vs_OBS"))
                row.update(_param_pair_metrics(g["obs_T_K"], g.get("T_MOST_INVheat_K", pd.Series(index=g.index, dtype=float)), "MOST_T_INVheat_vs_OBS"))
            row.update(_param_pair_metrics(g.get("T_used_K", pd.Series(index=g.index, dtype=float)), g.get("T_MOST_INVheat_K", pd.Series(index=g.index, dtype=float)), "MOST_T_INVheat_vs_used_profile"))
            t_rows.append(row)
    return pd.DataFrame(u_rows), pd.DataFrame(t_rows)


def run_gryning_only() -> None:
    """Run Gryning z0/u_star inversion and diagnostics for each tower/file."""
    root = Path(ML_READY_ROOT)
    files = sorted(root.glob(ML_READY_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found under {root} with glob {ML_READY_GLOB}")

    out_root = ensure_dir(Path(OUTPUT_ROOT))
    audit_root = ensure_dir(out_root / "00_audit")
    gryning_ts_frames = []
    gryning_u_point_frames = []
    gryning_t_point_frames = []

    for csv_path in files:
        rel = str(csv_path.relative_to(root))
        print("=" * 100)
        print(f"Gryning inversion: {csv_path}")
        df = read_ml_ready(csv_path)
        if "D10_10min_original" in GRYNING_DATASET_VERSIONS:
            ts, u_pts, t_pts = build_gryning_profile_inversion_for_dataset(
                df=df,
                csv_path=csv_path,
                rel_path=rel,
                dataset_version="D10_10min_original",
                out_dir=audit_root,
            )
            gryning_ts_frames.append(ts)
            gryning_u_point_frames.append(u_pts)
            gryning_t_point_frames.append(t_pts)

        if "D1H_1hour_aggregated" in GRYNING_DATASET_VERSIONS:
            df_h = build_hourly_dataset(df)
            ts, u_pts, t_pts = build_gryning_profile_inversion_for_dataset(
                df=df_h,
                csv_path=csv_path,
                rel_path=rel,
                dataset_version="D1H_1hour_aggregated",
                out_dir=audit_root,
            )
            gryning_ts_frames.append(ts)
            gryning_u_point_frames.append(u_pts)
            gryning_t_point_frames.append(t_pts)

    all_ts = pd.concat([x for x in gryning_ts_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in gryning_ts_frames) else pd.DataFrame()
    if all_ts.empty:
        print("⚠ No Gryning inversion rows were generated. Check OBS_WS/WRF_WS columns in ML-ready files.")
        return

    all_ts_path = audit_root / "13_gryning_z0_ustar_timeseries_all_files.csv"
    all_ts.to_csv(all_ts_path, index=False, encoding="utf-8-sig")

    summary = summarize_gryning_inversion(all_ts)
    summary_path = audit_root / "15_gryning_z0_ustar_summary_and_wrf_comparison_by_file.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if GRYNING_SAVE_METRIC_PLOTS and not summary.empty:
        metric_plot_root = ensure_dir(audit_root / "17_gryning_vs_wrf_parameter_metric_plots")
        for keys, g in summary.groupby(["relative_file", "dataset_version"], dropna=False):
            rel_path, dataset_version = keys
            rel_safe = safe_name(str(rel_path).replace("/", "__").replace(chr(92), "__"))
            stem = f"{safe_name(str(dataset_version))}__{rel_safe}"
            _plot_metric_summary(g, metric_plot_root / f"{stem}_parameter_metrics.png", f"{dataset_version} | {rel_path} | z0/u* comparison metrics")

    all_u_pts = pd.concat([x for x in gryning_u_point_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in gryning_u_point_frames) else pd.DataFrame()
    all_t_pts = pd.concat([x for x in gryning_t_point_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in gryning_t_point_frames) else pd.DataFrame()

    if GRYNING_SAVE_PROFILE_POINTS and not all_u_pts.empty:
        all_u_pts.to_csv(audit_root / "14_gryning_U_profile_points_all_files.csv", index=False, encoding="utf-8-sig")
    if GRYNING_RUN_T_PROFILE_DIAGNOSTICS and not all_t_pts.empty:
        all_t_pts.to_csv(audit_root / "19_gryning_T_profile_points_all_files.csv", index=False, encoding="utf-8-sig")

    u_metrics, t_metrics = build_profile_metric_tables(all_u_pts, all_t_pts)
    if not u_metrics.empty:
        u_metrics.to_csv(audit_root / "18_gryning_U_profile_metrics_by_height.csv", index=False, encoding="utf-8-sig")
        if GRYNING_SAVE_METRIC_PLOTS:
            u_metric_plot_root = ensure_dir(audit_root / "18_gryning_U_profile_metric_plots")
            u_methods = [
                ("WRF_vs_OBS", "WRF"),
                ("GRYNING_WRFparam_vs_OBS", "Gryning all-WRF"),
                ("GRYNING_INVzu_vs_OBS", "Gryning inv z0/u*"),
                ("LOGMOST_WRFparam_vs_OBS", "log-MOST all-WRF"),
                ("LOGMOST_INVzu_vs_OBS", "log-MOST inv z0/u*"),
            ]
            for keys, g in u_metrics.groupby(["relative_file", "dataset_version"], dropna=False):
                rel_path, dataset_version = keys
                rel_safe = safe_name(str(rel_path).replace("/", "__").replace(chr(92), "__"))
                stem = f"{safe_name(str(dataset_version))}__{rel_safe}"
                for metric in ["RMSE", "MAE", "bias"]:
                    _plot_profile_metric_by_height(g, u_metric_plot_root / f"{stem}_U_{metric}_by_height.png", f"{dataset_version} | {rel_path} | U {metric} by height", u_methods, metric_suffix=metric)
    if not t_metrics.empty:
        t_metrics.to_csv(audit_root / "21_gryning_T_profile_metrics_by_height.csv", index=False, encoding="utf-8-sig")
        if GRYNING_SAVE_METRIC_PLOTS:
            t_metric_plot_root = ensure_dir(audit_root / "21_gryning_T_profile_metric_plots")
            t_methods = [
                ("WRF_T_vs_OBS", "WRF T"),
                ("MOST_T_WRFparam_vs_OBS", "MOST-T WRF params"),
                ("MOST_T_INVzu_WRFhfx_vs_OBS", "MOST-T inv z0/u*, WRF HFX"),
                ("MOST_T_INVheat_vs_OBS", "MOST-T inv z0h/theta*"),
            ]
            for keys, g in t_metrics.groupby(["relative_file", "dataset_version"], dropna=False):
                rel_path, dataset_version = keys
                rel_safe = safe_name(str(rel_path).replace("/", "__").replace(chr(92), "__"))
                stem = f"{safe_name(str(dataset_version))}__{rel_safe}"
                for metric in ["RMSE", "MAE", "bias"]:
                    _plot_profile_metric_by_height(g, t_metric_plot_root / f"{stem}_T_{metric}_by_height.png", f"{dataset_version} | {rel_path} | T {metric} by height", t_methods, metric_suffix=metric)

    config = {
        "ML_READY_ROOT": str(root),
        "ML_READY_GLOB": ML_READY_GLOB,
        "OUTPUT_ROOT": str(out_root),
        "gryning_inversion": {
            "enabled": RUN_GRYNING_PROFILE_INVERSION,
            "dataset_versions": list(GRYNING_DATASET_VERSIONS),
            "wind_profile_formula": "Gryning et al. (2007) Eq.(8)/(15)/(21): neutral/stable/unstable wind profile",
            "L_MBL_parameterization": "Gryning et al. (2007) Eq.(31): u*/(f L_MBL)=[-2 ln(u*/(f z0))+55] exp(-((u*/(f L))^2)/400)",
            "source_priority": "OBS_WS at same height/time first; WRF_WS fallback only where OBS is missing/invalid",
            "min_heights_to_fit": GRYNING_MIN_HEIGHTS_TO_FIT,
            "z0_bounds_m": [GRYNING_Z0_MIN, GRYNING_Z0_MAX],
            "ustar_bounds_mps": [GRYNING_USTAR_MIN, GRYNING_USTAR_MAX],
            "pblh_as_zi": GRYNING_USE_PBLH_ZI,
            "pblh_fallback_m": GRYNING_PBLH_FALLBACK_M,
            "neutral_abs_L_threshold_m": GRYNING_NEUTRAL_ABS_L_THRESHOLD_M,
            "default_latitude_deg": GRYNING_LATITUDE_DEG_DEFAULT,
            "temperature_diagnostic": "MOST scalar profile. Outputs include MOST-T with all WRF parameters, MOST-T with inverted wind z0/u* but WRF HFX, and MOST-T with inverted z0h/theta_star; HFX_inv=-rho*cp*u_star*theta_star.",
            "temperature_fit_parameters": {
                "enabled": GRYNING_TEMP_FIT_INVHEAT,
                "min_T_heights_to_fit": GRYNING_TEMP_MIN_HEIGHTS_TO_FIT,
                "z0h_bounds_m": [GRYNING_Z0H_MIN, GRYNING_Z0H_MAX],
                "theta_star_bounds_K": [GRYNING_THETA_STAR_MIN_K, GRYNING_THETA_STAR_MAX_K],
                "source_priority": "OBS_T at same height/time first; WRF_T fallback only where OBS is missing/invalid",
            },
            "outputs": [
                "13_gryning_z0_ustar_timeseries_all_files.csv",
                "13_gryning_z0_ustar_timeseries/",
                "14_gryning_U_profile_points_all_files.csv",
                "14_gryning_profile_points/",
                "15_gryning_z0_ustar_summary_and_wrf_comparison_by_file.csv",
                "16_gryning_vs_wrf_parameter_timeseries_plots/",
                "17_gryning_vs_wrf_parameter_metric_plots/",
                "18_gryning_U_profile_metrics_by_height.csv",
                "18_gryning_U_profile_plots/",
                "18_gryning_U_profile_metric_plots/",
                "19_gryning_T_profile_points_all_files.csv",
                "19_gryning_T_profile_points/",
                "20_gryning_T_profile_plots/",
                "21_gryning_T_profile_metrics_by_height.csv",
                "21_gryning_T_profile_metric_plots/",
            ],
        },
    }
    with open(audit_root / "00_gryning_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("Gryning z0/u_star inversion and WRF comparison finished.")
    print(f"All-time output : {all_ts_path}")
    print(f"Summary output  : {summary_path}")
    if not u_metrics.empty:
        print(f"U metrics output: {audit_root / '18_gryning_U_profile_metrics_by_height.csv'}")
    if not t_metrics.empty:
        print(f"T metrics output: {audit_root / '21_gryning_T_profile_metrics_by_height.csv'}")


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
    hourly = add_most_temperature_profile_columns(hourly)

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

        wrf_t = _first_existing(hourly.columns, [f'WRF_T{h_label}_', f'WRF_T{h_label} ', f'WRF_T{h_label}', f'WRF_TEMP{h_label}_', f'WRF_TEMP{h_label} ', f'WRF_TEMP{h_label}'])
        t_mo = find_col(hourly.columns, f'T_MO{h_label}')
        if t_mo is not None and wrf_t is not None:
            hourly[f'MO_T_minus_WRF{h_label} (K)'] = pd.to_numeric(hourly[t_mo], errors='coerce') - pd.to_numeric(hourly[wrf_t], errors='coerce')

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


def main() -> None:
    if RUN_GRYNING_PROFILE_INVERSION:
        run_gryning_only()
        return

    root = Path(ML_READY_ROOT)
    files = sorted(root.glob(ML_READY_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found under {root} with glob {ML_READY_GLOB}")

    out_root = ensure_dir(Path(OUTPUT_ROOT))
    audit_root = ensure_dir(out_root / "00_audit")
    d10_root = ensure_dir(out_root / "D10_10min")
    d1h_root = ensure_dir(out_root / "D1H_1hour")

    axis_rows = []
    var_audit_frames = []
    build_rows = []
    most_diag_frames = []
    most_temp_diag_frames = []
    most_related_plot_frames = []

    for csv_path in files:
        rel = str(csv_path.relative_to(root))
        print("=" * 100)
        print(f"Processing: {csv_path}")
        df = read_ml_ready(csv_path)
        df_diag = add_most_temperature_profile_columns(df)

        if RUN_STAGE1_AUDIT:
            axis_rows.append(audit_time_axis(csv_path, df, rel))
            var_audit_frames.append(audit_variable_resolution(csv_path, df, rel))
            if RUN_MOST_WS_DIAGNOSTICS:
                most_diag_frames.append(build_most_ws_diagnostics_for_dataset(
                    df=df_diag,
                    csv_path=csv_path,
                    rel_path=rel,
                    dataset_version="D10_10min_original",
                    out_dir=audit_root,
                ))
                if RUN_MOST_TEMP_DIAGNOSTICS:
                    most_temp_diag_frames.append(build_most_temp_diagnostics_for_dataset(
                        df=df_diag,
                        csv_path=csv_path,
                        rel_path=rel,
                        dataset_version="D10_10min_original",
                        out_dir=audit_root,
                    ))
                if RUN_MOST_RELATED_VARIABLE_PLOTS:
                    most_related_plot_frames.append(build_most_related_variable_plots_for_dataset(
                        df=df_diag,
                        csv_path=csv_path,
                        rel_path=rel,
                        dataset_version='D10_10min_original',
                        out_dir=audit_root,
                    ))

        if RUN_STAGE2_BUILD_DATASETS:
            d10_path = d10_root / rel
            d1h_path = d1h_root / rel
            ensure_dir(d10_path.parent)
            ensure_dir(d1h_path.parent)
            if D10_WRITE_AUGMENTED_DIAGNOSTIC_COLUMNS:
                df_diag.to_csv(d10_path, index=False, encoding='utf-8-sig')
            else:
                shutil.copy2(csv_path, d10_path)

            df_h = build_hourly_dataset(df)
            df_h.to_csv(d1h_path, index=False, encoding="utf-8-sig")
            if RUN_MOST_WS_DIAGNOSTICS:
                most_diag_frames.append(build_most_ws_diagnostics_for_dataset(
                    df=df_h,
                    csv_path=d1h_path,
                    rel_path=rel,
                    dataset_version="D1H_1hour_aggregated",
                    out_dir=audit_root,
                ))
            if RUN_MOST_TEMP_DIAGNOSTICS:
                most_temp_diag_frames.append(build_most_temp_diagnostics_for_dataset(
                    df=df_h,
                    csv_path=d1h_path,
                    rel_path=rel,
                    dataset_version="D1H_1hour_aggregated",
                    out_dir=audit_root,
                ))
            if RUN_MOST_RELATED_VARIABLE_PLOTS:
                most_related_plot_frames.append(build_most_related_variable_plots_for_dataset(
                    df=df_h,
                    csv_path=d1h_path,
                    rel_path=rel,
                    dataset_version='D1H_1hour_aggregated',
                    out_dir=audit_root,
                ))
            build_rows.append({
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

    if RUN_MOST_WS_DIAGNOSTICS and most_diag_frames:
        most_df = pd.concat([x for x in most_diag_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in most_diag_frames) else pd.DataFrame()
        if not most_df.empty:
            most_df.to_csv(audit_root / "05_most_ws_obs_wrf_error_metrics_by_file.csv", index=False, encoding="utf-8-sig")
            metric_cols = [
                "WRF_vs_OBS_RMSE", "MOST_vs_OBS_RMSE",
                "WRF_vs_OBS_MAE", "MOST_vs_OBS_MAE",
                "MOST_skill_RMSE_pct_vs_WRF", "MOST_skill_MAE_pct_vs_WRF",
                "MOST_better_abs_error_ratio",
                "residual_corr_MOminusWRF_vs_OBSminusWRF",
                "residual_sign_agreement_ratio",
                "consistency_gate_ratio_sign_or_absdiff",
            ]
            summary = (
                most_df.groupby(["dataset_version"], dropna=False)[metric_cols]
                .mean(numeric_only=True)
                .reset_index()
            )
            summary.to_csv(audit_root / "06_most_ws_obs_wrf_error_metrics_summary.csv", index=False, encoding="utf-8-sig")
            _save_metric_plot_grouped(most_df, audit_root / '07_most_ws_obs_wrf_metrics_plots', 'ws', 'ws')

    if RUN_MOST_TEMP_DIAGNOSTICS and most_temp_diag_frames:
        temp_df = pd.concat([x for x in most_temp_diag_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in most_temp_diag_frames) else pd.DataFrame()
        if not temp_df.empty:
            temp_df.to_csv(audit_root / '08_most_temp_obs_wrf_error_metrics_by_file.csv', index=False, encoding='utf-8-sig')
            metric_cols = [
                'WRF_vs_OBS_RMSE', 'MOST_vs_OBS_RMSE',
                'WRF_vs_OBS_MAE', 'MOST_vs_OBS_MAE',
                'MOST_skill_RMSE_pct_vs_WRF', 'MOST_skill_MAE_pct_vs_WRF',
                'MOST_better_abs_error_ratio',
                'residual_corr_MOminusWRF_vs_OBSminusWRF',
                'residual_sign_agreement_ratio',
                'consistency_gate_ratio_sign_or_absdiff',
            ]
            temp_summary = (
                temp_df.groupby(['dataset_version'], dropna=False)[metric_cols]
                .mean(numeric_only=True)
                .reset_index()
            )
            temp_summary.to_csv(audit_root / '09_most_temp_obs_wrf_error_metrics_summary.csv', index=False, encoding='utf-8-sig')
            _save_metric_plot_grouped(temp_df, audit_root / '10_most_temp_obs_wrf_metrics_plots', 'temp', 'temp')

    if RUN_MOST_RELATED_VARIABLE_PLOTS and most_related_plot_frames:
        related_df = pd.concat([x for x in most_related_plot_frames if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in most_related_plot_frames) else pd.DataFrame()
        if not related_df.empty:
            related_df.to_csv(audit_root / '11_most_related_variable_mean_summary_by_file.csv', index=False, encoding='utf-8-sig')
            related_summary = (
                related_df.groupby(['dataset_version', 'group_kind', 'group_name'], dropna=False)[['n_valid_clean', 'mean_clean', 'median_clean', 'std_clean']]
                .mean(numeric_only=True)
                .reset_index()
            )
            related_summary.to_csv(audit_root / '12_most_related_variable_mean_summary_overall.csv', index=False, encoding='utf-8-sig')

    if RUN_STAGE2_BUILD_DATASETS:
        build_df = pd.DataFrame(build_rows)
        build_df.to_csv(audit_root / "04_dataset_build_manifest.csv", index=False, encoding="utf-8-sig")

    config = {
        "ML_READY_ROOT": str(root),
        "ML_READY_GLOB": ML_READY_GLOB,
        "OUTPUT_ROOT": str(out_root),
        "D10_ROOT": str(d10_root),
        "D1H_ROOT": str(d1h_root),
        "recommended_downstream_window_size": {
            "D10_10min": 144,
            "D1H_1hour": 24,
        },
        "most_ws_diagnostics": {
            "enabled": RUN_MOST_WS_DIAGNOSTICS,
            "min_samples": MOST_DIAG_MIN_SAMPLES,
            "absdiff_tolerance_mps": MOST_DIAG_ABSDIFF_TOL,
            "min_abs_obs_residual_mps": MOST_DIAG_MIN_ABS_OBS_RESID,
            "outputs": [
                "05_most_ws_obs_wrf_error_metrics_by_file.csv",
                "06_most_ws_obs_wrf_error_metrics_summary.csv",
                "05_most_ws_obs_wrf_timeseries/",
                "05_most_ws_obs_wrf_plots/",
                '07_most_ws_obs_wrf_metrics_plots/',
            ],
        },
        'most_temp_diagnostics': {
            'enabled': RUN_MOST_TEMP_DIAGNOSTICS,
            'min_samples': MOST_TEMP_DIAG_MIN_SAMPLES,
            'reference_height_m': MOST_TEMP_REF_HEIGHT_M,
            'outputs': [
                '08_most_temp_obs_wrf_error_metrics_by_file.csv',
                '09_most_temp_obs_wrf_error_metrics_summary.csv',
                '08_most_temp_obs_wrf_timeseries/',
                '08_most_temp_obs_wrf_plots/',
                '10_most_temp_obs_wrf_metrics_plots/',
            ],
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
    print("Stage 1/2 finished.")
    print(f"Audit outputs: {audit_root}")
    print(f"D10 dataset : {d10_root}")
    print(f"D1H dataset : {d1h_root}")
    if RUN_MOST_WS_DIAGNOSTICS:
        print("MOST diagnostics: 05_most_ws_obs_wrf_error_metrics_by_file.csv and per-height time-series/plots under 00_audit")
    if RUN_MOST_TEMP_DIAGNOSTICS:
        print("MOST temperature diagnostics: 08_most_temp_obs_wrf_error_metrics_by_file.csv and per-height time-series/plots under 00_audit")
    if RUN_MOST_RELATED_VARIABLE_PLOTS:
        print("MOST-related variable plots: 11_most_related_variable_timeseries/, 12_most_related_variable_mean_plots/, and mean-summary CSVs under 00_audit")
    print("Recommended: D10 WINDOW_SIZE=144; D1H WINDOW_SIZE=24 for equal 24-hour history.")


if __name__ == "__main__":
    main()
