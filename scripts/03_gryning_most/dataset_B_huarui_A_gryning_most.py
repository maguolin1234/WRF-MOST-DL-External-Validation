#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gryning / MOST full profile diagnostics for huarui_A.

This standalone diagnostic script reads ML_ready_ref10_*.csv files and produces:
  1) U-profile diagnostics with multiple z0/u* cases;
  2) T-profile diagnostics with multiple z0h/theta* cases;
  3) D10 / D1H / D6H temporal-resolution sensitivity;
  4) long-format time-series tables for all towers, heights, methods;
  5) by-height and time-series metrics and plots.

Important interpretation:
  - Free inversion cases use tower profiles at the same time step and are diagnostic upper bounds.
  - Station/sector/stability fixed z0/z0h cases reduce physically questionable 10-min jumping.
  - Fixed vegetation z0 uses the site-wise 500-m vegetation roughness values configured below.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# =============================================================================
# Case configuration
# =============================================================================

CASE_NAME = "huarui_A"
TIME_COL = "北京时间"
ML_READY_ROOT = Path(r"D:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\huarui_A_two_towers_full")
ML_READY_GLOB = r"**/ML_ready_ref10_*.csv"
OUTPUT_ROOT = Path(r"D:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\gryning_profile_full_diagnostics_huarui_A")

# D10 + D1H + D6H are all generated. D1H is recommended as the main-paper version.
DATASET_VERSIONS = ("D10_10min_original", "D1H_1hour_aggregated", "D6H_6hour_aggregated")
AGGREGATION_RULES = {
    "D1H_1hour_aggregated": {"rule": "1h", "min_source_records": 3},
    "D6H_6hour_aggregated": {"rule": "6h", "min_source_records": 18},
}

# Site-wise vegetation-based roughness length at 500 m radius.
# Huarui_A/C values are ESA/ESA-strict vegetation z0 from your table.
# Xilinhaote defaults to DEFAULT_VEGETATION_Z0_M if no site-specific value is provided.
VEGETATION_Z0_500M_BY_SITE = {
    "C039801": 0.072,
    "C039802": 0.032
}
DEFAULT_VEGETATION_Z0_M = 0.03

# Wind-sector and stability grouped fixed-z0 options.
N_WIND_SECTORS = 8                 # 8 sectors = 45 degrees each; change to 12 if needed.
REFERENCE_Z_FOR_STABILITY_M = 10.0 # z/L classification reference height.

# Fitting controls.
KAPPA = 0.40
RD_AIR = 287.05
CP_AIR = 1004.67
GRAV = 9.81
OMEGA = 7.2921159e-5
DEFAULT_LATITUDE_DEG = 34.63

MIN_HEIGHTS_TO_FIT_U = 2
MIN_HEIGHTS_TO_FIT_T = 2
MIN_WIND_SPEED = 0.05
MAX_WIND_SPEED = 75.0
VALID_T_RANGE_K = (180.0, 380.0)

Z0_MIN = 1.0e-4
Z0_MAX = 10.0
Z0_GRID_N = 160
USTAR_MIN = 0.001
USTAR_MAX = 10.0
USTAR_ITER_MAX = 8
USTAR_ITER_TOL = 1.0e-4

Z0H_MIN = 1.0e-5
Z0H_MAX = 2.0
Z0H_GRID_N = 120
THETA_STAR_MIN_K = -20.0
THETA_STAR_MAX_K = 20.0
TEMP_Z0H_OVER_Z0M = 0.10
TEMP_REF_HEIGHT_M = 2.0

# Gryning settings.
USE_PBLH_AS_ZI = True
PBLH_FALLBACK_M = 500.0
USE_LMBL_EQ31 = True
LMBL_FALLBACK_M = 150.0
NEUTRAL_ABS_L_THRESHOLD_M = 1.0e6
STABLE_BETA = 5.0
D_FACTOR = 0.0

# Plot controls.
SAVE_PLOTS = True
PLOT_DPI = 180
PLOT_MAX_POINTS = 2500
PLOT_SELECTED_METHODS_ONLY = True

# Runtime controls.
ROW_STRIDE = 1
MAX_ROWS_PER_FILE = None  # set to a small number for debugging; None = all rows.
PROGRESS_EVERY_ROWS = 5000

# Robust diagnostic filtering. These filters only affect diagnostics/inversions/metrics;
# they do not modify the source ML-ready CSV files.
USE_ROBUST_FILTERS = True
FILTER_INPUTS_BEFORE_INVERSION = True
FILTER_LONG_TABLES_BEFORE_METRICS = True
SPIKE_ROLLING_WINDOW = 9
SPIKE_HAMPEL_NSIGMA = 8.0
WS_METRIC_RANGE = (0.0, 45.0)
TEMP_METRIC_RANGE_K = (230.0, 330.0)
WS_JUMP_THRESHOLD = 8.0          # m/s between adjacent diagnostic samples
TEMP_JUMP_THRESHOLD_K = 10.0    # K between adjacent diagnostic samples
Z0_JUMP_LOG10_THRESHOLD = 1.0   # one order of magnitude between adjacent samples
USTAR_JUMP_THRESHOLD = 1.0      # m/s between adjacent diagnostic samples
Z0H_JUMP_LOG10_THRESHOLD = 1.0
THETA_STAR_JUMP_THRESHOLD_K = 5.0
U_PROFILE_RMSE_MAX_MPS = 8.0
T_PROFILE_RMSE_MAX_K = 10.0
HFX_INV_VALID_RANGE_WM2 = (-1500.0, 2000.0)

# Plot controls for time-series plots. Metrics are still computed for D10/D1H/D6H;
# by default only D1H time-series plots are saved to avoid excessive PNG output.
TIMESERIES_PLOT_DATASET_VERSIONS = ("D1H_1hour_aggregated",)
PLOT_METRIC_BY_HEIGHT_DATASET_VERSIONS = DATASET_VERSIONS


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: Any, max_len: int = 180) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(text))[:max_len]


def as_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_list = list(cols)
    lower_map = {str(c).lower(): c for c in col_list}
    # exact case-insensitive
    for cand in candidates:
        if cand is None:
            continue
        c = str(cand)
        if c in col_list:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    # prefix / contains fallback
    for cand in candidates:
        if cand is None:
            continue
        cand_l = str(cand).lower()
        for c in col_list:
            cl = str(c).lower()
            if cl.startswith(cand_l) or cand_l in cl:
                return c
    return None


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


def extract_site_id(rel_path: str, csv_path: Path) -> str:
    text = f"{rel_path} {csv_path.name}"
    patterns = [
        r"(NO[123]_[0-9]+)",
        r"(NOT_[0-9]+)",
        r"(C0?\d{5,6})",
        r"(C\d{4})",
        r"(t\d{3})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    stem = re.sub(r"ML_ready_ref10_", "", csv_path.stem)
    stem = re.sub(r"_d\d+$", "", stem)
    return stem


def site_z0_vegetation(site_id: str) -> float:
    # exact and case-insensitive lookup
    if site_id in VEGETATION_Z0_500M_BY_SITE:
        return float(VEGETATION_Z0_500M_BY_SITE[site_id])
    for k, v in VEGETATION_Z0_500M_BY_SITE.items():
        if str(k).lower() == str(site_id).lower():
            return float(v)
    return float(DEFAULT_VEGETATION_Z0_M)

# =============================================================================
# Column discovery
# =============================================================================

def normalize_height_label(label: str) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().replace("_", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*m?", s, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    if abs(val - round(val)) < 1e-9:
        return f"{int(round(val))}m"
    return f"{val:g}m"


def height_label_to_float(h_label: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(h_label))
    return float(m.group(1)) if m else np.nan


def collect_height_labels(cols: Iterable[str], kind: str) -> List[str]:
    labels = set()
    # kind: WS, WD, T/TEMP
    if kind == "WS":
        pats = [r"(?:^|_)OBS[_ ]?WS[_ ]?(\d+(?:\.\d+)?)\s*m?", r"(?:^|_)WRF[_ ]?WS[_ ]?(\d+(?:\.\d+)?)\s*m?"]
    elif kind == "WD":
        pats = [r"(?:^|_)OBS[_ ]?WD[_ ]?(\d+(?:\.\d+)?)\s*m?", r"(?:^|_)WRF[_ ]?WD[_ ]?(\d+(?:\.\d+)?)\s*m?"]
    else:
        pats = [
            r"(?:^|_)OBS[_ ]?T(?:EMP)?[_ ]?(\d+(?:\.\d+)?)\s*m?",
            r"(?:^|_)WRF[_ ]?T(?:EMP)?[_ ]?(\d+(?:\.\d+)?)\s*m?",
        ]
    for c in cols:
        cs = str(c)
        for pat in pats:
            m = re.search(pat, cs, flags=re.IGNORECASE)
            if m:
                lab = normalize_height_label(m.group(1) + "m")
                if lab:
                    labels.add(lab)
    return sorted(labels, key=height_label_to_float)


def find_ws_col(cols: Iterable[str], source: str, h_label: str) -> Optional[str]:
    h = h_label.replace("m", "")
    candidates = [
        f"{source}_WS_{h}m", f"{source}_WS{h}m", f"{source}_WS_{h}", f"{source}_WS{h}",
        f"{source}_WS_{h}m_interp", f"{source}_WS{h}m_interp",
    ]
    return first_existing(cols, candidates)


def find_wd_col(cols: Iterable[str], source: str, h_label: str) -> Optional[str]:
    h = h_label.replace("m", "")
    candidates = [
        f"{source}_WD_{h}m", f"{source}_WD{h}m", f"{source}_WD_{h}", f"{source}_WD{h}",
        f"{source}_WD_{h}m_interp", f"{source}_WD{h}m_interp",
    ]
    return first_existing(cols, candidates)


def find_t_col(cols: Iterable[str], source: str, h_label: str) -> Optional[str]:
    h = h_label.replace("m", "")
    candidates = [
        f"{source}_T_{h}m", f"{source}_T{h}m", f"{source}_T_{h}", f"{source}_T{h}",
        f"{source}_TEMP_{h}m", f"{source}_TEMP{h}m", f"{source}_TEMP_{h}", f"{source}_TEMP{h}",
        f"{source}_T_{h}m_interp", f"{source}_T{h}m_interp", f"{source}_TEMP_{h}m_interp", f"{source}_TEMP{h}m_interp",
    ]
    return first_existing(cols, candidates)


def find_scalar_col(cols: Iterable[str], key: str) -> Optional[str]:
    candidates_by_key = {
        "z0": ["Z0 (m)", "Z0", "ZNT", "ZNT (m)", "wrf_z0_m"],
        "ust": ["UST (m/s)", "UST", "USTAR", "USTAR (m/s)", "wrf_ustar_mps"],
        "L": ["WRF_L (m)", "L", "L (m)", "Monin_Obukhov_length", "WRF_MO_LENGTH"],
        "pblh": ["WRF_PBLH (m)", "PBLH", "PBLH (m)", "WRF_PBLH"],
        "hfx": ["HFX (W/m2)", "HFX", "HFX (W m-2)", "HFX_Wm2"],
        "psfc": ["PSFC (Pa)", "PSFC", "PSFC_Pa"],
        "t2": ["T2 (K)", "T2", "T2_K", "WRF_T2_K"],
    }
    return first_existing(cols, candidates_by_key.get(key, []))


def to_kelvin(v: float) -> float:
    v = as_float(v)
    if not np.isfinite(v):
        return np.nan
    # ML-ready may use degC or K. Values < 150 are treated as Celsius.
    return v + 273.15 if v < 150.0 else v


# =============================================================================
# Robust filtering and Stage2-like temporal aggregation
# =============================================================================

def circular_mean_degrees(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    rad = np.deg2rad(vals)
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    if not np.isfinite(s) or not np.isfinite(c):
        return np.nan
    return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)


def _is_probably_wd_column(col: str) -> bool:
    u = str(col).upper()
    return ("WD" in u or "WIND_DIRECTION" in u or "WDIR" in u) and "SIN" not in u and "COS" not in u


def _is_probably_ws_column(col: str) -> bool:
    u = str(col).upper()
    return any(k in u for k in ["WS", "WSPD", "WIND_SPEED", "U_MO", "MOST_WS"]) and not _is_probably_wd_column(col)


def _is_probably_temp_column(col: str) -> bool:
    u = str(col).upper()
    return any(k in u for k in ["TEMP", "_T", "T2", "T_MO", "OBS_T", "WRF_T"]) and not any(k in u for k in ["TKE", "THETA", "UST"])


def _hampel_keep(s: pd.Series, window: int = SPIKE_ROLLING_WINDOW, nsigma: float = SPIKE_HAMPEL_NSIGMA) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.rolling(window=window, center=True, min_periods=max(3, window // 3)).median()
    mad = (x - med).abs().rolling(window=window, center=True, min_periods=max(3, window // 3)).median()
    sigma = 1.4826 * mad
    keep = x.notna()
    spike = (sigma > 0) & ((x - med).abs() > nsigma * sigma)
    keep = keep & ~spike.fillna(False)
    return keep


def _range_jump_hampel_keep(s: pd.Series, kind: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    keep = x.notna()
    if kind == "ws":
        vmin, vmax = WS_METRIC_RANGE
        keep &= (x >= vmin) & (x <= vmax)
        keep &= x.diff().abs().fillna(0.0) <= WS_JUMP_THRESHOLD
    elif kind == "temp":
        vmin, vmax = TEMP_METRIC_RANGE_K
        keep &= (x >= vmin) & (x <= vmax)
        keep &= x.diff().abs().fillna(0.0) <= TEMP_JUMP_THRESHOLD_K
    elif kind == "z0":
        keep &= (x >= Z0_MIN) & (x <= Z0_MAX)
        keep &= np.log10(x.where(x > 0)).diff().abs().fillna(0.0) <= Z0_JUMP_LOG10_THRESHOLD
    elif kind == "z0h":
        keep &= (x >= Z0H_MIN) & (x <= Z0H_MAX)
        keep &= np.log10(x.where(x > 0)).diff().abs().fillna(0.0) <= Z0H_JUMP_LOG10_THRESHOLD
    elif kind == "ustar":
        keep &= (x >= USTAR_MIN) & (x <= USTAR_MAX)
        keep &= x.diff().abs().fillna(0.0) <= USTAR_JUMP_THRESHOLD
    elif kind == "theta_star":
        keep &= (x >= THETA_STAR_MIN_K) & (x <= THETA_STAR_MAX_K)
        keep &= x.diff().abs().fillna(0.0) <= THETA_STAR_JUMP_THRESHOLD_K
    if USE_ROBUST_FILTERS:
        keep &= _hampel_keep(x)
    return keep.fillna(False)


def robust_filter_dataframe_for_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Filter obvious spikes/range errors in diagnostic input columns only."""
    if not (USE_ROBUST_FILTERS and FILTER_INPUTS_BEFORE_INVERSION) or df.empty:
        return df
    out = df.copy()
    if TIME_COL in out.columns:
        out = out.sort_values(TIME_COL).reset_index(drop=True)
    for col in out.columns:
        if col == TIME_COL:
            continue
        kind = None
        if _is_probably_ws_column(col):
            kind = "ws"
        elif _is_probably_temp_column(col):
            kind = "temp"
        elif str(col).upper() in {"Z0", "Z0 (M)", "ZNT", "ZNT (M)"}:
            kind = "z0"
        elif "UST" in str(col).upper():
            kind = "ustar"
        if kind is None:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        sf = s + 273.15 if kind == "temp" and s.dropna().median() < 150.0 else s
        keep = _range_jump_hampel_keep(sf, kind)
        out.loc[~keep, col] = np.nan
    return out


def _parse_wind_component_col(col: str) -> Optional[Tuple[str, str, str]]:
    """Return (source, component, height_label) for broad OBS/WRF U/V naming patterns."""
    c = str(col)
    m = re.search(r"\b(OBS|WRF)[_ ]*([UV])[_ ]*(\d+(?:\.\d+)?)\s*m?", c, flags=re.IGNORECASE)
    if not m:
        return None
    return (m.group(1).upper(), m.group(2).upper(), normalize_height_label(m.group(3) + "m") or f"{m.group(3)}m")


def _find_matching_col_for_height(cols: Iterable[str], source: str, kind: str, h_label: str) -> Optional[str]:
    if kind == "WS":
        return find_ws_col(cols, source, h_label)
    if kind == "WD":
        return find_wd_col(cols, source, h_label)
    return None


def aggregate_dataset(df: pd.DataFrame, rule: str, min_source_records: int) -> pd.DataFrame:
    """Stage2-like aggregation: numeric means, WD circular mean, and WS/WD recomputed from mean U/V when available."""
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)
    if d.empty:
        return pd.DataFrame(columns=df.columns)

    numeric_cols = [c for c in d.columns if c != TIME_COL]
    out_parts = []
    for col in numeric_cols:
        s = pd.to_numeric(d[col], errors="coerce")
        if _is_probably_wd_column(col):
            res = s.resample(rule).apply(circular_mean_degrees)
        else:
            res = s.resample(rule).mean()
        out_parts.append(res.rename(col))
    out = pd.concat(out_parts, axis=1) if out_parts else pd.DataFrame(index=d.resample(rule).size().index)

    comp: Dict[Tuple[str, str], Dict[str, str]] = {}
    for c in numeric_cols:
        parsed = _parse_wind_component_col(c)
        if parsed is None:
            continue
        source, uv, h_label = parsed
        comp.setdefault((source, h_label), {})[uv] = c
    for (source, h_label), pair in comp.items():
        if "U" not in pair or "V" not in pair:
            continue
        u_mean = pd.to_numeric(d[pair["U"]], errors="coerce").resample(rule).mean()
        v_mean = pd.to_numeric(d[pair["V"]], errors="coerce").resample(rule).mean()
        ws = np.sqrt(u_mean ** 2 + v_mean ** 2)
        wd = (270.0 - np.degrees(np.arctan2(v_mean, u_mean))) % 360.0
        ws_col = _find_matching_col_for_height(out.columns, source, "WS", h_label)
        wd_col = _find_matching_col_for_height(out.columns, source, "WD", h_label)
        if ws_col is not None:
            out[ws_col] = ws
        if wd_col is not None:
            out[wd_col] = wd

    counts = d.resample(rule).size()
    out = out.loc[counts[counts >= int(min_source_records)].index]
    out = out.reset_index()

    if "hour_sin" in out.columns or "hour_cos" in out.columns:
        hour = pd.to_datetime(out[TIME_COL]).dt.hour + pd.to_datetime(out[TIME_COL]).dt.minute / 60.0
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    if "month_sin" in out.columns or "month_cos" in out.columns:
        month = pd.to_datetime(out[TIME_COL]).dt.month
        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def filter_long_group_for_metrics(g: pd.DataFrame, variable: str) -> pd.DataFrame:
    if not (USE_ROBUST_FILTERS and FILTER_LONG_TABLES_BEFORE_METRICS) or g.empty:
        return g
    out = g.sort_values("time").copy()
    kind = "ws" if variable == "U" else "temp"
    keep = pd.Series(True, index=out.index)
    keep &= _range_jump_hampel_keep(out["obs"], kind)
    keep &= _range_jump_hampel_keep(out["pred"], kind)
    return out.loc[keep].copy()

# =============================================================================
# MOST / Gryning formulas
# =============================================================================

def effective_L(L: float) -> float:
    L = as_float(L)
    if not np.isfinite(L) or abs(L) >= NEUTRAL_ABS_L_THRESHOLD_M or abs(L) < 1.0e-6:
        return np.nan
    return L


def psi_m_businger(zeta: np.ndarray) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    out = np.zeros_like(zeta, dtype=float)
    unstable = zeta < 0.0
    stable = zeta > 0.0
    if np.any(unstable):
        x = np.power(np.maximum(1.0 - 16.0 * zeta[unstable], 1.0e-12), 0.25)
        out[unstable] = 2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x*x) / 2.0) - 2.0 * np.arctan(x) + np.pi/2.0
    if np.any(stable):
        out[stable] = -5.0 * zeta[stable]
    return out


def psi_h_businger(zeta: np.ndarray) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    out = np.zeros_like(zeta, dtype=float)
    unstable = zeta < 0.0
    stable = zeta > 0.0
    if np.any(unstable):
        x = np.power(np.maximum(1.0 - 16.0 * zeta[unstable], 1.0e-12), 0.25)
        out[unstable] = 2.0 * np.log((1.0 + x*x) / 2.0)
    if np.any(stable):
        out[stable] = -5.0 * zeta[stable]
    return out


def coriolis_parameter(lat_deg: float) -> float:
    lat = np.deg2rad(float(lat_deg) if np.isfinite(lat_deg) else DEFAULT_LATITUDE_DEG)
    return float(2.0 * OMEGA * np.sin(lat))


def gryning_lmbl_eq31(ustar: float, z0: float, L: float, fcor: float) -> float:
    if not (np.isfinite(ustar) and np.isfinite(z0) and np.isfinite(fcor)):
        return np.nan
    ustar = float(ustar); z0 = float(z0); f = abs(float(fcor))
    if ustar <= 0.0 or z0 <= 0.0 or f <= 1.0e-8:
        return np.nan
    rough_arg = ustar / (f * z0)
    if rough_arg <= 1.0:
        return np.nan
    base = -2.0 * math.log(rough_arg) + 55.0
    if not np.isfinite(base) or base <= 0.0:
        return np.nan
    L_eff = effective_L(L)
    exp_term = 1.0
    if np.isfinite(L_eff):
        exp_term = math.exp(-((ustar / (f * L_eff)) ** 2) / 400.0)
    denom = f * base * exp_term
    if denom <= 0.0:
        return np.nan
    lmbl = ustar / denom
    if not np.isfinite(lmbl) or lmbl <= 0.0:
        return np.nan
    return float(lmbl)


def gryning_shape(z: np.ndarray, z0: float, L: float = np.nan, zi: float = np.nan, lmbl: float = np.nan) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z0 = float(z0)
    d = D_FACTOR * z0
    zeff = np.maximum(z - d, z0 * 1.01)
    g = np.log(zeff / z0)
    zi_use = float(zi) if np.isfinite(zi) and zi > 0 else np.nan
    lmbl_use = float(lmbl) if np.isfinite(lmbl) and lmbl > 0 else np.nan
    if USE_PBLH_AS_ZI and np.isfinite(zi_use) and np.isfinite(lmbl_use):
        zclip = np.minimum(zeff, 0.98 * zi_use)
        g = g + zclip / lmbl_use - (zclip ** 2) / (2.0 * lmbl_use * zi_use)
    else:
        zclip = zeff
    L_eff = effective_L(L)
    if np.isfinite(L_eff):
        if L_eff > 0.0:
            zi_for = zi_use if np.isfinite(zi_use) else np.inf
            g = g + STABLE_BETA * zclip / L_eff * (1.0 - zclip / (2.0 * zi_for))
        elif L_eff < 0.0:
            g = g - psi_m_businger(zclip / L_eff)
    return g


def wind_at_height(z: float, z0: float, ustar: float, L: float, zi: float = np.nan, lmbl: float = np.nan) -> float:
    if not (np.isfinite(z) and np.isfinite(z0) and np.isfinite(ustar)):
        return np.nan
    if z <= 0.0 or z0 <= 0.0 or ustar <= 0.0:
        return np.nan
    g = gryning_shape(np.asarray([float(z)]), z0, L=L, zi=zi, lmbl=lmbl)[0]
    val = (ustar / KAPPA) * g
    return float(val) if np.isfinite(val) and 0.0 <= val <= MAX_WIND_SPEED else np.nan


def fit_ustar_linear(z: np.ndarray, u: np.ndarray, z0: float, L: float, zi: float = np.nan, lmbl: float = np.nan) -> float:
    g = gryning_shape(z, z0, L=L, zi=zi, lmbl=lmbl)
    m = np.isfinite(g) & (g > 0.0) & np.isfinite(u)
    if int(m.sum()) < 1:
        return np.nan
    denom = float(np.sum(g[m] ** 2))
    if denom <= 1.0e-14:
        return np.nan
    beta = float(np.sum(g[m] * u[m]) / denom)
    ustar = KAPPA * beta
    if USTAR_MIN <= ustar <= USTAR_MAX:
        return float(ustar)
    return np.nan


def fit_ustar_given_z0(z: np.ndarray, u: np.ndarray, z0: float, pblh: float, L: float, lat_deg: float) -> Dict[str, float]:
    z = np.asarray(z, dtype=float); u = np.asarray(u, dtype=float)
    m = np.isfinite(z) & np.isfinite(u) & (z > 0) & (u >= MIN_WIND_SPEED)
    z = z[m]; u = u[m]
    if len(z) < MIN_HEIGHTS_TO_FIT_U or not np.isfinite(z0) or z0 <= 0:
        return {"z0": np.nan, "ustar": np.nan, "lmbl": np.nan, "rmse": np.nan, "mae": np.nan}
    zi = pblh if np.isfinite(pblh) and pblh > 0 else PBLH_FALLBACK_M
    fcor = coriolis_parameter(lat_deg)
    ustar = fit_ustar_linear(z, u, z0, L=L)
    if not np.isfinite(ustar):
        return {"z0": z0, "ustar": np.nan, "lmbl": np.nan, "rmse": np.nan, "mae": np.nan}
    lmbl = np.nan
    for _ in range(USTAR_ITER_MAX):
        lmbl = gryning_lmbl_eq31(ustar, z0, L, fcor) if USE_LMBL_EQ31 else np.nan
        if not np.isfinite(lmbl):
            lmbl = LMBL_FALLBACK_M
        new_ustar = fit_ustar_linear(z, u, z0, L=L, zi=zi, lmbl=lmbl)
        if not np.isfinite(new_ustar):
            break
        if abs(new_ustar - ustar) <= USTAR_ITER_TOL:
            ustar = new_ustar
            break
        ustar = new_ustar
    pred = np.asarray([wind_at_height(zz, z0, ustar, L, zi=zi, lmbl=lmbl) for zz in z], dtype=float)
    mm = np.isfinite(pred) & np.isfinite(u)
    if int(mm.sum()) == 0:
        return {"z0": z0, "ustar": ustar, "lmbl": lmbl, "rmse": np.nan, "mae": np.nan}
    e = pred[mm] - u[mm]
    return {"z0": float(z0), "ustar": float(ustar), "lmbl": float(lmbl), "rmse": float(np.sqrt(np.mean(e*e))), "mae": float(np.mean(np.abs(e)))}


def fit_gryning_z0_ustar(z: np.ndarray, u: np.ndarray, pblh: float, L: float, lat_deg: float, z0_hint: float = np.nan) -> Dict[str, float]:
    z = np.asarray(z, dtype=float); u = np.asarray(u, dtype=float)
    m = np.isfinite(z) & np.isfinite(u) & (z > 0) & (u >= MIN_WIND_SPEED)
    z = z[m]; u = u[m]
    if len(z) < MIN_HEIGHTS_TO_FIT_U:
        return {"z0": np.nan, "ustar": np.nan, "lmbl": np.nan, "rmse": np.nan, "mae": np.nan}
    z0_grid = np.geomspace(Z0_MIN, Z0_MAX, Z0_GRID_N)
    if np.isfinite(z0_hint) and Z0_MIN <= z0_hint <= Z0_MAX:
        local = np.geomspace(max(Z0_MIN, z0_hint / 5.0), min(Z0_MAX, z0_hint * 5.0), max(20, Z0_GRID_N // 3))
        z0_grid = np.unique(np.sort(np.concatenate([z0_grid, local, np.asarray([z0_hint])])) )
    best = None
    for z0 in z0_grid:
        fit = fit_ustar_given_z0(z, u, float(z0), pblh, L, lat_deg)
        if not np.isfinite(fit.get("rmse", np.nan)):
            continue
        if best is None or fit["rmse"] < best["rmse"]:
            best = fit
    if best is None:
        return {"z0": np.nan, "ustar": np.nan, "lmbl": np.nan, "rmse": np.nan, "mae": np.nan}
    return best


def temp_shape(z: np.ndarray, z0h: float, L: float, z_ref: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z0h = max(float(z0h), 1.0e-8)
    zeff = np.maximum(z, z0h * 1.01)
    zref_eff = max(float(z_ref), z0h * 1.01)
    L_eff = effective_L(L)
    if np.isfinite(L_eff):
        zeta = zeff / L_eff
        zeta_ref = zref_eff / L_eff
    else:
        zeta = np.zeros_like(zeff)
        zeta_ref = 0.0
    return np.log(zeff / zref_eff) - (psi_h_businger(zeta) - psi_h_businger(np.asarray([zeta_ref]))[0])


def temperature_with_hfx(z: float, z0h: float, ustar: float, L: float, hfx: float, psfc: float, t_ref_k: float, z_ref: float = TEMP_REF_HEIGHT_M) -> float:
    if not (np.isfinite(z) and np.isfinite(z0h) and np.isfinite(ustar) and np.isfinite(hfx) and np.isfinite(psfc) and np.isfinite(t_ref_k)):
        return np.nan
    if z <= 0 or z0h <= 0 or ustar <= 0:
        return np.nan
    rho = float(psfc) / (RD_AIR * max(float(t_ref_k), 150.0))
    theta_star = -float(hfx) / (rho * CP_AIR * float(ustar))
    H = temp_shape(np.asarray([float(z)]), z0h, L, z_ref)[0]
    val = float(t_ref_k) + (theta_star / KAPPA) * H
    return float(val) if np.isfinite(val) and VALID_T_RANGE_K[0] <= val <= VALID_T_RANGE_K[1] else np.nan


def fit_theta_star_given_z0h(z: np.ndarray, T: np.ndarray, z0h: float, L: float, t_ref_k: float, z_ref: float = TEMP_REF_HEIGHT_M) -> Dict[str, float]:
    z = np.asarray(z, dtype=float); T = np.asarray(T, dtype=float)
    m = np.isfinite(z) & np.isfinite(T) & (z > 0) & (T >= VALID_T_RANGE_K[0]) & (T <= VALID_T_RANGE_K[1])
    z = z[m]; T = T[m]
    if len(z) < MIN_HEIGHTS_TO_FIT_T or not np.isfinite(z0h) or z0h <= 0 or not np.isfinite(t_ref_k):
        return {"z0h": np.nan, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    H = temp_shape(z, z0h, L, z_ref)
    m2 = np.isfinite(H)
    if int(m2.sum()) < MIN_HEIGHTS_TO_FIT_T:
        return {"z0h": z0h, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    X = H[m2] / KAPPA
    y = T[m2] - float(t_ref_k)
    denom = float(np.sum(X*X))
    if denom <= 1.0e-14:
        return {"z0h": z0h, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    theta_star = float(np.sum(X*y) / denom)
    if not (THETA_STAR_MIN_K <= theta_star <= THETA_STAR_MAX_K):
        return {"z0h": z0h, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    pred = float(t_ref_k) + theta_star * X
    e = pred - y - float(t_ref_k) + 0.0  # overwritten below for clarity
    pred_T = float(t_ref_k) + theta_star * X
    err = pred_T - T[m2]
    return {"z0h": float(z0h), "theta_star": theta_star, "rmse": float(np.sqrt(np.mean(err*err))), "mae": float(np.mean(np.abs(err)))}


def fit_z0h_theta_star(z: np.ndarray, T: np.ndarray, L: float, t_ref_k: float, z0h_hint: float = np.nan, z_ref: float = TEMP_REF_HEIGHT_M) -> Dict[str, float]:
    if not np.isfinite(t_ref_k):
        return {"z0h": np.nan, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    z0h_grid = np.geomspace(Z0H_MIN, Z0H_MAX, Z0H_GRID_N)
    if np.isfinite(z0h_hint) and Z0H_MIN <= z0h_hint <= Z0H_MAX:
        local = np.geomspace(max(Z0H_MIN, z0h_hint / 5.0), min(Z0H_MAX, z0h_hint * 5.0), max(20, Z0H_GRID_N // 3))
        z0h_grid = np.unique(np.sort(np.concatenate([z0h_grid, local, np.asarray([z0h_hint])])) )
    best = None
    for z0h in z0h_grid:
        fit = fit_theta_star_given_z0h(z, T, float(z0h), L, t_ref_k, z_ref=z_ref)
        if not np.isfinite(fit.get("rmse", np.nan)):
            continue
        if best is None or fit["rmse"] < best["rmse"]:
            best = fit
    if best is None:
        return {"z0h": np.nan, "theta_star": np.nan, "rmse": np.nan, "mae": np.nan}
    return best


def temp_from_theta(z: float, z0h: float, theta_star: float, L: float, t_ref_k: float, z_ref: float = TEMP_REF_HEIGHT_M) -> float:
    if not (np.isfinite(z) and np.isfinite(z0h) and np.isfinite(theta_star) and np.isfinite(t_ref_k)):
        return np.nan
    H = temp_shape(np.asarray([float(z)]), z0h, L, z_ref)[0]
    val = float(t_ref_k) + (float(theta_star) / KAPPA) * H
    return float(val) if np.isfinite(val) and VALID_T_RANGE_K[0] <= val <= VALID_T_RANGE_K[1] else np.nan

# =============================================================================
# Metrics and classifications
# =============================================================================

def pair_metrics(obs: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n == 0:
        return {"N": 0, "bias": np.nan, "MAE": np.nan, "RMSE": np.nan, "Pearson_r": np.nan, "R2": np.nan, "mean_OBS": np.nan, "mean_pred": np.nan, "std_OBS": np.nan, "std_pred": np.nan, "P50_abs_error": np.nan, "P75_abs_error": np.nan, "P90_abs_error": np.nan, "P95_abs_error": np.nan, "max_abs_error": np.nan}
    e = b[m] - a[m]
    ae = np.abs(e)
    if n >= 2 and np.nanstd(a[m]) > 0 and np.nanstd(b[m]) > 0:
        r = float(np.corrcoef(a[m], b[m])[0, 1])
    else:
        r = np.nan
    ss_res = float(np.sum((b[m] - a[m]) ** 2))
    ss_tot = float(np.sum((a[m] - np.mean(a[m])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "N": n,
        "bias": float(np.mean(e)),
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "Pearson_r": r,
        "R2": r2,
        "mean_OBS": float(np.mean(a[m])),
        "mean_pred": float(np.mean(b[m])),
        "std_OBS": float(np.std(a[m])),
        "std_pred": float(np.std(b[m])),
        "P50_abs_error": float(np.percentile(ae, 50)),
        "P75_abs_error": float(np.percentile(ae, 75)),
        "P90_abs_error": float(np.percentile(ae, 90)),
        "P95_abs_error": float(np.percentile(ae, 95)),
        "max_abs_error": float(np.max(ae)),
    }


def build_metrics_table(long_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    obs_col = "obs"
    pred_col = "pred"
    group_cols = ["relative_file", "site_id", "dataset_version", "height_m", "method"]
    for keys, g in long_df.groupby(group_cols, dropna=False):
        g_metric = filter_long_group_for_metrics(g, variable)
        row = dict(zip(group_cols, keys))
        row["variable"] = variable
        row.update(pair_metrics(g_metric[obs_col], g_metric[pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_metrics(metrics_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in metrics_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        # Weighted by N where possible.
        w = pd.to_numeric(g["N"], errors="coerce").fillna(0.0)
        for m in ["bias", "MAE", "RMSE", "Pearson_r", "R2"]:
            vals = pd.to_numeric(g[m], errors="coerce")
            mask = vals.notna() & (w > 0)
            if mask.any():
                row[m] = float(np.average(vals[mask], weights=w[mask]))
            else:
                row[m] = np.nan
        row["N_total"] = int(w.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def wind_sector_label(wd_deg: float, n_sector: int = N_WIND_SECTORS) -> str:
    wd = as_float(wd_deg)
    if not np.isfinite(wd):
        return "unknown"
    width = 360.0 / n_sector
    idx = int(np.floor(((wd % 360.0) + width / 2.0) / width)) % n_sector
    center = idx * width
    if n_sector == 8:
        names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return names[idx]
    return f"sector_{idx:02d}_{center:.0f}deg"


def stability_class(L: float, z_ref: float = REFERENCE_Z_FOR_STABILITY_M) -> str:
    L = as_float(L)
    if not np.isfinite(L) or abs(L) < 1.0e-6:
        return "unknown"
    zL = z_ref / L
    if zL < -1.0:
        return "strong_unstable"
    if zL < -0.1:
        return "moderate_unstable"
    if zL < -0.01:
        return "weak_unstable"
    if zL <= 0.01:
        return "near_neutral"
    if zL <= 0.1:
        return "weak_stable"
    if zL <= 1.0:
        return "moderate_stable"
    return "strong_stable"

# =============================================================================
# Profile object construction
# =============================================================================

def scalar_at(row: pd.Series, col: Optional[str]) -> float:
    if col is None or col not in row.index:
        return np.nan
    return as_float(row[col])


def build_profile_objects(df: pd.DataFrame, csv_path: Path, rel_path: str, dataset_version: str) -> List[Dict[str, Any]]:
    cols = list(df.columns)
    site_id = extract_site_id(rel_path, csv_path)

    ws_labels = collect_height_labels(cols, "WS")
    t_labels = collect_height_labels(cols, "T")
    wd_labels = collect_height_labels(cols, "WD")

    ws_col_map = []
    for h in ws_labels:
        obs_col = find_ws_col(cols, "OBS", h)
        wrf_col = find_ws_col(cols, "WRF", h)
        if obs_col is not None or wrf_col is not None:
            ws_col_map.append((h, height_label_to_float(h), obs_col, wrf_col))

    t_col_map = []
    for h in t_labels:
        obs_col = find_t_col(cols, "OBS", h)
        wrf_col = find_t_col(cols, "WRF", h)
        if obs_col is not None or wrf_col is not None:
            t_col_map.append((h, height_label_to_float(h), obs_col, wrf_col))

    # sector direction: prefer OBS WD at lowest available height, then WRF WD.
    wd_col = None
    if wd_labels:
        preferred = sorted(wd_labels, key=lambda x: abs(height_label_to_float(x) - 10.0))
        for h in preferred:
            wd_col = find_wd_col(cols, "OBS", h) or find_wd_col(cols, "WRF", h)
            if wd_col:
                break
    if wd_col is None:
        wd_col = first_existing(cols, ["OBS_WD", "WRF_WD", "WD", "wind_direction"])

    z0_col = find_scalar_col(cols, "z0")
    ust_col = find_scalar_col(cols, "ust")
    L_col = find_scalar_col(cols, "L")
    pblh_col = find_scalar_col(cols, "pblh")
    hfx_col = find_scalar_col(cols, "hfx")
    psfc_col = find_scalar_col(cols, "psfc")
    t2_col = find_scalar_col(cols, "t2")

    objects = []
    nrows = len(df) if MAX_ROWS_PER_FILE is None else min(len(df), int(MAX_ROWS_PER_FILE))
    for i in range(0, nrows, int(ROW_STRIDE)):
        if PROGRESS_EVERY_ROWS and i > 0 and i % PROGRESS_EVERY_ROWS == 0:
            print(f"  {CASE_NAME} | {dataset_version} | {rel_path}: {i}/{len(df)} rows")
        row = df.iloc[i]
        t = row[TIME_COL]
        if pd.isna(t):
            continue
        wrf_z0 = scalar_at(row, z0_col)
        wrf_ust = scalar_at(row, ust_col)
        L = scalar_at(row, L_col)
        pblh = scalar_at(row, pblh_col)
        hfx = scalar_at(row, hfx_col)
        psfc = scalar_at(row, psfc_col)
        t2_k = to_kelvin(scalar_at(row, t2_col))
        wd = scalar_at(row, wd_col)
        sector = wind_sector_label(wd)
        stab = stability_class(L)
        lat = DEFAULT_LATITUDE_DEG

        # Wind profile arrays.
        z_ws, obs_ws, wrf_ws, used_ws, hlabels_ws, src_ws = [], [], [], [], [], []
        for hlabel, z, obs_col, wrf_col in ws_col_map:
            ov = scalar_at(row, obs_col)
            wv = scalar_at(row, wrf_col)
            use = np.nan; src = "missing"
            if np.isfinite(ov) and MIN_WIND_SPEED <= ov <= MAX_WIND_SPEED:
                use = ov; src = "OBS"
            elif np.isfinite(wv) and MIN_WIND_SPEED <= wv <= MAX_WIND_SPEED:
                use = wv; src = "WRF_fallback"
            z_ws.append(z); obs_ws.append(ov); wrf_ws.append(wv); used_ws.append(use); hlabels_ws.append(hlabel); src_ws.append(src)
        z_ws = np.asarray(z_ws, dtype=float); obs_ws = np.asarray(obs_ws, dtype=float); wrf_ws = np.asarray(wrf_ws, dtype=float); used_ws = np.asarray(used_ws, dtype=float)

        free_fit = fit_gryning_z0_ustar(z_ws, used_ws, pblh, L, lat, z0_hint=wrf_z0)

        # Temperature profile arrays.
        z_t, obs_T, wrf_T, used_T, hlabels_t, src_T = [], [], [], [], [], []
        for hlabel, z, obs_col, wrf_col in t_col_map:
            ov = to_kelvin(scalar_at(row, obs_col))
            wv = to_kelvin(scalar_at(row, wrf_col))
            use = np.nan; src = "missing"
            if np.isfinite(ov) and VALID_T_RANGE_K[0] <= ov <= VALID_T_RANGE_K[1]:
                use = ov; src = "OBS"
            elif np.isfinite(wv) and VALID_T_RANGE_K[0] <= wv <= VALID_T_RANGE_K[1]:
                use = wv; src = "WRF_fallback"
            z_t.append(z); obs_T.append(ov); wrf_T.append(wv); used_T.append(use); hlabels_t.append(hlabel); src_T.append(src)
        z_t = np.asarray(z_t, dtype=float); obs_T = np.asarray(obs_T, dtype=float); wrf_T = np.asarray(wrf_T, dtype=float); used_T = np.asarray(used_T, dtype=float)

        # Temperature references are separated for fairness.
        # HFX/WRF-parameter methods may only use WRF-derived reference temperature: T2 first,
        # then the lowest valid WRF_T profile value. They must not use OBS_T as fallback.
        t_ref_hfx_k = t2_k
        z_ref_hfx = TEMP_REF_HEIGHT_M
        t_ref_hfx_source = "T2" if np.isfinite(t_ref_hfx_k) else "missing"
        if not np.isfinite(t_ref_hfx_k):
            mw = np.isfinite(wrf_T)
            if mw.any():
                valid_indices = np.where(mw)[0]
                ii = valid_indices[int(np.nanargmin(z_t[mw]))]
                t_ref_hfx_k = float(wrf_T[ii])
                z_ref_hfx = float(z_t[ii])
                t_ref_hfx_source = "lowest_WRF_T_profile"

        # Free thermal inversion is a diagnostic upper bound and may use the tower-constrained
        # profile reference if WRF reference is unavailable.
        t_ref_fit_k = t_ref_hfx_k
        z_ref_fit = z_ref_hfx
        t_ref_fit_source = t_ref_hfx_source
        if not np.isfinite(t_ref_fit_k):
            mt = np.isfinite(used_T)
            if mt.any():
                valid_indices = np.where(mt)[0]
                ii = valid_indices[int(np.nanargmin(z_t[mt]))]
                t_ref_fit_k = float(used_T[ii])
                z_ref_fit = float(z_t[ii])
                t_ref_fit_source = "lowest_used_T_profile_diagnostic"

        z0h_hint = TEMP_Z0H_OVER_Z0M * free_fit.get("z0", np.nan) if np.isfinite(free_fit.get("z0", np.nan)) else np.nan
        free_t_fit = fit_z0h_theta_star(z_t, used_T, L, t_ref_fit_k, z0h_hint=z0h_hint, z_ref=z_ref_fit)

        objects.append({
            "relative_file": rel_path,
            "csv_file": str(csv_path),
            "site_id": site_id,
            "dataset_version": dataset_version,
            "time": t,
            "sector": sector,
            "wind_direction_deg": wd,
            "stability_class": stab,
            "L": L,
            "pblh": pblh,
            "wrf_z0": wrf_z0,
            "wrf_ust": wrf_ust,
            "hfx": hfx,
            "psfc": psfc,
            "t2_k": t2_k,
            "t_ref_hfx_k": t_ref_hfx_k,
            "z_ref_hfx": z_ref_hfx,
            "t_ref_hfx_source": t_ref_hfx_source,
            "t_ref_fit_k": t_ref_fit_k,
            "z_ref_fit": z_ref_fit,
            "t_ref_fit_source": t_ref_fit_source,
            "vegetation_z0": site_z0_vegetation(site_id),
            "u_free": free_fit,
            "t_free": free_t_fit,
            "z_ws": z_ws,
            "obs_ws": obs_ws,
            "wrf_ws": wrf_ws,
            "used_ws": used_ws,
            "hlabels_ws": hlabels_ws,
            "src_ws": src_ws,
            "z_t": z_t,
            "obs_T": obs_T,
            "wrf_T": wrf_T,
            "used_T": used_T,
            "hlabels_t": hlabels_t,
            "src_T": src_T,
        })
    return objects

# =============================================================================
# Fixed parameter summaries and long rows
# =============================================================================

def robust_median(values: Iterable[float]) -> float:
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return np.nan
    return float(v.median())


def build_fixed_maps(objects: List[Dict[str, Any]], param: str) -> Dict[str, Dict[Any, float]]:
    # param: "z0" or "z0h"
    val_get = (lambda o: o["u_free"].get("z0", np.nan)) if param == "z0" else (lambda o: o["t_free"].get("z0h", np.nan))
    # Prefer D1H as fixed-parameter reference; fallback to all versions when D1H absent.
    ref_objs = [o for o in objects if o["dataset_version"] == "D1H_1hour_aggregated"]
    if not ref_objs:
        ref_objs = objects
    maps = {"station": {}, "sector": {}, "stability": {}}
    df_rows = []
    for o in ref_objs:
        v = val_get(o)
        rmse = o["u_free"].get("rmse", np.nan) if param == "z0" else o["t_free"].get("rmse", np.nan)
        good_rmse = (np.isfinite(rmse) and ((param == "z0" and rmse <= U_PROFILE_RMSE_MAX_MPS) or (param == "z0h" and rmse <= T_PROFILE_RMSE_MAX_K)))
        good_range = ((param == "z0" and Z0_MIN <= v <= Z0_MAX) or (param == "z0h" and Z0H_MIN <= v <= Z0H_MAX))
        if np.isfinite(v) and v > 0 and good_rmse and good_range:
            df_rows.append({
                "site_id": o["site_id"],
                "sector": o["sector"],
                "stability_class": o["stability_class"],
                "value": float(v),
            })
    d = pd.DataFrame(df_rows)
    if d.empty:
        return maps
    for site, g in d.groupby("site_id"):
        maps["station"][site] = robust_median(g["value"])
    for (site, sector), g in d.groupby(["site_id", "sector"]):
        maps["sector"][(site, sector)] = robust_median(g["value"])
    for (site, stab), g in d.groupby(["site_id", "stability_class"]):
        maps["stability"][(site, stab)] = robust_median(g["value"])
    return maps


def fixed_lookup(maps: Dict[str, Dict[Any, float]], mode: str, obj: Dict[str, Any], fallback: float) -> float:
    if mode == "station":
        v = maps.get("station", {}).get(obj["site_id"], np.nan)
    elif mode == "sector":
        v = maps.get("sector", {}).get((obj["site_id"], obj["sector"]), np.nan)
    elif mode == "stability":
        v = maps.get("stability", {}).get((obj["site_id"], obj["stability_class"]), np.nan)
    else:
        v = np.nan
    if np.isfinite(v) and v > 0:
        return float(v)
    return float(fallback) if np.isfinite(fallback) and fallback > 0 else np.nan


def build_parameter_summary(objects: List[Dict[str, Any]], z0_maps: Dict[str, Dict[Any, float]], z0h_maps: Dict[str, Dict[Any, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_z0, rows_z0h = [], []
    for o in objects:
        base = {
            "relative_file": o["relative_file"], "site_id": o["site_id"], "dataset_version": o["dataset_version"],
            "time": o["time"], "sector": o["sector"], "stability_class": o["stability_class"],
        }
        rows_z0.append({**base, "param": "z0", "free": o["u_free"].get("z0", np.nan), "wrf": o["wrf_z0"], "vegetation_500m": o["vegetation_z0"],
                        "station_fixed": fixed_lookup(z0_maps, "station", o, o["vegetation_z0"]),
                        "sector_fixed": fixed_lookup(z0_maps, "sector", o, fixed_lookup(z0_maps, "station", o, o["vegetation_z0"])),
                        "stability_fixed": fixed_lookup(z0_maps, "stability", o, fixed_lookup(z0_maps, "station", o, o["vegetation_z0"]))})
        rows_z0h.append({**base, "param": "z0h", "free": o["t_free"].get("z0h", np.nan),
                         "station_fixed": fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan)),
                         "sector_fixed": fixed_lookup(z0h_maps, "sector", o, fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan))),
                         "stability_fixed": fixed_lookup(z0h_maps, "stability", o, fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan)))})
    return pd.DataFrame(rows_z0), pd.DataFrame(rows_z0h)


def build_long_tables(objects: List[Dict[str, Any]], z0_maps: Dict[str, Dict[Any, float]], z0h_maps: Dict[str, Dict[Any, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    u_rows, t_rows = [], []
    for o in objects:
        base = {
            "relative_file": o["relative_file"],
            "site_id": o["site_id"],
            "dataset_version": o["dataset_version"],
            "time": o["time"],
            "sector": o["sector"],
            "wind_direction_deg": o["wind_direction_deg"],
            "stability_class": o["stability_class"],
            "L": o["L"],
            "pblh": o["pblh"],
            "t_ref_hfx_source": o.get("t_ref_hfx_source", ""),
            "t_ref_fit_source": o.get("t_ref_fit_source", ""),
        }
        zi = o["pblh"] if np.isfinite(o["pblh"]) and o["pblh"] > 0 else PBLH_FALLBACK_M
        L = o["L"]
        fcor = coriolis_parameter(DEFAULT_LATITUDE_DEG)

        # U methods that do not need new fitting.
        u_method_params = []
        if np.isfinite(o["wrf_z0"]) and np.isfinite(o["wrf_ust"]):
            lmbl_wrf = gryning_lmbl_eq31(o["wrf_ust"], o["wrf_z0"], L, fcor) if USE_LMBL_EQ31 else np.nan
            if not np.isfinite(lmbl_wrf): lmbl_wrf = LMBL_FALLBACK_M
            u_method_params += [
                ("Gryning_all_WRF", o["wrf_z0"], o["wrf_ust"], zi, lmbl_wrf),
                ("logMOST_all_WRF", o["wrf_z0"], o["wrf_ust"], np.nan, np.nan),
            ]
        if np.isfinite(o["u_free"].get("z0", np.nan)) and np.isfinite(o["u_free"].get("ustar", np.nan)):
            u_method_params += [
                ("Gryning_free_inv_z0_ustar", o["u_free"]["z0"], o["u_free"]["ustar"], zi, o["u_free"].get("lmbl", np.nan)),
                ("logMOST_free_inv_z0_ustar", o["u_free"]["z0"], o["u_free"]["ustar"], np.nan, np.nan),
            ]
        # Fixed-z0 cases with per-time fitted ustar.
        fixed_z0_cases = [
            ("Gryning_station_fixed_z0", fixed_lookup(z0_maps, "station", o, o["vegetation_z0"]), "station"),
            ("Gryning_fixed_z0_vegetation500", o["vegetation_z0"], "vegetation_500m"),
            ("Gryning_sector_fixed_z0", fixed_lookup(z0_maps, "sector", o, fixed_lookup(z0_maps, "station", o, o["vegetation_z0"])), "sector"),
            ("Gryning_stability_fixed_z0", fixed_lookup(z0_maps, "stability", o, fixed_lookup(z0_maps, "station", o, o["vegetation_z0"])), "stability"),
        ]
        fixed_fit_cache = {}
        for method, z0_fixed, z0_case in fixed_z0_cases:
            fit = fit_ustar_given_z0(o["z_ws"], o["used_ws"], z0_fixed, o["pblh"], L, DEFAULT_LATITUDE_DEG)
            fixed_fit_cache[method] = fit
            if np.isfinite(fit.get("ustar", np.nan)):
                u_method_params.append((method, z0_fixed, fit["ustar"], zi, fit.get("lmbl", np.nan)))

        for j, z in enumerate(o["z_ws"]):
            obs = o["obs_ws"][j]
            # WRF baseline.
            pred = o["wrf_ws"][j]
            if np.isfinite(obs) or np.isfinite(pred):
                u_rows.append({**base, "height_label": o["hlabels_ws"][j], "height_m": z, "method": "WRF", "obs": obs, "pred": pred, "error": pred - obs if np.isfinite(obs) and np.isfinite(pred) else np.nan, "z0_used_m": o["wrf_z0"], "ustar_used_mps": o["wrf_ust"], "z0_case": "WRF", "source": o["src_ws"][j]})
            for method, z0, ust, zi_use, lmbl in u_method_params:
                pred = wind_at_height(z, z0, ust, L, zi=zi_use, lmbl=lmbl)
                u_rows.append({**base, "height_label": o["hlabels_ws"][j], "height_m": z, "method": method, "obs": obs, "pred": pred, "error": pred - obs if np.isfinite(obs) and np.isfinite(pred) else np.nan, "z0_used_m": z0, "ustar_used_mps": ust, "z0_case": method, "source": o["src_ws"][j]})

        # Temperature methods.
        t_method_params = []
        if np.isfinite(o["wrf_z0"]) and np.isfinite(o["wrf_ust"]):
            z0h = max(Z0H_MIN, TEMP_Z0H_OVER_Z0M * o["wrf_z0"])
            t_method_params.append(("MOST_T_WRF_params", "hfx", z0h, np.nan, o["wrf_ust"]))
        if np.isfinite(o["u_free"].get("z0", np.nan)) and np.isfinite(o["u_free"].get("ustar", np.nan)):
            z0h = max(Z0H_MIN, TEMP_Z0H_OVER_Z0M * o["u_free"]["z0"])
            t_method_params.append(("MOST_T_inv_z0_ustar_WRF_HFX", "hfx", z0h, np.nan, o["u_free"]["ustar"]))
        if np.isfinite(o["t_free"].get("z0h", np.nan)) and np.isfinite(o["t_free"].get("theta_star", np.nan)):
            t_method_params.append(("MOST_T_free_inv_z0h_theta", "theta", o["t_free"]["z0h"], o["t_free"]["theta_star"], np.nan))

        fixed_z0h_cases = [
            ("MOST_T_station_fixed_z0h_theta", fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan))),
            ("MOST_T_sector_fixed_z0h_theta", fixed_lookup(z0h_maps, "sector", o, fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan)))),
            ("MOST_T_stability_fixed_z0h_theta", fixed_lookup(z0h_maps, "stability", o, fixed_lookup(z0h_maps, "station", o, o["t_free"].get("z0h", np.nan)))),
        ]
        for method, z0h_fixed in fixed_z0h_cases:
            fit = fit_theta_star_given_z0h(o["z_t"], o["used_T"], z0h_fixed, L, o.get("t_ref_fit_k", np.nan), z_ref=o.get("z_ref_fit", TEMP_REF_HEIGHT_M))
            if np.isfinite(fit.get("theta_star", np.nan)):
                t_method_params.append((method, "theta", z0h_fixed, fit["theta_star"], np.nan))

        for j, z in enumerate(o["z_t"]):
            obs = o["obs_T"][j]
            pred = o["wrf_T"][j]
            if np.isfinite(obs) or np.isfinite(pred):
                t_rows.append({**base, "height_label": o["hlabels_t"][j], "height_m": z, "method": "WRF_T", "obs": obs, "pred": pred, "error": pred - obs if np.isfinite(obs) and np.isfinite(pred) else np.nan, "z0h_used_m": np.nan, "theta_star_used_K": np.nan, "HFX_inv_Wm2": np.nan, "source": o["src_T"][j]})
            for method, mode, z0h, theta_star, ust in t_method_params:
                if mode == "hfx":
                    pred = temperature_with_hfx(z, z0h, ust, L, o["hfx"], o["psfc"], o.get("t_ref_hfx_k", np.nan), z_ref=o.get("z_ref_hfx", TEMP_REF_HEIGHT_M))
                    hfx_inv = np.nan
                else:
                    pred = temp_from_theta(z, z0h, theta_star, L, o.get("t_ref_fit_k", np.nan), z_ref=o.get("z_ref_fit", TEMP_REF_HEIGHT_M))
                    # HFX implied when a representative ustar is available.
                    u_for_hfx = o["u_free"].get("ustar", np.nan)
                    if np.isfinite(o["psfc"]) and np.isfinite(o.get("t_ref_fit_k", np.nan)) and np.isfinite(u_for_hfx) and np.isfinite(theta_star):
                        rho = float(o["psfc"]) / (RD_AIR * max(float(o.get("t_ref_fit_k", np.nan)), 150.0))
                        hfx_inv = -rho * CP_AIR * u_for_hfx * theta_star
                    else:
                        hfx_inv = np.nan
                t_rows.append({**base, "height_label": o["hlabels_t"][j], "height_m": z, "method": method, "obs": obs, "pred": pred, "error": pred - obs if np.isfinite(obs) and np.isfinite(pred) else np.nan, "z0h_used_m": z0h, "theta_star_used_K": theta_star, "HFX_inv_Wm2": hfx_inv, "source": o["src_T"][j]})
    return pd.DataFrame(u_rows), pd.DataFrame(t_rows)

# =============================================================================
# Plotting
# =============================================================================

def plot_metric_by_height(metrics: pd.DataFrame, out_png: Path, title: str, metric: str, method_order: List[str]) -> None:
    if plt is None or metrics.empty:
        return
    ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    for method in method_order:
        g = metrics[metrics["method"] == method].sort_values("height_m")
        if g.empty or metric not in g.columns:
            continue
        ax.plot(g[metric], g["height_m"], marker="o", label=method)
    ax.set_xlabel(metric)
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def _downsample_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= PLOT_MAX_POINTS:
        return df
    step = int(math.ceil(len(df) / PLOT_MAX_POINTS))
    return df.iloc[::step].copy()


def plot_timeseries(long_df: pd.DataFrame, out_root: Path, variable: str, selected_methods: List[str]) -> None:
    if plt is None or long_df.empty:
        return
    ensure_dir(out_root)
    if TIMESERIES_PLOT_DATASET_VERSIONS:
        long_df = long_df[long_df["dataset_version"].isin(TIMESERIES_PLOT_DATASET_VERSIONS)].copy()
    for keys, g0 in long_df.groupby(["relative_file", "site_id", "dataset_version", "height_m"], dropna=False):
        rel_path, site_id, dataset_version, h = keys
        g0 = g0.copy()
        if PLOT_SELECTED_METHODS_ONLY:
            g0 = g0[g0["method"].isin(selected_methods)]
        if g0.empty:
            continue
        stem = f"{safe_name(dataset_version)}__{safe_name(site_id)}__{safe_name(Path(str(rel_path)).name)}__{variable}_{float(h):g}m"
        fig, ax = plt.subplots(figsize=(12, 4.8))
        # Plot OBS once from first method rows.
        obs_series = g0[["time", "obs"]].dropna().drop_duplicates(subset=["time"]).sort_values("time")
        obs_series = _downsample_for_plot(obs_series)
        if not obs_series.empty:
            ax.plot(obs_series["time"], obs_series["obs"], label="OBS", linewidth=1.6)
        for method in selected_methods:
            gm = g0[g0["method"] == method][["time", "pred"]].dropna().sort_values("time")
            if gm.empty:
                continue
            gm = _downsample_for_plot(gm)
            ax.plot(gm["time"], gm["pred"], label=method, linewidth=1.0, alpha=0.9)
        ax.set_title(f"{dataset_version} | {site_id} | {variable} {float(h):g} m")
        ax.set_ylabel("Wind speed (m/s)" if variable == "U" else "Temperature (K)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_root / f"{stem}_timeseries.png", dpi=PLOT_DPI)
        plt.close(fig)

        # Error time series.
        fig, ax = plt.subplots(figsize=(12, 4.2))
        for method in selected_methods:
            gm = g0[g0["method"] == method][["time", "error"]].dropna().sort_values("time")
            if gm.empty:
                continue
            gm = _downsample_for_plot(gm)
            ax.plot(gm["time"], gm["error"], label=method, linewidth=1.0, alpha=0.9)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_title(f"{dataset_version} | {site_id} | {variable} error {float(h):g} m")
        ax.set_ylabel("pred - obs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_root / f"{stem}_error_timeseries.png", dpi=PLOT_DPI)
        plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, out_root: Path, variable: str) -> None:
    if plt is None or summary.empty:
        return
    ensure_dir(out_root)
    for metric in ["RMSE", "MAE", "bias", "Pearson_r"]:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot_table(index="method", columns="dataset_version", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(pivot.index)), 5.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{variable} {metric}: D10 / D1H / D6H by method")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / f"{variable}_{metric}_resolution_method_bar.png", dpi=PLOT_DPI)
        plt.close(fig)

# =============================================================================
# Main
# =============================================================================

def run() -> None:
    out_root = ensure_dir(OUTPUT_ROOT)
    manifest_root = ensure_dir(out_root / "00_config_and_manifest")
    files = sorted(ML_READY_ROOT.glob(ML_READY_GLOB))
    if not files:
        raise FileNotFoundError(f"No ML-ready files found under {ML_READY_ROOT} with glob {ML_READY_GLOB}")
    print(f"[{CASE_NAME}] files found: {len(files)}")

    all_objects: List[Dict[str, Any]] = []
    file_manifest = []
    for csv_path in files:
        rel_path = str(csv_path.relative_to(ML_READY_ROOT))
        print("=" * 100)
        print(f"[{CASE_NAME}] Reading {csv_path}")
        df0_raw = read_ml_ready(csv_path)
        df0 = robust_filter_dataframe_for_diagnostics(df0_raw)
        file_manifest.append({"file": str(csv_path), "relative_file": rel_path, "n_rows_D10_raw": int(len(df0_raw)), "n_rows_D10": int(len(df0))})
        datasets = {"D10_10min_original": df0}
        for version, cfg in AGGREGATION_RULES.items():
            if version in DATASET_VERSIONS:
                agg = aggregate_dataset(df0, cfg["rule"], cfg["min_source_records"])
                datasets[version] = robust_filter_dataframe_for_diagnostics(agg)
                file_manifest[-1][f"n_rows_{version}"] = int(len(datasets[version]))
        for version in DATASET_VERSIONS:
            if version not in datasets:
                continue
            print(f"  Building profile objects: {version}, rows={len(datasets[version])}")
            objs = build_profile_objects(datasets[version], csv_path, rel_path, version)
            all_objects.extend(objs)
            print(f"  Objects: {len(objs)}")

    pd.DataFrame(file_manifest).to_csv(manifest_root / "input_file_manifest.csv", index=False, encoding="utf-8-sig")
    if all_objects:
        obj_summary = pd.DataFrame([{
            "relative_file": o["relative_file"],
            "site_id": o["site_id"],
            "dataset_version": o["dataset_version"],
            "n_ws_heights": int(np.isfinite(o["used_ws"]).sum()),
            "n_T_heights": int(np.isfinite(o["used_T"]).sum()),
            "has_u_fit": bool(np.isfinite(o["u_free"].get("z0", np.nan)) and np.isfinite(o["u_free"].get("ustar", np.nan))),
            "has_t_fit": bool(np.isfinite(o["t_free"].get("z0h", np.nan)) and np.isfinite(o["t_free"].get("theta_star", np.nan))),
        } for o in all_objects])
        obj_summary.groupby(["relative_file", "site_id", "dataset_version"], dropna=False).agg(
            n_objects=("site_id", "size"),
            median_ws_heights=("n_ws_heights", "median"),
            median_T_heights=("n_T_heights", "median"),
            u_fit_count=("has_u_fit", "sum"),
            t_fit_count=("has_t_fit", "sum"),
        ).reset_index().to_csv(manifest_root / "profile_object_validity_summary.csv", index=False, encoding="utf-8-sig")
    if not all_objects:
        raise RuntimeError("No profile objects were generated. Check column names and data availability.")

    z0_maps = build_fixed_maps(all_objects, "z0")
    z0h_maps = build_fixed_maps(all_objects, "z0h")
    z0_param_df, z0h_param_df = build_parameter_summary(all_objects, z0_maps, z0h_maps)
    param_root = ensure_dir(out_root / "10_parameter_diagnostics")
    z0_param_df.to_csv(param_root / "z0_parameter_timeseries_all.csv", index=False, encoding="utf-8-sig")
    z0h_param_df.to_csv(param_root / "z0h_parameter_timeseries_all.csv", index=False, encoding="utf-8-sig")

    # Fixed map summaries.
    fixed_rows = []
    for mode, mp in z0_maps.items():
        for k, v in mp.items():
            fixed_rows.append({"param": "z0", "mode": mode, "key": str(k), "value_m": v})
    for mode, mp in z0h_maps.items():
        for k, v in mp.items():
            fixed_rows.append({"param": "z0h", "mode": mode, "key": str(k), "value_m": v})
    pd.DataFrame(fixed_rows).to_csv(param_root / "fixed_z0_z0h_summary.csv", index=False, encoding="utf-8-sig")

    print("Building long U/T method tables...")
    u_long, t_long = build_long_tables(all_objects, z0_maps, z0h_maps)
    u_root = ensure_dir(out_root / "20_U_diagnostics")
    t_root = ensure_dir(out_root / "30_T_diagnostics")
    u_long.to_csv(u_root / "22_U_timeseries_long_all_methods_all_files.csv", index=False, encoding="utf-8-sig")
    t_long.to_csv(t_root / "32_T_timeseries_long_all_methods_all_files.csv", index=False, encoding="utf-8-sig")

    u_metrics = build_metrics_table(u_long, "U")
    t_metrics = build_metrics_table(t_long, "T")
    u_metrics.to_csv(u_root / "23_U_timeseries_metrics_by_file_height_method.csv", index=False, encoding="utf-8-sig")
    t_metrics.to_csv(t_root / "33_T_timeseries_metrics_by_file_height_method.csv", index=False, encoding="utf-8-sig")

    u_sum_method = build_summary_metrics(u_metrics, ["dataset_version", "method"])
    t_sum_method = build_summary_metrics(t_metrics, ["dataset_version", "method"])
    u_sum_site = build_summary_metrics(u_metrics, ["dataset_version", "site_id", "method"])
    t_sum_site = build_summary_metrics(t_metrics, ["dataset_version", "site_id", "method"])
    u_sum_height = build_summary_metrics(u_metrics, ["dataset_version", "height_m", "method"])
    t_sum_height = build_summary_metrics(t_metrics, ["dataset_version", "height_m", "method"])
    u_sum_method.to_csv(u_root / "24_U_summary_by_resolution_method.csv", index=False, encoding="utf-8-sig")
    t_sum_method.to_csv(t_root / "34_T_summary_by_resolution_method.csv", index=False, encoding="utf-8-sig")
    u_sum_site.to_csv(u_root / "25_U_summary_by_resolution_site_method.csv", index=False, encoding="utf-8-sig")
    t_sum_site.to_csv(t_root / "35_T_summary_by_resolution_site_method.csv", index=False, encoding="utf-8-sig")
    u_sum_height.to_csv(u_root / "26_U_summary_by_resolution_height_method.csv", index=False, encoding="utf-8-sig")
    t_sum_height.to_csv(t_root / "36_T_summary_by_resolution_height_method.csv", index=False, encoding="utf-8-sig")

    if SAVE_PLOTS and plt is not None:
        u_method_order = [
            "WRF", "Gryning_all_WRF", "Gryning_free_inv_z0_ustar",
            "Gryning_station_fixed_z0", "Gryning_fixed_z0_vegetation500",
            "Gryning_sector_fixed_z0", "Gryning_stability_fixed_z0",
            "logMOST_all_WRF", "logMOST_free_inv_z0_ustar",
        ]
        t_method_order = [
            "WRF_T", "MOST_T_WRF_params", "MOST_T_inv_z0_ustar_WRF_HFX",
            "MOST_T_free_inv_z0h_theta", "MOST_T_station_fixed_z0h_theta",
            "MOST_T_sector_fixed_z0h_theta", "MOST_T_stability_fixed_z0h_theta",
        ]
        # By-height metric line plots for every file/resolution.
        u_plot_root = ensure_dir(u_root / "27_U_metric_by_height_plots")
        for keys, g in u_metrics.groupby(["relative_file", "site_id", "dataset_version"], dropna=False):
            rel, site, version = keys
            if PLOT_METRIC_BY_HEIGHT_DATASET_VERSIONS and version not in PLOT_METRIC_BY_HEIGHT_DATASET_VERSIONS:
                continue
            stem = f"{safe_name(version)}__{safe_name(site)}__{safe_name(Path(str(rel)).name)}"
            for metric in ["bias", "MAE", "RMSE", "Pearson_r"]:
                plot_metric_by_height(g, u_plot_root / f"{stem}_U_{metric}_by_height.png", f"{version} | {site} | U {metric} by height", metric, u_method_order)
        t_plot_root = ensure_dir(t_root / "37_T_metric_by_height_plots")
        for keys, g in t_metrics.groupby(["relative_file", "site_id", "dataset_version"], dropna=False):
            rel, site, version = keys
            if PLOT_METRIC_BY_HEIGHT_DATASET_VERSIONS and version not in PLOT_METRIC_BY_HEIGHT_DATASET_VERSIONS:
                continue
            stem = f"{safe_name(version)}__{safe_name(site)}__{safe_name(Path(str(rel)).name)}"
            for metric in ["bias", "MAE", "RMSE", "Pearson_r"]:
                plot_metric_by_height(g, t_plot_root / f"{stem}_T_{metric}_by_height.png", f"{version} | {site} | T {metric} by height", metric, t_method_order)
        # Time-series plots for every tower and height.
        plot_timeseries(u_long, ensure_dir(u_root / "28_U_timeseries_plots"), "U", [
            "WRF", "Gryning_all_WRF", "Gryning_free_inv_z0_ustar", "Gryning_station_fixed_z0", "Gryning_fixed_z0_vegetation500", "Gryning_sector_fixed_z0"
        ])
        plot_timeseries(t_long, ensure_dir(t_root / "38_T_timeseries_plots"), "T", [
            "WRF_T", "MOST_T_WRF_params", "MOST_T_inv_z0_ustar_WRF_HFX", "MOST_T_free_inv_z0h_theta", "MOST_T_station_fixed_z0h_theta"
        ])
        plot_summary_bars(u_sum_method, ensure_dir(u_root / "29_U_summary_bar_plots"), "U")
        plot_summary_bars(t_sum_method, ensure_dir(t_root / "39_T_summary_bar_plots"), "T")

    config = {
        "case_name": CASE_NAME,
        "ml_ready_root": str(ML_READY_ROOT),
        "ml_ready_glob": ML_READY_GLOB,
        "output_root": str(OUTPUT_ROOT),
        "dataset_versions": list(DATASET_VERSIONS),
        "vegetation_z0_500m_by_site": VEGETATION_Z0_500M_BY_SITE,
        "default_vegetation_z0_m": DEFAULT_VEGETATION_Z0_M,
        "n_wind_sectors": N_WIND_SECTORS,
        "use_robust_filters": USE_ROBUST_FILTERS,
        "timeseries_plot_dataset_versions": list(TIMESERIES_PLOT_DATASET_VERSIONS),
        "T_reference_policy": "HFX/WRF-parameter T methods use T2 or lowest WRF_T only; free thermal inversion may use lowest used profile as diagnostic fallback.",
        "U_methods": ["WRF", "Gryning_all_WRF", "logMOST_all_WRF", "Gryning_free_inv_z0_ustar", "logMOST_free_inv_z0_ustar", "Gryning_station_fixed_z0", "Gryning_fixed_z0_vegetation500", "Gryning_sector_fixed_z0", "Gryning_stability_fixed_z0"],
        "T_methods": ["WRF_T", "MOST_T_WRF_params", "MOST_T_inv_z0_ustar_WRF_HFX", "MOST_T_free_inv_z0h_theta", "MOST_T_station_fixed_z0h_theta", "MOST_T_sector_fixed_z0h_theta", "MOST_T_stability_fixed_z0h_theta"],
        "interpretation_note": "Free inversion cases use same-time tower profiles and are diagnostic upper bounds; fixed z0/z0h cases reduce 10-min parameter jumping.",
    }
    with open(manifest_root / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"[{CASE_NAME}] diagnostics finished.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"U metrics : {u_root / '23_U_timeseries_metrics_by_file_height_method.csv'}")
    print(f"T metrics : {t_root / '33_T_timeseries_metrics_by_file_height_method.csv'}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print(traceback.format_exc())
        raise
