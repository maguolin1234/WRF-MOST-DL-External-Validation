#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huarui_A Code2 profile-model generalization application
========================================================

Applies saved Huarui_A multi-height Code2 models without retraining.

Station mapping
---------------
- C039801 / A1 uses the d03 ML-ready file.
- C039802 / A2 uses the d02 ML-ready file.

The script evaluates C2, C4b, C6m and C8d for CNN_LSTM and TCN. It reconstructs
direct WS, circular WD and log-residual TKE/temperature outputs, calculates
per-height and pooled ALL_PROFILE metrics, and interpolates only missing WRF and
Gryning/MOST predictor heights. Target-tower observations are never interpolated.

Tasks
-----
- cross_tower_same_time: source-model test endpoints shared with the target tower.

- cross_tower_other_time: all valid target-tower endpoints; overlapping calendar
  times are retained, so this is all-target-time coverage rather than strict
  non-overlapping temporal extrapolation.

For WD, saved heads remain at their physical height when target OBS/WRF exist;
only unavailable WD heights are rebased to the target tower's native 160-m WD
baseline for matched wind-vector and wind-rose-ready diagnostics.
"""
from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

TIME_RESOLUTION_MODE = "1h"
TIME_RESOLUTION_DATA_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\time_resolution_sensitivity_huarui_A"
)
DATASET_ROOTS: Dict[str, Path] = {
    "10min": TIME_RESOLUTION_DATA_ROOT / "D10_10min",
    "1h": TIME_RESOLUTION_DATA_ROOT / "D1H_1hour",
    "original": Path(
        r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\huarui_A_two_towers_full"
    ),
}
ML_READY_ROOT = DATASET_ROOTS[TIME_RESOLUTION_MODE]
CODE2_RESULT_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\correction\c2"
)
OUTPUT_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\correction\c2_generalization_huarui_A_same_and_all_target_time"
)
GRYNING_U_LONG_CSV = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\input_data\data_post_new\gryning_profile_full_diagnostics_huarui_A\20_U_diagnostics\22_U_timeseries_long_all_methods_all_files.csv"
)
GRYNING_SEARCH_ROOT = TIME_RESOLUTION_DATA_ROOT.parent
GRYNING_LONG_FILENAME = "22_U_timeseries_long_all_methods_all_files.csv"
GRYNING_AUTO_DISCOVER = True
GRYNING_U_METHOD = "Gryning_fixed_z0_vegetation500"
GRYNING_DATASET_VERSION = {
    "10min": "D10_10min_original",
    "1h": "D1H_1hour_aggregated",
    "original": "D10_10min_original",
}[TIME_RESOLUTION_MODE]
GRYNING_MIN_VALID_RATIO = 0.80

TIME_COL = "北京时间"
OBS_VALID_COL = "obs_valid_flag"
CASES_TO_APPLY = ["C2", "C4b", "C6m", "C8d"]
NETWORKS_TO_APPLY = ["CNN_LSTM", "TCN"]
SOURCE_SITES = ["C039801", "C039802"]
TARGET_SITES = ["C039801", "C039802"]

INTERPOLATE_FEATURES = True
INTERPOLATE_MISSING_MODEL_HEIGHTS = True
VERTICAL_INTERPOLATION_MAX_GAP_M = 100.0
AUDIT_ONLY = False
STOP_ON_ERROR = False
SAVE_PLOTS = True
PLOT_DPI = 300
MAX_TIMESERIES_POINTS = 2500

RUN_CROSS_TOWER_SAME_TIME = True
RUN_CROSS_TOWER_OTHER_TIME = True

REMAP_WD_HEAD_TO_TARGET_NATIVE_HEIGHT = True
TARGET_ALL_TIME_REQUIRE_VALID_WS = True

MOST_REFERENCE_INTERPRETATION = (
    "OBS-assisted diagnostic Gryning_fixed_z0_vegetation500; "
    "training-consistent but not strict deployment"
)
ALLOW_EXISTING_MOST_COLUMN_FALLBACK = True

SITE_CONFIG: Dict[str, Dict[str, Any]] = {
    "C039801": {"aliases": ["C039801", "A1"], "preferred_domain": "d03", "native_wd_m": 160},
    "C039802": {"aliases": ["C039802", "A2"], "preferred_domain": "d02", "native_wd_m": 160},
}

CASE_ALIASES: Dict[str, List[str]] = {
    "C2": ["C2", "noMOST_corrFDR"],
    "C4b": ["C4b", "MOSTinput_corrFDR_forceMOST"],
    "C6m": ["C6m", "PINN_MOSTmag_corrFDR_noSmooth"],
    "C8d": ["C8d", "MOSTinput_PINN_MOSTmag_corrFDR_forceMOST_noSmooth"],
}

# =============================================================================
# 2. DATA CLASSES
# =============================================================================

@dataclass
class TargetRecord:
    var: str
    height: int
    obs_col: str
    wrf_col: str
    target_col: str
    unit: str


@dataclass
class ModelArtifact:
    source_site: str
    network: str
    case: str
    model_dir: Path
    metadata_path: Path
    model_path: Path
    x_scaler_path: Path
    prediction_path: Optional[Path]
    config_path: Optional[Path]


@dataclass
class GeneralizationTask:
    name: str
    generalization_type: str
    source_site: str
    target_site: str
    time_mode: str

# =============================================================================
# 3. GENERIC UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: Any, max_len: int = 180) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(text))[:max_len]


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def path_contains_alias(path: Path, aliases: Sequence[str]) -> bool:
    """Match long station IDs anywhere, but short A1/A2 aliases only as path parts."""
    parts = {norm(x) for x in path.parts}
    full = norm(str(path))
    for alias in aliases:
        token = norm(alias)
        if not token:
            continue
        if token in parts:
            return True
        if len(token) >= 5 and token in full:
            return True
    return False



def normalize_network(value: Any) -> Optional[str]:
    n = norm(value)
    if "cnnlstm" in n:
        return "CNN_LSTM"
    if "tcn" in n:
        return "TCN"
    return None


def normalize_case(value: Any) -> Optional[str]:
    n = norm(value)
    for case, aliases in CASE_ALIASES.items():
        if any(n == norm(a) or n.endswith(norm(a)) for a in aliases):
            return case
    return None


def infer_site_from_path(path: Path) -> Optional[str]:
    for site, cfg in SITE_CONFIG.items():
        if path_contains_alias(path, cfg["aliases"]):
            return site
    return None


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    mapping = {norm(c): c for c in df.columns}
    for c in candidates:
        hit = mapping.get(norm(c))
        if hit is not None:
            return hit
    return None


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")



def parse_datetime_mixed_safe(series: pd.Series, label: str = "time") -> pd.Series:
    """Parse mixed timestamp formats and normalize timezone-aware values to naive Beijing time."""
    raw = pd.Series(series, copy=True)
    try:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, errors="coerce")

    def one(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.NaT
        if isinstance(value, (int, float, np.integer, np.floating)):
            fv = float(value)
            if 20000.0 <= fv <= 80000.0:
                try:
                    return pd.Timestamp("1899-12-30") + pd.to_timedelta(fv, unit="D")
                except Exception:
                    return pd.NaT
        try:
            ts = pd.Timestamp(pd.to_datetime(value, errors="coerce"))
        except Exception:
            return pd.NaT
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is not None:
            try:
                ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
            except Exception:
                try:
                    ts = ts.tz_localize(None)
                except Exception:
                    pass
        return ts

    if parsed.dtype == object or int(parsed.notna().sum()) < int(raw.notna().sum()):
        scalar = pd.to_datetime(raw.map(one), errors="coerce")
        if int(scalar.notna().sum()) >= int(parsed.notna().sum()):
            parsed = scalar
    try:
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    except Exception:
        pass
    return pd.Series(parsed, index=raw.index, name=getattr(series, "name", label))

def resolve_gryning_long_csv() -> Path:
    """Discover and select the Huarui_A Gryning table with the best two-site 160-m coverage."""
    global _GRYNING_RESOLVED_PATH
    cached = globals().get("_GRYNING_RESOLVED_PATH", None)
    if cached is not None and Path(cached).exists():
        return Path(cached)

    candidates: List[Path] = [Path(GRYNING_U_LONG_CSV)]
    root = Path(GRYNING_SEARCH_ROOT)
    if GRYNING_AUTO_DISCOVER and root.exists():
        candidates.extend(root.rglob(GRYNING_LONG_FILENAME))
        # Some corrected copies may have a different filename but the same schema.
        for p in root.rglob("*.csv"):
            if "gryning" in str(p).lower():
                candidates.append(p)

    required = {"time", "method", "height_m", "pred"}
    unique: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if not p.exists() or not p.is_file():
            continue
        try:
            if required.issubset(pd.read_csv(p, nrows=0).columns):
                unique.append(p)
        except Exception:
            continue
    if not unique:
        raise FileNotFoundError(
            f"No schema-compatible Gryning long table found. Configured={GRYNING_U_LONG_CSV}; search_root={GRYNING_SEARCH_ROOT}"
        )

    ranked: List[Tuple[int, int, int, Path]] = []
    for p in unique:
        try:
            g = pd.read_csv(p)
            g["time"] = parse_datetime_mixed_safe(g["time"], "time")
            g["height_m"] = numeric(g["height_m"])
            g["pred"] = numeric(g["pred"])
            mask = g["method"].astype(str).map(norm).eq(norm(GRYNING_U_METHOD))
            if "dataset_version" in g.columns:
                ds = g["dataset_version"].astype(str).map(norm)
                req = norm(GRYNING_DATASET_VERSION)
                mask &= ds.eq(req) | ds.str.contains(req, na=False)
            mask &= np.isclose(g["height_m"].to_numpy(dtype=float), 160.0, atol=1.0e-6, equal_nan=False)
            counts = []
            for site in SOURCE_SITES:
                sm = np.zeros(len(g), dtype=bool)
                for c in [x for x in ["site_id", "relative_file", "csv_file"] if x in g.columns]:
                    sm |= g[c].astype(str).str.contains(site, case=False, na=False, regex=False).to_numpy()
                counts.append(int(g.loc[mask & sm, "time"].dropna().nunique()))
            ranked.append((min(counts) if counts else 0, sum(counts), -len(str(p)), p))
        except Exception:
            ranked.append((0, 0, -len(str(p)), p))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = ranked[0][3]
    _GRYNING_RESOLVED_PATH = selected
    print(
        f"[Gryning] selected={selected} | "
        f"min_two_site_160m_unique_times={ranked[0][0]} | sum={ranked[0][1]}"
    )
    return selected


def merge_gryning_height_by_time(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    value_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merge one Gryning height using exact/floor/round and guarded constant-step alignment."""
    left = left_df.copy()
    right = right_df.copy()
    left[TIME_COL] = parse_datetime_mixed_safe(left[TIME_COL], TIME_COL)
    right["time"] = parse_datetime_mixed_safe(right["time"], "time")
    right["pred"] = numeric(right["pred"])
    right = right.dropna(subset=["time", "pred"])

    unit = "h" if str(TIME_RESOLUTION_MODE).lower() == "1h" else "10min"
    step = pd.Timedelta(hours=1) if unit == "h" else pd.Timedelta(minutes=10)
    max_steps = 24 if unit == "h" else 144
    lraw, rraw = left[TIME_COL], right["time"]

    def score(a, b) -> int:
        return len(set(pd.Series(a).dropna().unique()).intersection(set(pd.Series(b).dropna().unique())))

    candidates = [
        ("exact", 0, lraw, rraw),
        (f"floor_{unit}", 0, lraw.dt.floor(unit), rraw.dt.floor(unit)),
        (f"round_{unit}", 0, lraw.dt.round(unit), rraw.dt.round(unit)),
    ]
    base_l, base_r = lraw.dt.floor(unit), rraw.dt.floor(unit)
    for k in range(-max_steps, max_steps + 1):
        if k:
            candidates.append((f"floor_{unit}_shift", k, base_l, base_r + k * step))
    ranked = [(score(a, b), -abs(k), mode, k, a, b) for mode, k, a, b in candidates]
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    overlap, _, mode, shift_steps, lkey, rkey = ranked[0]

    left["__merge_time"] = pd.Series(lkey, index=left.index)
    right["__merge_time"] = pd.Series(rkey, index=right.index)
    rr = right.groupby("__merge_time", as_index=False)["pred"].mean().rename(columns={"pred": value_name})
    out = left.merge(rr, how="left", on="__merge_time").drop(columns=["__merge_time"])
    valid = int(numeric(out[value_name]).notna().sum())
    total = int(len(out))
    return out, {
        "alignment_mode": mode,
        "alignment_shift_steps": int(shift_steps),
        "alignment_shift_timedelta": str(shift_steps * step),
        "unique_time_overlap": int(overlap),
        "valid_after_merge": valid,
        "coverage_after_merge": float(valid / total) if total else 0.0,
        "ml_time_min": str(lraw.min()) if lraw.notna().any() else None,
        "ml_time_max": str(lraw.max()) if lraw.notna().any() else None,
        "gryning_time_min": str(rraw.min()) if rraw.notna().any() else None,
        "gryning_time_max": str(rraw.max()) if rraw.notna().any() else None,
    }

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to load saved Code2 models.") from exc


def load_keras_model(path: Path):
    tf = import_tf()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIME_COL in out.columns:
        t = pd.to_datetime(out[TIME_COL], errors="coerce")
        hour = t.dt.hour.astype(float) + t.dt.minute.astype(float) / 60.0
        month = t.dt.month.astype(float)
        if "hour_sin" not in out.columns:
            out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_cos" not in out.columns:
            out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        if "month_sin" not in out.columns:
            out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        if "month_cos" not in out.columns:
            out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    return out


def load_ml_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing time column {TIME_COL}: {path}")
    df[TIME_COL] = parse_datetime_mixed_safe(df[TIME_COL], TIME_COL)
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).drop_duplicates(TIME_COL).reset_index(drop=True)
    return add_time_features(df)


# =============================================================================
# 4. FILE AND MODEL DISCOVERY
# =============================================================================

def discover_ml_ready_optional(site: str) -> Optional[Path]:
    if not ML_READY_ROOT.exists():
        return None
    cfg = SITE_CONFIG[site]
    hits = [
        p for p in ML_READY_ROOT.rglob("ML_ready_ref10_*.csv")
        if path_contains_alias(p, cfg["aliases"])
    ]
    if not hits:
        return None
    preferred = str(cfg.get("preferred_domain", "")).lower()
    hits.sort(key=lambda p: (
        0 if preferred and preferred in p.name.lower() else 1,
        len(p.parts), len(str(p)),
    ))
    return hits[0]



def discover_code2_models() -> List[ModelArtifact]:
    found: List[ModelArtifact] = []
    for meta_path in CODE2_RESULT_ROOT.rglob("metadata.json"):
        model_dir = meta_path.parent
        model_path = model_dir / "model.keras"
        scaler_path = model_dir / "x_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            continue
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        config_path = model_dir / "case_v4_config.json"
        cfg = read_json(config_path) if config_path.exists() else {}
        case = (
            normalize_case(cfg.get("requested_case", ""))
            or normalize_case(cfg.get("feature_mode_name", ""))
            or normalize_case(meta.get("feature_mode", ""))
            or normalize_case(model_dir.name)
        )
        if case not in CASES_TO_APPLY:
            continue
        source_site = infer_site_from_path(meta_path)
        network = normalize_network(meta.get("network", "")) or normalize_network(meta.get("model_name", "")) or normalize_network(meta_path)
        if source_site not in SOURCE_SITES or network not in NETWORKS_TO_APPLY:
            continue
        prediction = model_dir / "predictions_train_val_test.csv"
        found.append(ModelArtifact(
            source_site=source_site, network=network, case=case,
            model_dir=model_dir, metadata_path=meta_path, model_path=model_path,
            x_scaler_path=scaler_path,
            prediction_path=prediction if prediction.exists() else None,
            config_path=config_path if config_path.exists() else None,
        ))
    best: Dict[Tuple[str, str, str], ModelArtifact] = {}
    for art in found:
        key = (art.source_site, art.network, art.case)
        score = (50 if "cases" in [x.lower() for x in art.model_dir.parts] else 0) - len(art.model_dir.parts)
        old = best.get(key)
        if old is None:
            best[key] = art
        else:
            old_score = (50 if "cases" in [x.lower() for x in old.model_dir.parts] else 0) - len(old.model_dir.parts)
            if score > old_score:
                best[key] = art
    return sorted(best.values(), key=lambda a: (a.source_site, a.network, a.case))


def source_times(artifact: ModelArtifact, split: Optional[str]) -> Optional[pd.DatetimeIndex]:
    if artifact.prediction_path is None or not artifact.prediction_path.exists():
        return None
    try:
        d = pd.read_csv(artifact.prediction_path, usecols=lambda c: c in {TIME_COL, "split"})
        d[TIME_COL] = parse_datetime_mixed_safe(d[TIME_COL], TIME_COL)
        if split is not None:
            d = d[d["split"].astype(str).str.lower().eq(split.lower())]
        d = d.dropna(subset=[TIME_COL])
        return pd.DatetimeIndex(d[TIME_COL].drop_duplicates().sort_values())
    except Exception:
        return None


# =============================================================================
# 5. GRYNING INJECTION
# =============================================================================

def wrf_ws_candidates(h: int) -> List[str]:
    return [
        f"WRF_WS{h}_interp (m/s)", f"WRF_WS_{h}m_interp (m/s)",
        f"WRF_WS{h} (m/s)", f"WRF_WS_{h}m (m/s)", f"WRF_WS{h}",
    ]


def wrf_u_candidates(h: int) -> List[str]:
    return [
        f"WRF_U{h}_interp (m/s)", f"WRF_U_{h}m_interp (m/s)",
        f"WRF_U{h} (m/s)", f"WRF_U_{h}m (m/s)", f"WRF_U{h}",
        f"WRF_U{h}_auto_from_WS_WD (m/s)",
    ]


def wrf_v_candidates(h: int) -> List[str]:
    return [
        f"WRF_V{h}_interp (m/s)", f"WRF_V_{h}m_interp (m/s)",
        f"WRF_V{h} (m/s)", f"WRF_V_{h}m (m/s)", f"WRF_V{h}",
        f"WRF_V{h}_auto_from_WS_WD (m/s)",
    ]


def most_speed_candidates(h: int) -> List[str]:
    return [
        f"U_MO{h} (m/s)", f"MOST_WS{h} (m/s)",
        f"U_MO_{h}m (m/s)", f"MOST_WS_{h}m (m/s)",
        f"U_MO{h}", f"MOST_WS{h}",
    ]


def _extract_available_heights(columns: Iterable[str], prefixes: Sequence[str]) -> List[int]:
    heights = set()
    for col in columns:
        compact = str(col).replace("_", "")
        for prefix in prefixes:
            m = re.search(rf"{re.escape(prefix)}(\d+)", compact, flags=re.IGNORECASE)
            if m:
                heights.add(int(m.group(1)))
                break
    return sorted(heights)


def _bracketing_heights(available: Sequence[int], target: int) -> Optional[Tuple[int, int]]:
    lower = [h for h in available if h < target]
    upper = [h for h in available if h > target]
    if not lower or not upper:
        return None
    h0, h1 = max(lower), min(upper)
    if float(h1 - h0) > float(VERTICAL_INTERPOLATION_MAX_GAP_M):
        return None
    return h0, h1


def _wrf_uv_at_height(df: pd.DataFrame, h: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Return WRF U/V at one height without using observations.

    Existing U/V columns are preferred. If unavailable, U/V are derived from
    WRF WS/WD using the meteorological convention used in training.
    """
    u_col = first_existing(df, wrf_u_candidates(h))
    v_col = first_existing(df, wrf_v_candidates(h))
    if u_col is not None and v_col is not None:
        return (
            numeric(df[u_col]).to_numpy(dtype=float),
            numeric(df[v_col]).to_numpy(dtype=float),
            {"height_m": h, "source": "existing_uv", "u_col": u_col, "v_col": v_col},
        )

    ws_col = first_existing(df, wrf_ws_candidates(h))
    wd_col = first_existing(df, wrf_wd_candidates(h))
    if ws_col is None or wd_col is None:
        return None, None, {
            "height_m": h, "source": "unavailable",
            "u_col": u_col, "v_col": v_col, "ws_col": ws_col, "wd_col": wd_col,
        }
    ws = numeric(df[ws_col]).to_numpy(dtype=float)
    wd = numeric(df[wd_col]).to_numpy(dtype=float)
    rad = np.deg2rad(wd)
    u = -ws * np.sin(rad)
    v = -ws * np.cos(rad)
    return u, v, {
        "height_m": h, "source": "derived_from_wrf_ws_wd",
        "ws_col": ws_col, "wd_col": wd_col,
    }


def _create_wrf_wind_height(df: pd.DataFrame, target_h: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create missing WRF U/V/WS/WD at target_h from bracketing levels."""
    out = df.copy()
    canonical = {
        "u": f"WRF_U{target_h}_interp (m/s)",
        "v": f"WRF_V{target_h}_interp (m/s)",
        "ws": f"WRF_WS{target_h}_interp (m/s)",
        "wd": f"WRF_WD{target_h}_interp (deg)",
    }
    existing = {k: first_existing(out, cands) for k, cands in {
        "u": wrf_u_candidates(target_h),
        "v": wrf_v_candidates(target_h),
        "ws": wrf_ws_candidates(target_h),
        "wd": wrf_wd_candidates(target_h),
    }.items()}
    if all(existing.values()):
        for key, col in existing.items():
            if canonical[key] not in out.columns:
                out[canonical[key]] = numeric(out[col])
        return out, {
            "height_m": target_h, "status": "already_available",
            "existing_columns": existing, "created_columns": [],
        }

    available = _extract_available_heights(out.columns, ["WRFU", "WRFV", "WRFWS", "WRFWD"])
    valid_heights = []
    source_details: Dict[int, Dict[str, Any]] = {}
    source_uv: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for h in available:
        u, v, detail = _wrf_uv_at_height(out, h)
        source_details[h] = detail
        if u is not None and v is not None:
            valid_heights.append(h)
            source_uv[h] = (u, v)
    bracket = _bracketing_heights(valid_heights, target_h)
    if bracket is None:
        return out, {
            "height_m": target_h, "status": "no_bracketing_wrf_levels",
            "available_heights": available, "valid_uv_heights": valid_heights,
        }

    h0, h1 = bracket
    alpha = (float(target_h) - float(h0)) / (float(h1) - float(h0))
    u0, v0 = source_uv[h0]
    u1, v1 = source_uv[h1]
    u = (1.0 - alpha) * u0 + alpha * u1
    v = (1.0 - alpha) * v0 + alpha * v1
    ws = np.sqrt(u ** 2 + v ** 2)
    wd = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    values = {"u": u, "v": v, "ws": ws, "wd": wd}
    created = []
    for key, col in canonical.items():
        if col not in out.columns or numeric(out[col]).notna().sum() == 0:
            out[col] = values[key]
            created.append(col)

    # Also fill an exact existing-but-empty alias when metadata refers to it.
    for key, col in existing.items():
        if col is not None and numeric(out[col]).notna().sum() == 0:
            out[col] = values[key]
            if col not in created:
                created.append(col)

    return out, {
        "height_m": target_h, "status": "interpolated_from_wrf_vector",
        "lower_height_m": h0, "upper_height_m": h1, "alpha": alpha,
        "lower_source": source_details.get(h0),
        "upper_source": source_details.get(h1),
        "created_columns": created,
        "valid_rows": int(np.isfinite(ws).sum()),
        "uses_target_observations": False,
    }


def _create_most_height(df: pd.DataFrame, target_h: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create missing Gryning/MOST speed at target_h from bracketing levels."""
    out = df.copy()
    u_target = f"U_MO{target_h} (m/s)"
    most_target = f"MOST_WS{target_h} (m/s)"
    diff_target = f"MO_minus_WRF{target_h} (m/s)"

    direct = first_existing(out, most_speed_candidates(target_h))
    if direct is not None and numeric(out[direct]).notna().sum() > 0:
        speed = numeric(out[direct])
        out[u_target] = speed
        out[most_target] = speed
        wrf_ws = first_existing(out, wrf_ws_candidates(target_h))
        if wrf_ws is not None:
            out[diff_target] = speed - numeric(out[wrf_ws])
        return out, {
            "height_m": target_h, "status": "already_available",
            "source_col": direct, "wrf_ws_col": wrf_ws,
            "uses_target_observations": False,
        }

    available = _extract_available_heights(out.columns, ["UMO", "MOSTWS"])
    valid = []
    cols: Dict[int, str] = {}
    for h in available:
        col = first_existing(out, most_speed_candidates(h))
        if col is not None and numeric(out[col]).notna().sum() > 0:
            valid.append(h)
            cols[h] = col
    bracket = _bracketing_heights(valid, target_h)
    if bracket is None:
        return out, {
            "height_m": target_h, "status": "no_bracketing_most_levels",
            "available_heights": available, "valid_heights": valid,
        }

    h0, h1 = bracket
    alpha = (float(target_h) - float(h0)) / (float(h1) - float(h0))
    s0 = numeric(out[cols[h0]]).to_numpy(dtype=float)
    s1 = numeric(out[cols[h1]]).to_numpy(dtype=float)
    speed = (1.0 - alpha) * s0 + alpha * s1
    out[u_target] = speed
    out[most_target] = speed
    wrf_ws = first_existing(out, wrf_ws_candidates(target_h))
    if wrf_ws is not None:
        out[diff_target] = speed - numeric(out[wrf_ws])

    return out, {
        "height_m": target_h, "status": "interpolated_from_most_speed",
        "lower_height_m": h0, "upper_height_m": h1, "alpha": alpha,
        "lower_col": cols[h0], "upper_col": cols[h1],
        "wrf_ws_col": wrf_ws,
        "created_columns": [u_target, most_target] + ([diff_target] if wrf_ws is not None else []),
        "valid_rows": int(np.isfinite(speed).sum()),
        "uses_target_observations": False,
    }


def ensure_model_height_support(
    df: pd.DataFrame,
    required_heights: Sequence[int],
    feature_cols: Sequence[str],
    records: Sequence[TargetRecord],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fill missing predictor heights required by the saved source model.

    Only WRF and Gryning/MOST predictors are interpolated. OBS fields are never
    interpolated or synthesized; consequently a target height without an actual
    observation remains unavailable for verification and contributes N=0.
    """
    out = df.copy()
    report: Dict[str, Any] = {
        "enabled": bool(INTERPOLATE_MISSING_MODEL_HEIGHTS),
        "required_heights": sorted({int(h) for h in required_heights}),
        "wrf_wind": [], "most_speed": [],
        "obs_interpolation_performed": False,
    }
    if not INTERPOLATE_MISSING_MODEL_HEIGHTS:
        return out, report

    # Determine which heights require WRF wind structure from saved features or outputs.
    wrf_needed = set()
    most_needed = set()
    for h in report["required_heights"]:
        feature_text = " ".join(str(c) for c in feature_cols)
        if any(token in feature_text for token in [f"WRF_U{h}", f"WRF_V{h}", f"WRF_WS{h}", f"WRF_WD{h}"]):
            wrf_needed.add(h)
        if any(token in feature_text for token in [f"U_MO{h}", f"MOST_WS{h}", f"MO_minus_WRF{h}"]):
            most_needed.add(h)
    for r in records:
        if r.var in {"ws", "wd"}:
            wrf_needed.add(int(r.height))

    for h in sorted(wrf_needed):
        needs_creation = (
            first_existing(out, wrf_u_candidates(h)) is None
            or first_existing(out, wrf_v_candidates(h)) is None
            or first_existing(out, wrf_ws_candidates(h)) is None
            or first_existing(out, wrf_wd_candidates(h)) is None
        )
        if needs_creation:
            out, r = _create_wrf_wind_height(out, h)
            report["wrf_wind"].append(r)

    # MOST interpolation is performed after WRF wind interpolation so that
    # MO_minus_WRF can be reconstructed consistently at the target height.
    for h in sorted(most_needed):
        needs_creation = any(
            c not in out.columns or numeric(out[c]).notna().sum() == 0
            for c in [f"U_MO{h} (m/s)", f"MOST_WS{h} (m/s)", f"MO_minus_WRF{h} (m/s)"]
        )
        if needs_creation:
            out, r = _create_most_height(out, h)
            report["most_speed"].append(r)

    return out, report


def inject_gryning(df: pd.DataFrame, site: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject the Huarui_A Gryning_fixed_z0_vegetation500 profile at all available heights."""
    out = df.copy()
    try:
        table = resolve_gryning_long_csv()
    except Exception as exc:
        return out, {
            "site": site,
            "gryning_csv": str(GRYNING_U_LONG_CSV),
            "method": GRYNING_U_METHOD,
            "dataset_version": GRYNING_DATASET_VERSION,
            "interpretation": MOST_REFERENCE_INTERPRETATION,
            "status": "missing_or_invalid_long_table",
            "error": repr(exc),
            "heights": [],
        }
    report: Dict[str, Any] = {
        "site": site,
        "gryning_csv": str(table),
        "method": GRYNING_U_METHOD,
        "dataset_version": GRYNING_DATASET_VERSION,
        "interpretation": MOST_REFERENCE_INTERPRETATION,
        "minimum_valid_ratio": GRYNING_MIN_VALID_RATIO,
        "heights": [],
    }
    g = pd.read_csv(table)
    g["time"] = parse_datetime_mixed_safe(g["time"], "time")
    g["height_m"] = numeric(g["height_m"])
    g["pred"] = numeric(g["pred"])
    method_norm = g["method"].astype(str).map(norm)
    mask = method_norm.eq(norm(GRYNING_U_METHOD))
    if "dataset_version" in g.columns:
        ds = g["dataset_version"].astype(str).map(norm)
        req = norm(GRYNING_DATASET_VERSION)
        mask &= ds.eq(req) | ds.str.contains(req, na=False)
    site_cols = [c for c in ["site_id", "relative_file", "csv_file"] if c in g.columns]
    if not site_cols:
        raise ValueError("Gryning table has no site_id/relative_file/csv_file column.")
    sm = np.zeros(len(g), dtype=bool)
    for c in site_cols:
        sm |= g[c].astype(str).str.contains(site, case=False, na=False, regex=False).to_numpy()
    mask &= sm
    gs = g.loc[mask].dropna(subset=["time", "height_m", "pred"]).copy()
    if gs.empty:
        report["status"] = "no_site_rows"
        return out, report

    for hval, gh in gs.groupby("height_m"):
        if not np.isfinite(hval):
            continue
        h = int(round(float(hval)))
        value_name = f"__gryn_{h}"
        out, align = merge_gryning_height_by_time(out, gh[["time", "pred"]], value_name)
        coverage = float(align["coverage_after_merge"])
        accepted = coverage >= float(GRYNING_MIN_VALID_RATIO)
        gryn = numeric(out[value_name]) if accepted else pd.Series(np.nan, index=out.index)
        u_col = f"U_MO{h} (m/s)"
        most_col = f"MOST_WS{h} (m/s)"
        diff_col = f"MO_minus_WRF{h} (m/s)"
        out[u_col] = gryn
        out[most_col] = gryn
        wrf_col = first_existing(out, [
            f"WRF_WS{h}_interp (m/s)", f"WRF_WS_{h}m_interp (m/s)",
            f"WRF_WS{h} (m/s)", f"WRF_WS_{h}m (m/s)", f"WRF_WS{h}",
        ])
        if wrf_col is not None:
            out[diff_col] = gryn - numeric(out[wrf_col])
        align.update({
            "height_m": h,
            "accepted": bool(accepted),
            "valid": int(gryn.notna().sum()),
            "wrf_ws_col": wrf_col,
        })
        report["heights"].append(align)
        out = out.drop(columns=[value_name])
    report["status"] = "injected"
    return out, report


# =============================================================================
# 6. TARGET RECORD ALIGNMENT AND DERIVED FEATURES
# =============================================================================

def obs_ws_candidates(h: int) -> List[str]:
    return [f"OBS_WS{h} (m/s)", f"OBS_WS_{h}m (m/s)", f"OBS_WS{h}"]


def obs_wd_candidates(h: int) -> List[str]:
    return [f"OBS_WD{h} (deg)", f"OBS_WD_{h}m (deg)", f"OBS_WD{h}"]


def obs_tke_candidates(h: int) -> List[str]:
    return [f"OBS_TKE{h} (m2/s2)", f"OBS_TKE{h}_from_TI (m2/s2)", f"OBS_TKE_{h}m (m2/s2)", f"OBS_TKE{h}"]


def wrf_wd_candidates(h: int) -> List[str]:
    return [f"WRF_WD{h}_interp (deg)", f"WRF_WD_{h}m_interp (deg)", f"WRF_WD{h} (deg)", f"WRF_WD{h}"]


def wrf_tke_candidates(h: int) -> List[str]:
    return [f"WRF_TKE{h} (m2/s2)", f"WRF_TKE_{h}m (m2/s2)", f"WRF_TKE{h}_interp (m2/s2)", f"WRF_TKE{h}"]


def obs_temp_candidates(h: int) -> List[str]:
    return [f"OBS_T{h} (K)", f"OBS_T_{h}m (K)", f"OBS_TEMP{h} (K)", f"OBS_T{h}"]


def wrf_temp_candidates(h: int) -> List[str]:
    return [f"WRF_T{h}_interp (K)", f"WRF_T_{h}m_interp (K)", f"WRF_TEMP{h}_interp (K)", f"WRF_T{h} (K)", f"WRF_T{h}"]


def derive_common_wrf_features(df: pd.DataFrame, heights: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for h in heights:
        ws_col = first_existing(out, wrf_ws_candidates(h))
        wd_col = first_existing(out, wrf_wd_candidates(h))
        if wd_col is not None:
            rad = np.deg2rad(numeric(out[wd_col]).to_numpy(dtype=float))
            if f"WRF_WD{h}_sin" not in out.columns:
                out[f"WRF_WD{h}_sin"] = np.sin(rad)
            if f"WRF_WD{h}_cos" not in out.columns:
                out[f"WRF_WD{h}_cos"] = np.cos(rad)
        if ws_col is not None and wd_col is not None:
            ws = numeric(out[ws_col]).to_numpy(dtype=float)
            rad = np.deg2rad(numeric(out[wd_col]).to_numpy(dtype=float))
            if f"WRF_U{h}_auto_from_WS_WD (m/s)" not in out.columns:
                out[f"WRF_U{h}_auto_from_WS_WD (m/s)"] = -ws * np.sin(rad)
            if f"WRF_V{h}_auto_from_WS_WD (m/s)" not in out.columns:
                out[f"WRF_V{h}_auto_from_WS_WD (m/s)"] = -ws * np.cos(rad)
    return out


def records_from_metadata(meta: Dict[str, Any]) -> List[TargetRecord]:
    return [TargetRecord(**r) for r in meta.get("target_records", [])]


def align_records(
    df: pd.DataFrame,
    saved: Sequence[TargetRecord],
    task: Optional[GeneralizationTask] = None,
) -> Tuple[pd.DataFrame, List[TargetRecord], List[Dict[str, Any]]]:
    """Align saved heads to Huarui_A target columns.

    Huarui_A may contain WD observations at more than one height (commonly 130
    and 160 m). A saved WD head therefore remains at its physical height whenever
    the target tower has real OBS and WRF data there. It is rebased to the target
    tower's configured native 160-m WD height only when the saved height is not
    evaluable. Target observations are never interpolated.
    """
    out = df.copy()
    aligned: List[TargetRecord] = []
    report: List[Dict[str, Any]] = []
    native_wd = None
    if REMAP_WD_HEAD_TO_TARGET_NATIVE_HEIGHT and task is not None and task.target_site in SITE_CONFIG:
        native_wd = int(SITE_CONFIG[task.target_site]["native_wd_m"])

    for r in saved:
        saved_h = int(r.height)
        aligned_h = saved_h
        mapping_mode = "saved_physical_height"

        if r.var == "ws":
            obs_candidates, wrf_candidates = obs_ws_candidates(saved_h), wrf_ws_candidates(saved_h)
        elif r.var == "wd":
            saved_obs = first_existing(out, obs_wd_candidates(saved_h))
            saved_wrf = first_existing(out, wrf_wd_candidates(saved_h))
            saved_evaluable = (
                saved_obs is not None and saved_wrf is not None
                and numeric(out[saved_obs]).notna().sum() > 0
                and numeric(out[saved_wrf]).notna().sum() > 0
            )
            if native_wd is not None and not saved_evaluable:
                aligned_h = native_wd
                obs_candidates, wrf_candidates = obs_wd_candidates(native_wd), wrf_wd_candidates(native_wd)
                mapping_mode = "target_native_wd_head_rebase_missing_saved_height"
            else:
                obs_candidates, wrf_candidates = obs_wd_candidates(saved_h), wrf_wd_candidates(saved_h)
        elif r.var == "log_tke":
            obs_candidates, wrf_candidates = obs_tke_candidates(saved_h), wrf_tke_candidates(saved_h)
        elif r.var == "temp":
            obs_candidates, wrf_candidates = obs_temp_candidates(saved_h), wrf_temp_candidates(saved_h)
        else:
            obs_candidates, wrf_candidates = [r.obs_col], [r.wrf_col]

        if r.var == "wd" and mapping_mode.startswith("target_native"):
            obs_col = first_existing(out, obs_candidates)
            wrf_col = first_existing(out, wrf_candidates)
        else:
            obs_col = r.obs_col if r.obs_col in out.columns else first_existing(out, obs_candidates)
            wrf_col = r.wrf_col if r.wrf_col in out.columns else first_existing(out, wrf_candidates)
        if obs_col is None:
            obs_col = r.obs_col
            if obs_col not in out.columns:
                out[obs_col] = np.nan
        if wrf_col is None:
            wrf_col = r.wrf_col
            if wrf_col not in out.columns:
                out[wrf_col] = np.nan
        if r.target_col not in out.columns:
            out[r.target_col] = np.nan

        aligned.append(TargetRecord(r.var, aligned_h, obs_col, wrf_col, r.target_col, r.unit))
        report.append({
            "var": r.var,
            "saved_height_m": saved_h,
            "aligned_height_m": aligned_h,
            "mapping_mode": mapping_mode,
            "saved_obs_col": r.obs_col,
            "aligned_obs_col": obs_col,
            "saved_wrf_col": r.wrf_col,
            "aligned_wrf_col": wrf_col,
            "obs_valid": int(numeric(out[obs_col]).notna().sum()),
            "wrf_valid": int(numeric(out[wrf_col]).notna().sum()),
            "uses_interpolated_target_observation": False,
        })
    return out, aligned, report


# =============================================================================
# 7. WINDOWING AND TIME MODES
# =============================================================================

def prepare_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c in out.columns:
            out[c] = numeric(out[c])
    if INTERPOLATE_FEATURES and feature_cols:
        out[list(feature_cols)] = out[list(feature_cols)].interpolate(method="linear", limit_direction="both").ffill().bfill()
    return out


def make_windows(df: pd.DataFrame, feature_cols: Sequence[str], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    x = df[list(feature_cols)].astype(float).to_numpy()
    wins, idx = [], []
    for i in range(window_size - 1, len(df)):
        w = x[i - window_size + 1:i + 1]
        if np.all(np.isfinite(w)):
            wins.append(w)
            idx.append(i)
    if not wins:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0,), dtype=int)
    return np.asarray(wins, dtype=float), np.asarray(idx, dtype=int)


def endpoint_selector(
    df: pd.DataFrame,
    idx: np.ndarray,
    task: GeneralizationTask,
    artifact: ModelArtifact,
    records: Sequence[TargetRecord],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    endpoint_times = pd.to_datetime(df.loc[idx, TIME_COL], errors="coerce")
    report: Dict[str, Any] = {
        "time_mode": task.time_mode,
        "n_candidate_windows": int(len(idx)),
    }

    if task.time_mode == "source_test_common_time":
        allowed = source_times(artifact, "test")
        if allowed is None:
            keep = np.flatnonzero(
                endpoint_times.notna().to_numpy(dtype=bool)
            ).astype(int)
            report.update({
                "status": (
                    "source_test_times_unavailable; using all finite-feature "
                    "target endpoints"
                ),
                "n_kept": int(len(keep)),
            })
            return keep, report
        allowed_set = set(allowed.asi8.tolist())
        keep = np.asarray([
            k for k, t in enumerate(endpoint_times)
            if pd.notna(t) and int(t.value) in allowed_set
        ], dtype=int)
        report.update({
            "status": "source_test_intersection",
            "n_source_test": int(len(allowed)),
            "n_kept": int(len(keep)),
        })
        return keep, report

    if task.time_mode == "target_all_valid":
        valid = endpoint_times.notna().to_numpy(dtype=bool)
        n_invalid_time = int((~valid).sum())

        n_invalid_qc = 0
        if OBS_VALID_COL in df.columns:
            qc = numeric(df.loc[idx, OBS_VALID_COL]).eq(1).to_numpy(dtype=bool)
            n_invalid_qc = int((valid & ~qc).sum())
            valid &= qc

        n_invalid_ws = 0
        if TARGET_ALL_TIME_REQUIRE_VALID_WS:
            valid_ws = np.zeros(len(idx), dtype=bool)
            ws_record_count = 0
            for r in records:
                if r.var != "ws":
                    continue
                if r.obs_col not in df.columns or r.wrf_col not in df.columns:
                    continue
                ws_record_count += 1
                obs = numeric(df.loc[idx, r.obs_col]).to_numpy(dtype=float)
                wrf = numeric(df.loc[idx, r.wrf_col]).to_numpy(dtype=float)
                valid_ws |= (
                    np.isfinite(obs) & np.isfinite(wrf)
                    & (obs >= 0.0) & (wrf >= 0.0)
                )
            if ws_record_count == 0:
                raise ValueError(
                    "No aligned WS records are available for target-all-time QC."
                )
            n_invalid_ws = int((valid & ~valid_ws).sum())
            valid &= valid_ws

        keep = np.flatnonzero(valid).astype(int)
        report.update({
            "status": "all_valid_target_endpoints",
            "definition": (
                "All target-site endpoints after existing obs_valid_flag QC "
                "and at least one finite/non-negative WS observation/WRF pair; "
                "source-time overlap is intentionally retained."
            ),
            "n_invalid_time": n_invalid_time,
            "n_removed_by_obs_valid_flag": n_invalid_qc,
            "n_removed_by_target_ws_validity": n_invalid_ws,
            "n_kept": int(len(keep)),
        })
        return keep, report

    if task.time_mode == "target_nonoverlap_with_source_development":
        source_all = source_times(artifact, None)
        if source_all is None:
            keep = np.flatnonzero(
                endpoint_times.notna().to_numpy(dtype=bool)
            ).astype(int)
            report.update({
                "status": (
                    "source_development_times_unavailable; "
                    "target assumed different-time"
                ),
                "n_kept": int(len(keep)),
            })
            return keep, report
        source_set = set(source_all.asi8.tolist())
        keep = np.asarray([
            k for k, t in enumerate(endpoint_times)
            if pd.notna(t) and int(t.value) not in source_set
        ], dtype=int)
        overlap = int(sum(
            pd.notna(t) and int(t.value) in source_set
            for t in endpoint_times
        ))
        report.update({
            "status": "nonoverlap_filter",
            "n_source_development": int(len(source_all)),
            "n_overlap_removed": overlap,
            "n_kept": int(len(keep)),
        })
        return keep, report

    keep = np.flatnonzero(
        endpoint_times.notna().to_numpy(dtype=bool)
    ).astype(int)
    report.update({"status": "all_target_endpoints", "n_kept": int(len(keep))})
    return keep, report

# =============================================================================
# 8. RECONSTRUCTION AND METRICS
# =============================================================================

def is_wd_sincos_record(r: TargetRecord) -> bool:
    s = r.target_col.lower()
    return r.var == "wd" and ("_sin_" in s or "_cos_" in s)


def wd_component(r: TargetRecord) -> Optional[str]:
    s = r.target_col.lower()
    return "sin" if "_sin_" in s else "cos" if "_cos_" in s else None


def wd_pair_indices(records: Sequence[TargetRecord], h: int, obs_col: str, wrf_col: str) -> Tuple[Optional[int], Optional[int]]:
    si = ci = None
    for j, r in enumerate(records):
        if int(r.height) == int(h) and r.obs_col == obs_col and r.wrf_col == wrf_col:
            if wd_component(r) == "sin":
                si = j
            elif wd_component(r) == "cos":
                ci = j
    return si, ci


def physical_arrays(df: pd.DataFrame, idx: np.ndarray, records: Sequence[TargetRecord], delta: np.ndarray) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    handled = set()
    for j, r in enumerate(records):
        if is_wd_sincos_record(r):
            key = (r.height, r.obs_col, r.wrf_col)
            if key in handled:
                continue
            si, ci = wd_pair_indices(records, r.height, r.obs_col, r.wrf_col)
            if si is None or ci is None:
                continue
            obs = numeric(df.loc[idx, r.obs_col]).to_numpy(dtype=float)
            base = numeric(df.loc[idx, r.wrf_col]).to_numpy(dtype=float)
            rad = np.deg2rad(base)
            sc = np.sin(rad) + delta[:, si]
            cc = np.cos(rad) + delta[:, ci]
            mag = np.sqrt(sc ** 2 + cc ** 2)
            pred = np.full_like(base, np.nan)
            ok = np.isfinite(mag) & (mag > 1e-12)
            pred[ok] = np.degrees(np.arctan2(sc[ok] / mag[ok], cc[ok] / mag[ok])) % 360.0
            data.append({"record": r, "variable": "WD", "obs": obs, "wrf": base, "pred": pred})
            handled.add(key)
            continue
        obs = numeric(df.loc[idx, r.obs_col]).to_numpy(dtype=float)
        base = numeric(df.loc[idx, r.wrf_col]).to_numpy(dtype=float)
        if r.var == "ws":
            pred = np.maximum(base + delta[:, j], 0.0); variable = "WS"
        elif r.var == "wd":
            pred = (base + delta[:, j]) % 360.0; variable = "WD"
        elif r.var == "log_tke":
            pred = np.maximum(np.expm1(np.log1p(np.maximum(base, 0.0)) + delta[:, j]), 0.0); variable = "TKE"
        elif r.var == "temp":
            pred = base + delta[:, j]; variable = "T"
        else:
            pred = base + delta[:, j]; variable = r.var.upper()
        data.append({"record": r, "variable": variable, "obs": obs, "wrf": base, "pred": pred})
    return data


def scalar_metrics(obs: np.ndarray, pred: np.ndarray, wrf: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wrf)
    if m.sum() < 2:
        return {"N": int(m.sum()), "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan, "Pearson_r": np.nan, "Raw_RMSE": np.nan, "Skill_RMSE_pct": np.nan}
    o, p, b = obs[m], pred[m], wrf[m]
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    raw = float(np.sqrt(np.mean((b - o) ** 2)))
    return {
        "N": int(m.sum()), "RMSE": rmse, "MAE": float(np.mean(np.abs(p - o))),
        "MBE": float(np.mean(p - o)),
        "Pearson_r": float(np.corrcoef(o, p)[0, 1]) if np.std(o) > 0 and np.std(p) > 0 else np.nan,
        "Raw_RMSE": raw,
        "Skill_RMSE_pct": float((raw - rmse) / raw * 100.0) if raw > 1e-12 else np.nan,
    }


def direction_metrics(obs: np.ndarray, pred: np.ndarray, wrf: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wrf)
    if m.sum() < 2:
        return {"N": int(m.sum()), "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan, "Pearson_r": np.nan, "Raw_RMSE": np.nan, "Skill_RMSE_pct": np.nan}
    e = ((pred[m] - obs[m] + 180.0) % 360.0) - 180.0
    eb = ((wrf[m] - obs[m] + 180.0) % 360.0) - 180.0
    rmse = float(np.sqrt(np.mean(e ** 2)))
    raw = float(np.sqrt(np.mean(eb ** 2)))
    return {
        "N": int(m.sum()), "RMSE": rmse, "MAE": float(np.mean(np.abs(e))), "MBE": float(np.mean(e)),
        "Pearson_r": np.nan, "Raw_RMSE": raw,
        "Skill_RMSE_pct": float((raw - rmse) / raw * 100.0) if raw > 1e-12 else np.nan,
    }


def build_outputs(
    df: pd.DataFrame,
    idx: np.ndarray,
    records: Sequence[TargetRecord],
    delta: np.ndarray,
    artifact: ModelArtifact,
    task: GeneralizationTask,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    items = physical_arrays(df, idx, records, delta)
    pred = pd.DataFrame({TIME_COL: df.loc[idx, TIME_COL].to_numpy()})
    pred["source_site"] = artifact.source_site
    pred["target_site"] = task.target_site
    pred["network"] = artifact.network
    pred["case"] = artifact.case
    pred["generalization_type"] = task.generalization_type
    pred["task_name"] = task.name

    rows: List[Dict[str, Any]] = []
    pooled: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for item in items:
        r: TargetRecord = item["record"]
        var = item["variable"]
        h = int(r.height)
        prefix = f"{var}{h}"
        pred[f"OBS_{prefix}"] = item["obs"]
        pred[f"WRF_{prefix}"] = item["wrf"]
        pred[f"CORR_{prefix}"] = item["pred"]
        row = {
            "source_site": artifact.source_site, "target_site": task.target_site,
            "network": artifact.network, "case": artifact.case,
            "generalization_type": task.generalization_type, "task_name": task.name,
            "variable": var, "height_m": h,
        }
        row.update(direction_metrics(item["obs"], item["pred"], item["wrf"]) if var == "WD" else scalar_metrics(item["obs"], item["pred"], item["wrf"]))
        rows.append(row)
        pooled.setdefault(var, {"obs": [], "pred": [], "wrf": []})
        pooled[var]["obs"].append(item["obs"])
        pooled[var]["pred"].append(item["pred"])
        pooled[var]["wrf"].append(item["wrf"])

    for var, arrays in pooled.items():
        o = np.concatenate(arrays["obs"]); p = np.concatenate(arrays["pred"]); b = np.concatenate(arrays["wrf"])
        row = {
            "source_site": artifact.source_site, "target_site": task.target_site,
            "network": artifact.network, "case": artifact.case,
            "generalization_type": task.generalization_type, "task_name": task.name,
            "variable": var, "height_m": "ALL_PROFILE",
        }
        row.update(direction_metrics(o, p, b) if var == "WD" else scalar_metrics(o, p, b))
        rows.append(row)
    return pred, pd.DataFrame(rows)

# =============================================================================
# 9. TASKS AND PLOTTING
# =============================================================================

def generate_tasks() -> List[GeneralizationTask]:
    tasks: List[GeneralizationTask] = []
    if RUN_CROSS_TOWER_SAME_TIME:
        tasks.extend([
            GeneralizationTask("C039801_to_C039802_same_time", "cross_tower_same_time", "C039801", "C039802", "source_test_common_time"),
            GeneralizationTask("C039802_to_C039801_same_time", "cross_tower_same_time", "C039802", "C039801", "source_test_common_time"),
        ])
    if RUN_CROSS_TOWER_OTHER_TIME:
        tasks.extend([
            GeneralizationTask("C039801_to_C039802_all_target_time", "cross_tower_other_time", "C039801", "C039802", "target_all_valid"),
            GeneralizationTask("C039802_to_C039801_all_target_time", "cross_tower_other_time", "C039802", "C039801", "target_all_valid"),
        ])
    return tasks



def plot_mean_ws_profile(pred: pd.DataFrame, out_png: Path, title: str) -> None:
    if plt is None or pred.empty:
        return
    rows = []
    for c in pred.columns:
        m = re.fullmatch(r"OBS_WS(\d+)", c)
        if not m:
            continue
        h = int(m.group(1))
        wc, pc = f"WRF_WS{h}", f"CORR_WS{h}"
        if wc in pred.columns and pc in pred.columns:
            rows.append((h, numeric(pred[c]).mean(), numeric(pred[wc]).mean(), numeric(pred[pc]).mean()))
    if len(rows) < 1:
        return
    d = pd.DataFrame(rows, columns=["height", "OBS", "WRF", "CORR"]).sort_values("height")
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    ax.plot(d["OBS"], d["height"], marker="o", label="OBS")
    ax.plot(d["WRF"], d["height"], marker="s", linestyle="--", label="WRF")
    ax.plot(d["CORR"], d["height"], marker="^", label="Corrected")
    ax.set_xlabel("Mean WS (m s$^{-1}$)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ws_timeseries(pred: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    if plt is None or pred.empty:
        return
    heights = []
    for c in pred.columns:
        m = re.fullmatch(r"OBS_WS(\d+)", c)
        if m:
            heights.append(int(m.group(1)))
    for h in sorted(set(heights)):
        cols = [f"OBS_WS{h}", f"WRF_WS{h}", f"CORR_WS{h}"]
        if not all(c in pred.columns for c in cols):
            continue
        d = pred[[TIME_COL] + cols].dropna().copy()
        if d.empty:
            continue
        if len(d) > MAX_TIMESERIES_POINTS:
            d = d.iloc[::int(math.ceil(len(d) / MAX_TIMESERIES_POINTS))]
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(d[TIME_COL], d[cols[0]], label="OBS", linewidth=1.2)
        ax.plot(d[TIME_COL], d[cols[1]], label="WRF", linestyle="--", linewidth=1.0)
        ax.plot(d[TIME_COL], d[cols[2]], label="Corrected", linewidth=1.0)
        ax.set_xlabel("Time"); ax.set_ylabel(f"WS at {h} m (m s$^{{-1}}$)")
        ax.set_title(f"{title_prefix} | {h} m")
        ax.grid(alpha=0.25); ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"timeseries_WS_h{h}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

# =============================================================================
# 10. APPLY ONE MODEL/TASK
# =============================================================================

def apply_model_task(
    artifact: ModelArtifact,
    task: GeneralizationTask,
    target_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    meta = read_json(artifact.metadata_path)
    feature_cols = [str(x) for x in meta.get("feature_cols", [])]
    records_saved = records_from_metadata(meta)
    heights = sorted(set(int(r.height) for r in records_saved))
    window_size = int(meta.get("window_size", 24))

    df, gryning_report = inject_gryning(target_raw, task.target_site)
    df, vertical_interpolation_report = ensure_model_height_support(
        df, heights, feature_cols, records_saved
    )
    df = derive_common_wrf_features(df, heights)
    df, records, alignment_report = align_records(df, records_saved, task=task)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features and ALLOW_EXISTING_MOST_COLUMN_FALLBACK:
        # Injection may have failed, but source ML-ready columns may already contain MOST fields.
        missing_features = [c for c in feature_cols if c not in df.columns]
    missing_wrf = [r.wrf_col for r in records if r.wrf_col not in df.columns or numeric(df[r.wrf_col]).notna().sum() == 0]

    audit: Dict[str, Any] = {
        "artifact": str(artifact.model_dir), "source_site": artifact.source_site,
        "target_site": task.target_site, "network": artifact.network, "case": artifact.case,
        "task": task.__dict__, "gryning": gryning_report,
        "vertical_interpolation": vertical_interpolation_report,
        "MOST_reference_interpretation": MOST_REFERENCE_INTERPRETATION,
        "wd_head_target_native_remap": REMAP_WD_HEAD_TO_TARGET_NATIVE_HEIGHT,
        "wd_head_target_native_remap_mode": "only_when_saved_WD_height_is_not_evaluable",
        "target_native_wd_heights_m": {
            site: int(cfg["native_wd_m"])
            for site, cfg in SITE_CONFIG.items()
        },
        "n_feature_cols": len(feature_cols), "missing_features": missing_features,
        "missing_or_empty_wrf_cols": missing_wrf,
        "record_alignment": alignment_report,
    }
    if missing_features or missing_wrf:
        raise ValueError(f"Missing features={missing_features[:30]}, missing WRF={missing_wrf[:30]}")

    df = prepare_features(df, feature_cols)
    X, idx = make_windows(df, feature_cols, window_size)
    keep, time_report = endpoint_selector(
        df, idx, task, artifact, records
    )
    X, idx = X[keep], idx[keep]
    audit["time_selection"] = time_report
    audit["n_application_windows"] = int(len(X))
    if len(X) == 0:
        raise ValueError("No valid target windows after feature and time filtering.")
    audit["time_start"] = str(df.loc[idx, TIME_COL].min())
    audit["time_end"] = str(df.loc[idx, TIME_COL].max())

    if AUDIT_ONLY:
        return pd.DataFrame(), pd.DataFrame(), audit

    x_scaler = joblib.load(artifact.x_scaler_path)
    Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    model = load_keras_model(artifact.model_path)
    pred_scaled = np.asarray(model.predict(Xs, verbose=0))
    y_mean = np.asarray(meta.get("y_mean", []), dtype=float)
    y_scale = np.asarray(meta.get("y_scale", []), dtype=float)
    if pred_scaled.shape[1] != len(records) or len(y_mean) != len(records) or len(y_scale) != len(records):
        raise ValueError(
            f"Output/metadata mismatch: model={pred_scaled.shape}, records={len(records)}, y_mean={len(y_mean)}, y_scale={len(y_scale)}"
        )
    delta = pred_scaled * y_scale[None, :] + y_mean[None, :]
    pred, metrics = build_outputs(df, idx, records, delta, artifact, task)
    return pred, metrics, audit

# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    audit_dir = ensure_dir(OUTPUT_ROOT / "00_audit")
    artifacts = discover_code2_models()
    if not artifacts:
        raise RuntimeError(f"No Code2 model artifacts found under {CODE2_RESULT_ROOT}")

    target_paths = {site: discover_ml_ready_optional(site) for site in TARGET_SITES}
    target_data: Dict[str, pd.DataFrame] = {}
    for site, path in target_paths.items():
        if path is not None:
            target_data[site] = load_ml_ready(path)

    pd.DataFrame([{
        "source_site": a.source_site, "network": a.network, "case": a.case,
        "model_dir": str(a.model_dir), "prediction": str(a.prediction_path or ""),
    } for a in artifacts]).to_csv(audit_dir / "model_discovery.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "target_site": s, "ml_ready_csv": str(p or ""), "exists": bool(p and p.exists()),
        "note": "Time-resolved ML-ready target data used for same-time and all-valid-target-time cross-tower evaluation."
    } for s, p in target_paths.items()]).to_csv(audit_dir / "target_data_discovery.csv", index=False, encoding="utf-8-sig")

    tasks = generate_tasks()
    with open(audit_dir / "generalization_tasks.json", "w", encoding="utf-8") as f:
        json.dump([t.__dict__ for t in tasks], f, ensure_ascii=False, indent=2)

    all_pred: List[pd.DataFrame] = []
    all_metrics: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    artifact_map = {(a.source_site, a.network, a.case): a for a in artifacts}
    for task in tasks:
        if task.target_site not in target_data:
            for network in NETWORKS_TO_APPLY:
                for case in CASES_TO_APPLY:
                    if (task.source_site, network, case) in artifact_map:
                        failures.append({
                            "source_site": task.source_site, "target_site": task.target_site,
                            "network": network, "case": case, "task": task.name,
                            "error": "Target ML-ready time-series CSV not found for cross-tower generalization.",
                        })
            continue
        for network in NETWORKS_TO_APPLY:
            for case in CASES_TO_APPLY:
                artifact = artifact_map.get((task.source_site, network, case))
                if artifact is None:
                    failures.append({
                        "source_site": task.source_site, "target_site": task.target_site,
                        "network": network, "case": case, "task": task.name,
                        "error": "Source model artifact not discovered",
                    })
                    continue
                print("=" * 110)
                print(f"Code2 generalization | {task.name} | {network} | {case}")
                try:
                    pred, metrics, audit = apply_model_task(artifact, task, target_data[task.target_site].copy())
                    audit["status"] = "PASS"
                    audit_rows.append(audit)
                    if not pred.empty:
                        out_dir = ensure_dir(
                            OUTPUT_ROOT / task.generalization_type / f"{task.source_site}_to_{task.target_site}" / network / case
                        )
                        pred.to_csv(out_dir / "predictions_generalization.csv", index=False, encoding="utf-8-sig")
                        metrics.to_csv(out_dir / "metrics_generalization.csv", index=False, encoding="utf-8-sig")
                        with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
                            json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
                        if SAVE_PLOTS:
                            plot_mean_ws_profile(pred, out_dir / "mean_WS_profile.png", f"{task.name} | {network} | {case}")
                            plot_ws_timeseries(pred, out_dir / "timeseries_WS", f"{task.name} | {network} | {case}")
                        all_pred.append(pred)
                        all_metrics.append(metrics)
                except Exception as exc:
                    failures.append({
                        "source_site": task.source_site, "target_site": task.target_site,
                        "network": network, "case": case, "task": task.name,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    })
                    print(f"[FAILED] {exc}")
                    if STOP_ON_ERROR:
                        raise

    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_csv(
            OUTPUT_ROOT / "ALL_code2_generalization_predictions.csv", index=False, encoding="utf-8-sig"
        )
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(OUTPUT_ROOT / "ALL_code2_generalization_metrics.csv", index=False, encoding="utf-8-sig")
        all_profile = metrics_all[metrics_all["height_m"].astype(str).eq("ALL_PROFILE")]
        all_profile.to_csv(OUTPUT_ROOT / "ALL_code2_ALL_PROFILE_generalization_comparison.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(audit_rows).to_json(audit_dir / "application_audit.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(failures).to_csv(audit_dir / "failures.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_models_discovered": len(artifacts), "n_tasks": len(tasks),
        "n_success": len(audit_rows), "n_failed": len(failures), "audit_only": AUDIT_ONLY,
        "scope": "Cross-tower same-time plus cross-tower all-valid-target-time generalization in both directions.",
        "other_time_definition": "All valid target-site times; source-period overlap is retained by design.",
        "MOST_reference_interpretation": MOST_REFERENCE_INTERPRETATION,
    }
    with open(audit_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=" * 110)
    print("Finished Code2 generalization")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
