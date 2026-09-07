#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huarui_A Code1 generalization application
==========================================

Applies the saved 160-m Huarui_A Code1 models without retraining.

Stations and WRF domains
------------------------
- C039801 / A1: preferred ML-ready file from d03.
- C039802 / A2: preferred ML-ready file from d02.

The saved Code1 models use canonical internal 10-m names, but their physical
training bundle is WS/WD/TKE at 160 m. For height-transfer tasks, only the
WS/U/V/MOST height is changed; WD, TKE and temperature use the target tower's
160-m physical bundle. This creates a height-wise WS pseudo-profile.

Formal tasks
------------
1. same_tower_other_height
2. cross_tower_same_height
3. cross_tower_other_height

4. cross_tower_other_time: all valid target-tower endpoints; timestamps are not
   removed merely because they overlap the source development period.

The script recursively discovers saved C2/C4/C6/C8 CNN_LSTM and TCN artifacts,
injects the same Gryning_fixed_z0_vegetation500 reference used in training, and
exports predictions, metrics, audits, time series and WS pseudo-profiles.
"""
from __future__ import annotations

import json
import math
import pickle
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

TIME_RESOLUTION_MODE = "1h"  # "1h", "10min", or "original"
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
CODE1_RESULT_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\correction\c1_new_v5_I"
)
OUTPUT_ROOT = Path(
    r"I:\wake\WRF_RANS_wake\huarui\A\bias_correction\1km\correction\c1_generalization_huarui_A_same_and_all_target_time"
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
MOST_REFERENCE_INTERPRETATION = (
    "OBS-assisted diagnostic Gryning_fixed_z0_vegetation500; "
    "training-consistent but not strict deployment"
)

TIME_COL = "北京时间"
OBS_VALID_COL = "obs_valid_flag"
CASES_TO_APPLY = ["C2", "C4", "C6", "C8"]
NETWORKS_TO_APPLY = ["CNN_LSTM", "TCN"]
SOURCE_SITES = ["C039801", "C039802"]
TARGET_SITES = ["C039801", "C039802"]

USE_SOURCE_CASE_TEST_ENDPOINTS = True
INTERPOLATE_FEATURES = True
AUDIT_ONLY = False
STOP_ON_ERROR = False
SAVE_PLOTS = True
PLOT_DPI = 300
MAX_TIMESERIES_POINTS = 2500

RUN_SAME_TOWER_OTHER_HEIGHT = True
RUN_CROSS_TOWER_SAME_HEIGHT = True
RUN_CROSS_TOWER_OTHER_HEIGHT = True

RUN_CROSS_TOWER_OTHER_TIME = True
TARGET_ALL_TIME_REQUIRE_VALID_WS = True

SITE_CONFIG: Dict[str, Dict[str, Any]] = {
    "C039801": {
        "aliases": ["C039801", "A1"],
        "preferred_domain": "d03",
        "source_train_ws_m": 160,
        "native_wd_m": 160,
        "native_tke_m": 160,
        "native_temp_m": 160,
    },
    "C039802": {
        "aliases": ["C039802", "A2"],
        "preferred_domain": "d02",
        "source_train_ws_m": 160,
        "native_wd_m": 160,
        "native_tke_m": 160,
        "native_temp_m": 160,
    },
}

CASE_ALIASES = {
    "C2": ["C2", "noMOST_corr"],
    "C4": ["C4", "MOSTinput_corr"],
    "C6": ["C6", "PG_MOSTloss_corr"],
    "C8": ["C8", "MOSTinput_PG_MOSTloss_corr"],
}

# =============================================================================
# 2. DATA CLASSES AND PICKLE COMPATIBILITY
# =============================================================================

@dataclass
class TargetSpec:
    name: str
    kind: str
    obs_col: str
    wrf_col: str
    target_col: str
    unit: str


@dataclass
class ModelArtifact:
    source_site: str
    network: str
    case: str
    scenario: str
    case_dir: Path
    metadata_path: Path
    model_path: Path
    x_scaler_path: Path
    y_scaler_path: Path
    features_path: Path
    prediction_path: Optional[Path]


@dataclass
class GeneralizationTask:
    name: str
    generalization_type: str
    source_site: str
    target_site: str
    time_mode: str
    apply_ws_height_m: int
    apply_wd_height_m: int
    apply_tke_height_m: int
    apply_temp_height_m: int


class MaskedYScaler:
    """Local replacement for the pickle class saved by Code1."""
    def __init__(self, mean_: np.ndarray, scale_: np.ndarray):
        self.mean_ = np.asarray(mean_, dtype=float)
        self.scale_ = np.asarray(scale_, dtype=float)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-12)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean_[None, :]) / self.scale_[None, :]

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled, dtype=float) * self.scale_[None, :] + self.mean_[None, :]


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "MaskedYScaler":
            return MaskedYScaler
        return super().find_class(module, name)


def load_pickle_compat(path: Path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()

# =============================================================================
# 3. GENERIC UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: Any, max_len: int = 160) -> str:
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



def normalize_network(path_or_value: Any) -> Optional[str]:
    n = norm(path_or_value)
    if "cnnlstm" in n or re.search(r"(^|[^a-z])cl([^a-z]|$)", str(path_or_value).lower()):
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


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    mapping = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in mapping:
            return mapping[norm(c)]
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

def import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:
        raise ImportError("TensorFlow is required to load the saved Code1 models.") from exc


def load_keras_model(path: Path):
    tf = import_tf()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIME_COL in out.columns:
        t = pd.to_datetime(out[TIME_COL], errors="coerce")
        hour = t.dt.hour.astype(float) + t.dt.minute.astype(float) / 60.0
        if "hour_sin" not in out.columns:
            out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_cos" not in out.columns:
            out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return out


def load_ml_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing time column {TIME_COL}: {path}")
    df[TIME_COL] = parse_datetime_mixed_safe(df[TIME_COL], TIME_COL)
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).drop_duplicates(TIME_COL).reset_index(drop=True)
    return add_time_features(df)


# =============================================================================
# 4. FILE DISCOVERY
# =============================================================================

def discover_ml_ready(site: str) -> Path:
    candidates = list(ML_READY_ROOT.rglob("ML_ready_ref10_*.csv"))
    cfg = SITE_CONFIG[site]
    hits = [p for p in candidates if path_contains_alias(p, cfg["aliases"])]
    if not hits:
        raise FileNotFoundError(f"No ML_ready_ref10 CSV found for {site} under {ML_READY_ROOT}")
    preferred = str(cfg.get("preferred_domain", "")).lower()
    hits.sort(key=lambda p: (
        0 if preferred and preferred in p.name.lower() else 1,
        len(p.parts), len(str(p)),
    ))
    return hits[0]



def infer_site_from_path(path: Path) -> Optional[str]:
    for site, cfg in SITE_CONFIG.items():
        if path_contains_alias(path, cfg["aliases"]):
            return site
    return None


def discover_code1_models() -> List[ModelArtifact]:
    artifacts: List[ModelArtifact] = []
    for meta_path in CODE1_RESULT_ROOT.rglob("metadata_*.json"):
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        scenario = str(meta.get("scenario", meta_path.stem.replace("metadata_", "")))
        case = normalize_case(scenario) or normalize_case(meta_path.parent.name)
        if case not in CASES_TO_APPLY:
            continue
        source_site = infer_site_from_path(meta_path)
        network = normalize_network(meta.get("network", "")) or normalize_network(meta_path)
        if source_site not in SOURCE_SITES or network not in NETWORKS_TO_APPLY:
            continue
        case_dir = meta_path.parent
        model_path = case_dir / str(meta.get("model_file", f"model_{scenario}.keras"))
        x_scaler_path = case_dir / str(meta.get("x_scaler_file", f"x_scaler_{scenario}.pkl"))
        y_scaler_path = case_dir / str(meta.get("y_scaler_file", f"y_scaler_{scenario}.pkl"))
        features_path = case_dir / str(meta.get("features_file", f"features_{scenario}.csv"))
        prediction_path = case_dir / f"predictions_{scenario}_supervised_multiout.csv"
        if not prediction_path.exists():
            matches = list(case_dir.glob("predictions_*_supervised_multiout.csv"))
            prediction_path = matches[0] if matches else None
        required = [model_path, x_scaler_path, y_scaler_path, features_path]
        if not all(p.exists() for p in required):
            continue
        artifacts.append(ModelArtifact(
            source_site=source_site,
            network=network,
            case=case,
            scenario=scenario,
            case_dir=case_dir,
            metadata_path=meta_path,
            model_path=model_path,
            x_scaler_path=x_scaler_path,
            y_scaler_path=y_scaler_path,
            features_path=features_path,
            prediction_path=prediction_path,
        ))
    # Deduplicate; prefer paths under core/01_cases.
    best: Dict[Tuple[str, str, str], ModelArtifact] = {}
    for art in artifacts:
        key = (art.source_site, art.network, art.case)
        score = (100 if "core" in [x.lower() for x in art.case_dir.parts] else 0) + (50 if "01_cases" in art.case_dir.parts else 0) - len(art.case_dir.parts)
        old = best.get(key)
        if old is None:
            best[key] = art
        else:
            old_score = (100 if "core" in [x.lower() for x in old.case_dir.parts] else 0) + (50 if "01_cases" in old.case_dir.parts else 0) - len(old.case_dir.parts)
            if score > old_score:
                best[key] = art
    return sorted(best.values(), key=lambda a: (a.source_site, a.network, a.case))


def source_test_endpoints(artifact: ModelArtifact) -> Optional[pd.DatetimeIndex]:
    if not USE_SOURCE_CASE_TEST_ENDPOINTS or artifact.prediction_path is None or not artifact.prediction_path.exists():
        return None
    try:
        d = pd.read_csv(artifact.prediction_path, usecols=lambda c: c in {TIME_COL, "split"})
        d[TIME_COL] = parse_datetime_mixed_safe(d[TIME_COL], TIME_COL)
        d = d[d["split"].astype(str).str.lower().eq("test")].dropna(subset=[TIME_COL])
        return pd.DatetimeIndex(d[TIME_COL].drop_duplicates().sort_values())
    except Exception:
        return None



# =============================================================================
# 4A. TRAINING-CONSISTENT GRYNING INJECTION
# =============================================================================

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
# 5. HEIGHT AND COLUMN MAPPING
# =============================================================================

def obs_ws_candidates(h: int) -> List[str]:
    return [f"OBS_WS{h} (m/s)", f"OBS_WS_{h}m (m/s)", f"OBS_WS{h}", f"OBS_WS_{h}m"]


def obs_wd_candidates(h: int) -> List[str]:
    return [f"OBS_WD{h} (deg)", f"OBS_WD_{h}m (deg)", f"OBS_WD{h}", f"OBS_WD_{h}m"]


def obs_tke_candidates(h: int) -> List[str]:
    return [f"OBS_TKE{h} (m2/s2)", f"OBS_TKE{h}_from_TI (m2/s2)", f"OBS_TKE_{h}m (m2/s2)", f"OBS_TKE{h}"]


def available_obs_ws_heights(df: pd.DataFrame) -> List[int]:
    hs = set()
    for c in df.columns:
        m = re.match(r"^OBS_WS_?(\d+)(?:m)?(?:\s*\(m/s\))?$", str(c).replace(" ", ""), flags=re.I)
        if m and numeric(df[c]).notna().sum() > 0:
            hs.add(int(m.group(1)))
        m2 = re.match(r"^OBS_WS(\d+)\s*\(m/s\)$", str(c), flags=re.I)
        if m2 and numeric(df[c]).notna().sum() > 0:
            hs.add(int(m2.group(1)))
    # More robust candidate-based scan.
    for h in range(1, 401):
        c = first_existing(df, obs_ws_candidates(h))
        if c is not None and numeric(df[c]).notna().sum() > 0:
            hs.add(h)
    return sorted(hs)


def alias_target_bundle(df: pd.DataFrame, task: GeneralizationTask) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    h_ws = int(task.apply_ws_height_m)
    h_wd = int(task.apply_wd_height_m)
    h_tke = int(task.apply_tke_height_m)
    h_t = int(task.apply_temp_height_m)
    mapping: Dict[str, str] = {}

    def alias(candidates: Sequence[str], dst: str, required: bool = False):
        src = first_existing(out, candidates)
        if src is not None:
            out[dst] = out[src]
            mapping[dst] = src
        elif required:
            mapping[dst] = "MISSING"

    # OBS target columns.
    alias(obs_ws_candidates(h_ws), "OBS_WS10 (m/s)")
    alias(obs_wd_candidates(h_wd), "OBS_WD10 (deg)")
    alias(obs_tke_candidates(h_tke), "OBS_TKE10 (m2/s2)")
    alias([f"OBS_T{h_t} (K)", f"OBS_T_{h_t}m (K)", f"OBS_TEMP{h_t} (K)", f"OBS_T{h_t}"], "OBS_T10 (K)")

    # WRF physical features and baselines.
    alias([f"WRF_WS{h_ws}_interp (m/s)", f"WRF_WS_{h_ws}m_interp (m/s)", f"WRF_WS{h_ws} (m/s)", f"WRF_WS{h_ws}"], "WRF_WS10_interp (m/s)", True)
    alias([f"WRF_WD{h_wd}_interp (deg)", f"WRF_WD_{h_wd}m_interp (deg)", f"WRF_WD{h_wd} (deg)", f"WRF_WD{h_wd}"], "WRF_WD10_interp (deg)", True)
    alias([f"WRF_U{h_ws}_interp (m/s)", f"WRF_U_{h_ws}m_interp (m/s)", f"WRF_U{h_ws} (m/s)", f"WRF_U{h_ws}"], "WRF_U10_interp (m/s)")
    alias([f"WRF_V{h_ws}_interp (m/s)", f"WRF_V_{h_ws}m_interp (m/s)", f"WRF_V{h_ws} (m/s)", f"WRF_V{h_ws}"], "WRF_V10_interp (m/s)")
    alias([f"WRF_WD{h_wd}_sin", f"WRF_WD_{h_wd}m_sin"], "WRF_WD10_sin")
    alias([f"WRF_WD{h_wd}_cos", f"WRF_WD_{h_wd}m_cos"], "WRF_WD10_cos")
    alias([f"WRF_TKE{h_tke} (m2/s2)", f"WRF_TKE_{h_tke}m (m2/s2)", f"WRF_TKE{h_tke}_interp (m2/s2)", f"WRF_TKE{h_tke}"], "WRF_TKE10 (m2/s2)", True)
    alias([f"WRF_T{h_t}_interp (K)", f"WRF_T_{h_t}m_interp (K)", f"WRF_TEMP{h_t}_interp (K)", f"WRF_T{h_t} (K)", f"WRF_T{h_t}"], "T2 (K)")

    # If U/V are absent, derive from WS/WD.
    if "WRF_U10_interp (m/s)" not in out.columns or "WRF_V10_interp (m/s)" not in out.columns:
        ws_col = first_existing(out, ["WRF_WS10_interp (m/s)"])
        wd_col = first_existing(out, ["WRF_WD10_interp (deg)"])
        if ws_col and wd_col:
            ws = numeric(out[ws_col]).to_numpy(dtype=float)
            wd = np.deg2rad(numeric(out[wd_col]).to_numpy(dtype=float))
            out["WRF_U10_interp (m/s)"] = -ws * np.sin(wd)
            out["WRF_V10_interp (m/s)"] = -ws * np.cos(wd)
            mapping["WRF_U10_interp (m/s)"] = "derived_from_WS_WD"
            mapping["WRF_V10_interp (m/s)"] = "derived_from_WS_WD"

    if "WRF_WD10_sin" not in out.columns or "WRF_WD10_cos" not in out.columns:
        wd_col = first_existing(out, ["WRF_WD10_interp (deg)"])
        if wd_col:
            rad = np.deg2rad(numeric(out[wd_col]).to_numpy(dtype=float))
            out["WRF_WD10_sin"] = np.sin(rad)
            out["WRF_WD10_cos"] = np.cos(rad)
            mapping["WRF_WD10_sin"] = "derived_from_WRF_WD"
            mapping["WRF_WD10_cos"] = "derived_from_WRF_WD"

    # MOST features follow WS height.
    alias([f"U_MO{h_ws} (m/s)", f"U_MO_{h_ws}m (m/s)", f"U_MO{h_ws}", f"MOST_WS{h_ws} (m/s)", f"MOST_WS{h_ws}"], "U_MO10 (m/s)")
    alias([f"MO_minus_WRF{h_ws} (m/s)", f"MO_minus_WRF_{h_ws}m (m/s)", f"MO_minus_WRF{h_ws}", f"MOST_minus_WRF{h_ws} (m/s)"], "MO_minus_WRF10 (m/s)")

    # Height-specific validity flag if exported.
    for fc in [f"obs_valid_flag_{h_ws}", f"OBS_valid_flag_{h_ws}", f"valid_flag_{h_ws}"]:
        if fc in out.columns:
            out[OBS_VALID_COL] = out[fc]
            mapping[OBS_VALID_COL] = fc
            break

    return out, mapping

# =============================================================================
# 6. WINDOWING, RECONSTRUCTION, AND METRICS
# =============================================================================


def select_task_endpoints(
    df: pd.DataFrame,
    idx: np.ndarray,
    task: GeneralizationTask,
    source_test_allowed: Optional[pd.DatetimeIndex],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select application endpoints according to the requested time definition.

    source_test_endpoints
        Restrict target application to the saved source model's test timestamps.
    target_all_valid
        Use every valid target endpoint. This is the requested "other-time"
        definition: all target times, not a strict non-overlap subset.
    """
    endpoint_times = pd.to_datetime(df.loc[idx, TIME_COL], errors="coerce")
    report: Dict[str, Any] = {
        "time_mode": task.time_mode,
        "n_candidate_windows": int(len(idx)),
    }

    if task.time_mode == "source_test_endpoints":
        keep = restrict_endpoints(df, idx, source_test_allowed)
        report.update({
            "status": (
                "source_test_intersection"
                if source_test_allowed is not None
                else "source_test_times_unavailable; using all finite-feature endpoints"
            ),
            "n_source_test_times": (
                int(len(source_test_allowed))
                if source_test_allowed is not None else 0
            ),
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
            obs_col = "OBS_WS10 (m/s)"
            wrf_col = "WRF_WS10_interp (m/s)"
            if obs_col not in df.columns or wrf_col not in df.columns:
                raise ValueError(
                    "Target-all-time filtering requires canonical WS observation "
                    f"and WRF columns, missing: "
                    f"{[c for c in [obs_col, wrf_col] if c not in df.columns]}"
                )
            obs = numeric(df.loc[idx, obs_col]).to_numpy(dtype=float)
            wrf = numeric(df.loc[idx, wrf_col]).to_numpy(dtype=float)
            valid_ws = (
                np.isfinite(obs) & np.isfinite(wrf)
                & (obs >= 0.0) & (wrf >= 0.0)
            )
            n_invalid_ws = int((valid & ~valid_ws).sum())
            valid &= valid_ws

        keep = np.flatnonzero(valid).astype(int)
        report.update({
            "status": "all_valid_target_endpoints",
            "definition": (
                "All target-site endpoints after existing obs_valid_flag QC "
                "and finite/non-negative target WS checks; source-time overlap "
                "is intentionally retained."
            ),
            "n_invalid_time": n_invalid_time,
            "n_removed_by_obs_valid_flag": n_invalid_qc,
            "n_removed_by_target_ws_validity": n_invalid_ws,
            "n_kept": int(len(keep)),
        })
        return keep, report

    keep = np.flatnonzero(endpoint_times.notna().to_numpy(dtype=bool)).astype(int)
    report.update({"status": "all_finite_time_endpoints", "n_kept": int(len(keep))})
    return keep, report


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
    windows, idx = [], []
    for i in range(window_size - 1, len(df)):
        w = x[i - window_size + 1:i + 1]
        if np.all(np.isfinite(w)):
            windows.append(w)
            idx.append(i)
    if not windows:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0,), dtype=int)
    return np.asarray(windows, dtype=float), np.asarray(idx, dtype=int)


def restrict_endpoints(df: pd.DataFrame, idx: np.ndarray, allowed: Optional[pd.DatetimeIndex]) -> np.ndarray:
    if allowed is None:
        return np.arange(len(idx), dtype=int)
    allowed_ns = set(pd.DatetimeIndex(allowed).asi8.tolist())
    t = pd.to_datetime(df.loc[idx, TIME_COL], errors="coerce")
    return np.asarray([k for k, ts in enumerate(t) if pd.notna(ts) and int(ts.value) in allowed_ns], dtype=int)


def specs_from_metadata(meta: Dict[str, Any]) -> List[TargetSpec]:
    return [TargetSpec(**x) for x in meta.get("target_specs", [])]


def collect_obs_wrf(df: pd.DataFrame, idx: np.ndarray, specs: Sequence[TargetSpec]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    obs, wrf = {}, {}
    for s in specs:
        obs[s.name] = numeric(df.loc[idx, s.obs_col]).to_numpy(dtype=float) if s.obs_col in df.columns else np.full(len(idx), np.nan)
        wrf[s.name] = numeric(df.loc[idx, s.wrf_col]).to_numpy(dtype=float) if s.wrf_col in df.columns else np.full(len(idx), np.nan)
    return obs, wrf


def find_sincos_indices(specs: Sequence[TargetSpec]) -> Tuple[Optional[int], Optional[int]]:
    names = [s.name for s in specs]
    return (
        names.index("WD10_sin_direct") if "WD10_sin_direct" in names else None,
        names.index("WD10_cos_direct") if "WD10_cos_direct" in names else None,
    )


def reconstruct(wrf: Dict[str, np.ndarray], delta: np.ndarray, specs: Sequence[TargetSpec], tke_eps: float) -> Dict[str, np.ndarray]:
    corr: Dict[str, np.ndarray] = {}
    for j, s in enumerate(specs):
        base = np.asarray(wrf[s.name], dtype=float)
        if s.kind == "log_tke":
            corr[s.name] = np.maximum(np.exp(np.log(np.maximum(base, 0.0) + tke_eps) + delta[:, j]) - tke_eps, 0.0)
        elif s.kind == "ws":
            corr[s.name] = np.maximum(base + delta[:, j], 0.0)
        elif s.kind == "wd_circular":
            corr[s.name] = (base + delta[:, j]) % 360.0
        else:
            corr[s.name] = base + delta[:, j]
    i_sin, i_cos = find_sincos_indices(specs)
    if i_sin is not None and i_cos is not None:
        wrf_wd = np.asarray(wrf[specs[i_sin].name], dtype=float)
        rad = np.deg2rad(wrf_wd)
        sin_corr = np.sin(rad) + delta[:, i_sin]
        cos_corr = np.cos(rad) + delta[:, i_cos]
        mag = np.sqrt(sin_corr ** 2 + cos_corr ** 2)
        good = np.isfinite(mag) & (mag > 1e-12)
        pred = np.full_like(wrf_wd, np.nan, dtype=float)
        pred[good] = np.degrees(np.arctan2(sin_corr[good] / mag[good], cos_corr[good] / mag[good])) % 360.0
        corr["WD10_direct"] = pred
    return corr


def scalar_metrics(obs: np.ndarray, pred: np.ndarray, wrf: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wrf)
    if m.sum() < 2:
        return {"N": int(m.sum()), "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan, "Pearson_r": np.nan, "Raw_RMSE": np.nan, "Skill_RMSE_pct": np.nan}
    o, p, b = obs[m], pred[m], wrf[m]
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    raw = float(np.sqrt(np.mean((b - o) ** 2)))
    return {
        "N": int(m.sum()),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(p - o))),
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


def build_physical_outputs(
    df: pd.DataFrame,
    idx: np.ndarray,
    specs: Sequence[TargetSpec],
    delta: np.ndarray,
    task: GeneralizationTask,
    artifact: ModelArtifact,
    tke_eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs, wrf = collect_obs_wrf(df, idx, specs)
    corr = reconstruct(wrf, delta, specs, tke_eps)
    out = pd.DataFrame({TIME_COL: df.loc[idx, TIME_COL].to_numpy()})
    out["source_site"] = artifact.source_site
    out["target_site"] = task.target_site
    out["network"] = artifact.network
    out["case"] = artifact.case
    out["scenario"] = artifact.scenario
    out["generalization_type"] = task.generalization_type
    out["task_name"] = task.name
    out["WS_height_m"] = task.apply_ws_height_m
    out["WD_height_m"] = task.apply_wd_height_m
    out["TKE_height_m"] = task.apply_tke_height_m

    metrics_rows: List[Dict[str, Any]] = []

    def add(variable: str, height: int, o: np.ndarray, b: np.ndarray, p: np.ndarray, direction: bool = False):
        out[f"OBS_{variable}"] = o
        out[f"WRF_{variable}"] = b
        out[f"CORR_{variable}"] = p
        row = {
            "source_site": artifact.source_site, "target_site": task.target_site,
            "network": artifact.network, "case": artifact.case, "scenario": artifact.scenario,
            "generalization_type": task.generalization_type, "task_name": task.name,
            "variable": variable, "height_m": height,
        }
        row.update(direction_metrics(o, p, b) if direction else scalar_metrics(o, p, b))
        metrics_rows.append(row)

    if "WS10_direct" in obs and "WS10_direct" in corr:
        add("WS", task.apply_ws_height_m, obs["WS10_direct"], wrf["WS10_direct"], corr["WS10_direct"])
    if "WD10_direct" in corr:
        if "WD10_sin_direct" in obs:
            add("WD", task.apply_wd_height_m, obs["WD10_sin_direct"], wrf["WD10_sin_direct"], corr["WD10_direct"], True)
        elif "WD10_direct" in obs:
            add("WD", task.apply_wd_height_m, obs["WD10_direct"], wrf["WD10_direct"], corr["WD10_direct"], True)
    if "logTKE10" in obs and "logTKE10" in corr:
        add("TKE", task.apply_tke_height_m, obs["logTKE10"], wrf["logTKE10"], corr["logTKE10"])
    if "T10" in obs and "T10" in corr:
        add("T", task.apply_temp_height_m, obs["T10"], wrf["T10"], corr["T10"])

    return out, pd.DataFrame(metrics_rows)

# =============================================================================
# 7. TASK GENERATION
# =============================================================================

def generate_tasks(source_site: str, target_data: Dict[str, pd.DataFrame]) -> List[GeneralizationTask]:
    tasks: List[GeneralizationTask] = []
    other_site = "C039802" if source_site == "C039801" else "C039801"

    def make(target_site: str, ws_h: int, gtype: str, time_mode: str = "source_test_endpoints") -> GeneralizationTask:
        cfg = SITE_CONFIG[target_site]
        kwargs = dict(
            name=f"{source_site}_to_{target_site}_{gtype}_WS{ws_h}",
            generalization_type=gtype,
            source_site=source_site,
            target_site=target_site,
            apply_ws_height_m=int(ws_h),
            apply_wd_height_m=int(cfg["native_wd_m"]),
            apply_tke_height_m=int(cfg["native_tke_m"]),
            apply_temp_height_m=int(cfg["native_temp_m"]),
        )
        if "time_mode" in GeneralizationTask.__dataclass_fields__:
            kwargs["time_mode"] = time_mode
        return GeneralizationTask(**kwargs)

    source_h = int(SITE_CONFIG[source_site]["source_train_ws_m"])
    if RUN_SAME_TOWER_OTHER_HEIGHT:
        for h in available_obs_ws_heights(target_data[source_site]):
            if h != source_h:
                tasks.append(make(source_site, h, "same_tower_other_height"))
    if RUN_CROSS_TOWER_SAME_HEIGHT and source_h in available_obs_ws_heights(target_data[other_site]):
        tasks.append(make(other_site, source_h, "cross_tower_same_height"))
    if RUN_CROSS_TOWER_OTHER_HEIGHT:
        for h in available_obs_ws_heights(target_data[other_site]):
            if h != source_h:
                tasks.append(make(other_site, h, "cross_tower_other_height"))
    if globals().get("RUN_CROSS_TOWER_OTHER_TIME", False):
        for h in available_obs_ws_heights(target_data[other_site]):
            tasks.append(make(other_site, h, "cross_tower_other_time", "target_all_valid"))
    return tasks


# =============================================================================
# 8. PLOTTING AND PSEUDO-PROFILE SUMMARIES
# =============================================================================

def plot_timeseries(pred: pd.DataFrame, out_png: Path, variable: str, height: int, title: str) -> None:
    if plt is None or pred.empty or not all(c in pred.columns for c in [TIME_COL, f"OBS_{variable}", f"WRF_{variable}", f"CORR_{variable}"]):
        return
    d = pred.dropna(subset=[f"OBS_{variable}", f"WRF_{variable}", f"CORR_{variable}"]).copy()
    if d.empty:
        return
    if len(d) > MAX_TIMESERIES_POINTS:
        d = d.iloc[::int(math.ceil(len(d) / MAX_TIMESERIES_POINTS))]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(d[TIME_COL], d[f"OBS_{variable}"], label="OBS", linewidth=1.2)
    ax.plot(d[TIME_COL], d[f"WRF_{variable}"], label="WRF", linestyle="--", linewidth=1.0)
    ax.plot(d[TIME_COL], d[f"CORR_{variable}"], label="Corrected", linewidth=1.0)
    unit = {"WS": "m s$^{-1}$", "WD": "°", "TKE": "m$^2$ s$^{-2}$", "T": "K"}.get(variable, "")
    ax.set_ylabel(f"{variable} at {height} m ({unit})")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def export_pseudo_profiles(all_pred: pd.DataFrame, all_metrics: pd.DataFrame, out_root: Path) -> None:
    if all_pred.empty or all_metrics.empty:
        return
    root = ensure_dir(out_root / "pseudo_profile_summary")
    keys = ["source_site", "target_site", "network", "case", "generalization_type"]
    rows = []
    for group_key, g in all_pred.groupby(keys, dropna=False):
        meta = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
        if "WS_height_m" not in g.columns:
            continue
        for h, gh in g.groupby("WS_height_m"):
            if not all(c in gh.columns for c in ["OBS_WS", "WRF_WS", "CORR_WS"]):
                continue
            rows.append({
                **meta, "height_m": int(h),
                "OBS_mean": numeric(gh["OBS_WS"]).mean(),
                "WRF_mean": numeric(gh["WRF_WS"]).mean(),
                "CORR_mean": numeric(gh["CORR_WS"]).mean(),
                "N": int(numeric(gh["OBS_WS"]).notna().sum()),
            })
    prof = pd.DataFrame(rows)
    if prof.empty:
        return
    prof.to_csv(root / "ALL_code1_generalization_pseudo_profiles.csv", index=False, encoding="utf-8-sig")
    if plt is None:
        return
    for keys_val, g in prof.groupby(keys, dropna=False):
        meta = dict(zip(keys, keys_val if isinstance(keys_val, tuple) else (keys_val,)))
        if g["height_m"].nunique() < 2:
            continue
        g = g.sort_values("height_m")
        fig, ax = plt.subplots(figsize=(5.6, 6.4))
        ax.plot(g["OBS_mean"], g["height_m"], marker="o", label="OBS")
        ax.plot(g["WRF_mean"], g["height_m"], marker="s", linestyle="--", label="WRF")
        ax.plot(g["CORR_mean"], g["height_m"], marker="^", label="Corrected")
        ax.set_xlabel("Mean WS (m s$^{-1}$)")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"{meta['source_site']}→{meta['target_site']} | {meta['network']} | {meta['case']}\n{meta['generalization_type']}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        name = "_".join(safe_name(meta[k]) for k in keys)
        fig.savefig(root / f"pseudo_profile_{name}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

# =============================================================================
# 9. APPLY ONE MODEL TO ONE TASK
# =============================================================================

def apply_model_task(
    artifact: ModelArtifact,
    task: GeneralizationTask,
    target_df_raw: pd.DataFrame,
    allowed_endpoints: Optional[pd.DatetimeIndex],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    meta = read_json(artifact.metadata_path)
    specs = specs_from_metadata(meta)
    feature_cols = pd.read_csv(
        artifact.features_path
    )["feature"].dropna().astype(str).tolist()
    window_size = int(meta.get("window_size", 24))
    tke_eps = float(meta.get("tke_eps", 1e-4))

    df_gryn, gryning_report = inject_gryning(
        target_df_raw, task.target_site
    )
    df, alias_map = alias_target_bundle(df_gryn, task)
    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_wrf = [s.wrf_col for s in specs if s.wrf_col not in df.columns]
    audit = {
        "artifact": str(artifact.case_dir),
        "source_site": artifact.source_site,
        "target_site": task.target_site,
        "network": artifact.network,
        "case": artifact.case,
        "task": task.__dict__,
        "alias_map": alias_map,
        "gryning": gryning_report,
        "MOST_reference_interpretation": MOST_REFERENCE_INTERPRETATION,
        "n_feature_cols": len(feature_cols),
        "missing_features": missing_features,
        "missing_wrf_reconstruction_cols": missing_wrf,
    }
    if missing_features or missing_wrf:
        raise ValueError(
            f"Missing target columns. features={missing_features[:20]}, "
            f"wrf={missing_wrf[:20]}"
        )

    df = prepare_features(df, feature_cols)
    X, idx = make_windows(df, feature_cols, window_size)
    keep, time_report = select_task_endpoints(
        df, idx, task, allowed_endpoints
    )
    X, idx = X[keep], idx[keep]
    audit["time_selection"] = time_report
    if len(X) == 0:
        raise ValueError(
            "No application windows remain after feature, QC and time filtering."
        )

    # Apply the target QC flag to every evaluation variable. For target-all-time
    # tasks the WS endpoint was already required to be valid; other variables
    # retain their own missing-value masks in the metric functions.
    if OBS_VALID_COL in df.columns:
        valid = numeric(df[OBS_VALID_COL]).eq(1)
        for s in specs:
            if s.obs_col in df.columns:
                df.loc[~valid, s.obs_col] = np.nan

    audit["n_application_windows"] = int(len(X))
    audit["time_start"] = str(df.loc[idx, TIME_COL].min())
    audit["time_end"] = str(df.loc[idx, TIME_COL].max())

    if AUDIT_ONLY:
        return pd.DataFrame(), pd.DataFrame(), audit

    x_scaler = load_pickle_compat(artifact.x_scaler_path)
    y_scaler = load_pickle_compat(artifact.y_scaler_path)
    nfeat = X.shape[-1]
    Xs = x_scaler.transform(X.reshape(-1, nfeat)).reshape(X.shape)
    model = load_keras_model(artifact.model_path)
    delta_scaled = np.asarray(model.predict(Xs, verbose=0))
    delta = y_scaler.inverse_transform(delta_scaled)
    pred, metrics = build_physical_outputs(
        df, idx, specs, delta, task, artifact, tke_eps
    )
    return pred, metrics, audit

# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    audit_dir = ensure_dir(OUTPUT_ROOT / "00_audit")

    target_paths = {site: discover_ml_ready(site) for site in TARGET_SITES}
    target_data = {site: load_ml_ready(path) for site, path in target_paths.items()}
    artifacts = discover_code1_models()
    if not artifacts:
        raise RuntimeError(f"No Code1 model artifacts discovered under {CODE1_RESULT_ROOT}")

    discovery_rows = [{
        "source_site": a.source_site, "network": a.network, "case": a.case,
        "scenario": a.scenario, "case_dir": str(a.case_dir),
        "model": str(a.model_path), "prediction": str(a.prediction_path or ""),
    } for a in artifacts]
    pd.DataFrame(discovery_rows).to_csv(audit_dir / "model_discovery.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"site": k, "ml_ready_csv": str(v)} for k, v in target_paths.items()]).to_csv(
        audit_dir / "target_data_discovery.csv", index=False, encoding="utf-8-sig"
    )

    all_pred: List[pd.DataFrame] = []
    all_metrics: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    tasks_by_source = {site: generate_tasks(site, target_data) for site in SOURCE_SITES}
    with open(audit_dir / "generated_tasks.json", "w", encoding="utf-8") as f:
        json.dump({k: [t.__dict__ for t in v] for k, v in tasks_by_source.items()}, f, ensure_ascii=False, indent=2)

    for artifact in artifacts:
        allowed = source_test_endpoints(artifact)
        for task in tasks_by_source[artifact.source_site]:
            print("=" * 110)
            print(f"Code1 generalization | {artifact.source_site}-{artifact.network}-{artifact.case} | {task.name}")
            try:
                pred, metrics, audit = apply_model_task(artifact, task, target_data[task.target_site].copy(), allowed)
                audit["status"] = "PASS"
                audit_rows.append(audit)
                if not pred.empty:
                    out_dir = ensure_dir(
                        OUTPUT_ROOT / task.generalization_type /
                        f"{artifact.source_site}_to_{task.target_site}" /
                        artifact.network / artifact.case / f"WS{task.apply_ws_height_m}"
                    )
                    pred.to_csv(out_dir / "predictions_generalization.csv", index=False, encoding="utf-8-sig")
                    metrics.to_csv(out_dir / "metrics_generalization.csv", index=False, encoding="utf-8-sig")
                    with open(out_dir / "task_config.json", "w", encoding="utf-8") as f:
                        json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
                    if SAVE_PLOTS:
                        for var, h in [("WS", task.apply_ws_height_m), ("WD", task.apply_wd_height_m), ("TKE", task.apply_tke_height_m)]:
                            plot_timeseries(pred, out_dir / f"timeseries_{var}.png", var, h, f"{task.name} | {artifact.network} | {artifact.case}")
                    all_pred.append(pred)
                    all_metrics.append(metrics)
            except Exception as exc:
                failure = {
                    "source_site": artifact.source_site, "target_site": task.target_site,
                    "network": artifact.network, "case": artifact.case, "task": task.name,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                }
                failure_rows.append(failure)
                print(f"[FAILED] {failure['error']}")
                if STOP_ON_ERROR:
                    raise

    if all_pred:
        pred_all = pd.concat(all_pred, ignore_index=True)
        pred_all.to_csv(OUTPUT_ROOT / "ALL_code1_generalization_predictions.csv", index=False, encoding="utf-8-sig")
    else:
        pred_all = pd.DataFrame()
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(OUTPUT_ROOT / "ALL_code1_generalization_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        metrics_all = pd.DataFrame()

    pd.DataFrame(audit_rows).to_json(audit_dir / "application_audit.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(failure_rows).to_csv(audit_dir / "failures.csv", index=False, encoding="utf-8-sig")
    export_pseudo_profiles(pred_all, metrics_all, OUTPUT_ROOT)

    summary = {
        "n_models_discovered": len(artifacts),
        "n_tasks_generated": sum(len(v) for v in tasks_by_source.values()),
        "n_success": len(audit_rows),
        "n_failed": len(failure_rows),
        "audit_only": AUDIT_ONLY,
        "scope": "Existing source-test-time tasks plus cross_tower_other_time at all valid target times and all evaluable WS heights.",
        "other_time_definition": "All valid target-site times; source-period overlap is retained by design.",
        "note": "WS height is varied; WD/TKE use the target tower's native evaluable heights. See every task_config.json.",
    }
    with open(audit_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=" * 110)
    print("Finished Code1 generalization")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
