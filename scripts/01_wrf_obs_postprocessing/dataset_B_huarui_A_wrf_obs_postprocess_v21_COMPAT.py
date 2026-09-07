#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两站点 huarui_A WRF tslist 完整后处理脚本（增强版）

增强内容：
V15 compatibility note: original outputs are kept; ML-ready output receives additional Code1-4-compatible aliases for TKE/temperature and clearer WS-residual columns.
1. 修复 wrfout 中 TKE_PBL 为 bottom_top_stag 时的高度不匹配问题；
2. 新增观测温度(tp) vs WRF 温度剖面对比；
3. 新增观测湍流强度(tbl) + 风速(wspd) 推算 TKE，并与 WRF TKE 对比；
4. 新增风速 / 风向 obs vs WRF 时序对比图与导出 CSV；
5. 廓线绘图中 WRF 使用原始模式层点（不连线，不使用插值后高度）；
6. Excel 中新增原始层时间段平均、观测平均剖面、时序对比等工作表；
7. 新增 obs vs WRF 的 L 时序对比，以及 WRF L 全空时的诊断信息。
8. 新增 obs vs WRF 全变量误差统计与误差绘图（Bias/MAE/RMSE/R/R²/NSE/IA等）。
9. 新增风玫瑰 / 风速频率分布 / Weibull 分布误差统计与绘图。
"""

import os
import re
import math
import traceback
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ====================== CONFIGURATION (edit as needed) ======================

HEIGHT_MATCH_METHOD = "interpolate"   # "interpolate" or "nearest"
OUT_OF_RANGE_POLICY = "boundary"      # "boundary" or "skip"
HEIGHT_WARNING_THRESHOLD = 5.0
DEFAULT_STABILITY_HEIGHTS = (10.0, 50.0)

# ====================== ML-ready ref10 / MOST output configuration ======================
# Note: the following options only add outputs and do not change the existing full_csv / period_mean / Excel logic.
# Low-level reference height for CNN_LSTM / PINN. The current setup uses 10 m.
REF_HEIGHT_ML = 10.0

# WRF interpolation heights added explicitly. 10 m is used for bias correction; 50 m is retained for Bulk Ri / shear diagnostics.
FORCE_EXTRA_HEIGHTS = (10.0, 50.0)

# Whether to export the 10 m machine-learning table read directly by CNN_LSTM/PINN.
EXPORT_ML_READY_REF10 = True

# Whether to add multi-height profile wide-format columns to the same ML_ready_ref10 file.
# If True, the file can be read by the 10 m single-height CNN_LSTM/PINN,
# and by the multi-height CNN_LSTM/PINN.
EXPORT_ML_READY_PROFILE_WIDE = True

# Multi-height ML-ready output heights. None uses all heights already present in result_df;
# to export only common LiDAR heights, use [40, 60, 80, 90, 100, 110, 120, 140, 160, 180, 200, 210].
ML_READY_PROFILE_HEIGHTS = None

# Whether to additionally calculate and export the theoretical MOST wind speed U_MO_profile / U_MO10.
# Note: whether MOST is used as a model input is determined by the downstream bias-correction ablation.
EXPORT_MOST_THEORY = True

# MOST parameters. If ZNT/Z0 is unavailable in wrfout, Z0_FALLBACK is used.
KAPPA = 0.40
MOST_D_FACTOR = 0.0          # d = MOST_D_FACTOR * z0; use 0 here, with no imposed displacement height.
Z0_FALLBACK = 0.03
MOST_MIN_Z0 = 1e-5
MOST_MAX_REASONABLE_U = 75.0
ML_MERGE_TOLERANCE = "5min"
OBS_WRF_MATCH_TOLERANCE = "5min"  # Windographer power files are hourly; avoid duplicating one obs over many 10-min WRF rows.
# Windographer explicitly says time stamps indicate the beginning of the time step.
# Therefore WRF 10-min averages are labelled by window beginning for obs matching.
TSLIST_AVG_TIME_LABEL = "begin"  # "begin" or "end"

NEST_LEVEL = 3
CASE_NAME = "huarui_A"

# ====================== Current case: Huarui_A (WRF path follows mysetup.sh: hurui/A/meso_1km) ======================
# According to mysetup.sh / namelist.input / namelist.wps:
# d01/d02/d03 = 9 km / 3 km / 1 km; tslist contains two met towers, t001/t002;
# the WRF simulation period is 2023-09-01 00:00 UTC to 2024-04-30 00:00 UTC; script outputs/comparisons use Beijing time.
WRF_OUTPUT_DIR = r"/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_1km"
WRFOUT_DIR = r"/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_1km"
OUTPUT_DIR = r"/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_1km/data_post_new/huarui_A_two_towers_full"

# WRF namelist times are UTC; the script converts WRF tslist/wrfout times to Beijing time (+8 h).
# Windographer time stamps indicate the beginning of the time step, so the comparison period is also set in Beijing time.
AVG_START_STR = "2023-09-01 08:00"
AVG_END_STR   = "2024-04-30 08:00"

PROFILE_ZOOM_MAX = 500.0

# ====================== Wind-rose / Weibull distribution diagnostic configuration ======================
# Note: the following options only add distribution-error outputs and do not change existing time-series, profile, or ML-ready outputs.
EXPORT_WIND_DISTRIBUTION_DIAGNOSTICS = True

# Number of wind-rose direction sectors. 16 gives one sector every 22.5 degrees.
WIND_ROSE_NUM_SECTORS = 16

# Wind-speed bins for joint wind-rose frequencies. Keep the final upper bound at 75 m/s to match the QC range.
WIND_ROSE_SPEED_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 75.0)

# Wind-speed bin width for the empirical Weibull frequency curve.
WEIBULL_SPEED_BIN_WIDTH = 1.0

# Minimum number of matched samples required at each height for Weibull fitting.
WEIBULL_MIN_SAMPLES = 20

OBS_VALID_RANGES = {
    "wspd": (0.0, 75.0),
    "wdir": (0.0, 360.0),
    "tp": (-60.0, 60.0),
    "rh": (0.0, 100.0),
    "tbl": (0.0, 2.0),
    "prs": (300.0, 1100.0),
    "tke": (0.0, 200.0),
}

# Some observation systems use 0 for missing or frozen values; by default, zero wspd/tbl values are treated as invalid here.
# Set this to False if zero values represent real calm wind or real TI.
OBS_ZERO_AS_NAN = {
    "wspd": False,
    "wdir": False,
    "tp": False,
    "rh": False,
    "tbl": False,
    "prs": False,
}

_CLEANED_OBS_LOGGED = set()

# The two met towers are defined in tslist:
# tower1 -> t001 -> C039801
# tower2 -> t002 -> C039802
# obs_csv is generated automatically from Windographer txt files at the start of the main program.
STATIONS = [
    {
        "station_name": "t001",
        "site_tag": "C039801",
        "obs_csv": "",
        "nest_level": 3,
    },
    {
        "station_name": "t002",
        "site_tag": "C039802",
        "obs_csv": "",
        # t002 is outside d03 in the current WRF output; use available d02 tslist files.
        "nest_level": 2,
    },
]

# ====================== Windographer observation txt auto-adapter configuration ======================
# Notes:
# 1) The power-calculation files contain gap-filled wind speed/direction and are suitable for full-period WS/WD validation;
# 2) The turbulence-calculation files retain standard deviation and are suitable for TI/TKE calculation;
# 3) This script automatically merges the original 8 txt files plus supplemental height-specific wind-speed txt files into the obs_csv format required by the original script;
#    wind speed can be extended to 10/40/80/100/130/140/160 m, while wind direction/TI/TKE mainly come from the 130/160 m files.
AUTO_BUILD_WINDOGRAPHER_OBS_CSV = True

# Place the Windographer txt files in this directory; if not found, the script also checks the current working directory and the script directory.
WINDOGRAPHER_OBS_DIR = r"/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_1km/data_post_new/obs"
# This path matches the obs folder used in the original setup.

# Data source for wind speed/direction in WRF-vs-OBS validation:
# "power"      -> use gap-filled wind data from the power-calculation files, typically at 1 h resolution;
# "turbulence" -> use original wind data from the turbulence-calculation files, typically at 10 min resolution; the 130 m direction may come from the 120 m vane.
WINDOGRAPHER_WIND_SOURCE_FOR_VALIDATION = "power"

# Supplemental files: wind speed at each height. When enabled, wspd{height}m is preferentially read from these files,
# supporting multi-height wind-speed validation and profile ML-ready output at 10/40/80/100/130/140/160 m and other available heights.
USE_SUPPLEMENTAL_ALL_HEIGHT_SPEED = True

WINDOGRAPHER_TOWER_CONFIGS = [
    {
        "station_name": "t001",
        "site_tag": "C039801",
        "nest_level": 3,
        "lat": 34.665893,
        "lon": 111.529245,
        "elevation": 783.0,
        "all_height_speed_file": "039801_20230901-20240901_各高度风速.txt",
        "heights": {
            130: {
                "power_file": "039801测风数据-130-计算发电量.txt",
                "turbulence_file": "039801测风数据-130-计算湍流.txt",
            },
            160: {
                "power_file": "039801测风数据-160-计算发电量.txt",
                "turbulence_file": "039801测风数据-160-计算湍流.txt",
            },
        },
    },
    {
        "station_name": "t002",
        "site_tag": "C039802",
        # t002 only has d01/d02 tslist outputs in the current run.
        "nest_level": 2,
        "lat": 34.596327,
        "lon": 111.660960,
        "elevation": 622.0,
        "all_height_speed_file": "039802_20230901-20240901_各高度风速.txt",
        "heights": {
            130: {
                "power_file": "039802测风数据-130-计算发电量.txt",
                "turbulence_file": "039802测风数据-130-计算湍流.txt",
            },
            160: {
                "power_file": "039802测风数据-160-计算发电量.txt",
                "turbulence_file": "039802测风数据-160-计算湍流.txt",
            },
        },
    },
]

OBS_VAR_META = {
    "wspd": {"label": "Wind speed", "unit": "m s$^{-1}$", "plot_name": "ws", "obs_col": "观测风速 (m/s)"},
    "wdir": {"label": "Wind direction", "unit": "deg", "plot_name": "wd", "obs_col": "观测风向 (°)"},
    "tp":   {"label": "Temperature", "unit": "°C", "plot_name": "T", "obs_col": "观测温度 (°C)"},
    "tbl":  {"label": "Turbulence intensity", "unit": "-", "plot_name": "TI", "obs_col": "观测湍流强度 (-)"},
    "tke":  {"label": "TKE", "unit": "m$^2$ s$^{-2}$", "plot_name": "TKE", "obs_col": "观测TKE (m2/s2)"},
    "l":    {"label": "Monin-Obukhov length", "unit": "m", "plot_name": "L", "obs_col": "观测L (m)"},
}


# ====================== Basic utility functions ======================

def file_exists_and_non_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def safe_sheet_name(name):
    """Excel sheet 名最长 31 字符"""
    name = re.sub(r"[:\\/?*\[\]]", "_", str(name))
    return name[:31]


def append_sheets_safe(excel_file, sheet_map, site_tag="", stage="追加写入"):
    """安全地向 Excel 追加多个 sheet。失败时只打印警告，不中断主流程。"""
    valid_map = {k: v for k, v in (sheet_map or {}).items() if v is not None and not v.empty}
    if not valid_map:
        return True
    try:
        with pd.ExcelWriter(excel_file, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            for sheet_name, df in valid_map.items():
                df.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)
        return True
    except Exception as e:
        tag = f"[{site_tag}] " if site_tag else ""
        print(f"{tag}⚠ {stage} 失败: {e}")
        print(traceback.format_exc())
        return False


def find_closest_level(target_height, height_levels):
    return int(np.argmin(np.abs(height_levels - target_height)))


def parse_ts_file(ts_file):
    with open(ts_file, 'r') as f:
        header = f.readline().strip()

    pattern = (
        r"(.{24})"
        r"\s+(\d+)\s+(\d+)"
        r"\s+(\S+)"
        r"\s+\(\s*([-]?\d+\.\d+)\s*,\s*([-]?\d+\.\d+)\s*\)"
        r"\s+\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        r"\s+\(\s*([-]?\d+\.\d+)\s*,\s*([-]?\d+\.\d+)\s*\)"
        r"\s+(\d+\.\d+)\s+meters"
        r"\s+(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"
    )

    match = re.search(pattern, header)
    if match:
        return {
            'station_name': match.group(1).strip(),
            'grid_id': int(match.group(2)),
            'ts_id': int(match.group(3)),
            'abbreviation': match.group(4),
            'lat': float(match.group(5)),
            'lon': float(match.group(6)),
            'ix': int(match.group(7)),
            'iy': int(match.group(8)),
            'grid_lat': float(match.group(9)),
            'grid_lon': float(match.group(10)),
            'elevation': float(match.group(11)),
            'start_time': datetime.strptime(match.group(12), '%Y-%m-%d_%H:%M:%S')
        }
    return None


def get_height_levels(wrfinput_file, ix, iy):
    with Dataset(wrfinput_file) as f:
        ph  = f.variables['PH'][0, :, iy - 1, ix - 1]
        phb = f.variables['PHB'][0, :, iy - 1, ix - 1]
        hgt = f.variables['HGT'][0, iy - 1, ix - 1]
        z_w = (ph + phb) / 9.81
        z_w_agl = z_w - hgt
        z_mass_agl = 0.5 * (z_w_agl[:-1] + z_w_agl[1:])
        return np.round(z_mass_agl, 4)


def read_profile_file(file_path):
    try:
        data = np.loadtxt(file_path, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time_hr = data[:, 0]
        profile_data = data[:, 1:]
        return time_hr, profile_data
    except Exception as e:
        print(f"警告: 读取文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return np.array([]), np.array([])


def calculate_vector_average(u_series, v_series):
    u_mean = np.mean(u_series)
    v_mean = np.mean(v_series)
    ws_avg = np.sqrt(u_mean ** 2 + v_mean ** 2)
    wd_rad = np.arctan2(v_mean, u_mean)
    wd_deg = np.degrees(wd_rad)
    wd_avg = (270.0 - wd_deg) % 360.0
    return ws_avg, wd_avg, u_mean, v_mean


def calculate_vector_average_from_df(group):
    return calculate_vector_average(group['平均_u (m/s)'].values, group['平均_v (m/s)'].values)


def classify_stability_zL(z_over_L):
    if np.isnan(z_over_L):
        return np.nan
    if z_over_L < -1.0:
        return "Strongly unstable"
    elif -1.0 <= z_over_L < -0.1:
        return "Moderately unstable"
    elif -0.1 <= z_over_L < -0.01:
        return "Weakly unstable / near neutral"
    elif -0.01 <= z_over_L <= 0.01:
        return "Neutral"
    elif 0.01 < z_over_L <= 0.1:
        return "Weakly stable"
    elif 0.1 < z_over_L <= 1.0:
        return "Moderately stable"
    else:
        return "Strongly stable"


# ====================== Observation CSV reading and derived variables ======================

def _match_obs_height(col_name, prefix):
    m = re.fullmatch(rf"{prefix}(\d+)m", str(col_name).strip())
    return float(m.group(1)) if m else None


def get_obs_available_heights_by_prefix(csv_file, prefixes=("wspd", "wdir", "tp", "tbl")):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"未找到观测 CSV: {csv_file}")

    cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
    out = {}
    for prefix in prefixes:
        hs = []
        for c in cols:
            h = _match_obs_height(c, prefix)
            if h is not None:
                hs.append(h)
        out[prefix] = sorted(set(hs))
    return out


def get_obs_union_heights_from_csv(csv_file, prefixes=("wspd", "wdir", "tp", "tbl")):
    info = get_obs_available_heights_by_prefix(csv_file, prefixes=prefixes)
    hs = sorted(set(h for vals in info.values() for h in vals))
    if not hs:
        raise ValueError(f"在 {csv_file} 中没有找到可用高度列")
    return hs, info


def _sanitize_obs_numeric_series(series, prefix=None):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if prefix in OBS_VALID_RANGES:
        vmin, vmax = OBS_VALID_RANGES[prefix]
        s = s.where((s >= vmin) & (s <= vmax), np.nan)
    if prefix in OBS_ZERO_AS_NAN and OBS_ZERO_AS_NAN[prefix]:
        s = s.where(s != 0, np.nan)
    return s


def clean_obs_dataframe(df, csv_file=None, verbose=True):
    df = df.copy()
    summary = []

    for c in df.columns:
        if c == "rec_time":
            continue

        prefix = None
        for cand in ["wspd", "wdir", "tp", "rh", "tbl", "tke"]:
            if _match_obs_height(c, cand) is not None:
                prefix = cand
                break
        if prefix is None and c == "prs":
            prefix = "prs"
        elif prefix is None and c in ["L", "L_level"]:
            prefix = None

        raw = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cleaned = _sanitize_obs_numeric_series(raw, prefix=prefix) if prefix is not None else raw
        removed = int(raw.notna().sum() - cleaned.notna().sum())
        if removed > 0:
            summary.append((c, removed))
        df[c] = cleaned

    if verbose and csv_file is not None and csv_file not in _CLEANED_OBS_LOGGED:
        if summary:
            total_removed = sum(x[1] for x in summary)
            print(f"[OBS清洗] {os.path.basename(csv_file)} 共剔除 {total_removed} 个异常/越界值。")
            for c, n in summary[:12]:
                print(f"    - {c}: 剔除 {n} 个")
            if len(summary) > 12:
                print(f"    - 其余 {len(summary)-12} 列也有剔除")
        else:
            print(f"[OBS清洗] {os.path.basename(csv_file)} 未发现需剔除的越界值。")
        _CLEANED_OBS_LOGGED.add(csv_file)
    return df


def load_obs_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    if 'rec_time' not in df.columns:
        raise ValueError(f"{csv_file} 中缺少 rec_time 列")
    df['rec_time'] = pd.to_datetime(df['rec_time'])
    df = clean_obs_dataframe(df, csv_file=csv_file, verbose=True)
    return df


def subset_obs_by_period(df, avg_start, avg_end):
    mask = (df['rec_time'] >= avg_start) & (df['rec_time'] <= avg_end)
    return df.loc[mask].copy()


def get_obs_valid_time_bounds(df, var_prefix):
    sub = df.copy()
    if var_prefix == "l":
        if "L" not in sub.columns:
            return None, None
        mask = pd.to_numeric(sub["L"], errors="coerce").notna()
    elif var_prefix == "tke":
        # V16 hurui case addition:
        # Windographer turbulence files can provide direct tke{height}m columns
        # computed from wind-speed standard deviation. Prefer direct TKE when present,
        # and keep the original fallback 1.5*(TI*WS)^2 for older obs CSVs.
        mask = pd.Series(False, index=sub.index)

        direct_cols = [c for c in sub.columns if _match_obs_height(c, "tke") is not None]
        for c in direct_cols:
            tke_direct = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mask = mask | tke_direct.notna()

        wspd_cols = [c for c in sub.columns if _match_obs_height(c, "wspd") is not None]
        tbl_cols = [c for c in sub.columns if _match_obs_height(c, "tbl") is not None]
        common_heights = sorted(set(_match_obs_height(c, "wspd") for c in wspd_cols).intersection(
            set(_match_obs_height(c, "tbl") for c in tbl_cols)
        ))
        for h in common_heights:
            ws_col = next((c for c in wspd_cols if _match_obs_height(c, "wspd") == h), None)
            ti_col = next((c for c in tbl_cols if _match_obs_height(c, "tbl") == h), None)
            if ws_col is None or ti_col is None:
                continue
            ws = pd.to_numeric(sub[ws_col], errors="coerce")
            ti = pd.to_numeric(sub[ti_col], errors="coerce")
            tke = 1.5 * (ti * ws) ** 2
            mask = mask | tke.replace([np.inf, -np.inf], np.nan).notna()
    else:
        cols = [c for c in sub.columns if _match_obs_height(c, var_prefix) is not None]
        if not cols:
            return None, None
        mask = pd.Series(False, index=sub.index)
        for c in cols:
            mask = mask | pd.to_numeric(sub[c], errors="coerce").notna()

    if not mask.any():
        return None, None
    valid_times = sub.loc[mask, 'rec_time']
    return valid_times.min(), valid_times.max()

def get_common_profile_period(obs_csv, wrf_time_like, var_prefix, avg_start, avg_end):
    obs_df = subset_obs_by_period(load_obs_dataframe(obs_csv), avg_start, avg_end)
    obs_start, obs_end = get_obs_valid_time_bounds(obs_df, var_prefix)
    if obs_start is None or obs_end is None:
        return None, None

    if wrf_time_like is None:
        return None, None
    wrf_times = pd.to_datetime(pd.Series(wrf_time_like)).dropna()
    wrf_times = wrf_times[(wrf_times >= avg_start) & (wrf_times <= avg_end)]
    if wrf_times.empty:
        return None, None

    common_start = max(pd.Timestamp(avg_start), pd.Timestamp(obs_start), pd.Timestamp(wrf_times.min()))
    common_end = min(pd.Timestamp(avg_end), pd.Timestamp(obs_end), pd.Timestamp(wrf_times.max()))
    if common_start > common_end:
        return None, None
    return common_start.to_pydatetime(), common_end.to_pydatetime()


def build_obs_period_profiles(csv_file, avg_start, avg_end, instrument_name=None):
    """
    输出 dict:
      wspd, wdir, tp, tbl, tke, l
    每个元素都是按给定时间窗统计后的 DataFrame。
    obs 数据在读入时会先清洗 NaN/越界值；做与 WRF 的廓线对比时，
    应传入与 WRF 一致的共同时间窗。

    V16 hurui case addition:
    - If direct tke{height}m columns exist, use them first.
    - Otherwise keep the original TKE estimate from TI and WS: 1.5*(TI*WS)^2.
    """
    df = load_obs_dataframe(csv_file)
    sub = subset_obs_by_period(df, avg_start, avg_end)
    inst = instrument_name if instrument_name else os.path.basename(csv_file)

    if sub.empty:
        return {k: pd.DataFrame() for k in ["wspd", "wdir", "tp", "tbl", "tke", "l"]}

    out = {}
    for prefix in ["wspd", "wdir", "tp", "tbl"]:
        rows = []
        for c in sub.columns:
            h = _match_obs_height(c, prefix)
            if h is None:
                continue
            val = pd.to_numeric(sub[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            mean_val = val.mean(skipna=True)
            valid_n = int(val.notna().sum())
            if pd.notna(mean_val) and valid_n > 0:
                rows.append({
                    "instrument": inst,
                    "height": float(h),
                    OBS_VAR_META[prefix]["obs_col"]: float(mean_val),
                    "有效样本数": valid_n,
                    "比较时间段开始 (北京)": avg_start,
                    "比较时间段结束 (北京)": avg_end,
                })
        out[prefix] = pd.DataFrame(rows).sort_values("height").reset_index(drop=True) if rows else pd.DataFrame()

    tke_rows = []
    direct_tke_hs = {_match_obs_height(c, "tke"): c for c in sub.columns if _match_obs_height(c, "tke") is not None}
    wspd_hs = {_match_obs_height(c, "wspd"): c for c in sub.columns if _match_obs_height(c, "wspd") is not None}
    tbl_hs = {_match_obs_height(c, "tbl"): c for c in sub.columns if _match_obs_height(c, "tbl") is not None}
    all_tke_hs = sorted(set(direct_tke_hs.keys()).union(set(wspd_hs.keys()).intersection(tbl_hs.keys())))

    for h in all_tke_hs:
        if h in direct_tke_hs:
            tke_ts = pd.to_numeric(sub[direct_tke_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
        else:
            ws = pd.to_numeric(sub[wspd_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
            ti = pd.to_numeric(sub[tbl_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
            tke_ts = (1.5 * (ti * ws) ** 2).replace([np.inf, -np.inf], np.nan)

        mean_tke = tke_ts.mean(skipna=True)
        valid_n = int(tke_ts.notna().sum())
        if pd.notna(mean_tke) and valid_n > 0:
            tke_rows.append({
                "instrument": inst,
                "height": float(h),
                OBS_VAR_META["tke"]["obs_col"]: float(mean_tke),
                "有效样本数": valid_n,
                "比较时间段开始 (北京)": avg_start,
                "比较时间段结束 (北京)": avg_end,
            })
    out["tke"] = pd.DataFrame(tke_rows).sort_values("height").reset_index(drop=True) if tke_rows else pd.DataFrame()

    if "L" in sub.columns:
        L_series = pd.to_numeric(sub["L"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        L_mean = L_series.mean(skipna=True)
        valid_n = int(L_series.notna().sum())
        if pd.notna(L_mean) and valid_n > 0:
            out["l"] = pd.DataFrame([{
                "instrument": inst,
                OBS_VAR_META["l"]["obs_col"]: float(L_mean),
                "有效样本数": valid_n,
                "比较时间段开始 (北京)": avg_start,
                "比较时间段结束 (北京)": avg_end,
            }])
        else:
            out["l"] = pd.DataFrame()
    else:
        out["l"] = pd.DataFrame()

    return out

def build_obs_timeseries_long(csv_file, avg_start, avg_end, var_prefix, heights=None):
    """
    将 obs 变量转成长表，并在输出前剔除 NaN/无效值。

    V16 hurui case addition:
    - For TKE, direct tke{height}m columns are preferred when present.
    - Otherwise, TKE is estimated from TI and wind speed as in the original script.
    """
    df = load_obs_dataframe(csv_file)
    sub = subset_obs_by_period(df, avg_start, avg_end)
    if sub.empty:
        return pd.DataFrame()

    rows = []
    heights = None if heights is None else sorted(set(float(h) for h in heights))

    if var_prefix == "l":
        if "L" not in sub.columns:
            return pd.DataFrame()
        out = pd.DataFrame({
            "北京时间": sub["rec_time"].values,
            OBS_VAR_META["l"]["obs_col"]: pd.to_numeric(sub["L"], errors="coerce").replace([np.inf, -np.inf], np.nan).values
        })
        return out.dropna(subset=[OBS_VAR_META["l"]["obs_col"]]).reset_index(drop=True)

    if var_prefix == "tke":
        direct_tke_hs = {_match_obs_height(c, "tke"): c for c in sub.columns if _match_obs_height(c, "tke") is not None}
        wspd_hs = {_match_obs_height(c, "wspd"): c for c in sub.columns if _match_obs_height(c, "wspd") is not None}
        tbl_hs = {_match_obs_height(c, "tbl"): c for c in sub.columns if _match_obs_height(c, "tbl") is not None}
        common_hs = sorted(set(direct_tke_hs.keys()).union(set(wspd_hs.keys()).intersection(tbl_hs.keys())))
        if heights is not None:
            common_hs = [h for h in common_hs if h in heights]
        for h in common_hs:
            if h in direct_tke_hs:
                tke_ts = pd.to_numeric(sub[direct_tke_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
            else:
                ws = pd.to_numeric(sub[wspd_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
                ti = pd.to_numeric(sub[tbl_hs[h]], errors='coerce').replace([np.inf, -np.inf], np.nan)
                tke_ts = (1.5 * (ti * ws) ** 2).replace([np.inf, -np.inf], np.nan)

            tmp = pd.DataFrame({
                "北京时间": sub["rec_time"].values,
                "离地高度 (m)": float(h),
                OBS_VAR_META["tke"]["obs_col"]: tke_ts.values
            }).dropna(subset=[OBS_VAR_META["tke"]["obs_col"]])
            if not tmp.empty:
                rows.append(tmp)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    obs_col_name = OBS_VAR_META[var_prefix]["obs_col"]
    for c in sub.columns:
        h = _match_obs_height(c, var_prefix)
        if h is None:
            continue
        if heights is not None and float(h) not in heights:
            continue
        val = pd.to_numeric(sub[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        tmp = pd.DataFrame({
            "北京时间": sub["rec_time"].values,
            "离地高度 (m)": float(h),
            obs_col_name: val.values
        }).dropna(subset=[obs_col_name])
        if not tmp.empty:
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ====================== Height matching / interpolation functions ======================
# ====================== Height matching / interpolation functions ======================

def get_height_mapping_table(target_heights, wrf_heights, height_match_method=None, out_of_range_policy=None):
    rows = []
    wrf_heights = np.asarray(wrf_heights, dtype=float)
    height_match_method = HEIGHT_MATCH_METHOD if height_match_method is None else height_match_method
    out_of_range_policy = OUT_OF_RANGE_POLICY if out_of_range_policy is None else out_of_range_policy

    for h in target_heights:
        nearest_idx = find_closest_level(h, wrf_heights)
        nearest_h = float(wrf_heights[nearest_idx])
        nearest_diff = nearest_h - float(h)

        below = wrf_heights[wrf_heights <= h]
        above = wrf_heights[wrf_heights >= h]

        lower_h = float(below.max()) if below.size > 0 else np.nan
        upper_h = float(above.min()) if above.size > 0 else np.nan

        if height_match_method == "nearest":
            method = "nearest"
            used_h = nearest_h
            used_diff = nearest_diff
        else:
            if np.isnan(lower_h) or np.isnan(upper_h):
                method = f"interpolate->{out_of_range_policy}"
                if out_of_range_policy == "boundary":
                    used_h = nearest_h
                    used_diff = nearest_diff
                else:
                    used_h = np.nan
                    used_diff = np.nan
            elif np.isclose(lower_h, upper_h):
                method = "exact_or_same_level"
                used_h = lower_h
                used_diff = used_h - float(h)
            else:
                method = "interpolate"
                used_h = float(h)
                used_diff = 0.0

        rows.append({
            "观测高度 (m)": float(h),
            "最近WRF层高度 (m)": nearest_h,
            "最近层高度差 (m)": nearest_diff,
            "WRF下层高度 (m)": lower_h,
            "WRF上层高度 (m)": upper_h,
            "提取方式": method,
            "最终代表高度 (m)": used_h,
            "最终高度差 (m)": used_diff,
            "是否超阈值": abs(nearest_diff) > HEIGHT_WARNING_THRESHOLD
        })

    return pd.DataFrame(rows)


def sample_profile_at_height(profile_values, height_levels, target_height,
                             method="interpolate", out_of_range_policy="boundary"):
    z = np.asarray(height_levels, dtype=float)
    v = np.asarray(profile_values, dtype=float)

    if z.size != v.size:
        raise ValueError(f"height_levels 与 profile_values 长度不一致: {z.size} vs {v.size}")

    nearest_idx = find_closest_level(target_height, z)
    nearest_h = float(z[nearest_idx])
    nearest_diff = nearest_h - float(target_height)

    meta = {
        "target_height": float(target_height),
        "nearest_height": nearest_h,
        "nearest_diff": nearest_diff,
        "lower_height": np.nan,
        "upper_height": np.nan,
        "method_used": None,
        "out_of_range": False,
    }

    if method == "nearest":
        meta["method_used"] = "nearest"
        return float(v[nearest_idx]), meta

    below_idx = np.where(z <= target_height)[0]
    above_idx = np.where(z >= target_height)[0]

    if below_idx.size == 0 or above_idx.size == 0:
        meta["out_of_range"] = True
        if out_of_range_policy == "boundary":
            meta["method_used"] = "boundary"
            return float(v[nearest_idx]), meta
        meta["method_used"] = "skip"
        return np.nan, meta

    i0 = int(below_idx[-1])
    i1 = int(above_idx[0])

    z0 = float(z[i0])
    z1 = float(z[i1])
    meta["lower_height"] = z0
    meta["upper_height"] = z1

    if i0 == i1 or np.isclose(z0, z1):
        meta["method_used"] = "exact_or_same_level"
        return float(v[i0]), meta

    frac = (float(target_height) - z0) / (z1 - z0)
    val = float(v[i0] + frac * (v[i1] - v[i0]))
    meta["method_used"] = "interpolate"
    return val, meta


# ====================== Stability and mean profiles ======================

def compute_stability_bulk_Ri(result_df, stability_heights=None):
    if result_df.empty:
        return result_df

    unique_heights = sorted(result_df['离地高度 (m)'].unique())

    if stability_heights is None:
        if len(unique_heights) < 2:
            print("高度层不足，无法计算 Ri")
            return result_df
        h1, h2 = unique_heights[0], unique_heights[1]
    else:
        req1, req2 = stability_heights
        h1 = min(unique_heights, key=lambda z: abs(z - req1))
        h2 = min(unique_heights, key=lambda z: abs(z - req2))

    print(f"用于计算稳定度的两层高度(实际使用): {h1} m, {h2} m")

    g = 9.81
    result_df['Bulk_Ri'] = np.nan
    result_df['Stability_Ri'] = np.nan
    result_df['z_over_L'] = np.nan
    result_df['Stability_L'] = np.nan

    z_ref = float(min(h1, h2))

    for _, group in result_df.groupby('北京时间'):
        sub1 = group[group['离地高度 (m)'] == h1]
        sub2 = group[group['离地高度 (m)'] == h2]

        if sub1.empty or sub2.empty:
            continue

        r1 = sub1.iloc[0]
        r2 = sub2.iloc[0]

        z1, z2 = float(h1), float(h2)
        dz = z2 - z1
        if dz == 0:
            continue

        theta1 = r1['十分钟平均温度 (K)']
        theta2 = r2['十分钟平均温度 (K)']
        dtheta = theta2 - theta1

        du = r2['平均_u (m/s)'] - r1['平均_u (m/s)']
        dv = r2['平均_v (m/s)'] - r1['平均_v (m/s)']
        shear2 = (du / dz) ** 2 + (dv / dz) ** 2

        if shear2 == 0:
            Ri = np.nan
        else:
            theta_mean = 0.5 * (theta1 + theta2)
            Ri = (g / theta_mean) * (dtheta / dz) / shear2

        if np.isnan(Ri):
            stab_Ri = np.nan
        elif Ri < 0:
            stab_Ri = 'Unstable'
        elif Ri <= 0.25:
            stab_Ri = 'Neutral/Weakly stable'
        else:
            stab_Ri = 'Stable'

        result_df.loc[group.index, 'Bulk_Ri'] = Ri
        result_df.loc[group.index, 'Stability_Ri'] = stab_Ri

        L_val = group.iloc[0]['L (m)']
        if (not pd.isna(L_val)) and (L_val != 0):
            zL = z_ref / L_val
            stab_L = classify_stability_zL(zL)
            result_df.loc[group.index, 'z_over_L'] = zL
            result_df.loc[group.index, 'Stability_L'] = stab_L

    result_df['Stability'] = result_df['Stability_Ri']
    return result_df


def compute_period_mean(result_df, avg_start=None, avg_end=None):
    if avg_start is None or avg_end is None or result_df.empty:
        return None

    mask = (result_df['北京时间'] >= avg_start) & (result_df['北京时间'] <= avg_end)
    sub = result_df[mask].copy()
    if sub.empty:
        print("给定时间段内没有数据，无法计算时间段平均。")
        return None

    rows = []
    for h, group in sub.groupby('离地高度 (m)'):
        ws_avg, wd_avg, u_mean, v_mean = calculate_vector_average_from_df(group)
        avg_p = group['十分钟平均气压 (KPa)'].mean()
        avg_T = group['十分钟平均温度 (K)'].mean()
        avg_TKE = group['TKE (m2/s2)'].mean() if 'TKE (m2/s2)' in group.columns else np.nan
        valid_n = int(group['北京时间'].notna().sum())

        rows.append({
            '离地高度 (m)': h,
            '时间段开始 (北京)': avg_start,
            '时间段结束 (北京)': avg_end,
            '时间段平均风速 (m/s)': round(ws_avg, 4),
            '时间段平均风向 (°)': round(wd_avg, 4),
            '时间段平均气压 (KPa)': round(avg_p, 4),
            '时间段平均温度 (K)': round(avg_T, 4),
            '时间段平均温度 (°C)': round(avg_T - 273.15, 4) if pd.notna(avg_T) else np.nan,
            '时间段平均_u (m/s)': round(u_mean, 6),
            '时间段平均_v (m/s)': round(v_mean, 6),
            '时间段平均TKE (m2/s2)': round(avg_TKE, 6) if not np.isnan(avg_TKE) else np.nan,
            '样本数': valid_n,
            '实际使用时间段开始 (北京)': avg_start,
            '实际使用时间段结束 (北京)': avg_end,
            '是否matched': False,
        })

    return pd.DataFrame(rows).sort_values('离地高度 (m)').reset_index(drop=True)


def compute_native_period_mean(profile_windows, height_levels, avg_start=None, avg_end=None):
    if avg_start is None or avg_end is None or not profile_windows:
        return None

    selected = [w for w in profile_windows if avg_start <= w["window_end_time"] <= avg_end]
    if not selected:
        return None

    U = np.vstack([w["u_prof_mean"] for w in selected])
    V = np.vstack([w["v_prof_mean"] for w in selected])
    TH = np.vstack([w["th_prof_mean"] for w in selected])
    PR = np.vstack([w["pr_prof_mean"] for w in selected])

    u_mean = np.nanmean(U, axis=0)
    v_mean = np.nanmean(V, axis=0)
    ws_mean = np.sqrt(u_mean ** 2 + v_mean ** 2)
    wd_mean = (270.0 - np.degrees(np.arctan2(v_mean, u_mean))) % 360.0
    T_mean_K = np.nanmean(TH * (PR / 100000.0) ** 0.2854, axis=0)
    P_mean_kPa = np.nanmean(PR / 1000.0, axis=0)
    n = len(selected)

    return pd.DataFrame({
        "离地高度 (m)": np.asarray(height_levels, dtype=float),
        "WRF原始层平均风速 (m/s)": ws_mean,
        "WRF原始层平均风向 (°)": wd_mean,
        "WRF原始层平均温度 (K)": T_mean_K,
        "WRF原始层平均温度 (°C)": T_mean_K - 273.15,
        "WRF原始层平均气压 (KPa)": P_mean_kPa,
        "WRF原始层平均_u (m/s)": u_mean,
        "WRF原始层平均_v (m/s)": v_mean,
        "样本数": n,
        "实际使用时间段开始 (北京)": avg_start,
        "实际使用时间段结束 (北京)": avg_end,
        "是否matched": False,
    }).sort_values("离地高度 (m)").reset_index(drop=True)


def compute_native_tke_period_mean(dfL, TKE_profiles, z_tke_agl, avg_start=None, avg_end=None):
    if avg_start is None or avg_end is None or dfL is None or len(dfL) == 0:
        return None
    mask = (dfL["time_bj"] >= avg_start) & (dfL["time_bj"] <= avg_end)
    if not mask.any():
        return None
    tke_mean = np.nanmean(TKE_profiles[mask.values, :], axis=0)
    return pd.DataFrame({
        "离地高度 (m)": np.asarray(z_tke_agl, dtype=float),
        "WRF原始层平均TKE (m2/s2)": tke_mean,
        "样本数": int(mask.sum()),
        "实际使用时间段开始 (北京)": avg_start,
        "实际使用时间段结束 (北京)": avg_end,
        "是否matched": False,
    }).sort_values("离地高度 (m)").reset_index(drop=True)


# ====================== wrfout processing ======================

def compute_MO_and_TKE_profiles_from_wrfout(wrfout_dir, domain_id, ix, iy):
    """
    返回:
    dfL, TKE_profiles, z_tke_agl, tke_name, l_diag
    关键修复：
    - 若 wrfout 中是 TKE_PBL(bottom_top_stag)，则高度使用 z_w_agl（W-stag）
    - 若是 TKE/QKE(bottom_top)，则高度使用 z_mass_agl
    - 额外返回 L 计算诊断，便于判断为什么 L 全空
    """
    kappa = 0.4
    g = 9.81
    Rd = 287.0
    cp = 1004.0
    tiny = 1.0e-12

    j = iy - 1
    i = ix - 1

    prefix = f"wrfout_{domain_id}_"
    file_list = sorted(f for f in os.listdir(wrfout_dir) if f.startswith(prefix))
    if not file_list:
        raise FileNotFoundError(f"未找到 wrfout 文件前缀: {prefix}")

    all_time_bj = []
    all_L = []
    all_PBLH = []
    all_TKE = []

    # Added: retain surface physical variables for ML-ready ref10 / MOST diagnostic output.
    all_UST = []
    all_HFX = []
    all_PSFC = []
    all_T2 = []
    all_Z0 = []

    z_tke_agl = None
    tke_name_global = None

    l_diag = {
        "total_steps": 0,
        "valid_L_steps": 0,
        "ust_le_zero": 0,
        "hfx_zero": 0,
        "hfx_nan": 0,
        "ust_nan": 0,
        "psfc_nan": 0,
        "t2_nan": 0,
        "wtheta_too_small": 0,
        "rho_invalid": 0,
    }

    for fname in file_list:
        path = os.path.join(wrfout_dir, fname)
        print("读取 wrfout:", path)

        with Dataset(path) as nc:
            Times = ["".join(t.astype(str)) for t in nc['Times'][:]]
            time_utc = [datetime.strptime(t, "%Y-%m-%d_%H:%M:%S") for t in Times]
            time_bj = [t + timedelta(hours=8) for t in time_utc]

            UST = np.asarray(nc['UST'][:, j, i], dtype=float)
            HFX = np.asarray(nc['HFX'][:, j, i], dtype=float)
            PSFC = np.asarray(nc['PSFC'][:, j, i], dtype=float)
            T2 = np.asarray(nc['T2'][:, j, i], dtype=float)
            PBLH = np.asarray(nc['PBLH'][:, j, i], dtype=float) if "PBLH" in nc.variables else np.full_like(UST, np.nan)

            # Added: roughness length. WRF commonly uses ZNT; some outputs may use Z0.
            # If unavailable, use Z0_FALLBACK and interpret the resulting MOST values cautiously in downstream ablations.
            if "ZNT" in nc.variables:
                Z0_arr = np.asarray(nc['ZNT'][:, j, i], dtype=float)
            elif "Z0" in nc.variables:
                Z0_arr = np.asarray(nc['Z0'][:, j, i], dtype=float)
            else:
                Z0_arr = np.full_like(UST, Z0_FALLBACK, dtype=float)

            tke_name = None
            for cand in ["TKE_PBL", "TKE", "QKE", "TKE_1", "TKE_2"]:
                if cand in nc.variables:
                    tke_name = cand
                    break
            if tke_name is None:
                raise ValueError("未找到 TKE 类变量 (TKE_PBL / TKE / QKE / TKE_1 / TKE_2)")
            if tke_name_global is None:
                tke_name_global = tke_name

            PH = np.asarray(nc['PH'][0, :, j, i], dtype=float)
            PHB = np.asarray(nc['PHB'][0, :, j, i], dtype=float)
            HGT = float(nc['HGT'][0, j, i])
            z_w = (PH + PHB) / g
            z_w_agl = z_w - HGT
            z_mass_agl = 0.5 * (z_w_agl[:-1] + z_w_agl[1:])

            if z_tke_agl is None:
                if tke_name == "TKE_PBL":
                    z_tke_agl = np.asarray(z_w_agl, dtype=float)
                else:
                    z_tke_agl = np.asarray(z_mass_agl, dtype=float)

            TKE_raw = np.asarray(nc[tke_name][:, :, j, i], dtype=float)
            if TKE_raw.ndim != 2:
                raise ValueError(f"{tke_name} 维度异常: {TKE_raw.shape}")

            if TKE_raw.shape[1] != len(z_tke_agl):
                if TKE_raw.shape[1] == len(z_mass_agl):
                    z_tke_agl = np.asarray(z_mass_agl, dtype=float)
                elif TKE_raw.shape[1] == len(z_w_agl):
                    z_tke_agl = np.asarray(z_w_agl, dtype=float)
                else:
                    raise ValueError(
                        f"{tke_name} 层数({TKE_raw.shape[1]})与可用高度长度不匹配: "
                        f"mass={len(z_mass_agl)}, w={len(z_w_agl)}"
                    )

            p0 = 100000.0
            theta = T2 * (p0 / PSFC) ** (Rd / cp)
            rho = PSFC / (Rd * T2)

            for k in range(len(time_bj)):
                l_diag["total_steps"] += 1

                ust_k = UST[k]
                hfx_k = HFX[k]
                psfc_k = PSFC[k]
                t2_k = T2[k]
                rho_k = rho[k]
                theta_k = theta[k]

                if np.isnan(ust_k):
                    l_diag["ust_nan"] += 1
                    L_val = np.nan
                elif ust_k <= 0:
                    l_diag["ust_le_zero"] += 1
                    L_val = np.nan
                elif np.isnan(hfx_k):
                    l_diag["hfx_nan"] += 1
                    L_val = np.nan
                elif hfx_k == 0:
                    l_diag["hfx_zero"] += 1
                    L_val = np.nan
                elif np.isnan(psfc_k):
                    l_diag["psfc_nan"] += 1
                    L_val = np.nan
                elif np.isnan(t2_k):
                    l_diag["t2_nan"] += 1
                    L_val = np.nan
                elif np.isnan(rho_k) or rho_k <= 0:
                    l_diag["rho_invalid"] += 1
                    L_val = np.nan
                else:
                    wtheta = hfx_k / (rho_k * cp)
                    if np.isnan(wtheta) or abs(wtheta) < tiny:
                        l_diag["wtheta_too_small"] += 1
                        L_val = np.nan
                    else:
                        L_val = -theta_k * (ust_k ** 3) / (kappa * g * wtheta)

                if pd.notna(L_val):
                    l_diag["valid_L_steps"] += 1

                all_time_bj.append(time_bj[k])
                all_L.append(L_val)
                all_PBLH.append(float(PBLH[k]) if np.isfinite(PBLH[k]) else np.nan)
                all_TKE.append(TKE_raw[k, :])

                # Added: save surface physical variables for later merging into result_df and export to ML-ready ref10.
                all_UST.append(float(UST[k]) if np.isfinite(UST[k]) else np.nan)
                all_HFX.append(float(HFX[k]) if np.isfinite(HFX[k]) else np.nan)
                all_PSFC.append(float(PSFC[k]) if np.isfinite(PSFC[k]) else np.nan)
                all_T2.append(float(T2[k]) if np.isfinite(T2[k]) else np.nan)
                all_Z0.append(float(Z0_arr[k]) if np.isfinite(Z0_arr[k]) else np.nan)

    TKE_profiles = np.vstack([np.asarray(t).reshape(1, -1) for t in all_TKE])
    df = pd.DataFrame({
        "time_bj": all_time_bj,
        "L (m)": all_L,
        "PBLH (m)": all_PBLH,
        "UST (m/s)": all_UST,
        "HFX (W/m2)": all_HFX,
        "PSFC (Pa)": all_PSFC,
        "T2 (K)": all_T2,
        "Z0 (m)": all_Z0,
    })
    return df, TKE_profiles, np.asarray(z_tke_agl, dtype=float), tke_name_global, l_diag



# ====================== ML-ready ref10 / MOST helper functions ======================

def psi_m_businger(zeta):
    """
    MOST momentum stability correction psi_m.
    稳定：psi_m = -5 zeta；不稳定：常用 Paulson/Businger-Dyer 形式。
    """
    zeta = np.asarray(zeta, dtype=float)
    psi = np.zeros_like(zeta, dtype=float)
    stable = zeta > 0
    unstable = zeta < 0
    psi[stable] = -5.0 * zeta[stable]
    if np.any(unstable):
        x = np.maximum(1.0 - 16.0 * zeta[unstable], 1e-6) ** 0.25
        psi[unstable] = (
            2.0 * np.log((1.0 + x) / 2.0)
            + np.log((1.0 + x**2) / 2.0)
            - 2.0 * np.arctan(x)
            + 0.5 * np.pi
        )
    return psi


def most_wind_from_ustar(z, ustar, z0, L, d=0.0, kappa=KAPPA):
    """
    用 u*, z0, L 计算 MOST 理论风速 U_MO(z)。
    这里的 U_MO 只作为候选物理参考量；是否进入 CNN_LSTM/PINN 由后续 ablation 决定。
    """
    z = np.asarray(z, dtype=float)
    ustar = np.asarray(ustar, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    L = np.asarray(L, dtype=float)
    d = np.asarray(d, dtype=float)

    z0 = np.maximum(z0, MOST_MIN_Z0)
    zeff = np.maximum(z - d, z0 * 1.01)

    neutral = (~np.isfinite(L)) | (np.abs(L) < 1e-6)
    zeta = np.where(neutral, 0.0, zeff / L)
    zeta0 = np.where(neutral, 0.0, z0 / L)
    psi = psi_m_businger(zeta) - psi_m_businger(zeta0)

    U = (ustar / kappa) * (np.log(zeff / z0) - psi)
    U = np.where(np.isfinite(U), U, np.nan)
    U = np.where((U >= 0.0) & (U <= MOST_MAX_REASONABLE_U), U, np.nan)
    return U


def add_most_profile_to_result(result_df):
    """在 result_df 中新增 U_MO_profile (m/s)，不改变原有列。"""
    if result_df is None or result_df.empty:
        return result_df
    if not EXPORT_MOST_THEORY:
        return result_df
    required = ["离地高度 (m)", "UST (m/s)", "Z0 (m)", "L (m)"]
    missing = [c for c in required if c not in result_df.columns]
    if missing:
        print(f"⚠ 缺少 MOST 计算所需列 {missing}，U_MO_profile 填充 NaN。")
        result_df["U_MO_profile (m/s)"] = np.nan
        return result_df

    z0 = pd.to_numeric(result_df["Z0 (m)"], errors="coerce")
    d_eff = MOST_D_FACTOR * z0
    result_df["U_MO_profile (m/s)"] = most_wind_from_ustar(
        z=pd.to_numeric(result_df["离地高度 (m)"], errors="coerce").values,
        ustar=pd.to_numeric(result_df["UST (m/s)"], errors="coerce").values,
        z0=z0.values,
        L=pd.to_numeric(result_df["L (m)"], errors="coerce").values,
        d=d_eff.values,
    )
    return result_df


def _cyclic_time_features(df, time_col="北京时间"):
    t = pd.to_datetime(df[time_col], errors="coerce")
    hour = t.dt.hour + t.dt.minute / 60.0
    month = t.dt.month
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    return df


def _height_label(h):
    """Return a compact height label used in ML-ready column names, e.g. 40 or 40p5."""
    h = float(h)
    if np.isclose(h, round(h), atol=1e-6):
        return str(int(round(h)))
    return (f"{h:.3f}".rstrip("0").rstrip(".")).replace(".", "p")


def _extract_obs_ref10(obs_csv, ref_height=REF_HEIGHT_ML):
    """
    Extract reference-height observation columns for single-height training.

    V15 compatibility additions are additive only:
      - keep the original OBS_TKE10_from_TI column;
      - also provide OBS_TKE10 (m2/s2), which Code 1/2 target detection expects;
      - provide OBS_T10 (K) / OBS_TEMP10 (K) if tp10m is available.
    """
    obs_wide = _extract_obs_profile_wide(obs_csv, [ref_height])
    h_label = _height_label(ref_height)
    rename = {}
    for c in obs_wide.columns:
        rename[c] = c.replace(f"{h_label}", "10") if c != "北京时间" else c
    out = obs_wide.rename(columns=rename)

    # Keep original columns and add standard aliases expected by Code 1-4.
    if "OBS_TKE10_from_TI (m2/s2)" in out.columns and "OBS_TKE10 (m2/s2)" not in out.columns:
        out["OBS_TKE10 (m2/s2)"] = out["OBS_TKE10_from_TI (m2/s2)"]
    if "OBS_TEMP10 (K)" in out.columns and "OBS_T10 (K)" not in out.columns:
        out["OBS_T10 (K)"] = out["OBS_TEMP10 (K)"]
    if "OBS_T10 (K)" in out.columns and "OBS_TEMP10 (K)" not in out.columns:
        out["OBS_TEMP10 (K)"] = out["OBS_T10 (K)"]

    # Guarantee core columns exist, so downstream code can run and simply mask missing targets.
    core_cols = [
        "OBS_WS10 (m/s)", "OBS_WD10 (deg)",
        "OBS_TKE10_from_TI (m2/s2)", "OBS_TKE10 (m2/s2)",
        "OBS_T10 (K)", "OBS_TEMP10 (K)",
    ]
    for c in core_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[["北京时间"] + core_cols]


def _extract_obs_profile_wide(obs_csv, heights):
    """
    从观测 CSV 中提取多高度观测风速/风向/TI-derived TKE，输出宽表。
    只为观测文件中真实存在的高度创建 OBS_* 列，避免多高度训练误把全 NaN 高度当作有效 target。

    V16 hurui case addition:
    - Windographer converted CSV can provide direct tke{height}m from speed SD.
    - Direct TKE is used as OBS_TKE{height}; TI-derived TKE is still kept as _from_TI when possible.
    """
    obs_df = load_obs_dataframe(obs_csv)
    out = pd.DataFrame({"北京时间": pd.to_datetime(obs_df["rec_time"], errors="coerce")})

    for h in sorted(set(float(x) for x in heights)):
        h_int = int(round(h))
        h_label = _height_label(h)
        ws_col = f"wspd{h_int}m"
        wd_col = f"wdir{h_int}m"
        ti_col = f"tbl{h_int}m"
        tp_col = f"tp{h_int}m"
        tke_col = f"tke{h_int}m"

        has_ws = ws_col in obs_df.columns
        has_wd = wd_col in obs_df.columns
        has_ti = ti_col in obs_df.columns
        has_tp = tp_col in obs_df.columns
        has_tke_direct = tke_col in obs_df.columns

        if has_ws:
            out[f"OBS_WS{h_label} (m/s)"] = pd.to_numeric(obs_df[ws_col], errors="coerce")
        if has_wd:
            out[f"OBS_WD{h_label} (deg)"] = pd.to_numeric(obs_df[wd_col], errors="coerce")

        if has_tke_direct:
            out[f"OBS_TKE{h_label} (m2/s2)"] = pd.to_numeric(obs_df[tke_col], errors="coerce")

        if has_ws and has_ti:
            ws = pd.to_numeric(obs_df[ws_col], errors="coerce")
            ti = pd.to_numeric(obs_df[ti_col], errors="coerce")
            tke_from_ti = 1.5 * (ti * ws) ** 2
            out[f"OBS_TKE{h_label}_from_TI (m2/s2)"] = tke_from_ti
            # Standard alias expected by Code 1-4. If direct TKE exists, keep it as the main target.
            if not has_tke_direct:
                out[f"OBS_TKE{h_label} (m2/s2)"] = tke_from_ti

        if has_tp:
            # Observation tp is configured/cleaned in Celsius; export Kelvin for ML targets.
            temp_c = pd.to_numeric(obs_df[tp_col], errors="coerce")
            out[f"OBS_T{h_label} (K)"] = temp_c + 273.15
            out[f"OBS_TEMP{h_label} (K)"] = temp_c + 273.15

    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["北京时间"]).sort_values("北京时间")

def export_ml_ready_ref10(result_df, obs_csv, output_dir, site_tag, domain_id,
                          ref_height=REF_HEIGHT_ML, tolerance=ML_MERGE_TOLERANCE):
    """
    导出 CNN_LSTM/PINN 用 ML-ready 表。

    兼容两类后续代码：
    1) 单高度 10 m bias-correction 代码：保留旧列名，例如 WRF_WS10_interp、OBS_WS10、U_MO10、target_delta_U10。
    2) 多高度 profile bias-correction 代码：若 EXPORT_ML_READY_PROFILE_WIDE=True，额外加入
       OBS_WS40/WRF_WS40_interp/U_MO40/MO_minus_WRF40/target_delta_U40 等多高度宽表列。

    输出文件名仍为 ML_ready_ref10_{site_tag}_{domain_id}.csv，避免破坏原 10 m 脚本的读取逻辑。
    """
    if result_df is None or result_df.empty:
        return None

    df = result_df.copy()
    df["北京时间"] = pd.to_datetime(df["北京时间"], errors="coerce")
    df["离地高度 (m)"] = pd.to_numeric(df["离地高度 (m)"], errors="coerce")
    df = df.dropna(subset=["北京时间", "离地高度 (m)"])

    h_all = sorted(df["离地高度 (m)"].dropna().unique().astype(float))
    if ML_READY_PROFILE_HEIGHTS is not None:
        wanted = [float(x) for x in ML_READY_PROFILE_HEIGHTS]
        h_all = [h for h in h_all if any(np.isclose(h, w, atol=1e-6) for w in wanted)]
    if float(ref_height) not in h_all and any(np.isclose(df["离地高度 (m)"], float(ref_height), atol=1e-6)):
        h_all = sorted(set(h_all + [float(ref_height)]))

    href_mask = np.isclose(df["离地高度 (m)"], float(ref_height), atol=1e-6)
    wrf_ref = df[href_mask].copy()
    if wrf_ref.empty:
        print(f"[{site_tag}] ⚠ result_df 中没有 {ref_height} m 行，无法导出 ML-ready ref10/profile。")
        return None

    wrf_ref = wrf_ref.sort_values("北京时间")
    keep_cols = [
        "北京时间", "站点标签", "经度 (°)", "纬度 (°)", "海拔高度 (m)",
        "WRF嵌套域", "离地高度 (m)", "提取方式", "最近WRF层高度 (m)", "最近层高度差 (m)",
        "WRF下层高度 (m)", "WRF上层高度 (m)",
        "十分钟平均风速 (m/s)", "十分钟平均风向 (°)", "平均_u (m/s)", "平均_v (m/s)",
        "十分钟平均温度 (K)", "十分钟平均温度 (°C)", "十分钟平均气压 (KPa)",
        "TKE (m2/s2)", "L (m)", "PBLH (m)", "Bulk_Ri", "z_over_L",
        "Stability_Ri", "Stability_L", "Stability",
        "UST (m/s)", "HFX (W/m2)", "PSFC (Pa)", "T2 (K)", "Z0 (m)",
        "U_MO_profile (m/s)",
    ]
    keep_cols = [c for c in keep_cols if c in wrf_ref.columns]
    ml = wrf_ref[keep_cols].copy()

    # Column names required by the legacy 10 m code: keep full compatibility.
    rename_map = {
        "十分钟平均风速 (m/s)": "WRF_WS10_interp (m/s)",
        "十分钟平均风向 (°)": "WRF_WD10_interp (deg)",
        "平均_u (m/s)": "WRF_U10_interp (m/s)",
        "平均_v (m/s)": "WRF_V10_interp (m/s)",
        "十分钟平均温度 (K)": "WRF_T10_interp (K)",
        "TKE (m2/s2)": "WRF_TKE10 (m2/s2)",
        "L (m)": "WRF_L (m)",
        "PBLH (m)": "WRF_PBLH (m)",
        "Bulk_Ri": "WRF_Bulk_Ri",
        "U_MO_profile (m/s)": "U_MO10 (m/s)",
    }
    ml = ml.rename(columns=rename_map)
    # V15 compatibility aliases for single-height temperature detection.
    if "WRF_T10_interp (K)" in ml.columns and "WRF_T10 (K)" not in ml.columns:
        ml["WRF_T10 (K)"] = ml["WRF_T10_interp (K)"]
    if "WRF_T10_interp (K)" in ml.columns and "WRF_TEMP10 (K)" not in ml.columns:
        ml["WRF_TEMP10 (K)"] = ml["WRF_T10_interp (K)"]

    # Multi-height WRF/MOST profile wide-format columns: the same CSV can be used by both single-height and profile models.
    if EXPORT_ML_READY_PROFILE_WIDE:
        for h in h_all:
            h_label = _height_label(h)
            sub = df[np.isclose(df["离地高度 (m)"], h, atol=1e-6)].copy().sort_values("北京时间")
            if sub.empty:
                continue
            cols_to_take = ["北京时间"]
            col_map = {}
            raw_to_new = {
                "十分钟平均风速 (m/s)": f"WRF_WS{h_label}_interp (m/s)",
                "十分钟平均风向 (°)": f"WRF_WD{h_label}_interp (deg)",
                "平均_u (m/s)": f"WRF_U{h_label}_interp (m/s)",
                "平均_v (m/s)": f"WRF_V{h_label}_interp (m/s)",
                "十分钟平均温度 (K)": f"WRF_T{h_label}_interp (K)",
                "TKE (m2/s2)": f"WRF_TKE{h_label} (m2/s2)",
                "U_MO_profile (m/s)": f"U_MO{h_label} (m/s)",
            }
            for old_c, new_c in raw_to_new.items():
                if old_c in sub.columns:
                    cols_to_take.append(old_c)
                    col_map[old_c] = new_c
            if len(cols_to_take) <= 1:
                continue
            tmp = sub[cols_to_take].rename(columns=col_map)
            ml = pd.merge(ml, tmp, on="北京时间", how="left", suffixes=("", "_dup"))
            dup_cols = [c for c in ml.columns if c.endswith("_dup")]
            if dup_cols:
                ml = ml.drop(columns=dup_cols)

            wd_col = f"WRF_WD{h_label}_interp (deg)"
            if wd_col in ml.columns:
                wd = pd.to_numeric(ml[wd_col], errors="coerce")
                ml[f"WRF_WD{h_label}_sin"] = np.sin(np.deg2rad(wd))
                ml[f"WRF_WD{h_label}_cos"] = np.cos(np.deg2rad(wd))

            u_mo_col = f"U_MO{h_label} (m/s)"
            wrf_ws_col = f"WRF_WS{h_label}_interp (m/s)"
            if u_mo_col in ml.columns and wrf_ws_col in ml.columns:
                ml[f"MO_minus_WRF{h_label} (m/s)"] = (
                    pd.to_numeric(ml[u_mo_col], errors="coerce")
                    - pd.to_numeric(ml[wrf_ws_col], errors="coerce")
                )
            wrf_t_col = f"WRF_T{h_label}_interp (K)"
            if wrf_t_col in ml.columns:
                ml[f"WRF_TEMP{h_label}_interp (K)"] = ml[wrf_t_col]

    # Merge the wide-format observation table. Create only OBS height columns that actually exist.
    obs_heights = h_all if EXPORT_ML_READY_PROFILE_WIDE else [ref_height]
    obs_wide = _extract_obs_profile_wide(obs_csv, obs_heights)
    obs_wide["北京时间"] = pd.to_datetime(obs_wide["北京时间"], errors="coerce")
    obs_wide = obs_wide.dropna(subset=["北京时间"]).sort_values("北京时间")

    ml = pd.merge_asof(
        ml.sort_values("北京时间"),
        obs_wide.sort_values("北京时间"),
        on="北京时间",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    # OBS wind-direction sin/cos, target residual, and obs_valid_flag. Legacy 10 m column names are also retained.
    target_heights_for_flags = h_all if EXPORT_ML_READY_PROFILE_WIDE else [ref_height]
    for h in target_heights_for_flags:
        h_label = _height_label(h)
        obs_ws_col = f"OBS_WS{h_label} (m/s)"
        obs_wd_col = f"OBS_WD{h_label} (deg)"
        wrf_ws_col = f"WRF_WS{h_label}_interp (m/s)"

        if obs_wd_col in ml.columns:
            wd_obs = pd.to_numeric(ml[obs_wd_col], errors="coerce")
            ml[f"OBS_WD{h_label}_sin"] = np.sin(np.deg2rad(wd_obs))
            ml[f"OBS_WD{h_label}_cos"] = np.cos(np.deg2rad(wd_obs))

        if obs_ws_col in ml.columns and wrf_ws_col in ml.columns:
            ws_residual = (
                pd.to_numeric(ml[obs_ws_col], errors="coerce")
                - pd.to_numeric(ml[wrf_ws_col], errors="coerce")
            )
            # Original legacy column is kept for backward compatibility, although it is a WS residual.
            ml[f"target_delta_U{h_label}_obs_minus_wrf (m/s)"] = ws_residual
            # New clearer column name.
            ml[f"target_delta_WS{h_label}_obs_minus_wrf (m/s)"] = ws_residual
            ml[f"obs_valid_flag_{h_label}"] = pd.to_numeric(ml[obs_ws_col], errors="coerce").notna().astype(int)

        tke_from_ti_col = f"OBS_TKE{h_label}_from_TI (m2/s2)"
        tke_std_col = f"OBS_TKE{h_label} (m2/s2)"
        if tke_from_ti_col in ml.columns and tke_std_col not in ml.columns:
            ml[tke_std_col] = ml[tke_from_ti_col]
        temp_col = f"OBS_T{h_label} (K)"
        temp_alias_col = f"OBS_TEMP{h_label} (K)"
        if temp_col in ml.columns and temp_alias_col not in ml.columns:
            ml[temp_alias_col] = ml[temp_col]
        if temp_alias_col in ml.columns and temp_col not in ml.columns:
            ml[temp_col] = ml[temp_alias_col]

    # Compatibility with the single-height 10 m script: ensure the core columns exist.
    for c in ["OBS_WS10 (m/s)", "OBS_WD10 (deg)", "OBS_TKE10_from_TI (m2/s2)", "OBS_TKE10 (m2/s2)", "OBS_T10 (K)", "OBS_TEMP10 (K)"]:
        if c not in ml.columns:
            ml[c] = np.nan
    if "OBS_TKE10_from_TI (m2/s2)" in ml.columns and "OBS_TKE10 (m2/s2)" in ml.columns:
        ml["OBS_TKE10 (m2/s2)"] = ml["OBS_TKE10 (m2/s2)"].where(ml["OBS_TKE10 (m2/s2)"].notna(), ml["OBS_TKE10_from_TI (m2/s2)"])
    if "OBS_T10 (K)" in ml.columns and "OBS_TEMP10 (K)" in ml.columns:
        ml["OBS_T10 (K)"] = ml["OBS_T10 (K)"].where(ml["OBS_T10 (K)"].notna(), ml["OBS_TEMP10 (K)"])

    if "WRF_WD10_interp (deg)" in ml.columns:
        wd = pd.to_numeric(ml["WRF_WD10_interp (deg)"], errors="coerce")
        ml["WRF_WD10_sin"] = np.sin(np.deg2rad(wd))
        ml["WRF_WD10_cos"] = np.cos(np.deg2rad(wd))

    if "OBS_WD10 (deg)" in ml.columns:
        wd_obs = pd.to_numeric(ml["OBS_WD10 (deg)"], errors="coerce")
        ml["OBS_WD10_sin"] = np.sin(np.deg2rad(wd_obs))
        ml["OBS_WD10_cos"] = np.cos(np.deg2rad(wd_obs))

    if "U_MO10 (m/s)" in ml.columns and "WRF_WS10_interp (m/s)" in ml.columns:
        ml["MO_minus_WRF10 (m/s)"] = (
            pd.to_numeric(ml["U_MO10 (m/s)"], errors="coerce")
            - pd.to_numeric(ml["WRF_WS10_interp (m/s)"], errors="coerce")
        )

    if "OBS_WS10 (m/s)" in ml.columns and "WRF_WS10_interp (m/s)" in ml.columns:
        ws10_residual = (
            pd.to_numeric(ml["OBS_WS10 (m/s)"], errors="coerce")
            - pd.to_numeric(ml["WRF_WS10_interp (m/s)"], errors="coerce")
        )
        # Original legacy column is kept; clearer WS residual column is added.
        ml["target_delta_U10_obs_minus_wrf (m/s)"] = ws10_residual
        ml["target_delta_WS10_obs_minus_wrf (m/s)"] = ws10_residual

    ml["obs_valid_flag"] = pd.to_numeric(ml["OBS_WS10 (m/s)"], errors="coerce").notna().astype(int)

    # Whether the multi-height profile is fully valid. This column is optional downstream but useful for diagnostics.
    profile_target_cols = [c for c in ml.columns if c.startswith("target_delta_U") and c.endswith("_obs_minus_wrf (m/s)")]
    if profile_target_cols:
        ml["obs_profile_valid_flag_all_detected_heights"] = ml[profile_target_cols].notna().all(axis=1).astype(int)
        ml["obs_profile_valid_count_detected_heights"] = ml[profile_target_cols].notna().sum(axis=1)

    ml = _cyclic_time_features(ml, time_col="北京时间")

    out_csv = os.path.join(output_dir, f"ML_ready_ref10_{site_tag}_{domain_id}.csv")
    ml.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if EXPORT_ML_READY_PROFILE_WIDE:
        print(f"[{site_tag}] ✅ ML-ready 10 m + 多高度宽表已保存: {out_csv}")
    else:
        print(f"[{site_tag}] ✅ ML-ready 10 m 表已保存: {out_csv}")
    return ml


# ====================== Plotting and export functions ======================

def filter_profile_by_height(df, zcol, zmax=None):
    if df is None or df.empty:
        return df
    if zmax is None:
        return df.copy()
    return df[df[zcol] <= zmax].copy()


def plot_obs_vs_wrf_profile_points(wrf_df, obs_df, wrf_x_col, obs_x_col, output_dir,
                                   fig_name, xlabel, title, zmax=None, wrf_aligned_df=None, wrf_aligned_x_col=None):
    if (wrf_df is None or wrf_df.empty) and (obs_df is None or obs_df.empty):
        print(f"{fig_name}: WRF 与 obs 都为空，跳过。")
        return

    os.makedirs(output_dir, exist_ok=True)
    wrf_plot = filter_profile_by_height(wrf_df, "离地高度 (m)", zmax=zmax) if wrf_df is not None else None
    obs_plot = filter_profile_by_height(obs_df, "height", zmax=zmax) if obs_df is not None else None

    plt.figure(figsize=(6, 8 if zmax is None else 6))

    if wrf_plot is not None and not wrf_plot.empty and wrf_x_col in wrf_plot.columns:
        plt.scatter(wrf_plot[wrf_x_col].values, wrf_plot["离地高度 (m)"].values, s=24, alpha=0.7, label="WRF native levels")

    if wrf_aligned_df is not None and not wrf_aligned_df.empty and wrf_aligned_x_col in wrf_aligned_df.columns:
        wrf_aligned_plot = filter_profile_by_height(wrf_aligned_df, "离地高度 (m)", zmax=zmax)
        if wrf_aligned_plot is not None and not wrf_aligned_plot.empty:
            plt.scatter(wrf_aligned_plot[wrf_aligned_x_col].values,
                        wrf_aligned_plot["离地高度 (m)"].values,
                        s=46, marker="o", facecolors="none", linewidths=1.2,
                        label="WRF aligned@obs heights")

    if obs_plot is not None and not obs_plot.empty and obs_x_col in obs_plot.columns:
        for inst, group in obs_plot.groupby("instrument"):
            plt.scatter(group[obs_x_col].values, group["height"].values, s=45, marker="s", label=f"{inst} obs")

    plt.xlabel(xlabel)
    plt.ylabel("Height (m)")
    plt.title(title if zmax is None else f"{title} (0–{int(zmax)} m)")
    if zmax is not None:
        plt.ylim(0.0, zmax)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"已保存: {fig_path}")




def _normalize_timeseries_for_asof(df, time_col="北京时间", height_col=None, value_cols=None):
    """
    统一 merge_asof 的键类型，避免 object / datetime64 / mixed 类型导致的
    `Function call with ambiguous argument types`。
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    # Remove timezone information when present and standardize to naive datetime64[ns].
    try:
        if getattr(out[time_col].dt, 'tz', None) is not None:
            out[time_col] = out[time_col].dt.tz_localize(None)
    except Exception:
        pass

    if height_col is not None and height_col in out.columns:
        out[height_col] = pd.to_numeric(out[height_col], errors="coerce").astype(float)

    if value_cols is not None:
        for c in value_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

    subset = [time_col]
    if height_col is not None and height_col in out.columns:
        subset.append(height_col)
    out = out.dropna(subset=subset).sort_values(subset).reset_index(drop=True)
    return out

def _merge_asof_on_time_ns(left_df, right_df, left_time_col, right_time_col, tolerance=OBS_WRF_MATCH_TOLERANCE, right_keep_cols=None):
    """用 int64 纳秒时间戳做 merge_asof，避免 datetime/by 类型歧义。"""
    if left_df is None or left_df.empty or right_df is None or right_df.empty:
        return pd.DataFrame()

    left = left_df.copy()
    right = right_df.copy()

    left[left_time_col] = pd.to_datetime(left[left_time_col], errors="coerce")
    right[right_time_col] = pd.to_datetime(right[right_time_col], errors="coerce")

    left = left.dropna(subset=[left_time_col]).sort_values(left_time_col).reset_index(drop=True)
    right = right.dropna(subset=[right_time_col]).sort_values(right_time_col).reset_index(drop=True)
    if left.empty or right.empty:
        return pd.DataFrame()

    left["_time_ns"] = left[left_time_col].astype("int64")
    right["_time_ns"] = right[right_time_col].astype("int64")
    right["_matched_right_time"] = right[right_time_col]

    cols = ["_time_ns", "_matched_right_time"]
    if right_keep_cols:
        cols += [c for c in right_keep_cols if c in right.columns]

    merged = pd.merge_asof(
        left.sort_values("_time_ns"),
        right[cols].sort_values("_time_ns"),
        on="_time_ns",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance).value
    )
    merged[left_time_col] = pd.to_datetime(merged["_time_ns"], unit="ns", errors="coerce")
    if "_matched_right_time" in merged.columns:
        merged["WRF匹配时间 (北京)"] = pd.to_datetime(merged["_matched_right_time"], errors="coerce")
        merged["时间差 (min)"] = (merged[left_time_col] - merged["WRF匹配时间 (北京)"]).abs().dt.total_seconds() / 60.0
        merged = merged.drop(columns=["_matched_right_time"])
    return merged.drop(columns=["_time_ns"])


def merge_obs_wrf_timeseries(obs_ts, wrf_ts, obs_col, wrf_col, tolerance=OBS_WRF_MATCH_TOLERANCE):
    if obs_ts is None or obs_ts.empty or wrf_ts is None or wrf_ts.empty:
        return pd.DataFrame()

    obs_ts = _normalize_timeseries_for_asof(
        obs_ts, time_col="北京时间", height_col="离地高度 (m)", value_cols=[obs_col]
    )
    wrf_ts = _normalize_timeseries_for_asof(
        wrf_ts, time_col="北京时间", height_col="离地高度 (m)", value_cols=[wrf_col]
    )
    if obs_ts.empty or wrf_ts.empty:
        return pd.DataFrame()

    obs_heights = set(np.round(obs_ts["离地高度 (m)"].dropna().astype(float).values, 6))
    wrf_heights = set(np.round(wrf_ts["离地高度 (m)"].dropna().astype(float).values, 6))
    common_heights = sorted(obs_heights.intersection(wrf_heights))

    out = []
    for h in common_heights:
        sub_obs = obs_ts[np.isclose(obs_ts["离地高度 (m)"].astype(float), float(h), atol=1e-6)].copy()
        sub_wrf = wrf_ts[np.isclose(wrf_ts["离地高度 (m)"].astype(float), float(h), atol=1e-6)].copy()
        if sub_obs.empty or sub_wrf.empty:
            continue

        sub_obs = sub_obs[["北京时间", "离地高度 (m)", obs_col]].dropna(subset=["北京时间", obs_col]).copy()
        sub_wrf = sub_wrf[["北京时间", wrf_col]].dropna(subset=["北京时间", wrf_col]).copy()
        if sub_obs.empty or sub_wrf.empty:
            continue

        merged = _merge_asof_on_time_ns(
            sub_obs,
            sub_wrf,
            left_time_col="北京时间",
            right_time_col="北京时间",
            tolerance=tolerance,
            right_keep_cols=[wrf_col]
        )
        if merged.empty:
            continue
        merged["离地高度 (m)"] = float(h)
        out.append(merged[["北京时间", "离地高度 (m)", obs_col, wrf_col]])

    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame()


def merge_obs_wrf_scalar_timeseries(obs_ts, wrf_ts, obs_col, wrf_col, tolerance=OBS_WRF_MATCH_TOLERANCE):
    if obs_ts is None or obs_ts.empty or wrf_ts is None or wrf_ts.empty:
        return pd.DataFrame()

    sub_obs = _normalize_timeseries_for_asof(obs_ts, time_col="北京时间", value_cols=[obs_col])
    sub_wrf = _normalize_timeseries_for_asof(wrf_ts[["北京时间", wrf_col]].copy(), time_col="北京时间", value_cols=[wrf_col])
    if sub_obs.empty or sub_wrf.empty:
        return pd.DataFrame()

    return _merge_asof_on_time_ns(
        sub_obs,
        sub_wrf,
        left_time_col="北京时间",
        right_time_col="北京时间",
        tolerance=tolerance,
        right_keep_cols=[wrf_col]
    )


def plot_scalar_timeseries_comparison_dual(obs_ts, wrf_ts, obs_col, wrf_col, output_dir, fig_name, y_label, title, gap="60min"):
    if obs_ts is None or obs_ts.empty or wrf_ts is None or wrf_ts.empty:
        print(f"{fig_name}: obs 或 WRF 标量时序为空，跳过。")
        return
    os.makedirs(output_dir, exist_ok=True)

    obs_ts = _normalize_timeseries_for_asof(obs_ts, time_col="北京时间", value_cols=[obs_col])
    wrf_ts = _normalize_timeseries_for_asof(wrf_ts, time_col="北京时间", value_cols=[wrf_col])
    if obs_ts.empty or wrf_ts.empty:
        print(f"{fig_name}: 归一化后 obs 或 WRF 标量时序为空，跳过。")
        return

    obs_df = _insert_time_gap_nans(obs_ts, "北京时间", obs_col, max_gap=gap)
    wrf_df = _insert_time_gap_nans(wrf_ts, "北京时间", wrf_col, max_gap=gap)

    plt.figure(figsize=(10, 4))
    plt.plot(wrf_df["北京时间"], wrf_df[wrf_col], label="WRF", linewidth=1.0)
    plt.scatter(obs_ts["北京时间"], obs_ts[obs_col], label="Obs", s=8, alpha=0.8)
    plt.plot(obs_df["北京时间"], obs_df[obs_col], linewidth=0.6, alpha=0.6)
    plt.xlabel("Beijing Time")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"已保存: {fig_path}")


def print_wrf_L_diagnostics(dfL, l_diag, site_tag=None):
    tag = f"[{site_tag}] " if site_tag else ""
    if dfL is None or dfL.empty:
        print(f"{tag}WRF L 诊断: dfL 为空。")
        return

    total_steps = int(l_diag.get("total_steps", len(dfL))) if isinstance(l_diag, dict) else len(dfL)
    valid_steps = int(l_diag.get("valid_L_steps", int(dfL["L (m)"].notna().sum()))) if isinstance(l_diag, dict) else int(dfL["L (m)"].notna().sum())
    finite_vals = pd.to_numeric(dfL["L (m)"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    print(f"{tag}WRF L 诊断: 总时次={total_steps}, 有效L={valid_steps}, 缺失L={total_steps - valid_steps}")
    if not finite_vals.empty:
        print(f"{tag}WRF L 统计: min={finite_vals.min():.6f}, max={finite_vals.max():.6f}, mean={finite_vals.mean():.6f}")
    else:
        print(f"{tag}⚠ WRF L 全为空。可能原因如下：")
        if isinstance(l_diag, dict):
            print(f"{tag}  UST<=0 次数      : {l_diag.get('ust_le_zero', 0)}")
            print(f"{tag}  UST为NaN 次数    : {l_diag.get('ust_nan', 0)}")
            print(f"{tag}  HFX==0 次数      : {l_diag.get('hfx_zero', 0)}")
            print(f"{tag}  HFX为NaN 次数    : {l_diag.get('hfx_nan', 0)}")
            print(f"{tag}  PSFC为NaN 次数   : {l_diag.get('psfc_nan', 0)}")
            print(f"{tag}  T2为NaN 次数     : {l_diag.get('t2_nan', 0)}")
            print(f"{tag}  rho无效 次数     : {l_diag.get('rho_invalid', 0)}")
            print(f"{tag}  wtheta过小 次数  : {l_diag.get('wtheta_too_small', 0)}")
        print(f"{tag}  当前代码中，UST<=0、HFX==0、HFX/UST缺失、wtheta过小都会令 L 记为 NaN。")


def _prepare_wrapped_direction_for_plot(series_deg, wrap_threshold=180.0):
    """
    将 0-360° 风向序列整理为更适合绘图的形式：
    1) 统一映射到 [0, 360)
    2) 当相邻时刻跨越 0/360 且跳变过大时插入 NaN，避免折线横跨整张图
    3) 跳过 NaN / inf，避免 np.mod 触发 RuntimeWarning
    """
    vals = pd.to_numeric(pd.Series(series_deg), errors="coerce").to_numpy(dtype=float)

    # Apply modulo only to finite values; keep NaN / inf as NaN.
    finite_mask = np.isfinite(vals)
    out = vals.copy()
    out[finite_mask] = np.mod(out[finite_mask], 360.0)
    out[~finite_mask] = np.nan

    if out.size <= 1:
        return out

    for i in range(1, len(out)):
        if np.isnan(out[i]) or np.isnan(out[i - 1]):
            continue
        raw_diff = abs(out[i] - out[i - 1])
        circ_diff = min(raw_diff, 360.0 - raw_diff)
        if raw_diff > wrap_threshold and circ_diff < (360.0 - wrap_threshold):
            out[i] = np.nan
    return out


def build_aligned_profile_means_from_compare(compare_df, obs_col, wrf_col, instrument_name=None):
    """保留旧接口，但内部调用新的 matched summary 构造。"""
    _summary_df, obs_df, wrf_df = build_matched_profile_summary(compare_df, obs_col, wrf_col, instrument_name=instrument_name)
    return obs_df, wrf_df


def build_matched_profile_summary(compare_df, obs_col, wrf_col, instrument_name=None):
    """由按时间配对后的 compare_df 生成统一口径的最终廓线统计表。"""
    if compare_df is None or compare_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows = []
    obs_rows = []
    wrf_rows = []
    inst = instrument_name if instrument_name else "Obs"

    for h, grp in compare_df.groupby("离地高度 (m)"):
        obs_s = pd.to_numeric(grp[obs_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        wrf_s = pd.to_numeric(grp[wrf_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pair_mask = obs_s.notna() & wrf_s.notna()
        if not pair_mask.any():
            continue
        grp_valid = grp.loc[pair_mask].copy()
        obs_valid = obs_s.loc[pair_mask]
        wrf_valid = wrf_s.loc[pair_mask]
        valid_n = int(pair_mask.sum())
        t_start = pd.to_datetime(grp_valid["北京时间"], errors="coerce").min()
        t_end = pd.to_datetime(grp_valid["北京时间"], errors="coerce").max()

        rows.append({
            "instrument": inst,
            "height": float(h),
            obs_col: float(obs_valid.mean()),
            wrf_col: float(wrf_valid.mean()),
            "样本数": valid_n,
            "实际使用时间段开始 (北京)": t_start,
            "实际使用时间段结束 (北京)": t_end,
            "是否matched": True,
        })
        obs_rows.append({
            "instrument": inst,
            "height": float(h),
            obs_col: float(obs_valid.mean()),
            "样本数": valid_n,
            "实际使用时间段开始 (北京)": t_start,
            "实际使用时间段结束 (北京)": t_end,
            "是否matched": True,
        })
        wrf_rows.append({
            "离地高度 (m)": float(h),
            wrf_col: float(wrf_valid.mean()),
            "样本数": valid_n,
            "实际使用时间段开始 (北京)": t_start,
            "实际使用时间段结束 (北京)": t_end,
            "是否matched": True,
        })

    summary_df = pd.DataFrame(rows).sort_values("height").reset_index(drop=True) if rows else pd.DataFrame()
    obs_df = pd.DataFrame(obs_rows).sort_values("height").reset_index(drop=True) if obs_rows else pd.DataFrame()
    wrf_df = pd.DataFrame(wrf_rows).sort_values("离地高度 (m)").reset_index(drop=True) if wrf_rows else pd.DataFrame()
    return summary_df, obs_df, wrf_df


def build_full_timeseries_export(obs_ts, wrf_ts, obs_col, wrf_col, is_scalar=False):
    if (obs_ts is None or obs_ts.empty) and (wrf_ts is None or wrf_ts.empty):
        return pd.DataFrame()

    obs_norm = _normalize_timeseries_for_asof(obs_ts, time_col="北京时间", height_col=None if is_scalar else "离地高度 (m)", value_cols=[obs_col]) if obs_ts is not None and not obs_ts.empty else pd.DataFrame()
    wrf_norm = _normalize_timeseries_for_asof(wrf_ts, time_col="北京时间", height_col=None if is_scalar else "离地高度 (m)", value_cols=[wrf_col]) if wrf_ts is not None and not wrf_ts.empty else pd.DataFrame()

    parts = []
    base_cols = ["北京时间"] + ([] if is_scalar else ["离地高度 (m)"])
    if not obs_norm.empty:
        obs_part = obs_norm[base_cols + [obs_col]].copy()
        obs_part[wrf_col] = np.nan
        obs_part["source"] = "Obs"
        obs_part["是否matched"] = False
        parts.append(obs_part)
    if not wrf_norm.empty:
        wrf_part = wrf_norm[base_cols + [wrf_col]].copy()
        wrf_part[obs_col] = np.nan
        wrf_part["source"] = "WRF"
        wrf_part["是否matched"] = False
        parts.append(wrf_part)
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True, sort=False)
    group_cols = ["source"] + ([] if is_scalar else ["离地高度 (m)"])
    meta = out.groupby(group_cols, dropna=False).agg(
        样本数=("北京时间", "count"),
        **{"实际使用时间段开始 (北京)": ("北京时间", "min"),
           "实际使用时间段结束 (北京)": ("北京时间", "max")}
    ).reset_index()
    out = out.merge(meta, on=group_cols, how="left")
    sort_cols = base_cols + ["source"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_matched_timeseries_export(compare_df, obs_col, wrf_col, is_scalar=False):
    if compare_df is None or compare_df.empty:
        return pd.DataFrame()
    out = compare_df.copy()
    out["是否matched"] = True
    group_cols = [] if is_scalar else ["离地高度 (m)"]
    if group_cols:
        meta = out.groupby(group_cols, dropna=False).agg(
            样本数=("北京时间", "count"),
            **{"实际使用时间段开始 (北京)": ("北京时间", "min"),
               "实际使用时间段结束 (北京)": ("北京时间", "max")}
        ).reset_index()
        out = out.merge(meta, on=group_cols, how="left")
    else:
        out["样本数"] = len(out)
        out["实际使用时间段开始 (北京)"] = pd.to_datetime(out["北京时间"], errors="coerce").min()
        out["实际使用时间段结束 (北京)"] = pd.to_datetime(out["北京时间"], errors="coerce").max()
    return out


def _insert_time_gap_nans(df, time_col, value_col, max_gap="60min"):
    if df is None or df.empty:
        return pd.DataFrame(columns=[time_col, value_col])
    sub = df[[time_col, value_col]].copy()
    sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if sub.empty:
        return sub
    out_rows = [sub.iloc[0].to_dict()]
    gap = pd.Timedelta(max_gap)
    for i in range(1, len(sub)):
        t_prev = pd.Timestamp(sub.iloc[i-1][time_col])
        t_now = pd.Timestamp(sub.iloc[i][time_col])
        if t_now - t_prev > gap:
            out_rows.append({time_col: t_prev + gap/2, value_col: np.nan})
        out_rows.append(sub.iloc[i].to_dict())
    return pd.DataFrame(out_rows)


def plot_timeseries_comparison_dual(obs_ts, wrf_ts, obs_col, wrf_col, output_dir, fig_prefix, y_label,
                                    is_direction=False, gap="60min"):
    """直接叠加 obs 有效观测与 WRF 完整时序；不再因为 obs 缺测而裁掉 WRF。"""
    if obs_ts is None or obs_ts.empty or wrf_ts is None or wrf_ts.empty:
        print(f"{fig_prefix}: obs 或 WRF 时序为空，跳过。")
        return
    os.makedirs(output_dir, exist_ok=True)

    obs_ts = _normalize_timeseries_for_asof(obs_ts, time_col="北京时间", height_col="离地高度 (m)", value_cols=[obs_col])
    wrf_ts = _normalize_timeseries_for_asof(wrf_ts, time_col="北京时间", height_col="离地高度 (m)", value_cols=[wrf_col])
    common_heights = sorted(set(np.round(obs_ts["离地高度 (m)"].dropna(), 6)).intersection(
                            set(np.round(wrf_ts["离地高度 (m)"].dropna(), 6))))
    for h in common_heights:
        sub_obs = obs_ts[np.isclose(obs_ts["离地高度 (m)"].astype(float), float(h), atol=1e-6)].copy()
        sub_wrf = wrf_ts[np.isclose(wrf_ts["离地高度 (m)"].astype(float), float(h), atol=1e-6)].copy()
        if sub_obs.empty or sub_wrf.empty:
            continue
        plt.figure(figsize=(10,4))
        if is_direction:
            obs_df = _insert_time_gap_nans(sub_obs, "北京时间", obs_col, max_gap=gap)
            wrf_df = _insert_time_gap_nans(sub_wrf, "北京时间", wrf_col, max_gap=gap)
            obs_vals = _prepare_wrapped_direction_for_plot(obs_df[obs_col])
            wrf_vals = _prepare_wrapped_direction_for_plot(wrf_df[wrf_col])
            plt.scatter(sub_obs["北京时间"], np.mod(pd.to_numeric(sub_obs[obs_col], errors='coerce'), 360.0), s=8, label="Obs", alpha=0.7)
            plt.plot(obs_df["北京时间"], obs_vals, linewidth=0.8, alpha=0.8)
            plt.plot(wrf_df["北京时间"], wrf_vals, linewidth=1.0, alpha=0.9, label="WRF")
            plt.ylim(0,360); plt.yticks(np.arange(0,361,45))
        else:
            obs_df = _insert_time_gap_nans(sub_obs, "北京时间", obs_col, max_gap=gap)
            wrf_df = _insert_time_gap_nans(sub_wrf, "北京时间", wrf_col, max_gap=gap)
            plt.plot(wrf_df["北京时间"], wrf_df[wrf_col], label="WRF", linewidth=1.0)
            plt.scatter(sub_obs["北京时间"], sub_obs[obs_col], label="Obs", s=8, alpha=0.8)
            plt.plot(obs_df["北京时间"], obs_df[obs_col], linewidth=0.6, alpha=0.6)
        plt.xlabel("Beijing Time")
        plt.ylabel(y_label)
        plt.title(f"{fig_prefix} at {h:g} m")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig_name = f"{fig_prefix}_{int(round(h))}m.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")


def plot_timeseries_comparison(compare_df, obs_col, wrf_col, output_dir, fig_prefix, y_label, is_direction=False):
    if compare_df is None or compare_df.empty:
        print(f"{fig_prefix}: 时序对比数据为空，跳过。")
        return

    os.makedirs(output_dir, exist_ok=True)
    for h in sorted(compare_df["离地高度 (m)"].dropna().unique()):
        sub = compare_df[compare_df["离地高度 (m)"] == h].copy().sort_values("北京时间")
        if sub.empty:
            continue

        plt.figure(figsize=(10, 4))

        if is_direction:
            obs_vals = pd.to_numeric(sub[obs_col], errors='coerce').to_numpy(dtype=float)
            wrf_vals = pd.to_numeric(sub[wrf_col], errors='coerce').to_numpy(dtype=float)

            obs_plot = _prepare_wrapped_direction_for_plot(obs_vals)
            wrf_plot = _prepare_wrapped_direction_for_plot(wrf_vals)

            plt.scatter(sub["北京时间"], np.mod(obs_vals, 360.0), s=10, label="Obs", alpha=0.8)
            plt.scatter(sub["北京时间"], np.mod(wrf_vals, 360.0), s=10, label="WRF", alpha=0.8)
            plt.plot(sub["北京时间"], obs_plot, linewidth=0.8, alpha=0.9)
            plt.plot(sub["北京时间"], wrf_plot, linewidth=0.8, alpha=0.9)
            plt.ylim(0.0, 360.0)
            plt.yticks(np.arange(0, 361, 45))
        else:
            plt.plot(sub["北京时间"], sub[obs_col], label="Obs")
            plt.plot(sub["北京时间"], sub[wrf_col], label="WRF")

        plt.xlabel("Beijing Time")
        plt.ylabel(y_label)
        plt.title(f"{fig_prefix} at {h:g} m")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        fig_name = f"{fig_prefix}_{int(round(h))}m.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")


# ====================== OBS vs WRF error statistics and plots (added, minor change) ======================

def _circular_diff_deg(wrf_deg, obs_deg):
    """风向误差，返回 [-180, 180) 内的 WRF - OBS 环形差值。"""
    wrf = np.asarray(wrf_deg, dtype=float)
    obs = np.asarray(obs_deg, dtype=float)
    return (wrf - obs + 180.0) % 360.0 - 180.0


def _circular_mean_deg(values_deg):
    """风向环形平均，避免 359° 和 1° 被普通平均成 180°。"""
    vals = pd.to_numeric(pd.Series(values_deg), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if vals.size == 0:
        return np.nan
    rad = np.deg2rad(vals)
    return float(np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))) % 360.0)


def _calc_error_metrics(obs_values, wrf_values, is_direction=False):
    """
    计算 obs vs WRF 误差指标。
    误差定义统一为 error = WRF - OBS。
    风向使用环形误差；其他变量使用普通差值。
    """
    obs = pd.to_numeric(pd.Series(obs_values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    wrf = pd.to_numeric(pd.Series(wrf_values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = obs.notna() & wrf.notna()
    obs = obs.loc[mask].astype(float)
    wrf = wrf.loc[mask].astype(float)

    if len(obs) == 0:
        return {
            "样本数": 0,
            "OBS均值": np.nan,
            "WRF均值": np.nan,
            "Bias_WRF减OBS": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_percent": np.nan,
            "NMB_percent": np.nan,
            "R": np.nan,
            "R2": np.nan,
            "NSE": np.nan,
            "IA": np.nan,
            "误差最小值": np.nan,
            "误差最大值": np.nan,
        }

    if is_direction:
        err = pd.Series(_circular_diff_deg(wrf.values, obs.values), index=obs.index).replace([np.inf, -np.inf], np.nan)
        obs_mean = _circular_mean_deg(obs.values)
        wrf_mean = _circular_mean_deg(wrf.values)
        # Do not report linear correlation, NSE, IA, or MAPE directly for wind direction to avoid misleading interpretation.
        R = R2 = NSE = IA = MAPE = NMB = np.nan
    else:
        err = (wrf - obs).replace([np.inf, -np.inf], np.nan)
        obs_mean = float(obs.mean())
        wrf_mean = float(wrf.mean())

        if len(obs) >= 2 and obs.std(skipna=True) > 0 and wrf.std(skipna=True) > 0:
            R = float(np.corrcoef(obs.values, wrf.values)[0, 1])
            R2 = R ** 2
        else:
            R = R2 = np.nan

        denom_nse = float(np.sum((obs.values - obs_mean) ** 2))
        NSE = float(1.0 - np.sum((wrf.values - obs.values) ** 2) / denom_nse) if denom_nse > 0 else np.nan

        denom_ia = float(np.sum((np.abs(wrf.values - obs_mean) + np.abs(obs.values - obs_mean)) ** 2))
        IA = float(1.0 - np.sum((wrf.values - obs.values) ** 2) / denom_ia) if denom_ia > 0 else np.nan

        nonzero = np.abs(obs.values) > 1e-12
        MAPE = float(np.nanmean(np.abs((wrf.values[nonzero] - obs.values[nonzero]) / obs.values[nonzero])) * 100.0) if np.any(nonzero) else np.nan
        denom_sum = float(np.nansum(obs.values))
        NMB = float(np.nansum(wrf.values - obs.values) / denom_sum * 100.0) if abs(denom_sum) > 1e-12 else np.nan

    err = err.replace([np.inf, -np.inf], np.nan).dropna()
    if err.empty:
        bias = mae = rmse = err_min = err_max = np.nan
    else:
        bias = float(err.mean())
        mae = float(np.mean(np.abs(err.values)))
        rmse = float(np.sqrt(np.mean(err.values ** 2)))
        err_min = float(err.min())
        err_max = float(err.max())

    return {
        "样本数": int(len(err)),
        "OBS均值": obs_mean,
        "WRF均值": wrf_mean,
        "Bias_WRF减OBS": bias,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": MAPE,
        "NMB_percent": NMB,
        "R": R,
        "R2": R2,
        "NSE": NSE,
        "IA": IA,
        "误差最小值": err_min,
        "误差最大值": err_max,
    }


def build_error_timeseries(compare_df, obs_col, wrf_col, variable_name, unit="", is_direction=False, is_scalar=False):
    """由 matched obs-WRF 表生成逐时次误差表。"""
    if compare_df is None or compare_df.empty:
        return pd.DataFrame()

    out = compare_df.copy()
    out["北京时间"] = pd.to_datetime(out["北京时间"], errors="coerce")
    out[obs_col] = pd.to_numeric(out[obs_col], errors="coerce")
    out[wrf_col] = pd.to_numeric(out[wrf_col], errors="coerce")
    out = out.dropna(subset=["北京时间", obs_col, wrf_col]).copy()
    if out.empty:
        return pd.DataFrame()

    if is_direction:
        out["误差_WRF减OBS"] = _circular_diff_deg(out[wrf_col].values, out[obs_col].values)
        out["绝对误差"] = np.abs(out["误差_WRF减OBS"])
    else:
        out["误差_WRF减OBS"] = out[wrf_col] - out[obs_col]
        out["绝对误差"] = np.abs(out["误差_WRF减OBS"])

    out["平方误差"] = out["误差_WRF减OBS"] ** 2
    out["变量"] = variable_name
    out["单位"] = unit
    if is_scalar and "离地高度 (m)" in out.columns:
        out = out.drop(columns=["离地高度 (m)"])
    sort_cols = ["北京时间"] if is_scalar else ["离地高度 (m)", "北京时间"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_error_statistics(compare_df, obs_col, wrf_col, variable_name, unit="", is_direction=False, is_scalar=False):
    """生成整体 + 分高度误差统计。"""
    if compare_df is None or compare_df.empty:
        return pd.DataFrame()

    rows = []
    df = compare_df.copy()
    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
    df[wrf_col] = pd.to_numeric(df[wrf_col], errors="coerce")
    df["北京时间"] = pd.to_datetime(df["北京时间"], errors="coerce")
    df = df.dropna(subset=["北京时间", obs_col, wrf_col]).copy()
    if df.empty:
        return pd.DataFrame()

    def _one_row(sub, group_name, height_val=np.nan):
        m = _calc_error_metrics(sub[obs_col], sub[wrf_col], is_direction=is_direction)
        row = {
            "变量": variable_name,
            "单位": unit,
            "统计范围": group_name,
            "离地高度 (m)": height_val,
            "误差定义": "WRF - OBS；风向为环形差值[-180,180)" if is_direction else "WRF - OBS",
            "实际使用时间段开始 (北京)": pd.to_datetime(sub["北京时间"], errors="coerce").min(),
            "实际使用时间段结束 (北京)": pd.to_datetime(sub["北京时间"], errors="coerce").max(),
        }
        row.update(m)
        return row

    rows.append(_one_row(df, "overall", np.nan))

    if (not is_scalar) and ("离地高度 (m)" in df.columns):
        for h, sub in df.groupby("离地高度 (m)"):
            if sub.empty:
                continue
            rows.append(_one_row(sub, "by_height", float(h)))

    return pd.DataFrame(rows)


def plot_error_profile(error_stats_df, output_dir, fig_name, title, x_label_suffix=""):
    """绘制分高度 Bias / MAE / RMSE 廓线。"""
    if error_stats_df is None or error_stats_df.empty:
        return
    if "离地高度 (m)" not in error_stats_df.columns:
        return
    sub = error_stats_df[(error_stats_df["统计范围"] == "by_height") & error_stats_df["离地高度 (m)"].notna()].copy()
    if sub.empty:
        return
    sub = sub.sort_values("离地高度 (m)")

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    for metric in ["Bias_WRF减OBS", "MAE", "RMSE"]:
        if metric in sub.columns and pd.to_numeric(sub[metric], errors="coerce").notna().any():
            plt.plot(pd.to_numeric(sub[metric], errors="coerce"), sub["离地高度 (m)"], marker="o", label=metric)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel(f"Error metric{x_label_suffix}")
    plt.ylabel("Height (m)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"已保存: {fig_path}")


def plot_obs_wrf_scatter(compare_df, obs_col, wrf_col, output_dir, fig_name, title, x_label, y_label, is_direction=False):
    """绘制 obs vs WRF 散点图，带 1:1 参考线。"""
    if compare_df is None or compare_df.empty:
        return
    df = compare_df.copy()
    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
    df[wrf_col] = pd.to_numeric(df[wrf_col], errors="coerce")
    df = df.dropna(subset=[obs_col, wrf_col])
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5.5, 5.5))
    if is_direction:
        x = np.mod(df[obs_col].values, 360.0)
        y = np.mod(df[wrf_col].values, 360.0)
        vmin, vmax = 0.0, 360.0
    else:
        x = df[obs_col].values
        y = df[wrf_col].values
        vmin = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        vmax = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        pad = 0.05 * (vmax - vmin)
        vmin -= pad
        vmax += pad
    plt.scatter(x, y, s=10, alpha=0.5)
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.0, label="1:1")
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"已保存: {fig_path}")


def plot_error_timeseries(error_ts_df, output_dir, fig_prefix, y_label, is_scalar=False):
    """绘制误差时序图。分高度变量每个高度一张；标量变量一张。"""
    if error_ts_df is None or error_ts_df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = error_ts_df.copy()
    df["北京时间"] = pd.to_datetime(df["北京时间"], errors="coerce")
    df["误差_WRF减OBS"] = pd.to_numeric(df["误差_WRF减OBS"], errors="coerce")
    df = df.dropna(subset=["北京时间", "误差_WRF减OBS"]).sort_values("北京时间")
    if df.empty:
        return

    if is_scalar or "离地高度 (m)" not in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["北京时间"], df["误差_WRF减OBS"], linewidth=0.8)
        plt.axhline(0.0, linewidth=0.8)
        plt.xlabel("Beijing Time")
        plt.ylabel(y_label)
        plt.title(fig_prefix)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{fig_prefix}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")
        return

    for h, sub in df.groupby("离地高度 (m)"):
        sub = sub.sort_values("北京时间")
        if sub.empty:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(sub["北京时间"], sub["误差_WRF减OBS"], linewidth=0.8)
        plt.axhline(0.0, linewidth=0.8)
        plt.xlabel("Beijing Time")
        plt.ylabel(y_label)
        plt.title(f"{fig_prefix} at {float(h):g} m")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{fig_prefix}_{int(round(float(h)))}m.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")


def export_error_diagnostics(compare_df, obs_col, wrf_col, var_key, var_label, unit,
                             output_dir, site_tag, domain_id, is_direction=False, is_scalar=False):
    """
    一次性导出某变量的误差时序、误差统计和误差图。
    返回: (error_stats_df, error_timeseries_df)
    """
    if compare_df is None or compare_df.empty:
        print(f"[{site_tag}] {var_key} 没有 matched obs-WRF 样本，跳过误差统计。")
        return pd.DataFrame(), pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"{site_tag}_{domain_id}"

    err_ts = build_error_timeseries(
        compare_df, obs_col, wrf_col,
        variable_name=var_label,
        unit=unit,
        is_direction=is_direction,
        is_scalar=is_scalar,
    )
    err_stats = build_error_statistics(
        compare_df, obs_col, wrf_col,
        variable_name=var_label,
        unit=unit,
        is_direction=is_direction,
        is_scalar=is_scalar,
    )

    if err_ts is not None and not err_ts.empty:
        ts_csv = os.path.join(output_dir, f"error_timeseries_{var_key}_{suffix}.csv")
        err_ts.to_csv(ts_csv, index=False, encoding="utf-8-sig")
        print(f"[{site_tag}] 误差时序已保存: {ts_csv}")

    if err_stats is not None and not err_stats.empty:
        stats_csv = os.path.join(output_dir, f"error_stats_{var_key}_{suffix}.csv")
        err_stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
        print(f"[{site_tag}] 误差统计已保存: {stats_csv}")

    unit_label = f" ({unit})" if unit else ""
    if err_stats is not None and not err_stats.empty and not is_scalar:
        plot_error_profile(
            err_stats,
            output_dir=output_dir,
            fig_name=f"Error_profile_{var_key}_{suffix}.png",
            title=f"Error metrics profile ({var_label}, {site_tag})",
            x_label_suffix=unit_label,
        )

    plot_obs_wrf_scatter(
        compare_df,
        obs_col=obs_col,
        wrf_col=wrf_col,
        output_dir=output_dir,
        fig_name=f"Scatter_obs_wrf_{var_key}_{suffix}.png",
        title=f"Obs vs WRF scatter ({var_label}, {site_tag})",
        x_label=f"Obs {var_label}{unit_label}",
        y_label=f"WRF {var_label}{unit_label}",
        is_direction=is_direction,
    )

    if err_ts is not None and not err_ts.empty:
        plot_error_timeseries(
            err_ts,
            output_dir=output_dir,
            fig_prefix=f"Error_timeseries_{var_key}_{suffix}",
            y_label=f"WRF - OBS{unit_label}",
            is_scalar=is_scalar,
        )

    return err_stats, err_ts



# ====================== Wind-rose / Weibull distribution error statistics and plots (added, minor change) ======================

def _standardize_distribution_inputs(ws_compare, wd_compare):
    """
    将风速 matched 表和风向 matched 表合并为同一时间、同一高度的风分布样本。
    这里会再次剔除 NaN、无穷值和不合理范围，确保风玫瑰 / Weibull 只使用有效 obs-WRF 配对样本。
    """
    if ws_compare is None or ws_compare.empty or wd_compare is None or wd_compare.empty:
        return pd.DataFrame()

    ws_obs_col = OBS_VAR_META["wspd"]["obs_col"]
    wd_obs_col = OBS_VAR_META["wdir"]["obs_col"]
    ws_wrf_col = "十分钟平均风速 (m/s)"
    wd_wrf_col = "十分钟平均风向 (°)"

    need_ws = ["北京时间", "离地高度 (m)", ws_obs_col, ws_wrf_col]
    need_wd = ["北京时间", "离地高度 (m)", wd_obs_col, wd_wrf_col]
    if any(c not in ws_compare.columns for c in need_ws):
        return pd.DataFrame()
    if any(c not in wd_compare.columns for c in need_wd):
        return pd.DataFrame()

    ws = ws_compare[need_ws].copy()
    wd = wd_compare[need_wd].copy()

    ws = ws.rename(columns={ws_obs_col: "OBS_WS", ws_wrf_col: "WRF_WS"})
    wd = wd.rename(columns={wd_obs_col: "OBS_WD", wd_wrf_col: "WRF_WD"})

    for df in (ws, wd):
        df["北京时间"] = pd.to_datetime(df["北京时间"], errors="coerce")
        df["离地高度 (m)"] = pd.to_numeric(df["离地高度 (m)"], errors="coerce")

    ws["OBS_WS"] = pd.to_numeric(ws["OBS_WS"], errors="coerce")
    ws["WRF_WS"] = pd.to_numeric(ws["WRF_WS"], errors="coerce")
    wd["OBS_WD"] = pd.to_numeric(wd["OBS_WD"], errors="coerce")
    wd["WRF_WD"] = pd.to_numeric(wd["WRF_WD"], errors="coerce")

    df = pd.merge(
        ws,
        wd,
        on=["北京时间", "离地高度 (m)"],
        how="inner",
    )
    if df.empty:
        return pd.DataFrame()

    # Apply a second QC pass: OBS has already been cleaned in load_obs_dataframe; this is a safeguard before distribution diagnostics.
    # Filter WRF using the same physical range to prevent outliers from affecting distribution fitting.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["北京时间", "离地高度 (m)", "OBS_WS", "WRF_WS", "OBS_WD", "WRF_WD"]).copy()

    ws_min, ws_max = OBS_VALID_RANGES.get("wspd", (0.0, 75.0))
    wd_min, wd_max = OBS_VALID_RANGES.get("wdir", (0.0, 360.0))
    df = df[(df["OBS_WS"] >= ws_min) & (df["OBS_WS"] <= ws_max)]
    df = df[(df["WRF_WS"] >= ws_min) & (df["WRF_WS"] <= ws_max)]
    df = df[(df["OBS_WD"] >= wd_min) & (df["OBS_WD"] <= wd_max)]
    df = df[(df["WRF_WD"] >= wd_min) & (df["WRF_WD"] <= wd_max)]

    if OBS_ZERO_AS_NAN.get("wspd", False):
        df = df[(df["OBS_WS"] > 0.0) & (df["WRF_WS"] > 0.0)]

    df["OBS_WD"] = np.mod(df["OBS_WD"].values, 360.0)
    df["WRF_WD"] = np.mod(df["WRF_WD"].values, 360.0)
    return df.sort_values(["离地高度 (m)", "北京时间"]).reset_index(drop=True)


def _sector_index(direction_deg, n_sectors=WIND_ROSE_NUM_SECTORS):
    wd = np.mod(np.asarray(direction_deg, dtype=float), 360.0)
    sector_width = 360.0 / float(n_sectors)
    return (np.floor((wd + 0.5 * sector_width) / sector_width).astype(int) % n_sectors)


def _sector_frequency(direction_deg, n_sectors=WIND_ROSE_NUM_SECTORS):
    idx = _sector_index(direction_deg, n_sectors=n_sectors)
    counts = np.bincount(idx, minlength=n_sectors).astype(float)
    total = float(np.sum(counts))
    freq_pct = counts / total * 100.0 if total > 0 else np.full(n_sectors, np.nan)
    return counts, freq_pct


def _speed_bin_frequency(speed_values, bins):
    vals = pd.to_numeric(pd.Series(speed_values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    counts, _ = np.histogram(vals, bins=np.asarray(bins, dtype=float))
    counts = counts.astype(float)
    total = float(np.sum(counts))
    freq_pct = counts / total * 100.0 if total > 0 else np.full(len(bins) - 1, np.nan)
    return counts, freq_pct


def _distribution_metrics(obs_freq_pct, wrf_freq_pct, prefix=""):
    obs = np.asarray(obs_freq_pct, dtype=float)
    wrf = np.asarray(wrf_freq_pct, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(wrf)
    if not np.any(mask):
        return {
            f"{prefix}MAE_pct": np.nan,
            f"{prefix}RMSE_pct": np.nan,
            f"{prefix}Bias_mean_pct": np.nan,
            f"{prefix}MaxAbsDiff_pct": np.nan,
            f"{prefix}TotalVariationDistance": np.nan,
        }
    diff = wrf[mask] - obs[mask]
    return {
        f"{prefix}MAE_pct": float(np.mean(np.abs(diff))),
        f"{prefix}RMSE_pct": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}Bias_mean_pct": float(np.mean(diff)),
        f"{prefix}MaxAbsDiff_pct": float(np.max(np.abs(diff))),
        f"{prefix}TotalVariationDistance": float(0.5 * np.sum(np.abs(diff)) / 100.0),
    }


def _joint_windrose_frequency(speed_values, direction_deg, speed_bins, n_sectors=WIND_ROSE_NUM_SECTORS):
    spd = pd.to_numeric(pd.Series(speed_values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    wd = pd.to_numeric(pd.Series(direction_deg), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = spd.notna() & wd.notna()
    spd = spd.loc[mask].values
    wd = wd.loc[mask].values

    speed_bins = np.asarray(speed_bins, dtype=float)
    n_speed = len(speed_bins) - 1
    table = np.zeros((n_sectors, n_speed), dtype=float)
    if spd.size == 0:
        return table, np.full_like(table, np.nan, dtype=float)

    sec = _sector_index(wd, n_sectors=n_sectors)
    speed_idx = np.digitize(spd, speed_bins, right=False) - 1
    speed_idx = np.clip(speed_idx, 0, n_speed - 1)
    for s, b in zip(sec, speed_idx):
        table[int(s), int(b)] += 1.0
    total = float(table.sum())
    freq_pct = table / total * 100.0 if total > 0 else np.full_like(table, np.nan, dtype=float)
    return table, freq_pct


def _windrose_sector_label(center_deg, sector_width):
    left = (center_deg - 0.5 * sector_width) % 360.0
    right = (center_deg + 0.5 * sector_width) % 360.0
    return f"{left:.1f}–{right:.1f}°"


def build_windrose_diagnostics(distribution_df, site_tag, domain_id,
                               n_sectors=WIND_ROSE_NUM_SECTORS,
                               speed_bins=WIND_ROSE_SPEED_BINS):
    """生成风玫瑰方向频率、方向-风速联合频率和分布误差统计。"""
    if distribution_df is None or distribution_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sector_width = 360.0 / float(n_sectors)
    centers = np.arange(n_sectors, dtype=float) * sector_width
    speed_bins = np.asarray(speed_bins, dtype=float)
    sector_rows = []
    joint_rows = []
    stats_rows = []

    for h, sub in distribution_df.groupby("离地高度 (m)"):
        if sub.empty:
            continue
        obs_count, obs_freq = _sector_frequency(sub["OBS_WD"].values, n_sectors=n_sectors)
        wrf_count, wrf_freq = _sector_frequency(sub["WRF_WD"].values, n_sectors=n_sectors)
        diff = wrf_freq - obs_freq

        for i, center in enumerate(centers):
            sector_rows.append({
                "站点": site_tag,
                "WRF嵌套域": domain_id,
                "离地高度 (m)": float(h),
                "sector_index": int(i),
                "sector_center_deg": float(center),
                "sector_label": _windrose_sector_label(center, sector_width),
                "OBS_count": float(obs_count[i]),
                "WRF_count": float(wrf_count[i]),
                "OBS_freq_pct": float(obs_freq[i]),
                "WRF_freq_pct": float(wrf_freq[i]),
                "Diff_WRF_minus_OBS_pct": float(diff[i]),
            })

        obs_joint_count, obs_joint_freq = _joint_windrose_frequency(sub["OBS_WS"], sub["OBS_WD"], speed_bins, n_sectors=n_sectors)
        wrf_joint_count, wrf_joint_freq = _joint_windrose_frequency(sub["WRF_WS"], sub["WRF_WD"], speed_bins, n_sectors=n_sectors)
        joint_diff = wrf_joint_freq - obs_joint_freq
        for i, center in enumerate(centers):
            for b in range(len(speed_bins) - 1):
                joint_rows.append({
                    "站点": site_tag,
                    "WRF嵌套域": domain_id,
                    "离地高度 (m)": float(h),
                    "sector_index": int(i),
                    "sector_center_deg": float(center),
                    "speed_bin_left (m/s)": float(speed_bins[b]),
                    "speed_bin_right (m/s)": float(speed_bins[b + 1]),
                    "OBS_count": float(obs_joint_count[i, b]),
                    "WRF_count": float(wrf_joint_count[i, b]),
                    "OBS_freq_pct": float(obs_joint_freq[i, b]),
                    "WRF_freq_pct": float(wrf_joint_freq[i, b]),
                    "Diff_WRF_minus_OBS_pct": float(joint_diff[i, b]),
                })

        row = {
            "站点": site_tag,
            "WRF嵌套域": domain_id,
            "变量": "Wind rose / direction distribution",
            "统计范围": "by_height",
            "离地高度 (m)": float(h),
            "样本数": int(len(sub)),
            "OBS prevailing sector center (deg)": float(centers[int(np.nanargmax(obs_freq))]) if np.isfinite(obs_freq).any() else np.nan,
            "WRF prevailing sector center (deg)": float(centers[int(np.nanargmax(wrf_freq))]) if np.isfinite(wrf_freq).any() else np.nan,
            "Prevailing sector circular diff (deg)": np.nan,
        }
        if np.isfinite(row["OBS prevailing sector center (deg)"]) and np.isfinite(row["WRF prevailing sector center (deg)"]):
            row["Prevailing sector circular diff (deg)"] = float(_circular_diff_deg(
                row["WRF prevailing sector center (deg)"], row["OBS prevailing sector center (deg)"]
            ))
        row.update(_distribution_metrics(obs_freq, wrf_freq, prefix="direction_freq_"))
        row.update(_distribution_metrics(obs_joint_freq.ravel(), wrf_joint_freq.ravel(), prefix="joint_speed_direction_freq_"))
        stats_rows.append(row)

    return pd.DataFrame(stats_rows), pd.DataFrame(sector_rows), pd.DataFrame(joint_rows)


def plot_windrose_direction_compare(sector_df, output_dir, site_tag, domain_id):
    """绘制方向频率风玫瑰对比和频率差值图。"""
    if sector_df is None or sector_df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)

    for h, sub in sector_df.groupby("离地高度 (m)"):
        sub = sub.sort_values("sector_index")
        if sub.empty:
            continue
        theta = np.deg2rad(sub["sector_center_deg"].values.astype(float))
        obs = pd.to_numeric(sub["OBS_freq_pct"], errors="coerce").values
        wrf = pd.to_numeric(sub["WRF_freq_pct"], errors="coerce").values
        diff = pd.to_numeric(sub["Diff_WRF_minus_OBS_pct"], errors="coerce").values

        theta_closed = np.r_[theta, theta[0]]
        obs_closed = np.r_[obs, obs[0]]
        wrf_closed = np.r_[wrf, wrf[0]]
        diff_closed = np.r_[diff, diff[0]]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(theta_closed, obs_closed, marker="o", label="Obs")
        ax.plot(theta_closed, wrf_closed, marker="o", label="WRF")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Wind direction frequency rose ({site_tag}, {float(h):g} m)")
        ax.set_ylabel("Frequency (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"Windrose_direction_compare_{site_tag}_{domain_id}_{int(round(float(h)))}m.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"已保存: {fig_path}")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(theta_closed, diff_closed, marker="o")
        ax.axhline(0.0, linewidth=0.8)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Wind-rose frequency error: WRF - OBS ({site_tag}, {float(h):g} m)")
        ax.set_ylabel("Frequency difference (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"Windrose_direction_error_{site_tag}_{domain_id}_{int(round(float(h)))}m.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"已保存: {fig_path}")


def _fit_weibull_moments(speed_values, min_samples=WEIBULL_MIN_SAMPLES):
    """
    用矩估计拟合二参数 Weibull：shape k, scale c, loc 固定为 0。
    这样不依赖 scipy，适合直接在服务器上运行。
    """
    vals = pd.to_numeric(pd.Series(speed_values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
    vals = vals[vals > 0.0]
    if vals.size < min_samples:
        return {"n": int(vals.size), "k": np.nan, "c": np.nan, "mean": np.nan, "std": np.nan}

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan
    if not np.isfinite(mean) or mean <= 0.0 or not np.isfinite(std) or std <= 0.0:
        return {"n": int(vals.size), "k": np.nan, "c": np.nan, "mean": mean, "std": std}

    cv = std / mean

    def cv_from_k(k):
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        val = g2 / (g1 ** 2) - 1.0
        return math.sqrt(max(val, 0.0))

    lo, hi = 0.2, 20.0
    # cv_from_k decreases as k increases. Use a standard approximation as a fallback outside the valid range.
    try:
        if not (cv_from_k(hi) <= cv <= cv_from_k(lo)):
            k = cv ** (-1.086) if cv > 0 else np.nan
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if cv_from_k(mid) > cv:
                    lo = mid
                else:
                    hi = mid
            k = 0.5 * (lo + hi)
    except Exception:
        k = cv ** (-1.086) if cv > 0 else np.nan

    if not np.isfinite(k) or k <= 0.0:
        c = np.nan
    else:
        c = mean / math.gamma(1.0 + 1.0 / k)

    return {"n": int(vals.size), "k": float(k), "c": float(c), "mean": mean, "std": std}


def _weibull_pdf(x, k, c):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(k) or not np.isfinite(c) or k <= 0.0 or c <= 0.0:
        return np.full_like(x, np.nan, dtype=float)
    y = (k / c) * (x / c) ** (k - 1.0) * np.exp(-((x / c) ** k))
    y = np.where(np.isfinite(y), y, np.nan)
    return y


def _dynamic_speed_bins(obs_ws, wrf_ws, bin_width=WEIBULL_SPEED_BIN_WIDTH):
    vals = pd.to_numeric(pd.Series(np.r_[obs_ws, wrf_ws]), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return np.arange(0.0, 11.0, bin_width)
    upper = float(np.nanpercentile(vals, 99.5) + 2.0 * bin_width)
    upper = min(max(upper, 10.0), OBS_VALID_RANGES.get("wspd", (0.0, 75.0))[1])
    upper = math.ceil(upper / bin_width) * bin_width
    return np.arange(0.0, upper + bin_width, bin_width)


def build_weibull_diagnostics(distribution_df, site_tag, domain_id,
                              bin_width=WEIBULL_SPEED_BIN_WIDTH,
                              min_samples=WEIBULL_MIN_SAMPLES):
    """生成 Weibull 参数误差与风速频率曲线误差。"""
    if distribution_df is None or distribution_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stats_rows = []
    bin_rows = []
    for h, sub in distribution_df.groupby("离地高度 (m)"):
        if sub.empty:
            continue
        obs_fit = _fit_weibull_moments(sub["OBS_WS"].values, min_samples=min_samples)
        wrf_fit = _fit_weibull_moments(sub["WRF_WS"].values, min_samples=min_samples)
        bins = _dynamic_speed_bins(sub["OBS_WS"].values, sub["WRF_WS"].values, bin_width=bin_width)
        obs_count, obs_freq = _speed_bin_frequency(sub["OBS_WS"].values, bins)
        wrf_count, wrf_freq = _speed_bin_frequency(sub["WRF_WS"].values, bins)
        diff = wrf_freq - obs_freq

        row = {
            "站点": site_tag,
            "WRF嵌套域": domain_id,
            "变量": "Wind-speed distribution / Weibull",
            "统计范围": "by_height",
            "离地高度 (m)": float(h),
            "样本数": int(len(sub)),
            "OBS_Weibull_shape_k": obs_fit["k"],
            "WRF_Weibull_shape_k": wrf_fit["k"],
            "Delta_k_WRF_minus_OBS": wrf_fit["k"] - obs_fit["k"] if np.isfinite(wrf_fit["k"]) and np.isfinite(obs_fit["k"]) else np.nan,
            "OBS_Weibull_scale_c (m/s)": obs_fit["c"],
            "WRF_Weibull_scale_c (m/s)": wrf_fit["c"],
            "Delta_c_WRF_minus_OBS (m/s)": wrf_fit["c"] - obs_fit["c"] if np.isfinite(wrf_fit["c"]) and np.isfinite(obs_fit["c"]) else np.nan,
            "OBS_mean_ws (m/s)": obs_fit["mean"],
            "WRF_mean_ws (m/s)": wrf_fit["mean"],
            "Mean_ws_bias_WRF_minus_OBS (m/s)": wrf_fit["mean"] - obs_fit["mean"] if np.isfinite(wrf_fit["mean"]) and np.isfinite(obs_fit["mean"]) else np.nan,
            "OBS_std_ws (m/s)": obs_fit["std"],
            "WRF_std_ws (m/s)": wrf_fit["std"],
        }
        row.update(_distribution_metrics(obs_freq, wrf_freq, prefix="speed_bin_freq_"))
        stats_rows.append(row)

        for i in range(len(bins) - 1):
            bin_rows.append({
                "站点": site_tag,
                "WRF嵌套域": domain_id,
                "离地高度 (m)": float(h),
                "speed_bin_left (m/s)": float(bins[i]),
                "speed_bin_right (m/s)": float(bins[i + 1]),
                "speed_bin_center (m/s)": float(0.5 * (bins[i] + bins[i + 1])),
                "OBS_count": float(obs_count[i]),
                "WRF_count": float(wrf_count[i]),
                "OBS_freq_pct": float(obs_freq[i]),
                "WRF_freq_pct": float(wrf_freq[i]),
                "Diff_WRF_minus_OBS_pct": float(diff[i]),
                "OBS_Weibull_shape_k": obs_fit["k"],
                "OBS_Weibull_scale_c (m/s)": obs_fit["c"],
                "WRF_Weibull_shape_k": wrf_fit["k"],
                "WRF_Weibull_scale_c (m/s)": wrf_fit["c"],
            })

    return pd.DataFrame(stats_rows), pd.DataFrame(bin_rows)


def plot_weibull_distribution_compare(weibull_bins_df, output_dir, site_tag, domain_id):
    """绘制风速经验频率、Weibull 拟合曲线，以及风速频率误差。"""
    if weibull_bins_df is None or weibull_bins_df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)

    for h, sub in weibull_bins_df.groupby("离地高度 (m)"):
        sub = sub.sort_values("speed_bin_center (m/s)").copy()
        if sub.empty:
            continue
        x = pd.to_numeric(sub["speed_bin_center (m/s)"], errors="coerce").values
        obs_freq = pd.to_numeric(sub["OBS_freq_pct"], errors="coerce").values
        wrf_freq = pd.to_numeric(sub["WRF_freq_pct"], errors="coerce").values
        diff = pd.to_numeric(sub["Diff_WRF_minus_OBS_pct"], errors="coerce").values
        if len(x) < 2:
            continue
        bin_width = float(np.nanmedian(np.diff(x))) if len(x) > 1 else WEIBULL_SPEED_BIN_WIDTH
        obs_k = pd.to_numeric(sub["OBS_Weibull_shape_k"], errors="coerce").dropna()
        obs_c = pd.to_numeric(sub["OBS_Weibull_scale_c (m/s)"], errors="coerce").dropna()
        wrf_k = pd.to_numeric(sub["WRF_Weibull_shape_k"], errors="coerce").dropna()
        wrf_c = pd.to_numeric(sub["WRF_Weibull_scale_c (m/s)"], errors="coerce").dropna()
        obs_k = float(obs_k.iloc[0]) if not obs_k.empty else np.nan
        obs_c = float(obs_c.iloc[0]) if not obs_c.empty else np.nan
        wrf_k = float(wrf_k.iloc[0]) if not wrf_k.empty else np.nan
        wrf_c = float(wrf_c.iloc[0]) if not wrf_c.empty else np.nan

        x_fit = np.linspace(max(0.001, np.nanmin(x) - bin_width), np.nanmax(x) + bin_width, 250)
        obs_fit_pct = _weibull_pdf(x_fit, obs_k, obs_c) * bin_width * 100.0
        wrf_fit_pct = _weibull_pdf(x_fit, wrf_k, wrf_c) * bin_width * 100.0

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, obs_freq, marker="o", linestyle="", label="Obs empirical")
        plt.plot(x, wrf_freq, marker="s", linestyle="", label="WRF empirical")
        if np.isfinite(obs_fit_pct).any():
            plt.plot(x_fit, obs_fit_pct, label="Obs Weibull fit")
        if np.isfinite(wrf_fit_pct).any():
            plt.plot(x_fit, wrf_fit_pct, label="WRF Weibull fit")
        plt.xlabel("Wind speed (m s$^{-1}$)")
        plt.ylabel("Frequency per bin (%)")
        plt.title(f"Weibull / wind-speed distribution ({site_tag}, {float(h):g} m)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"Weibull_distribution_compare_{site_tag}_{domain_id}_{int(round(float(h)))}m.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")

        plt.figure(figsize=(7, 4.0))
        plt.bar(x, diff, width=0.8 * bin_width, align="center")
        plt.axhline(0.0, linewidth=0.8)
        plt.xlabel("Wind speed (m s$^{-1}$)")
        plt.ylabel("Frequency difference WRF - OBS (%)")
        plt.title(f"Wind-speed frequency error ({site_tag}, {float(h):g} m)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"Weibull_frequency_error_{site_tag}_{domain_id}_{int(round(float(h)))}m.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")


def export_wind_distribution_diagnostics(ws_compare, wd_compare, output_dir, site_tag, domain_id):
    """
    导出风玫瑰、风速频率和 Weibull 分布误差诊断。
    所有计算均基于 matched obs-WRF 样本，并且会再次剔除 obs/WRF 缺测和不合理值。
    """
    if not EXPORT_WIND_DISTRIBUTION_DIAGNOSTICS:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    dist_df = _standardize_distribution_inputs(ws_compare, wd_compare)
    if dist_df.empty:
        print(f"[{site_tag}] 风玫瑰/Weibull 没有有效 matched 风速+风向样本，跳过。")
        return {}

    suffix = f"{site_tag}_{domain_id}"
    out = {"wind_distribution_samples": dist_df}
    sample_csv = os.path.join(output_dir, f"wind_distribution_matched_samples_{suffix}.csv")
    dist_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    print(f"[{site_tag}] 风分布 matched 样本已保存: {sample_csv}")

    windrose_stats, sector_df, joint_df = build_windrose_diagnostics(
        dist_df, site_tag=site_tag, domain_id=domain_id,
        n_sectors=WIND_ROSE_NUM_SECTORS,
        speed_bins=WIND_ROSE_SPEED_BINS,
    )
    weibull_stats, weibull_bins = build_weibull_diagnostics(
        dist_df, site_tag=site_tag, domain_id=domain_id,
        bin_width=WEIBULL_SPEED_BIN_WIDTH,
        min_samples=WEIBULL_MIN_SAMPLES,
    )

    export_items = {
        f"windrose_error_stats_{suffix}.csv": windrose_stats,
        f"windrose_sector_frequency_{suffix}.csv": sector_df,
        f"windrose_joint_speed_direction_frequency_{suffix}.csv": joint_df,
        f"weibull_error_stats_{suffix}.csv": weibull_stats,
        f"weibull_speed_frequency_{suffix}.csv": weibull_bins,
    }
    for fname, df in export_items.items():
        if df is not None and not df.empty:
            fpath = os.path.join(output_dir, fname)
            df.to_csv(fpath, index=False, encoding="utf-8-sig")
            print(f"[{site_tag}] 已保存: {fpath}")

    if sector_df is not None and not sector_df.empty:
        plot_windrose_direction_compare(sector_df, output_dir, site_tag=site_tag, domain_id=domain_id)
    if weibull_bins is not None and not weibull_bins.empty:
        plot_weibull_distribution_compare(weibull_bins, output_dir, site_tag=site_tag, domain_id=domain_id)

    if windrose_stats is not None and not windrose_stats.empty:
        out["windrose_error_stats"] = windrose_stats
    if sector_df is not None and not sector_df.empty:
        out["windrose_sector_frequency"] = sector_df
    if joint_df is not None and not joint_df.empty:
        out["windrose_joint_frequency"] = joint_df
    if weibull_stats is not None and not weibull_stats.empty:
        out["weibull_error_stats"] = weibull_stats
    if weibull_bins is not None and not weibull_bins.empty:
        out["weibull_speed_frequency"] = weibull_bins

    return out

def plot_wrf_only_profiles(native_period_df, output_dir, prefix="WRF_profile", zmax=500.0):
    if native_period_df is None or native_period_df.empty:
        print("native_period_df 为空，无法绘制 WRF-only 剖面。")
        return

    os.makedirs(output_dir, exist_ok=True)

    specs = [
        ("WRF原始层平均温度 (°C)", "Temperature (°C)", "WRF Temperature profile (native levels)", f"{prefix}_T.png"),
        ("WRF原始层平均气压 (KPa)", "Pressure (kPa)", "WRF Pressure profile (native levels)", f"{prefix}_P.png"),
        ("WRF原始层平均风向 (°)", "Wind direction (deg)", "WRF Wind-direction profile (native levels)", f"{prefix}_WD.png"),
    ]

    for col, xlabel, title, fig_name in specs:
        for zz, suffix in [(None, ""), (zmax, f"_0-{int(zmax)}m")]:
            sub = filter_profile_by_height(native_period_df, "离地高度 (m)", zmax=zz)
            if sub is None or sub.empty:
                continue
            plt.figure(figsize=(6, 8 if zz is None else 6))
            plt.scatter(sub[col].values, sub["离地高度 (m)"].values, s=25)
            plt.xlabel(xlabel)
            plt.ylabel("Height (m)")
            plt.title(title if zz is None else f"{title} (0–{int(zmax)} m)")
            if zz is not None:
                plt.ylim(0.0, zz)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out_name = fig_name.replace(".png", f"{suffix}.png")
            fig_path = os.path.join(output_dir, out_name)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"已保存: {fig_path}")


def plot_native_tke_profile(native_tke_df, output_dir, fig_name="TKE_profile.png", zmax=500.0):
    if native_tke_df is None or native_tke_df.empty:
        print("native_tke_df 为空，无法绘制 TKE 剖面。")
        return
    for zz, suffix in [(None, ""), (zmax, f"_0-{int(zmax)}m")]:
        sub = filter_profile_by_height(native_tke_df, "离地高度 (m)", zmax=zz)
        if sub is None or sub.empty:
            continue
        plt.figure(figsize=(6, 8 if zz is None else 6))
        plt.scatter(sub["WRF原始层平均TKE (m2/s2)"].values, sub["离地高度 (m)"].values, s=25)
        plt.xlabel("TKE (m$^2$ s$^{-2}$)")
        plt.ylabel("Height (m)")
        plt.title("WRF TKE vertical profile (native levels)" if zz is None else f"WRF TKE vertical profile (0–{int(zmax)} m)")
        if zz is not None:
            plt.ylim(0.0, zz)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_name = fig_name.replace(".png", f"{suffix}.png")
        fig_path = os.path.join(output_dir, out_name)
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"已保存: {fig_path}")


def plot_PBLH_timeseries(result_df, avg_start, avg_end, output_dir, fig_name="PBLH_timeseries.png"):
    if avg_start is None or avg_end is None or result_df.empty:
        print("无法绘制 PBLH 时间序列。")
        return np.nan

    mask = (result_df['北京时间'] >= avg_start) & (result_df['北京时间'] <= avg_end)
    sub = result_df[mask].copy().sort_values('北京时间')
    if sub.empty:
        print("指定时间段内没有数据，无法绘制 PBLH 时间序列。")
        return np.nan

    mean_PBLH = sub['PBLH (m)'].mean(skipna=True)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    # Multiple heights are repeated at each time; remove duplicates first.
    sub_line = sub[["北京时间", "PBLH (m)"]].drop_duplicates().sort_values("北京时间")
    plt.plot(sub_line['北京时间'], sub_line['PBLH (m)'], marker='o')
    plt.xlabel("Beijing Time")
    plt.ylabel("PBLH (m)")
    plt.title("PBLH timeseries")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"PBLH 时间序列图已保存: {fig_path}")

    return float(mean_PBLH)


def plot_Ri_L_timeseries(result_df, avg_start, avg_end, output_dir, fig_name="Ri_L_timeseries.png"):
    if avg_start is None or avg_end is None or result_df.empty:
        print("无法绘制 Ri/L 时间序列。")
        return np.nan, np.nan

    mask = (result_df['北京时间'] >= avg_start) & (result_df['北京时间'] <= avg_end)
    sub = result_df[mask].copy().sort_values('北京时间')
    if sub.empty:
        print("指定时间段内没有数据，无法绘制 Ri/L 时间序列。")
        return np.nan, np.nan

    sub_line = sub[["北京时间", "Bulk_Ri", "L (m)"]].drop_duplicates().sort_values("北京时间")
    mean_Ri = sub_line['Bulk_Ri'].mean(skipna=True)
    mean_L = sub_line['L (m)'].mean(skipna=True)

    print(f"Ri_mean = {mean_Ri:.6f}")
    print(f"L_mean  = {mean_L:.6f} m")

    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(sub_line['北京时间'], sub_line['Bulk_Ri'], marker='o', label='Bulk Ri')
    ax1.set_ylabel('Bulk Ri')
    ax1.set_xlabel('Beijing Time')

    ax2 = ax1.twinx()
    ax2.plot(sub_line['北京时间'], sub_line['L (m)'], marker='s', linestyle='--', label='L (m)')
    ax2.set_ylabel('Monin-Obukhov length L (m)')

    fig.autofmt_xdate()
    fig.tight_layout()

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Ri / L 时间序列图已保存: {fig_path}")

    return float(mean_Ri), float(mean_L)


# ====================== Simulation end-time check ======================

def get_simulation_end_time(wrf_output_dir):
    namelist_path = os.path.join(wrf_output_dir, "namelist.wps")
    if not os.path.exists(namelist_path):
        raise FileNotFoundError(f"未找到 namelist.wps 文件: {namelist_path}")

    with open(namelist_path, 'r') as f:
        content = f.read()

    pattern = r"end_date\s*=\s*'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})'"
    matches = re.findall(pattern, content)
    if matches:
        return datetime.strptime(matches[-1], '%Y-%m-%d_%H:%M:%S')
    raise ValueError("无法在 namelist.wps 中找到结束时间")


def check_simulation_end(excel_file, end_time_utc):
    try:
        df = pd.read_excel(excel_file, sheet_name="所有高度")
        if not df.empty:
            last_time = pd.to_datetime(df.iloc[-1]['北京时间'])
            end_time_bj = end_time_utc + timedelta(hours=8)
            print(f"最后记录时间: {last_time}, 模拟结束(北京): {end_time_bj}")
            return last_time >= end_time_bj
    except Exception as e:
        print(f"检查结束时间时出错: {str(e)}")
    return False


# ====================== Core processing function ======================

def process_monitoring_data(
    wrf_output_dir,
    output_dir,
    heights,
    nest_level,
    station_name,
    avg_start=None,
    avg_end=None,
    stability_heights=None,
    wrfout_dir=None,
    height_match_method="interpolate",
    out_of_range_policy="boundary",
    site_tag=None,
):
    domain_id = f"d{nest_level:02d}"
    ts_file = os.path.join(wrf_output_dir, f"{station_name}.{domain_id}.TS")
    if not os.path.exists(ts_file):
        raise FileNotFoundError(f"未找到指定的 TS 文件: {ts_file}")

    meta = parse_ts_file(ts_file)
    if not meta:
        raise ValueError(f"无法解析 TS 文件: {ts_file}")

    wrfinput_file = os.path.join(wrf_output_dir, f"wrfinput_{domain_id}")
    if not os.path.exists(wrfinput_file):
        raise FileNotFoundError(f"找不到 wrfinput 文件: {wrfinput_file}")

    base_path = os.path.join(wrf_output_dir, f"{station_name}.{domain_id}")
    required_files = [".UU", ".VV", ".TH", ".PR"]
    for suffix in required_files:
        file_path = f"{base_path}{suffix}"
        if not file_exists_and_non_empty(file_path):
            raise FileNotFoundError(f"文件不存在或为空: {file_path}")

    uu_time, uu_profiles = read_profile_file(f"{base_path}.UU")
    vv_time, vv_profiles = read_profile_file(f"{base_path}.VV")
    th_time, th_profiles = read_profile_file(f"{base_path}.TH")
    pr_time, pr_profiles = read_profile_file(f"{base_path}.PR")

    min_length = min(len(uu_time), len(vv_time), len(th_time), len(pr_time))
    if min_length == 0:
        raise ValueError("至少有一个数据文件为空")

    uu_time, uu_profiles = uu_time[:min_length], uu_profiles[:min_length, :]
    vv_time, vv_profiles = vv_time[:min_length], vv_profiles[:min_length, :]
    th_time, th_profiles = th_time[:min_length], th_profiles[:min_length, :]
    pr_time, pr_profiles = pr_time[:min_length], pr_profiles[:min_length, :]

    nlev_prof = uu_profiles.shape[1]

    height_levels_full = get_height_levels(wrfinput_file, meta['ix'], meta['iy'])
    if len(height_levels_full) < nlev_prof:
        raise ValueError(f"wrfinput 高度层数 ({len(height_levels_full)}) 少于 TS profile 层数 ({nlev_prof})")

    height_levels = np.asarray(height_levels_full[:nlev_prof], dtype=float)

    if heights is None or (isinstance(heights, (list, tuple)) and any(h is None for h in heights)):
        heights = [float(h) for h in height_levels]
        print(f"未指定有效 heights，自动使用所有模式高度层，共 {len(heights)} 层。")
    else:
        heights = sorted(set(float(h) for h in heights))

    mapping_df = get_height_mapping_table(heights, height_levels, height_match_method, out_of_range_policy)
    mapping_csv = os.path.join(output_dir, f"height_mapping_{site_tag if site_tag else station_name}_{domain_id}.csv")
    os.makedirs(output_dir, exist_ok=True)
    mapping_df.to_csv(mapping_csv, index=False, encoding='utf-8-sig')
    print(f"高度匹配说明表已保存: {mapping_csv}")

    warn_df = mapping_df[mapping_df["是否超阈值"] == True]
    if not warn_df.empty:
        print("以下高度与最近 WRF 层差异较大，请特别注意：")
        print(warn_df[["观测高度 (m)", "最近WRF层高度 (m)", "最近层高度差 (m)", "提取方式"]])

    if len(uu_time) < 2:
        raise ValueError("时间长度太短，无法自动推断时间步长。")

    dt_hours = uu_time[1] - uu_time[0]
    dt_seconds = dt_hours * 3600.0
    if dt_seconds <= 0:
        raise ValueError(f"检测到非法时间步长: {dt_seconds} 秒")

    window_size = int(round(600.0 / dt_seconds))
    if window_size < 1:
        window_size = 1

    print(f"自动识别的时间步长约为 {dt_seconds:.1f} 秒，每 10 分钟窗口包含 {window_size} 个时间步。")

    num_windows = len(uu_time) // window_size
    time_deltas = [timedelta(seconds=t * 3600) for t in uu_time]
    beijing_times = [meta['start_time'] + delta + timedelta(hours=8) for delta in time_deltas]

    profile_windows = []
    for iw in range(num_windows):
        start_idx = iw * window_size
        end_idx = (iw + 1) * window_size
        if end_idx > len(uu_time):
            continue

        u_prof_mean = np.mean(uu_profiles[start_idx:end_idx, :], axis=0)
        v_prof_mean = np.mean(vv_profiles[start_idx:end_idx, :], axis=0)
        th_prof_mean = np.mean(th_profiles[start_idx:end_idx, :], axis=0)
        pr_prof_mean = np.mean(pr_profiles[start_idx:end_idx, :], axis=0)

        if str(TSLIST_AVG_TIME_LABEL).lower().strip() == "begin":
            window_label_time = beijing_times[start_idx]
        else:
            window_label_time = beijing_times[end_idx - 1]

        profile_windows.append({
            # Keep the historical key name to avoid changing downstream logic.
            # In this huarui_A case its value is the 10-min window beginning,
            # because the Windographer observation timestamps are beginning times.
            "window_end_time": window_label_time,
            "u_prof_mean": u_prof_mean,
            "v_prof_mean": v_prof_mean,
            "th_prof_mean": th_prof_mean,
            "pr_prof_mean": pr_prof_mean,
        })

    native_period_df = compute_native_period_mean(profile_windows, height_levels, avg_start=avg_start, avg_end=avg_end)

    result_data = []
    for h in heights:
        for win in profile_windows:
            u_mean_h, meta_u = sample_profile_at_height(
                win["u_prof_mean"], height_levels, h,
                method=height_match_method,
                out_of_range_policy=out_of_range_policy
            )
            v_mean_h, _ = sample_profile_at_height(
                win["v_prof_mean"], height_levels, h,
                method=height_match_method,
                out_of_range_policy=out_of_range_policy
            )
            th_mean_h, _ = sample_profile_at_height(
                win["th_prof_mean"], height_levels, h,
                method=height_match_method,
                out_of_range_policy=out_of_range_policy
            )
            pr_mean_h, _ = sample_profile_at_height(
                win["pr_prof_mean"], height_levels, h,
                method=height_match_method,
                out_of_range_policy=out_of_range_policy
            )

            avg_ws = np.sqrt(u_mean_h ** 2 + v_mean_h ** 2) if pd.notna(u_mean_h) and pd.notna(v_mean_h) else np.nan
            if pd.notna(u_mean_h) and pd.notna(v_mean_h):
                wd_rad = np.arctan2(v_mean_h, u_mean_h)
                wd_deg = np.degrees(wd_rad)
                avg_wd = (270.0 - wd_deg) % 360.0
            else:
                avg_wd = np.nan

            if pd.notna(th_mean_h) and pd.notna(pr_mean_h):
                avg_temperature = th_mean_h * (pr_mean_h / 100000.0) ** 0.2854
                avg_pressure_kpa = pr_mean_h / 1000.0
            else:
                avg_temperature = np.nan
                avg_pressure_kpa = np.nan

            result_data.append({
                '北京时间': win["window_end_time"],
                '站点标签': site_tag if site_tag else station_name,
                'TS前缀': station_name,
                '站点名称': meta['station_name'],
                '经度 (°)': meta['lon'],
                '纬度 (°)': meta['lat'],
                '海拔高度 (m)': meta['elevation'],
                'WRF嵌套域': domain_id,
                '离地高度 (m)': float(h),
                '提取方式': meta_u["method_used"],
                '最近WRF层高度 (m)': meta_u["nearest_height"],
                '最近层高度差 (m)': meta_u["nearest_diff"],
                'WRF下层高度 (m)': meta_u["lower_height"],
                'WRF上层高度 (m)': meta_u["upper_height"],
                '十分钟平均风速 (m/s)': round(avg_ws, 6) if pd.notna(avg_ws) else np.nan,
                '十分钟平均风向 (°)': round(avg_wd, 6) if pd.notna(avg_wd) else np.nan,
                '十分钟平均气压 (KPa)': round(avg_pressure_kpa, 6) if pd.notna(avg_pressure_kpa) else np.nan,
                '十分钟平均温度 (K)': round(avg_temperature, 6) if pd.notna(avg_temperature) else np.nan,
                '十分钟平均温度 (°C)': round(avg_temperature - 273.15, 6) if pd.notna(avg_temperature) else np.nan,
                '平均_u (m/s)': round(u_mean_h, 6) if pd.notna(u_mean_h) else np.nan,
                '平均_v (m/s)': round(v_mean_h, 6) if pd.notna(v_mean_h) else np.nan,
            })

    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values(['离地高度 (m)', '北京时间']).reset_index(drop=True)

    dfL = None
    TKE_profiles = None
    z_tke_agl = None
    native_tke_period_df = None
    tke_source_name = None
    l_diag = {}

    if (wrfout_dir is not None) and os.path.isdir(wrfout_dir):
        try:
            dfL, TKE_profiles, z_tke_agl, tke_source_name, l_diag = compute_MO_and_TKE_profiles_from_wrfout(
                wrfout_dir, domain_id, meta['ix'], meta['iy']
            )
            print(f"TKE 来源变量: {tke_source_name}, 高度层数: {len(z_tke_agl)}")
            print_wrf_L_diagnostics(dfL, l_diag, site_tag=site_tag if site_tag else station_name)

            dfL_sorted = dfL.sort_values('time_bj')
            result_sorted = result_df.sort_values('北京时间')

            merged = pd.merge_asof(
                result_sorted,
                dfL_sorted[[c for c in ['time_bj', 'L (m)', 'PBLH (m)', 'UST (m/s)', 'HFX (W/m2)', 'PSFC (Pa)', 'T2 (K)', 'Z0 (m)'] if c in dfL_sorted.columns]],
                left_on='北京时间',
                right_on='time_bj',
                direction='nearest'
            )
            merged.drop(columns=['time_bj'], inplace=True)
            result_df = merged

            tke_rows = []
            time_bj_arr = dfL_sorted['time_bj'].values
            for it, tb in enumerate(time_bj_arr):
                tke_prof = TKE_profiles[it, :]
                for h in heights:
                    tke_val, _ = sample_profile_at_height(
                        tke_prof, z_tke_agl, h,
                        method=height_match_method,
                        out_of_range_policy=out_of_range_policy
                    )
                    tke_rows.append({
                        "time_bj": tb,
                        "离地高度 (m)": float(h),
                        "TKE (m2/s2)": float(tke_val) if pd.notna(tke_val) else np.nan,
                    })

            dfTKE = pd.DataFrame(tke_rows)

            merged_list = []
            for h in heights:
                h_float = float(h)
                sub_res = result_df[result_df['离地高度 (m)'] == h_float].copy()
                if sub_res.empty:
                    continue
                sub_res = sub_res.sort_values('北京时间')

                sub_tke = dfTKE[dfTKE['离地高度 (m)'] == h_float].copy()
                if sub_tke.empty:
                    sub_res['TKE (m2/s2)'] = np.nan
                    merged_list.append(sub_res)
                    continue
                sub_tke = sub_tke.sort_values('time_bj')

                sub_merged = pd.merge_asof(
                    sub_res,
                    sub_tke[['time_bj', 'TKE (m2/s2)']],
                    left_on='北京时间',
                    right_on='time_bj',
                    direction='nearest'
                )
                sub_merged.drop(columns=['time_bj'], inplace=True)
                merged_list.append(sub_merged)

            if merged_list:
                result_df = pd.concat(merged_list, ignore_index=True)
                result_df = result_df.sort_values(['离地高度 (m)', '北京时间']).reset_index(drop=True)
            else:
                result_df['TKE (m2/s2)'] = np.nan

            if "L (m)" in result_df.columns and result_df["L (m)"].notna().sum() == 0:
                print(f"[{site_tag if site_tag else station_name}] ⚠ 合并到结果表后的 WRF L 全为空。")
                print_wrf_L_diagnostics(dfL, l_diag, site_tag=site_tag if site_tag else station_name)

            native_tke_period_df = compute_native_tke_period_mean(
                dfL, TKE_profiles, z_tke_agl, avg_start=avg_start, avg_end=avg_end
            )

        except Exception as e:
            print(f"计算或合并 L/TKE 时出错: {e}")
            print(f"[{site_tag if site_tag else station_name}] ⚠ 由于 wrfout 处理失败，WRF 的 L/PBLH/TKE 将全部填充为 NaN。")
            result_df['L (m)'] = np.nan
            result_df['PBLH (m)'] = np.nan
            result_df['TKE (m2/s2)'] = np.nan
            for _c in ['UST (m/s)', 'HFX (W/m2)', 'PSFC (Pa)', 'T2 (K)', 'Z0 (m)']:
                if _c not in result_df.columns:
                    result_df[_c] = np.nan
    else:
        print("wrfout_dir 未设置或不存在，L/TKE 将填充为 NaN。")
        result_df['L (m)'] = np.nan
        result_df['PBLH (m)'] = np.nan
        result_df['TKE (m2/s2)'] = np.nan

    # Added: calculate the theoretical MOST wind speed U_MO_profile. This column does not alter existing outputs and only provides a candidate physical variable for ML/PINN ablation.
    result_df = add_most_profile_to_result(result_df)

    use_stability_heights = stability_heights
    if stability_heights is not None:
        req1, req2 = stability_heights
        unique_heights = sorted(result_df['离地高度 (m)'].unique())
        if req1 not in unique_heights or req2 not in unique_heights:
            print(f"站点 {site_tag if site_tag else station_name} 不包含指定 Ri 高度 {stability_heights}，自动改用最近可用高度。")
            use_stability_heights = stability_heights

    result_df = compute_stability_bulk_Ri(result_df, stability_heights=use_stability_heights)
    period_mean_df = compute_period_mean(result_df, avg_start=avg_start, avg_end=avg_end)

    os.makedirs(output_dir, exist_ok=True)
    tag = site_tag if site_tag else station_name
    output_excel = os.path.join(output_dir, f"风资源数据_{tag}_{domain_id}.xlsx")
    full_csv = os.path.join(output_dir, f"风资源数据_{tag}_{domain_id}_all_10min.csv")
    period_csv = os.path.join(output_dir, f"风资源数据_{tag}_{domain_id}_period_mean.csv")
    native_csv = os.path.join(output_dir, f"WRF_native_period_mean_{tag}_{domain_id}.csv")
    native_tke_csv = os.path.join(output_dir, f"WRF_native_TKE_period_mean_{tag}_{domain_id}.csv")

    result_df.to_csv(full_csv, index=False, encoding='utf-8-sig')
    print(f"全部 10 分钟平均数据已保存为 CSV: {full_csv}")

    # Added: export the ML-ready 10 m table. Existing outputs are unchanged; this is an additional file.
    ml_ready_ref10_df = None
    if EXPORT_ML_READY_REF10:
        try:
            # obs_csv is injected from the main program through a function attribute; if unset, only ML-ready export is skipped and the original workflow is unaffected.
            _obs_csv_for_ml = getattr(process_monitoring_data, '_current_obs_csv', None)
            if _obs_csv_for_ml is None:
                print(f"[{tag}] ⚠ 未提供 obs_csv，跳过 ML-ready ref10 导出。")
            else:
                ml_ready_ref10_df = export_ml_ready_ref10(
                    result_df=result_df,
                    obs_csv=_obs_csv_for_ml,
                    output_dir=output_dir,
                    site_tag=tag,
                    domain_id=domain_id,
                    ref_height=REF_HEIGHT_ML,
                    tolerance=ML_MERGE_TOLERANCE,
                )
        except Exception as e:
            print(f"[{tag}] ⚠ ML-ready ref10 导出失败，但原有输出继续: {e}")
            print(traceback.format_exc())

    if period_mean_df is not None:
        period_mean_df.to_csv(period_csv, index=False, encoding='utf-8-sig')
        print(f"时间段平均剖面已保存为 CSV: {period_csv}")
    if native_period_df is not None:
        native_period_df.to_csv(native_csv, index=False, encoding='utf-8-sig')
        print(f"WRF原始层时间段平均已保存为 CSV: {native_csv}")
    if native_tke_period_df is not None:
        native_tke_period_df.to_csv(native_tke_csv, index=False, encoding='utf-8-sig')
        print(f"WRF原始层TKE时间段平均已保存为 CSV: {native_tke_csv}")

    with pd.ExcelWriter(output_excel) as writer:
        result_df.to_excel(writer, sheet_name="所有高度", index=False)

        for height in sorted(result_df['离地高度 (m)'].unique()):
            sub_df = result_df[result_df['离地高度 (m)'] == height]
            if not sub_df.empty:
                sheet_name = safe_sheet_name(f"{int(round(height))}米")
                sub_df.to_excel(writer, sheet_name=sheet_name, index=False)

        mapping_df.to_excel(writer, sheet_name="高度匹配说明", index=False)

        if period_mean_df is not None:
            period_mean_df.to_excel(writer, sheet_name="时间段平均", index=False)
        if native_period_df is not None:
            native_period_df.to_excel(writer, sheet_name="WRF原始层平均", index=False)
        if native_tke_period_df is not None:
            native_tke_period_df.to_excel(writer, sheet_name="WRF原始层TKE", index=False)
        if ml_ready_ref10_df is not None and not ml_ready_ref10_df.empty:
            ml_ready_ref10_df.to_excel(writer, sheet_name="ML_ready_ref10", index=False)

    print(f"Excel 文件生成完成: {output_excel}")

    return {
        "output_excel": output_excel,
        "result_df": result_df,
        "period_mean_df": period_mean_df,
        "mapping_df": mapping_df,
        "native_period_df": native_period_df,
        "native_tke_period_df": native_tke_period_df,
        "tke_source_name": tke_source_name,
        "profile_windows": profile_windows,
        "height_levels": height_levels,
        "dfL": dfL,
        "TKE_profiles": TKE_profiles,
        "z_tke_agl": z_tke_agl,
        "ml_ready_ref10_df": ml_ready_ref10_df,
    }



# ====================== Windographer txt observation adapter (Huarui_A case; path: hurui/A) ======================

def _read_text_with_encoding_fallback(file_path):
    """Read Windographer txt with common encodings; Chinese-export files may be GB18030 or UTF-8-BOM."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin1"):
        try:
            return PathLikeText(file_path).read_text(encoding=enc), enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取文本文件编码: {file_path}; last error = {last_err}")


class PathLikeText:
    """Tiny wrapper to avoid adding pathlib dependency to the original script style."""
    def __init__(self, path):
        self.path = os.fspath(path)

    def read_text(self, encoding):
        with open(self.path, "r", encoding=encoding) as f:
            return f.read()


def _resolve_windographer_file(file_name_or_path):
    """
    Resolve a Windographer file path.
    Search order:
      1) absolute path in config;
      2) WINDOGRAPHER_OBS_DIR;
      3) current working directory;
      4) script directory.
    """
    candidates = []
    p0 = os.fspath(file_name_or_path)
    if os.path.isabs(p0):
        candidates.append(p0)
    else:
        candidates.append(os.path.join(WINDOGRAPHER_OBS_DIR, p0))
        candidates.append(os.path.join(os.getcwd(), p0))
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates.append(os.path.join(script_dir, p0))
        except Exception:
            pass

    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p

    raise FileNotFoundError(
        "未找到 Windographer 观测文件: "
        + str(file_name_or_path)
        + "\n已尝试:\n  - "
        + "\n  - ".join(candidates)
        + "\n请把 Windographer txt 放到 WINDOGRAPHER_OBS_DIR，或把配置改成绝对路径。"
    )


def read_windographer_table(file_path):
    """
    Read Windographer exported txt.

    The file has metadata lines before the real tab-separated table.
    The real table starts at the line beginning with 'Date/Time'.
    """
    file_path = _resolve_windographer_file(file_path)
    txt, enc = _read_text_with_encoding_fallback(file_path)
    lines = txt.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip("\ufeff").strip().startswith("Date/Time"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"未找到 Date/Time 表头行: {file_path}")

    table_txt = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(table_txt), sep="\t", engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    if df.empty:
        raise ValueError(f"Windographer 表为空: {file_path}")

    if df.columns[0] != "Date/Time":
        df = df.rename(columns={df.columns[0]: "Date/Time"})

    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df = df.dropna(subset=["Date/Time"]).copy()
    for c in df.columns:
        if c != "Date/Time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[Windographer] 读取 {os.path.basename(file_path)}: {len(df)} rows, encoding={enc}, "
          f"{df['Date/Time'].min()} -> {df['Date/Time'].max()}")
    return df


def _identify_windographer_columns(df, expect_sd=False):
    """Identify speed, direction and optional standard-deviation columns from Windographer headers."""
    cols = [c for c in df.columns if c != "Date/Time"]

    def low(c):
        return str(c).lower()

    sd_cols = [c for c in cols if re.search(r"(sd|stdev|standard)", low(c))]
    speed_cols = [
        c for c in cols
        if c not in sd_cols and (
            "speed" in low(c)
            or "_ws_" in low(c)
            or "[m/s]" in low(c)
        )
    ]
    direction_cols = [
        c for c in cols
        if (
            "direction" in low(c)
            or "_wd_" in low(c)
            or "wd_" in low(c)
            or "[°" in low(c)
            or "[癩" in low(c)
        )
    ]

    if not speed_cols:
        raise ValueError(f"无法识别风速列，现有列: {df.columns.tolist()}")
    if not direction_cols:
        raise ValueError(f"无法识别风向列，现有列: {df.columns.tolist()}")
    if expect_sd and not sd_cols:
        raise ValueError(f"无法识别风速标准差列，现有列: {df.columns.tolist()}")

    return speed_cols[0], direction_cols[0], (sd_cols[0] if sd_cols else None)


def _build_windographer_height_obs(power_file, turbulence_file, height, wind_source="power"):
    """
    Build one-height observation DataFrame with original script-compatible column names.

    Output columns:
      rec_time
      wspd{height}m, wdir{height}m
      tbl{height}m       = speed SD / speed
      tke{height}m       = 1.5 * SD^2
      raw helper columns = wspd_raw..., sd...
    """
    h = int(round(float(height)))
    power_df = read_windographer_table(power_file)
    turb_df = read_windographer_table(turbulence_file)

    p_ws, p_wd, _ = _identify_windographer_columns(power_df, expect_sd=False)
    t_ws, t_wd, t_sd = _identify_windographer_columns(turb_df, expect_sd=True)

    power_part = pd.DataFrame({
        "rec_time": power_df["Date/Time"],
        f"wspd_power{h}m": pd.to_numeric(power_df[p_ws], errors="coerce"),
        f"wdir_power{h}m": pd.to_numeric(power_df[p_wd], errors="coerce"),
    })

    raw_ws = pd.to_numeric(turb_df[t_ws], errors="coerce")
    raw_wd = pd.to_numeric(turb_df[t_wd], errors="coerce")
    raw_sd = pd.to_numeric(turb_df[t_sd], errors="coerce")
    ti = raw_sd / raw_ws.replace(0, np.nan)
    tke_direct = 1.5 * raw_sd ** 2

    turb_part = pd.DataFrame({
        "rec_time": turb_df["Date/Time"],
        f"wspd_raw{h}m": raw_ws,
        f"wdir_raw{h}m": raw_wd,
        f"sd{h}m": raw_sd,
        f"tbl{h}m": ti,
        f"tke{h}m": tke_direct,
    })

    merged = pd.merge(power_part, turb_part, on="rec_time", how="outer").sort_values("rec_time")

    source = str(wind_source).lower().strip()
    if source == "power":
        merged[f"wspd{h}m"] = merged[f"wspd_power{h}m"]
        merged[f"wdir{h}m"] = merged[f"wdir_power{h}m"]
    elif source == "turbulence":
        merged[f"wspd{h}m"] = merged[f"wspd_raw{h}m"]
        merged[f"wdir{h}m"] = merged[f"wdir_raw{h}m"]
    else:
        raise ValueError("WINDOGRAPHER_WIND_SOURCE_FOR_VALIDATION must be 'power' or 'turbulence'.")

    return merged



def _extract_height_from_speed_column(col_name):
    """
    Extract height from all-height speed columns, e.g.:
      F1_WS_160_202.5_K620A [m/s] -> 160
      Speed 130 m Synthesized [m/s] -> 130
    """
    c = str(col_name)
    m = re.search(r"_WS_(\d+(?:\.\d+)?)_", c)
    if m:
        return float(m.group(1))
    m = re.search(r"Speed\s+(\d+(?:\.\d+)?)\s*m", c, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def build_all_height_speed_obs(all_height_speed_file):
    """
    Convert supplemental all-height wind-speed txt into columns:
      rec_time, wspd10m, wspd40m, wspd80m, ...
    It contains wind speed only; direction/TI/TKE still come from the original
    power/turbulence files when available.
    """
    df = read_windographer_table(all_height_speed_file)
    out = pd.DataFrame({"rec_time": df["Date/Time"]})
    height_cols = []
    for c in df.columns:
        if c == "Date/Time":
            continue
        h = _extract_height_from_speed_column(c)
        if h is None:
            continue
        h_label = int(round(float(h)))
        out[f"wspd{h_label}m"] = pd.to_numeric(df[c], errors="coerce")
        out[f"wspd_allheight{h_label}m"] = pd.to_numeric(df[c], errors="coerce")
        height_cols.append((h_label, c))
    if not height_cols:
        raise ValueError(f"各高度风速文件中没有识别到风速高度列: {all_height_speed_file}")
    print("[All-height speed] 已识别高度列: " + ", ".join([f"{h}m<-{c}" for h, c in height_cols]))
    return out.sort_values("rec_time")


def build_windographer_tower_obs_csv(tower_cfg, output_dir, wind_source=None):
    """
    Convert one tower's Windographer txt files into the original obs CSV format.

    V17 huarui_A addition:
    - If all_height_speed_file exists and USE_SUPPLEMENTAL_ALL_HEIGHT_SPEED=True,
      wind-speed observation columns are expanded from the supplemental all-height
      speed file. This supports speed-profile comparison at 10/40/80/100/130/140/160 m
      when available.
    - Direction/TI/TKE still come from the previous 130/160 power/turbulence files.
    """
    wind_source = WINDOGRAPHER_WIND_SOURCE_FOR_VALIDATION if wind_source is None else wind_source
    site_tag = tower_cfg["site_tag"]
    os.makedirs(output_dir, exist_ok=True)

    tower_df = None
    for h, file_cfg in sorted(tower_cfg["heights"].items()):
        h_df = _build_windographer_height_obs(
            power_file=file_cfg["power_file"],
            turbulence_file=file_cfg["turbulence_file"],
            height=h,
            wind_source=wind_source,
        )
        tower_df = h_df if tower_df is None else pd.merge(tower_df, h_df, on="rec_time", how="outer")

    # New supplemental all-height speed file has higher-priority wind-speed columns.
    # It may include 10/40/80/100/130/140/160 m for C039801 and
    # 40/80/100/130/140/160 m for C039802.
    if USE_SUPPLEMENTAL_ALL_HEIGHT_SPEED and tower_cfg.get("all_height_speed_file"):
        try:
            speed_profile_df = build_all_height_speed_obs(tower_cfg["all_height_speed_file"])
            tower_df = pd.merge(tower_df, speed_profile_df, on="rec_time", how="outer", suffixes=("", "_allheight_dup"))

            # Replace/define main wspd{height}m columns by the supplemental file values.
            for c in list(tower_df.columns):
                if re.fullmatch(r"wspd_allheight\d+m", str(c)):
                    h = _match_obs_height(c.replace("wspd_allheight", "wspd"), "wspd")
                    if h is None:
                        continue
                    h_label = int(round(float(h)))
                    main_col = f"wspd{h_label}m"
                    tower_df[main_col] = pd.to_numeric(tower_df[c], errors="coerce").combine_first(
                        pd.to_numeric(tower_df[main_col], errors="coerce") if main_col in tower_df.columns else pd.Series(np.nan, index=tower_df.index)
                    )

            dup_cols = [c for c in tower_df.columns if str(c).endswith("_allheight_dup")]
            if dup_cols:
                tower_df = tower_df.drop(columns=dup_cols)
        except Exception as e:
            print(f"[{site_tag}] ⚠ 各高度风速补充文件读取失败，将仅使用 130/160 原始配置: {e}")
            print(traceback.format_exc())

    tower_df = tower_df.sort_values("rec_time").replace([np.inf, -np.inf], np.nan)
    out_csv = os.path.join(output_dir, f"obs_{site_tag}_windographer_converted.csv")
    tower_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[{site_tag}] Windographer obs 已转换: {out_csv}")
    print(f"[{site_tag}] case={CASE_NAME}; wind_source={wind_source}; columns={tower_df.columns.tolist()}")
    return out_csv

def build_all_windographer_obs_csvs(output_dir):
    """
    Build converted obs CSVs and return updated STATIONS entries.
    """
    converted_dir = os.path.join(output_dir, "_converted_obs")
    updated = []
    for cfg in WINDOGRAPHER_TOWER_CONFIGS:
        out_csv = build_windographer_tower_obs_csv(cfg, converted_dir)
        updated.append({
            "station_name": cfg["station_name"],
            "site_tag": cfg["site_tag"],
            "obs_csv": out_csv,
            "nest_level": int(cfg.get("nest_level", NEST_LEVEL)),
        })
    return updated


# ====================== Main program ======================

if __name__ == "__main__":
    if AVG_START_STR and AVG_END_STR:
        avg_start = datetime.strptime(AVG_START_STR, "%Y-%m-%d %H:%M")
        avg_end = datetime.strptime(AVG_END_STR, "%Y-%m-%d %H:%M")
    else:
        avg_start = avg_end = None

    print("=" * 90)
    print(f"开始批量处理 {len(STATIONS)} 个站点, 默认嵌套层: d{NEST_LEVEL:02d}；各站点可单独设置 nest_level")
    print(f"高度匹配方式: {HEIGHT_MATCH_METHOD}, 越界策略: {OUT_OF_RANGE_POLICY}")
    print("=" * 90)

    try:
        end_time_utc = get_simulation_end_time(WRF_OUTPUT_DIR)
        print(f"模拟结束时间(UTC): {end_time_utc}")
    except Exception as e:
        print(f"获取模拟结束时间失败: {str(e)}")
        end_time_utc = None

    if AUTO_BUILD_WINDOGRAPHER_OBS_CSV:
        try:
            STATIONS = build_all_windographer_obs_csvs(OUTPUT_DIR)
        except Exception as e:
            print(f"⚠ Windographer 观测 txt 自动转换失败: {e}")
            print(traceback.format_exc())
            print("将继续使用 STATIONS 中已有 obs_csv；若 obs_csv 为空，后续站点会失败。")

    for site in STATIONS:
        station_name = site["station_name"]
        site_tag = site["site_tag"]
        obs_csv = site["obs_csv"]
        site_nest_level = int(site.get("nest_level", NEST_LEVEL))
        site_domain_id = f"d{site_nest_level:02d}"

        print("\n" + "=" * 90)
        print(f"开始处理站点: {site_tag} ({station_name}), 使用嵌套域: {site_domain_id}")
        print("=" * 90)

        site_output_dir = os.path.join(OUTPUT_DIR, site_tag)
        os.makedirs(site_output_dir, exist_ok=True)

        try:
            obs_union_heights, obs_height_info = get_obs_union_heights_from_csv(
                obs_csv, prefixes=("wspd", "wdir", "tp", "tbl")
            )
            print(f"[{site_tag}] 观测可用高度汇总 = {obs_union_heights}")
            print(f"[{site_tag}] 分变量高度 = {obs_height_info}")

            # Added: explicitly include the 10 m reference height and 50 m stability-diagnostic height.
            # Existing observation-height outputs are unchanged; WRF interpolation output is only added at these heights.
            obs_union_heights_aug = sorted(set(float(h) for h in obs_union_heights).union(FORCE_EXTRA_HEIGHTS))
            print(f"[{site_tag}] 加入 FORCE_EXTRA_HEIGHTS 后的输出高度 = {obs_union_heights_aug}")

            # Pass the current obs_csv to the ML-ready export function in process_monitoring_data without changing the original function-argument compatibility.
            process_monitoring_data._current_obs_csv = obs_csv

            result_pack = process_monitoring_data(
                wrf_output_dir=WRF_OUTPUT_DIR,
                output_dir=site_output_dir,
                heights=obs_union_heights_aug,
                nest_level=site_nest_level,
                station_name=station_name,
                avg_start=avg_start,
                avg_end=avg_end,
                stability_heights=DEFAULT_STABILITY_HEIGHTS,
                wrfout_dir=WRFOUT_DIR,
                height_match_method=HEIGHT_MATCH_METHOD,
                out_of_range_policy=OUT_OF_RANGE_POLICY,
                site_tag=site_tag,
            )

            output_file = result_pack["output_excel"]
            result_df = result_pack["result_df"]
            period_mean_df = result_pack["period_mean_df"]
            native_period_df = result_pack["native_period_df"]
            native_tke_period_df = result_pack["native_tke_period_df"]
            profile_windows = result_pack["profile_windows"]
            height_levels = result_pack["height_levels"]
            dfL = result_pack["dfL"]
            TKE_profiles = result_pack["TKE_profiles"]
            z_tke_agl = result_pack["z_tke_agl"]
            ml_ready_ref10_df = result_pack.get("ml_ready_ref10_df")

            if output_file and end_time_utc is not None:
                if check_simulation_end(output_file, end_time_utc):
                    print(f"[{site_tag}] ✅ 模拟已达到结束时间。")
                else:
                    print(f"[{site_tag}] ✅ 处理完成，结果已更新。")

            # Time series and mean values.
            if avg_start is not None and avg_end is not None and (result_df is not None) and (not result_df.empty):
                Ri_mean, L_mean = plot_Ri_L_timeseries(
                    result_df,
                    avg_start,
                    avg_end,
                    output_dir=site_output_dir,
                    fig_name=f"Ri_L_timeseries_{site_tag}_{site_domain_id}.png"
                )
                mean_PBLH = plot_PBLH_timeseries(
                    result_df,
                    avg_start,
                    avg_end,
                    output_dir=site_output_dir,
                    fig_name=f"PBLH_timeseries_{site_tag}_{site_domain_id}.png"
                )

                mean_txt = os.path.join(site_output_dir, f"Ri_L_mean_{site_tag}_{site_domain_id}.txt")
                with open(mean_txt, "w", encoding="utf-8") as f:
                    f.write(f"Ri_mean,{Ri_mean:.6f}\n")
                    f.write(f"L_mean,{L_mean:.6f}\n")
                    if pd.notna(mean_PBLH):
                        f.write(f"PBLH_mean,{mean_PBLH:.2f}\n")
                print(f"[{site_tag}] Ri / L / PBLH 平均值已保存到: {mean_txt}")

            # OBS-WRF profile comparison: calculate statistics over the actual shared time window so both means use the same period.
            common_periods = {
                "wspd": get_common_profile_period(obs_csv, [w["window_end_time"] for w in profile_windows], "wspd", avg_start, avg_end),
                "wdir": get_common_profile_period(obs_csv, [w["window_end_time"] for w in profile_windows], "wdir", avg_start, avg_end),
                "tp":   get_common_profile_period(obs_csv, [w["window_end_time"] for w in profile_windows], "tp", avg_start, avg_end),
                "tke":  get_common_profile_period(obs_csv, [] if dfL is None else dfL["time_bj"], "tke", avg_start, avg_end),
            }

            obs_profiles = {k: pd.DataFrame() for k in ["wspd", "wdir", "tp", "tbl", "tke", "l"]}
            wrf_profile_map = {"wspd": None, "wdir": None, "tp": None, "tke": None}

            for var in ["wspd", "wdir", "tp"]:
                common_start, common_end = common_periods[var]
                if common_start is None or common_end is None:
                    print(f"[{site_tag}] {var} 廓线对比缺少共同时间窗，跳过该变量。")
                    continue
                print(f"[{site_tag}] {var} 廓线对比共同时间窗: {common_start} ~ {common_end}")
                obs_pack_var = build_obs_period_profiles(obs_csv, common_start, common_end, instrument_name=site_tag)
                obs_profiles[var] = obs_pack_var.get(var, pd.DataFrame())
                wrf_profile_map[var] = compute_native_period_mean(profile_windows, height_levels, avg_start=common_start, avg_end=common_end)

            common_start_tke, common_end_tke = common_periods["tke"]
            if common_start_tke is None or common_end_tke is None:
                print(f"[{site_tag}] tke 廓线对比缺少共同时间窗，跳过该变量。")
            else:
                print(f"[{site_tag}] tke 廓线对比共同时间窗: {common_start_tke} ~ {common_end_tke}")
                obs_pack_tke = build_obs_period_profiles(obs_csv, common_start_tke, common_end_tke, instrument_name=site_tag)
                obs_profiles["tke"] = obs_pack_tke.get("tke", pd.DataFrame())
                if dfL is not None and TKE_profiles is not None and z_tke_agl is not None:
                    wrf_profile_map["tke"] = compute_native_tke_period_mean(dfL, TKE_profiles, z_tke_agl, avg_start=common_start_tke, avg_end=common_end_tke)

            # Also export a site-level mean L using the requested time window and cleaned OBS.
            obs_profiles["l"] = build_obs_period_profiles(obs_csv, avg_start, avg_end, instrument_name=site_tag).get("l", pd.DataFrame())

            # Do not use the common-window mean directly as the final comparison output here.
            # Final comparison plots, CSV files, and Excel files all use the matched-sample mean;
            # wrf_profile_map is retained only as background reference using native model-level points.

            # Profile comparison: wind speed, wind direction, temperature, and TKE (all use native WRF level points without connecting lines and the same time window as OBS).
            if wrf_profile_map.get("wspd") is not None:
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["wspd"], obs_profiles.get("wspd"),
                    wrf_x_col="WRF原始层平均风速 (m/s)",
                    obs_x_col=OBS_VAR_META["wspd"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_ws_profile_{site_tag}_{site_domain_id}_full.png",
                    xlabel="Wind speed (m s$^{-1}$)",
                    title="WRF vs Observations (Wind-speed profile)",
                    zmax=None
                )
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["wspd"], obs_profiles.get("wspd"),
                    wrf_x_col="WRF原始层平均风速 (m/s)",
                    obs_x_col=OBS_VAR_META["wspd"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_ws_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                    xlabel="Wind speed (m s$^{-1}$)",
                    title="WRF vs Observations (Wind-speed profile)",
                    zmax=PROFILE_ZOOM_MAX
                )

            if wrf_profile_map.get("wdir") is not None:
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["wdir"], obs_profiles.get("wdir"),
                    wrf_x_col="WRF原始层平均风向 (°)",
                    obs_x_col=OBS_VAR_META["wdir"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_wd_profile_{site_tag}_{site_domain_id}_full.png",
                    xlabel="Wind direction (deg)",
                    title="WRF vs Observations (Wind-direction profile)",
                    zmax=None
                )
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["wdir"], obs_profiles.get("wdir"),
                    wrf_x_col="WRF原始层平均风向 (°)",
                    obs_x_col=OBS_VAR_META["wdir"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_wd_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                    xlabel="Wind direction (deg)",
                    title="WRF vs Observations (Wind-direction profile)",
                    zmax=PROFILE_ZOOM_MAX
                )

            if wrf_profile_map.get("tp") is not None:
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["tp"], obs_profiles.get("tp"),
                    wrf_x_col="WRF原始层平均温度 (°C)",
                    obs_x_col=OBS_VAR_META["tp"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_tp_profile_{site_tag}_{site_domain_id}_full.png",
                    xlabel="Temperature (°C)",
                    title="WRF vs Observations (Temperature profile)",
                    zmax=None
                )
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["tp"], obs_profiles.get("tp"),
                    wrf_x_col="WRF原始层平均温度 (°C)",
                    obs_x_col=OBS_VAR_META["tp"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_tp_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                    xlabel="Temperature (°C)",
                    title="WRF vs Observations (Temperature profile)",
                    zmax=PROFILE_ZOOM_MAX
                )

            if wrf_profile_map.get("tke") is not None:
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["tke"], obs_profiles.get("tke"),
                    wrf_x_col="WRF原始层平均TKE (m2/s2)",
                    obs_x_col=OBS_VAR_META["tke"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_tke_profile_{site_tag}_{site_domain_id}_full.png",
                    xlabel="TKE (m$^2$ s$^{-2}$)",
                    title="WRF vs Observations (TKE profile)",
                    zmax=None
                )
                plot_obs_vs_wrf_profile_points(
                    wrf_profile_map["tke"], obs_profiles.get("tke"),
                    wrf_x_col="WRF原始层平均TKE (m2/s2)",
                    obs_x_col=OBS_VAR_META["tke"]["obs_col"],
                    output_dir=site_output_dir,
                    fig_name=f"WRF_vs_obs_tke_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                    xlabel="TKE (m$^2$ s$^{-2}$)",
                    title="WRF vs Observations (TKE profile)",
                    zmax=PROFILE_ZOOM_MAX
                )

            # WRF-only native-level profiles and TKE native-level profiles are still exported over the user-specified broad time window.
            if avg_start is not None and avg_end is not None and native_period_df is not None:
                plot_wrf_only_profiles(
                    native_period_df,
                    output_dir=site_output_dir,
                    prefix=f"WRF_profile_{site_tag}_{site_domain_id}",
                    zmax=PROFILE_ZOOM_MAX
                )

                if native_tke_period_df is not None:
                    plot_native_tke_profile(
                        native_tke_period_df,
                        output_dir=site_output_dir,
                        fig_name=f"TKE_profile_{site_tag}_{site_domain_id}.png",
                        zmax=PROFILE_ZOOM_MAX
                    )

            # Time-series comparison: wind speed / wind direction / L.
            if avg_start is not None and avg_end is not None and result_df is not None and not result_df.empty:
                obs_ws_ts = build_obs_timeseries_long(obs_csv, avg_start, avg_end, "wspd", heights=obs_height_info.get("wspd"))
                obs_wd_ts = build_obs_timeseries_long(obs_csv, avg_start, avg_end, "wdir", heights=obs_height_info.get("wdir"))
                obs_tp_ts = build_obs_timeseries_long(obs_csv, avg_start, avg_end, "tp", heights=obs_height_info.get("tp"))
                obs_tke_ts = build_obs_timeseries_long(obs_csv, avg_start, avg_end, "tke", heights=obs_height_info.get("tbl"))
                obs_l_ts = build_obs_timeseries_long(obs_csv, avg_start, avg_end, "l")

                wrf_ws_ts = result_df[["北京时间", "离地高度 (m)", "十分钟平均风速 (m/s)"]].copy()
                wrf_wd_ts = result_df[["北京时间", "离地高度 (m)", "十分钟平均风向 (°)"]].copy()
                wrf_tp_ts = result_df[["北京时间", "离地高度 (m)", "十分钟平均温度 (°C)"]].copy()
                wrf_tke_ts = result_df[["北京时间", "离地高度 (m)", "TKE (m2/s2)"]].copy() if "TKE (m2/s2)" in result_df.columns else pd.DataFrame()
                wrf_l_ts = result_df[["北京时间", "L (m)"]].drop_duplicates(subset=["北京时间"]).sort_values("北京时间").copy()

                ws_compare = merge_obs_wrf_timeseries(obs_ws_ts, wrf_ws_ts, obs_col=OBS_VAR_META["wspd"]["obs_col"], wrf_col="十分钟平均风速 (m/s)", tolerance=OBS_WRF_MATCH_TOLERANCE)
                wd_compare = merge_obs_wrf_timeseries(obs_wd_ts, wrf_wd_ts, obs_col=OBS_VAR_META["wdir"]["obs_col"], wrf_col="十分钟平均风向 (°)", tolerance=OBS_WRF_MATCH_TOLERANCE)
                tp_compare = merge_obs_wrf_timeseries(obs_tp_ts, wrf_tp_ts, obs_col=OBS_VAR_META["tp"]["obs_col"], wrf_col="十分钟平均温度 (°C)", tolerance=OBS_WRF_MATCH_TOLERANCE)
                tke_compare = merge_obs_wrf_timeseries(obs_tke_ts, wrf_tke_ts, obs_col=OBS_VAR_META["tke"]["obs_col"], wrf_col="TKE (m2/s2)", tolerance=OBS_WRF_MATCH_TOLERANCE) if not wrf_tke_ts.empty else pd.DataFrame()
                l_compare = merge_obs_wrf_scalar_timeseries(obs_l_ts, wrf_l_ts, obs_col=OBS_VAR_META["l"]["obs_col"], wrf_col="L (m)", tolerance=OBS_WRF_MATCH_TOLERANCE)

                print(f"[{site_tag}] 时序匹配结果: ws={ws_compare.shape}, wd={wd_compare.shape}, tp={tp_compare.shape}, tke={tke_compare.shape}, L={l_compare.shape}")

                full_ws = build_full_timeseries_export(obs_ws_ts, wrf_ws_ts, OBS_VAR_META["wspd"]["obs_col"], "十分钟平均风速 (m/s)", is_scalar=False)
                full_wd = build_full_timeseries_export(obs_wd_ts, wrf_wd_ts, OBS_VAR_META["wdir"]["obs_col"], "十分钟平均风向 (°)", is_scalar=False)
                full_L = build_full_timeseries_export(obs_l_ts, wrf_l_ts, OBS_VAR_META["l"]["obs_col"], "L (m)", is_scalar=True)

                matched_ws = build_matched_timeseries_export(ws_compare, OBS_VAR_META["wspd"]["obs_col"], "十分钟平均风速 (m/s)", is_scalar=False)
                matched_wd = build_matched_timeseries_export(wd_compare, OBS_VAR_META["wdir"]["obs_col"], "十分钟平均风向 (°)", is_scalar=False)
                matched_L = build_matched_timeseries_export(l_compare, OBS_VAR_META["l"]["obs_col"], "L (m)", is_scalar=True)

                export_map = {
                    f"full_timeseries_ws_{site_tag}_{site_domain_id}.csv": full_ws,
                    f"matched_timeseries_ws_{site_tag}_{site_domain_id}.csv": matched_ws,
                    f"full_timeseries_wd_{site_tag}_{site_domain_id}.csv": full_wd,
                    f"matched_timeseries_wd_{site_tag}_{site_domain_id}.csv": matched_wd,
                    f"full_timeseries_L_{site_tag}_{site_domain_id}.csv": full_L,
                    f"matched_timeseries_L_{site_tag}_{site_domain_id}.csv": matched_L,
                }
                for fname, df_exp in export_map.items():
                    if df_exp is not None and not df_exp.empty:
                        fpath = os.path.join(site_output_dir, fname)
                        df_exp.to_csv(fpath, index=False, encoding="utf-8-sig")
                        print(f"[{site_tag}] 已保存: {fpath}")

                if not full_ws.empty:
                    plot_timeseries_comparison_dual(
                        obs_ws_ts, wrf_ws_ts,
                        obs_col=OBS_VAR_META["wspd"]["obs_col"],
                        wrf_col="十分钟平均风速 (m/s)",
                        output_dir=site_output_dir,
                        fig_prefix=f"Timeseries_ws_compare_{site_tag}_{site_domain_id}",
                        y_label="Wind speed (m s$^{-1}$)",
                        is_direction=False
                    )
                if not full_wd.empty:
                    plot_timeseries_comparison_dual(
                        obs_wd_ts, wrf_wd_ts,
                        obs_col=OBS_VAR_META["wdir"]["obs_col"],
                        wrf_col="十分钟平均风向 (°)",
                        output_dir=site_output_dir,
                        fig_prefix=f"Timeseries_wd_compare_{site_tag}_{site_domain_id}",
                        y_label="Wind direction (deg)",
                        is_direction=True
                    )
                if not full_L.empty:
                    plot_scalar_timeseries_comparison_dual(
                        obs_l_ts, wrf_l_ts,
                        obs_col=OBS_VAR_META["l"]["obs_col"],
                        wrf_col="L (m)",
                        output_dir=site_output_dir,
                        fig_name=f"Timeseries_L_compare_{site_tag}_{site_domain_id}.png",
                        y_label="Monin-Obukhov length L (m)",
                        title=f"L comparison ({site_tag})"
                    )
                else:
                    if obs_l_ts.empty:
                        print(f"[{site_tag}] 观测文件中没有可用的 L 时序，跳过 obs-vs-WRF L 图。")
                    elif wrf_l_ts.empty or wrf_l_ts['L (m)'].notna().sum() == 0:
                        print(f"[{site_tag}] WRF L 时序为空，无法生成 obs-vs-WRF L 图。")
                        print_wrf_L_diagnostics(dfL, l_diag, site_tag=site_tag)

                matched_profile_tables = {}
                aligned_profiles = {}
                compare_pack = {
                    "wspd": (ws_compare, OBS_VAR_META["wspd"]["obs_col"], "十分钟平均风速 (m/s)"),
                    "wdir": (wd_compare, OBS_VAR_META["wdir"]["obs_col"], "十分钟平均风向 (°)"),
                    "tp":   (tp_compare, OBS_VAR_META["tp"]["obs_col"],   "十分钟平均温度 (°C)"),
                    "tke":  (tke_compare, OBS_VAR_META["tke"]["obs_col"],  "TKE (m2/s2)"),
                }

                # Added: complete OBS vs WRF error statistics and error plots.
                # Errors are consistently defined as WRF - OBS; wind direction uses circular error to avoid artificial large errors across the 0/360-degree boundary.
                error_stats_tables = {}
                error_timeseries_tables = {}
                error_specs = {
                    "wspd": {"label": "Wind speed", "unit": "m/s", "is_direction": False, "is_scalar": False},
                    "wdir": {"label": "Wind direction", "unit": "deg", "is_direction": True,  "is_scalar": False},
                    "tp":   {"label": "Temperature", "unit": "°C",  "is_direction": False, "is_scalar": False},
                    "tke":  {"label": "TKE", "unit": "m2/s2", "is_direction": False, "is_scalar": False},
                    "L":    {"label": "Monin-Obukhov length", "unit": "m", "is_direction": False, "is_scalar": True},
                }

                for key, (cmp_df, obs_c, wrf_c) in compare_pack.items():
                    spec = error_specs[key]
                    err_stats, err_ts = export_error_diagnostics(
                        cmp_df, obs_c, wrf_c,
                        var_key=key,
                        var_label=spec["label"],
                        unit=spec["unit"],
                        output_dir=site_output_dir,
                        site_tag=site_tag,
                        domain_id=site_domain_id,
                        is_direction=spec["is_direction"],
                        is_scalar=spec["is_scalar"],
                    )
                    if err_stats is not None and not err_stats.empty:
                        error_stats_tables[key] = err_stats
                    if err_ts is not None and not err_ts.empty:
                        error_timeseries_tables[key] = err_ts

                err_stats_L, err_ts_L = export_error_diagnostics(
                    l_compare, OBS_VAR_META["l"]["obs_col"], "L (m)",
                    var_key="L",
                    var_label=error_specs["L"]["label"],
                    unit=error_specs["L"]["unit"],
                    output_dir=site_output_dir,
                    site_tag=site_tag,
                    domain_id=site_domain_id,
                    is_direction=False,
                    is_scalar=True,
                )
                if err_stats_L is not None and not err_stats_L.empty:
                    error_stats_tables["L"] = err_stats_L
                if err_ts_L is not None and not err_ts_L.empty:
                    error_timeseries_tables["L"] = err_ts_L

                if error_stats_tables:
                    error_stats_all = pd.concat(error_stats_tables.values(), ignore_index=True)
                    error_stats_all_csv = os.path.join(site_output_dir, f"error_stats_all_{site_tag}_{site_domain_id}.csv")
                    error_stats_all.to_csv(error_stats_all_csv, index=False, encoding="utf-8-sig")
                    print(f"[{site_tag}] 全变量误差统计已保存: {error_stats_all_csv}")
                else:
                    error_stats_all = pd.DataFrame()

                # Added: wind-rose / Weibull / wind-speed frequency-distribution error statistics and plots.
                # Note: this section uses matched wind-speed and wind-direction samples and again removes missing or physically unreasonable values.
                distribution_diag_tables = export_wind_distribution_diagnostics(
                    ws_compare, wd_compare,
                    output_dir=site_output_dir,
                    site_tag=site_tag,
                    domain_id=site_domain_id,
                )
                for key, (cmp_df, obs_c, wrf_c) in compare_pack.items():
                    if cmp_df is None or cmp_df.empty:
                        continue
                    summary_df, obs_aligned, wrf_aligned = build_matched_profile_summary(cmp_df, obs_c, wrf_c, instrument_name=site_tag)
                    if summary_df is None or summary_df.empty:
                        continue
                    summary_df["变量"] = key
                    matched_profile_tables[key] = summary_df
                    aligned_profiles[key] = (obs_aligned, wrf_aligned)
                    out_csv = os.path.join(site_output_dir, f"matched_profile_{key}_{site_tag}_{site_domain_id}.csv")
                    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                    print(f"[{site_tag}] matched 廓线统计已保存: {out_csv}")

                if "wspd" in aligned_profiles:
                    obs_aligned, wrf_aligned = aligned_profiles["wspd"]
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("wspd"), obs_aligned,
                        wrf_x_col="WRF原始层平均风速 (m/s)", obs_x_col=OBS_VAR_META["wspd"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_ws_profile_{site_tag}_{site_domain_id}_full.png",
                        xlabel="Wind speed (m s$^{-1}$)", title="WRF vs Observations (Wind-speed profile)",
                        zmax=None, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均风速 (m/s)"
                    )
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("wspd"), obs_aligned,
                        wrf_x_col="WRF原始层平均风速 (m/s)", obs_x_col=OBS_VAR_META["wspd"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_ws_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                        xlabel="Wind speed (m s$^{-1}$)", title="WRF vs Observations (Wind-speed profile)",
                        zmax=PROFILE_ZOOM_MAX, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均风速 (m/s)"
                    )
                if "wdir" in aligned_profiles:
                    obs_aligned, wrf_aligned = aligned_profiles["wdir"]
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("wdir"), obs_aligned,
                        wrf_x_col="WRF原始层平均风向 (°)", obs_x_col=OBS_VAR_META["wdir"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_wd_profile_{site_tag}_{site_domain_id}_full.png",
                        xlabel="Wind direction (deg)", title="WRF vs Observations (Wind-direction profile)",
                        zmax=None, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均风向 (°)"
                    )
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("wdir"), obs_aligned,
                        wrf_x_col="WRF原始层平均风向 (°)", obs_x_col=OBS_VAR_META["wdir"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_wd_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                        xlabel="Wind direction (deg)", title="WRF vs Observations (Wind-direction profile)",
                        zmax=PROFILE_ZOOM_MAX, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均风向 (°)"
                    )
                if "tp" in aligned_profiles:
                    obs_aligned, wrf_aligned = aligned_profiles["tp"]
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("tp"), obs_aligned,
                        wrf_x_col="WRF原始层平均温度 (°C)", obs_x_col=OBS_VAR_META["tp"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_tp_profile_{site_tag}_{site_domain_id}_full.png",
                        xlabel="Temperature (°C)", title="WRF vs Observations (Temperature profile)",
                        zmax=None, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均温度 (°C)"
                    )
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("tp"), obs_aligned,
                        wrf_x_col="WRF原始层平均温度 (°C)", obs_x_col=OBS_VAR_META["tp"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_tp_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                        xlabel="Temperature (°C)", title="WRF vs Observations (Temperature profile)",
                        zmax=PROFILE_ZOOM_MAX, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="十分钟平均温度 (°C)"
                    )
                if "tke" in aligned_profiles:
                    obs_aligned, wrf_aligned = aligned_profiles["tke"]
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("tke"), obs_aligned,
                        wrf_x_col="WRF原始层平均TKE (m2/s2)", obs_x_col=OBS_VAR_META["tke"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_tke_profile_{site_tag}_{site_domain_id}_full.png",
                        xlabel="TKE (m$^2$ s$^{-2}$)", title="WRF vs Observations (TKE profile)",
                        zmax=None, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="TKE (m2/s2)"
                    )
                    plot_obs_vs_wrf_profile_points(
                        wrf_profile_map.get("tke"), obs_aligned,
                        wrf_x_col="WRF原始层平均TKE (m2/s2)", obs_x_col=OBS_VAR_META["tke"]["obs_col"],
                        output_dir=site_output_dir,
                        fig_name=f"WRF_vs_obs_tke_profile_{site_tag}_{site_domain_id}_0-{int(PROFILE_ZOOM_MAX)}m.png",
                        xlabel="TKE (m$^2$ s$^{-2}$)", title="WRF vs Observations (TKE profile)",
                        zmax=PROFILE_ZOOM_MAX, wrf_aligned_df=wrf_aligned, wrf_aligned_x_col="TKE (m2/s2)"
                    )

                if os.path.exists(output_file):
                    sheet_map = {}
                    for key, df_prof in matched_profile_tables.items():
                        if df_prof is not None and not df_prof.empty:
                            sheet_map[f"matched_{key}_profile"] = df_prof
                    if not full_ws.empty:
                        sheet_map["full_ws_ts"] = full_ws
                    if not matched_ws.empty:
                        sheet_map["matched_ws_ts"] = matched_ws
                    if not full_wd.empty:
                        sheet_map["full_wd_ts"] = full_wd
                    if not matched_wd.empty:
                        sheet_map["matched_wd_ts"] = matched_wd
                    if not full_L.empty:
                        sheet_map["full_L_ts"] = full_L
                    if not matched_L.empty:
                        sheet_map["matched_L_ts"] = matched_L
                    if 'error_stats_all' in locals() and error_stats_all is not None and not error_stats_all.empty:
                        sheet_map["error_stats_all"] = error_stats_all
                    if 'error_stats_tables' in locals():
                        for key, df_err in error_stats_tables.items():
                            if df_err is not None and not df_err.empty:
                                sheet_map[f"error_{key}_stats"] = df_err
                    if 'distribution_diag_tables' in locals() and distribution_diag_tables:
                        for key, df_dist in distribution_diag_tables.items():
                            if df_dist is not None and not df_dist.empty:
                                sheet_map[key] = df_dist
                    append_sheets_safe(output_file, sheet_map, site_tag=site_tag, stage="写入v13 matched profile / timeseries / error / distribution sheet")

        except Exception as e:
            print(f"[{site_tag}] ❌ 处理过程中出错: {str(e)}")
            print(traceback.format_exc())

    print("\n" + "=" * 90)
    print("全部站点处理完成")
    print("=" * 90)
