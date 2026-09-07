# Long-duration external validation of the WRF–MOST deep-learning correction framework

This repository contains the data and code chain supporting the long-duration external transfer validation reported in Appendix B of the manuscript **“Improving Near-Surface Wind Speed Forecasts: A Physics-Informed Deep Learning Framework for Bias Correction of Weather Research and Forecasting Model Outputs.”**

Two independent multi-tower datasets are included:

- **Dataset A — Xilinhaote** (Northern China)
- **Dataset B — Huarui_A** (Southern China)

The repository provides the observational input data, WRF/WPS configuration files, WRF/OBS postprocessing code, hourly-data construction, Gryning/MOST reference generation, model training, cross-tower/multi-height generalization, final v8 result processing, and the key result tables used to support Appendix B.

## What is not included

Large WRF output files (`wrfout*`) are not distributed. The WRF configurations are provided so the simulations can be reproduced with the required external WPS/WRF input datasets. Precomputed Gryning/MOST diagnostic CSVs are also not distributed; the provided Gryning/MOST scripts regenerate the reference from the ML-ready data.

## Reproduction chain

```text
OBS + WRF configuration
        |
        v
run WRF (wrfout is not distributed)
        |
        v
WRF extraction + OBS matching
        |
        v
ML-ready data
        |
        v
1 h data construction
        |
        v
Gryning/MOST reference generation
        |
        v
Code1 single-height + Code2 multi-height training
        |
        v
cross-tower / multi-height transfer without target-site retraining
        |
        v
final v8 time alignment and result processing
        |
        v
Appendix B key results
```

## Repository structure

```text
data/
  dataset_A_xilinhaote/obs/
  dataset_B_huarui_A/obs/
wrf_config/
  dataset_A_xilinhaote/
  dataset_B_huarui_A/
scripts/
  01_wrf_obs_postprocessing/
  02_time_resolution/
  03_gryning_most/
  04_training/
  05_generalization/
  06_result_processing/
results/
  dataset_A_xilinhaote/
  dataset_B_huarui_A/
  appendix_B/
docs/
```

## Final physical reference

The external learning cases use the **`Gryning_fixed_z0_vegetation500`** wind-speed reference. The fixed roughness lengths are:

| Dataset | Tower | z0 (m) |
|---|---:|---:|
| Dataset A — Xilinhaote | NO1_1159 | 0.030 |
| Dataset A — Xilinhaote | NO2_1107 | 0.030 |
| Dataset A — Xilinhaote | NO3_1071 | 0.030 |
| Dataset A — Xilinhaote | NOT_1166 | 0.030 |
| Dataset B — Huarui_A | C039801 | 0.072 |
| Dataset B — Huarui_A | C039802 | 0.032 |

## Appendix-B result convention

Appendix B reports **TCN** as the representative architecture. Corresponding **CNN-LSTM** key results are also supplied in this repository. The manuscript-facing comparison is:

- Data-driven correction
- MOST-based loss
- MOST-informed input

Internal case labels are documented in `docs/METHOD_MAPPING.md`. Combined MOST-input + MOST-loss cases remain in the full training/generalization scripts but are not included in the Appendix-B summary table.

The final TCN Appendix-support table is `results/appendix_B/Table_B1_TCN.csv`. The exact v8 source values give an RMSE reduction range of **27.4–60.5%** for MOST-informed input relative to the corresponding data-driven correction across the reported single- and multi-height transfers.

> Note: the final v8 source values give 60.5% (not 60.6%) for the largest one-decimal reduction after calculation from the unrounded RMSE values.

## Quick environment setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The original executed scripts contain machine-specific paths and run switches in their user-configuration sections. Edit those configuration values before running on a different system. In particular, the Xilinhaote time-resolution script must have its dataset-build switch enabled when regenerating hourly inputs, and the Huarui_A time-resolution script should be restricted to `RUN_CASES = ["huarui_A"]`. Scientific calculation logic has not been rewritten for this repository.

## Result verification

To regenerate the two Appendix-support CSV files from the included exact key metrics:

```bash
python scripts/06_result_processing/build_appendix_B_tables.py
```

See `results/appendix_B/verification_summary.txt` and `results/appendix_B/selected_sample_time_audit.csv` for a compact audit of values, sample counts, and periods.

## Documentation

- `docs/DATA_DESCRIPTION.md` — datasets, z0 values, WRF-data scope, and sample interpretation
- `docs/METHOD_MAPPING.md` — manuscript labels and internal cases
- `docs/REPRODUCIBILITY.md` — step-by-step execution chain
