# Reproducibility workflow

The repository is organized to expose the full data/code chain used for the external validation while avoiding distribution of large WRF output files.

## 1. Run WRF

Use the files in `wrf_config/dataset_A_xilinhaote/` or `wrf_config/dataset_B_huarui_A/`. The repository does not contain `wrfout*`.

## 2. Build ML-ready tower data

Run the matching script in `scripts/01_wrf_obs_postprocessing/` after editing its user-configuration paths. These scripts read the raw observations and WRF output and export ML-ready tables containing the WRF predictors, observations, profile variables, and compatibility columns required downstream.

## 3. Build the 1 h dataset

Run the corresponding script in `scripts/02_time_resolution/` when the downstream runner is configured for the hourly dataset. Wind direction is aggregated through vector components rather than a direct arithmetic mean. For the supplied Xilinhaote time-resolution script, enable the Stage-2 dataset build when regenerating `D1H_1hour`. For the Huarui_A script, set `RUN_CASES = ["huarui_A"]` before running this repository workflow. The exact training input roots and time-resolution modes are preserved in the individual training scripts.

## 4. Generate the Gryning/MOST reference

Run the corresponding script in `scripts/03_gryning_most/`. The final learning cases use `Gryning_fixed_z0_vegetation500`. No precomputed Gryning/MOST reference CSV is distributed; it is regenerated from the ML-ready data.

Fixed z0 values:
- Dataset A / Xilinhaote: 0.030 m for NO1_1159, NO2_1107, NO3_1071, and NOT_1166.
- Dataset B / Huarui_A: 0.072 m for C039801 and 0.032 m for C039802.

## 5. Train models

Run the Code1 single-height and Code2 multi-height scripts in `scripts/04_training/`. Both CNN-LSTM and TCN are supported. The original local filesystem paths are preserved; edit only the user-configuration blocks for a new machine.

## 6. Apply cross-tower transfer

Run the all-target-time Code1 and Code2 scripts in `scripts/05_generalization/`. Target-site models are applied without target-site retraining.

## 7. Process final results

The v8 processing scripts in `scripts/06_result_processing/` implement the final time-alignment and comparison logic used for the supplied key results. The repository intentionally includes only the key result subset needed to audit Appendix B, not every intermediate plot or output file.

Run:

```bash
python scripts/06_result_processing/build_appendix_B_tables.py
```

to rebuild the TCN and CNN-LSTM Appendix-support CSVs from the included key metrics.

## Code preservation

The scientific logic of the supplied analysis scripts is retained from the executed versions. For repository readability, Chinese Python comments in the two WRF/OBS postprocessing scripts were translated to English. Data-field strings and other executable strings were not translated because downstream parsing depends on them.
