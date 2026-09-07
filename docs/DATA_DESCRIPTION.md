# Data description

## Dataset A — Xilinhaote

Raw observational CSV files are provided for four towers: `NO1_1159`, `NO2_1107`, `NO3_1071`, and `NOT_1166`. The files contain tower observations at multiple heights, including wind speed and other available variables. The Appendix-B transfer tests use models sourced from `NO3_1071` and transferred to `NO1_1159`, `NO2_1107`, and `NOT_1166`. The single-height wind-speed evaluation uses 10 m.

The fixed vegetation roughness length used by the Gryning/MOST reference is `z0 = 0.030 m` for all four Xilinhaote towers.

## Dataset B — Huarui_A

Raw Windographer text files are provided for towers `C039801` and `C039802`, including the multi-height wind-speed files and the 130/160 m files used by the original preprocessing workflow. Both cross-tower transfer directions are evaluated. The single-height wind-speed evaluation uses 160 m.

The fixed vegetation roughness lengths used by the Gryning/MOST reference are:

- `C039801`: `z0 = 0.072 m`
- `C039802`: `z0 = 0.032 m`

## WRF data

WRF output (`wrfout*`) is not distributed in this repository. The WRF/WPS configuration and `tslist` files used for the two external datasets are provided under `wrf_config/`. Users must provide the required WPS meteorological/geographical inputs and run WRF separately. The supplied postprocessing scripts then extract and match the tower-level WRF variables to the observations.

## Time basis

Appendix-B sample counts are valid hourly target samples retained by the final all-target-time/common-time result processing. They are evaluation sample counts, not source-model training sample counts. Training-time resolution follows the exact configuration preserved in each training script; the Appendix-B N values should not be used to infer the training sample size.
