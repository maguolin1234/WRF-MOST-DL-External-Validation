# Result provenance

The key result CSVs in this repository were extracted without averaging across architectures from the final v8 result packages used in the revision:

- Dataset A: `xlh_v8_fair_shared_obs(1).zip` -> `FOR_ANALYSIS/generalization_key_metrics.csv`
- Dataset B: `generalization_compare_huarui_A_v8_global_common_time(1).zip` -> `FOR_ANALYSIS/generalization_key_metrics.csv`

Filters used for Appendix B:

- variable: WS
- generalization type: `cross_tower_other_time` (all valid target-time evaluation)
- single-height: Code1, 10 m for Dataset A and 160 m for Dataset B
- multi-height: Code2, `ALL_PROFILE` pooled RMSE
- methods: Data-driven, MOST loss, MOST input
- representative network shown in the manuscript: TCN
- CNN-LSTM is retained as the corresponding alternate architecture in the GitHub result tables

RMSE reduction is calculated as `(RMSE_data-driven - RMSE_MOST-input) / RMSE_data-driven * 100` using the unrounded source RMSE values.
