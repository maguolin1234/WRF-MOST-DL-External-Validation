# Method mapping

The manuscript-facing method labels are intentionally used in the result tables. Internal case labels remain in the original scripts because they are part of the executed workflow.

| Manuscript label | Code1 internal case | Code2 internal case | Meaning |
|---|---|---|---|
| Data-driven | C2 | C2 | Neural-network correction without the selected MOST constraint/input |
| MOST-informed input | C4 | C4b | MOST/Gryning wind-speed reference supplied as a model input |
| MOST-based loss | C6 | C6m | MOST/Gryning wind-speed reference used through the physics-loss formulation |
| MOST input + loss | C8 | C8d | Combined strategy; retained in the full scripts but not shown in Appendix B Table B1 |

Appendix B reports representative TCN results. The corresponding CNN-LSTM key results are included in `results/appendix_B/Table_B1_CNN_LSTM.csv`.
