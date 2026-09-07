#!/usr/bin/env python3
"""Rebuild Appendix-B CSV tables from the key result files included in this repository."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = pd.read_csv(ROOT / "results/dataset_A_xilinhaote/key_metrics_all_networks.csv")
B = pd.read_csv(ROOT / "results/dataset_B_huarui_A/key_metrics_all_networks.csv")
OUT = ROOT / "results/appendix_B"
ROLES = ["Data-driven", "MOST loss", "MOST input"]

SPECS = [
    ("Dataset A - Xilinhaote", A, "NO3_1071", "NO1_1159", 10, [10, 30, 50, 70]),
    ("Dataset A - Xilinhaote", A, "NO3_1071", "NO2_1107", 10, [10, 30, 50, 70]),
    ("Dataset A - Xilinhaote", A, "NO3_1071", "NOT_1166", 10, [10, 50, 70]),
    ("Dataset B - Huarui_A", B, "C039801", "C039802", 160, [40, 80, 100, 130, 140, 160]),
    ("Dataset B - Huarui_A", B, "C039802", "C039801", 160, [40, 80, 100, 130, 140, 160]),
]

def table(network: str) -> pd.DataFrame:
    rows=[]
    for dataset,d,src,tgt,href,hmulti in SPECS:
        for code,evaluation,heights,hsel in [
            ("Code1","Single-height",str(href),str(float(href))),
            ("Code2","Multi-height",", ".join(map(str,hmulti)),"ALL_PROFILE"),
        ]:
            q=d[(d.source_site==src)&(d.target_site==tgt)&(d.network==network)&(d.code==code)&
                (d.generalization_type=="cross_tower_other_time")&(d.variable=="WS")]
            if code=="Code1":
                q=q[pd.to_numeric(q.height_m,errors="coerce")==href]
            else:
                q=q[q.height_m.astype(str)=="ALL_PROFILE"]
            v={r.role:r for r in q[q.role.isin(ROLES)].itertuples()}
            data=float(v["Data-driven"].RMSE); inp=float(v["MOST input"].RMSE)
            rows.append({
                "Dataset":dataset,"Transfer":f"{src} → {tgt}","Evaluation":evaluation,"Heights_m":heights,
                "N":int(v["Data-driven"].N),"Raw_WRF":float(v["Data-driven"].Raw_RMSE),
                "Data_driven":data,"MOST_based_loss":float(v["MOST loss"].RMSE),
                "MOST_informed_input":inp,"RMSE_reduction_pct":(data-inp)/data*100,
            })
    out=pd.DataFrame(rows)
    out["__order"]=out.Evaluation.map({"Single-height":0,"Multi-height":1})
    return out.sort_values("__order",kind="stable").drop(columns="__order").reset_index(drop=True)

if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    for network,fn in [("TCN","Table_B1_TCN.csv"),("CNN_LSTM","Table_B1_CNN_LSTM.csv")]:
        df=table(network)
        df.to_csv(OUT/fn,index=False,float_format="%.12g")
        print(network, f"reduction range: {df.RMSE_reduction_pct.min():.1f}-{df.RMSE_reduction_pct.max():.1f}%")
