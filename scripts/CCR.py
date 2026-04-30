"""Prototype: CCR DEA with 1 input (price) and 1 output (intelligence index)."""
import json
from pathlib import Path

import pandas as pd
from Pyfrontier.frontier_model import MultipleDEA

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "5models.json"
RESULTS_FILE = ROOT / "results" / "dea_results_CCR.csv"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    json_data = json.load(f)

rows = []
for element in json_data["data"]:
    rows.append({
        "model": element["name"],
        "provider": element["model_creator"]["name"],
        "release_date": element["release_date"],
        "price_blended": element["pricing"]["price_1m_blended_3_to_1"],
        "intelligence_index": element["evaluations"]["artificial_analysis_intelligence_index"],
    })
df = pd.DataFrame(rows).set_index("model")
print(df)

X = df[["price_blended"]].to_numpy()
Y = df[["intelligence_index"]].to_numpy()

dea = MultipleDEA(frontier="CRS", orient="in")
dea.fit(X, Y)

results_data = []
for r in dea.result:
    results_data.append({
        "model": df.index[r.id],
        "efficiency": r.score,
        "is_efficient": r.is_efficient,
        "x_weight": r.x_weight,
        "y_weight": r.y_weight,
        "bias": r.bias,
    })

results_df = pd.DataFrame(results_data).set_index("model").join(df)

print("\n=== DEA Results ===")
print(results_df[["efficiency", "is_efficient", "price_blended", "intelligence_index", "x_weight", "y_weight"]])

print("\n=== Detailed DEA Results ===")
for idx, row in results_df.iterrows():
    print(f"\n{idx}:")
    print(f"  Efficiency:   {row['efficiency']:.6f}")
    print(f"  Is Efficient: {row['is_efficient']}")
    print(f"  X Weight:     {row['x_weight']}")
    print(f"  Y Weight:     {row['y_weight']}")

results_df.to_csv(RESULTS_FILE)
print(f"\nResults saved to {RESULTS_FILE}")
print(dea.cross_efficiency)
