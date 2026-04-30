"""Prototype: Envelop DEA (dual formulation) with 1 input and 1 output."""
import json
from pathlib import Path

import pandas as pd
from Pyfrontier.frontier_model import EnvelopDEA

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "5models.json"
RESULTS_FILE = ROOT / "results" / "dea_results_DUAL.csv"

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

dea = EnvelopDEA(frontier="CRS", orient="in")
dea.fit(X, Y)

results_data = []
for r in dea.result:
    results_data.append({
        "model": df.index[r.id],
        "efficiency": r.score,
        "is_efficient": r.is_efficient,
        "weights": r.weights,
        "x_slack": r.x_slack,
        "y_slack": r.y_slack,
    })

results_df = pd.DataFrame(results_data).set_index("model").join(df)

print("\n=== DEA Results ===")
print(results_df[["efficiency", "is_efficient", "price_blended", "intelligence_index", "weights", "x_slack", "y_slack"]])

print("\n=== Detailed DEA Results ===")
for idx, row in results_df.iterrows():
    print(f"\n{idx}:")
    print(f"  Efficiency:   {row['efficiency']:.6f}")
    print(f"  Is Efficient: {row['is_efficient']}")
    print(f"  Weights:      {row['weights']}")
    print(f"  X Slack:      {row['x_slack']}")
    print(f"  Y Slack:      {row['y_slack']}")

results_df.to_csv(RESULTS_FILE)
print(f"\nResults saved to {RESULTS_FILE}")

optimal_solutions = results_df[results_df["efficiency"] == 1.0]
print(f"\n{len(optimal_solutions)} efficient DMU(s) found.")