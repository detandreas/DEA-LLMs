import json
import pandas as pd
import numpy as np
from Pyfrontier.frontier_model import EnvelopDEA

with open("data/5models.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

data = json_data["data"]

rows = []
for element in data:
    rows.append({
        "model" : element["name"],
        "provider" : element["model_creator"]["name"],
        "release_date" : element["release_date"],
        "price_blended" : element["pricing"]["price_1m_blended_3_to_1"],
        "intelligence_index" : element["evaluations"]["artificial_analysis_intelligence_index"]
    })
df = pd.DataFrame(rows).set_index("model")

print(df)

X = df[["price_blended"]].to_numpy()
Y = df[["intelligence_index"]].to_numpy()

dea = EnvelopDEA(frontier="CRS", orient="in")

dea.fit(X, Y)
result = dea.result

results_data = []
for r in result:
    results_data.append({
        "model": df.index[r.id],
        "efficiency": r.score,
        "is_efficient": r.is_efficient,
        "weights": r.weights,
        "x_slack": r.x_slack,
        "y_slack": r.y_slack
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.set_index("model")

results_data = []
for r in result:
    results_data.append({
        "model": df.index[r.id],
        "efficiency": r.score,
        "is_efficient": r.is_efficient,
        "weights": r.weights,
        "x_slack": r.x_slack,
        "y_slack": r.y_slack
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.set_index("model")
results_df = results_df.join(df)

print("\n=== Αποτελέσματα DEA ===")
print(results_df[["efficiency", "is_efficient", "price_blended", "intelligence_index", "weights", "x_slack", "y_slack"]])

print("\n=== Αναλυτικά Αποτελέσματα DEA ===")
for idx, row in results_df.iterrows():
    print(f"\n{idx}:")
    print(f"  Efficiency: {row['efficiency']:.6f}")
    print(f"  Is Efficient: {row['is_efficient']}")
    print(f"  Weights: {row['weights']}")
    print(f"  X Slack (input): {row['x_slack']}")
    print(f"  Y Slack (output): {row['y_slack']}")

results_df.to_csv("results/dea_results.csv")
print("\nΑποτελέσματα αποθηκεύτηκαν στο results/dea_results_DUAL.csv")

# Αποθήκευση μόνο των βέλτιστων λύσεων (efficiency = 1)
optimal_solutions = results_df[results_df["efficiency"] == 1.0]