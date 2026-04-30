"""
CCR DEA with Cross-Efficiency for LLM benchmarking.

Inputs  : price_blended (cost per 1M tokens), median_time_to_first_answer_token (latency)
Outputs : intelligence_index, coding_index, math_index, mmlu_pro
Model   : CCR (Constant Returns to Scale), input-oriented
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Pyfrontier.frontier_model import MultipleDEA

ROOT = Path(__file__).parent
OUTPUT_COLS = ["intelligence_index", "coding_index", "math_index", "mmlu_pro"]


def load_data(json_file: str) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(json_data: dict) -> pd.DataFrame:
    rows = []
    for element in json_data["data"]:
        rows.append({
            "model": element["name"],
            "provider": element["model_creator"]["name"],
            "release_date": element["release_date"],
            "price_blended": element["pricing"]["price_1m_blended_3_to_1"],
            "median_time_to_first_answer": element["median_time_to_first_answer_token"],
            "intelligence_index": element["evaluations"]["artificial_analysis_intelligence_index"],
            "coding_index": element["evaluations"]["artificial_analysis_coding_index"],
            "math_index": element["evaluations"]["artificial_analysis_math_index"],
            "mmlu_pro": element["evaluations"]["mmlu_pro"],
        })
    return pd.DataFrame(rows).set_index("model")


def run_dea_analysis(df: pd.DataFrame) -> tuple:
    X = df[["price_blended", "median_time_to_first_answer"]].to_numpy()
    # Missing output scores are treated as 0 (conservative assumption)
    Y = df[OUTPUT_COLS].fillna(0).to_numpy()

    dea = MultipleDEA(frontier="CRS", orient="in")
    dea.fit(X, Y)
    return dea, dea.result, dea.cross_efficiency


def create_results_dataframe(result, cross_efficiency_scores, df: pd.DataFrame) -> pd.DataFrame:
    results_data = []
    for i, r in enumerate(result):
        results_data.append({
            "model": df.index[r.id],
            "efficiency": r.score,
            "cross_efficiency": cross_efficiency_scores[i],
            "is_efficient": r.is_efficient,
            "x_weight": r.x_weight,
            "y_weight": r.y_weight,
            "bias": r.bias,
        })
    return pd.DataFrame(results_data).set_index("model").join(df)


def print_results(results_df: pd.DataFrame) -> None:
    print("\n=== DEA Results ===")
    print(results_df[[
        "efficiency", "cross_efficiency", "is_efficient",
        "price_blended", "median_time_to_first_answer",
        *OUTPUT_COLS, "x_weight", "y_weight",
    ]])

    print("\n=== Detailed DEA Results ===")
    for idx, row in results_df.iterrows():
        print(f"\n{idx}:")
        print(f"  Efficiency:        {row['efficiency']:.6f}")
        print(f"  Cross Efficiency:  {row['cross_efficiency']:.6f}")
        print(f"  Is Efficient:      {row['is_efficient']}")
        print(f"  Inputs:")
        print(f"    Price Blended:                 {row['price_blended']:.4f}")
        print(f"    Median Time to First Answer:   {row['median_time_to_first_answer']:.4f}")
        print(f"  Outputs:")
        for col in OUTPUT_COLS:
            val = row[col]
            label = col.replace("_", " ").title()
            fmt = f"{val:.4f}" if pd.notna(val) else "N/A"
            print(f"    {label}: {fmt}")
        print(f"  X Weights: {row['x_weight']}")
        print(f"  Y Weights: {row['y_weight']}")


def save_results(results_df: pd.DataFrame, filename: str = "results/dea_results.csv") -> None:
    path = ROOT / filename
    results_df.to_csv(path)
    print(f"\nResults saved to {path}")


def _calculate_extended_frontier(eff: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    """Builds a piecewise-linear CRS frontier line for 2-D projection plots."""
    max_x = results_df["price_blended"].max() * 1.1

    if len(eff) == 1:
        p = (eff["price_blended"].iloc[0], eff["intelligence_index"].iloc[0])
        slope = p[1] / p[0] if p[0] > 0 else 0
        frontier_x = [0, p[0], max_x]
        frontier_y = [0, p[1], slope * max_x]
    else:
        last = (eff["price_blended"].iloc[-1], eff["intelligence_index"].iloc[-1])
        second_last = (eff["price_blended"].iloc[-2], eff["intelligence_index"].iloc[-2])
        dx = last[0] - second_last[0]
        tail_slope = (last[1] - second_last[1]) / dx if dx != 0 else 0

        frontier_x = [0] + eff["price_blended"].tolist() + [max_x]
        frontier_y = [0] + eff["intelligence_index"].tolist() + [last[1] + tail_slope * (max_x - last[0])]

    return frontier_x, frontier_y


def plot_efficiency_comparison(results_df: pd.DataFrame, dea) -> None:
    df_sorted = results_df.sort_values("efficiency", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted.index, df_sorted["efficiency"])
    plt.ylabel("Efficiency Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.title("DEA Efficiency by Model")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    cross_df = pd.DataFrame(
        dea._cross_efficiency_matrix(),
        index=results_df.index,
        columns=results_df.index,
    )
    plt.figure(figsize=(16, 12))
    sns.heatmap(cross_df, annot=True, fmt=".5f", cmap="viridis",
                cbar_kws={"label": "Cross-Efficiency"})
    plt.title("Cross-Efficiency Matrix")
    plt.xlabel("Evaluated by DMU")
    plt.ylabel("DMU")
    plt.tight_layout()
    plt.show()


def plot_frontier(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))

    plt.scatter(
        results_df["price_blended"], results_df["intelligence_index"],
        s=100, alpha=0.6, color="blue", edgecolors="black", linewidth=1,
    )
    for idx, row in results_df.iterrows():
        plt.annotate(idx, (row["price_blended"], row["intelligence_index"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9, ha="left")

    eff = results_df[results_df["is_efficient"]].sort_values("price_blended")
    if len(eff) > 0:
        frontier_x, frontier_y = _calculate_extended_frontier(eff, results_df)
        plt.plot(frontier_x, frontier_y, color="red", linewidth=2, linestyle="--",
                 label="Efficient Frontier", zorder=3)
        plt.scatter(eff["price_blended"], eff["intelligence_index"],
                    s=150, color="red", marker="o", edgecolors="darkred",
                    linewidth=2, zorder=5, label="Efficient DMUs")

    max_y = results_df["intelligence_index"].max()
    max_y_rounded = ((int(max_y) // 5) + 1) * 5
    plt.ylim(bottom=0, top=max_y_rounded)
    plt.yticks(np.arange(0, max_y_rounded + 1, 5))

    plt.xlabel("Price Blended (Input 1)", fontsize=12)
    plt.ylabel("Intelligence Index (Output 1)", fontsize=12)
    plt.title("DEA Frontier — 2D Projection\n(Full model: 2 inputs, 4 outputs)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    json_data = load_data(ROOT / "data" / "models.json")
    df = prepare_data(json_data)
    print(df)

    dea, result, cross_efficiency_scores = run_dea_analysis(df)
    results_df = create_results_dataframe(result, cross_efficiency_scores, df)

    print_results(results_df)
    save_results(results_df)

    plot_efficiency_comparison(results_df, dea)
    plot_frontier(results_df)


if __name__ == "__main__":
    main()
