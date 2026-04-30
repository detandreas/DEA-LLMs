"""
CCR DEA with Cross-Efficiency for LLM benchmarking.

Pipeline (run as script):
    1. Download fresh model data from the Artificial Analysis API.
    2. Filter the dataset to "similar capability" models (frontier tier).
    3. Run CCR DEA + cross-efficiency on the filtered dataset.
    4. Print, save, and plot results.

Inputs  : price_blended (cost per 1M tokens), median_time_to_first_answer_token (latency)
Outputs : intelligence_index, coding_index, math_index, mmlu_pro
Model   : CCR (Constant Returns to Scale), input-oriented
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from Pyfrontier.frontier_model import MultipleDEA

ROOT = Path(__file__).parent
RAW_DATA_FILE = ROOT / "data" / "models_latest.json"
FILTERED_DATA_FILE = ROOT / "data" / "models_frontier.json"
RESULTS_FILE = ROOT / "results" / "dea_results.csv"

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
INPUT_COLS = ["price_blended", "median_time_to_first_answer"]
OUTPUT_COLS = ["intelligence_index", "coding_index", "math_index", "mmlu_pro"]

# "Similar capability" = frontier-tier models above this intelligence threshold
INTELLIGENCE_THRESHOLD = 40.0
# Maximum DMUs to evaluate (top-N by intelligence index after quality filter)
MAX_MODELS = 15

# Pipeline step 1 — download
def download_latest_data(output_path: Path = RAW_DATA_FILE) -> dict:
    """
    Fetch the latest model dataset from the Artificial Analysis API.

    Requires the ARTIFICIAL_ANALYSIS_API_KEY environment variable.
    If the key is missing or the request fails, falls back to the cached
    file at output_path (or to data/models2.json as legacy fallback).
    """
    api_key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")

    if api_key:
        try:
            print(f"[1/4] Downloading latest data from {API_URL} ...")
            response = requests.get(API_URL, headers={"x-api-key": api_key}, timeout=30)
            response.raise_for_status()
            data = response.json()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"      Saved {len(data['data'])} models → {output_path}")
            return data
        except requests.RequestException as e:
            print(f"      Download failed: {e}. Falling back to cached data.")
    else:
        print("[1/4] ARTIFICIAL_ANALYSIS_API_KEY not set; using cached data.")

    for candidate in (output_path, ROOT / "data" / "models2.json"):
        if candidate.exists():
            print(f"      Using cached file: {candidate}")
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        "No cached data available. Set ARTIFICIAL_ANALYSIS_API_KEY to download "
        "fresh data, or place a dataset at data/models_latest.json."
    )

# Pipeline step 2 — filter to "similar capability" models
def filter_similar_capability_models(
    raw_data: dict,
    intelligence_threshold: float = INTELLIGENCE_THRESHOLD,
    max_models: int = MAX_MODELS,
    output_path: Path = FILTERED_DATA_FILE,
) -> dict:
    """
    Build a filtered dataset of frontier-tier models suitable for fair DEA comparison.

    A model is kept iff:
      * intelligence_index >= intelligence_threshold
      * both DEA inputs (price, latency) are present and strictly positive
        (DEA requires positive inputs; zeros indicate missing data and would
        break the LP and the cross-efficiency normalisation)
      * the core DEA outputs (intelligence_index, coding_index) are present
    From the qualifying models the top max_models by intelligence_index are
    selected, so the DEA set stays small enough to be discriminating.
    Missing math_index / mmlu_pro values are imputed as 0 in the analysis step.
    """
    candidates = []
    for m in raw_data.get("data", []):
        evals = m.get("evaluations", {}) or {}
        pricing = m.get("pricing", {}) or {}

        idx = evals.get("artificial_analysis_intelligence_index")
        price = pricing.get("price_1m_blended_3_to_1")
        ttfa = m.get("median_time_to_first_answer_token")
        coding = evals.get("artificial_analysis_coding_index")

        if idx is None or idx < intelligence_threshold:
            continue
        if not price or price <= 0:
            continue
        if not ttfa or ttfa <= 0:
            continue
        if coding is None:
            continue

        candidates.append(m)

    # Keep only the top-N by intelligence index for a focused, comparable set
    candidates.sort(
        key=lambda m: m["evaluations"]["artificial_analysis_intelligence_index"],
        reverse=True,
    )
    kept = candidates[:max_models]

    filtered = {
        "status": raw_data.get("status", 200),
        "prompt_options": raw_data.get("prompt_options", {}),
        "filter": {
            "intelligence_threshold": intelligence_threshold,
            "max_models": max_models,
            "candidates": len(candidates),
            "kept": len(kept),
            "total": len(raw_data.get("data", [])),
        },
        "data": kept,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)

    print(
        f"[2/4] Selected top {len(kept)} of {len(candidates)} qualifying models "
        f"(intelligence_index >= {intelligence_threshold}, max {max_models}) → {output_path}"
    )
    return filtered


# Pipeline step 3 — evaluate
def prepare_data(json_data: dict) -> pd.DataFrame:
    rows = []
    for element in json_data["data"]:
        evals = element.get("evaluations", {}) or {}
        rows.append({
            "model": element["name"],
            "provider": element.get("model_creator", {}).get("name"),
            "release_date": element.get("release_date"),
            "price_blended": element["pricing"]["price_1m_blended_3_to_1"],
            "median_time_to_first_answer": element["median_time_to_first_answer_token"],
            "intelligence_index": evals.get("artificial_analysis_intelligence_index"),
            "coding_index": evals.get("artificial_analysis_coding_index"),
            "math_index": evals.get("artificial_analysis_math_index"),
            "mmlu_pro": evals.get("mmlu_pro"),
        })
    return pd.DataFrame(rows).set_index("model")


def run_dea_analysis(df: pd.DataFrame) -> tuple:
    X = df[INPUT_COLS].to_numpy()
    # Missing benchmark scores are treated as 0 (conservative assumption).
    # Cast to float first so fillna doesn't trigger pandas downcasting warning.
    Y_df = df[OUTPUT_COLS].astype(float).fillna(0)

    # Drop output columns that are entirely zero — they add no discriminating
    # power and cause LP degeneracy (solver returns None for those weights).
    active_cols = [c for c in OUTPUT_COLS if Y_df[c].gt(0).any()]
    dropped = [c for c in OUTPUT_COLS if c not in active_cols]
    if dropped:
        print(f"      Warning: dropping all-zero output columns: {dropped}")
    if not active_cols:
        raise ValueError("All output columns are zero — DEA cannot be run.")

    Y = Y_df[active_cols].to_numpy()

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


# Reporting
def print_results(results_df: pd.DataFrame) -> None:
    print("\n=== DEA Results ===")
    print(results_df[[
        "efficiency", "cross_efficiency", "is_efficient",
        "price_blended", "median_time_to_first_answer",
        *OUTPUT_COLS,
    ]].sort_values("cross_efficiency", ascending=False))

    print("\n=== Detailed DEA Results ===")
    for idx, row in results_df.sort_values("cross_efficiency", ascending=False).iterrows():
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
            print(f"    {label}: {val:.4f}" if pd.notna(val) else f"    {label}: N/A")


def save_results(results_df: pd.DataFrame, path: Path = RESULTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path)
    print(f"\nResults saved to {path}")


# Plotting (adaptive sizing for many models)
def plot_efficiency_comparison(results_df: pd.DataFrame, dea) -> None:
    n = len(results_df)
    df_sorted = results_df.sort_values("efficiency", ascending=False)
    # --- Bar chart: width scales with n, capped to a screen-friendly maximum ---
    bar_width = min(28, max(12, 0.45 * n))
    if n <= 15:
        label_fontsize, rotation = 10, 45
    elif n <= 30:
        label_fontsize, rotation = 8, 60
    elif n <= 60:
        label_fontsize, rotation = 7, 75
    else:
        label_fontsize, rotation = 6, 90

    plt.figure(figsize=(bar_width, 7))
    plt.bar(df_sorted.index, df_sorted["efficiency"])
    plt.ylabel("Efficiency Score")
    plt.xlabel("Model")
    plt.xticks(rotation=rotation, ha="right", fontsize=label_fontsize)
    plt.title(f"DEA Efficiency by Model (n={n})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Heatmap: square, capped size; annotations dropped past ~30 cells per side ---
    cross_df = pd.DataFrame(
        dea._cross_efficiency_matrix(),
        index=results_df.index,
        columns=results_df.index,
    )
    heat_size = min(24, max(10, 0.45 * n))

    if n <= 12:
        annot, fmt, annot_size = True, ".4f", 9
    elif n <= 20:
        annot, fmt, annot_size = True, ".2f", 7
    elif n <= 30:
        annot, fmt, annot_size = True, ".2f", 6
    else:
        annot, fmt, annot_size = False, ".2f", 6  # Too crowded to annotate

    plt.figure(figsize=(heat_size, heat_size * 0.9))
    sns.heatmap(
        cross_df, annot=annot, fmt=fmt, cmap="viridis",
        annot_kws={"size": annot_size},
        cbar_kws={"label": "Cross-Efficiency"},
        linewidths=0.0,
        square=False,
    )
    plt.title(f"Cross-Efficiency Matrix (n={n})")
    plt.xlabel("Evaluated by DMU")
    plt.ylabel("DMU")
    plt.xticks(rotation=75, ha="right", fontsize=label_fontsize)
    plt.yticks(rotation=0, fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()


def _upper_left_envelope(eff: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of efficient DMUs that form the *2-D* upper-left envelope
    in (price, intelligence). A point (p, i) is on the envelope iff no other
    point dominates it (p' <= p AND i' >= i, with at least one strict).
    """
    sorted_eff = eff.sort_values("price_blended")
    envelope = []
    best_intel = -np.inf
    for _, row in sorted_eff.iterrows():
        if row["intelligence_index"] > best_intel:
            envelope.append(row)
            best_intel = row["intelligence_index"]
    return pd.DataFrame(envelope)


def _truncate(label: str, width: int = 28) -> str:
    return label if len(label) <= width else label[: width - 1] + "…"


def _stagger_offset(i: int) -> tuple:
    """Cycle through label offset directions to reduce overlap."""
    offsets = [(7, 7), (7, -12), (-90, 7), (-90, -12), (7, 18), (-90, 18)]
    return offsets[i % len(offsets)]


def plot_frontier(results_df: pd.DataFrame) -> None:
    n = len(results_df)
    eff = results_df[results_df["is_efficient"]].sort_values("price_blended")
    plt.figure(figsize=(14, 8))

    # Use log scale on price — prices span ~2 orders of magnitude ($0.15–$30).
    plt.xscale("log")

    plt.scatter(
        results_df["price_blended"], results_df["intelligence_index"],
        s=70, alpha=0.5, color="steelblue", edgecolors="black", linewidth=0.6,
        label="Inefficient DMUs", zorder=2,
    )

    # Frontier: connect only the upper-left envelope of efficient DMUs.
    if len(eff) > 0:
        env = _upper_left_envelope(eff)
        plt.plot(env["price_blended"], env["intelligence_index"],
                 color="red", linewidth=2, linestyle="--",
                 label="Efficient Frontier (upper envelope)", zorder=3)
        plt.scatter(eff["price_blended"], eff["intelligence_index"],
                    s=140, color="red", marker="o", edgecolors="darkred",
                    linewidth=1.5, zorder=5, label="Efficient DMUs")

    # Labelling strategy:
    #   - n <= 15:                   label every model
    #   - 15 < n <= 35:              label every efficient DMU
    #   - n > 35:                    label only envelope DMUs (the visible frontier)
    if n <= 15:
        to_label = list(results_df.iterrows())
        label_fontsize = 9
    elif n <= 35:
        to_label = list(eff.iterrows())
        label_fontsize = 8
    else:
        to_label = list(_upper_left_envelope(eff).iterrows()) if len(eff) else []
        label_fontsize = 8

    for i, (idx, row) in enumerate(to_label):
        dx, dy = _stagger_offset(i)
        plt.annotate(
            _truncate(str(idx)),
            (row["price_blended"], row["intelligence_index"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=label_fontsize, ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5),
            zorder=6,
        )

    max_y = results_df["intelligence_index"].max()
    max_y_rounded = ((int(max_y) // 5) + 1) * 5
    plt.ylim(bottom=0, top=max_y_rounded)
    plt.yticks(np.arange(0, max_y_rounded + 1, 5))

    plt.xlabel("Price Blended (USD per 1M tokens, Input 1) — log scale", fontsize=12)
    plt.ylabel("Intelligence Index (Output 1)", fontsize=12)

    if n <= 15:
        suffix = ""
    elif n <= 35:
        suffix = " — efficient DMUs labelled"
    else:
        suffix = " — frontier-envelope DMUs labelled"
    plt.title(
        f"DEA Frontier — 2D Projection (n={n}){suffix}\n(Full model: 2 inputs, 4 outputs)",
        fontsize=13, fontweight="bold",
    )
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Entry point
def main() -> None:
    raw_data = download_latest_data()
    filtered_data = filter_similar_capability_models(raw_data)

    df = prepare_data(filtered_data)
    print(f"\n[3/4] Running DEA on {len(df)} models")
    print(df)

    dea, result, cross_efficiency_scores = run_dea_analysis(df)
    results_df = create_results_dataframe(result, cross_efficiency_scores, df)

    print_results(results_df)
    save_results(results_df)

    print("\n[4/4] Generating plots ...")
    plot_efficiency_comparison(results_df, dea)
    plot_frontier(results_df)


if __name__ == "__main__":
    main()
