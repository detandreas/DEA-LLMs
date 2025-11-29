import json
import pandas as pd
import numpy as np
from Pyfrontier.frontier_model import MultipleDEA
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(json_file: str) -> dict:
    """Φορτώνει δεδομένα από JSON αρχείο."""
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(json_data: dict) -> pd.DataFrame:
    """Προετοιμάζει DataFrame από JSON δεδομένα."""
    data = json_data["data"]
    
    rows = []
    for element in data:
        rows.append({
            "model": element["name"],
            "provider": element["model_creator"]["name"],
            "release_date": element["release_date"],
            "price_blended": element["pricing"]["price_1m_blended_3_to_1"],
            "intelligence_index": element["evaluations"]["artificial_analysis_intelligence_index"]
        })
    
    return pd.DataFrame(rows).set_index("model")


def run_dea_analysis(df: pd.DataFrame) -> tuple:
    """Εκτελεί DEA ανάλυση και επιστρέφει το μοντέλο και τα αποτελέσματα."""
    X = df[["price_blended"]].to_numpy()
    Y = df[["intelligence_index"]].to_numpy()
    
    dea = MultipleDEA(frontier="CRS", orient="in")
    dea.fit(X, Y)
    
    return dea, dea.result, dea.cross_efficiency


def create_results_dataframe(result, cross_efficiency_scores, df: pd.DataFrame) -> pd.DataFrame:
    """Δημιουργεί DataFrame με τα αποτελέσματα DEA."""
    results_data = []
    for i, r in enumerate(result):
        results_data.append({
            "model": df.index[r.id],
            "efficiency": r.score,
            "cross_efficiency": cross_efficiency_scores[i],
            "is_efficient": r.is_efficient,
            "x_weight": r.x_weight,
            "y_weight": r.y_weight,
            "bias": r.bias
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.set_index("model")
    results_df = results_df.join(df)
    
    return results_df


def print_results(results_df: pd.DataFrame):
    """Εκτυπώνει τα αποτελέσματα DEA."""
    print("\n=== Αποτελέσματα DEA ===")
    print(results_df[["efficiency", "cross_efficiency", "is_efficient", 
                      "price_blended", "intelligence_index", "x_weight", "y_weight"]])
    
    print("\n=== Αναλυτικά Αποτελέσματα DEA ===")
    for idx, row in results_df.iterrows():
        print(f"\n{idx}:")
        print(f"  Efficiency: {row['efficiency']:.6f}")
        print(f"  Cross Efficiency: {row['cross_efficiency']:.6f}")
        print(f"  Is Efficient: {row['is_efficient']}")
        print(f"  X Weight (input multiplier): {row['x_weight']}")
        print(f"  Y Weight (output multiplier): {row['y_weight']}")


def save_results(results_df: pd.DataFrame, filename: str = "dea_results.csv"):
    """Αποθηκεύει τα αποτελέσματα σε CSV αρχείο."""
    results_df.to_csv(filename)
    print(f"\nΑποτελέσματα αποθηκεύτηκαν στο {filename}")


def _calculate_extended_frontier(eff: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    """Υπολογίζει την επεκταμένη frontier γραμμή."""
    first_point = (eff["price_blended"].iloc[0], eff["intelligence_index"].iloc[0])
    
    # Κλίση από (0,0) στο πρώτο efficient point
    if first_point[0] > 0:
        slope_from_origin = first_point[1] / first_point[0]
    else:
        slope_from_origin = 0
    
    # Επέκταση προς τα δεξιά
    max_x = results_df["price_blended"].max() * 1.1
    
    if len(eff) == 1:
        # Αν υπάρχει μόνο ένα efficient point
        frontier_x = [0, eff["price_blended"].iloc[0], max_x]
        frontier_y = [0, eff["intelligence_index"].iloc[0], slope_from_origin * max_x]
    else:
        # Υπολογισμός κλίσης μεταξύ efficient points
        last_point = (eff["price_blended"].iloc[-1], eff["intelligence_index"].iloc[-1])
        
        if last_point[0] != first_point[0]:
            slope_between = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        else:
            slope_between = 0
        
        # Επέκταση προς τα δεξιά με την ίδια κλίση
        y_at_max = last_point[1] + slope_between * (max_x - last_point[0])
        
        # Δημιουργία extended frontier
        frontier_x = [0] + eff["price_blended"].tolist() + [max_x]
        frontier_y = [0] + eff["intelligence_index"].tolist() + [y_at_max]
    
    return frontier_x, frontier_y


def plot_efficiency_comparison(results_df: pd.DataFrame, dea):
    """Δημιουργεί plot με bar chart και heatmap."""
    df_sorted = results_df.sort_values("efficiency", ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.bar(df_sorted.index, df_sorted["efficiency"])
    ax1.set_ylabel("Efficiency score")
    ax1.set_xlabel("Model")
    ax1.set_xticklabels(df_sorted.index, rotation=45, ha="right")
    ax1.set_title("DEA Efficiency by Model")
    ax1.grid(axis='y', alpha=0.3)
    
    cross_efficiency_matrix = dea._cross_efficiency_matrix()
    cross_df = pd.DataFrame(
        cross_efficiency_matrix,
        index=results_df.index,
        columns=results_df.index
    )
    sns.heatmap(cross_df, annot=True, fmt='.5f', cmap="viridis", 
                cbar_kws={'label': 'Cross-Efficiency'}, ax=ax2)
    ax2.set_title("Cross-Efficiency Matrix")
    ax2.set_xlabel("Evaluated by DMU")
    ax2.set_ylabel("DMU")
    
    plt.tight_layout()
    plt.show()


def plot_frontier(results_df: pd.DataFrame):
    """Δημιουργεί plot με την efficient frontier."""
    plt.figure(figsize=(10, 7))
    
    plt.scatter(results_df["price_blended"], results_df["intelligence_index"], 
                s=100, alpha=0.6, color='blue', edgecolors='black', linewidth=1)
    
    for idx, row in results_df.iterrows():
        plt.annotate(idx, 
                    (row["price_blended"], row["intelligence_index"]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    eff = results_df[results_df["is_efficient"] == True].sort_values("price_blended")
    
    if len(eff) > 0:
        frontier_x, frontier_y = _calculate_extended_frontier(eff, results_df)
        
        plt.plot(frontier_x, frontier_y, 
                 color="red", linewidth=2, linestyle='--', 
                 label='Efficient Frontier', zorder=3)
        
        plt.scatter(eff["price_blended"], eff["intelligence_index"], 
                   s=150, color='red', marker='o', edgecolors='darkred', 
                   linewidth=2, zorder=5, label='Efficient DMUs')
    
    max_y = results_df["intelligence_index"].max()
    max_y_rounded = ((int(max_y) // 5) + 1) * 5
    plt.ylim(bottom=0, top=max_y_rounded)
    plt.yticks(np.arange(0, max_y_rounded + 1, 5))
    
    plt.xlabel("Price Blended (Input)", fontsize=12)
    plt.ylabel("Intelligence Index (Output)", fontsize=12)
    plt.title("DEA Frontier", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Κύρια συνάρτηση που εκτελεί όλη την ανάλυση."""
    json_data = load_data("5models.json")
    
    df = prepare_data(json_data)
    print(df)
    
    dea, result, cross_efficiency_scores = run_dea_analysis(df)
    
    results_df = create_results_dataframe(result, cross_efficiency_scores, df)
    
    print_results(results_df)
    
    save_results(results_df, "dea_results.csv")
    
    plot_efficiency_comparison(results_df, dea)
    plot_frontier(results_df)


if __name__ == "__main__":
    main()