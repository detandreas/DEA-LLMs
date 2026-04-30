# DEA Analysis of Large Language Models

A Data Envelopment Analysis (DEA) framework that measures the **relative efficiency** of Large Language Models — evaluating how much benchmark performance each model delivers per unit of cost and latency.

---

## What is DEA?

Data Envelopment Analysis is a non-parametric linear-programming method for measuring the productive efficiency of a set of *Decision Making Units* (DMUs). It identifies a **best-practice frontier** from the observed data and scores every unit relative to that frontier (1.0 = efficient, < 1.0 = inefficient).

This project applies the **CCR model** (Charnes, Cooper & Rhodes, 1978) with *Constant Returns to Scale* and *input orientation*, asking: *"By how much could a model reduce its cost and latency while producing the same benchmark outputs?"*

---

## Model Setup

| | Variables |
|---|---|
| **Inputs** (minimize) | Price per 1M tokens (blended 3:1 ratio) · Median time to first answer token |
| **Outputs** (maximize) | Intelligence Index · Coding Index · Math Index · MMLU Pro |

**Cross-efficiency** scores are also computed: each model is evaluated not only by its own optimal weights but by the weights of every other model, yielding a more discriminating ranking that removes the self-appraisal advantage of standard DEA.

---

## Data

Model data is sourced from the [Artificial Analysis](https://artificialanalysis.ai/) API (`/v2/data/llms/models`). The dataset covers frontier models across all major providers, including pricing, latency, and a standardised suite of capability benchmarks.

---

## Project Structure

```
DEA-LLMs/
├── CCR_Cross_Efficiency.py   # Main pipeline: download → filter → DEA → plots
├── scripts/
│   ├── CCR.py                # Prototype — 1-input / 1-output CCR model
│   ├── DUAL.py               # Prototype — Envelop DEA (dual formulation)
│   ├── findmodels.py         # Standalone filter utility (thresholds on index & price)
│   └── download_json.py      # Standalone API downloader (gitignored; needs API key)
├── data/
│   ├── models_latest.json    # Cached raw API download (auto-refreshed by main script)
│   ├── models_frontier.json  # Filtered "similar capability" dataset (auto-generated)
│   ├── models.json           # Legacy curated dataset (13 models)
│   ├── 5models.json          # Small sample used by prototype scripts
│   └── ...                   # Other subsets produced by findmodels.py
├── images/                   # Saved chart exports
└── results/                  # CSV outputs (gitignored)
```

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/detandreas/DEA-LLMs.git
cd DEA-LLMs

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn Pyfrontier requests
```

---

## Usage

### Run the full pipeline

```bash
export ARTIFICIAL_ANALYSIS_API_KEY="your_key_here"   # optional; uses cache if unset
python CCR_Cross_Efficiency.py
```

The script runs four stages end-to-end:

1. **Download** — fetches the latest model dataset from the Artificial Analysis API
   and caches it to `data/models_latest.json`. Falls back to the cache if no API
   key is set or the request fails.
2. **Filter** — keeps only **frontier-tier comparable models** (intelligence
   index ≥ 40, with positive price and latency, and a coding score). The result
   is saved to `data/models_frontier.json` — the new "similar capability" dataset.
3. **Evaluate** — runs the CCR DEA with cross-efficiency on the filtered set
   and writes scores to `results/dea_results.csv`.
4. **Plot** — renders three figures with adaptive sizing for any number of
   models (see *Adaptive Plotting* below).

To tune the filter, edit `INTELLIGENCE_THRESHOLD` at the top of the script
(default = 40.0). At threshold 40 you get ~58 models; at 45 you get ~34;
at 50 you get the top ~18.

### Adaptive Plotting

All three charts scale automatically with the number of models so labels never
overlap, even at n = 70+:

| Chart | Adaptive behaviour |
|---|---|
| **Bar chart** | Width scales with *n* (capped at 28″); rotation 45→90° and font size 10→6 as *n* grows. |
| **Heatmap** | Square size scales with *n* (capped at 24″); annotations are dropped past *n* > 30 (unreadable at that density). |
| **Frontier (2-D)** | Log-scale x-axis (prices span orders of magnitude); only the upper-left envelope DMUs are labelled when *n* > 35; long names truncated. |

### Prototype scripts

The `scripts/` folder contains earlier, simpler experiments:

```bash
python scripts/CCR.py          # 1-input / 1-output CCR (5 models)
python scripts/DUAL.py         # Envelop DEA dual formulation (5 models)
python scripts/findmodels.py   # Filter models by benchmark threshold
```

---

## Outputs

| Output | Description |
|---|---|
| **Efficiency score** | Standard CCR score (1.0 = on the frontier) |
| **Cross-efficiency score** | Average score when evaluated by all other models' weights — more robust ranking |
| **Bar chart** | Efficiency scores sorted descending |
| **Heatmap** | Full cross-efficiency matrix (DMU × DMU) |
| **Frontier plot** | 2-D projection: Price vs. Intelligence Index with frontier line |

---

## Key Idea

The analysis surfaces models that achieve high benchmark performance at relatively low cost and latency — the efficient frontier — and quantifies how far off-frontier each other model sits.

> **Note:** The frontier is computed in the full 6-dimensional space (2 inputs, 4 outputs). The 2-D frontier plot is a projection onto the Price × Intelligence Index plane and is for visual reference only.

---

## References

- Charnes, A., Cooper, W. W., & Rhodes, E. (1978). *Measuring the efficiency of decision making units.* European Journal of Operational Research, 2(6), 429–444.
- Doyle, J., & Green, R. (1994). *Efficiency and cross-efficiency in DEA.* Journal of the Operational Research Society, 45(5), 567–578.
- [Artificial Analysis](https://artificialanalysis.ai/) — LLM benchmark & pricing data
- [Pyfrontier](https://github.com/NibuTake/PyFrontier) — DEA library for Python
