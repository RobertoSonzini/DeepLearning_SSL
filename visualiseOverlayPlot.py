import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load results
# -------------------------
with open("rearranged_code/results/ssl_budget_results.json", "r") as f:
    ssl_results = json.load(f)

with open("rearranged_code/results/supervised_budget_results.json", "r") as f:
    sup_results = json.load(f)

budgets = sorted([float(b) for b in ssl_results.keys()])

def extract(results, metric):
    means = [results[str(b)][metric]["mean"] for b in budgets]
    stds  = [results[str(b)][metric]["std"]  for b in budgets]
    return np.array(means), np.array(stds)

# -------------------------
# Metrics to compare
# -------------------------
metrics = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "loss": "Loss",
    "FPR": "False-Positive-Rate",
    "precision": "Precision",
    "recall": "Recall"
}

# -------------------------
# Plot
# -------------------------
for metric, ylabel in metrics.items():
    ssl_mean, ssl_std = extract(ssl_results, metric)
    sup_mean, sup_std = extract(sup_results, metric)

    plt.figure(figsize=(6, 4))

    plt.errorbar(
        budgets, ssl_mean, yerr=ssl_std,
        marker='o', capsize=4, linewidth=2,
        label="FixMatch (SSL)"
    )

    plt.errorbar(
        budgets, sup_mean, yerr=sup_std,
        marker='s', capsize=4, linewidth=2,
        label="Supervised"
    )

    plt.xscale("log")
    plt.xlabel("Labeling Budget (fraction of labeled data)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: FixMatch vs Supervised")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"rearranged_code/results/OverlayPlot_{metric}_vs_budget.png", dpi=300)

