import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load results
# -------------------------
with open("rearranged_code/results/supervised_budget_results.json", "r") as f:
    results = json.load(f)

# Convert budget keys to float and sort
budgets = sorted([float(b) for b in results.keys()])

def extract_metric(metric):
    means = [results[str(b)][metric]["mean"] for b in budgets]
    stds  = [results[str(b)][metric]["std"]  for b in budgets]
    return np.array(means), np.array(stds)

# -------------------------
# Metrics to plot
# -------------------------
metrics = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "loss": "Loss",
    "FPR": "False Positive Rate"
}

# -------------------------
# Plot
# -------------------------
for metric, ylabel in metrics.items():
    means, stds = extract_metric(metric)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        budgets,
        means,
        yerr=stds,
        marker='o',
        capsize=4,
        linewidth=2
    )

    plt.xscale("log")
    plt.xlabel("Labeling Budget (fraction of labeled data)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Labeling Budget")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"rearranged_code/results/supervised_{metric}_vs_budget.png", dpi=300)
    plt.close()
