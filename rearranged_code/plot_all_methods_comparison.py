import json
import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
FILES = {
    "Self-supervised": "results/self_supervised_all_budgets.json",
    "Supervised": "results/supervised_budget_results.json",
    "Semi-supervised": "results/ssl_budget_results.json",
}

OUT_DIR = "results/plots_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Load results
# -------------------------
all_results = {}
for name, path in FILES.items():
    with open(path, "r") as f:
        all_results[name] = json.load(f)

# Collect all budgets appearing anywhere
all_budgets = sorted(
    {b for res in all_results.values() for b in res.keys()},
    key=float
)


# -------------------------
# Helpers
# -------------------------
def get_metric(res, budget, metric):
    """Safely extract scalar metric or mean if dict."""
    if budget not in res:
        return None
    if metric not in res[budget]:
        return None

    val = res[budget][metric]
    if isinstance(val, dict):
        return val.get("mean", None)
    return val


def get_robustness_curve(res, budget, corr):
    """Return (severities, accuracies) or None if missing."""
    if budget not in res:
        return None
    if "robustness" not in res[budget]:
        return None
    if corr not in res[budget]["robustness"]:
        return None

    r = res[budget]["robustness"][corr]
    if "mean_accuracy" in r:
        return r["severities"], r["mean_accuracy"]
    return r["severities"], r["accuracy"]


# =========================
# 1. Accuracy vs Budget
# =========================
plt.figure(figsize=(6, 4))

for method, res in all_results.items():
    xs, ys = [], []
    for b in all_budgets:
        acc = get_metric(res, b, "accuracy")
        if acc is not None:
            xs.append(float(b))
            ys.append(acc)

    if ys:
        plt.plot(xs, ys, marker="o", label=method)

plt.xlabel("Label budget")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Label Budget")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_vs_budget.png", dpi=200)
plt.close()


# =========================
# 2. Calibration vs Budget
# =========================
for metric in ["ECE", "ECE_TS"]:
    plt.figure(figsize=(6, 4))

    for method, res in all_results.items():
        xs, ys = [], []
        for b in all_budgets:
            v = get_metric(res, b, metric)
            if v is not None:
                xs.append(float(b))
                ys.append(v)

        if ys:
            plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("Label budget")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Label Budget")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{metric.lower()}_vs_budget.png", dpi=200)
    plt.close()


# =========================
# 3. Robustness curves (highest budget with robustness)
# =========================
# Find highest budget that has robustness data
highest_budget = None
for b in sorted(all_budgets, key=float, reverse=True):
    if any(
        "robustness" in res.get(b, {})
        for res in all_results.values()
    ):
        highest_budget = b
        break

if highest_budget is not None:
    # Collect corruption names safely
    corruptions = set()
    for res in all_results.values():
        if highest_budget in res and "robustness" in res[highest_budget]:
            corruptions.update(res[highest_budget]["robustness"].keys())

    # ---- Plot curves ----
    for corr in sorted(corruptions):
        plt.figure(figsize=(6, 4))

        for method, res in all_results.items():
            curve = get_robustness_curve(res, highest_budget, corr)
            if curve is None:
                continue

            severities, accs = curve
            plt.plot(severities, accs, marker="o", label=method)

        plt.xlabel("Severity")
        plt.ylabel("Accuracy")
        plt.title(f"Robustness – {corr} (budget={highest_budget})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{corr}_curve.png", dpi=200)
        plt.close()

    # ---- Robustness drop ----
    for corr in sorted(corruptions):
        plt.figure(figsize=(6, 4))

        for method, res in all_results.items():
            curve = get_robustness_curve(res, highest_budget, corr)
            if curve is None:
                continue

            severities, accs = curve
            accs = np.array(accs)
            drops = accs[0] - accs
            plt.plot(severities, drops, marker="o", label=method)

        plt.xlabel("Severity")
        plt.ylabel("Accuracy drop")
        plt.title(f"Robustness drop – {corr} (budget={highest_budget})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{corr}_drop.png", dpi=200)
        plt.close()


print(f"✅ All comparison plots saved to `{OUT_DIR}/`")
