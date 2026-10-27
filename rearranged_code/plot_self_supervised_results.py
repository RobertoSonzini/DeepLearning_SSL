import json
import os
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# Config
# -------------------------
RESULTS_FILE = "results/self_supervised_all_budgets.json"
OUT_DIR = "results/plots_ssl_pretrain_probe"

os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Load results
# -------------------------
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

budgets = sorted(results.keys(), key=float)
budgets_f = [float(b) for b in budgets]


# =========================
# 1. Accuracy vs Budget
# =========================
accs = [results[b]["accuracy"] for b in budgets]

plt.figure(figsize=(5, 4))
plt.plot(budgets_f, accs, marker="o")
plt.xlabel("Label budget")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Label Budget (SSL pretrain + probe)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_vs_budget.png", dpi=200)
plt.close()


# =========================
# 2. Calibration vs Budget
# =========================
ece = [results[b]["ECE"] for b in budgets]
ece_ts = [results[b]["ECE_TS"] for b in budgets]

plt.figure(figsize=(5, 4))
plt.plot(budgets_f, ece, marker="o", label="ECE")
plt.plot(budgets_f, ece_ts, marker="o", label="ECE (Temp. scaled)")
plt.xlabel("Label budget")
plt.ylabel("ECE")
plt.title("Calibration vs Label Budget")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/calibration_vs_budget.png", dpi=200)
plt.close()


# =========================
# 3. Robustness curves
# =========================
corruptions = list(next(iter(results.values()))["robustness"].keys())

for corr in corruptions:
    plt.figure(figsize=(6, 4))

    for b in budgets:
        sev = results[b]["robustness"][corr]["severities"]
        acc = results[b]["robustness"][corr]["accuracy"]

        plt.plot(sev, acc, marker="o", label=f"budget={b}")

    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(f"Robustness to {corr}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/robustness_curve_{corr}.png", dpi=200)
    plt.close()


# =========================
# 4. Robustness drop plots
# =========================
for corr in corruptions:
    plt.figure(figsize=(6, 4))

    for b in budgets:
        acc = np.array(results[b]["robustness"][corr]["accuracy"])
        drop = acc[0] - acc

        plt.plot(
            results[b]["robustness"][corr]["severities"],
            drop,
            marker="o",
            label=f"budget={b}"
        )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy drop")
    plt.title(f"Robustness drop: {corr}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/robustness_drop_{corr}.png", dpi=200)
    plt.close()


print(f"All plots saved to `{OUT_DIR}/`")
