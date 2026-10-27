import json
import matplotlib.pyplot as plt

# -------------------------
# Load results
# -------------------------
with open("results/self_supervised_all_budgets.json", "r") as f:
    selfsup = json.load(f)

with open("results/supervised_budget_results.json", "r") as f:
    sup = json.load(f)

# -------------------------
# Budgets (sorted numerically)
# -------------------------
budgets = sorted(selfsup.keys(), key=float)
budgets_f = [float(b) for b in budgets]

# -------------------------
# Extract accuracies
# -------------------------
acc_selfsup = [selfsup[b]["accuracy"] for b in budgets]
acc_sup = [sup[b]["accuracy"]["mean"] for b in budgets]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(7, 5))

plt.plot(budgets_f, acc_selfsup, marker="o", label="Self-supervised")
plt.plot(budgets_f, acc_sup, marker="o", label="Supervised")

plt.xlabel("Label budget")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Label Budget")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("results/plots_comparison/self_super_vs_super.png")
