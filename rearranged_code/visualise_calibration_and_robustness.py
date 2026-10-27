import json
import numpy as np
import matplotlib.pyplot as plt
import os


# -------------------------
# Load results
# -------------------------
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------
# Calibration plots
# -------------------------
def plot_calibration_vs_budget(results, out_dir):
    budgets = sorted(results.keys(), key=float)

    ece = [results[b]["ECE"]["mean"] for b in budgets]
    ece_ts = [results[b]["ECE_TS"]["mean"] for b in budgets]

    plt.figure()
    plt.plot(budgets, ece, marker="o", label="ECE")
    plt.plot(budgets, ece_ts, marker="o", label="ECE (Temp. Scaled)")
    plt.xlabel("Label budget")
    plt.ylabel("ECE")
    plt.title("Calibration vs Label Budget")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "calibration_vs_budget.png"), dpi=300)
    plt.close()


def plot_calibration_gain(results, out_dir):
    budgets = sorted(results.keys(), key=float)
    gain = [
        results[b]["ECE"]["mean"] - results[b]["ECE_TS"]["mean"]
        for b in budgets
    ]

    plt.figure()
    plt.plot(budgets, gain, marker="o")
    plt.xlabel("Label budget")
    plt.ylabel("ECE − ECE_TS")
    plt.title("Calibration Gain from Temperature Scaling")
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "calibration_gain.png"), dpi=300)
    plt.close()


# -------------------------
# Robustness plots
# -------------------------
def plot_robustness_curves(results, corruption, out_dir):
    budgets = sorted(results.keys(), key=float)

    plt.figure()
    for b in budgets:
        acc = results[b]["robustness"][corruption]["mean_accuracy"]
        sev = results[b]["robustness"][corruption]["severities"]
        plt.plot(sev, acc, marker="o", label=f"budget={b}")

    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(f"Robustness Curve – {corruption}")
    plt.legend()
    plt.grid(True)

    fname = f"robustness_curve_{corruption}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()


def plot_robustness_drop(results, corruption, out_dir):
    budgets = sorted(results.keys(), key=float)
    drops = []

    for b in budgets:
        acc = results[b]["robustness"][corruption]["mean_accuracy"]
        drops.append(acc[0] - acc[-1])  # clean − max severity

    plt.figure()
    plt.plot(budgets, drops, marker="o")
    plt.xlabel("Label budget")
    plt.ylabel("Accuracy drop")
    plt.title(f"Robustness Drop vs Budget – {corruption}")
    plt.grid(True)

    fname = f"robustness_drop_{corruption}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    input_path = "results/ssl_budget_results.json"
    out_dir = "results/plots"
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(input_path)

    # -------- Calibration --------
    plot_calibration_vs_budget(results, out_dir)
    plot_calibration_gain(results, out_dir)

    # -------- Robustness --------
    corruptions = list(next(iter(results.values()))["robustness"].keys())

    for corr in corruptions:
        plot_robustness_curves(results, corr, out_dir)
        plot_robustness_drop(results, corr, out_dir)

    print(f"Plots saved in `{out_dir}/`")
