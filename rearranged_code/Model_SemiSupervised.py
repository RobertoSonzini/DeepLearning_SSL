import torch
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import DataLoader

from dataset_manager import DatasetManager
from CNNClassifier import CNNClassifier
from train_manager import SSLTrainer_threshold

from calibration_utils import (
    collect_logits_and_labels,
    expected_calibration_error,
    fit_temperature,
)

from robustness_utils import CorruptedDataset, CORRUPTIONS


# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_results(results, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    # numpy backup
    np.save(os.path.join(out_dir, "ssl_budget_results.npy"), results)

    # JSON conversion
    def to_python(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        return x

    json_results = json.loads(json.dumps(results, default=to_python))
    with open(os.path.join(out_dir, "ssl_budget_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved in `{out_dir}/`")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed()

    # Fixed settings
    dataset_name = "FashionMNIST"
    lambda_u = 2.0
    threshold = 0.95

    budgets = [0.01, 0.05, 0.1, 1.0]

    n_epochs = 5
    n_splits = 5

    batch_size_labelled = 32
    batch_size_unlabelled = 64

    device = get_device()
    print(f"Using device: {device}")

    results = {}
    robustness_per_budget = {}

    print(f"\n{'CROSS-VALIDATION SSL (FIXED LAMBDA)':=^80}")
    print(f"lambda_u = {lambda_u}")

    # Loop over budgets
    for budget in budgets:
        print(f"\n{'BUDGET':-^80}")
        print(f"Budget = {budget}")

        manager = DatasetManager.create(
            dataset_name,
            budget=budget,
            root="./data"
        )
        manager.print_summary()

        kfold_loaders = list(
            manager.get_kfold_loaders(
                n_splits=n_splits,
                batch_size_labelled=batch_size_labelled,
                batch_size_unlabelled=batch_size_unlabelled
            )
        )

        fold_metrics = {
            key: [] for key in
            ['loss', 'accuracy', 'precision', 'recall', 'f1', 'TPR', 'FPR',
             'ECE', 'ECE_TS', 'T']
        }

        robustness_per_budget[budget] = []

        # Cross-validation folds
        for fold_idx, (labeled_loader, unlabeled_loader, val_loader) in enumerate(kfold_loaders):
            print(f"  Fold {fold_idx + 1}/{n_splits}")

            in_channels = 1 if dataset_name.lower() == "fashionmnist" else 3
            model = CNNClassifier(
                in_channels=in_channels,
                num_classes=10,
                dropout=0.5
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            trainer = SSLTrainer_threshold(
                model,
                device,
                lambda_u=lambda_u,
                threshold=threshold
            )

            # -------- Training --------
            for _ in range(n_epochs):
                trainer.train_epoch(
                    labeled_loader,
                    unlabeled_loader if unlabeled_loader is not None else None,
                    optimizer
                )

            # -------- Standard validation --------
            val_metrics = trainer.util_validate(val_loader)

            # -------- Calibration Testing --------
            logits_val, labels_val = collect_logits_and_labels(model, val_loader, device)

            ece = expected_calibration_error(logits_val, labels_val)

            scaler = fit_temperature(model, val_loader, device)
            T = scaler.temperature
            with torch.no_grad():
                scaled_logits_val = scaler(logits_val.to(device)).cpu()

            ece_ts = expected_calibration_error(scaled_logits_val, labels_val)

            val_metrics["ECE"] = ece
            val_metrics["ECE_TS"] = ece_ts
            val_metrics["T"] = T

            for key in fold_metrics:
                fold_metrics[key].append(val_metrics[key])

            # -------- Robustness Testing --------
            base_val_dataset = val_loader.dataset
            severities = [0, 1, 2, 3, 4, 5]
            robust_results_fold = {}

            for corr_name in CORRUPTIONS.keys():
                accs = []
                for sev in severities:
                    corr_ds = CorruptedDataset(base_val_dataset, corr_name, sev)
                    corr_loader = DataLoader(
                        corr_ds,
                        batch_size=val_loader.batch_size,
                        shuffle=False,
                        num_workers=0
                    )
                    m = trainer.util_validate(corr_loader)
                    accs.append(m["accuracy"])
                robust_results_fold[corr_name] = accs

            robustness_per_budget[budget].append(robust_results_fold)

        # Collect metrics
        results[budget] = {
            key: {
                "mean": float(np.mean(fold_metrics[key])),
                "std": float(np.std(fold_metrics[key])),
                "values": fold_metrics[key]
            }
            for key in fold_metrics
        }

        # -------- Aggregate robustness --------
        rob_agg = {}
        for corr_name in CORRUPTIONS.keys():
            mat = np.array(
                [fold[corr_name] for fold in robustness_per_budget[budget]],
                dtype=float
            )
            rob_agg[corr_name] = {
                "severities": severities,
                "mean_accuracy": mat.mean(axis=0).tolist(),
                "std_accuracy": mat.std(axis=0).tolist(),
            }

        results[budget]["robustness"] = rob_agg

    save_results(results)
