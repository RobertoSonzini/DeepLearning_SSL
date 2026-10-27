import torch
import torch.optim as optim
import numpy as np
import json
import os

from dataset_manager import DatasetManager
from CNNClassifier import CNNClassifier
from train_manager import SupervisedTrainer

# Utilities
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
    with open(os.path.join(out_dir, "supervised_budget_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved in `{out_dir}/`")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed()

    dataset_name = "FashionMNIST"
    budgets = [0.01, 0.05, 0.1, 1.0]

    n_epochs = 5
    n_splits = 5
    batch_size = 32

    device = get_device()
    print(f"Using device: {device}")

    results = {}

    print(f"\n{'CROSS-VALIDATION SUPERVISED BASELINE':=^80}")

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
                batch_size_labelled=batch_size,
                batch_size_unlabelled=None  # IMPORTANT
            )
        )

        fold_metrics = {
            key: [] for key in
            ['loss', 'accuracy', 'precision', 'recall', 'f1', 'TPR', 'FPR']
        }

        for fold_idx, (labeled_loader, _, val_loader) in enumerate(kfold_loaders):
            print(f"  Fold {fold_idx + 1}/{n_splits}")

            in_channels = 1 if dataset_name.lower() == "fashionmnist" else 3
            model = CNNClassifier(
                in_channels=in_channels,
                num_classes=10,
                dropout=0.5
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            trainer = SupervisedTrainer(model, device)

            for epoch in range(n_epochs):
                trainer.train_epoch(labeled_loader, optimizer)

            val_metrics = trainer.util_validate(val_loader)
            for key in fold_metrics:
                fold_metrics[key].append(val_metrics[key])

        results[budget] = {
            key: {
                "mean": float(np.mean(fold_metrics[key])),
                "std": float(np.std(fold_metrics[key])),
                "values": fold_metrics[key]
            }
            for key in fold_metrics
        }

    save_results(results)
