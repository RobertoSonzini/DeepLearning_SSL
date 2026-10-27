import torch
import torch.optim as optim
import numpy as np
import json
import os

from dataset_manager import DatasetManager
from CNNClassifier import CNNClassifier
from train_manager import SupervisedTrainer


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


def save_results(results, out_dir="results", filename="supervised_full_test_results.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    # JSON-safe conversion
    def to_python(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        return x

    json_results = json.loads(json.dumps(results, default=to_python))
    with open(path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to `{path}`")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed()

    dataset_name = "FashionMNIST"
    budgets = [0.05]

    n_epochs = 5
    batch_size = 32

    device = get_device()
    print(f"Using device: {device}")

    results = {
        "dataset": dataset_name,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "budgets": {},
    }

    print(f"\n{'SUPERVISED BASELINE (TRAIN FULL) -> TEST EVAL':=^80}")

    for budget in budgets:
        print(f"\n{'BUDGET':-^80}")
        print(f"Budget = {budget}")

        manager = DatasetManager.create(
            dataset_name,
            budget=budget,
            root="./data"
        )
        manager.print_summary()

        # ---------------------------------------------------------
        # IMPORTANT:
        # This assumes DatasetManager exposes a "full train + test" loader getter.
        # If your method name differs, change ONLY this block.
        # ---------------------------------------------------------
        labeled_loader, _, test_loader = manager.get_full_loaders(
            batch_size_labelled=batch_size,
            batch_size_unlabelled=None
        )
        # ---------------------------------------------------------

        in_channels = 1 if dataset_name.lower() == "fashionmnist" else 3
        model = CNNClassifier(
            in_channels=in_channels,
            num_classes=10,
            dropout=0.5
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = SupervisedTrainer(model, device)

        # -------- Train on full labeled training set --------
        print("Training on full labeled training set...")
        for epoch in range(n_epochs):
            trainer.train_epoch(labeled_loader, optimizer)
            print(f"  Epoch {epoch + 1}/{n_epochs} done")

        # -------- Evaluate on test set --------
        print("Evaluating on test set...")
        test_metrics = trainer.util_validate(test_loader)

        results["budgets"][str(budget)] = {
            "metrics": test_metrics
        }

        print(f"Test accuracy @ budget {budget}: {test_metrics.get('accuracy', None)}")

    save_results(results)
