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


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed()

    dataset_name = "FashionMNIST"
    budget = 0.05

    n_epochs = 5
    batch_size = 32

    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # Dataset (2-fold split)
    # -------------------------
    manager = DatasetManager.create(
        dataset_name,
        budget=budget,
        root="./data"
    )
    manager.print_summary()

    # IMPORTANT: must use n_splits >= 2
    kfold_loaders = list(
        manager.get_kfold_loaders(
            n_splits=2,
            batch_size_labelled=batch_size,
            batch_size_unlabelled=None
        )
    )

    # Take the first split only
    labeled_loader, _, test_loader = kfold_loaders[0]

    # -------------------------
    # Model
    # -------------------------
    model = CNNClassifier(
        in_channels=1,
        num_classes=10,
        dropout=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = SupervisedTrainer(model, device)

    # -------------------------
    # Train
    # -------------------------
    print("Training supervised model...")
    for _ in range(n_epochs):
        trainer.train_epoch(labeled_loader, optimizer)

    # -------------------------
    # Test & collect predictions
    # -------------------------
    print("Evaluating on test set...")
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # -------------------------
    # Save outputs (same format as SSL)
    # -------------------------
    out_dir = "results/confusion_supervised_budget_0.05"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    with open(os.path.join(out_dir, "confusion_data.json"), "w") as f:
        json.dump(
            {
                "method": "supervised",
                "budget": budget,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
            },
            f,
            indent=2
        )

    print(f"Saved supervised confusion matrix data to `{out_dir}/`")
