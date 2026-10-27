import torch
import torch.optim as optim
import numpy as np
import json
import os

from dataset_manager import DatasetManager
from CNNClassifier import CNNClassifier
from train_manager import SSLTrainer_threshold


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

    lambda_u = 2.0
    threshold = 0.95
    n_epochs = 5

    batch_size_labelled = 32
    batch_size_unlabelled = 64

    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # Dataset (use 2-fold split)
    # -------------------------
    manager = DatasetManager.create(
        dataset_name,
        budget=budget,
        root="./data"
    )
    manager.print_summary()

    kfold_loaders = list(
        manager.get_kfold_loaders(
            n_splits=2,  # IMPORTANT: must be >= 2
            batch_size_labelled=batch_size_labelled,
            batch_size_unlabelled=batch_size_unlabelled
        )
    )

    # Take the first split
    labeled_loader, unlabeled_loader, test_loader = kfold_loaders[0]

    # -------------------------
    # Model
    # -------------------------
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

    # -------------------------
    # Train
    # -------------------------
    print("Training model...")
    for _ in range(n_epochs):
        trainer.train_epoch(
            labeled_loader,
            unlabeled_loader if unlabeled_loader is not None else None,
            optimizer
        )

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
    # Save outputs
    # -------------------------
    out_dir = "results/confusion_budget_0.05"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    with open(os.path.join(out_dir, "confusion_data.json"), "w") as f:
        json.dump(
            {
                "budget": budget,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
            },
            f,
            indent=2
        )

    print(f"Saved confusion matrix data to `{out_dir}/`")
