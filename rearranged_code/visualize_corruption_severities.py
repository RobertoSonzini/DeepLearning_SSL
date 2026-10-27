import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset_manager import DatasetManager
from robustness_utils import CorruptedDataset, CORRUPTIONS


# -------------------------
# HARD SAFETY FIX (IMPORTANT)
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # -------------------------
    # Config
    # -------------------------
    dataset_name = "FashionMNIST"
    budget = 0.05

    CORRUPTION_NAME = "rotate"   # choose from CORRUPTIONS.keys()
    IMAGE_INDEX = 0
    SEVERITIES = [0, 1, 2, 3, 4, 5]

    OUT_DIR = "results/corruption_examples"
    os.makedirs(OUT_DIR, exist_ok=True)

    assert CORRUPTION_NAME in CORRUPTIONS, (
        f"Unknown corruption {CORRUPTION_NAME}. "
        f"Choose from {list(CORRUPTIONS.keys())}"
    )

    # -------------------------
    # Load dataset (NO DataLoader usage)
    # -------------------------
    manager = DatasetManager.create(
        dataset_name,
        budget=budget,
        root="./data"
    )

    # We only need the dataset object, not loaders
    kfold_loaders = list(
        manager.get_kfold_loaders(
            n_splits=2,
            batch_size_labelled=32,
            batch_size_unlabelled=None
        )
    )

    base_dataset = kfold_loaders[0][2].dataset

    # -------------------------
    # Create figure
    # -------------------------
    fig, axes = plt.subplots(
        1, len(SEVERITIES),
        figsize=(3 * len(SEVERITIES), 3)
    )

    fig.suptitle(
        f"{CORRUPTION_NAME.capitalize()} corruption across severity levels",
        fontsize=14
    )

    # -------------------------
    # Apply corruption + plot
    # -------------------------
    for ax, severity in zip(axes, SEVERITIES):
        corr_dataset = CorruptedDataset(
            base_dataset,
            CORRUPTION_NAME,
            severity
        )

        img, _ = corr_dataset[IMAGE_INDEX]
        img_np = img.squeeze().numpy()

        ax.imshow(img_np, cmap="gray")
        ax.set_title(f"Severity {severity}")
        ax.axis("off")

    # -------------------------
    # Save ONE image
    # -------------------------
    out_path = os.path.join(
        OUT_DIR,
        f"{CORRUPTION_NAME}_all_severities.png"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved combined corruption image to:\n{out_path}")
