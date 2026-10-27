import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset_manager import DatasetManager


# -------------------------
# CRITICAL SAFETY FIXES
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
    results_dir = "results/confusion_supervised_budget_0.05"

    TRUE_CLASS = 6   # Shirt
    PRED_CLASS = 0   # Coat
    MAX_IMAGES = 16

    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # -------------------------
    # Load predictions
    # -------------------------
    y_true = np.load(f"{results_dir}/y_true.npy")
    y_pred = np.load(f"{results_dir}/y_pred.npy")

    # -------------------------
    # Rebuild test dataset (NO iteration)
    # -------------------------
    manager = DatasetManager.create(
        dataset_name,
        budget=budget,
        root="./data"
    )

    kfold_loaders = list(
        manager.get_kfold_loaders(
            n_splits=2,
            batch_size_labelled=32,
            batch_size_unlabelled=None
        )
    )

    # IMPORTANT: dataset only — no DataLoader usage
    test_dataset = kfold_loaders[0][2].dataset

    # -------------------------
    # Find misclassified indices
    # -------------------------
    mis_idx = np.where(
        (y_true == TRUE_CLASS) &
        (y_pred == PRED_CLASS)
    )[0]

    print(
        f"Found {len(mis_idx)} samples: "
        f"{CLASS_NAMES[TRUE_CLASS]} → {CLASS_NAMES[PRED_CLASS]}"
    )

    # -------------------------
    # Visualize
    # -------------------------
    plt.figure(figsize=(8, 8))

    for i, idx in enumerate(mis_idx[:MAX_IMAGES]):
        img, _ = test_dataset[idx]

        plt.subplot(4, 4, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(
        f"{CLASS_NAMES[TRUE_CLASS]} → {CLASS_NAMES[PRED_CLASS]}",
        fontsize=14
    )
    plt.tight_layout()
    s = "Misclas_" + f"{CLASS_NAMES[TRUE_CLASS]}" + "_as_" + f"{CLASS_NAMES[PRED_CLASS]}" + ".png"
    s = s.replace("/", "-")

    plt.savefig("results/" + s)
    plt.close()