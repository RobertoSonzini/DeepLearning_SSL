import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset_manager import DatasetManager
from calibration_utils import (
    collect_logits_and_labels,
    expected_calibration_error,
    fit_temperature,
)
from robustness_utils import CorruptedDataset, CORRUPTIONS

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# UTILITIES
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# SIMPLE CNN BACKBONE
class SmallCNN(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14×14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 7×7
        )
        self.fc = nn.Linear(64 * 7 * 7, feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FullModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.backbone(x))



# PROJECTION HEAD FOR REPRESENTATION LEARNING
class ProjectionHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)



# NT-CrossEntropy LOSS
def nt_xent_loss(z1, z2, temperature=0.2):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])

    return nn.functional.cross_entropy(sim, labels)



# Image AUGMENTATIONS
ssl_aug = T.Compose([
    T.RandomResizedCrop(28, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
])


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    dataset_name = "FashionMNIST"
    budgets = [0.01, 0.05, 0.1, 1.0]

    pretrain_epochs = 20
    probe_epochs = 5
    batch_size = 256
    feat_dim = 128

    all_results = {}

    for budget in budgets:
        print(f"\n{'='*80}")
        print(f"SELF-SUPERVISED (SmallCNN) | Budget = {budget}")
        print(f"{'='*80}")

        results = {
            "method": "self_supervised_pretrain_then_probe",
            "dataset": dataset_name,
            "budget": budget,
            "pretrain_loss": [],
            "probe_loss": [],
            "accuracy": None,
            "ECE": None,
            "ECE_TS": None,
            "T": None,
            "robustness": {}
        }
        manager = DatasetManager.create(dataset_name, budget, "./data")

        labeled_loader, unlabeled_loader, val_loader = list(
            manager.get_kfold_loaders(
                n_splits=2,
                batch_size_labelled=batch_size,
                batch_size_unlabelled=batch_size
            )
        )[0]

        # -------------------------
        # MODELS
        # -------------------------
        backbone = SmallCNN(feat_dim).to(device)
        projector = ProjectionHead(feat_dim).to(device)

        optimizer = optim.AdamW(
            list(backbone.parameters()) + list(projector.parameters()),
            lr=3e-4,
            weight_decay=1e-4
        )

        # =========================
        # STAGE 1: SELF-SUPERVISED
        # =========================
        if unlabeled_loader is not None:
            print("[Stage 1] Self-supervised pretraining")
            backbone.train()
            projector.train()

            for epoch in range(pretrain_epochs):
                total_loss = 0.0

                for batch in unlabeled_loader:
                    x = batch[0].to(device)

                    x1 = torch.stack([ssl_aug(img.cpu()).to(device) for img in x])
                    x2 = torch.stack([ssl_aug(img.cpu()).to(device) for img in x])

                    z1 = projector(backbone(x1))
                    z2 = projector(backbone(x2))

                    loss = nt_xent_loss(z1, z2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(unlabeled_loader)
                results["pretrain_loss"].append(avg_loss)
                print(f"[SSL] Epoch {epoch+1}: {avg_loss:.4f}")
        else:
            print("[Stage 1] Skipped (budget = 1.0)")

        # =========================
        # STAGE 2: LINEAR PROBE
        # =========================
        for p in backbone.parameters():
            p.requires_grad = False

        classifier = nn.Linear(feat_dim, 10).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(probe_epochs):
            classifier.train()
            total_loss = 0.0

            for x, y in labeled_loader:
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    feats = backbone(x)

                loss = criterion(classifier(feats), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(labeled_loader)
            results["probe_loss"].append(avg_loss)
            print(f"[Probe] Epoch {epoch+1}: {avg_loss:.4f}")

        full_model = FullModel(backbone, classifier).to(device)
        full_model.eval()

        # ACCURACY
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = full_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        results["accuracy"] = acc
        print(f"Accuracy: {acc:.4f}")

        # CALIBRATION
        logits_val, labels_val = collect_logits_and_labels(
            full_model, val_loader, device
        )

        ece = expected_calibration_error(logits_val, labels_val)
        scaler = fit_temperature(full_model, val_loader, device)
        T = scaler.temperature

        with torch.no_grad():
            scaled_logits = scaler(logits_val.to(device)).cpu()

        ece_ts = expected_calibration_error(scaled_logits, labels_val)

        results["ECE"] = float(ece)
        results["ECE_TS"] = float(ece_ts)
        results["T"] = float(T)

        # ROBUSTNESS
        base_dataset = val_loader.dataset
        severities = [0, 1, 2, 3, 4, 5]

        for corr in CORRUPTIONS.keys():
            accs = []
            for sev in severities:
                corr_ds = CorruptedDataset(base_dataset, corr, sev)
                corr_loader = DataLoader(
                    corr_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )

                correct = total = 0
                with torch.no_grad():
                    for x, y in corr_loader:
                        x, y = x.to(device), y.to(device)
                        preds = full_model(x).argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)

                accs.append(correct / total)

            results["robustness"][corr] = {
                "severities": severities,
                "accuracy": accs
            }

        all_results[str(budget)] = results

    os.makedirs("results", exist_ok=True)
    out_path = "results/self_supervised_smallcnn_all_budgets.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {out_path}")
