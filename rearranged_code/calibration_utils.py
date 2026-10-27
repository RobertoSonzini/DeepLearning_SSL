import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def expected_calibration_error(logits, labels, n_bins=15):
    """
    Standard ECE over max-softmax confidence bins.
    logits: (N, C) torch tensor
    labels: (N,) torch tensor
    """
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    acc = preds.eq(labels).float()

    conf = conf.numpy()
    acc = acc.numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge on last bin
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if mask.sum() == 0:
            continue

        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(bin_acc - bin_conf)

    return float(ece)


def plot_reliability_diagram(logits, labels, n_bins=15, title="Reliability Diagram"):
    probs = torch.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    acc = preds.eq(labels).float()

    conf = conf.numpy()
    acc = acc.numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accs = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        bin_accs.append(acc[mask].mean() if mask.sum() > 0 else 0.0)

    ece = expected_calibration_error(logits, labels, n_bins=n_bins)

    plt.figure(figsize=(5, 5))
    plt.bar(bin_centers, bin_accs, width=1.0/n_bins, edgecolor="black")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title}\nECE={ece:.4f}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return ece


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling: scaled_logits = logits / T, where T = exp(log_T) > 0
    """
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T

    @property
    def temperature(self):
        return float(torch.exp(self.log_T).detach().cpu().item())


def fit_temperature(model, val_loader, device, max_iter=50, lr=0.05):
    """
    Fit T on validation set by minimizing NLL (cross-entropy).
    """
    scaler = TemperatureScaler().to(device)
    model.eval()

    logits_cpu, labels_cpu = collect_logits_and_labels(model, val_loader, device)
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)

    optimizer = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler
