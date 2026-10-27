import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _clamp01(x):
    return x.clamp(0.0, 1.0)


def corruption_gaussian_noise(x, severity):
    # severity in [0..5]
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    sigma = sigmas[severity]
    return _clamp01(x + sigma * torch.randn_like(x))


def corruption_blur(x, severity):
    # simple gaussian blur kernel sizes
    ks = [1, 3, 5, 7, 9, 11]
    k = ks[severity]
    if k == 1:
        return x
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=None)


def corruption_brightness(x, severity):
    factors = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]  # darker
    return _clamp01(TF.adjust_brightness(x, factors[severity]))


def corruption_contrast(x, severity):
    factors = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3]
    return _clamp01(TF.adjust_contrast(x, factors[severity]))


def corruption_rotate(x, severity):
    degrees = [0, 5, 10, 15, 20, 30]
    return TF.rotate(x, degrees[severity])


def corruption_cutout(x, severity):
    # cutout square relative size
    fracs = [0.0, 0.12, 0.18, 0.25, 0.32, 0.40]
    frac = fracs[severity]
    if frac <= 0:
        return x

    c, h, w = x.shape
    cut_h = int(h * frac)
    cut_w = int(w * frac)

    # random top-left
    top = torch.randint(0, max(1, h - cut_h + 1), (1,)).item()
    left = torch.randint(0, max(1, w - cut_w + 1), (1,)).item()

    x2 = x.clone()
    x2[:, top:top + cut_h, left:left + cut_w] = 0.0
    return x2


CORRUPTIONS = {
    "gaussian_noise": corruption_gaussian_noise,
    "blur": corruption_blur,
    "brightness": corruption_brightness,
    "contrast": corruption_contrast,
    "rotate": corruption_rotate,
    "cutout": corruption_cutout,
}


class CorruptedDataset(Dataset):
    """
    Wrap a base dataset that returns (x, y) where x is a torch tensor in [0,1].
    Applies a chosen corruption at __getitem__ time.
    """
    def __init__(self, base_dataset, corruption_name, severity):
        self.base = base_dataset
        self.fn = CORRUPTIONS[corruption_name]
        self.corruption_name = corruption_name
        self.severity = int(severity)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # Ensure tensor
        if not torch.is_tensor(x):
            x = TF.to_tensor(x)
        x = self.fn(x, self.severity)
        return x, y
