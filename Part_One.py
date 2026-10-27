import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset
import numpy as np

# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
])

# Load dataset
full_train = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

test_set = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# Train/val split
train_size = 55000
val_size = 5000
train_set, val_set = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)

# Label-budget subsampling
def get_labeled_subset(dataset, fraction):
    n_labeled = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), n_labeled, replace=False)
    return Subset(dataset, indices)

labeled_1pct = get_labeled_subset(train_set, 0.01)
labeled_5pct = get_labeled_subset(train_set, 0.05)
labeled_10pct = get_labeled_subset(train_set, 0.10)

# Optional - 100 percent labeled
labeled_100pc = get_labeled_subset(train_set, 1.0)

print("DONE")

