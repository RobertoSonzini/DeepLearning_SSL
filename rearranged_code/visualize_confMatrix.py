import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------
# FashionMNIST class labels
# -------------------------
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# -------------------------
# Load predictions
# -------------------------
y_true = np.load("results/confusion_budget_0.05/y_true.npy")
y_pred = np.load("results/confusion_budget_0.05/y_pred.npy")

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – SSL (budget = 0.05)")
plt.tight_layout()
plt.savefig("results/confusion_budget_0.05/confusion_matrix.png")
