import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Standard transform for labeled data
transform_standard = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
])

# Augmented transform for unlabeled data (consistency regularization)
transform_augmented = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
])

# Load dataset
full_train = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform_standard
)

test_set = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform_standard
)

# Train/val split
from torch.utils.data import random_split
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
    return Subset(dataset, indices), indices

labeled_1pct, labeled_1pct_indices = get_labeled_subset(train_set, 0.01)
labeled_5pct, labeled_5pct_indices = get_labeled_subset(train_set, 0.05)
labeled_10pct, labeled_10pct_indices = get_labeled_subset(train_set, 0.10)

# ============================================================================
# SEMI-SUPERVISED DATASET
# ============================================================================

class SSLDataset(Dataset):
    """Dataset that returns both original and augmented versions for SSL"""
    def __init__(self, dataset, transform_weak, transform_strong):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get raw image and label - handle nested Subsets
        current = self.dataset
        
        # Traverse through potential nested Subsets
        actual_idx = idx
        while isinstance(current, Subset):
            actual_idx = current.indices[actual_idx]
            current = current.dataset
        
        # Now current is the base dataset
        img = current.data[actual_idx]
        label = current.targets[actual_idx]
        
        # Convert to PIL for transforms
        from PIL import Image
        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy(), mode='L')
        else:
            img = Image.fromarray(img, mode='L')
        
        # Apply transforms
        img_weak = self.transform_weak(img)
        img_strong = self.transform_strong(img)
        
        return img_weak, img_strong, label

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# SEMI-SUPERVISED LEARNING TRAINER
# ============================================================================

class SSLTrainer:
    def __init__(self, model, device, lambda_u=1.0, threshold=0.90):
        """
        Args:
            model: Neural network model
            device: cuda or cpu
            lambda_u: Weight for unsupervised loss
            threshold: Confidence threshold for pseudo-labels
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_u = lambda_u
        self.threshold = threshold
    
    def train_epoch(self, labeled_loader, unlabeled_loader, optimizer):
        self.model.train()
        total_loss = 0
        supervised_loss_total = 0
        unsupervised_loss_total = 0
        
        # Create iterator for unlabeled data
        unlabeled_iter = iter(unlabeled_loader)
        
        for batch_idx, batch_data in enumerate(labeled_loader):
            # Unpack labeled data (SSLDataset returns 3 values)
            x_l, _, y_l = batch_data
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            
            # Supervised loss
            logits_l = self.model(x_l)
            loss_supervised = F.cross_entropy(logits_l, y_l)
            
            # Unsupervised loss with pseudo-labeling
            try:
                x_u_weak, x_u_strong, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u_weak, x_u_strong, _ = next(unlabeled_iter)
            
            x_u_weak = x_u_weak.to(self.device)
            x_u_strong = x_u_strong.to(self.device)
            
            # Generate pseudo-labels from weak augmentation
            with torch.no_grad():
                logits_u_weak = self.model(x_u_weak)
                probs_u = F.softmax(logits_u_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs_u, dim=1)
                
                # Only use high-confidence predictions
                mask = max_probs >= self.threshold
            
            if mask.sum() > 0:
                # Predict on strong augmentation
                logits_u_strong = self.model(x_u_strong)
                
                # Unsupervised loss (consistency regularization)
                loss_unsupervised = (F.cross_entropy(
                    logits_u_strong, pseudo_labels, reduction='none'
                ) * mask).mean()
            else:
                loss_unsupervised = torch.tensor(0.).to(self.device)
            
            # Combined loss
            loss = loss_supervised + self.lambda_u * loss_unsupervised
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            supervised_loss_total += loss_supervised.item()
            unsupervised_loss_total += loss_unsupervised.item()
        
        n_batches = len(labeled_loader)
        return {
            'total_loss': total_loss / n_batches,
            'supervised_loss': supervised_loss_total / n_batches,
            'unsupervised_loss': unsupervised_loss_total / n_batches
        }
    
    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return 100 * correct / total

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ssl_model(labeled_subset, train_set, val_set, test_set, 
                    labeled_indices, epochs=50, batch_size=64, lr=0.001):
    """
    Train a semi-supervised model
    
    Args:
        labeled_subset: Subset with labeled data
        train_set: Full training set (for unlabeled data)
        val_set: Validation set
        test_set: Test set
        labeled_indices: Indices of labeled samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Create unlabeled dataset (exclude labeled samples)
    all_indices = set(range(len(train_set)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    unlabeled_subset = Subset(train_set, unlabeled_indices)
    
    print(f"Labeled samples: {len(labeled_subset)}")
    print(f"Unlabeled samples: {len(unlabeled_subset)}")
    
    # Create SSL datasets
    labeled_ssl = SSLDataset(labeled_subset, transform_standard, transform_standard)
    unlabeled_ssl = SSLDataset(unlabeled_subset, transform_standard, transform_augmented)
    
    # Create dataloaders
    labeled_loader = DataLoader(labeled_ssl, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ssl, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = CNN(num_classes=10)
    trainer = SSLTrainer(model, device, lambda_u=2.0, threshold=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        losses = trainer.train_epoch(labeled_loader, unlabeled_loader, optimizer)
        val_acc = trainer.evaluate(val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {losses['total_loss']:.4f} "
                  f"(Sup: {losses['supervised_loss']:.4f}, "
                  f"Unsup: {losses['unsupervised_loss']:.4f})")
            print(f"  Val Acc: {val_acc:.2f}%")
    
    # Load best model and test
    model.load_state_dict(best_model_state)
    test_acc = trainer.evaluate(test_loader)
    
    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    
    return model, test_acc

# ============================================================================
# SUPERVISED BASELINE FOR COMPARISON
# ============================================================================

def train_supervised_baseline(labeled_subset, val_set, test_set, 
                              epochs=50, batch_size=64, lr=0.001):
    """Train supervised baseline (no SSL)"""
    
    labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model = CNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.2f}%")
    
    # Test
    model.load_state_dict(best_model_state)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    return model, test_acc

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENTS: Supervised Baseline vs Semi-Supervised Learning")
print("="*70)

results = {}

for name, labeled_data, indices in [
    ("1%", labeled_1pct, labeled_1pct_indices),
    ("5%", labeled_5pct, labeled_5pct_indices),
    ("10%", labeled_10pct, labeled_10pct_indices)
]:
    print(f"\n{'='*70}")
    print(f"Training with {name} labeled data ({len(labeled_data)} samples)")
    print(f"{'='*70}")
    
    print(f"\n--- Supervised Baseline ({name}) ---")
    _, baseline_acc = train_supervised_baseline(labeled_data, val_set, test_set, epochs=50)
    
    print(f"\n--- Semi-Supervised Learning ({name}) ---")
    _, ssl_acc = train_ssl_model(labeled_data, train_set, val_set, test_set, 
                                  indices, epochs=50)
    
    results[name] = {
        'supervised': baseline_acc,
        'ssl': ssl_acc,
        'improvement': ssl_acc - baseline_acc
    }

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Label %':<10} {'Supervised':<15} {'SSL':<15} {'Improvement':<15}")
print("-"*70)
for name, res in results.items():
    print(f"{name:<10} {res['supervised']:<15.2f} {res['ssl']:<15.2f} "
          f"{res['improvement']:<15.2f}")
print("="*70)