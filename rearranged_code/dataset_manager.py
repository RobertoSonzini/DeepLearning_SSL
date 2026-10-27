import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class SSLDataset(Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x_weak = self.weak_transform(x)
        x_strong = self.strong_transform(x)
        return x_weak, x_strong, -1

class DatasetManager:
    """Base class to manage the datasets"""

    @staticmethod
    def create(dataset_name, budget=1.0, root="./data", seed = 42):
        """Factory method"""
        if dataset_name.lower() == 'fashionmnist':
            return ManagerFashionMNIST(seed, budget, root)
        elif dataset_name.lower() == 'cifar10':
            return ManagerCIFAR10(seed, budget, root)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. \nUse: 'FashionMNIST' o 'CIFAR10'\n")

    def __init__(self, seed, budget=1.0, root="./data"):
        self.budget = budget
        self.root = root
        self.seed = seed
        
        self.mean = None
        self.std = None
        
        self._compute_dataset_stats()
    
    def _get_dataset_class(self):
        raise NotImplementedError("No chosen dataset")
    
    def _get_in_channels(self):
        raise NotImplementedError("No chosen dataset")
    
    def _get_augmentation_transforms(self):
        raise NotImplementedError("No chosen dataset")
    
    def _compute_dataset_stats(self):
        """Compute mean and std on training set"""
        dataset_class = self._get_dataset_class()
        
        full_train = dataset_class(
            root=self.root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        
        loader = DataLoader(
            full_train, batch_size=1024, shuffle=False, num_workers=2
        )
        
        mean = 0.
        std = 0.
        total_samples = 0
        
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples
        
        mean /= total_samples
        std /= total_samples
        
        self.mean = mean
        self.std = std
    
    def get_transforms_base_list(self):
        transform_base_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]

        return transform_base_list
    
    def get_transforms_base(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def get_transforms_weak(self):
        # Eventually no augmentation for the weak case
        weak_aug = []
        base_transform = self.get_transforms_base_list()
        return transforms.Compose(weak_aug + base_transform)

    def get_transforms_strong(self):
        strong_aug = [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
        base_transform = self.get_transforms_base_list()
        return transforms.Compose(strong_aug + base_transform)
    
    def get_labeled_dataset(self, full_dataset, indices):
        subset = Subset(full_dataset, indices)
        return TransformSubset(subset, self.get_transforms_base())

    def get_unlabeled_dataset(self, full_dataset, indices):
        unlabeled_subset = Subset(full_dataset, indices)
        return SSLDataset(
            unlabeled_subset,
            weak_transform=self.get_transforms_weak(),
            strong_transform=self.get_transforms_strong()
        )

    def get_full_loaders(
            self,
            batch_size_labelled=32,
            batch_size_unlabelled=None,
    ):
        """
        Reuse get_kfold_loaders to produce ONE split:
        - train = labeled (+ unlabeled if requested)
        - test = test set
        No validation split.
        """

        # Ask for exactly ONE fold
        kfold_iter = self.get_kfold_loaders(
            n_splits=1,
            batch_size_labelled=batch_size_labelled,
            batch_size_unlabelled=batch_size_unlabelled,
        )

        labeled_loader, unlabeled_loader, _ = next(kfold_iter)

        # get_kfold_loaders always builds the test set internally
        # but exposes only val_loader; we need test_loader separately

        # So we reconstruct the test loader the SAME way the manager does
        test_loader = self.get_test_loader(
            batch_size=batch_size_labelled
        )

        return labeled_loader, unlabeled_loader, test_loader

    def _create_loader(self, dataset, batch_size, shuffle):
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    
    def get_kfold_loaders(self, n_splits=5, batch_size_labelled = 64, batch_size_unlabelled = 64):
        full_train_dataset = self._get_dataset_class()(
            root=self.root, train=True, download=True, transform=None
        )
        
        all_labels = [label for _, label in full_train_dataset]
        indices = list(range(len(full_train_dataset)))

        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(indices, all_labels)):
            
            train_labels = [all_labels[i] for i in train_idx]
            labeled_idx, unlabeled_idx = self._stratified_split(
                train_idx, train_labels, self.budget
            )

            labeled_loader = self._create_loader(
                self.get_labeled_dataset(full_train_dataset, labeled_idx), shuffle=True, batch_size=batch_size_labelled
            )
            unlabeled_loader = self._create_loader(
                self.get_unlabeled_dataset(full_train_dataset, unlabeled_idx), shuffle=True, batch_size=batch_size_unlabelled
            )
            val_loader = self._create_loader(
                self.get_labeled_dataset(full_train_dataset, val_idx), shuffle=False, batch_size=batch_size_labelled
            )
            
            yield labeled_loader, unlabeled_loader, val_loader
    
    def _stratified_split(self, indices, labels, labeled_ratio):
        class_indices = defaultdict(list)
        for idx, label in zip(indices, labels):
            class_indices[label].append(idx)
        
        labeled_idx = []
        unlabeled_idx = []
        
        for class_label, class_idx in class_indices.items():
            n_labeled = max(1, int(len(class_idx) * labeled_ratio))
            labeled_idx.extend(class_idx[:n_labeled])
            unlabeled_idx.extend(class_idx[n_labeled:])
        
        return labeled_idx, unlabeled_idx
    
    def print_summary(self):
        print(f"\n{'DATASET SUMMARY':=^70}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Budget: {self.budget * 100:.1f}%")
        print("\nNORMALIZATION (computed on training set):")
        print(f"  Mean: {tuple(self.mean.tolist())}")
        print(f"  Std:  {tuple(self.std.tolist())}")
        print("="*70)

class ManagerFashionMNIST(DatasetManager):
    """Manager FashionMNIST"""
    
    def __init__(self, seed = 42, budget=1.0, root="./data"):
        self.dataset_name = 'FashionMNIST'
        super().__init__(seed, budget, root)

        self.in_channels = 1

    def _get_in_channels(self):
        return self.in_channels
    
    def _get_dataset_class(self):
        return datasets.FashionMNIST

class ManagerCIFAR10(DatasetManager):
    """Manager CIFAR10"""
    
    def __init__(self, seed = 42, budget=1.0, root="./data"):
        self.dataset_name = 'CIFAR10'
        super().__init__(seed, budget, root)

        self.in_channels = 3

    def _get_in_channels(self):
        return self.in_channels
    
    def _get_dataset_class(self):
        return datasets.CIFAR10