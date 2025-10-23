"""
Data loading utilities for PCOS NAS
Handles stratified train/val split and data augmentation
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np


class PCOSDataLoader:
    """
    PCOS Dataset loader with stratified split
    Automatically splits training data into train/val for DARTS
    """
    
    def __init__(self, data_path, test_path, batch_size=64, num_workers=4, 
                 val_split=0.5, image_size=224, use_augmentation=False):
        """
        Args:
            data_path: Path to training data directory
            test_path: Path to test data directory
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            val_split: Fraction of training data to use for validation (0.5 = 50%)
            image_size: Image size for resizing
            use_augmentation: Whether to use data augmentation (False for search, True for final training)
        """
        self.data_path = data_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        
        # Define transforms
        self._setup_transforms()
        
        # Load datasets
        self._load_datasets()
        
        # Calculate class weights
        self._calculate_class_weights()
    
    def _setup_transforms(self):
        """Setup data transformations"""
        
        # Normalization (ImageNet stats - common for transfer learning)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.use_augmentation:
            # Aggressive augmentation for final training
            self.train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # Minimal augmentation for architecture search (faster)
            self.train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        
        # Validation/Test transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    def _load_datasets(self):
        """Load datasets and create stratified split"""
        
        print("Loading PCOS dataset...")
        print("=" * 70)
        
        # Load full training dataset
        full_train_dataset = datasets.ImageFolder(
            self.data_path,
            transform=self.train_transform
        )
        
        # Get all targets for stratification
        targets = np.array([label for _, label in full_train_dataset.samples])
        
        # Get indices
        indices = np.arange(len(full_train_dataset))
        
        # Stratified split
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_split,
            stratify=targets,
            random_state=42
        )
        
        # Create samplers
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)
        
        # Store datasets
        self.train_dataset = full_train_dataset
        
        # Create validation dataset with val transform
        self.val_dataset = datasets.ImageFolder(
            self.data_path,
            transform=self.val_transform
        )
        
        # Load test dataset
        self.test_dataset = datasets.ImageFolder(
            self.test_path,
            transform=self.val_transform
        )
        
        # Print statistics
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]
        test_targets = np.array([label for _, label in self.test_dataset.samples])
        
        print(f"\nDataset Split Statistics:")
        print(f"{'Split':<15} {'Total':<10} {'Negative':<10} {'Positive':<10} {'Pos %':<10}")
        print("-" * 70)
        
        print(f"{'Train':<15} {len(train_idx):<10} "
              f"{np.sum(train_targets == 0):<10} {np.sum(train_targets == 1):<10} "
              f"{np.sum(train_targets == 1) / len(train_targets) * 100:.1f}%")
        
        print(f"{'Validation':<15} {len(val_idx):<10} "
              f"{np.sum(val_targets == 0):<10} {np.sum(val_targets == 1):<10} "
              f"{np.sum(val_targets == 1) / len(val_targets) * 100:.1f}%")
        
        print(f"{'Test':<15} {len(self.test_dataset):<10} "
              f"{np.sum(test_targets == 0):<10} {np.sum(test_targets == 1):<10} "
              f"{np.sum(test_targets == 1) / len(test_targets) * 100:.1f}%")
        
        print(f"{'Total':<15} "
              f"{len(train_idx) + len(val_idx) + len(self.test_dataset):<10}")
        
        print("=" * 70)
        
        # Store class names
        self.class_names = self.train_dataset.classes
        print(f"\nClass mapping: {dict(enumerate(self.class_names))}")
        print()
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        targets = np.array([label for _, label in self.train_dataset.samples])
        train_targets = targets[list(self.train_sampler)]
        
        # Count samples per class
        class_counts = np.bincount(train_targets)
        
        # Calculate weights (inverse frequency)
        total = len(train_targets)
        weights = total / (len(class_counts) * class_counts)
        
        # Normalize
        weights = weights / weights.sum() * len(weights)
        
        self.class_weights = torch.FloatTensor(weights)
        
        print(f"Class weights (for loss function): {self.class_weights.numpy()}")
        print()
    
    def get_train_loader(self, batch_size=None):
        """Get training data loader"""
        batch_size = batch_size or self.batch_size
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
    
    def get_val_loader(self, batch_size=None):
        """Get validation data loader"""
        batch_size = batch_size or self.batch_size
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_test_loader(self, batch_size=None):
        """Get test data loader"""
        batch_size = batch_size or self.batch_size
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


if __name__ == '__main__':
    # Test data loader
    print("\nTesting PCOS Data Loader...")
    print("=" * 70)
    
    loader = PCOSDataLoader(
        data_path='./data/train',
        test_path='./data/test',
        batch_size=32,
        num_workers=4,
        val_split=0.5,
        image_size=224,
        use_augmentation=False
    )
    
    # Test train loader
    train_loader = loader.get_train_loader()
    print(f"Train loader: {len(train_loader)} batches")
    
    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test val loader
    val_loader = loader.get_val_loader()
    print(f"\nValidation loader: {len(val_loader)} batches")
    
    # Test test loader
    test_loader = loader.get_test_loader()
    print(f"Test loader: {len(test_loader)} batches")
    
    print("\n✓ Data loader working correctly!")
