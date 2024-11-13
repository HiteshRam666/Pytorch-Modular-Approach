
"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set the number of workers based on the available CPUs
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 2

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders for image classification.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (torchvision.transforms.Compose): Image transformations.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: Training DataLoader, testing DataLoader, class names.
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
