import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import os

# Base directory for all datasets
BASE_DIR = Path(os.getcwd())

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_data(dataset_name: str, partition_id: int, num_partitions: int):
    """Load partition of the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset folder (e.g., "chest_xray", "Alzheimer")
        partition_id (int): Client ID (0 to num_partitions-1)
        num_partitions (int): Total number of clients
        
    Returns:
        trainloader, valloader, num_classes
    """
    
    data_root = BASE_DIR / dataset_name
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found at {train_dir}")
    if not val_dir.exists():
        # Fallback: if 'val' doesn't exist, use 'test' or split train? 
        # For now, let's assume structure is consistent or check 'test'
        test_dir = data_root / "test"
        if test_dir.exists():
            val_dir = test_dir
        else:
            raise FileNotFoundError(f"Validation/Test folder not found at {val_dir} or {test_dir}")

    transform = get_transforms()
    
    # Load entire training set
    full_train_dataset = ImageFolder(root=train_dir, transform=transform)
    
    # Determine classes dynamically
    classes = full_train_dataset.classes
    num_classes = len(classes)
    
    # Split training set into partitions
    num_train = len(full_train_dataset)
    partition_size = num_train // num_partitions
    lengths = [partition_size] * num_partitions
    # Add remainder to the last partition
    lengths[-1] += num_train - sum(lengths)
    
    # Use a fixed generator for reproducibility across client launches
    datasets = random_split(full_train_dataset, lengths, generator=torch.Generator().manual_seed(42))
    
    if partition_id >= len(datasets):
        raise ValueError(f"Partition ID {partition_id} out of range for {len(datasets)} partitions.")
        
    train_partition = datasets[partition_id]
    
    # Load validation set
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32)

    return trainloader, valloader, num_classes

def get_num_classes(dataset_name: str) -> int:
    """Get the number of classes for a dataset without loading loaders."""
    data_root = BASE_DIR / dataset_name
    train_dir = data_root / "train"
    if not train_dir.exists():
        # Quick check for fallback or error
        return 2 # Default fallback or raise error
    
    # We can use ImageFolder on just the root to find classes quickly, 
    # but ImageFolder expects subfolders.
    # A faster way is just os.listdir or utilizing ImageFolder.find_classes
    try:
        # Check standard ImageFolder structure
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        return len(classes)
    except Exception:
        return 2
