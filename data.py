from flwr_datasets import FederatedDataset
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10

from clients import NUM_CLIENTS

TEST_SPLIT = 0.2

def load_data_flwr(node_id):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(node_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=TEST_SPLIT)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=32)
    return train_loader, test_loader

def load_data_server_flwr():
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 1})
    partition = fds.load_partition(0)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=1)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(partition_train_test["train"], batch_size=32)#, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=32)
    return train_loader, test_loader

def load_data():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = CIFAR10("./data", train=True, transform=transform)
    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    generator = torch.Generator().manual_seed(49)

    train_datasets = random_split(train_dataset, [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)], generator)
    test_datasets = random_split(test_dataset, [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)], generator)
    
    train_loaders  = [DataLoader(dataset,  batch_size=32, shuffle=True, pin_memory=True) for dataset in train_datasets]
    test_loaders  = [DataLoader(dataset,  batch_size=32, shuffle=True, pin_memory=True) for dataset in test_datasets]

    return list(zip(train_loaders, test_loaders))

def load_data_server():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=True, pin_memory=True)
    train_dataset  = CIFAR10("./data", train=True, transform=transform)
    
    train_loader  = DataLoader(train_dataset,  batch_size=32, shuffle=True, pin_memory=True)
    
    return train_loader, test_loader