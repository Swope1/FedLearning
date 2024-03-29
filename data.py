import numpy as np
import random
from flwr_datasets import FederatedDataset
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10, MNIST

from clients import NUM_CLIENTS

BATCH_SIZE = 32
TEST_SPLIT = 0.2
NUM_CLASSES = 10
SEED = 1

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
    train_loader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
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
    train_loader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE)#, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    return train_loader, test_loader

def load_data_CIFAR10():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = CIFAR10("./data", train=True, transform=transform)
    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    generator = torch.Generator().manual_seed(SEED)

    train_datasets = random_split(train_dataset, [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)], generator)
    test_datasets = random_split(test_dataset, [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)], generator)
    
    train_loaders  = [DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in train_datasets]
    test_loaders  = [DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in test_datasets]

    return list(zip(train_loaders, test_loaders))

def load_data_server_CIFAR10():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    train_dataset  = CIFAR10("./data", train=True, transform=transform)
    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    train_loader  = DataLoader(train_dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    return train_loader, test_loader

def load_data_non_iid_CIFAR10(num_classes_per_node):
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    random.seed(SEED)

    train_dataset = CIFAR10("./data", train=True, transform=transform)
    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    classes_included = np.array([0])
    while NUM_CLASSES < NUM_CLIENTS * num_classes_per_node and len(np.unique(classes_included)) != NUM_CLASSES:
        classes_included = np.array([random.sample(range(NUM_CLASSES), num_classes_per_node) for _ in range(NUM_CLIENTS)])
    
    classes_included_counts = [np.count_nonzero(classes_included == i) for i in range(NUM_CLASSES)]
    
    generator = torch.Generator().manual_seed(SEED)
    
    train_dataset_class_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in range(NUM_CLASSES)]
    test_dataset_class_indices = [np.where(np.array(test_dataset.targets) == i)[0] for i in range(NUM_CLASSES)]
    
    train_datasets_per_class = [Subset(train_dataset, train_dataset_class_indices[i]) for i in range(NUM_CLASSES)]
    test_datasets_per_class = [Subset(test_dataset, test_dataset_class_indices[i]) for i in range(NUM_CLASSES)]
    
    train_splits = [[int(1/classes_included_counts[i] * len(train_datasets_per_class[i])) for _ in range(classes_included_counts[i])] for i in range(NUM_CLASSES)]
    test_splits = [[int(1/classes_included_counts[i] * len(test_datasets_per_class[i])) for _ in range(classes_included_counts[i])] for i in range(NUM_CLASSES)]
    
    for i in range(NUM_CLASSES):
        train_splits[i][0] += len(train_datasets_per_class[i]) - sum(train_splits[i])
        test_splits[i][0] += len(test_datasets_per_class[i]) - sum(test_splits[i])
        
    train_datasets_per_class = [iter(random_split(train_datasets_per_class[i], train_splits[i], generator)) for i in range(NUM_CLASSES)]
    test_datasets_per_class = [iter(random_split(test_datasets_per_class[i], test_splits[i], generator)) for i in range(NUM_CLASSES)]
    
    train_datasets = [[next(train_datasets_per_class[class_num]) for class_num in client] for client in classes_included]
    test_datasets = [[next(test_datasets_per_class[class_num]) for class_num in client] for client in classes_included]
    
    train_datasets = [ConcatDataset(dataset) for dataset in train_datasets]
    test_datasets = [ConcatDataset(dataset) for dataset in test_datasets]
    
    train_loaders  = [DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in train_datasets]
    test_loaders  = [DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in test_datasets]

    return list(zip(train_loaders, test_loaders))

def load_pr_data_CIFAR10():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    test_dataset  = CIFAR10("./data", train=False, transform=transform)
    
    test_loader  = DataLoader(test_dataset,  batch_size=10000, shuffle=True, pin_memory=True)
    
    return test_loader

def load_data_MNIST():
    transform = Compose([
        ToTensor(),
        Normalize((0.1307), (0.3081))
    ])

    client_dataset = MNIST("./data", train=True, transform=transform)
    server_test_dataset  = MNIST("./data", train=False, transform=transform)
    
    generator = torch.Generator().manual_seed(SEED)

    datasets = random_split(client_dataset, [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)], generator)
    
    datasets = [random_split(dataset, [1 - TEST_SPLIT,  TEST_SPLIT]) for dataset in datasets]
    
    loaders  = [[DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in client] for client in datasets]
    server_test_loader  = DataLoader(server_test_dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    return loaders, server_test_loader

def load_data_non_iid_MNIST(num_classes_per_node):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307), (0.3081))
    ])

    random.seed(SEED)

    dataset = MNIST("./data", train=True, transform=transform)
    server_test_dataset  = MNIST("./data", train=False, transform=transform)
    
    classes_included = np.array([0])
    while NUM_CLASSES < NUM_CLIENTS * num_classes_per_node and len(np.unique(classes_included)) != NUM_CLASSES:
        classes_included = np.array([random.sample(range(NUM_CLASSES), num_classes_per_node) for _ in range(NUM_CLIENTS)])
    
    classes_included_counts = [np.count_nonzero(classes_included == i) for i in range(NUM_CLASSES)]
    
    generator = torch.Generator().manual_seed(SEED)
    
    dataset_class_indices = [np.where(np.array(dataset.targets) == i)[0] for i in range(NUM_CLASSES)]
    
    datasets_per_class = [Subset(dataset, dataset_class_indices[i]) for i in range(NUM_CLASSES)]
    
    dataset_splits = [[int(1/classes_included_counts[i] * len(datasets_per_class[i])) for _ in range(classes_included_counts[i])] for i in range(NUM_CLASSES)]
    
    for i in range(NUM_CLASSES):
        dataset_splits[i][0] += len(datasets_per_class[i]) - sum(dataset_splits[i])
        
    datasets_per_class = [iter(random_split(datasets_per_class[i], dataset_splits[i], generator)) for i in range(NUM_CLASSES)]
    
    datasets = [[next(datasets_per_class[class_num]) for class_num in client] for client in classes_included]
    
    datasets = [ConcatDataset(dataset) for dataset in datasets]
    
    datasets = [random_split(dataset, [1 - TEST_SPLIT,  TEST_SPLIT]) for dataset in datasets]
    
    loaders  = [[DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) for dataset in client] for client in datasets]
    server_test_loader  = DataLoader(server_test_dataset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    return loaders, server_test_loader

def load_pr_data_MNIST():
    transform = Compose([
        ToTensor(),
        Normalize((0.1307), (0.3081))
    ])

    test_dataset  = MNIST("./data", train=False, transform=transform)
    
    test_loader  = DataLoader(test_dataset,  batch_size=10000, shuffle=True, pin_memory=True)
    
    return test_loader