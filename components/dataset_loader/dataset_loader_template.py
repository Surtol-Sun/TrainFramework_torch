import os
import torch
import torchvision


def cifar10_dataloader(dataloader_config):
    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    # num_workers = os.cpu_count()
    num_workers = 1

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)])
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(dataset_path, train=True, transform=train_transform, download=False)
    test_data = torchvision.datasets.CIFAR10(dataset_path, train=False, transform=test_transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def cifar100_dataloader(dataloader_config):
    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    num_workers = os.cpu_count()

    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)])
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR100(dataset_path, train=True, transform=train_transform, download=False)
    test_data = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

