import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    Get CIFAR-10 dataloaders for training and testing.
    
    Args:
        batch_size (int): batch size
        num_workers (int): number of workers for data loading
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Just normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def get_cifar100_dataloaders(batch_size=128, num_workers=4):
    """
    Get CIFAR-100 dataloaders for training and testing.
    
    Args:
        batch_size (int): batch size
        num_workers (int): number of workers for data loading
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Just normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def get_svhn_dataloaders(batch_size=128, num_workers=4):
    """
    Get SVHN dataloaders for training and testing.
    
    Args:
        batch_size (int): batch size
        num_workers (int): number of workers for data loading
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Normalization for training and testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def generate_synthetic_data(num_samples=50000):
    """
    Generate synthetic data from a mixture of 25 Gaussians.
    
    Args:
        num_samples (int): number of samples to generate
    
    Returns:
        tuple: (samples, mode_centers)
    """
    # Define the mixture of Gaussians
    num_modes = 25
    mode_centers = []
    for x in [-4, -2, 0, 2, 4]:
        for y in [-4, -2, 0, 2, 4]:
            mode_centers.append([x, y])
    mode_centers = np.array(mode_centers)
    
    # Generate samples
    samples = np.zeros((num_samples, 2))
    mode_indices = np.random.choice(num_modes, num_samples)
    
    for i in range(num_samples):
        mode_idx = mode_indices[i]
        samples[i] = np.random.multivariate_normal(
            mean=mode_centers[mode_idx],
            cov=np.array([[0.03, 0], [0, 0.03]])
        )
    
    return samples, mode_centers

