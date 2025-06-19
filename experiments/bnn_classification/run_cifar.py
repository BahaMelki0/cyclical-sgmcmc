import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import time
import copy

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.resnet import ResNet18
from samplers.sgld import SGLD
from samplers.sghmc import SGHMC
from samplers.cyclical_sgld import CyclicalSGLD
from samplers.cyclical_sghmc import CyclicalSGHMC
from utils.data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders
from utils.metrics import accuracy, negative_log_likelihood
from utils.visualization import plot_mds_visualization, plot_interpolation


def train_epoch(model, optimizer, train_loader, device, epoch, args):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): model to train
        optimizer: optimizer
        train_loader: training data loader
        device: device to use
        epoch (int): current epoch
        args: command line arguments
    
    Returns:
        tuple: (train_loss, train_acc)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')
    
    return train_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, device):
    """
    Test the model.
    
    Args:
        model (nn.Module): model to test
        test_loader: test data loader
        device: device to use
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def test_ensemble(models, test_loader, device):
    """
    Test an ensemble of models.
    
    Args:
        models (list): list of models
        test_loader: test data loader
        device: device to use
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    for model in models:
        model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Average predictions from all models
            outputs = None
            for model in models:
                if outputs is None:
                    outputs = F.softmax(model(inputs), dim=1)
                else:
                    outputs += F.softmax(model(inputs), dim=1)
            
            outputs /= len(models)
            loss = F.nll_loss(torch.log(outputs), targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def collect_samples(model, optimizer, train_loader, test_loader, device, start_epoch, num_epochs, num_samples, args):
    """
    Collect samples from the posterior distribution.
    
    Args:
        model (nn.Module): model to train
        optimizer: optimizer
        train_loader: training data loader
        test_loader: test data loader
        device: device to use
        start_epoch (int): starting epoch
        num_epochs (int): number of epochs to train
        num_samples (int): number of samples to collect
        args: command line arguments
    
    Returns:
        tuple: (samples, test_losses, test_accs)
    """
    samples = []
    test_losses = []
    test_accs = []
    
    # Calculate epochs at which to collect samples
    if isinstance(optimizer, (CyclicalSGLD, CyclicalSGHMC)):
        # For cyclical samplers, collect samples at the end of each cycle
        cycle_length = num_epochs // args.num_cycles
        sample_epochs = [start_epoch + (i + 1) * cycle_length - cycle_length // 5 for i in range(args.num_cycles)]
        sample_epochs = sample_epochs[:num_samples]
    else:
        # For traditional samplers, collect samples evenly spaced
        sample_epochs = [start_epoch + i * (num_epochs // num_samples) for i in range(num_samples)]
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
        test_loss, test_acc = test(model, test_loader, device)
        
        print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
        
        if epoch in sample_epochs:
            print(f'Collecting sample at epoch {epoch}')
            samples.append(copy.deepcopy(model))
            test_losses.append(test_loss)
            test_accs.append(test_acc)
    
    return samples, test_losses, test_accs


def flatten_params(model):
    """
    Flatten model parameters into a single vector.
    
    Args:
        model (nn.Module): model
    
    Returns:
        torch.Tensor: flattened parameters
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def interpolate_models(model1, model2, alpha):
    """
    Interpolate between two models.
    
    Args:
        model1 (nn.Module): first model
        model2 (nn.Module): second model
        alpha (float): interpolation parameter (0 = model1, 1 = model2)
    
    Returns:
        nn.Module: interpolated model
    """
    interpolated_model = copy.deepcopy(model1)
    
    with torch.no_grad():
        for p1, p2, p_interp in zip(model1.parameters(), model2.parameters(), interpolated_model.parameters()):
            p_interp.data.copy_((1 - alpha) * p1.data + alpha * p2.data)
    
    return interpolated_model


def main():
    parser = argparse.ArgumentParser(description='Run Bayesian neural network experiments on CIFAR')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.0045, help='Temperature for MCMC methods')
    parser.add_argument('--num_cycles', type=int, default=4, help='Number of cycles for cyclical methods')
    parser.add_argument('--exploration_ratio', type=float, default=0.8, help='Proportion of exploration stage in each cycle')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to collect')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='../../results/bnn_classification', help='Directory to save results')
    parser.add_argument('--methods', type=str, nargs='+', default=['sgd', 'sgdm', 'sgld', 'sghmc', 'csgld', 'csghmc', 'snapshot'], help='Methods to run')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloaders(args.batch_size)
        num_classes = 10
        exploration_ratio = 0.8
    else:  # cifar100
        train_loader, test_loader = get_cifar100_dataloaders(args.batch_size)
        num_classes = 100
        exploration_ratio = 0.94
    
    # Dictionary to store results
    results = {}
    
    # Run experiments for each method
    for method in args.methods:
        print(f"\nRunning experiment with method: {method}")
        
        # Create model
        model = ResNet18(num_classes=num_classes).to(device)
        
        # Set optimizer
        if method == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            # Train
            samples = []
            test_losses = []
            test_accs = []
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
                test_loss, test_acc = test(model, test_loader, device)
                scheduler.step()
                
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
                
                if epoch == args.epochs - 1:
                    samples.append(copy.deepcopy(model))
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
            
            # Test
            ensemble_test_loss, ensemble_test_acc = test(samples[0], test_loader, device)
            
        elif method == 'sgdm':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            # Train
            samples = []
            test_losses = []
            test_accs = []
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
                test_loss, test_acc = test(model, test_loader, device)
                scheduler.step()
                
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
                
                if epoch == args.epochs - 1:
                    samples.append(copy.deepcopy(model))
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
            
            # Test
            ensemble_test_loss, ensemble_test_acc = test(samples[0], test_loader, device)
            
        elif method == 'sgld':
            # For traditional SG-MCMC, avoid noise injection for the first 150 epochs
            # First train with SGD
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
            
            for epoch in range(150):
                train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
                test_loss, test_acc = test(model, test_loader, device)
                scheduler.step()
                
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
            
            # Then switch to SGLD
            optimizer = SGLD(model.parameters(), lr=0.01, num_data=len(train_loader.dataset), temperature=args.temperature, weight_decay=args.weight_decay)
            
            # Collect samples
            samples, test_losses, test_accs = collect_samples(model, optimizer, train_loader, test_loader, device, 150, 50, args.num_samples, args)
            
            # Test ensemble
            ensemble_test_loss, ensemble_test_acc = test_ensemble(samples, test_loader, device)
            
        elif method == 'sghmc':
            # For traditional SG-MCMC, avoid noise injection for the first 150 epochs
            # First train with SGDM
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
            
            for epoch in range(150):
                train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
                test_loss, test_acc = test(model, test_loader, device)
                scheduler.step()
                
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
            
            # Then switch to SGHMC
            optimizer = SGHMC(model.parameters(), lr=0.01, num_data=len(train_loader.dataset), momentum=args.momentum, temperature=args.temperature, weight_decay=args.weight_decay)
            
            # Collect samples
            samples, test_losses, test_accs = collect_samples(model, optimizer, train_loader, test_loader, device, 150, 50, args.num_samples, args)
            
            # Test ensemble
            ensemble_test_loss, ensemble_test_acc = test_ensemble(samples, test_loader, device)
            
        elif method == 'csgld':
            optimizer = CyclicalSGLD(
                model.parameters(),
                init_lr=0.5,
                num_data=len(train_loader.dataset),
                num_cycles=args.num_cycles,
                total_iterations=args.epochs * len(train_loader),
                exploration_ratio=exploration_ratio,
                temperature=args.temperature,
                weight_decay=args.weight_decay
            )
            
            # Collect samples
            samples, test_losses, test_accs = collect_samples(model, optimizer, train_loader, test_loader, device, 0, args.epochs, 3 * args.num_cycles, args)
            
            # Test ensemble
            ensemble_test_loss, ensemble_test_acc = test_ensemble(samples, test_loader, device)
            
        elif method == 'csghmc':
            optimizer = CyclicalSGHMC(
                model.parameters(),
                init_lr=0.5,
                num_data=len(train_loader.dataset),
                num_cycles=args.num_cycles,
                total_iterations=args.epochs * len(train_loader),
                exploration_ratio=exploration_ratio,
                momentum=args.momentum,
                temperature=args.temperature,
                weight_decay=args.weight_decay
            )
            
            # Collect samples
            samples, test_losses, test_accs = collect_samples(model, optimizer, train_loader, test_loader, device, 0, args.epochs, 3 * args.num_cycles, args)
            
            # Test ensemble
            ensemble_test_loss, ensemble_test_acc = test_ensemble(samples, test_loader, device)
            
        elif method == 'snapshot':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
            # Snapshot ensemble uses a cyclical learning rate schedule
            cycle_length = args.epochs // args.num_cycles
            
            samples = []
            test_losses = []
            test_accs = []
            
            for epoch in range(args.epochs):
                # Calculate current learning rate using cyclical schedule
                cycle_position = epoch % cycle_length
                lr = args.lr * 0.5 * (np.cos(np.pi * cycle_position / cycle_length) + 1)
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, args)
                test_loss, test_acc = test(model, test_loader, device)
                
                print(f'Epoch: {epoch} | LR: {lr:.6f} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
                
                # Save model at the end of each cycle
                if (epoch + 1) % cycle_length == 0:
                    samples.append(copy.deepcopy(model))
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
            
            # Test ensemble
            ensemble_test_loss, ensemble_test_acc = test_ensemble(samples, test_loader, device)
        
        # Store results
        results[method] = {
            'samples': samples,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'ensemble_test_loss': ensemble_test_loss,
            'ensemble_test_acc': ensemble_test_acc
        }
        
        print(f"{method} ensemble test accuracy: {ensemble_test_acc:.2f}%")
    
    # Save results
    torch.save(results, os.path.join(args.results_dir, f'{args.dataset}_results.pt'))
    
    # Visualize weight space using MDS
    if 'sgld' in results and 'csgld' in results:
        # Flatten parameters
        sgld_weights = [flatten_params(model) for model in results['sgld']['samples']]
        csgld_weights = [flatten_params(model) for model in results['csgld']['samples']]
        
        # Visualize SGLD weights
        plot_mds_visualization(sgld_weights, save_path=os.path.join(args.results_dir, f'{args.dataset}_sgld_mds.png'))
        
        # Visualize cSGLD weights
        plot_mds_visualization(csgld_weights, save_path=os.path.join(args.results_dir, f'{args.dataset}_csgld_mds.png'))
    
    # Interpolate between models
    if 'csgld' in results and len(results['csgld']['samples']) >= 2:
        model1 = results['csgld']['samples'][0]
        model2 = results['csgld']['samples'][-1]
        
        interpolation_errors = []
        alphas = np.linspace(0, 1, 11)
        
        for alpha in alphas:
            interpolated_model = interpolate_models(model1, model2, alpha)
            _, test_acc = test(interpolated_model, test_loader, device)
            interpolation_errors.append(100 - test_acc)  # Convert accuracy to error
        
        plot_interpolation(interpolation_errors, save_path=os.path.join(args.results_dir, f'{args.dataset}_interpolation.png'))
    
    # Print summary of results
    print("\nSummary of results:")
    for method in args.methods:
        print(f"{method}: {results[method]['ensemble_test_acc']:.2f}%")


if __name__ == '__main__':
    main()

