#!/usr/bin/env python3
"""
Basic usage example of Cyclical SG-MCMC for CIFAR-10 classification.

This example demonstrates:
1. Setting up a ResNet-18 model
2. Using CyclicalSGLD optimizer
3. Training with sample collection
4. Evaluating ensemble performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm

# Import cyclical SG-MCMC components
from cyclical_sgmcmc import CyclicalSGLD, ResNet18, get_cifar10_dataloaders
from cyclical_sgmcmc.utils import accuracy


def train_epoch(model, optimizer, train_loader, device, epoch):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{optimizer.get_current_lr():.6f}',
            'Stage': 'Sampling' if optimizer.in_sampling_stage() else 'Exploration'
        })
    
    return train_loss / len(train_loader), 100. * correct / total


def test_model(model, test_loader, device):
    """Test a single model."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def test_ensemble(models, test_loader, device):
    """Test an ensemble of models."""
    for model in models:
        model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Average predictions from all models
            ensemble_output = torch.zeros_like(models[0](data))
            for model in models:
                ensemble_output += F.softmax(model(data), dim=1)
            ensemble_output /= len(models)
            
            test_loss += F.nll_loss(torch.log(ensemble_output), target, reduction='sum').item()
            pred = ensemble_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def main():
    # Configuration
    config = {
        'epochs': 50,           # Reduced for quick example
        'batch_size': 128,
        'init_lr': 0.1,
        'num_cycles': 4,
        'exploration_ratio': 0.8,
        'temperature': 0.0045,
        'weight_decay': 5e-4,
        'samples_per_cycle': 3,
        'seed': 42
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = ResNet18(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    total_iterations = config['epochs'] * len(train_loader)
    optimizer = CyclicalSGLD(
        model.parameters(),
        init_lr=config['init_lr'],
        num_data=len(train_loader.dataset),
        num_cycles=config['num_cycles'],
        total_iterations=total_iterations,
        exploration_ratio=config['exploration_ratio'],
        temperature=config['temperature'],
        weight_decay=config['weight_decay']
    )
    
    print(f"Total iterations: {total_iterations}")
    print(f"Iterations per cycle: {total_iterations // config['num_cycles']}")
    
    # Training loop
    samples = []
    cycle_length = total_iterations // config['num_cycles']
    sample_interval = cycle_length // (config['samples_per_cycle'] * 2)  # Sample during sampling stage
    
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch)
        test_loss, test_acc = test_model(model, test_loader, device)
        
        # Collect samples during sampling stages
        current_iter = epoch * len(train_loader)
        if (optimizer.in_sampling_stage() and 
            current_iter % sample_interval == 0 and 
            len(samples) < config['num_cycles'] * config['samples_per_cycle']):
            print(f"Collecting sample {len(samples) + 1}")
            samples.append(copy.deepcopy(model))
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    
    # Evaluate ensemble
    if samples:
        print(f"\nEvaluating ensemble of {len(samples)} models...")
        ensemble_loss, ensemble_acc = test_ensemble(samples, test_loader, device)
        
        # Compare with final single model
        final_loss, final_acc = test_model(model, test_loader, device)
        
        print(f"\nResults:")
        print(f"Final single model: {final_acc:.2f}% accuracy")
        print(f"Ensemble model: {ensemble_acc:.2f}% accuracy")
        print(f"Improvement: {ensemble_acc - final_acc:.2f}%")
        
        # Analyze sample diversity
        print(f"\nSample diversity analysis:")
        print(f"Collected {len(samples)} samples across {config['num_cycles']} cycles")
        
        # Compute pairwise distances between samples (simplified)
        if len(samples) >= 2:
            distances = []
            for i in range(len(samples)):
                for j in range(i+1, len(samples)):
                    # Compute L2 distance between flattened parameters
                    params1 = torch.cat([p.flatten() for p in samples[i].parameters()])
                    params2 = torch.cat([p.flatten() for p in samples[j].parameters()])
                    dist = torch.norm(params1 - params2).item()
                    distances.append(dist)
            
            print(f"Average pairwise distance: {np.mean(distances):.3f}")
            print(f"Distance std: {np.std(distances):.3f}")
    else:
        print("No samples collected during training!")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

