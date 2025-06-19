import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils import generate_synthetic_data
from utils.metrics import mode_coverage
from utils.visualization import plot_density_2d, plot_learning_rate_schedule

class GaussianMixtureSampler:
    """
    Sampler for a mixture of Gaussians.
    """
    def __init__(self, mode_centers, cov=np.array([[0.03, 0], [0, 0.03]])):
        """
        Initialize the sampler.
        
        Args:
            mode_centers (numpy.ndarray): centers of the modes
            cov (numpy.ndarray): covariance matrix for each mode
        """
        self.mode_centers = mode_centers
        self.cov = cov
        self.num_modes = len(mode_centers)
        self.dim = mode_centers.shape[1]
        
        # Initialize parameters
        self.theta = np.random.randn(self.dim)
    
    def log_density(self, x):
        """
        Compute the log density of the mixture of Gaussians.
        
        Args:
            x (numpy.ndarray): point to evaluate
        
        Returns:
            float: log density
        """
        log_probs = []
        for i in range(self.num_modes):
            mean = self.mode_centers[i]
            diff = x - mean
            log_prob = -0.5 * np.dot(diff, np.linalg.solve(self.cov, diff)) - 0.5 * np.log(np.linalg.det(self.cov)) - 0.5 * self.dim * np.log(2 * np.pi)
            log_probs.append(log_prob)
        
        # Log sum exp trick for numerical stability
        max_log_prob = np.max(log_probs)
        log_probs_shifted = log_probs - max_log_prob
        return max_log_prob + np.log(np.sum(np.exp(log_probs_shifted)) / self.num_modes)
    
    def grad_log_density(self, x):
        """
        Compute the gradient of the log density.
        
        Args:
            x (numpy.ndarray): point to evaluate
        
        Returns:
            numpy.ndarray: gradient of log density
        """
        grad = np.zeros(self.dim)
        probs = []
        
        for i in range(self.num_modes):
            mean = self.mode_centers[i]
            diff = x - mean
            log_prob = -0.5 * np.dot(diff, np.linalg.solve(self.cov, diff)) - 0.5 * np.log(np.linalg.det(self.cov)) - 0.5 * self.dim * np.log(2 * np.pi)
            probs.append(np.exp(log_prob))
        
        probs = np.array(probs) / np.sum(probs)
        
        for i in range(self.num_modes):
            mean = self.mode_centers[i]
            diff = x - mean
            grad += probs[i] * (-np.linalg.solve(self.cov, diff))
        
        return grad
    
    def sgld_step(self, lr):
        """
        Perform a SGLD step.
        
        Args:
            lr (float): learning rate (stepsize)
        """
        grad = self.grad_log_density(self.theta)
        noise = np.random.randn(self.dim) * np.sqrt(2 * lr)
        self.theta = self.theta + lr * grad + noise
    
    def csgld_step(self, lr, in_exploration):
        """
        Perform a cSGLD step.
        
        Args:
            lr (float): learning rate (stepsize)
            in_exploration (bool): whether in exploration stage
        """
        grad = self.grad_log_density(self.theta)
        
        if in_exploration:
            # In exploration stage, perform SGD update (no noise)
            self.theta = self.theta + lr * grad
        else:
            # In sampling stage, perform SGLD update
            noise = np.random.randn(self.dim) * np.sqrt(2 * lr)
            self.theta = self.theta + lr * grad + noise


def run_sgld(mode_centers, num_iterations=50000, initial_lr=0.05, gamma=0.55):
    """
    Run SGLD on the mixture of Gaussians.
    
    Args:
        mode_centers (numpy.ndarray): centers of the modes
        num_iterations (int): number of iterations
        initial_lr (float): initial learning rate
        gamma (float): decay rate for learning rate
    
    Returns:
        numpy.ndarray: samples from SGLD
    """
    sampler = GaussianMixtureSampler(mode_centers)
    samples = np.zeros((num_iterations, 2))
    
    for i in tqdm(range(num_iterations)):
        # Decreasing stepsize schedule
        lr = initial_lr * (1 + i) ** (-gamma)
        
        # Update
        sampler.sgld_step(lr)
        
        # Save sample
        samples[i] = sampler.theta
    
    return samples


def run_csgld(mode_centers, num_iterations=50000, initial_lr=0.09, num_cycles=30, exploration_ratio=0.25):
    """
    Run cSGLD on the mixture of Gaussians.
    
    Args:
        mode_centers (numpy.ndarray): centers of the modes
        num_iterations (int): number of iterations
        initial_lr (float): initial learning rate
        num_cycles (int): number of cycles
        exploration_ratio (float): proportion of exploration stage in each cycle
    
    Returns:
        numpy.ndarray: samples from cSGLD
    """
    sampler = GaussianMixtureSampler(mode_centers)
    samples = np.zeros((num_iterations, 2))
    
    cycle_length = num_iterations // num_cycles
    
    for i in tqdm(range(num_iterations)):
        # Calculate current learning rate using cyclical schedule
        cycle_position = i % cycle_length
        cycle_ratio = cycle_position / cycle_length
        
        # Cosine annealing learning rate
        lr = (initial_lr / 2) * (np.cos(np.pi * cycle_position / cycle_length) + 1)
        
        # Determine if in exploration or sampling stage
        in_exploration = cycle_ratio < exploration_ratio
        
        # Update
        sampler.csgld_step(lr, in_exploration)
        
        # Save sample
        samples[i] = sampler.theta
    
    return samples


def run_parallel_sgld(mode_centers, num_chains=4, num_iterations=50000, initial_lr=0.05, gamma=0.55):
    """
    Run parallel SGLD on the mixture of Gaussians.
    
    Args:
        mode_centers (numpy.ndarray): centers of the modes
        num_chains (int): number of parallel chains
        num_iterations (int): number of iterations per chain
        initial_lr (float): initial learning rate
        gamma (float): decay rate for learning rate
    
    Returns:
        numpy.ndarray: samples from parallel SGLD
    """
    samples = np.zeros((num_chains * num_iterations, 2))
    
    for c in range(num_chains):
        sampler = GaussianMixtureSampler(mode_centers)
        
        for i in tqdm(range(num_iterations), desc=f"Chain {c+1}/{num_chains}"):
            # Decreasing stepsize schedule
            lr = initial_lr * (1 + i) ** (-gamma)
            
            # Update
            sampler.sgld_step(lr)
            
            # Save sample
            samples[c * num_iterations + i] = sampler.theta
    
    return samples


def run_parallel_csgld(mode_centers, num_chains=4, num_iterations=50000, initial_lr=0.09, num_cycles=30, exploration_ratio=0.25):
    """
    Run parallel cSGLD on the mixture of Gaussians.
    
    Args:
        mode_centers (numpy.ndarray): centers of the modes
        num_chains (int): number of parallel chains
        num_iterations (int): number of iterations per chain
        initial_lr (float): initial learning rate
        num_cycles (int): number of cycles
        exploration_ratio (float): proportion of exploration stage in each cycle
    
    Returns:
        numpy.ndarray: samples from parallel cSGLD
    """
    samples = np.zeros((num_chains * num_iterations, 2))
    
    for c in range(num_chains):
        sampler = GaussianMixtureSampler(mode_centers)
        cycle_length = num_iterations // num_cycles
        
        for i in tqdm(range(num_iterations), desc=f"Chain {c+1}/{num_chains}"):
            # Calculate current learning rate using cyclical schedule
            cycle_position = i % cycle_length
            cycle_ratio = cycle_position / cycle_length
            
            # Cosine annealing learning rate
            lr = (initial_lr / 2) * (np.cos(np.pi * cycle_position / cycle_length) + 1)
            
            # Determine if in exploration or sampling stage
            in_exploration = cycle_ratio < exploration_ratio
            
            # Update
            sampler.csgld_step(lr, in_exploration)
            
            # Save sample
            samples[c * num_iterations + i] = sampler.theta
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Run synthetic experiments for SGLD and cSGLD')
    parser.add_argument('--num_iterations', type=int, default=50000, help='Number of iterations')
    parser.add_argument('--num_chains', type=int, default=4, help='Number of parallel chains')
    parser.add_argument('--sgld_lr', type=float, default=0.05, help='Initial learning rate for SGLD')
    parser.add_argument('--csgld_lr', type=float, default=0.09, help='Initial learning rate for cSGLD')
    parser.add_argument('--num_cycles', type=int, default=30, help='Number of cycles for cSGLD')
    parser.add_argument('--exploration_ratio', type=float, default=0.25, help='Proportion of exploration stage in each cycle')
    parser.add_argument('--results_dir', type=str, default='../../results/synthetic', help='Directory to save results')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Generate synthetic data
    _, mode_centers = generate_synthetic_data(num_samples=1000)
    
    # Plot learning rate schedules
    plot_learning_rate_schedule('decay', args.num_iterations, save_path=os.path.join(args.results_dir, 'decay_lr_schedule.png'))
    plot_learning_rate_schedule('cyclical', args.num_iterations, save_path=os.path.join(args.results_dir, 'cyclical_lr_schedule.png'))
    
    # Run single-chain experiments
    print("Running single-chain SGLD...")
    sgld_samples = run_sgld(mode_centers, args.num_iterations, args.sgld_lr)
    
    print("Running single-chain cSGLD...")
    csgld_samples = run_csgld(mode_centers, args.num_iterations, args.csgld_lr, args.num_cycles, args.exploration_ratio)
    
    # Plot single-chain results
    plot_density_2d(sgld_samples, save_path=os.path.join(args.results_dir, 'sgld_density.png'))
    plot_density_2d(csgld_samples, save_path=os.path.join(args.results_dir, 'csgld_density.png'))
    
    # Run parallel experiments
    print("Running parallel SGLD...")
    parallel_sgld_samples = run_parallel_sgld(mode_centers, args.num_chains, args.num_iterations // args.num_chains, args.sgld_lr)
    
    print("Running parallel cSGLD...")
    parallel_csgld_samples = run_parallel_csgld(mode_centers, args.num_chains, args.num_iterations // args.num_chains, args.csgld_lr, args.num_cycles // args.num_chains, args.exploration_ratio)
    
    # Plot parallel results
    plot_density_2d(parallel_sgld_samples, save_path=os.path.join(args.results_dir, 'parallel_sgld_density.png'))
    plot_density_2d(parallel_csgld_samples, save_path=os.path.join(args.results_dir, 'parallel_csgld_density.png'))
    
    # Calculate mode coverage
    radius = 0.25
    threshold = 100
    
    sgld_coverage = mode_coverage(sgld_samples, mode_centers, radius, threshold)
    csgld_coverage = mode_coverage(csgld_samples, mode_centers, radius, threshold)
    parallel_sgld_coverage = mode_coverage(parallel_sgld_samples, mode_centers, radius, threshold)
    parallel_csgld_coverage = mode_coverage(parallel_csgld_samples, mode_centers, radius, threshold)
    
    print(f"Mode coverage (out of {len(mode_centers)}):")
    print(f"SGLD: {sgld_coverage}")
    print(f"cSGLD: {csgld_coverage}")
    print(f"Parallel SGLD: {parallel_sgld_coverage}")
    print(f"Parallel cSGLD: {parallel_csgld_coverage}")
    
    # Save results
    results = {
        'sgld_coverage': sgld_coverage,
        'csgld_coverage': csgld_coverage,
        'parallel_sgld_coverage': parallel_sgld_coverage,
        'parallel_csgld_coverage': parallel_csgld_coverage
    }
    
    np.save(os.path.join(args.results_dir, 'mode_coverage.npy'), results)


if __name__ == '__main__':
    main()

