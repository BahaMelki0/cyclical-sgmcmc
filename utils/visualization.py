import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import MDS
import seaborn as sns
from matplotlib.colors import LogNorm

def plot_density_2d(samples, bins=100, range_min=-6, range_max=6, save_path=None):
    """
    Plot the density of 2D samples.
    
    Args:
        samples (numpy.ndarray): 2D samples from the distribution
        bins (int): number of bins for the histogram
        range_min (float): minimum range for both axes
        range_max (float): maximum range for both axes
        save_path (str, optional): path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1], 
        bins=bins, 
        range=[[range_min, range_max], [range_min, range_max]]
    )
    
    # Normalize the histogram
    hist = hist / hist.sum()
    
    # Plot the histogram using pcolormesh
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    plt.pcolormesh(X, Y, hist.T, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Normalized Density')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density Estimation')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_learning_rate_schedule(schedule, num_iterations, save_path=None):
    """
    Plot the learning rate schedule.
    
    Args:
        schedule (str): type of schedule ('cyclical' or 'decay')
        num_iterations (int): number of iterations
        save_path (str, optional): path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    iterations = np.arange(num_iterations)
    
    if schedule == 'cyclical':
        # Cyclical learning rate schedule
        num_cycles = 4
        cycle_length = num_iterations // num_cycles
        init_lr = 0.1
        
        lr_values = []
        for i in iterations:
            cycle_position = i % cycle_length
            lr = (init_lr / 2) * (np.cos(np.pi * cycle_position / cycle_length) + 1)
            lr_values.append(lr)
        
        plt.plot(iterations, lr_values, 'r-', label='Cyclical Schedule')
        
        # Mark exploration and sampling stages
        exploration_ratio = 0.8
        for i in range(num_cycles):
            plt.axvspan(i * cycle_length, i * cycle_length + exploration_ratio * cycle_length, 
                       alpha=0.2, color='blue', label='Exploration Stage' if i == 0 else None)
            plt.axvspan(i * cycle_length + exploration_ratio * cycle_length, (i + 1) * cycle_length, 
                       alpha=0.2, color='green', label='Sampling Stage' if i == 0 else None)
    
    elif schedule == 'decay':
        # Decreasing learning rate schedule
        init_lr = 0.1
        gamma = 0.55
        a = init_lr
        b = 1
        
        lr_values = [a * (b + i) ** (-gamma) for i in iterations]
        plt.plot(iterations, lr_values, 'b-', label='Decreasing Schedule')
    
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_mds_visualization(weights_list, labels=None, save_path=None):
    """
    Plot MDS visualization of weights in 2D space.
    
    Args:
        weights_list (list): list of flattened weight vectors
        labels (list, optional): labels for each weight vector
        save_path (str, optional): path to save the figure
    """
    # Convert weights to numpy arrays
    weights_np = np.array([w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in weights_list])
    
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
    weights_2d = mds.fit_transform(weights_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(weights_2d[mask, 0], weights_2d[mask, 1], label=f'Cluster {label}')
    else:
        plt.scatter(weights_2d[:, 0], weights_2d[:, 1])
    
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title('MDS Visualization of Weight Space')
    
    if labels is not None:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_interpolation(errors, save_path=None):
    """
    Plot test errors along the interpolation path between two models.
    
    Args:
        errors (list): list of test errors along the interpolation path
        save_path (str, optional): path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    alphas = np.linspace(0, 1, len(errors))
    plt.plot(alphas, errors, 'o-')
    
    plt.xlabel('Interpolation Parameter (α)')
    plt.ylabel('Test Error (%)')
    plt.title('Test Error Along Interpolation Path')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_uncertainty_histogram(confidences, correct, save_path=None):
    """
    Plot histogram of prediction confidences for correct and incorrect predictions.
    
    Args:
        confidences (numpy.ndarray): confidence values
        correct (numpy.ndarray): boolean array indicating correct predictions
        save_path (str, optional): path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(confidences[correct], bins=20, alpha=0.5, label='Correct Predictions')
    plt.hist(confidences[~correct], bins=20, alpha=0.5, label='Incorrect Predictions')
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Confidences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

