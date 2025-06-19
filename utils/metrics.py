import torch
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(output, target):
    """
    Compute the accuracy.
    
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): ground truth
    
    Returns:
        float: accuracy
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

def negative_log_likelihood(output, target):
    """
    Compute the negative log-likelihood.
    
    Args:
        output (torch.Tensor): model output (log probabilities)
        target (torch.Tensor): ground truth
    
    Returns:
        float: negative log-likelihood
    """
    return torch.nn.functional.nll_loss(output, target).item()

def brier_score(output, target, num_classes):
    """
    Compute the Brier score.
    
    Args:
        output (torch.Tensor): model output (probabilities)
        target (torch.Tensor): ground truth
        num_classes (int): number of classes
    
    Returns:
        float: Brier score
    """
    target_one_hot = torch.zeros_like(output)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    return torch.mean(torch.sum((output - target_one_hot) ** 2, dim=1)).item()

def expected_calibration_error(output, target, num_bins=15):
    """
    Compute the Expected Calibration Error (ECE).
    
    Args:
        output (torch.Tensor): model output (probabilities)
        target (torch.Tensor): ground truth
        num_bins (int): number of bins for confidence histogram
    
    Returns:
        float: Expected Calibration Error
    """
    pred = output.argmax(dim=1, keepdim=True)
    confidence = output.gather(1, pred).view(-1).cpu().numpy()
    pred = pred.view(-1).cpu().numpy()
    target = target.cpu().numpy()
    
    acc = np.zeros(num_bins)
    conf = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_mask = np.logical_and(confidence > bin_edges[i], confidence <= bin_edges[i+1])
        if np.sum(bin_mask) > 0:
            acc[i] = np.mean(pred[bin_mask] == target[bin_mask])
            conf[i] = np.mean(confidence[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    ece = np.sum(bin_counts * np.abs(acc - conf)) / np.sum(bin_counts)
    return ece

def mode_coverage(samples, mode_centers, radius, threshold):
    """
    Compute the mode coverage metric for synthetic experiments.
    
    Args:
        samples (numpy.ndarray): samples from the distribution
        mode_centers (numpy.ndarray): centers of the modes
        radius (float): radius around each mode center
        threshold (int): minimum number of samples to consider a mode covered
    
    Returns:
        int: number of modes covered
    """
    covered_modes = 0
    for center in mode_centers:
        distances = np.sqrt(np.sum((samples - center) ** 2, axis=1))
        samples_in_mode = np.sum(distances < radius)
        if samples_in_mode >= threshold:
            covered_modes += 1
    
    return covered_modes

