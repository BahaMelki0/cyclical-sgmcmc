import torch
from torch.optim import Optimizer
import math

class SGHMC(Optimizer):
    """
    Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) optimizer.
    
    Implementation of SGHMC from the paper:
    "Stochastic Gradient Hamiltonian Monte Carlo"
    by Chen et al., 2014.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (stepsize)
        num_data (int): total number of data points in the dataset
        momentum (float, optional): momentum factor (default: 0.9)
        temperature (float, optional): temperature parameter (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        noise_scale (float, optional): scale of the noise estimate (default: 0)
    """
    def __init__(self, params, lr, num_data, momentum=0.9, temperature=1.0, weight_decay=0, noise_scale=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            num_data=num_data,
            momentum=momentum,
            temperature=temperature,
            weight_decay=weight_decay,
            noise_scale=noise_scale
        )
        super(SGHMC, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            temperature = group['temperature']
            lr = group['lr']
            num_data = group['num_data']
            noise_scale = group['noise_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # Scale gradient by number of data points (for minibatch)
                d_p.mul_(num_data)
                
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_buffer']
                
                # Update momentum
                buf.mul_(momentum).add_(-lr, d_p)
                
                # Add noise
                if temperature > 0:
                    noise = torch.randn_like(p.data) * math.sqrt(2.0 * lr * temperature * (1.0 - momentum) - 2.0 * lr * lr * noise_scale)
                    buf.add_(noise)
                
                # Update parameters
                p.data.add_(buf)
        
        return loss
    
    def set_lr(self, lr):
        """
        Set the learning rate (stepsize).
        
        Args:
            lr (float): new learning rate
        """
        for group in self.param_groups:
            group['lr'] = lr

