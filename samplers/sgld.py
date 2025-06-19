import torch
from torch.optim import Optimizer
import math

class SGLD(Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) optimizer.
    
    Implementation of SGLD from the paper:
    "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
    by Welling and Teh, 2011.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (stepsize)
        num_data (int): total number of data points in the dataset
        temperature (float, optional): temperature parameter (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr, num_data, temperature=1.0, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            num_data=num_data,
            temperature=temperature,
            weight_decay=weight_decay
        )
        super(SGLD, self).__init__(params, defaults)
    
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
            temperature = group['temperature']
            lr = group['lr']
            num_data = group['num_data']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # Scale gradient by number of data points (for minibatch)
                d_p.mul_(num_data)
                
                # Add noise
                if temperature > 0:
                    noise = torch.randn_like(p.data) * math.sqrt(2.0 * lr * temperature)
                    p.data.add_(-lr, d_p).add_(noise)
                else:
                    # If temperature is 0, perform SGD update (no noise)
                    p.data.add_(-lr, d_p)
        
        return loss
    
    def set_lr(self, lr):
        """
        Set the learning rate (stepsize).
        
        Args:
            lr (float): new learning rate
        """
        for group in self.param_groups:
            group['lr'] = lr

