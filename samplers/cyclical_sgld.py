import torch
from torch.optim import Optimizer
import math
import numpy as np

class CyclicalSGLD(Optimizer):
    """
    Cyclical Stochastic Gradient Langevin Dynamics (cSGLD) optimizer.
    
    Implementation of cSGLD from the paper:
    "Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning"
    by Zhang et al., 2020.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        init_lr (float): initial learning rate (stepsize)
        num_data (int): total number of data points in the dataset
        num_cycles (int): number of cycles
        total_iterations (int): total number of iterations
        exploration_ratio (float): proportion of exploration stage in each cycle (default: 0.8)
        temperature (float, optional): temperature parameter (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, init_lr, num_data, num_cycles, total_iterations, exploration_ratio=0.8, 
                 temperature=1.0, weight_decay=0):
        if init_lr < 0.0:
            raise ValueError(f"Invalid learning rate: {init_lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if num_cycles <= 0:
            raise ValueError(f"Invalid number of cycles: {num_cycles}")
        if total_iterations <= 0:
            raise ValueError(f"Invalid total iterations: {total_iterations}")
        if exploration_ratio < 0.0 or exploration_ratio > 1.0:
            raise ValueError(f"Invalid exploration ratio: {exploration_ratio}")
        
        defaults = dict(
            init_lr=init_lr,
            num_data=num_data,
            num_cycles=num_cycles,
            total_iterations=total_iterations,
            exploration_ratio=exploration_ratio,
            temperature=temperature,
            weight_decay=weight_decay,
            iteration=0
        )
        super(CyclicalSGLD, self).__init__(params, defaults)
    
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
            init_lr = group['init_lr']
            num_data = group['num_data']
            num_cycles = group['num_cycles']
            total_iterations = group['total_iterations']
            exploration_ratio = group['exploration_ratio']
            iteration = group['iteration']
            
            # Calculate current learning rate using cyclical schedule
            cycle_length = math.ceil(total_iterations / num_cycles)
            cycle_position = iteration % cycle_length
            cycle_ratio = cycle_position / cycle_length
            
            # Cosine annealing learning rate
            lr = (init_lr / 2) * (math.cos(math.pi * cycle_position / cycle_length) + 1)
            
            # Determine if in exploration or sampling stage
            in_exploration = cycle_ratio < exploration_ratio
            
            # Set temperature based on stage
            current_temp = 0.0 if in_exploration else temperature
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # Scale gradient by number of data points (for minibatch)
                d_p.mul_(num_data)
                
                # Add noise only in sampling stage
                if not in_exploration and current_temp > 0:
                    noise = torch.randn_like(p.data) * math.sqrt(2.0 * lr * current_temp)
                    p.data.add_(-lr, d_p).add_(noise)
                else:
                    # In exploration stage, perform SGD update (no noise)
                    p.data.add_(-lr, d_p)
            
            # Increment iteration counter
            group['iteration'] = iteration + 1
        
        return loss
    
    def get_lr(self):
        """
        Get the current learning rate.
        
        Returns:
            float: current learning rate
        """
        for group in self.param_groups:
            init_lr = group['init_lr']
            num_cycles = group['num_cycles']
            total_iterations = group['total_iterations']
            iteration = group['iteration']
            
            cycle_length = math.ceil(total_iterations / num_cycles)
            cycle_position = iteration % cycle_length
            
            # Cosine annealing learning rate
            lr = (init_lr / 2) * (math.cos(math.pi * cycle_position / cycle_length) + 1)
            return lr
        
        return 0.0
    
    def is_sampling(self):
        """
        Check if the sampler is in the sampling stage.
        
        Returns:
            bool: True if in sampling stage, False if in exploration stage
        """
        for group in self.param_groups:
            num_cycles = group['num_cycles']
            total_iterations = group['total_iterations']
            exploration_ratio = group['exploration_ratio']
            iteration = group['iteration']
            
            cycle_length = math.ceil(total_iterations / num_cycles)
            cycle_position = iteration % cycle_length
            cycle_ratio = cycle_position / cycle_length
            
            return cycle_ratio >= exploration_ratio
        
        return False

