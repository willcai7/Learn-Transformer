from collections.abc import Callable, Iterable
from typing import Optional 
import torch 
import math  

class AdamW(torch.optim.Optimizer): 
    def __init__(self, 
        params, lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}") 
        defaults = {"lr": lr, "betas":betas, "eps":eps, "decay":weight_decay} 
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups:  
            lr = group["lr"] # Get the learning rate. 
            beta1, beta2 = group["betas"] # Get the beta values.
            eps = group["eps"] # Get the epsilon value.
            weight_decay = group["decay"] # Get the weight decay value.
            for p in group["params"]: 
                if p.grad is None: 
                    continue 
                state = self.state[p] # Get state associated with p.
                
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                m = state.get("m", torch.zeros_like(p)) # Get the first moment estimate.
                v = state.get("v", torch.zeros_like(p)) # Get the second moment estimate.
                grad = p.grad.data # Get the gradient of loss with respect to p.  

                m = beta1 * m + (1 - beta1) * grad # Update biased first moment estimate.
                v = beta2 * v + (1 - beta2) * grad**2 # Update biased second raw moment estimate.
                lr_t = lr * math.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1)) # Compute the effective learning rate.
                p.data -= lr_t * (m / (torch.sqrt(v) + eps) ) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data # Apply weight decay.
                
                state["t"] = t + 1 # Increment iteration number. 
                state["m"] = m # Update first moment estimate.
                state["v"] = v # Update second moment estimate.
        return loss

def lr_cosine_schedule(t, lr_max, lr_min, t_w, t_c):
    if t < t_w:
        return lr_max * t/t_w
    else:
        if t <= t_c:
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w)))
        else:
            return lr_min

def gradient_clipping(parameters, max_norm, eps=1e-6):
    total_norm2 = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm2 += torch.sum(p.grad ** 2)
    total_norm = total_norm2 ** 0.5
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.mul_(clip_coef)