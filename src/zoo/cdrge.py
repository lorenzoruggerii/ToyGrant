"""Implement the Central-Difference Random-vector Gradient Estimation for RNN"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from typing import Callable
from torch.utils.data import DataLoader
from torch.func import functional_call, vmap

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_perturbations(params: torch.Tensor, seed: int, scale: float = 1.0):
    """Generate perturbations {+1, -1}^d where d is the number of parameters"""

    # Define the seed
    g = torch.Generator(device=params.device)
    g.manual_seed(seed)

    shape = params.shape

    perturb = torch.randint(0, 2, shape, device=params.device, generator=g, dtype=torch.float32) * 2 - 1
    
    return perturb * scale

def calculate_loss(batch, loss_fn: Callable, model: nn.Module, device: str):

    inputs, labels = batch['input_ids'], batch['signal']
        
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds, _ = model(inputs)
    loss = loss_fn(preds, labels)

    return loss.item()

def cdrge_minimal(
        model: nn.Module,
        batch,
        loss_calculator: Callable,
        lr: float,
        epsilon: float,
        num_perturbations: int = 512,
    ):

    """
    Approximate the gradient as 1 / 2 * npert * \sum(L(theta + eps * p) - L(theta - eps * p) * p)
    
    """

    import random

    all_seeds = [random.randint(0, 2**32 - 1) for _ in range(num_perturbations)]

    grad_buffers = [torch.zeros_like(p.data) for p in model.parameters()]

    # Take input ids and labels
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    loss = 0.0

    for seed in all_seeds:
        probes = []

        # Generate the probes and apply
        for idx, param in enumerate(model.parameters()):
            probe = generate_perturbations(param, seed=seed+idx)
            param.data.add_(epsilon * probe)
            probes.append(probe)

        # Compute positive loss
        with torch.no_grad():
            preds, _ = model(input_ids)
            pos_loss = loss_calculator(preds, labels)

        # Apply -2eps probes
        for idx, param in enumerate(model.parameters()):
            param.data.add_(-2 * epsilon * probes[idx])

        # Compute pos loss
        with torch.no_grad():
            preds, _ = model(input_ids)
            neg_loss = loss_calculator(preds, labels)

        # restore original parameters
        for idx, param in enumerate(model.parameters()):
            param.data.add_(epsilon * probes[idx])

        # Calculate gradient
        grad_coeff = (pos_loss.item() - neg_loss.item()) / (2.0 * num_perturbations)
        for buf, probe in zip(grad_buffers, probes):
            buf.add_(probe, alpha=grad_coeff)


        # Define loss as the mid point
        loss += (pos_loss + neg_loss) / 2.0

        # free cached memory
        del preds, pos_loss, neg_loss, probes
        torch.cuda.empty_cache()

    # Now in buffer we have the sum of the losses, let's update
    with torch.no_grad():
        for param, buf in zip(model.parameters(), grad_buffers):
            param.data.add_(buf, alpha=-lr)
        
    
    return loss / num_perturbations

def cdrge_parallel(
        loss_fn: Callable,
        model: nn.Module,
        batch,
        lr: float,
        epsilon: float,
        num_perturbations: int = 512,
    ):

    def gen_noise(p):
        return torch.randint(0, 2, size=p.shape) * 2 - 1
    
    # Extract inputs and labels and push to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    noises = [
        {k: gen_noise(v) for k, v in model.named_parameters()}
        for _ in range(num_perturbations)
    ]

    # Stack noises onto the batch dimension
    stacked_noises = {
        k: torch.stack([n[k] for n in noises], dim=0).to(device) for k, _ in model.named_parameters()
    }

    # Create positive perturbations
    stacked_params_pos = {
        k: v.unsqueeze(0) + epsilon * stacked_noises[k] 
        for k, v in model.named_parameters()
    }

    stacked_params_neg = {
        k: v.unsqueeze(0) - epsilon * stacked_noises[k]
        for k, v in model.named_parameters()
    }

    # Define the loss function that works on a single set of parameters
    def loss_for_single_perturbation(pos_params, neg_params):
        pos_forw = functional_call(model, pos_params, input_ids) # (num_pert, out)
        pos_loss = loss_fn(pos_forw, labels) # (num_pert)

        neg_forw = functional_call(model, neg_params, input_ids) # (mum_pert, out)
        neg_loss = loss_fn(neg_forw, labels) # (num_pert)

        return pos_loss, neg_loss
    
    # Use vmap to parallelize across the batch dimension (dim=0)
    batched_loss = vmap(loss_for_single_perturbation, in_dims=0)

    # Execute in parallel
    pos_losses, neg_losses = batched_loss(stacked_params_pos, stacked_params_neg)

    loss_diff = (pos_losses - neg_losses) / (2.0) # here is the factor of 2, no epsilon because tied with lr

    # Calculate gradients times probes to get gradients per perturb
    gradxprobes = [
        {k: n * loss_diff[i] for k, n in noises[i].items()}
        for i in range(num_perturbations)    
    ]

    # Apply mean to get approx gradients (same as sum and dividing by num_perturbations)
    grads = {
        k: torch.stack([n[k] for n in gradxprobes]).mean(dim=0) for k, _ in model.named_parameters()
    }

    # Update parameters
    for name, param in model.named_parameters():
        old_weights = param.data.clone()
        param.data.add_(grads[name], alpha=-lr)
        print(f"Diff for {name}: {(param.data - old_weights).abs().mean()}")

    # Return averaged smoothed loss
    avg_loss = ((pos_losses + neg_losses) / 2.0).mean()

    return avg_loss
