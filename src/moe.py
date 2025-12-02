import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    """
    Mixture of Experts layer with Top-1 routing.
    """

    def __init__(self, d_model, d_hidden, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

        # Gate mechanism
        self.gate = nn.Linear(d_model, num_experts)

        # Expert MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_model)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (B, T, d_model)
        """

        B, T, D = x.shape

        # Compute gating weights
        gate_logits = self.gate(x) # (B, T, E)
        gate_scores = F.softmax(gate_logits, dim=-1) # (B, T, E)

        # Top-1 expert routing
        top1_idx = torch.argmax(gate_scores, dim=-1) # (B, T, 1)
        top1_mask = F.one_hot(top1_idx, num_classes=self.num_experts).float()
        top1_weight = (gate_scores * top1_mask).sum(-1, keepdim=True) # (B, T, 1)

        outputs = []
        for expert_id, expert in enumerate(self.experts):
            mask = top1_mask[..., expert_id].unsqueeze(-1)
            x_e = x * mask
            out_e = expert(x_e) * mask
            outputs.append(out_e)

        # Combine outputs weighted by gate prob
        # sum(outputs) sum the outputs from each expert
        # here each token is only processed by one expert
        y = sum(outputs) * top1_weight

        return y