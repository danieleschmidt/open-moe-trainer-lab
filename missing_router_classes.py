"""Missing router classes for architectures.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SwitchRouter(nn.Module):
    """Switch Transformer router implementation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        jitter_noise: float = 0.1,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.jitter_noise = jitter_noise
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(self, hidden_states: torch.Tensor):
        """Forward pass with Switch routing logic."""
        batch_size, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)
        
        # Add jitter noise
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        # Get top-1 expert (Switch uses single expert per token)
        expert_weights, selected_experts = torch.max(F.softmax(router_logits, dim=-1), dim=-1)
        selected_experts = selected_experts.unsqueeze(-1)
        expert_weights = expert_weights.unsqueeze(-1)
        
        # Create routing info
        from .router import RoutingInfo
        probs = F.softmax(router_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=selected_experts,
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, selected_experts, expert_weights, routing_info