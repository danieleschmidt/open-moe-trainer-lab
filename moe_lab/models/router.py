"""Router implementations for MoE models."""

from typing import Tuple, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class RoutingInfo:
    """Information about routing decisions."""
    
    expert_weights: torch.Tensor  # Weights assigned to each expert
    selected_experts: torch.Tensor  # Which experts were selected
    router_logits: torch.Tensor  # Raw router outputs
    load_variance: float  # Variance in expert load
    entropy: float  # Routing entropy
    
    
class TopKRouter(nn.Module):
    """Top-K router that selects top K experts per token."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        normalize_weights: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.normalize_weights = normalize_weights
        
        # Router network - simple linear layer
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize router weights
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Forward pass through router."""
        batch_size, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch_size, num_experts]
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Convert to probabilities
        if self.normalize_weights:
            # Softmax over top-k experts only
            expert_weights = F.softmax(top_k_logits, dim=-1)
        else:
            # Use raw logits as weights
            expert_weights = top_k_logits
            
        # Compute routing statistics
        probs = F.softmax(router_logits, dim=-1)
        
        # Load variance - how evenly distributed are tokens across experts
        expert_load = probs.mean(dim=0)  # Average probability per expert
        load_variance = expert_load.var().item()
        
        # Routing entropy - higher means more diverse routing
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_indices,
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, top_k_indices, expert_weights, routing_info


class ExpertChoice(nn.Module):
    """Expert-choice router where tokens compete for expert capacity."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Forward pass with expert choice routing."""
        batch_size, hidden_size = hidden_states.shape
        
        # Get router logits and probabilities
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Calculate expert capacity
        tokens_per_expert = int(batch_size * self.capacity_factor / self.num_experts)
        
        # Expert choice: each expert selects its top tokens
        selected_experts = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        expert_weights = torch.zeros(batch_size, device=hidden_states.device)
        
        for expert_idx in range(self.num_experts):
            # Get this expert's affinity for all tokens
            expert_affinities = router_probs[:, expert_idx]
            
            # Select top tokens for this expert
            top_token_values, top_token_indices = torch.topk(
                expert_affinities, 
                min(tokens_per_expert, batch_size), 
                dim=0
            )
            
            # Assign tokens to this expert
            selected_experts[top_token_indices] = expert_idx
            expert_weights[top_token_indices] = top_token_values
            
        # Compute statistics
        expert_load = torch.bincount(selected_experts, minlength=self.num_experts).float()
        expert_load = expert_load / expert_load.sum()
        load_variance = expert_load.var().item()
        
        # Entropy calculation
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights.unsqueeze(-1),  # Match expected shape
            selected_experts=selected_experts.unsqueeze(-1),  # Match expected shape
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, selected_experts.unsqueeze(-1), expert_weights.unsqueeze(-1), routing_info


class SwitchRouter(nn.Module):
    """Switch transformer router - routes each token to single expert."""
    
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
        
        # Initialize weights  
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Forward pass with switch routing."""
        batch_size, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)
        
        # Add jitter noise
        if self.training and self.jitter_noise > 0:
            noise = torch.rand_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        # Convert to probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select single expert per token (argmax)
        selected_experts = torch.argmax(router_probs, dim=-1)
        expert_weights = torch.max(router_probs, dim=-1)[0]
        
        # Calculate expert capacity
        capacity = int(batch_size * self.capacity_factor / self.num_experts)
        
        # Check capacity constraints
        expert_counts = torch.bincount(selected_experts, minlength=self.num_experts)
        
        if self.drop_tokens:
            # Drop tokens that exceed capacity
            for expert_idx in range(self.num_experts):
                expert_mask = (selected_experts == expert_idx)
                expert_token_count = expert_mask.sum()
                
                if expert_token_count > capacity:
                    # Keep only top-capacity tokens for this expert
                    expert_token_indices = torch.where(expert_mask)[0]
                    expert_token_weights = expert_weights[expert_mask]
                    
                    # Sort by weight and keep top tokens
                    sorted_indices = torch.argsort(expert_token_weights, descending=True)
                    keep_indices = expert_token_indices[sorted_indices[:capacity]]
                    drop_indices = expert_token_indices[sorted_indices[capacity:]]
                    
                    # Set dropped tokens to -1 (no expert)
                    selected_experts[drop_indices] = -1
                    expert_weights[drop_indices] = 0.0
        
        # Compute statistics
        valid_assignments = (selected_experts >= 0)
        if valid_assignments.any():
            valid_expert_counts = torch.bincount(
                selected_experts[valid_assignments], 
                minlength=self.num_experts
            ).float()
            expert_load = valid_expert_counts / valid_expert_counts.sum()
            load_variance = expert_load.var().item()
        else:
            load_variance = 0.0
            
        # Entropy
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights.unsqueeze(-1),
            selected_experts=selected_experts.unsqueeze(-1),
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, selected_experts.unsqueeze(-1), expert_weights.unsqueeze(-1), routing_info

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

