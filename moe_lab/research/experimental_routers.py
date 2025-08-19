"""Novel experimental router algorithms for MoE research."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..models.router import RoutingInfo


class AdaptiveRouter(nn.Module):
    """Router that adapts top-k dynamically based on input complexity."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        min_k: int = 1,
        max_k: int = 4,
        complexity_threshold: float = 0.5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k
        self.complexity_threshold = complexity_threshold
        
        # Main router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.router.weight, std=0.02)
        for layer in self.complexity_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        batch_size, hidden_size = hidden_states.shape
        
        # Predict input complexity
        complexity_scores = self.complexity_predictor(hidden_states).squeeze(-1)
        
        # Get router logits
        router_logits = self.router(hidden_states)
        
        # Determine adaptive top-k for each token
        adaptive_k = torch.where(
            complexity_scores > self.complexity_threshold,
            torch.full_like(complexity_scores, self.max_k),
            torch.full_like(complexity_scores, self.min_k)
        ).long()
        
        # Route with adaptive k
        selected_experts = torch.zeros(batch_size, self.max_k, dtype=torch.long, device=hidden_states.device)
        expert_weights = torch.zeros(batch_size, self.max_k, device=hidden_states.device)
        
        for i in range(batch_size):
            k = adaptive_k[i].item()
            top_k_logits, top_k_indices = torch.topk(router_logits[i], k)
            selected_experts[i, :k] = top_k_indices
            expert_weights[i, :k] = F.softmax(top_k_logits, dim=0)
        
        # Compute routing statistics
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


class HierarchicalRouter(nn.Module):
    """Two-level hierarchical routing: first route to expert groups, then to experts."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_groups: int = 4,
        experts_per_group: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_groups = num_groups
        
        if experts_per_group is None:
            self.experts_per_group = num_experts // num_groups
        else:
            self.experts_per_group = experts_per_group
            
        # Group-level router
        self.group_router = nn.Linear(hidden_size, num_groups, bias=False)
        
        # Expert-level routers (one per group)
        self.expert_routers = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.group_router.weight, std=0.02)
        for router in self.expert_routers:
            nn.init.normal_(router.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        batch_size, hidden_size = hidden_states.shape
        
        # First level: route to groups
        group_logits = self.group_router(hidden_states)
        group_probs = F.softmax(group_logits, dim=-1)
        selected_groups = torch.argmax(group_probs, dim=-1)
        
        # Second level: route to experts within selected groups
        selected_experts = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        expert_weights = torch.zeros(batch_size, device=hidden_states.device)
        all_expert_logits = torch.zeros(batch_size, self.num_experts, device=hidden_states.device)
        
        for i in range(batch_size):
            group_idx = selected_groups[i].item()
            expert_logits = self.expert_routers[group_idx](hidden_states[i:i+1])
            expert_probs = F.softmax(expert_logits, dim=-1)
            
            # Select best expert in group
            local_expert_idx = torch.argmax(expert_probs, dim=-1).item()
            global_expert_idx = group_idx * self.experts_per_group + local_expert_idx
            
            selected_experts[i] = global_expert_idx
            expert_weights[i] = expert_probs[0, local_expert_idx]
            
            # Store logits for statistics
            start_idx = group_idx * self.experts_per_group
            end_idx = start_idx + self.experts_per_group
            all_expert_logits[i, start_idx:end_idx] = expert_logits[0]
        
        # Compute statistics
        probs = F.softmax(all_expert_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights.unsqueeze(-1),
            selected_experts=selected_experts.unsqueeze(-1),
            router_logits=all_expert_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return all_expert_logits, selected_experts.unsqueeze(-1), expert_weights.unsqueeze(-1), routing_info


class LearnedSparseRouter(nn.Module):
    """Router with learned sparsity patterns using learnable masking."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        sparsity_level: float = 0.8,
        temperature: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.sparsity_level = sparsity_level
        self.temperature = temperature
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Learnable sparsity mask
        self.sparsity_mask = nn.Parameter(torch.randn(num_experts, hidden_size))
        
        # Gating network for sparsity
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.sparsity_mask, std=0.02)
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        batch_size, hidden_size = hidden_states.shape
        
        # Compute gates for sparsity
        gates = self.gate_network(hidden_states)
        
        # Apply learnable sparsity mask
        masked_weights = self.router.weight * torch.sigmoid(self.sparsity_mask)
        
        # Compute router logits with sparsity
        router_logits = F.linear(hidden_states, masked_weights)
        
        # Apply gating
        gated_logits = router_logits * gates
        
        # Temperature scaling
        scaled_logits = gated_logits / self.temperature
        
        # Select experts based on learned sparsity
        k = max(1, int(self.num_experts * (1 - self.sparsity_level)))
        top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute statistics
        probs = F.softmax(scaled_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_indices,
            router_logits=scaled_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return scaled_logits, top_k_indices, expert_weights, routing_info


class DynamicTopKRouter(nn.Module):
    """Router that learns optimal top-k values during training."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        max_k: int = 8,
        k_learning_rate: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.max_k = max_k
        self.k_learning_rate = k_learning_rate
        
        # Main router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Learnable k parameter (sigmoid to ensure 0 < k < max_k)
        self.k_logit = nn.Parameter(torch.tensor(0.0))
        
        # K predictor network
        self.k_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.router.weight, std=0.02)
        for layer in self.k_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def get_current_k(self, hidden_states: Optional[torch.Tensor] = None) -> int:
        """Get current top-k value."""
        if hidden_states is not None:
            # Use predictor network for token-specific k
            k_scores = self.k_predictor(hidden_states).mean()
            k = int(1 + k_scores * (self.max_k - 1))
        else:
            # Use global learnable k
            k = int(1 + torch.sigmoid(self.k_logit) * (self.max_k - 1))
        return max(1, min(k, self.max_k))
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        batch_size, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)
        
        # Determine dynamic k
        k = self.get_current_k(hidden_states)
        
        # Route with dynamic k
        top_k_logits, top_k_indices = torch.topk(router_logits, k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute statistics
        probs = F.softmax(router_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_indices,
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, top_k_indices, expert_weights, routing_info


class ContextAwareRouter(nn.Module):
    """Router that considers broader context for routing decisions."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        context_window: int = 64,
        top_k: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.context_window = context_window
        self.top_k = top_k
        
        # Context aggregation
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Context-aware router
        self.router = nn.Linear(hidden_size * 2, num_experts, bias=False)  # Concatenated features
        
        # Context encoding
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.router.weight, std=0.02)
        for layer in self.context_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        batch_size, hidden_size = hidden_states.shape
        
        # For context-aware routing, we need to process sequences
        # For now, treat each token independently but add context encoding
        
        # Encode context (simplified - in practice would use sequence context)
        context_features = self.context_encoder(hidden_states)
        
        # Self-attention for context aggregation
        attended_features, _ = self.context_attention(
            hidden_states.unsqueeze(1),  # Add sequence dimension
            hidden_states.unsqueeze(1),
            hidden_states.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Combine original and context features
        combined_features = torch.cat([hidden_states, attended_features], dim=-1)
        
        # Route based on combined features
        router_logits = self.router(combined_features)
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute statistics
        probs = F.softmax(router_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_indices,
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, top_k_indices, expert_weights, routing_info