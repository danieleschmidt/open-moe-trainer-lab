"""Adaptive routing optimization for MoE models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from ..models.router import RoutingInfo
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdaptiveRoutingConfig:
    """Configuration for adaptive routing."""
    
    adaptation_rate: float = 0.01
    temperature_schedule: str = "cosine"  # "cosine", "linear", "exponential"
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    load_balance_target: float = 0.1  # Target load variance
    entropy_target: float = 0.8  # Target normalized entropy
    adaptation_window: int = 100  # Steps to average over


class AdaptiveRouter(nn.Module):
    """Router with adaptive temperature and load balancing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        config: Optional[AdaptiveRoutingConfig] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.config = config or AdaptiveRoutingConfig()
        
        # Core router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Adaptive parameters
        self.register_buffer('temperature', torch.tensor(1.0))
        self.register_buffer('step_count', torch.tensor(0))
        
        # Statistics tracking
        self.register_buffer('load_history', torch.zeros(self.config.adaptation_window))
        self.register_buffer('entropy_history', torch.zeros(self.config.adaptation_window))
        self.register_buffer('history_ptr', torch.tensor(0))
        
        # Expert importance weights (learned)
        self.expert_importance = nn.Parameter(torch.ones(num_experts))
        
        # Initialize weights
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Forward pass with adaptive routing."""
        batch_size, hidden_size = hidden_states.shape
        
        # Compute base router logits
        router_logits = self.router(hidden_states)
        
        # Apply expert importance weights
        importance_weights = F.softmax(self.expert_importance, dim=0)
        router_logits = router_logits + torch.log(importance_weights + 1e-8)
        
        # Apply adaptive temperature
        router_logits = router_logits / self.temperature
        
        # Add noise during training for exploration
        if self.training:
            noise_scale = max(0.01, 0.1 * (1.0 - self.step_count.float() / 10000))
            noise = torch.randn_like(router_logits) * noise_scale
            router_logits = router_logits + noise
            
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute routing statistics
        probs = F.softmax(router_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        normalized_entropy = entropy / math.log(self.num_experts)
        
        # Update statistics and adapt
        if self.training:
            self._update_statistics(load_variance, normalized_entropy)
            self._adapt_parameters()
            
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_indices,
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, top_k_indices, expert_weights, routing_info
        
    def _update_statistics(self, load_variance: float, entropy: float):
        """Update running statistics."""
        ptr = self.history_ptr.item() % self.config.adaptation_window
        self.load_history[ptr] = load_variance
        self.entropy_history[ptr] = entropy
        self.history_ptr += 1
        
    def _adapt_parameters(self):
        """Adapt routing parameters based on statistics."""
        if self.step_count < self.config.adaptation_window:
            self.step_count += 1
            return
            
        # Get current statistics
        current_load_var = self.load_history.mean().item()
        current_entropy = self.entropy_history.mean().item()
        
        # Adapt temperature based on load balance and entropy
        target_temp = 1.0
        
        # Increase temperature if load is too imbalanced
        if current_load_var > self.config.load_balance_target:
            target_temp *= 1.1
            
        # Decrease temperature if entropy is too high (too random)
        if current_entropy > self.config.entropy_target:
            target_temp *= 0.95
        elif current_entropy < self.config.entropy_target * 0.5:
            target_temp *= 1.05
            
        # Clamp temperature
        target_temp = max(self.config.min_temperature, min(self.config.max_temperature, target_temp))
        
        # Smooth adaptation
        new_temp = self.temperature * (1 - self.config.adaptation_rate) + target_temp * self.config.adaptation_rate
        self.temperature.data = torch.tensor(new_temp)
        
        self.step_count += 1
        
    def get_routing_stats(self) -> Dict[str, float]:
        """Get current routing statistics."""
        return {
            'temperature': self.temperature.item(),
            'mean_load_variance': self.load_history.mean().item(),
            'mean_entropy': self.entropy_history.mean().item(),
            'expert_importance_entropy': -(
                F.softmax(self.expert_importance, dim=0) * 
                F.log_softmax(self.expert_importance, dim=0)
            ).sum().item()
        }


class DynamicCapacityRouter(nn.Module):
    """Router with dynamic expert capacity allocation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        base_capacity_factor: float = 1.25,
        min_capacity_factor: float = 1.0,
        max_capacity_factor: float = 2.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.base_capacity_factor = base_capacity_factor
        self.min_capacity_factor = min_capacity_factor
        self.max_capacity_factor = max_capacity_factor
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Capacity predictor
        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts)
        )
        
        # Expert load tracker
        self.register_buffer('expert_loads', torch.ones(num_experts))
        self.register_buffer('load_momentum', torch.tensor(0.99))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.router.weight, std=0.02)
        for layer in self.capacity_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
                
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Forward pass with dynamic capacity."""
        batch_size, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Predict dynamic capacities
        capacity_logits = self.capacity_predictor(hidden_states.mean(dim=0, keepdim=True))
        capacity_weights = F.softmax(capacity_logits, dim=-1).squeeze(0)
        
        # Compute dynamic capacity factors
        capacity_factors = (
            self.base_capacity_factor + 
            (capacity_weights - 0.5) * (self.max_capacity_factor - self.min_capacity_factor)
        )
        capacity_factors = torch.clamp(capacity_factors, self.min_capacity_factor, self.max_capacity_factor)
        
        # Select experts with capacity constraints
        selected_experts = []
        expert_weights = []
        dropped_tokens = 0
        
        for expert_idx in range(self.num_experts):
            # Calculate capacity for this expert
            expert_capacity = int(batch_size * capacity_factors[expert_idx] / self.num_experts)
            
            # Get tokens that want this expert
            expert_probs = router_probs[:, expert_idx]
            top_values, top_indices = torch.topk(expert_probs, min(expert_capacity, batch_size))
            
            # Store selections
            for i, (token_idx, weight) in enumerate(zip(top_indices, top_values)):
                if i < expert_capacity:
                    selected_experts.append((token_idx.item(), expert_idx))
                    expert_weights.append(weight.item())
                else:
                    dropped_tokens += 1
                    
        # Convert to tensors
        if selected_experts:
            token_indices, expert_indices = zip(*selected_experts)
            selections = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
            weights = torch.zeros(batch_size, device=hidden_states.device)
            
            for token_idx, expert_idx, weight in zip(token_indices, expert_indices, expert_weights):
                selections[token_idx] = expert_idx
                weights[token_idx] = weight
        else:
            selections = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
            weights = torch.zeros(batch_size, device=hidden_states.device)
            
        # Update expert load tracking
        if self.training:
            current_loads = torch.bincount(selections, minlength=self.num_experts).float()
            current_loads = current_loads / current_loads.sum()
            self.expert_loads = self.load_momentum * self.expert_loads + (1 - self.load_momentum) * current_loads
            
        # Compute statistics
        load_variance = self.expert_loads.var().item()
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=weights.unsqueeze(-1),
            selected_experts=selections.unsqueeze(-1),
            router_logits=router_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return router_logits, selections.unsqueeze(-1), weights.unsqueeze(-1), routing_info


class HierarchicalRouter(nn.Module):
    """Hierarchical routing with coarse-to-fine expert selection."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_groups: int = 4,
        experts_per_group: Optional[int] = None,
        top_k: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group or (num_experts // num_groups)
        self.top_k = top_k
        
        # Group router (coarse selection)
        self.group_router = nn.Linear(hidden_size, num_groups, bias=False)
        
        # Expert routers (fine selection within groups)
        self.expert_routers = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        
        # Group assignment
        self.expert_to_group = self._create_expert_groups()
        
        self._init_weights()
        
    def _create_expert_groups(self) -> torch.Tensor:
        """Create mapping from experts to groups."""
        expert_to_group = torch.zeros(self.num_experts, dtype=torch.long)
        
        for expert_idx in range(self.num_experts):
            group_idx = expert_idx // self.experts_per_group
            group_idx = min(group_idx, self.num_groups - 1)
            expert_to_group[expert_idx] = group_idx
            
        return expert_to_group
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.group_router.weight, std=0.02)
        for router in self.expert_routers:
            nn.init.normal_(router.weight, std=0.02)
            
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Hierarchical routing forward pass."""
        batch_size, hidden_size = hidden_states.shape
        
        # Step 1: Group selection
        group_logits = self.group_router(hidden_states)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Select top groups
        top_groups_logits, top_groups = torch.topk(group_logits, min(2, self.num_groups), dim=-1)
        group_weights = F.softmax(top_groups_logits, dim=-1)
        
        # Step 2: Expert selection within groups
        all_expert_logits = torch.full((batch_size, self.num_experts), float('-inf'), device=hidden_states.device)
        
        for token_idx in range(batch_size):
            for group_idx_pos, group_weight in zip(top_groups[token_idx], group_weights[token_idx]):
                group_idx = group_idx_pos.item()
                
                # Get expert logits for this group
                expert_logits = self.expert_routers[group_idx](hidden_states[token_idx:token_idx+1])
                
                # Map to global expert indices
                start_expert = group_idx * self.experts_per_group
                end_expert = min(start_expert + self.experts_per_group, self.num_experts)
                
                # Weight by group probability
                weighted_logits = expert_logits.squeeze(0) + torch.log(group_weight + 1e-8)
                all_expert_logits[token_idx, start_expert:end_expert] = weighted_logits[:end_expert-start_expert]
                
        # Step 3: Final top-k expert selection
        top_k_logits, top_k_experts = torch.topk(all_expert_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute statistics
        final_probs = F.softmax(all_expert_logits, dim=-1)
        expert_load = final_probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(final_probs * torch.log(final_probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_experts,
            router_logits=all_expert_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return all_expert_logits, top_k_experts, expert_weights, routing_info
        
    def get_group_statistics(self) -> Dict[str, Any]:
        """Get group-level routing statistics."""
        # This would be populated during forward pass
        return {
            'num_groups': self.num_groups,
            'experts_per_group': self.experts_per_group,
            'expert_to_group_mapping': self.expert_to_group.tolist()
        }


class MetaLearningRouter(nn.Module):
    """Router that meta-learns routing strategies."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        meta_lr: float = 0.01,
        adaptation_steps: int = 5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        
        # Base router
        self.base_router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Meta-learning components
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.router_adapter = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Task context memory
        self.register_buffer('context_memory', torch.zeros(100, hidden_size // 4))
        self.register_buffer('memory_ptr', torch.tensor(0))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.base_router.weight, std=0.02)
        for module in [self.context_encoder, self.router_adapter]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutingInfo]:
        """Meta-learning routing forward pass."""
        batch_size, hidden_size = hidden_states.shape
        
        # Encode current context
        context = self.context_encoder(hidden_states.mean(dim=0, keepdim=True))
        
        # Retrieve similar contexts from memory
        if self.context_memory.norm() > 0:
            similarities = F.cosine_similarity(context, self.context_memory.unsqueeze(0), dim=-1)
            top_similarities, top_indices = torch.topk(similarities.squeeze(0), min(5, similarities.size(0)))
            
            # Weight by similarity
            weights = F.softmax(top_similarities, dim=0)
            retrieved_context = (weights.unsqueeze(-1) * self.context_memory[top_indices]).sum(dim=0, keepdim=True)
        else:
            retrieved_context = context
            
        # Adapt router based on context
        base_logits = self.base_router(hidden_states)
        adaptation = self.router_adapter(retrieved_context)
        adapted_logits = base_logits + adaptation
        
        # Standard top-k routing
        top_k_logits, top_k_experts = torch.topk(adapted_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Update context memory
        if self.training:
            ptr = self.memory_ptr.item() % self.context_memory.size(0)
            self.context_memory[ptr] = context.squeeze(0).detach()
            self.memory_ptr += 1
            
        # Compute statistics
        probs = F.softmax(adapted_logits, dim=-1)
        expert_load = probs.mean(dim=0)
        load_variance = expert_load.var().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        routing_info = RoutingInfo(
            expert_weights=expert_weights,
            selected_experts=top_k_experts,
            router_logits=adapted_logits,
            load_variance=load_variance,
            entropy=entropy
        )
        
        return adapted_logits, top_k_experts, expert_weights, routing_info
        
    def adapt_to_task(self, task_data: torch.Tensor, task_targets: torch.Tensor):
        """Adapt router to specific task."""
        # This would implement meta-learning adaptation
        # For now, just update context memory with task representation
        if task_data.size(0) > 0:
            task_context = self.context_encoder(task_data.mean(dim=0, keepdim=True))
            
            # Simple adaptation: add to memory
            ptr = self.memory_ptr.item() % self.context_memory.size(0)
            self.context_memory[ptr] = task_context.squeeze(0).detach()
            self.memory_ptr += 1