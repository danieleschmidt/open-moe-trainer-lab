"""Pre-built MoE architectures."""

from typing import Optional, List
import torch
import torch.nn as nn

from .moe_model import MoEModel, MoELayer
from .router import SwitchRouter, ExpertChoice


class SwitchTransformer(MoEModel):
    """Switch Transformer implementation with single expert per token."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_experts: int = 128,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 2048,
        expert_capacity_factor: float = 1.25,
        jitter_noise: float = 0.1,
        drop_tokens: bool = True,
        **kwargs
    ):
        # Initialize base model without MoE layers first
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            experts_per_token=1,  # Switch uses single expert
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            moe_layers=None,  # We'll override layers
            **kwargs
        )
        
        # Replace MoE layers with Switch-specific layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 1:  # Every other layer is MoE
                layer = SwitchMoELayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    capacity_factor=expert_capacity_factor,
                    jitter_noise=jitter_noise,
                    drop_tokens=drop_tokens,
                    **kwargs
                )
                # Note: moe_layers is a set, initialized in parent
                if not hasattr(self, 'moe_layers'):
                    self.moe_layers = set()
                self.moe_layers.add(i)
            else:
                # Standard transformer layer
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                )
            self.layers.append(layer)


class SwitchMoELayer(nn.Module):
    """MoE layer using Switch routing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.1,
        drop_tokens: bool = True,
        expert_hidden_size: Optional[int] = None,
        activation: str = "gelu",
        aux_loss_coef: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.aux_loss_coef = aux_loss_coef
        
        # Switch router
        self.router = SwitchRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            jitter_noise=jitter_noise,
            capacity_factor=capacity_factor,
            drop_tokens=drop_tokens
        )
        
        # Expert pool
        from .expert import ExpertPool
        expert_hidden_size = expert_hidden_size or hidden_size * 4
        self.experts = ExpertPool(
            num_experts=num_experts,
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            activation=activation
        )
        
    def forward(self, hidden_states: torch.Tensor):
        """Forward pass through Switch MoE layer."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Route tokens
        router_logits, selected_experts, expert_weights, routing_info = self.router(hidden_states_flat)
        
        # Process through experts
        output = self.experts.forward_all(hidden_states_flat, expert_weights, selected_experts)
        
        # Compute auxiliary loss
        aux_loss = self._compute_auxiliary_loss(router_logits, selected_experts)
        total_loss = self.aux_loss_coef * aux_loss
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, routing_info, total_loss
        
    def _compute_auxiliary_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for Switch Transformer."""
        # Get routing probabilities
        routing_probs = torch.softmax(router_logits, dim=-1)
        
        # Compute mean probability per expert (fraction of tokens)
        mean_probs = routing_probs.mean(dim=0)
        
        # Compute fraction of tokens assigned to each expert
        assignments = torch.zeros_like(mean_probs)
        for expert_idx in range(self.num_experts):
            assignments[expert_idx] = (selected_experts.squeeze(-1) == expert_idx).float().mean()
            
        # Auxiliary loss is mean_probs * assignments summed over experts
        aux_loss = (mean_probs * assignments).sum() * self.num_experts
        
        return aux_loss


class MixtralModel(MoEModel):
    """Mixtral-style MoE model with top-2 routing."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_experts: int = 8,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 32768,
        normalize_expert_weights: bool = True,
        **kwargs
    ):
        # Set MoE layers (in Mixtral, specific layers are MoE)
        moe_layers = list(range(0, num_layers, 4))  # Every 4th layer
        
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            experts_per_token=2,  # Top-2 routing
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            moe_layers=moe_layers,
            expert_hidden_size=intermediate_size,
            **kwargs
        )
        
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        
        # Use GLU experts (like Mixtral)
        for i in range(num_layers):
            if i in self.moe_layers:
                # Replace with Mixtral-style MoE layer
                from .expert import ExpertPool
                self.layers[i].experts = ExpertPool(
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    expert_hidden_size=intermediate_size,
                    expert_type="glu",  # Use GLU experts
                    activation="swish"
                )


class GLaMMoE(MoEModel):
    """Generalist Language Model MoE for multi-task learning."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_experts: int = 64,
        num_shared_experts: int = 2,
        experts_per_token: int = 2,
        num_layers: int = 24,
        task_embedding_dim: int = 64,
        num_tasks: int = 10,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            num_layers=num_layers,
            **kwargs
        )
        
        self.num_shared_experts = num_shared_experts
        self.task_embedding_dim = task_embedding_dim
        self.num_tasks = num_tasks
        
        # Task embeddings for task-aware routing
        self.task_embeddings = nn.Embedding(num_tasks, task_embedding_dim)
        
        # Modify routers to be task-aware
        for i in range(num_layers):
            if i in self.moe_layers:
                layer = self.layers[i]
                # Add task conditioning to router
                original_router = layer.router.router
                layer.router.router = nn.Linear(
                    hidden_size + task_embedding_dim, 
                    num_experts,
                    bias=False
                )
                # Initialize new router with original weights
                with torch.no_grad():
                    layer.router.router.weight[:, :hidden_size] = original_router.weight
                    # Initialize task part randomly
                    nn.init.normal_(layer.router.router.weight[:, hidden_size:], std=0.02)
                    
        # Shared experts that always participate
        if num_shared_experts > 0:
            from .expert import ExpertPool
            self.shared_experts = ExpertPool(
                num_experts=num_shared_experts,
                hidden_size=hidden_size,
                expert_hidden_size=hidden_size * 4
            )
            
    def forward(
        self,
        input_ids: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with task conditioning."""
        if task_ids is not None:
            # Add task embeddings to input
            task_embeds = self.task_embeddings(task_ids)
            # Task embeddings will be concatenated to hidden states in router
            kwargs['task_embeddings'] = task_embeds
            
        return super().forward(input_ids, **kwargs)


class CustomMoE(MoEModel):
    """Customizable MoE model for experimentation."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_experts: int = 8,
        experts_per_token: int = 2,
        num_layers: int = 12,
        router_type: str = "top_k",
        expert_type: str = "standard",
        routing_strategy: str = "learned",
        load_balancing_strategy: str = "auxiliary_loss",
        expert_specialization: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            num_layers=num_layers,
            router_type=router_type,
            **kwargs
        )
        
        self.expert_type = expert_type
        self.routing_strategy = routing_strategy
        self.load_balancing_strategy = load_balancing_strategy
        self.expert_specialization = expert_specialization
        
        # Customize expert pools based on configuration
        for i in range(num_layers):
            if i in self.moe_layers:
                layer = self.layers[i]
                # Replace expert pool with custom configuration
                from .expert import ExpertPool
                try:
                    from .expert import SharedExpertPool
                except ImportError:
                    SharedExpertPool = None
                
                if expert_type == "shared" and SharedExpertPool is not None:
                    layer.experts = SharedExpertPool(
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        expert_hidden_size=hidden_size * 4,
                        shared_expert_ratio=0.5
                    )
                else:
                    layer.experts = ExpertPool(
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        expert_hidden_size=hidden_size * 4
                    )
                    
        # Initialize expert specialization if provided
        if expert_specialization:
            self._initialize_expert_specialization()
            
    def _initialize_expert_specialization(self):
        """Initialize experts with domain-specific weights."""
        # This would implement domain-specific initialization
        # For now, just add some bias to different experts
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers:
                for expert_idx, expert in enumerate(layer.experts.experts):
                    if expert_idx < len(self.expert_specialization):
                        # Add small bias based on specialization
                        domain = self.expert_specialization[expert_idx]
                        bias_scale = hash(domain) % 100 / 1000.0  # Small domain-specific bias
                        with torch.no_grad():
                            expert.fc1.bias += bias_scale