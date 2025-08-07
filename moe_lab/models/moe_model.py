"""Core MoE model implementation."""

from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .router import TopKRouter, RoutingInfo
from .expert import ExpertPool


@dataclass
class MoEOutput:
    """Output from MoE model forward pass."""
    
    last_hidden_state: torch.Tensor
    routing_info: RoutingInfo
    load_balancing_loss: Optional[torch.Tensor] = None
    router_z_loss: Optional[torch.Tensor] = None
    expert_weights: Optional[Dict[int, torch.Tensor]] = None


class MoELayer(nn.Module):
    """Single MoE layer with experts and routing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        experts_per_token: int = 2,
        router_type: str = "top_k",
        expert_hidden_size: Optional[int] = None,
        activation: str = "gelu",
        router_jitter_noise: float = 0.0,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        
        # Initialize router
        if router_type == "top_k":
            self.router = TopKRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=experts_per_token,
                jitter_noise=router_jitter_noise
            )
        else:
            raise ValueError(f"Unsupported router type: {router_type}")
            
        # Initialize expert pool
        expert_hidden_size = expert_hidden_size or hidden_size * 4
        self.experts = ExpertPool(
            num_experts=num_experts,
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            activation=activation
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, RoutingInfo, torch.Tensor]:
        """Forward pass through MoE layer."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Route tokens to experts
        router_logits, selected_experts, expert_weights, routing_info = self.router(hidden_states_flat)
        
        # Process through experts using the improved method
        expert_outputs = self.experts.forward_all(
            hidden_states_flat, 
            expert_weights, 
            selected_experts
        )
            
        # Compute auxiliary losses
        aux_loss = self._compute_auxiliary_loss(router_logits, selected_experts)
        z_loss = self._compute_z_loss(router_logits)
        total_loss = self.aux_loss_coef * aux_loss + self.z_loss_coef * z_loss
        
        # Reshape back to original dimensions
        output = expert_outputs.view(batch_size, seq_len, hidden_size)
        
        return output, routing_info, total_loss
        
    def _compute_auxiliary_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        # Expert selection frequency
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (selected_experts == expert_idx).sum().float()
            
        # Normalize to get load distribution
        total_tokens = selected_experts.numel()
        expert_load = expert_counts / total_tokens
        
        # Target uniform distribution
        target_load = torch.ones_like(expert_load) / self.num_experts
        
        # L2 loss between actual and target load
        aux_loss = F.mse_loss(expert_load, target_load)
        
        return aux_loss
        
    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute router z-loss to encourage lower logit variance."""
        return torch.logsumexp(router_logits, dim=-1).mean()


class MoEModel(nn.Module):
    """Complete MoE model with multiple layers."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_experts: int = 8,
        experts_per_token: int = 2,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 2048,
        router_type: str = "top_k",
        moe_layers: Optional[list] = None,  # Which layers are MoE
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.num_layers = num_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Determine which layers are MoE
        if moe_layers is None:
            # By default, make every other layer MoE starting from layer 1
            moe_layers = list(range(1, num_layers, 2))
        self.moe_layers = set(moe_layers)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in self.moe_layers:
                layer = MoELayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    experts_per_token=experts_per_token,
                    router_type=router_type,
                    **kwargs
                )
            else:
                # Standard transformer layer for non-MoE layers
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                )
            self.layers.append(layer)
            
        # Output head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> MoEOutput:
        """Forward pass through MoE model."""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
        token_embeddings = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = token_embeddings + position_embeddings
        
        # Tracking for routing info and losses
        all_routing_info = []
        total_load_balancing_loss = 0.0
        total_router_z_loss = 0.0
        expert_weights = {}
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers:
                # MoE layer
                hidden_states, routing_info, layer_loss = layer(hidden_states)
                all_routing_info.append(routing_info)
                total_load_balancing_loss += layer_loss
                
                # Collect expert weights for analysis
                expert_weights[i] = routing_info.expert_weights
            else:
                # Standard transformer layer
                if attention_mask is not None:
                    # Convert attention mask for transformer layer
                    src_key_padding_mask = (attention_mask == 0)
                    hidden_states = layer(hidden_states, src_key_padding_mask=src_key_padding_mask)
                else:
                    hidden_states = layer(hidden_states)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        
        # Combine routing info
        combined_routing_info = RoutingInfo(
            expert_weights=torch.stack([info.expert_weights for info in all_routing_info]) if all_routing_info else None,
            selected_experts=torch.stack([info.selected_experts for info in all_routing_info]) if all_routing_info else None,
            router_logits=torch.stack([info.router_logits for info in all_routing_info]) if all_routing_info else None,
            load_variance=sum(info.load_variance for info in all_routing_info) / len(all_routing_info) if all_routing_info else 0.0,
            entropy=sum(info.entropy for info in all_routing_info) / len(all_routing_info) if all_routing_info else 0.0
        )
        
        return MoEOutput(
            last_hidden_state=hidden_states,
            routing_info=combined_routing_info,
            load_balancing_loss=total_load_balancing_loss / len(self.moe_layers) if self.moe_layers else None,
            router_z_loss=total_router_z_loss / len(self.moe_layers) if self.moe_layers else None,
            expert_weights=expert_weights if return_routing_info else None
        )
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the MoE model."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids)
                hidden_states = outputs.last_hidden_state
                
                # Get logits for next token
                logits = self.lm_head(hidden_states[:, -1, :])
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                    
                # Sample next token
                if do_sample:
                    # Nucleus sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        return input_ids
        
    def get_expert_weights(self) -> Dict[int, torch.Tensor]:
        """Get expert weights from all MoE layers."""
        expert_weights = {}
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers:
                expert_weights[i] = layer.experts.get_weights()
        return expert_weights
        
    def set_expert_parallel_group(self, group) -> None:
        """Set expert parallel group for distributed training."""
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers:
                layer.experts.set_parallel_group(group)