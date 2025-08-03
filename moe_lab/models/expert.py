"""Expert network implementations."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network (feed-forward network)."""
    
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        
        # Expert network layers
        self.fc1 = nn.Linear(hidden_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize expert weights."""
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GLUExpert(nn.Module):
    """Expert with Gated Linear Unit (GLU) activation."""
    
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        activation: str = "swish",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        
        # GLU requires double the intermediate size for gating
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Activation for gating
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation for GLU: {activation}")
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize GLU expert weights."""
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(layer.weight, std=0.02)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with GLU."""
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(self.dropout(intermediate))
        return output


class ExpertPool(nn.Module):
    """Pool of expert networks."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        expert_hidden_size: int,
        activation: str = "gelu",
        expert_type: str = "standard",
        dropout: float = 0.1,
        expert_dropout: float = 0.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.expert_dropout = expert_dropout
        
        # Create expert networks
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if expert_type == "glu":
                expert = GLUExpert(
                    hidden_size=hidden_size,
                    expert_hidden_size=expert_hidden_size,
                    activation=activation,
                    dropout=dropout
                )
            else:  # standard expert
                expert = Expert(
                    hidden_size=hidden_size,
                    expert_hidden_size=expert_hidden_size,
                    activation=activation,
                    dropout=dropout
                )
            self.experts.append(expert)
            
        # For distributed training
        self.parallel_group = None
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        expert_idx: int
    ) -> torch.Tensor:
        """Forward pass through specific expert."""
        if expert_idx >= self.num_experts:
            raise ValueError(f"Expert index {expert_idx} >= num_experts {self.num_experts}")
            
        # Apply expert dropout during training
        if self.training and self.expert_dropout > 0.0:
            if torch.rand(1).item() < self.expert_dropout:
                # Return zeros if expert is dropped out
                return torch.zeros_like(hidden_states)
                
        return self.experts[expert_idx](hidden_states)
        
    def forward_all(
        self, 
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Forward through multiple experts with routing weights."""
        batch_size, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Process each token through its selected experts
        for batch_idx in range(batch_size):
            token_hidden_states = hidden_states[batch_idx:batch_idx+1]  # Keep batch dim
            token_experts = selected_experts[batch_idx]
            token_weights = expert_weights[batch_idx]
            
            # Aggregate outputs from selected experts
            token_output = torch.zeros_like(token_hidden_states)
            for k in range(len(token_experts)):
                expert_idx = token_experts[k].item()
                if expert_idx >= 0:  # Valid expert (not dropped)
                    expert_output = self.forward(token_hidden_states, expert_idx)
                    token_output += token_weights[k] * expert_output
                    
            output[batch_idx] = token_output.squeeze(0)
            
        return output
        
    def get_weights(self) -> Dict[int, torch.Tensor]:
        """Get weights from all experts for analysis."""
        weights = {}
        for i, expert in enumerate(self.experts):
            if hasattr(expert, 'fc1'):
                weights[i] = expert.fc1.weight.detach().clone()
            elif hasattr(expert, 'up_proj'):
                weights[i] = expert.up_proj.weight.detach().clone()
        return weights
        
    def set_parallel_group(self, group):
        """Set distributed training group."""
        self.parallel_group = group
        
    def get_expert_utilization(self, routing_history: list) -> Dict[int, float]:
        """Calculate how often each expert is used."""
        expert_counts = torch.zeros(self.num_experts)
        total_tokens = 0
        
        for routing_info in routing_history:
            selected_experts = routing_info.selected_experts
            total_tokens += selected_experts.numel()
            
            for expert_idx in range(self.num_experts):
                count = (selected_experts == expert_idx).sum().item()
                expert_counts[expert_idx] += count
                
        # Convert to utilization percentages
        utilization = {}
        for expert_idx in range(self.num_experts):
            utilization[expert_idx] = (expert_counts[expert_idx] / total_tokens).item()
            
        return utilization
        
    def get_expert_parameters(self) -> Dict[str, Any]:
        """Get expert parameter statistics."""
        total_params = 0
        expert_params = []
        
        for expert in self.experts:
            expert_param_count = sum(p.numel() for p in expert.parameters())
            expert_params.append(expert_param_count)
            total_params += expert_param_count
            
        return {
            "total_parameters": total_params,
            "parameters_per_expert": expert_params,
            "average_parameters": total_params / self.num_experts,
            "expert_parameter_std": torch.tensor(expert_params, dtype=torch.float).std().item()
        }


class SharedExpertPool(ExpertPool):
    """Expert pool with shared parameters for memory efficiency."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        expert_hidden_size: int,
        shared_expert_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(num_experts, hidden_size, expert_hidden_size, **kwargs)
        
        self.shared_expert_ratio = shared_expert_ratio
        self.num_shared_experts = int(num_experts * shared_expert_ratio)
        
        # Create shared expert parameters
        if self.num_shared_experts > 0:
            self.shared_fc1 = nn.Linear(hidden_size, expert_hidden_size)
            self.shared_fc2 = nn.Linear(expert_hidden_size, hidden_size)
            
            # Initialize shared parameters
            nn.init.normal_(self.shared_fc1.weight, std=0.02)
            nn.init.normal_(self.shared_fc2.weight, std=0.02)
            nn.init.zeros_(self.shared_fc1.bias)
            nn.init.zeros_(self.shared_fc2.bias)
            
    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Forward with shared parameters for some experts."""
        if expert_idx < self.num_shared_experts and hasattr(self, 'shared_fc1'):
            # Use shared parameters
            x = self.shared_fc1(hidden_states)
            x = F.gelu(x)  # Use fixed activation for shared experts
            x = self.shared_fc2(x)
            return x
        else:
            # Use dedicated expert
            return super().forward(hidden_states, expert_idx)