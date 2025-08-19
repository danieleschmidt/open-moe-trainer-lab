"""Router decision analysis for MoE models."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingAnalysis:
    """Analysis results for routing decisions."""
    
    expert_specialization: Dict[int, Dict[str, float]]
    token_expert_affinity: Dict[str, Dict[int, float]]
    routing_patterns: Dict[str, Any]
    load_distribution: np.ndarray
    entropy_distribution: np.ndarray


class RouterAnalyzer:
    """Analyzer for router decision patterns and expert specialization."""
    
    def __init__(self, model: MoEModel, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.routing_cache = {}
        
    def analyze_batch(
        self, 
        inputs: torch.Tensor,
        return_attention_maps: bool = False
    ) -> Dict[str, Any]:
        """Analyze routing decisions for a batch of inputs."""
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = self.model(**inputs, return_routing_info=True)
            else:
                outputs = self.model(inputs, return_routing_info=True)
                
        routing_info = outputs.routing_info
        
        analysis = {
            'batch_size': inputs['input_ids'].shape[0] if isinstance(inputs, dict) else inputs.shape[0],
            'sequence_length': inputs['input_ids'].shape[1] if isinstance(inputs, dict) else inputs.shape[1],
            'expert_weights': routing_info.expert_weights,
            'selected_experts': routing_info.selected_experts,
            'load_variance': routing_info.load_variance,
            'entropy': routing_info.entropy,
            'expert_utilization': self._compute_expert_utilization(routing_info.selected_experts)
        }
        
        if return_attention_maps:
            analysis['attention_maps'] = self._compute_attention_maps(routing_info)
            
        return analysis
        
    def _compute_expert_utilization(self, selected_experts: torch.Tensor) -> Dict[int, float]:
        """Compute how often each expert is selected."""
        experts_flat = selected_experts.flatten()
        total_selections = len(experts_flat)
        
        utilization = {}
        for expert_idx in range(self.model.num_experts):
            count = (experts_flat == expert_idx).sum().item()
            utilization[expert_idx] = count / total_selections if total_selections > 0 else 0.0
            
        return utilization
        
    def _compute_attention_maps(self, routing_info) -> Dict[str, np.ndarray]:
        """Compute attention-like maps for routing decisions."""
        expert_weights = routing_info.expert_weights
        selected_experts = routing_info.selected_experts
        
        # Create routing attention matrix
        batch_size, seq_len, top_k = selected_experts.shape
        attention_map = np.zeros((batch_size, seq_len, self.model.num_experts))
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(top_k):
                    expert_idx = selected_experts[b, s, k].item()
                    if expert_idx >= 0:
                        weight = expert_weights[b, s, k].item()
                        attention_map[b, s, expert_idx] = weight
                        
        return {
            'routing_attention': attention_map,
            'expert_activation_pattern': attention_map.sum(axis=(0, 1))
        }
        
    def compute_expert_specialization(
        self, 
        dataset,
        num_samples: int = 1000,
        token_categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[int, Dict[str, float]]:
        """Analyze what types of content each expert specializes in."""
        if token_categories is None:
            token_categories = {
                'math': ['calculate', 'equation', 'number', '+', '-', '*', '/', '='],
                'code': ['def', 'class', 'import', 'function', 'variable', '{', '}', '(', ')'],
                'text': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has'],
                'punctuation': ['.', ',', '!', '?', ':', ';', '"', "'"]
            }
            
        expert_specialization = defaultdict(lambda: defaultdict(float))
        category_counts = defaultdict(int)
        
        self.model.eval()
        
        # Sample from dataset
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in sample_indices:
                try:
                    batch = dataset[idx]
                    if isinstance(batch, dict):
                        input_ids = batch['input_ids']
                        outputs = self.model(**batch, return_routing_info=True)
                    else:
                        input_ids = batch
                        outputs = self.model(batch, return_routing_info=True)
                        
                    selected_experts = outputs.routing_info.selected_experts
                    expert_weights = outputs.routing_info.expert_weights
                    
                    # Decode tokens if tokenizer available
                    if self.tokenizer is not None:
                        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.flatten())
                    else:
                        tokens = [f"token_{i}" for i in input_ids.flatten()]
                        
                    # Analyze token-expert assignments
                    for pos, token in enumerate(tokens):
                        if pos < selected_experts.shape[1]:  # Within sequence length
                            # Find which category this token belongs to
                            token_lower = token.lower().strip()
                            for category, category_tokens in token_categories.items():
                                if any(cat_token in token_lower for cat_token in category_tokens):
                                    category_counts[category] += 1
                                    
                                    # Get experts for this position
                                    pos_experts = selected_experts[0, pos]  # Assuming batch size 1
                                    pos_weights = expert_weights[0, pos]
                                    
                                    for k in range(len(pos_experts)):
                                        expert_idx = pos_experts[k].item()
                                        if expert_idx >= 0:
                                            weight = pos_weights[k].item()
                                            expert_specialization[expert_idx][category] += weight
                                            
                except Exception as e:
                    logger.warning(f"Error processing sample {idx}: {e}")
                    continue
                    
        # Normalize specialization scores
        normalized_specialization = {}
        for expert_idx, categories in expert_specialization.items():
            normalized_categories = {}
            total_weight = sum(categories.values())
            for category, weight in categories.items():
                if category_counts[category] > 0:
                    normalized_categories[category] = weight / category_counts[category]
                else:
                    normalized_categories[category] = 0.0
            normalized_specialization[expert_idx] = normalized_categories
            
        return normalized_specialization
        
    def plot_expert_token_affinity(
        self, 
        tokens: List[str],
        save_to: Optional[str] = None
    ):
        """Plot which experts have affinity for specific tokens."""
        if self.tokenizer is None:
            logger.warning("Tokenizer not available for token analysis")
            return
            
        # Tokenize input tokens
        token_ids = []
        for token in tokens:
            try:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids:
                    token_ids.append(ids[0])
            except:
                logger.warning(f"Could not tokenize '{token}'")
                
        if not token_ids:
            logger.error("No valid tokens to analyze")
            return
            
        # Create input tensor
        input_tensor = torch.tensor([token_ids], device=next(self.model.parameters()).device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor, return_routing_info=True)
            
        routing_info = outputs.routing_info
        expert_weights = routing_info.expert_weights[0]  # Remove batch dimension
        selected_experts = routing_info.selected_experts[0]
        
        # Create affinity matrix
        affinity_matrix = np.zeros((len(tokens), self.model.num_experts))
        
        for pos, token in enumerate(tokens):
            if pos < len(selected_experts):
                pos_experts = selected_experts[pos]
                pos_weights = expert_weights[pos]
                
                for k in range(len(pos_experts)):
                    expert_idx = pos_experts[k].item()
                    if expert_idx >= 0:
                        weight = pos_weights[k].item()
                        affinity_matrix[pos, expert_idx] = weight
                        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            affinity_matrix,
            xticklabels=[f"Expert {i}" for i in range(self.model.num_experts)],
            yticklabels=tokens,
            annot=True,
            fmt='.3f',
            cmap='Blues'
        )
        plt.title('Token-Expert Affinity Heatmap')
        plt.xlabel('Experts')
        plt.ylabel('Tokens')
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            logger.info(f"Token affinity plot saved to {save_to}")
        else:
            plt.show()
            
    def analyze_routing_consistency(
        self, 
        inputs: torch.Tensor,
        num_trials: int = 5
    ) -> Dict[str, float]:
        """Analyze how consistent routing decisions are across multiple runs."""
        self.model.eval()
        
        all_routings = []
        
        for trial in range(num_trials):
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs, return_routing_info=True)
                else:
                    outputs = self.model(inputs, return_routing_info=True)
                    
                selected_experts = outputs.routing_info.selected_experts
                all_routings.append(selected_experts.cpu().numpy())
                
        # Compute consistency metrics
        all_routings = np.array(all_routings)  # [trials, batch, seq, top_k]
        
        # Position-wise consistency (how often same expert is chosen for same position)
        position_consistency = []
        for pos in range(all_routings.shape[2]):  # For each sequence position
            pos_routings = all_routings[:, 0, pos, 0]  # Top expert for each trial
            unique_experts = len(np.unique(pos_routings))
            consistency = 1.0 - (unique_experts - 1) / (num_trials - 1) if num_trials > 1 else 1.0
            position_consistency.append(consistency)
            
        # Overall routing variance
        routing_variance = np.var(all_routings)
        
        # Expert usage consistency
        expert_usage_variance = []
        for expert_idx in range(self.model.num_experts):
            usage_across_trials = []
            for trial in range(num_trials):
                usage = (all_routings[trial] == expert_idx).sum()
                usage_across_trials.append(usage)
            expert_usage_variance.append(np.var(usage_across_trials))
            
        return {
            'mean_position_consistency': np.mean(position_consistency),
            'routing_variance': float(routing_variance),
            'expert_usage_variance': np.mean(expert_usage_variance),
            'position_consistencies': position_consistency
        }
        
    def generate_routing_report(
        self, 
        dataset,
        num_samples: int = 100,
        save_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive routing analysis report."""
        logger.info("Generating routing analysis report...")
        
        # Collect routing statistics
        all_load_variances = []
        all_entropies = []
        all_utilizations = []
        
        self.model.eval()
        
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in sample_indices:
                try:
                    batch = dataset[idx]
                    analysis = self.analyze_batch(batch)
                    
                    all_load_variances.append(analysis['load_variance'])
                    all_entropies.append(analysis['entropy'])
                    all_utilizations.append(analysis['expert_utilization'])
                    
                except Exception as e:
                    logger.warning(f"Error analyzing sample {idx}: {e}")
                    continue
                    
        # Aggregate utilization statistics
        avg_utilization = defaultdict(float)
        for util_dict in all_utilizations:
            for expert_idx, util in util_dict.items():
                avg_utilization[expert_idx] += util / len(all_utilizations)
                
        # Compute expert specialization
        specialization = self.compute_expert_specialization(dataset, num_samples)
        
        report = {
            'summary': {
                'num_samples_analyzed': len(sample_indices),
                'avg_load_variance': np.mean(all_load_variances),
                'std_load_variance': np.std(all_load_variances),
                'avg_entropy': np.mean(all_entropies),
                'std_entropy': np.std(all_entropies),
                'expert_utilization_balance': np.std(list(avg_utilization.values()))
            },
            'expert_utilization': dict(avg_utilization),
            'expert_specialization': specialization,
            'load_variance_distribution': all_load_variances,
            'entropy_distribution': all_entropies
        }
        
        if save_to:
            import json
            with open(save_to, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in report.items()
                }
                json.dump(json_report, f, indent=2)
            logger.info(f"Routing report saved to {save_to}")
            
        return report