"""Routing-specific validation for MoE models."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingValidationResult:
    """Result of routing validation."""
    
    is_valid: bool
    load_balance_score: float
    routing_efficiency: float
    expert_specialization_score: float
    errors: List[str]
    warnings: List[str]
    detailed_metrics: Dict[str, Any]


class RoutingValidator:
    """Specialized validator for MoE routing behavior."""
    
    def __init__(self, model: MoEModel, tolerance: float = 0.1):
        self.model = model
        self.tolerance = tolerance
        self.device = next(model.parameters()).device
        
    def validate_routing_behavior(
        self, 
        num_samples: int = 200,
        sequence_lengths: List[int] = None
    ) -> RoutingValidationResult:
        """Comprehensive routing behavior validation."""
        
        if sequence_lengths is None:
            sequence_lengths = [16, 32, 64, 128]
            
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Collect routing data
            routing_data = self._collect_routing_data(num_samples, sequence_lengths)
            
            # Validate load balancing
            load_balance_score, lb_errors, lb_warnings, lb_metrics = self._validate_load_balancing(routing_data)
            errors.extend(lb_errors)
            warnings.extend(lb_warnings)
            metrics.update(lb_metrics)
            
            # Validate routing efficiency
            efficiency_score, eff_errors, eff_warnings, eff_metrics = self._validate_routing_efficiency(routing_data)
            errors.extend(eff_errors)
            warnings.extend(eff_warnings)
            metrics.update(eff_metrics)
            
            # Validate expert specialization
            specialization_score, spec_errors, spec_warnings, spec_metrics = self._validate_expert_specialization(routing_data)
            errors.extend(spec_errors)
            warnings.extend(spec_warnings)
            metrics.update(spec_metrics)
            
            # Validate routing consistency
            cons_errors, cons_warnings, cons_metrics = self._validate_routing_consistency(routing_data)
            errors.extend(cons_errors)
            warnings.extend(cons_warnings)
            metrics.update(cons_metrics)
            
            # Validate capacity utilization
            cap_errors, cap_warnings, cap_metrics = self._validate_capacity_utilization(routing_data)
            errors.extend(cap_errors)
            warnings.extend(cap_warnings)
            metrics.update(cap_metrics)
            
        except Exception as e:
            errors.append(f"Routing validation failed: {str(e)}")
            load_balance_score = 0.0
            efficiency_score = 0.0
            specialization_score = 0.0
            
        is_valid = len(errors) == 0
        
        return RoutingValidationResult(
            is_valid=is_valid,
            load_balance_score=load_balance_score,
            routing_efficiency=efficiency_score,
            expert_specialization_score=specialization_score,
            errors=errors,
            warnings=warnings,
            detailed_metrics=metrics
        )
        
    def _collect_routing_data(
        self, 
        num_samples: int, 
        sequence_lengths: List[int]
    ) -> Dict[str, Any]:
        """Collect routing data for analysis."""
        
        routing_data = {
            'expert_selections': [],
            'expert_weights': [],
            'load_variances': [],
            'entropies': [],
            'sequence_lengths': [],
            'routing_logits': []
        }
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                # Random sequence length
                seq_len = np.random.choice(sequence_lengths)
                
                # Generate input
                input_ids = torch.randint(0, 1000, (1, seq_len), device=self.device)
                
                # Forward pass
                outputs = self.model(input_ids, return_routing_info=True)
                
                if outputs.routing_info is not None:
                    routing_info = outputs.routing_info
                    
                    routing_data['expert_selections'].append(
                        routing_info.selected_experts.cpu() if routing_info.selected_experts is not None else None
                    )
                    routing_data['expert_weights'].append(
                        routing_info.expert_weights.cpu() if routing_info.expert_weights is not None else None
                    )
                    routing_data['load_variances'].append(routing_info.load_variance)
                    routing_data['entropies'].append(routing_info.entropy)
                    routing_data['sequence_lengths'].append(seq_len)
                    routing_data['routing_logits'].append(
                        routing_info.router_logits.cpu() if routing_info.router_logits is not None else None
                    )
                    
        return routing_data
        
    def _validate_load_balancing(self, routing_data: Dict[str, Any]) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Validate load balancing across experts."""
        errors = []
        warnings = []
        metrics = {}
        
        # Collect expert usage statistics
        expert_counts = torch.zeros(self.model.num_experts)
        total_tokens = 0
        
        for expert_selection in routing_data['expert_selections']:
            if expert_selection is not None:
                experts = expert_selection.flatten()
                total_tokens += len(experts)
                
                for expert_idx in range(self.model.num_experts):
                    count = (experts == expert_idx).sum().item()
                    expert_counts[expert_idx] += count
                    
        # Compute utilization statistics
        if total_tokens > 0:
            expert_utilizations = expert_counts / total_tokens
            target_utilization = 1.0 / self.model.num_experts
            
            # Load balance metrics
            utilization_variance = expert_utilizations.var().item()
            utilization_cv = expert_utilizations.std().item() / expert_utilizations.mean().item()
            gini_coefficient = self._compute_gini_coefficient(expert_utilizations)
            
            metrics.update({
                'expert_utilizations': expert_utilizations.tolist(),
                'utilization_variance': utilization_variance,
                'utilization_cv': utilization_cv,
                'gini_coefficient': gini_coefficient,
                'target_utilization': target_utilization
            })
            
            # Load balance score (higher is better)
            load_balance_score = 1.0 - min(1.0, utilization_cv)
            
            # Check for load imbalance
            if utilization_cv > 0.5:
                errors.append(f"Severe load imbalance detected (CV: {utilization_cv:.3f})")
            elif utilization_cv > 0.3:
                warnings.append(f"Moderate load imbalance (CV: {utilization_cv:.3f})")
                
            # Check for unused experts
            unused_experts = (expert_utilizations < target_utilization * 0.01).sum().item()
            if unused_experts > 0:
                errors.append(f"{unused_experts} experts are severely under-utilized")
                
            # Check load variance trends
            load_variances = [lv for lv in routing_data['load_variances'] if lv is not None]
            if load_variances:
                mean_load_variance = np.mean(load_variances)
                metrics['mean_load_variance'] = mean_load_variance
                
                if mean_load_variance > 0.1:
                    warnings.append(f"High average load variance: {mean_load_variance:.4f}")
                    
        else:
            load_balance_score = 0.0
            errors.append("No routing data available for load balance analysis")
            
        return load_balance_score, errors, warnings, metrics
        
    def _validate_routing_efficiency(self, routing_data: Dict[str, Any]) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Validate routing efficiency and effectiveness."""
        errors = []
        warnings = []
        metrics = {}
        
        # Analyze routing entropy
        entropies = [e for e in routing_data['entropies'] if e is not None]
        
        if entropies:
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            max_possible_entropy = np.log(self.model.num_experts)
            
            # Normalized entropy (0-1)
            normalized_entropy = mean_entropy / max_possible_entropy
            
            metrics.update({
                'mean_entropy': mean_entropy,
                'std_entropy': std_entropy,
                'normalized_entropy': normalized_entropy,
                'max_possible_entropy': max_possible_entropy
            })
            
            # Efficiency score based on entropy utilization
            efficiency_score = normalized_entropy
            
            # Check entropy levels
            if normalized_entropy < 0.3:
                warnings.append(f"Low routing entropy: {normalized_entropy:.3f} (poor diversity)")
            elif normalized_entropy > 0.95:
                warnings.append(f"Very high routing entropy: {normalized_entropy:.3f} (may indicate random routing)")
                
            # Check entropy consistency
            if std_entropy > mean_entropy * 0.5:
                warnings.append(f"High entropy variance: {std_entropy:.3f}")
                
        else:
            efficiency_score = 0.0
            errors.append("No entropy data available for efficiency analysis")
            
        # Analyze expert weight distributions
        if routing_data['expert_weights']:
            weight_analysis = self._analyze_expert_weights(routing_data['expert_weights'])
            metrics.update(weight_analysis)
            
        return efficiency_score, errors, warnings, metrics
        
    def _validate_expert_specialization(self, routing_data: Dict[str, Any]) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Validate expert specialization patterns."""
        errors = []
        warnings = []
        metrics = {}
        
        # Analyze position-based routing patterns
        position_patterns = self._analyze_position_routing(routing_data)
        metrics.update(position_patterns)
        
        # Analyze sequence length effects
        length_effects = self._analyze_sequence_length_effects(routing_data)
        metrics.update(length_effects)
        
        # Compute specialization score
        specialization_score = self._compute_specialization_score(routing_data)
        metrics['specialization_score'] = specialization_score
        
        # Check for over-specialization
        if specialization_score > 0.9:
            warnings.append(f"Very high specialization: {specialization_score:.3f} (risk of expert collapse)")
        elif specialization_score < 0.1:
            warnings.append(f"Very low specialization: {specialization_score:.3f} (experts may not be specializing)")
            
        return specialization_score, errors, warnings, metrics
        
    def _validate_routing_consistency(self, routing_data: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate routing consistency across similar inputs."""
        errors = []
        warnings = []
        metrics = {}
        
        # Test consistency with repeated identical inputs
        test_input = torch.randint(0, 1000, (1, 32), device=self.device)
        
        self.model.eval()
        routing_results = []
        
        with torch.no_grad():
            for _ in range(10):
                outputs = self.model(test_input, return_routing_info=True)
                if outputs.routing_info and outputs.routing_info.selected_experts is not None:
                    routing_results.append(outputs.routing_info.selected_experts.cpu())
                    
        if len(routing_results) > 1:
            # Compute consistency metrics
            consistency_scores = []
            for i in range(len(routing_results)):
                for j in range(i + 1, len(routing_results)):
                    consistency = (routing_results[i] == routing_results[j]).float().mean().item()
                    consistency_scores.append(consistency)
                    
            if consistency_scores:
                mean_consistency = np.mean(consistency_scores)
                std_consistency = np.std(consistency_scores)
                
                metrics.update({
                    'routing_consistency': mean_consistency,
                    'consistency_std': std_consistency
                })
                
                # Check consistency levels
                if mean_consistency < 0.7:
                    warnings.append(f"Low routing consistency: {mean_consistency:.3f}")
                elif mean_consistency > 0.99:
                    warnings.append(f"Overly deterministic routing: {mean_consistency:.3f}")
                    
        return errors, warnings, metrics
        
    def _validate_capacity_utilization(self, routing_data: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate expert capacity utilization."""
        errors = []
        warnings = []
        metrics = {}
        
        # Analyze token dropping rates
        total_tokens = 0
        valid_tokens = 0
        
        for expert_selection in routing_data['expert_selections']:
            if expert_selection is not None:
                experts = expert_selection.flatten()
                total_tokens += len(experts)
                valid_tokens += (experts >= 0).sum().item()
                
        if total_tokens > 0:
            drop_rate = 1.0 - (valid_tokens / total_tokens)
            metrics['token_drop_rate'] = drop_rate
            
            if drop_rate > 0.1:
                errors.append(f"High token drop rate: {drop_rate:.2%}")
            elif drop_rate > 0.05:
                warnings.append(f"Moderate token drop rate: {drop_rate:.2%}")
                
        # Analyze capacity utilization efficiency
        if routing_data['expert_weights']:
            weight_efficiency = self._analyze_weight_efficiency(routing_data['expert_weights'])
            metrics.update(weight_efficiency)
            
        return errors, warnings, metrics
        
    def _compute_gini_coefficient(self, utilizations: torch.Tensor) -> float:
        """Compute Gini coefficient for load balance measurement."""
        sorted_utils = torch.sort(utilizations)[0]
        n = len(sorted_utils)
        index = torch.arange(1, n + 1, dtype=torch.float)
        
        gini = (2 * torch.sum(index * sorted_utils)) / (n * torch.sum(sorted_utils)) - (n + 1) / n
        return gini.item()
        
    def _analyze_expert_weights(self, expert_weights: List) -> Dict[str, Any]:
        """Analyze expert weight distributions."""
        metrics = {}
        
        all_weights = []
        for weights in expert_weights:
            if weights is not None:
                all_weights.extend(weights.flatten().tolist())
                
        if all_weights:
            all_weights = np.array(all_weights)
            
            metrics.update({
                'weight_mean': np.mean(all_weights),
                'weight_std': np.std(all_weights),
                'weight_min': np.min(all_weights),
                'weight_max': np.max(all_weights),
                'weight_entropy': -np.sum(all_weights * np.log(all_weights + 1e-8))
            })
            
        return metrics
        
    def _analyze_position_routing(self, routing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze position-based routing patterns."""
        metrics = {}
        
        # Collect position-expert mappings
        position_expert_counts = {}
        
        for expert_selection in routing_data['expert_selections']:
            if expert_selection is not None and expert_selection.dim() >= 2:
                seq_len = expert_selection.shape[1]
                
                for pos in range(min(seq_len, 50)):  # Limit to first 50 positions
                    if pos not in position_expert_counts:
                        position_expert_counts[pos] = torch.zeros(self.model.num_experts)
                        
                    pos_experts = expert_selection[0, pos]  # Assuming batch size 1
                    for expert_idx in pos_experts:
                        if expert_idx >= 0:
                            position_expert_counts[pos][expert_idx] += 1
                            
        # Compute position specialization
        if position_expert_counts:
            position_entropies = []
            for pos, counts in position_expert_counts.items():
                if counts.sum() > 0:
                    probs = counts / counts.sum()
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    position_entropies.append(entropy)
                    
            if position_entropies:
                metrics.update({
                    'position_specialization_entropy': np.mean(position_entropies),
                    'position_entropy_std': np.std(position_entropies)
                })
                
        return metrics
        
    def _analyze_sequence_length_effects(self, routing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how routing changes with sequence length."""
        metrics = {}
        
        # Group by sequence length
        length_groups = {}
        for i, seq_len in enumerate(routing_data['sequence_lengths']):
            if seq_len not in length_groups:
                length_groups[seq_len] = []
            length_groups[seq_len].append(i)
            
        # Analyze each length group
        length_entropies = {}
        for seq_len, indices in length_groups.items():
            entropies = [routing_data['entropies'][i] for i in indices if routing_data['entropies'][i] is not None]
            if entropies:
                length_entropies[seq_len] = np.mean(entropies)
                
        if len(length_entropies) > 1:
            entropy_values = list(length_entropies.values())
            metrics.update({
                'length_entropy_variance': np.var(entropy_values),
                'length_entropy_correlation': self._compute_length_entropy_correlation(length_entropies)
            })
            
        return metrics
        
    def _compute_specialization_score(self, routing_data: Dict[str, Any]) -> float:
        """Compute overall expert specialization score."""
        if not routing_data['expert_selections']:
            return 0.0
            
        # Compute expert co-occurrence matrix
        co_occurrence = torch.zeros(self.model.num_experts, self.model.num_experts)
        
        for expert_selection in routing_data['expert_selections']:
            if expert_selection is not None:
                experts = expert_selection.flatten()
                valid_experts = experts[experts >= 0]
                
                # Count co-occurrences
                for i in range(len(valid_experts)):
                    for j in range(i + 1, len(valid_experts)):
                        exp_i, exp_j = valid_experts[i].item(), valid_experts[j].item()
                        co_occurrence[exp_i, exp_j] += 1
                        co_occurrence[exp_j, exp_i] += 1
                        
        # Normalize and compute specialization
        if co_occurrence.sum() > 0:
            co_occurrence = co_occurrence / co_occurrence.sum()
            
            # Higher diagonal values indicate specialization
            diagonal_sum = torch.diag(co_occurrence).sum().item()
            total_sum = co_occurrence.sum().item()
            
            specialization_score = diagonal_sum / total_sum if total_sum > 0 else 0.0
        else:
            specialization_score = 0.0
            
        return specialization_score
        
    def _analyze_weight_efficiency(self, expert_weights: List) -> Dict[str, Any]:
        """Analyze efficiency of expert weight utilization."""
        metrics = {}
        
        effective_weights = []
        weight_concentration = []
        
        for weights in expert_weights:
            if weights is not None and weights.numel() > 0:
                # Flatten and analyze
                w = weights.flatten()
                
                # Effective number of experts (based on weight distribution)
                probs = torch.softmax(w, dim=-1)
                effective_experts = torch.exp(-(probs * torch.log(probs + 1e-8)).sum())
                effective_weights.append(effective_experts.item())
                
                # Weight concentration (Gini coefficient)
                if len(w) > 1:
                    concentration = self._compute_gini_coefficient(w)
                    weight_concentration.append(concentration)
                    
        if effective_weights:
            metrics.update({
                'mean_effective_experts': np.mean(effective_weights),
                'std_effective_experts': np.std(effective_weights)
            })
            
        if weight_concentration:
            metrics.update({
                'mean_weight_concentration': np.mean(weight_concentration),
                'std_weight_concentration': np.std(weight_concentration)
            })
            
        return metrics
        
    def _compute_length_entropy_correlation(self, length_entropies: Dict[int, float]) -> float:
        """Compute correlation between sequence length and entropy."""
        if len(length_entropies) < 3:
            return 0.0
            
        lengths = list(length_entropies.keys())
        entropies = list(length_entropies.values())
        
        correlation = np.corrcoef(lengths, entropies)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0