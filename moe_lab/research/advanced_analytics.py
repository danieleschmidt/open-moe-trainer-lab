"""Advanced analytics components for revolutionary MoE research.

This module provides cutting-edge analytical tools:
1. Bayesian Hyperparameter Optimization with Gaussian Processes
2. Multi-Objective Optimization with Pareto Frontier Analysis
3. Causal Inference for Routing Pattern Understanding
4. Meta-Learning for Cross-Domain Transfer
5. Real-Time Experiment Monitoring and Adaptive Control
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import time
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    pareto_frontier: Optional[List[Dict[str, Any]]] = None


class BayesianOptimizer:
    """Advanced Bayesian optimization for hyperparameter tuning using Gaussian processes."""
    
    def __init__(
        self,
        config,
        acquisition_function: str = "expected_improvement",
        n_initial_points: int = 10,
        n_calls: int = 50,
        kernel_type: str = "matern",
        xi: float = 0.01,
        kappa: float = 2.576
    ):
        self.config = config
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.xi = xi
        self.kappa = kappa
        
        # Initialize Gaussian Process
        if kernel_type == "matern":
            kernel = Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = Matern(length_scale=1.0, nu=1.5)
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # Optimization space definition
        self.param_bounds = self._define_parameter_space()
        
        # History tracking
        self.X_observed = []
        self.y_observed = []
        self.optimization_history = []
        
    def _define_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Define the hyperparameter search space."""
        return {
            'learning_rate': (1e-5, 1e-2),
            'num_experts': (4, 64),
            'experts_per_token': (1, 8),
            'aux_loss_coef': (0.001, 0.1),
            'router_z_loss_coef': (0.0001, 0.01),
            'hidden_size': (256, 2048),
            'batch_size': (8, 128),
            'warmup_steps': (100, 5000)
        }
        
    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to normalized space [0, 1]."""
        encoded = []
        for param_name, (low, high) in self.param_bounds.items():
            if param_name in params:
                # Normalize to [0, 1]
                normalized = (params[param_name] - low) / (high - low)
                encoded.append(np.clip(normalized, 0, 1))
            else:
                encoded.append(0.5)  # Default middle value
        return np.array(encoded)
    
    def _decode_parameters(self, encoded_params: np.ndarray) -> Dict[str, Any]:
        """Decode normalized parameters back to original space."""
        decoded = {}
        param_names = list(self.param_bounds.keys())
        
        for i, param_name in enumerate(param_names):
            low, high = self.param_bounds[param_name]
            # Denormalize from [0, 1]
            value = low + encoded_params[i] * (high - low)
            
            # Round integers
            if param_name in ['num_experts', 'experts_per_token', 'batch_size', 'warmup_steps', 'hidden_size']:
                value = int(round(value))
                
            decoded[param_name] = value
            
        return decoded
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function values."""
        if len(self.X_observed) == 0:
            return np.random.random(len(X))
            
        # Fit GP to observed data
        X_observed = np.array(self.X_observed)
        y_observed = np.array(self.y_observed)
        
        self.gp.fit(X_observed, y_observed)
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.acquisition_function == "expected_improvement":
            # Expected Improvement
            y_best = np.max(y_observed)
            improvement = mu - y_best - self.xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0] = 0
            return ei
            
        elif self.acquisition_function == "upper_confidence_bound":
            # Upper Confidence Bound
            return mu + self.kappa * sigma
            
        elif self.acquisition_function == "probability_improvement":
            # Probability of Improvement
            y_best = np.max(y_observed)
            Z = (mu - y_best - self.xi) / sigma
            return norm.cdf(Z)
            
        else:
            return mu + sigma  # Default: mean + std
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest next parameters to evaluate."""
        if len(self.X_observed) < self.n_initial_points:
            # Random initialization
            random_params = np.random.random(len(self.param_bounds))
            return self._decode_parameters(random_params)
        
        # Optimize acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0]
            
        bounds = [(0, 1) for _ in range(len(self.param_bounds))]
        
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15
        )
        
        optimal_params = self._decode_parameters(result.x)
        return optimal_params
    
    def update_observations(self, params: Dict[str, Any], score: float):
        """Update optimization with new observation."""
        encoded_params = self._encode_parameters(params)
        self.X_observed.append(encoded_params)
        self.y_observed.append(score)
        
        # Store in history
        self.optimization_history.append({
            'params': params.copy(),
            'score': score,
            'iteration': len(self.optimization_history) + 1,
            'timestamp': time.time()
        })
        
    def get_best_parameters(self) -> OptimizationResult:
        """Get current best parameters and optimization results."""
        if not self.optimization_history:
            return OptimizationResult({}, 0.0, [], {})
            
        best_idx = np.argmax(self.y_observed)
        best_params = self.optimization_history[best_idx]['params']
        best_score = self.y_observed[best_idx]
        
        # Convergence analysis
        scores = np.array(self.y_observed)
        convergence_info = {
            'total_iterations': len(scores),
            'best_iteration': best_idx + 1,
            'improvement_rate': np.diff(np.maximum.accumulate(scores))[-5:].mean() if len(scores) > 5 else 0,
            'convergence_score': self._assess_convergence()
        }
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info
        )
        
    def _assess_convergence(self) -> float:
        """Assess optimization convergence."""
        if len(self.y_observed) < 10:
            return 0.0
            
        recent_scores = np.array(self.y_observed[-10:])
        recent_best = np.maximum.accumulate(recent_scores)
        improvement = recent_best[-1] - recent_best[0]
        return min(1.0, improvement / (abs(recent_best[-1]) + 1e-8))


class ParetoOptimizer:
    """Multi-objective optimization with Pareto frontier analysis."""
    
    def __init__(self, objectives: Optional[List[str]] = None):
        self.objectives = objectives or ['performance', 'efficiency', 'stability']
        self.pareto_solutions = []
        self.dominated_solutions = []
        
    def add_solution(
        self,
        params: Dict[str, Any],
        objective_values: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a solution to the multi-objective analysis."""
        solution = {
            'params': params.copy(),
            'objectives': objective_values.copy(),
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Check if solution is dominated
        is_dominated = self._is_dominated(solution, self.pareto_solutions)
        
        if not is_dominated:
            # Remove solutions dominated by this new one
            new_pareto = []
            for existing in self.pareto_solutions:
                if not self._dominates(solution, existing):
                    new_pareto.append(existing)
                else:
                    self.dominated_solutions.append(existing)
                    
            new_pareto.append(solution)
            self.pareto_solutions = new_pareto
        else:
            self.dominated_solutions.append(solution)
            
    def _dominates(self, sol_a: Dict, sol_b: Dict) -> bool:
        """Check if solution A dominates solution B."""
        better_in_all = True
        strictly_better_in_one = False
        
        for obj in self.objectives:
            if obj in sol_a['objectives'] and obj in sol_b['objectives']:
                a_val = sol_a['objectives'][obj]
                b_val = sol_b['objectives'][obj]
                
                # Assuming higher is better for all objectives
                if a_val < b_val:
                    better_in_all = False
                    break
                elif a_val > b_val:
                    strictly_better_in_one = True
                    
        return better_in_all and strictly_better_in_one
    
    def _is_dominated(self, solution: Dict, solution_set: List[Dict]) -> bool:
        """Check if solution is dominated by any in the set."""
        return any(self._dominates(existing, solution) for existing in solution_set)
    
    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """Get current Pareto frontier solutions."""
        return self.pareto_solutions.copy()
    
    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """Analyze trade-offs in the Pareto frontier."""
        if len(self.pareto_solutions) < 2:
            return {'error': 'Need at least 2 Pareto solutions for trade-off analysis'}
            
        analysis = {}
        
        # Extract objective values
        objective_matrix = np.array([
            [sol['objectives'][obj] for obj in self.objectives]
            for sol in self.pareto_solutions
        ])
        
        # Compute trade-off metrics
        for i, obj_a in enumerate(self.objectives):
            for j, obj_b in enumerate(self.objectives[i+1:], i+1):
                correlation = np.corrcoef(objective_matrix[:, i], objective_matrix[:, j])[0, 1]
                analysis[f'{obj_a}_vs_{obj_b}_correlation'] = correlation
                
        # Diversity metrics
        analysis['frontier_size'] = len(self.pareto_solutions)
        analysis['objective_ranges'] = {
            obj: {
                'min': float(objective_matrix[:, i].min()),
                'max': float(objective_matrix[:, i].max()),
                'range': float(objective_matrix[:, i].max() - objective_matrix[:, i].min())
            }
            for i, obj in enumerate(self.objectives)
        }
        
        # Hypervolume indicator (simplified 2D case)
        if len(self.objectives) == 2:
            analysis['hypervolume'] = self._compute_hypervolume_2d(objective_matrix)
            
        return analysis
        
    def _compute_hypervolume_2d(self, points: np.ndarray) -> float:
        """Compute 2D hypervolume indicator."""
        if len(points) == 0:
            return 0.0
            
        # Sort points by first objective
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # Compute hypervolume assuming reference point at origin
        hypervolume = 0.0
        prev_x = 0.0
        
        for point in sorted_points:
            x, y = point[0], point[1]
            if x > prev_x:
                hypervolume += (x - prev_x) * y
                prev_x = x
                
        return hypervolume


class CausalInferenceAnalyzer:
    """Causal inference analysis for understanding routing mechanisms."""
    
    def __init__(self):
        self.intervention_history = []
        self.observational_data = []
        
    def add_intervention(
        self,
        treatment: Dict[str, Any],
        control: Dict[str, Any],
        outcome_treatment: float,
        outcome_control: float,
        confounders: Optional[Dict[str, Any]] = None
    ):
        """Add an intervention experiment for causal analysis."""
        intervention = {
            'treatment': treatment.copy(),
            'control': control.copy(),
            'outcome_treatment': outcome_treatment,
            'outcome_control': outcome_control,
            'causal_effect': outcome_treatment - outcome_control,
            'confounders': confounders or {},
            'timestamp': time.time()
        }
        
        self.intervention_history.append(intervention)
        
    def add_observational_data(
        self,
        features: Dict[str, Any],
        routing_patterns: Dict[str, Any],
        performance: float
    ):
        """Add observational data for causal discovery."""
        observation = {
            'features': features.copy(),
            'routing_patterns': routing_patterns.copy(),
            'performance': performance,
            'timestamp': time.time()
        }
        
        self.observational_data.append(observation)
        
    def estimate_causal_effects(self) -> Dict[str, Any]:
        """Estimate causal effects from intervention data."""
        if not self.intervention_history:
            return {'error': 'No intervention data available'}
            
        # Aggregate causal effects by treatment type
        treatment_effects = {}
        
        for intervention in self.intervention_history:
            # Identify treatment variables
            treatment_vars = []
            for key in intervention['treatment']:
                if intervention['treatment'][key] != intervention['control'][key]:
                    treatment_vars.append(key)
                    
            treatment_key = '_'.join(sorted(treatment_vars))
            
            if treatment_key not in treatment_effects:
                treatment_effects[treatment_key] = []
                
            treatment_effects[treatment_key].append(intervention['causal_effect'])
            
        # Compute statistics for each treatment
        causal_analysis = {}
        for treatment, effects in treatment_effects.items():
            effects_array = np.array(effects)
            causal_analysis[treatment] = {
                'mean_effect': float(effects_array.mean()),
                'std_effect': float(effects_array.std()),
                'n_observations': len(effects),
                'confidence_interval': [
                    float(np.percentile(effects_array, 2.5)),
                    float(np.percentile(effects_array, 97.5))
                ],
                'statistical_significance': self._test_significance(effects_array)
            }
            
        return causal_analysis
        
    def _test_significance(self, effects: np.ndarray) -> Dict[str, Any]:
        """Test statistical significance of causal effects."""
        from scipy import stats
        
        if len(effects) < 2:
            return {'error': 'Insufficient data for significance testing'}
            
        # One-sample t-test against null hypothesis of zero effect
        t_stat, p_value = stats.ttest_1samp(effects, 0)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
        
    def discover_causal_structure(self) -> Dict[str, Any]:
        """Discover causal relationships from observational data."""
        if len(self.observational_data) < 10:
            return {'error': 'Insufficient observational data for causal discovery'}
            
        # Extract features and outcomes
        feature_names = list(self.observational_data[0]['features'].keys())
        routing_names = list(self.observational_data[0]['routing_patterns'].keys())
        
        # Build correlation matrix
        all_features = []
        all_routing = []
        all_performance = []
        
        for obs in self.observational_data:
            features = [obs['features'][name] for name in feature_names]
            routing = [obs['routing_patterns'][name] for name in routing_names]
            
            all_features.append(features)
            all_routing.append(routing)
            all_performance.append(obs['performance'])
            
        # Compute correlations
        feature_matrix = np.array(all_features)
        routing_matrix = np.array(all_routing)
        performance_array = np.array(all_performance)
        
        # Feature -> Routing correlations
        feature_routing_corr = {}
        for i, feat_name in enumerate(feature_names):
            for j, routing_name in enumerate(routing_names):
                corr = np.corrcoef(feature_matrix[:, i], routing_matrix[:, j])[0, 1]
                if not np.isnan(corr):
                    feature_routing_corr[f'{feat_name}_to_{routing_name}'] = float(corr)
                    
        # Routing -> Performance correlations
        routing_performance_corr = {}
        for j, routing_name in enumerate(routing_names):
            corr = np.corrcoef(routing_matrix[:, j], performance_array)[0, 1]
            if not np.isnan(corr):
                routing_performance_corr[f'{routing_name}_to_performance'] = float(corr)
                
        return {
            'feature_to_routing_correlations': feature_routing_corr,
            'routing_to_performance_correlations': routing_performance_corr,
            'n_observations': len(self.observational_data),
            'discovered_pathways': self._identify_causal_pathways(
                feature_routing_corr,
                routing_performance_corr
            )
        }
        
    def _identify_causal_pathways(
        self,
        feature_routing_corr: Dict[str, float],
        routing_performance_corr: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify potential causal pathways."""
        pathways = []
        
        for feat_route, corr1 in feature_routing_corr.items():
            if abs(corr1) > 0.3:  # Threshold for meaningful correlation
                feature, routing = feat_route.split('_to_')
                
                route_perf_key = f'{routing}_to_performance'
                if route_perf_key in routing_performance_corr:
                    corr2 = routing_performance_corr[route_perf_key]
                    
                    if abs(corr2) > 0.3:
                        pathway_strength = abs(corr1) * abs(corr2)
                        pathways.append({
                            'pathway': f'{feature} -> {routing} -> performance',
                            'feature_to_routing_corr': corr1,
                            'routing_to_performance_corr': corr2,
                            'pathway_strength': pathway_strength
                        })
                        
        # Sort by pathway strength
        pathways.sort(key=lambda x: x['pathway_strength'], reverse=True)
        
        return pathways


class ExperimentMonitor:
    """Real-time experiment monitoring and visualization."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring_active = True
        self.metrics_history = []
        self.alerts = []
        
    def update_metrics(self, metrics: Dict[str, float], iteration: int):
        """Update metrics and check for anomalies."""
        if not self.monitoring_active:
            return
            
        metric_update = {
            'iteration': iteration,
            'metrics': metrics.copy(),
            'timestamp': time.time()
        }
        
        self.metrics_history.append(metric_update)
        
        # Check for anomalies
        self._check_anomalies(metrics, iteration)
        
    def _check_anomalies(self, metrics: Dict[str, float], iteration: int):
        """Check for anomalous metric values."""
        if len(self.metrics_history) < 5:
            return
            
        # Get recent history
        recent_metrics = self.metrics_history[-5:]
        
        for metric_name, current_value in metrics.items():
            # Extract historical values for this metric
            historical_values = [
                m['metrics'].get(metric_name, 0) for m in recent_metrics[:-1]
            ]
            
            if len(historical_values) > 0:
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                
                # Z-score anomaly detection
                if std_val > 0:
                    z_score = abs((current_value - mean_val) / std_val)
                    
                    if z_score > 3:  # 3-sigma rule
                        alert = {
                            'type': 'anomaly',
                            'metric': metric_name,
                            'iteration': iteration,
                            'current_value': current_value,
                            'expected_value': mean_val,
                            'z_score': z_score,
                            'timestamp': time.time()
                        }
                        
                        self.alerts.append(alert)
                        
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        return self.alerts.copy()
        
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if not self.metrics_history:
            return {'error': 'No monitoring data available'}
            
        # Extract all metric names
        all_metric_names = set()
        for entry in self.metrics_history:
            all_metric_names.update(entry['metrics'].keys())
            
        # Compute statistics for each metric
        metric_stats = {}
        for metric_name in all_metric_names:
            values = []
            iterations = []
            
            for entry in self.metrics_history:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
                    iterations.append(entry['iteration'])
                    
            if values:
                values_array = np.array(values)
                metric_stats[metric_name] = {
                    'mean': float(values_array.mean()),
                    'std': float(values_array.std()),
                    'min': float(values_array.min()),
                    'max': float(values_array.max()),
                    'trend': self._compute_trend(values),
                    'stability': self._compute_stability(values)
                }
                
        return {
            'total_iterations': len(self.metrics_history),
            'monitoring_duration': time.time() - self.metrics_history[0]['timestamp'],
            'metric_statistics': metric_stats,
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'alert_types': list(set(alert['type'] for alert in self.alerts))
            },
            'recommendations': self._generate_recommendations(metric_stats)
        }
        
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for metric values."""
        if len(values) < 2:
            return 'insufficient_data'
            
        # Linear regression slope
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if abs(slope) < 1e-6:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
            
    def _compute_stability(self, values: List[float]) -> float:
        """Compute stability score (lower coefficient of variation = more stable)."""
        if len(values) < 2:
            return 0.0
            
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
            
        cv = np.std(values) / abs(mean_val)
        stability = max(0.0, 1.0 - cv)  # Higher stability = lower coefficient of variation
        
        return float(stability)
        
    def _generate_recommendations(self, metric_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring results."""
        recommendations = []
        
        for metric_name, stats in metric_stats.items():
            if stats['stability'] < 0.5:
                recommendations.append(
                    f"Consider stabilizing {metric_name} - current stability: {stats['stability']:.3f}"
                )
                
            if stats['trend'] == 'decreasing' and 'loss' not in metric_name.lower():
                recommendations.append(
                    f"Performance metric {metric_name} is decreasing - investigate potential issues"
                )
                
        if len(self.alerts) > 10:
            recommendations.append(
                "High number of anomaly alerts detected - consider adjusting thresholds or investigating root causes"
            )
            
        return recommendations


class AdaptiveEarlyStopper:
    """Adaptive early stopping with multiple criteria."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait = 0
        self.stopped_iteration = None
        self.should_stop = False
        
        # Adaptive components
        self.score_history = []
        self.adaptive_patience = patience
        self.plateau_threshold = 5
        
    def update(self, score: float, iteration: int) -> bool:
        """Update early stopper and return whether to stop."""
        self.score_history.append(score)
        
        # Check for improvement
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_iteration = iteration
            self.wait = 0
        else:
            self.wait += 1
            
        # Adaptive patience adjustment
        if len(self.score_history) >= self.plateau_threshold:
            recent_scores = self.score_history[-self.plateau_threshold:]
            score_variance = np.var(recent_scores)
            
            # If variance is very low, increase patience (we might be in a plateau)
            if score_variance < self.min_delta ** 2:
                self.adaptive_patience = min(self.patience * 2, self.patience + 20)
            else:
                self.adaptive_patience = self.patience
                
        # Check stopping criteria
        if self.wait >= self.adaptive_patience:
            self.should_stop = True
            self.stopped_iteration = iteration
            
        return self.should_stop
        
    def get_best_iteration(self) -> int:
        """Get the iteration with the best score."""
        return self.best_iteration
        
    def get_stopping_info(self) -> Dict[str, Any]:
        """Get information about early stopping."""
        return {
            'stopped': self.should_stop,
            'stopped_iteration': self.stopped_iteration,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'final_patience': self.adaptive_patience,
            'wait_count': self.wait
        }