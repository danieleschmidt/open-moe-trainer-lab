#!/usr/bin/env python3
"""
Self-Improving Patterns for MoE Trainer Lab
Autonomous learning, adaptation, and optimization capabilities
"""

import os
import json
import time
import math
import random
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import asyncio
from collections import defaultdict, deque

# Mock imports for advanced dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data): 
            if not data: return 0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val)**2 for x in data) / len(data))**0.5
        class random_np:
            @staticmethod
            def normal(loc=0, scale=1, size=None): 
                import random as rand
                return [rand.gauss(loc, scale) for _ in range(size if size else 1)]
        random = random_np

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Learning strategies for self-improvement"""
    REINFORCEMENT = "reinforcement"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    META_LEARNING = "meta_learning"

class AdaptationTrigger(Enum):
    """Triggers for adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    WORKLOAD_CHANGE = "workload_change"
    ERROR_RATE_INCREASE = "error_rate_increase"
    RESOURCE_CONSTRAINT = "resource_constraint"
    SCHEDULED_OPTIMIZATION = "scheduled_optimization"
    EXTERNAL_FEEDBACK = "external_feedback"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningExperience:
    """Learning experience record"""
    experiment_id: str
    strategy: LearningStrategy
    state_before: Dict[str, Any]
    action_taken: Dict[str, Any]
    state_after: Dict[str, Any]
    reward: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdaptationEvent:
    """Adaptation event record"""
    event_id: str
    trigger: AdaptationTrigger
    trigger_data: Dict[str, Any]
    adaptations_applied: List[Dict[str, Any]]
    performance_impact: Dict[str, float]
    timestamp: float
    success: bool

class ExperienceReplay:
    """Experience replay buffer for learning from past experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences: deque = deque(maxlen=max_size)
        self.experience_index: Dict[str, List[int]] = defaultdict(list)
    
    def add_experience(self, experience: LearningExperience):
        """Add learning experience to replay buffer"""
        index = len(self.experiences)
        self.experiences.append(experience)
        
        # Index by strategy for efficient retrieval
        self.experience_index[experience.strategy.value].append(index)
    
    def sample_experiences(self, strategy: Optional[LearningStrategy] = None, 
                          count: int = 10) -> List[LearningExperience]:
        """Sample experiences for learning"""
        if strategy:
            indices = self.experience_index.get(strategy.value, [])
            if not indices:
                return []
            
            sample_indices = random.sample(
                indices, min(count, len(indices))
            )
            return [self.experiences[i] for i in sample_indices]
        else:
            if len(self.experiences) == 0:
                return []
            
            sample_indices = random.sample(
                range(len(self.experiences)), 
                min(count, len(self.experiences))
            )
            return [self.experiences[i] for i in sample_indices]
    
    def get_best_experiences(self, strategy: LearningStrategy, 
                           count: int = 5) -> List[LearningExperience]:
        """Get best experiences for a strategy based on reward"""
        experiences = self.sample_experiences(strategy, count * 3)  # Get larger sample
        if not experiences:
            return []
        
        # Sort by reward and return top experiences
        sorted_experiences = sorted(experiences, key=lambda x: x.reward, reverse=True)
        return sorted_experiences[:count]

class PerformancePredictor:
    """Predicts performance outcomes for different configurations"""
    
    def __init__(self):
        self.prediction_models: Dict[str, Dict[str, Any]] = {}
        self.historical_data: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        self.confidence_threshold = 0.7
    
    def train_predictor(self, metric_name: str, training_data: List[Tuple[Dict, float]]):
        """Train prediction model for a specific metric"""
        if len(training_data) < 5:
            logger.warning(f"Insufficient data to train predictor for {metric_name}")
            return False
        
        # Simple linear regression-like model
        model_data = {
            "metric_name": metric_name,
            "sample_count": len(training_data),
            "feature_importance": self._calculate_feature_importance(training_data),
            "baseline_performance": np.mean([y for _, y in training_data]) if NUMPY_AVAILABLE else sum(y for _, y in training_data) / len(training_data),
            "trained_timestamp": time.time()
        }
        
        self.prediction_models[metric_name] = model_data
        logger.info(f"Trained predictor for {metric_name} with {len(training_data)} samples")
        return True
    
    def _calculate_feature_importance(self, training_data: List[Tuple[Dict, float]]) -> Dict[str, float]:
        """Calculate simple feature importance"""
        if not training_data:
            return {}
        
        # Extract all feature keys
        all_features = set()
        for features, _ in training_data:
            all_features.update(features.keys())
        
        # Calculate correlation-like importance
        importance = {}
        for feature in all_features:
            feature_values = []
            target_values = []
            
            for features, target in training_data:
                if feature in features and isinstance(features[feature], (int, float)):
                    feature_values.append(features[feature])
                    target_values.append(target)
            
            if len(feature_values) > 1:
                # Simple correlation calculation
                if NUMPY_AVAILABLE:
                    correlation = abs(np.corrcoef(feature_values, target_values)[0, 1]) if len(feature_values) > 1 else 0
                else:
                    # Manual correlation calculation
                    n = len(feature_values)
                    sum_x = sum(feature_values)
                    sum_y = sum(target_values)
                    sum_xy = sum(x * y for x, y in zip(feature_values, target_values))
                    sum_x2 = sum(x * x for x in feature_values)
                    sum_y2 = sum(y * y for y in target_values)
                    
                    numerator = n * sum_xy - sum_x * sum_y
                    denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5
                    correlation = abs(numerator / denominator) if denominator != 0 else 0
                
                importance[feature] = correlation
        
        return importance
    
    def predict_performance(self, config: Dict[str, Any], 
                          metric_name: str) -> Tuple[float, float]:
        """Predict performance for given configuration"""
        if metric_name not in self.prediction_models:
            return 0.0, 0.0  # No prediction available
        
        model = self.prediction_models[metric_name]
        baseline = model["baseline_performance"]
        feature_importance = model["feature_importance"]
        
        # Simple prediction based on feature importance
        adjustment = 0.0
        confidence = 0.5
        
        for feature, value in config.items():
            if feature in feature_importance and isinstance(value, (int, float)):
                importance = feature_importance[feature]
                # Simple scaling factor
                adjustment += importance * (value - 1.0) * 0.1
        
        predicted_value = baseline + adjustment
        confidence = min(0.9, 0.3 + sum(feature_importance.values()) / len(feature_importance) if feature_importance else 0.3)
        
        return predicted_value, confidence

class AdaptiveOptimizer:
    """Adaptive optimization engine that learns from experience"""
    
    def __init__(self):
        self.experience_replay = ExperienceReplay()
        self.performance_predictor = PerformancePredictor()
        self.adaptation_history: List[AdaptationEvent] = []
        self.optimization_parameters: Dict[str, Any] = {
            "learning_rate": 0.01,
            "exploration_factor": 0.2,
            "adaptation_threshold": 0.05,  # 5% performance change threshold
            "confidence_threshold": 0.6,
            "max_adaptations_per_cycle": 3
        }
        self.current_strategies: Dict[str, LearningStrategy] = {}
    
    def register_performance_metrics(self, metrics: List[PerformanceMetric]):
        """Register performance metrics for analysis"""
        for metric in metrics:
            # Update predictor with new data
            config_context = metric.context.get("config", {})
            if config_context:
                training_point = (config_context, metric.value)
                self.performance_predictor.historical_data.append(training_point)
                
                # Retrain predictor if enough data
                recent_data = [
                    (ctx["config"], val) 
                    for ctx, val in self.performance_predictor.historical_data[-50:]
                    if "config" in ctx
                ]
                
                if len(recent_data) >= 10:
                    self.performance_predictor.train_predictor(metric.name, recent_data)
    
    def detect_adaptation_triggers(self, current_metrics: List[PerformanceMetric], 
                                 historical_metrics: List[PerformanceMetric]) -> List[AdaptationTrigger]:
        """Detect conditions that trigger adaptation"""
        triggers = []
        
        if not historical_metrics or not current_metrics:
            return triggers
        
        # Group metrics by name
        current_by_name = {m.name: m for m in current_metrics}
        historical_by_name = defaultdict(list)
        for m in historical_metrics:
            historical_by_name[m.name].append(m.value)
        
        for metric_name, current_metric in current_by_name.items():
            if metric_name not in historical_by_name:
                continue
            
            historical_values = historical_by_name[metric_name]
            if len(historical_values) < 3:
                continue
            
            historical_mean = sum(historical_values) / len(historical_values)
            
            # Check for performance degradation
            if "latency" in metric_name.lower() or "error" in metric_name.lower():
                # Higher is worse
                if current_metric.value > historical_mean * (1 + self.optimization_parameters["adaptation_threshold"]):
                    triggers.append(AdaptationTrigger.PERFORMANCE_DEGRADATION)
            else:
                # Lower is worse (throughput, accuracy, etc.)
                if current_metric.value < historical_mean * (1 - self.optimization_parameters["adaptation_threshold"]):
                    triggers.append(AdaptationTrigger.PERFORMANCE_DEGRADATION)
        
        return list(set(triggers))  # Remove duplicates
    
    def generate_adaptation_strategies(self, triggers: List[AdaptationTrigger], 
                                     current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate potential adaptation strategies"""
        strategies = []
        
        for trigger in triggers:
            if trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
                # Learn from best past experiences
                best_experiences = self.experience_replay.get_best_experiences(
                    LearningStrategy.REINFORCEMENT, 5
                )
                
                if best_experiences:
                    # Extract successful adaptations
                    for exp in best_experiences:
                        adaptation = {
                            "strategy": LearningStrategy.REINFORCEMENT.value,
                            "action": exp.action_taken,
                            "expected_reward": exp.reward,
                            "confidence": 0.7,
                            "trigger": trigger.value
                        }
                        strategies.append(adaptation)
                
                # Also generate exploratory strategies
                exploration_strategy = self._generate_exploration_strategy(current_config)
                if exploration_strategy:
                    strategies.append(exploration_strategy)
        
        return strategies[:self.optimization_parameters["max_adaptations_per_cycle"]]
    
    def _generate_exploration_strategy(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploratory adaptation strategy"""
        # Simple parameter perturbation strategy
        exploration_actions = {}
        
        for param, value in current_config.items():
            if isinstance(value, (int, float)) and param in ["learning_rate", "batch_size", "num_experts", "top_k"]:
                # Add small random perturbation
                perturbation = random.uniform(-0.1, 0.1) * self.optimization_parameters["exploration_factor"]
                if isinstance(value, int):
                    new_value = max(1, int(value * (1 + perturbation)))
                else:
                    new_value = max(0.001, value * (1 + perturbation))
                
                exploration_actions[param] = new_value
        
        if exploration_actions:
            return {
                "strategy": LearningStrategy.EVOLUTIONARY.value,
                "action": exploration_actions,
                "expected_reward": 0.0,  # Unknown
                "confidence": 0.3,  # Low confidence for exploration
                "trigger": "exploration"
            }
        
        return None
    
    def apply_adaptations(self, strategies: List[Dict[str, Any]], 
                         current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation strategies and return new configuration"""
        if not strategies:
            return current_config
        
        # Sort by confidence and expected reward
        sorted_strategies = sorted(
            strategies, 
            key=lambda s: (s.get("confidence", 0) * s.get("expected_reward", 0)),
            reverse=True
        )
        
        # Apply best strategy
        best_strategy = sorted_strategies[0]
        new_config = current_config.copy()
        
        for param, value in best_strategy["action"].items():
            if param in new_config:
                new_config[param] = value
        
        # Record adaptation
        adaptation_event = AdaptationEvent(
            event_id=hashlib.sha256(f"adapt_{time.time()}".encode()).hexdigest()[:16],
            trigger=AdaptationTrigger(best_strategy["trigger"]) if best_strategy["trigger"] in [t.value for t in AdaptationTrigger] else AdaptationTrigger.SCHEDULED_OPTIMIZATION,
            trigger_data={"confidence": best_strategy["confidence"]},
            adaptations_applied=[best_strategy],
            performance_impact={},  # Will be filled later
            timestamp=time.time(),
            success=True  # Assumed for now
        )
        
        self.adaptation_history.append(adaptation_event)
        
        return new_config
    
    def learn_from_experience(self, config_before: Dict[str, Any], 
                            config_after: Dict[str, Any],
                            performance_before: List[PerformanceMetric],
                            performance_after: List[PerformanceMetric],
                            strategy_used: LearningStrategy):
        """Learn from adaptation experience"""
        # Calculate reward based on performance change
        reward = self._calculate_reward(performance_before, performance_after)
        
        experience = LearningExperience(
            experiment_id=hashlib.sha256(f"exp_{time.time()}".encode()).hexdigest()[:16],
            strategy=strategy_used,
            state_before=config_before,
            action_taken=self._diff_configs(config_before, config_after),
            state_after=config_after,
            reward=reward,
            timestamp=time.time(),
            metadata={
                "performance_before": [asdict(m) for m in performance_before],
                "performance_after": [asdict(m) for m in performance_after]
            }
        )
        
        self.experience_replay.add_experience(experience)
        
        # Update adaptation history with performance impact
        if self.adaptation_history:
            latest_adaptation = self.adaptation_history[-1]
            latest_adaptation.performance_impact = {
                "reward": reward,
                "improvement": reward > 0
            }
            latest_adaptation.success = reward > 0
        
        logger.info(f"Learned from experience: reward={reward:.3f}, strategy={strategy_used.value}")
    
    def _calculate_reward(self, before: List[PerformanceMetric], 
                         after: List[PerformanceMetric]) -> float:
        """Calculate reward based on performance improvement"""
        if not before or not after:
            return 0.0
        
        reward = 0.0
        compared_metrics = 0
        
        before_by_name = {m.name: m.value for m in before}
        after_by_name = {m.name: m.value for m in after}
        
        for metric_name in before_by_name:
            if metric_name in after_by_name:
                before_val = before_by_name[metric_name]
                after_val = after_by_name[metric_name]
                
                if before_val == 0:
                    continue
                
                # Calculate relative improvement
                if "latency" in metric_name.lower() or "error" in metric_name.lower():
                    # Lower is better
                    improvement = (before_val - after_val) / before_val
                else:
                    # Higher is better
                    improvement = (after_val - before_val) / before_val
                
                reward += improvement
                compared_metrics += 1
        
        return reward / compared_metrics if compared_metrics > 0 else 0.0
    
    def _diff_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate difference between configurations"""
        diff = {}
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                diff[key] = val2
        
        return diff

class SelfImprovingMoESystem:
    """Self-improving MoE system with autonomous learning and adaptation"""
    
    def __init__(self):
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.performance_history: List[PerformanceMetric] = []
        self.config_history: List[Dict[str, Any]] = []
        self.improvement_cycles = 0
        self.auto_adaptation_enabled = True
        self.learning_statistics = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "performance_improvements": 0,
            "learning_rate": 0.0,
            "adaptation_frequency": 0.0
        }
    
    def register_current_performance(self, metrics: List[PerformanceMetric], 
                                   config: Dict[str, Any]):
        """Register current system performance and configuration"""
        self.performance_history.extend(metrics)
        self.config_history.append({
            "config": config,
            "timestamp": time.time(),
            "metrics": [asdict(m) for m in metrics]
        })
        
        # Update optimizer
        self.adaptive_optimizer.register_performance_metrics(metrics)
        
        # Trigger adaptation if enabled
        if self.auto_adaptation_enabled and len(self.performance_history) >= 10:
            self._trigger_adaptation_cycle()
    
    def _trigger_adaptation_cycle(self):
        """Trigger an adaptation cycle"""
        if len(self.config_history) < 2:
            return
        
        current_config = self.config_history[-1]["config"]
        current_metrics = [
            PerformanceMetric(**m) for m in self.config_history[-1]["metrics"]
        ]
        
        # Get historical metrics from last 10 records
        historical_metrics = []
        for record in self.config_history[-11:-1]:  # Skip current, take last 10
            for m_data in record["metrics"]:
                historical_metrics.append(PerformanceMetric(**m_data))
        
        # Detect adaptation triggers
        triggers = self.adaptive_optimizer.detect_adaptation_triggers(
            current_metrics, historical_metrics
        )
        
        if triggers:
            logger.info(f"Adaptation triggers detected: {[t.value for t in triggers]}")
            
            # Generate and apply adaptations
            strategies = self.adaptive_optimizer.generate_adaptation_strategies(
                triggers, current_config
            )
            
            if strategies:
                new_config = self.adaptive_optimizer.apply_adaptations(
                    strategies, current_config
                )
                
                self._apply_configuration_changes(current_config, new_config)
                self.improvement_cycles += 1
                self.learning_statistics["total_adaptations"] += 1
    
    def _apply_configuration_changes(self, old_config: Dict[str, Any], 
                                   new_config: Dict[str, Any]):
        """Apply configuration changes (simulation)"""
        changes = self.adaptive_optimizer._diff_configs(old_config, new_config)
        if changes:
            logger.info(f"Applying configuration changes: {changes}")
            
            # Simulate performance measurement after change
            simulated_new_metrics = self._simulate_performance_after_change(
                old_config, new_config
            )
            
            # Learn from experience
            old_metrics = [
                PerformanceMetric(**m) for m in self.config_history[-1]["metrics"]
            ]
            
            self.adaptive_optimizer.learn_from_experience(
                old_config, new_config, old_metrics, simulated_new_metrics,
                LearningStrategy.REINFORCEMENT
            )
            
            # Update statistics
            reward = self.adaptive_optimizer._calculate_reward(
                old_metrics, simulated_new_metrics
            )
            
            if reward > 0:
                self.learning_statistics["successful_adaptations"] += 1
                self.learning_statistics["performance_improvements"] += 1
            
            # Update learning rate
            self.learning_statistics["learning_rate"] = (
                self.learning_statistics["successful_adaptations"] / 
                max(1, self.learning_statistics["total_adaptations"])
            )
    
    def _simulate_performance_after_change(self, old_config: Dict[str, Any],
                                         new_config: Dict[str, Any]) -> List[PerformanceMetric]:
        """Simulate performance metrics after configuration change"""
        # Simple simulation based on configuration changes
        base_latency = 100.0  # ms
        base_throughput = 1000.0  # req/sec
        base_accuracy = 0.95
        
        # Apply config-based modifications
        latency_factor = 1.0
        throughput_factor = 1.0
        accuracy_factor = 1.0
        
        for param, value in new_config.items():
            if param == "batch_size":
                throughput_factor *= (value / old_config.get("batch_size", 8))
                latency_factor *= (old_config.get("batch_size", 8) / value) * 0.8
            elif param == "num_experts":
                accuracy_factor *= (1 + (value - old_config.get("num_experts", 4)) * 0.02)
                latency_factor *= (value / old_config.get("num_experts", 4))
            elif param == "learning_rate":
                accuracy_factor *= (1 + random.uniform(-0.1, 0.1))
        
        # Add some randomness
        latency_factor *= random.uniform(0.9, 1.1)
        throughput_factor *= random.uniform(0.9, 1.1)
        accuracy_factor *= random.uniform(0.98, 1.02)
        
        return [
            PerformanceMetric(
                name="latency_ms",
                value=base_latency * latency_factor,
                timestamp=time.time()
            ),
            PerformanceMetric(
                name="throughput_req_per_sec",
                value=base_throughput * throughput_factor,
                timestamp=time.time()
            ),
            PerformanceMetric(
                name="accuracy",
                value=min(1.0, base_accuracy * accuracy_factor),
                timestamp=time.time()
            )
        ]
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights into the learning process"""
        recent_experiences = self.adaptive_optimizer.experience_replay.sample_experiences(
            count=20
        )
        
        insights = {
            "learning_statistics": self.learning_statistics,
            "improvement_cycles": self.improvement_cycles,
            "total_experiences": len(self.adaptive_optimizer.experience_replay.experiences),
            "adaptation_history_count": len(self.adaptive_optimizer.adaptation_history),
            "recent_experiences_summary": {
                "count": len(recent_experiences),
                "avg_reward": np.mean([exp.reward for exp in recent_experiences]) if recent_experiences and NUMPY_AVAILABLE else 0,
                "strategies_used": list(set(exp.strategy.value for exp in recent_experiences))
            },
            "performance_trends": self._analyze_performance_trends(),
            "top_performing_configurations": self._get_top_configurations(),
            "learning_recommendations": self._generate_learning_recommendations()
        }
        
        return insights
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.config_history) < 5:
            return {"status": "insufficient_data"}
        
        recent_records = self.config_history[-10:]
        trends = {}
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for record in recent_records:
            for metric_data in record["metrics"]:
                metrics_by_name[metric_data["name"]].append(metric_data["value"])
        
        for metric_name, values in metrics_by_name.items():
            if len(values) >= 3:
                # Simple trend analysis
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if "latency" in metric_name.lower() or "error" in metric_name.lower():
                    improvement = (first_avg - second_avg) / first_avg if first_avg != 0 else 0
                else:
                    improvement = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
                
                trends[metric_name] = {
                    "improvement_percentage": improvement * 100,
                    "trend": "improving" if improvement > 0.05 else "degrading" if improvement < -0.05 else "stable"
                }
        
        return trends
    
    def _get_top_configurations(self, count: int = 3) -> List[Dict[str, Any]]:
        """Get top performing configurations"""
        if len(self.config_history) < count:
            return self.config_history
        
        # Score configurations based on multiple metrics
        scored_configs = []
        
        for record in self.config_history:
            score = 0.0
            metric_count = 0
            
            for metric_data in record["metrics"]:
                metric_name = metric_data["name"]
                value = metric_data["value"]
                
                # Normalize and score based on metric type
                if "latency" in metric_name.lower() or "error" in metric_name.lower():
                    # Lower is better - invert score
                    score += 1.0 / (1.0 + value / 100.0)
                else:
                    # Higher is better
                    score += value if value <= 1.0 else math.log(1 + value)
                
                metric_count += 1
            
            if metric_count > 0:
                average_score = score / metric_count
                scored_configs.append((average_score, record))
        
        # Sort by score and return top configurations
        sorted_configs = sorted(scored_configs, key=lambda x: x[0], reverse=True)
        return [record for _, record in sorted_configs[:count]]
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for improving the learning process"""
        recommendations = []
        
        if self.learning_statistics["total_adaptations"] > 0:
            success_rate = self.learning_statistics["learning_rate"]
            
            if success_rate < 0.3:
                recommendations.append("Consider reducing exploration factor to focus on exploitation")
                recommendations.append("Increase confidence threshold for adaptations")
            elif success_rate > 0.8:
                recommendations.append("Consider increasing exploration factor to discover new optima")
                recommendations.append("Expand the adaptation search space")
        
        if len(self.adaptive_optimizer.experience_replay.experiences) < 100:
            recommendations.append("Continue collecting experience data for better learning")
        
        if self.improvement_cycles < 5:
            recommendations.append("Allow more adaptation cycles to establish learning patterns")
        
        return recommendations

async def demo_self_improving_patterns():
    """Demonstrate self-improving patterns"""
    print("🧬 Self-Improving MoE System Demo")
    print("="*50)
    
    # Initialize self-improving system
    system = SelfImprovingMoESystem()
    
    # Simulate training cycles with different configurations and performance
    configurations = [
        {"batch_size": 8, "num_experts": 4, "learning_rate": 0.001, "top_k": 2},
        {"batch_size": 16, "num_experts": 4, "learning_rate": 0.001, "top_k": 2},
        {"batch_size": 16, "num_experts": 8, "learning_rate": 0.001, "top_k": 2},
        {"batch_size": 32, "num_experts": 8, "learning_rate": 0.002, "top_k": 3},
    ]
    
    print("🔄 Simulating Training Cycles with Learning:")
    
    for cycle in range(15):  # Run multiple cycles
        # Use configurations in sequence, then let system adapt
        if cycle < len(configurations):
            config = configurations[cycle]
        else:
            # System will auto-adapt based on learning
            config = configurations[-1]  # Base config
        
        # Simulate performance metrics with some variance and trends
        base_performance = {
            "latency_ms": 120.0 - cycle * 2,  # Improving over time
            "throughput_req_per_sec": 800.0 + cycle * 30,  # Improving
            "accuracy": 0.92 + cycle * 0.005,  # Slight improvement
        }
        
        # Add some noise and configuration-based effects
        metrics = []
        for name, base_value in base_performance.items():
            noise_factor = random.uniform(0.9, 1.1)
            
            # Configuration effects
            if name == "throughput_req_per_sec":
                config_effect = config["batch_size"] / 8.0  # Batch size affects throughput
            elif name == "latency_ms":
                config_effect = 8.0 / config["batch_size"]  # Inverse for latency
            else:
                config_effect = 1.0 + (config["num_experts"] - 4) * 0.01  # Experts affect accuracy
            
            final_value = base_value * noise_factor * config_effect
            
            metrics.append(PerformanceMetric(
                name=name,
                value=final_value,
                timestamp=time.time(),
                context={"config": config, "cycle": cycle}
            ))
        
        # Register performance
        system.register_current_performance(metrics, config)
        
        print(f"  Cycle {cycle + 1}: Config={config}, " +
              f"Latency={metrics[0].value:.1f}ms, " +
              f"Throughput={metrics[1].value:.1f}req/s, " +
              f"Accuracy={metrics[2].value:.3f}")
        
        # Brief pause to simulate time
        await asyncio.sleep(0.1)
    
    # Get learning insights
    insights = system.get_learning_insights()
    
    print(f"\n📊 Learning Insights:")
    print(f"  Total Improvement Cycles: {insights['improvement_cycles']}")
    print(f"  Total Adaptations: {insights['learning_statistics']['total_adaptations']}")
    print(f"  Successful Adaptations: {insights['learning_statistics']['successful_adaptations']}")
    print(f"  Learning Success Rate: {insights['learning_statistics']['learning_rate']:.2%}")
    print(f"  Total Experiences: {insights['total_experiences']}")
    
    print(f"\n📈 Performance Trends:")
    for metric_name, trend_data in insights["performance_trends"].items():
        if isinstance(trend_data, dict):
            print(f"  {metric_name}: {trend_data['trend']} ({trend_data['improvement_percentage']:+.1f}%)")
    
    print(f"\n🏆 Top Performing Configurations:")
    for i, config_record in enumerate(insights["top_performing_configurations"][:3]):
        config = config_record["config"]
        print(f"  #{i+1}: batch_size={config['batch_size']}, num_experts={config['num_experts']}, lr={config['learning_rate']}")
    
    print(f"\n💡 Learning Recommendations:")
    for rec in insights["learning_recommendations"]:
        print(f"  • {rec}")
    
    # Save comprehensive results
    results = {
        "demo_timestamp": datetime.now(timezone.utc).isoformat(),
        "system_config": {
            "auto_adaptation_enabled": system.auto_adaptation_enabled,
            "improvement_cycles": system.improvement_cycles
        },
        "learning_insights": insights,
        "performance_history_sample": [
            asdict(m) for m in system.performance_history[-10:]  # Last 10 metrics
        ],
        "adaptation_history": [
            {
                "event_id": event.event_id,
                "trigger": event.trigger.value,
                "success": event.success,
                "timestamp": event.timestamp,
                "adaptations_count": len(event.adaptations_applied)
            } for event in system.adaptive_optimizer.adaptation_history
        ],
        "features_demonstrated": [
            "experience_replay_learning",
            "performance_prediction",
            "adaptive_optimization",
            "automatic_trigger_detection",
            "configuration_adaptation",
            "reward_based_learning",
            "trend_analysis",
            "learning_insights_generation"
        ]
    }
    
    with open("/root/repo/self_improving_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Self-Improving Patterns Demo completed!")
    print(f"📁 Results saved to self_improving_demo_results.json")
    
    return results

if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(demo_self_improving_patterns())
    print(f"\n🎯 Self-Improving Patterns Implementation Status: COMPLETE")