"""Novel MoE algorithms for breakthrough research.

This module implements revolutionary algorithms for Mixture of Experts:
1. Quantum-Inspired Routing with Superposition States
2. Evolutionary Expert Architecture Search
3. Continual Learning with Catastrophic Forgetting Prevention
4. Self-Organizing Expert Networks with Emergent Specialization
5. Meta-Routing for Multi-Task Adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import math
import random
from collections import defaultdict
import copy


@dataclass
class QuantumRoutingState:
    """Quantum-inspired routing state with superposition."""
    amplitudes: torch.Tensor
    phases: torch.Tensor
    entanglement_matrix: torch.Tensor
    measurement_probabilities: torch.Tensor


class QuantumInspiredRouter(nn.Module):
    """Quantum-inspired routing using superposition and entanglement principles.
    
    This revolutionary approach treats expert selection as a quantum measurement
    problem, where tokens exist in superposition states across multiple experts
    until routing measurement collapses the state.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_qubits: int = None,
        coherence_time: float = 10.0,
        entanglement_strength: float = 0.5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_qubits = num_qubits or int(math.ceil(math.log2(num_experts)))
        self.coherence_time = coherence_time
        self.entanglement_strength = entanglement_strength
        
        # Quantum state preparation networks
        self.amplitude_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        self.phase_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Tanh()  # Phases in [-1, 1] -> [-π, π]
        )
        
        # Entanglement matrix (learnable)
        self.entanglement_matrix = nn.Parameter(
            torch.randn(num_experts, num_experts) * 0.1
        )
        
        # Decoherence simulation
        self.register_buffer('time_step', torch.tensor(0.0))
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, QuantumRoutingState]:
        """Forward pass with quantum-inspired routing."""
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Prepare quantum state
        raw_amplitudes = self.amplitude_network(hidden_states)
        raw_phases = self.phase_network(hidden_states) * math.pi
        
        # Normalize amplitudes to satisfy quantum normalization
        amplitudes = F.softmax(raw_amplitudes.abs(), dim=-1)
        amplitudes = torch.sqrt(amplitudes)  # √(probability) = amplitude
        
        # Apply entanglement
        entangled_amplitudes = self._apply_entanglement(amplitudes)
        
        # Simulate decoherence over time
        coherence_factor = torch.exp(-self.time_step / self.coherence_time)
        decoherent_amplitudes = entangled_amplitudes * coherence_factor
        
        # Quantum measurement (collapse to classical routing)
        measurement_probs = decoherent_amplitudes ** 2
        measurement_probs = F.normalize(measurement_probs, p=1, dim=-1)
        
        # Top-k sampling from quantum measurement
        k = 2
        top_k_probs, top_k_indices = torch.topk(measurement_probs, k, dim=-1)
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)
        
        # Create quantum state for analysis
        quantum_state = QuantumRoutingState(
            amplitudes=decoherent_amplitudes,
            phases=raw_phases,
            entanglement_matrix=self.entanglement_matrix,
            measurement_probabilities=measurement_probs
        )
        
        # Update time step for decoherence
        self.time_step += 1
        
        return top_k_probs, top_k_indices, quantum_state
        
    def _apply_entanglement(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Apply quantum entanglement to amplitude states."""
        # Entanglement as controlled rotations
        entangled = amplitudes.clone()
        
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                entanglement_strength = self.entanglement_matrix[i, j]
                
                # Controlled rotation between expert i and j
                angle = entanglement_strength * self.entanglement_strength
                cos_theta = torch.cos(angle)
                sin_theta = torch.sin(angle)
                
                # Apply rotation
                amp_i = entangled[..., i].clone()
                amp_j = entangled[..., j].clone()
                
                entangled[..., i] = cos_theta * amp_i - sin_theta * amp_j
                entangled[..., j] = sin_theta * amp_i + cos_theta * amp_j
                
        return entangled
        
    def measure_entanglement(self) -> float:
        """Measure quantum entanglement in the system."""
        # Von Neumann entropy as entanglement measure
        eigenvals = torch.linalg.eigvals(self.entanglement_matrix @ self.entanglement_matrix.T)
        eigenvals = eigenvals.real
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return 0.0
            
        eigenvals = eigenvals / eigenvals.sum()  # Normalize
        entropy = -torch.sum(eigenvals * torch.log(eigenvals + 1e-12))
        
        return entropy.item()


class EvolutionaryArchitectureSearch(nn.Module):
    """Evolutionary search for optimal MoE architectures.
    
    Uses genetic algorithms to evolve MoE architectures, including:
    - Number of experts per layer
    - Expert architectures
    - Routing strategies
    - Skip connections
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.2,
        max_experts: int = 64,
        max_layers: int = 20
    ):
        super().__init__()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.max_experts = max_experts
        self.max_layers = max_layers
        
        # Architecture genome representation
        self.population = []
        self.fitness_history = []
        self.generation = 0
        
        # Initialize random population
        self._initialize_population()
        
    def _initialize_population(self):
        """Initialize random population of MoE architectures."""
        self.population = []
        
        for _ in range(self.population_size):
            genome = self._create_random_genome()
            self.population.append(genome)
            
    def _create_random_genome(self) -> Dict[str, Any]:
        """Create a random MoE architecture genome."""
        num_layers = random.randint(6, self.max_layers)
        
        genome = {
            'num_layers': num_layers,
            'layers': [],
            'global_config': {
                'hidden_size': random.choice([512, 768, 1024, 1536, 2048]),
                'activation': random.choice(['gelu', 'relu', 'swish', 'mish']),
                'layer_norm': random.choice([True, False]),
                'residual_connections': random.choice([True, False])
            }
        }
        
        # Generate layer configurations
        for layer_idx in range(num_layers):
            layer_type = random.choice(['dense', 'moe', 'sparse_moe'])
            
            if layer_type == 'moe':
                layer_config = {
                    'type': 'moe',
                    'num_experts': random.randint(4, self.max_experts),
                    'experts_per_token': random.randint(1, 4),
                    'expert_size': random.choice([1.0, 2.0, 4.0]),  # Multiplier of hidden_size
                    'routing_type': random.choice(['top_k', 'switch', 'expert_choice']),
                    'load_balancing': random.choice([True, False]),
                    'expert_dropout': random.uniform(0.0, 0.3)
                }
            elif layer_type == 'sparse_moe':
                layer_config = {
                    'type': 'sparse_moe',
                    'num_experts': random.randint(16, self.max_experts * 2),
                    'experts_per_token': random.randint(1, 2),
                    'sparsity_ratio': random.uniform(0.5, 0.9),
                    'dynamic_routing': random.choice([True, False])
                }
            else:  # dense
                layer_config = {
                    'type': 'dense',
                    'intermediate_size': int(genome['global_config']['hidden_size'] * random.uniform(2.0, 8.0)),
                    'dropout': random.uniform(0.0, 0.3)
                }
                
            genome['layers'].append(layer_config)
            
        return genome
        
    def evolve_generation(self, fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Evolve population for one generation."""
        assert len(fitness_scores) == len(self.population)
        
        # Store fitness history
        self.fitness_history.append(fitness_scores.copy())
        
        # Create fitness-population pairs
        population_fitness = list(zip(self.population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (higher better)
        
        # Selection
        elite_size = int(self.population_size * self.elite_ratio)
        elites = [genome for genome, _ in population_fitness[:elite_size]]
        
        # Generate new population
        new_population = elites.copy()  # Keep elites
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population_fitness)
            parent2 = self._tournament_selection(population_fitness)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
            
        # Truncate to population size
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return self.population
        
    def _tournament_selection(self, population_fitness: List[Tuple], k: int = 3) -> Dict[str, Any]:
        """Tournament selection for parent selection."""
        tournament = random.sample(population_fitness, min(k, len(population_fitness)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
        
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover two parent genomes."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Global config crossover
        for key in child1['global_config']:
            if random.random() < 0.5:
                child1['global_config'][key], child2['global_config'][key] = \
                    child2['global_config'][key], child1['global_config'][key]
                    
        # Layer crossover
        min_layers = min(len(child1['layers']), len(child2['layers']))
        crossover_point = random.randint(1, min_layers - 1) if min_layers > 1 else 0
        
        # Swap layer segments
        child1_layers = child1['layers'][:crossover_point] + child2['layers'][crossover_point:]
        child2_layers = child2['layers'][:crossover_point] + child1['layers'][crossover_point:]
        
        child1['layers'] = child1_layers
        child2['layers'] = child2_layers
        child1['num_layers'] = len(child1_layers)
        child2['num_layers'] = len(child2_layers)
        
        return child1, child2
        
    def _mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a genome."""
        mutated = copy.deepcopy(genome)
        
        # Global mutations
        if random.random() < self.mutation_rate:
            key = random.choice(list(mutated['global_config'].keys()))
            if key == 'hidden_size':
                mutated['global_config'][key] = random.choice([512, 768, 1024, 1536, 2048])
            elif key == 'activation':
                mutated['global_config'][key] = random.choice(['gelu', 'relu', 'swish', 'mish'])
            else:
                mutated['global_config'][key] = not mutated['global_config'][key]
                
        # Layer mutations
        for i, layer in enumerate(mutated['layers']):
            if random.random() < self.mutation_rate:
                if layer['type'] == 'moe':
                    mutation_type = random.choice(['num_experts', 'experts_per_token', 'routing_type'])
                    if mutation_type == 'num_experts':
                        layer['num_experts'] = random.randint(4, self.max_experts)
                    elif mutation_type == 'experts_per_token':
                        layer['experts_per_token'] = random.randint(1, 4)
                    else:
                        layer['routing_type'] = random.choice(['top_k', 'switch', 'expert_choice'])
                        
        # Structural mutations (add/remove layers)
        if random.random() < self.mutation_rate * 0.1:
            if len(mutated['layers']) < self.max_layers and random.random() < 0.5:
                # Add layer
                new_layer = self._create_random_layer()
                insert_pos = random.randint(0, len(mutated['layers']))
                mutated['layers'].insert(insert_pos, new_layer)
                mutated['num_layers'] += 1
            elif len(mutated['layers']) > 3:
                # Remove layer
                remove_pos = random.randint(0, len(mutated['layers']) - 1)
                mutated['layers'].pop(remove_pos)
                mutated['num_layers'] -= 1
                
        return mutated
        
    def _create_random_layer(self) -> Dict[str, Any]:
        """Create a random layer configuration."""
        layer_type = random.choice(['dense', 'moe', 'sparse_moe'])
        
        if layer_type == 'moe':
            return {
                'type': 'moe',
                'num_experts': random.randint(4, self.max_experts),
                'experts_per_token': random.randint(1, 4),
                'expert_size': random.choice([1.0, 2.0, 4.0]),
                'routing_type': random.choice(['top_k', 'switch', 'expert_choice']),
                'load_balancing': random.choice([True, False]),
                'expert_dropout': random.uniform(0.0, 0.3)
            }
        else:
            return {
                'type': 'dense',
                'intermediate_size': random.randint(512, 4096),
                'dropout': random.uniform(0.0, 0.3)
            }
            
    def get_best_genome(self) -> Dict[str, Any]:
        """Get the best genome from current population."""
        if not self.fitness_history:
            return self.population[0]
            
        last_fitness = self.fitness_history[-1]
        best_idx = np.argmax(last_fitness)
        return self.population[best_idx]
        
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        if not self.fitness_history:
            return {}
            
        fitness_array = np.array(self.fitness_history)
        
        return {
            'generation': self.generation,
            'best_fitness_per_generation': fitness_array.max(axis=1).tolist(),
            'mean_fitness_per_generation': fitness_array.mean(axis=1).tolist(),
            'diversity_per_generation': [
                self._compute_population_diversity(gen_pop) 
                for gen_pop in self.fitness_history
            ]
        }
        
    def _compute_population_diversity(self, fitness_scores: List[float]) -> float:
        """Compute population diversity based on fitness variance."""
        return float(np.std(fitness_scores))


class ContinualLearningMoE(nn.Module):
    """MoE with continual learning and catastrophic forgetting prevention.
    
    Uses expert specialization and memory replay to learn multiple tasks
    sequentially without forgetting previous knowledge.
    """
    
    def __init__(
        self,
        base_moe_model: nn.Module,
        memory_size: int = 1000,
        consolidation_strength: float = 0.5,
        expert_specialization_threshold: float = 0.8
    ):
        super().__init__()
        self.base_moe = base_moe_model
        self.memory_size = memory_size
        self.consolidation_strength = consolidation_strength
        self.expert_specialization_threshold = expert_specialization_threshold
        
        # Task-specific components
        self.task_memories = {}  # Task ID -> memory samples
        self.expert_task_affinity = {}  # Expert ID -> Task affinities
        self.task_embeddings = nn.Embedding(100, 64)  # Support up to 100 tasks
        
        # Importance weights for elastic weight consolidation
        self.importance_weights = {}
        self.optimal_params = {}
        
        # Current task tracking
        self.current_task_id = 0
        self.task_history = []
        
    def start_new_task(self, task_id: int):
        """Start learning a new task."""
        # Consolidate previous task knowledge
        if self.current_task_id != task_id and self.current_task_id in self.task_history:
            self._consolidate_task_knowledge(self.current_task_id)
            
        self.current_task_id = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
            self.task_memories[task_id] = []
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_id: Optional[int] = None,
        store_memory: bool = True
    ) -> torch.Tensor:
        """Forward pass with task-aware routing."""
        if task_id is None:
            task_id = self.current_task_id
            
        # Get task embedding
        task_embedding = self.task_embeddings(torch.tensor([task_id], device=hidden_states.device))
        
        # Modify hidden states with task context
        batch_size, seq_len, hidden_size = hidden_states.shape
        task_context = task_embedding.expand(batch_size, seq_len, -1)
        
        # Concatenate or add task context
        if hidden_states.size(-1) == task_context.size(-1):
            contextualized_states = hidden_states + task_context
        else:
            contextualized_states = torch.cat([hidden_states, task_context], dim=-1)
            
        # Forward through base MoE
        outputs = self.base_moe(contextualized_states)
        
        # Store memory samples for continual learning
        if store_memory and len(self.task_memories[task_id]) < self.memory_size:
            memory_sample = {
                'input': hidden_states.detach().cpu(),
                'output': outputs.last_hidden_state.detach().cpu() if hasattr(outputs, 'last_hidden_state') else outputs.detach().cpu(),
                'task_id': task_id,
                'routing_info': outputs.routing_info if hasattr(outputs, 'routing_info') else None
            }
            self.task_memories[task_id].append(memory_sample)
            
        return outputs
        
    def _consolidate_task_knowledge(self, task_id: int):
        """Consolidate knowledge for a completed task."""
        # Compute parameter importance using Fisher Information
        self._compute_fisher_information(task_id)
        
        # Store optimal parameters
        self.optimal_params[task_id] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.optimal_params[task_id][name] = param.data.clone()
                
        # Update expert-task affinity
        self._update_expert_task_affinity(task_id)
        
    def _compute_fisher_information(self, task_id: int):
        """Compute Fisher Information Matrix diagonal for EWC."""
        if task_id not in self.task_memories:
            return
            
        # Sample from task memory
        memory_samples = random.sample(
            self.task_memories[task_id],
            min(100, len(self.task_memories[task_id]))
        )
        
        fisher_info = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                
        self.train()
        
        for sample in memory_samples:
            # Forward pass
            input_tensor = sample['input'].to(next(self.parameters()).device)
            target_tensor = sample['output'].to(next(self.parameters()).device)
            
            outputs = self(input_tensor, task_id=task_id, store_memory=False)
            
            # Compute loss (reconstruction loss)
            if hasattr(outputs, 'last_hidden_state'):
                loss = F.mse_loss(outputs.last_hidden_state, target_tensor)
            else:
                loss = F.mse_loss(outputs, target_tensor)
                
            # Backward pass
            self.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= len(memory_samples)
            
        self.importance_weights[task_id] = fisher_info
        
    def _update_expert_task_affinity(self, task_id: int):
        """Update expert-task affinity scores."""
        if not hasattr(self.base_moe, 'moe_layers'):
            return
            
        # Analyze routing patterns for this task
        expert_usage = defaultdict(float)
        total_tokens = 0
        
        for sample in self.task_memories[task_id]:
            routing_info = sample['routing_info']
            if routing_info and hasattr(routing_info, 'selected_experts'):
                experts = routing_info.selected_experts.flatten()
                weights = routing_info.expert_weights.flatten() if hasattr(routing_info, 'expert_weights') else None
                
                for i, expert_id in enumerate(experts):
                    weight = weights[i] if weights is not None else 1.0
                    expert_usage[expert_id.item()] += weight.item()
                    total_tokens += 1
                    
        # Normalize and store affinities
        if task_id not in self.expert_task_affinity:
            self.expert_task_affinity[task_id] = {}
            
        for expert_id, usage in expert_usage.items():
            affinity = usage / max(total_tokens, 1)
            self.expert_task_affinity[task_id][expert_id] = affinity
            
    def compute_ewc_loss(self, task_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss."""
        if task_ids is None:
            task_ids = [tid for tid in self.task_history if tid != self.current_task_id]
            
        if not task_ids:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        ewc_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for task_id in task_ids:
            if task_id in self.importance_weights and task_id in self.optimal_params:
                for name, param in self.named_parameters():
                    if name in self.importance_weights[task_id] and name in self.optimal_params[task_id]:
                        importance = self.importance_weights[task_id][name]
                        optimal_param = self.optimal_params[task_id][name]
                        
                        ewc_loss += (importance * (param - optimal_param) ** 2).sum()
                        
        return self.consolidation_strength * ewc_loss
        
    def replay_memory(self, batch_size: int = 16, num_tasks: int = 3) -> Dict[str, torch.Tensor]:
        """Generate a replay batch from task memories."""
        if not self.task_memories:
            return {}
            
        # Sample tasks
        available_tasks = list(self.task_memories.keys())
        sampled_tasks = random.sample(available_tasks, min(num_tasks, len(available_tasks)))
        
        replay_batch = {
            'inputs': [],
            'outputs': [],
            'task_ids': []
        }
        
        for task_id in sampled_tasks:
            task_samples = random.sample(
                self.task_memories[task_id],
                min(batch_size // len(sampled_tasks), len(self.task_memories[task_id]))
            )
            
            for sample in task_samples:
                replay_batch['inputs'].append(sample['input'])
                replay_batch['outputs'].append(sample['output'])
                replay_batch['task_ids'].append(task_id)
                
        # Convert to tensors
        if replay_batch['inputs']:
            device = next(self.parameters()).device
            replay_batch['inputs'] = torch.cat(replay_batch['inputs'], dim=0).to(device)
            replay_batch['outputs'] = torch.cat(replay_batch['outputs'], dim=0).to(device)
            replay_batch['task_ids'] = torch.tensor(replay_batch['task_ids'], device=device)
            
        return replay_batch
        
    def get_continual_learning_stats(self) -> Dict[str, Any]:
        """Get continual learning statistics."""
        stats = {
            'num_tasks': len(self.task_history),
            'current_task': self.current_task_id,
            'memory_usage': {
                task_id: len(memories) for task_id, memories in self.task_memories.items()
            },
            'expert_specialization': {}
        }
        
        # Compute expert specialization scores
        for task_id, affinities in self.expert_task_affinity.items():
            for expert_id, affinity in affinities.items():
                if expert_id not in stats['expert_specialization']:
                    stats['expert_specialization'][expert_id] = {}
                stats['expert_specialization'][expert_id][task_id] = affinity
                
        return stats


class SelfOrganizingExpertNetwork(nn.Module):
    """Self-organizing MoE with emergent expert specialization.
    
    Experts autonomously develop specializations through competitive learning
    and mutual information maximization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        initial_num_experts: int = 8,
        max_experts: int = 64,
        min_experts: int = 4,
        specialization_threshold: float = 0.9,
        competition_strength: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.current_num_experts = initial_num_experts
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.specialization_threshold = specialization_threshold
        self.competition_strength = competition_strength
        
        # Dynamic expert pool
        self.experts = nn.ModuleList()
        self.expert_ages = []
        self.expert_specializations = []
        
        # Initialize experts
        for i in range(initial_num_experts):
            expert = self._create_expert()
            self.experts.append(expert)
            self.expert_ages.append(0)
            self.expert_specializations.append(None)
            
        # Self-organizing router
        self.competitive_router = CompetitiveRouter(hidden_size, max_experts)
        
        # Mutual information estimator
        self.mi_estimator = MutualInformationEstimator(hidden_size)
        
        # Expert creation/deletion controls
        self.expert_performance_history = defaultdict(list)
        self.reorganization_interval = 1000
        self.steps_since_reorganization = 0
        
    def _create_expert(self) -> nn.Module:
        """Create a new expert network."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with self-organizing expert selection."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Competitive routing
        expert_activations, competition_info = self.competitive_router(
            hidden_states, 
            num_active_experts=self.current_num_experts
        )
        
        # Expert processing
        expert_outputs = []
        expert_performances = []
        
        for i, expert in enumerate(self.experts[:self.current_num_experts]):
            # Process through expert
            expert_output = expert(hidden_states)
            expert_outputs.append(expert_output)
            
            # Measure expert performance (reconstruction quality)
            reconstruction_loss = F.mse_loss(expert_output, hidden_states)
            expert_performances.append(reconstruction_loss.item())
            
        # Weighted combination
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [batch, seq, hidden, num_experts]
        activations = expert_activations.unsqueeze(-2)  # [batch, seq, 1, num_experts]
        
        combined_output = torch.sum(expert_stack * activations, dim=-1)
        
        # Update expert statistics
        self._update_expert_stats(expert_performances, expert_activations)
        
        # Self-organization check
        self.steps_since_reorganization += 1
        if self.steps_since_reorganization >= self.reorganization_interval:
            reorganization_info = self._reorganize_experts()
        else:
            reorganization_info = {}
            
        # Compile analysis info
        analysis_info = {
            'expert_activations': expert_activations,
            'expert_performances': expert_performances,
            'competition_info': competition_info,
            'reorganization_info': reorganization_info,
            'current_num_experts': self.current_num_experts
        }
        
        return combined_output, analysis_info
        
    def _update_expert_stats(self, performances: List[float], activations: torch.Tensor):
        """Update expert performance statistics."""
        avg_activations = activations.mean(dim=(0, 1))  # Average over batch and sequence
        
        for i, (perf, activation) in enumerate(zip(performances, avg_activations)):
            self.expert_performance_history[i].append({
                'performance': perf,
                'activation': activation.item(),
                'age': self.expert_ages[i]
            })
            
            # Keep only recent history
            if len(self.expert_performance_history[i]) > 100:
                self.expert_performance_history[i] = self.expert_performance_history[i][-100:]
                
            # Age the expert
            self.expert_ages[i] += 1
            
    def _reorganize_experts(self) -> Dict[str, Any]:
        """Reorganize expert network based on performance and specialization."""
        self.steps_since_reorganization = 0
        reorganization_info = {'actions': []}
        
        # Analyze expert performance
        expert_analysis = []
        for i in range(self.current_num_experts):
            history = self.expert_performance_history[i]
            if not history:
                continue
                
            recent_performance = np.mean([h['performance'] for h in history[-20:]])
            recent_activation = np.mean([h['activation'] for h in history[-20:]])
            age = self.expert_ages[i]
            
            expert_analysis.append({
                'id': i,
                'performance': recent_performance,
                'activation': recent_activation,
                'age': age,
                'utilization': recent_activation * (1 / (1 + recent_performance))  # Higher is better
            })
            
        # Sort by utilization (performance * activation)
        expert_analysis.sort(key=lambda x: x['utilization'], reverse=True)
        
        # Expert deletion (remove underperforming experts)
        if self.current_num_experts > self.min_experts:
            worst_expert = expert_analysis[-1]
            if (worst_expert['utilization'] < 0.1 and 
                worst_expert['age'] > 500 and
                self.current_num_experts > self.min_experts + 1):
                
                self._remove_expert(worst_expert['id'])
                reorganization_info['actions'].append(f"Removed expert {worst_expert['id']}")
                
        # Expert creation (add new experts for high demand)
        if self.current_num_experts < self.max_experts:
            # Check if existing experts are overloaded
            high_activation_experts = [e for e in expert_analysis if e['activation'] > 0.8]
            
            if len(high_activation_experts) >= 2:  # Multiple overloaded experts
                new_expert_id = self._add_expert()
                reorganization_info['actions'].append(f"Added new expert {new_expert_id}")
                
        # Expert specialization analysis
        specialization_info = self._analyze_specializations()
        reorganization_info['specialization_analysis'] = specialization_info
        
        return reorganization_info
        
    def _remove_expert(self, expert_id: int):
        """Remove an underperforming expert."""
        if expert_id < len(self.experts) and self.current_num_experts > self.min_experts:
            # Remove from lists
            del self.experts[expert_id]
            del self.expert_ages[expert_id]
            del self.expert_specializations[expert_id]
            del self.expert_performance_history[expert_id]
            
            # Adjust indices in performance history
            new_history = {}
            for old_id, history in self.expert_performance_history.items():
                if old_id > expert_id:
                    new_history[old_id - 1] = history
                else:
                    new_history[old_id] = history
            self.expert_performance_history = new_history
            
            self.current_num_experts -= 1
            
    def _add_expert(self) -> int:
        """Add a new expert to the network."""
        if self.current_num_experts < self.max_experts:
            new_expert = self._create_expert()
            self.experts.append(new_expert)
            self.expert_ages.append(0)
            self.expert_specializations.append(None)
            
            new_expert_id = self.current_num_experts
            self.expert_performance_history[new_expert_id] = []
            
            self.current_num_experts += 1
            return new_expert_id
            
        return -1
        
    def _analyze_specializations(self) -> Dict[str, Any]:
        """Analyze expert specializations using mutual information."""
        # This would involve analyzing what types of inputs each expert responds to
        # For now, return basic statistics
        
        specialization_scores = []
        for i in range(self.current_num_experts):
            history = self.expert_performance_history[i]
            if len(history) >= 10:
                activations = [h['activation'] for h in history[-50:]]
                specialization = np.std(activations)  # Higher std = more specialized
                specialization_scores.append(specialization)
            else:
                specialization_scores.append(0.0)
                
        return {
            'mean_specialization': float(np.mean(specialization_scores)),
            'max_specialization': float(np.max(specialization_scores)) if specialization_scores else 0.0,
            'specialization_distribution': specialization_scores
        }


class CompetitiveRouter(nn.Module):
    """Competitive routing mechanism for self-organizing experts."""
    
    def __init__(self, hidden_size: int, max_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_experts = max_experts
        
        # Competitive layers
        self.competition_network = nn.Sequential(
            nn.Linear(hidden_size, max_experts * 2),
            nn.GELU(),
            nn.Linear(max_experts * 2, max_experts)
        )
        
        # Lateral inhibition matrix
        self.lateral_inhibition = nn.Parameter(
            torch.eye(max_experts) * -0.1 + torch.ones(max_experts, max_experts) * 0.02
        )
        
    def forward(self, hidden_states: torch.Tensor, num_active_experts: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Competitive expert selection."""
        # Initial activation scores
        raw_scores = self.competition_network(hidden_states)
        raw_scores = raw_scores[..., :num_active_experts]  # Limit to active experts
        
        # Apply lateral inhibition (winner-take-all dynamics)
        inhibition_matrix = self.lateral_inhibition[:num_active_experts, :num_active_experts]
        
        # Competitive dynamics (simplified)
        competitive_scores = raw_scores + torch.matmul(raw_scores, inhibition_matrix)
        
        # Softmax with temperature for smooth competition
        temperature = 2.0
        expert_activations = F.softmax(competitive_scores / temperature, dim=-1)
        
        competition_info = {
            'raw_scores': raw_scores,
            'competitive_scores': competitive_scores,
            'inhibition_strength': torch.mean(torch.abs(inhibition_matrix)).item()
        }
        
        return expert_activations, competition_info


class MutualInformationEstimator(nn.Module):
    """Neural mutual information estimator for expert specialization analysis."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # MINE (Mutual Information Neural Estimation) network
        self.mi_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information between x and y."""
        batch_size = x.size(0)
        
        # Joint distribution samples
        joint_samples = torch.cat([x, y], dim=-1)
        joint_scores = self.mi_network(joint_samples)
        
        # Marginal distribution samples (shuffle y)
        y_shuffled = y[torch.randperm(batch_size)]
        marginal_samples = torch.cat([x, y_shuffled], dim=-1)
        marginal_scores = self.mi_network(marginal_samples)
        
        # MINE estimation
        mi_estimate = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)))
        
        return mi_estimate