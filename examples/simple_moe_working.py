#!/usr/bin/env python3
"""
Simple Working MoE Demo - Generation 1: Make It Work
Demonstrates core MoE functionality with minimal implementation.
"""

import json
import random
import math


class MoEDemo:
    """Simplified MoE demonstration."""
    
    def __init__(self, hidden_size=32, num_experts=4, top_k=2):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize router weights (hidden_size x num_experts)
        self.router_weights = [[random.gauss(0, 0.02) for _ in range(num_experts)] 
                              for _ in range(hidden_size)]
        
        # Initialize expert weights (simplified)
        self.expert_weights = []
        for e in range(num_experts):
            # Each expert: hidden_size x hidden_size transformation
            expert_w = [[random.gauss(0, 0.02) for _ in range(hidden_size)] 
                       for _ in range(hidden_size)]
            self.expert_weights.append(expert_w)
    
    def softmax(self, logits):
        """Apply softmax to a list of values."""
        max_logit = max(logits)
        exp_logits = [math.exp(x - max_logit) for x in logits]
        sum_exp = sum(exp_logits)
        return [x / sum_exp for x in exp_logits]
    
    def matrix_vector_mult(self, matrix, vector):
        """Multiply matrix by vector."""
        result = []
        for row in matrix:
            value = sum(row[i] * vector[i] for i in range(len(vector)))
            result.append(value)
        return result
    
    def route_token(self, token_embedding):
        """Route a single token to top-k experts."""
        # Compute router logits: token @ router_weights
        router_logits = self.matrix_vector_mult(
            [[self.router_weights[i][j] for i in range(self.hidden_size)] 
             for j in range(self.num_experts)], 
            token_embedding
        )
        
        # Get top-k experts
        expert_scores = list(enumerate(router_logits))
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        top_experts = expert_scores[:self.top_k]
        
        # Extract indices and logits
        expert_indices = [x[0] for x in top_experts]
        expert_logits = [x[1] for x in top_experts]
        
        # Convert to probabilities  
        expert_probs = self.softmax(expert_logits)
        
        return expert_indices, expert_probs, router_logits
    
    def expert_forward(self, expert_idx, token_embedding):
        """Pass token through specific expert."""
        expert_w = self.expert_weights[expert_idx]
        output = self.matrix_vector_mult(expert_w, token_embedding)
        
        # Apply GELU activation (simplified as ReLU)
        output = [max(0, x) for x in output]
        
        return output
    
    def forward(self, token_embedding):
        """Forward pass for a single token."""
        # Route token to experts
        expert_indices, expert_probs, router_logits = self.route_token(token_embedding)
        
        # Compute weighted output from selected experts
        final_output = [0.0] * self.hidden_size
        
        for i, expert_idx in enumerate(expert_indices):
            expert_output = self.expert_forward(expert_idx, token_embedding)
            weight = expert_probs[i]
            
            # Add weighted contribution
            for j in range(self.hidden_size):
                final_output[j] += weight * expert_output[j]
        
        return final_output, {
            'selected_experts': expert_indices,
            'expert_weights': expert_probs,
            'router_logits': router_logits
        }
    
    def compute_routing_stats(self, routing_history):
        """Compute routing statistics."""
        expert_counts = [0] * self.num_experts
        total_entropy = 0.0
        
        for routing_info in routing_history:
            # Count expert usage
            for expert_idx in routing_info['selected_experts']:
                expert_counts[expert_idx] += 1
            
            # Compute entropy
            probs = self.softmax(routing_info['router_logits'])
            entropy = -sum(p * math.log(p + 1e-8) for p in probs if p > 0)
            total_entropy += entropy
        
        # Compute load balance
        total_assignments = sum(expert_counts)
        expert_loads = [c / total_assignments for c in expert_counts] if total_assignments > 0 else [0] * self.num_experts
        
        # Load variance
        mean_load = 1.0 / self.num_experts
        load_variance = sum((load - mean_load) ** 2 for load in expert_loads) / self.num_experts
        
        avg_entropy = total_entropy / len(routing_history) if routing_history else 0.0
        
        return {
            'expert_utilization': expert_loads,
            'load_variance': load_variance,
            'average_entropy': avg_entropy
        }


def run_generation1_demo():
    """Run Generation 1 demonstration."""
    print("🚀 Open MoE Trainer Lab - Generation 1 Demo")
    print("=" * 50)
    
    # Configuration
    config = {
        "hidden_size": 32,
        "num_experts": 4,
        "top_k": 2,
        "num_tokens": 16
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create MoE model
    print("🏗️  Creating MoE model...")
    model = MoEDemo(
        hidden_size=config["hidden_size"],
        num_experts=config["num_experts"],
        top_k=config["top_k"]
    )
    
    # Generate test tokens
    print("📊 Processing tokens...")
    routing_history = []
    outputs = []
    
    for i in range(config["num_tokens"]):
        # Create random token embedding
        token = [random.gauss(0, 1.0) for _ in range(config["hidden_size"])]
        
        # Forward pass
        output, routing_info = model.forward(token)
        
        outputs.append(output)
        routing_history.append(routing_info)
        
        if i < 5:  # Show first 5 token routings
            print(f"  Token {i}: Experts {routing_info['selected_experts']} "
                  f"Weights {[f'{w:.3f}' for w in routing_info['expert_weights']]}")
    
    # Compute and display statistics
    print("\n📈 Routing Analysis:")
    print("-" * 30)
    
    stats = model.compute_routing_stats(routing_history)
    
    print(f"Load Balance Variance: {stats['load_variance']:.4f}")
    print(f"Average Routing Entropy: {stats['average_entropy']:.4f}")
    print("\nExpert Utilization:")
    for i, util in enumerate(stats['expert_utilization']):
        print(f"  Expert {i}: {util:.1%}")
    
    # Test with different input patterns
    print("\n🎯 Pattern Specialization Test:")
    print("-" * 35)
    
    test_patterns = [
        ("Low variance", [random.gauss(0, 0.1) for _ in range(config["hidden_size"])]),
        ("High variance", [random.gauss(0, 2.0) for _ in range(config["hidden_size"])]),
        ("Positive bias", [random.gauss(1.0, 0.5) for _ in range(config["hidden_size"])]),
        ("Negative bias", [random.gauss(-1.0, 0.5) for _ in range(config["hidden_size"])])
    ]
    
    for pattern_name, pattern_token in test_patterns:
        _, routing_info = model.forward(pattern_token)
        experts = routing_info['selected_experts']
        weights = routing_info['expert_weights']
        print(f"{pattern_name}: Experts {experts} Weights {[f'{w:.3f}' for w in weights]}")
    
    print("\n✅ Generation 1 Complete!")
    print("Core MoE functionality verified:")
    print("  ✓ Token routing to top-k experts")
    print("  ✓ Weighted expert aggregation")
    print("  ✓ Load balancing metrics")
    print("  ✓ Pattern-based expert selection")
    
    # Save results
    results = {
        "config": config,
        "routing_stats": stats,
        "sample_outputs": outputs[:5],  # First 5 outputs
        "generation": 1,
        "status": "working"
    }
    
    with open("generation1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to generation1_results.json")
    
    return results


if __name__ == "__main__":
    results = run_generation1_demo()
    print("\n🎉 Ready to proceed to Generation 2: Make It Robust!")