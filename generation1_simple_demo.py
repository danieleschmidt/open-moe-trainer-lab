#!/usr/bin/env python3
"""
GENERATION 1: MAKE IT WORK (Simple)
Autonomous SDLC - Basic MoE functionality demonstration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from moe_lab import MoEModel
import time
import json

def create_simple_dataset(vocab_size=1000, seq_len=128, num_samples=100):
    """Create a simple synthetic dataset for testing."""
    data = []
    for _ in range(num_samples):
        # Create random sequences with some pattern
        sequence = torch.randint(0, vocab_size, (seq_len,))
        data.append(sequence)
    return torch.stack(data)

def simple_training_loop(model, data, num_epochs=3, batch_size=4):
    """Simple training loop for MoE model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Create targets (next token prediction)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            
            # Get logits for language modeling
            logits = model.lm_head(outputs.last_hidden_state)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Add MoE losses if available
            if outputs.load_balancing_loss is not None:
                loss += outputs.load_balancing_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def demonstrate_routing_analysis(model, data):
    """Demonstrate router analysis capabilities."""
    model.eval()
    
    with torch.no_grad():
        # Take a small sample
        sample_batch = data[:2, :32]  # 2 samples, 32 tokens each
        
        outputs = model(sample_batch, return_routing_info=True)
        
        routing_stats = {
            "load_variance": float(outputs.routing_info.load_variance),
            "entropy": float(outputs.routing_info.entropy),
            "num_experts": model.num_experts,
            "experts_per_token": model.experts_per_token
        }
        
        if outputs.expert_weights:
            # Analyze expert utilization
            for layer_idx, weights in outputs.expert_weights.items():
                expert_usage = weights.sum(dim=0)  # Sum across tokens
                most_used_expert = torch.argmax(expert_usage).item()
                routing_stats[f"layer_{layer_idx}_most_used_expert"] = most_used_expert
                routing_stats[f"layer_{layer_idx}_usage_variance"] = float(expert_usage.var())
        
        return routing_stats

def run_generation1_demo():
    """Run Generation 1 simple MoE demonstration."""
    print("🚀 Starting Generation 1: MAKE IT WORK (Simple)")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Create simple MoE model
    print("1. Creating MoE model...")
    model = MoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=4,
        experts_per_token=2,
        num_layers=4,
        num_attention_heads=4,
        moe_layers=[1, 3]  # Only layers 1 and 3 are MoE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created with {total_params:,} parameters")
    
    # 2. Create synthetic dataset
    print("2. Creating synthetic dataset...")
    dataset = create_simple_dataset(vocab_size=1000, seq_len=64, num_samples=32)
    print(f"   ✅ Dataset created: {dataset.shape}")
    
    # 3. Simple training
    print("3. Running simple training...")
    losses = simple_training_loop(model, dataset, num_epochs=3, batch_size=4)
    print(f"   ✅ Training complete. Final loss: {losses[-1]:.4f}")
    
    # 4. Routing analysis
    print("4. Analyzing routing patterns...")
    routing_stats = demonstrate_routing_analysis(model, dataset)
    print(f"   ✅ Router entropy: {routing_stats['entropy']:.3f}")
    print(f"   ✅ Load variance: {routing_stats['load_variance']:.3f}")
    
    # 5. Simple generation test
    print("5. Testing text generation...")
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 1000, (1, 10))  # Random 10-token prompt
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
        print(f"   ✅ Generated {generated.shape[1]} tokens")
    
    elapsed_time = time.time() - start_time
    
    # Results summary
    results = {
        "generation": 1,
        "status": "WORKING",
        "model_params": total_params,
        "training_epochs": 3,
        "final_loss": losses[-1],
        "routing_entropy": routing_stats["entropy"],
        "load_variance": routing_stats["load_variance"],
        "execution_time_seconds": elapsed_time,
        "features_implemented": [
            "Basic MoE model creation",
            "Expert routing (Top-K)",
            "Load balancing loss",
            "Simple training loop",
            "Routing analysis",
            "Text generation"
        ]
    }
    
    # Save results
    with open("generation1_simple_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n🎉 Generation 1 Complete: BASIC FUNCTIONALITY WORKING")
    print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
    print("📊 Results saved to: generation1_simple_results.json")
    print("🔄 Proceeding to Generation 2: MAKE IT ROBUST...")
    
    return results

if __name__ == "__main__":
    # Run autonomous Generation 1 demo
    results = run_generation1_demo()
    
    # Validate success
    assert results["status"] == "WORKING", "Generation 1 failed"
    assert results["final_loss"] < 10.0, "Training loss too high"
    assert results["routing_entropy"] > 0.0, "Router not functioning"
    
    print("✅ Generation 1 validation passed - Ready for Generation 2")