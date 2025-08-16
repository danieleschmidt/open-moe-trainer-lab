#!/usr/bin/env python3
"""
Generation 1 Demo: Simple MoE functionality working
AUTONOMOUS SDLC EXECUTION - GENERATION 1 IMPLEMENTATION
"""

import torch
import torch.nn.functional as F
from moe_lab import MoEModel, MoETrainer
from moe_lab.models import SwitchTransformer, MixtralModel, CustomMoE
import json
import time
from datetime import datetime

def demonstrate_core_functionality():
    """Demonstrate Generation 1: Core MoE functionality working."""
    
    results = {
        "generation": 1,
        "phase": "MAKE IT WORK (Simple)",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    print("🚀 GENERATION 1: MAKE IT WORK (Simple)")
    print("=" * 50)
    
    # Test 1: Basic MoE Model Creation
    print("\n1. Testing Basic MoE Model Creation...")
    try:
        model = MoEModel(
            hidden_size=768, 
            num_experts=4, 
            experts_per_token=2, 
            num_layers=6,
            num_attention_heads=12,
            vocab_size=1000
        )
        
        result = {
            "test": "Basic MoE Model Creation",
            "status": "PASS",
            "details": {
                "num_experts": model.num_experts,
                "hidden_size": model.hidden_size,
                "experts_per_token": model.experts_per_token,
                "moe_layers": list(model.moe_layers),
                "total_params": sum(p.numel() for p in model.parameters())
            }
        }
        
        print(f"   ✅ Model created successfully")
        print(f"   ✅ Experts: {model.num_experts}, Hidden: {model.hidden_size}")
        print(f"   ✅ MoE layers: {list(model.moe_layers)}")
        print(f"   ✅ Total parameters: {result['details']['total_params']:,}")
        
    except Exception as e:
        result = {"test": "Basic MoE Model Creation", "status": "FAIL", "error": str(e)}
        print(f"   ❌ Failed: {e}")
    
    results["tests"].append(result)
    
    # Test 2: Forward Pass
    print("\n2. Testing Forward Pass...")
    try:
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids, return_routing_info=True)
        
        result = {
            "test": "Forward Pass", 
            "status": "PASS",
            "details": {
                "input_shape": list(input_ids.shape),
                "output_shape": list(output.last_hidden_state.shape),
                "has_routing_info": output.routing_info is not None,
                "has_load_balancing_loss": output.load_balancing_loss is not None,
                "load_variance": float(output.routing_info.load_variance) if output.routing_info else None
            }
        }
        
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Input: {input_ids.shape} -> Output: {output.last_hidden_state.shape}")
        print(f"   ✅ Routing info available: {output.routing_info is not None}")
        print(f"   ✅ Load balancing loss: {output.load_balancing_loss}")
        
    except Exception as e:
        result = {"test": "Forward Pass", "status": "FAIL", "error": str(e)}
        print(f"   ❌ Failed: {e}")
    
    results["tests"].append(result)
    
    # Test 3: Different MoE Architectures
    print("\n3. Testing Different MoE Architectures...")
    
    architectures = [
        ("Switch Transformer", lambda: SwitchTransformer(
            hidden_size=768, 
            num_experts=8, 
            num_layers=4,
            num_attention_heads=12,
            vocab_size=1000
        )),
        ("Mixtral Style", lambda: MixtralModel(
            hidden_size=768, 
            num_experts=4, 
            num_layers=4,
            num_attention_heads=12,
            vocab_size=1000
        )),
        ("Custom MoE", lambda: CustomMoE(
            hidden_size=768, 
            num_experts=6, 
            num_layers=4,
            num_attention_heads=12,
            vocab_size=1000,
            expert_type="standard"
        ))
    ]
    
    for arch_name, arch_constructor in architectures:
        try:
            print(f"   Testing {arch_name}...")
            arch_model = arch_constructor()
            
            # Quick forward pass test
            test_input = torch.randint(0, 1000, (1, 5))
            with torch.no_grad():
                arch_output = arch_model(test_input)
            
            result = {
                "test": f"{arch_name} Architecture",
                "status": "PASS", 
                "details": {
                    "model_type": arch_name,
                    "num_experts": arch_model.num_experts,
                    "output_shape": list(arch_output.last_hidden_state.shape)
                }
            }
            
            print(f"     ✅ {arch_name} working")
            
        except Exception as e:
            result = {
                "test": f"{arch_name} Architecture",
                "status": "FAIL",
                "error": str(e)
            }
            print(f"     ❌ {arch_name} failed: {e}")
        
        results["tests"].append(result)
    
    # Test 4: Expert Routing Analysis
    print("\n4. Testing Expert Routing Analysis...")
    try:
        model.eval()
        test_input = torch.randint(0, 1000, (4, 8))  # Small batch for analysis
        
        with torch.no_grad():
            output = model(test_input, return_routing_info=True)
            
        routing_info = output.routing_info
        expert_usage = {}
        
        if routing_info and routing_info.selected_experts is not None:
            # Analyze expert selection patterns
            selected = routing_info.selected_experts.flatten()
            unique_experts, counts = torch.unique(selected, return_counts=True)
            
            for expert_idx, count in zip(unique_experts.tolist(), counts.tolist()):
                expert_usage[expert_idx] = count
        
        result = {
            "test": "Expert Routing Analysis",
            "status": "PASS",
            "details": {
                "expert_usage": expert_usage,
                "routing_entropy": float(routing_info.entropy) if routing_info else None,
                "load_variance": float(routing_info.load_variance) if routing_info else None,
                "num_active_experts": len(expert_usage)
            }
        }
        
        print(f"   ✅ Routing analysis successful")
        print(f"   ✅ Expert usage: {expert_usage}")
        print(f"   ✅ Active experts: {len(expert_usage)}/{model.num_experts}")
        print(f"   ✅ Routing entropy: {routing_info.entropy:.3f}")
        
    except Exception as e:
        result = {"test": "Expert Routing Analysis", "status": "FAIL", "error": str(e)}
        print(f"   ❌ Failed: {e}")
    
    results["tests"].append(result)
    
    # Test 5: Text Generation (Simple)
    print("\n5. Testing Simple Text Generation...")
    try:
        model.eval()
        prompt = torch.randint(0, 1000, (1, 5))  # Random prompt
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=True
            )
        
        result = {
            "test": "Simple Text Generation",
            "status": "PASS",
            "details": {
                "prompt_length": prompt.shape[1],
                "generated_length": generated.shape[1],
                "new_tokens": generated.shape[1] - prompt.shape[1]
            }
        }
        
        print(f"   ✅ Generation successful")
        print(f"   ✅ Prompt length: {prompt.shape[1]} -> Generated: {generated.shape[1]}")
        print(f"   ✅ New tokens: {generated.shape[1] - prompt.shape[1]}")
        
    except Exception as e:
        result = {"test": "Simple Text Generation", "status": "FAIL", "error": str(e)}
        print(f"   ❌ Failed: {e}")
    
    results["tests"].append(result)
    
    # Summary
    passed_tests = sum(1 for test in results["tests"] if test["status"] == "PASS")
    total_tests = len(results["tests"])
    
    print(f"\n🎯 GENERATION 1 SUMMARY")
    print("=" * 30)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    results["summary"] = {
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "success_rate": passed_tests/total_tests,
        "status": "COMPLETE" if passed_tests == total_tests else "PARTIAL"
    }
    
    if passed_tests == total_tests:
        print("✅ Generation 1: COMPLETE - Core functionality working!")
    else:
        print("⚠️  Generation 1: PARTIAL - Some issues need fixing")
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    results = demonstrate_core_functionality()
    execution_time = time.time() - start_time
    
    results["execution_time_seconds"] = execution_time
    
    # Save results
    with open("generation1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"📁 Results saved to: generation1_results.json")