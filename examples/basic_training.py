#!/usr/bin/env python3
"""
Basic MoE Training Example

This example demonstrates how to train a simple MoE model from scratch using the Open MoE Trainer Lab.
It covers the essential components: model initialization, data preparation, training configuration, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from pathlib import Path

from moe_lab import MoEModel, MoETrainer
from moe_lab.data import TextDataset, MoEDataCollator
from moe_lab.utils.logging import setup_logging


def create_sample_data():
    """Create sample training data for demonstration."""
    sample_texts = [
        "The field of machine learning has evolved rapidly over the past decade.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models use neural networks with multiple layers.",
        "Mixture of Experts models can scale to very large parameter counts efficiently.",
        "Transformer architectures have revolutionized natural language understanding.",
        "Computer vision applications benefit from convolutional neural networks.",
        "Reinforcement learning algorithms learn through interaction with environments.",
        "Large language models demonstrate emergent capabilities at scale.",
        "Distributed training enables training of massive neural networks.",
        "Fine-tuning pre-trained models is more efficient than training from scratch.",
        # Add more diverse examples
        "The quick brown fox jumps over the lazy dog repeatedly.",
        "Scientific research requires careful experimental design and analysis.",
        "Software engineering principles ensure maintainable and scalable code.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence systems require ethical considerations and safety measures."
    ] * 20  # Repeat for more training data
    
    return sample_texts


def main():
    """Main training function."""
    print("🚀 Starting Basic MoE Training Example")
    
    # Setup logging
    setup_logging()
    
    # Configuration
    config = {
        'model': {
            'vocab_size': 32000,
            'hidden_size': 512,
            'num_experts': 8,
            'experts_per_token': 2,
            'num_layers': 6,
            'num_attention_heads': 8,
            'max_position_embeddings': 1024,
            'moe_layers': [1, 3, 5]  # Every other layer starting from 1
        },
        'training': {
            'learning_rate': 3e-4,
            'num_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 250,
            'load_balancing_loss_coef': 0.01,
            'router_z_loss_coef': 0.001,
            'max_grad_norm': 1.0
        },
        'data': {
            'max_seq_length': 256,
            'preprocessing': {
                'lowercase': False,
                'remove_special_chars': False
            }
        }
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize tokenizer (using a simple mock for this example)
    print("📝 Setting up tokenizer...")
    # In real usage, you'd use: tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # For this example, we'll create a simple mock
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
            self.pad_token_id = 0
            self.eos_token_id = 1
            
        def encode(self, text, max_length=None, truncation=True):
            # Simple mock encoding - convert text to list of integers
            return [hash(char) % self.vocab_size for char in text[:max_length or len(text)]]
        
        def decode(self, token_ids):
            return ''.join([chr(65 + (id % 26)) for id in token_ids])
    
    tokenizer = MockTokenizer()
    
    # Create sample data
    print("📊 Creating sample training data...")
    sample_texts = create_sample_data()
    
    # Split into train/eval
    split_idx = int(0.8 * len(sample_texts))
    train_texts = sample_texts[:split_idx]
    eval_texts = sample_texts[split_idx:]
    
    print(f"📈 Training samples: {len(train_texts)}")
    print(f"📉 Evaluation samples: {len(eval_texts)}")
    
    # Create datasets
    print("🏗️  Creating datasets...")
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    
    eval_dataset = TextDataset(
        texts=eval_texts,
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    
    # Initialize model
    print("🧠 Initializing MoE model...")
    model = MoEModel(**config['model'])
    model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 MoE layers: {model.moe_layers}")
    print(f"👥 Experts per MoE layer: {config['model']['num_experts']}")
    print(f"🎪 Experts per token: {config['model']['experts_per_token']}")
    
    # Initialize data collator
    data_collator = MoEDataCollator(
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    
    # Initialize trainer
    print("🏃 Initializing trainer...")
    trainer = MoETrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        **config['training']
    )
    
    # Display training info
    print("\n" + "="*50)
    print("🎯 TRAINING CONFIGURATION")
    print("="*50)
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Load balancing coef: {config['training']['load_balancing_loss_coef']}")
    print(f"Router z-loss coef: {config['training']['router_z_loss_coef']}")
    print("="*50 + "\n")
    
    # Start training
    print("🚀 Starting training...")
    try:
        training_result = trainer.train()
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED")
        print("="*50)
        print(f"Final loss: {training_result.final_loss:.4f}")
        print(f"Best loss: {training_result.best_loss:.4f}")
        print(f"Training time: {training_result.total_time:.2f} seconds")
        print(f"Steps completed: {training_result.global_step}")
        print("="*50 + "\n")
        
        # Run evaluation
        print("📊 Running final evaluation...")
        eval_result = trainer.evaluate()
        
        print("📈 EVALUATION RESULTS")
        print("-" * 30)
        print(f"Perplexity: {eval_result.perplexity:.4f}")
        print(f"Eval loss: {eval_result.eval_loss:.4f}")
        print(f"Expert load variance: {eval_result.load_variance:.4f}")
        print(f"Router entropy: {eval_result.router_entropy:.4f}")
        print("-" * 30)
        
        # Save model
        output_dir = Path("./outputs/basic_training_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving model to {output_dir}")
        trainer.save_model(str(output_dir))
        
        # Save configuration
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Generate sample text
        print("📝 Generating sample text...")
        model.eval()
        with torch.no_grad():
            # Create a simple input
            input_text = "The future of artificial intelligence"
            input_ids = torch.tensor([tokenizer.encode(input_text)[:50]]).to(device)
            
            # Generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True
            )
            
            generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
            print(f"Input: {input_text}")
            print(f"Generated: {generated_text}")
        
        print("\n🎉 Example completed successfully!")
        print(f"📁 Check outputs in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()