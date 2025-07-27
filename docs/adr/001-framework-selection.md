# ADR-001: Primary Framework Selection

## Status
Accepted

## Date
2025-01-27

## Context
We need to select a primary deep learning framework for the Open MoE Trainer Lab. The framework choice will impact:
- Development velocity and ease of use
- Community support and ecosystem
- Performance characteristics
- Multi-GPU and distributed training capabilities
- Integration with existing MoE implementations

### Options Considered

1. **PyTorch**
   - Pros: Excellent dynamic graph support, large community, strong MoE ecosystem
   - Cons: Historically weaker production deployment story
   
2. **TensorFlow**
   - Pros: Strong production deployment, comprehensive tooling
   - Cons: More complex for research, fewer MoE implementations
   
3. **JAX**
   - Pros: Functional programming, excellent for research, XLA compilation
   - Cons: Smaller ecosystem, steeper learning curve

## Decision
We will use **PyTorch** as the primary framework for the following reasons:

1. **MoE Ecosystem**: Most existing MoE implementations (Mixtral, OLMoE, Switch Transformer) use PyTorch
2. **Research Flexibility**: Dynamic computation graphs are ideal for experimental routing algorithms
3. **Distributed Training**: PyTorch's DDP and FSDP provide robust distributed training capabilities
4. **Community**: Largest ML research community uses PyTorch
5. **Integration**: Easy integration with HuggingFace Transformers library

## Consequences

### Positive
- Faster development due to existing MoE codebases
- Access to pre-trained models through HuggingFace
- Strong community support for troubleshooting
- Excellent debugging capabilities

### Negative
- Need to invest in production deployment tooling
- TorchScript compilation can be complex for dynamic models
- Memory management requires careful attention

### Mitigation Strategies
- Use TorchServe for production model serving
- Implement comprehensive memory profiling
- Leverage PyTorch's native compilation features (torch.compile)
- Consider ONNX export for deployment flexibility