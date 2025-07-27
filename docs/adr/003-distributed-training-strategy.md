# ADR-003: Distributed Training Strategy

## Status
Accepted

## Date
2025-01-27

## Context
MoE models require sophisticated distributed training strategies due to:
- Large total parameter counts (8x-128x+ experts)
- Uneven expert utilization leading to load imbalance
- Communication overhead from token routing
- Memory constraints with large expert pools

### Parallelism Options

1. **Data Parallelism**: Replicate model across GPUs
2. **Model Parallelism**: Split model layers across GPUs  
3. **Expert Parallelism**: Distribute experts across GPUs
4. **Pipeline Parallelism**: Pipeline different layers
5. **Hybrid Approaches**: Combine multiple strategies

### Current Solutions Analysis

- **DeepSpeed-MoE**: Expert parallelism with all-to-all communication
- **FairScale**: Model parallel MoE layers
- **Megatron-LM**: Pipeline + tensor parallelism for MoE
- **Alpa**: Automatic parallelization strategies

## Decision
We will implement a **hierarchical parallelism strategy** with the following architecture:

### Primary Strategy: Expert Parallelism + Data Parallelism
```python
class DistributedMoE:
    def __init__(self, 
                 world_size: int,
                 expert_parallel_size: int,
                 data_parallel_size: int):
        assert world_size == expert_parallel_size * data_parallel_size
        self.ep_group = create_expert_parallel_group(expert_parallel_size)
        self.dp_group = create_data_parallel_group(data_parallel_size)
```

### Communication Pattern
1. **Local Expert Computation**: Each GPU handles subset of experts
2. **All-to-All Exchange**: Route tokens to appropriate expert GPUs
3. **Expert Processing**: Parallel expert computation
4. **All-to-All Return**: Gather expert outputs back to original GPUs
5. **Data Parallel Reduction**: Gradient synchronization across data parallel groups

### Fallback Strategies
- **Model Parallelism**: For very large individual experts
- **Pipeline Parallelism**: For deep MoE models with memory constraints
- **ZeRO**: For optimizer state partitioning

## Consequences

### Positive
- Scales to large expert counts (128+ experts)
- Efficient memory utilization
- Good load balancing with proper routing
- Composable with other parallelism strategies

### Negative
- High communication overhead for small batch sizes
- Requires careful load balancing to avoid stragglers
- Complex implementation and debugging
- All-to-all communication can be network bottleneck

### Mitigation Strategies
1. **Adaptive Batching**: Increase batch size to amortize communication
2. **Load Balancing**: Implement auxiliary losses and routing constraints
3. **Communication Optimization**: 
   - Overlapping computation and communication
   - Communication compression
   - Hierarchical all-to-all for large clusters
4. **Monitoring**: Real-time communication and load balancing metrics

### Implementation Roadmap
1. **Phase 1**: Basic expert parallelism with PyTorch DDP
2. **Phase 2**: Optimized all-to-all communication
3. **Phase 3**: Hybrid strategies (EP + MP + PP)
4. **Phase 4**: Integration with DeepSpeed/FairScale backends
5. **Phase 5**: Automatic parallelization strategy selection