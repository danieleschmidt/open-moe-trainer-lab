# ADR-002: Routing Algorithm Architecture

## Status
Accepted

## Date
2025-01-27

## Context
The router is the core component that determines which experts process which tokens. We need to design a flexible architecture that supports multiple routing algorithms while maintaining performance.

### Routing Algorithms to Support

1. **Top-K Routing**: Select K highest-scoring experts per token
2. **Expert Choice**: Experts select tokens up to capacity
3. **Switch Routing**: Single expert per token with capacity factor
4. **Hash-based Routing**: Deterministic routing based on token features
5. **Learned Routing**: Attention-based or transformer-based routing

### Design Requirements
- Plugin architecture for new routing algorithms
- Efficient implementation for large-scale training
- Support for load balancing mechanisms
- Gradient flow through routing decisions
- Real-time routing analytics

## Decision
We will implement a **Strategy Pattern** based router architecture:

```python
class Router(nn.Module):
    def __init__(self, routing_strategy: RoutingStrategy):
        self.strategy = routing_strategy
    
    def forward(self, hidden_states):
        return self.strategy.route(hidden_states)

class RoutingStrategy(ABC):
    @abstractmethod
    def route(self, hidden_states) -> RoutingDecision
    
class TopKRouting(RoutingStrategy):
    def route(self, hidden_states):
        # Implementation
        pass
```

### Key Design Decisions

1. **Modular Strategies**: Each routing algorithm is a separate strategy class
2. **Common Interface**: All strategies implement the same routing interface
3. **Configurable Load Balancing**: Load balancing is configurable per strategy
4. **Analytics Integration**: Built-in hooks for routing decision tracking
5. **Gradient Support**: All routing decisions support gradient flow

## Consequences

### Positive
- Easy to add new routing algorithms
- Clear separation of concerns
- Testable components
- Performance optimization per algorithm
- Research-friendly for novel routing methods

### Negative
- Additional abstraction layer overhead
- More complex codebase
- Potential performance impact from virtual function calls

### Implementation Plan
1. Implement base Router and RoutingStrategy classes
2. Create TopKRouting as reference implementation
3. Add SwitchRouting for Switch Transformer compatibility
4. Implement ExpertChoiceRouting for token selection
5. Add routing analytics hooks
6. Performance optimization and benchmarking