# Monitoring & Observability Playbook

## Overview

This playbook provides guidance for monitoring and troubleshooting the Open MoE Trainer Lab infrastructure and applications.

## Monitoring Stack

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notifications
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing

## Key Metrics

### Training Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `moe_lab_training_tokens_per_second` | Training throughput | < 500/sec |
| `moe_lab_training_loss` | Training loss value | No decrease for 1h |
| `moe_lab_expert_utilization_variance` | Expert load balance | > 0.3 |
| `moe_lab_gpu_memory_utilization` | GPU memory usage | > 95% |
| `moe_lab_training_step_duration` | Time per training step | > 10s |

### Inference Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `moe_lab_inference_latency_p99` | 99th percentile latency | > 1s |
| `moe_lab_inference_requests_per_second` | Request throughput | < 10/sec |
| `moe_lab_inference_error_rate` | Error percentage | > 5% |
| `moe_lab_expert_cache_hit_rate` | Cache efficiency | < 80% |
| `moe_lab_model_load_time` | Model loading duration | > 30s |

### System Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `node_cpu_utilization` | CPU usage | > 90% |
| `node_memory_utilization` | Memory usage | > 90% |
| `node_disk_utilization` | Disk usage | > 85% |
| `nvidia_gpu_temperature` | GPU temperature | > 85°C |
| `container_memory_usage` | Container memory | > limit |

## Alert Runbooks

### High Training Loss Alert

**Alert**: `MoETrainingLossHigh`

**Description**: Training loss has not decreased in the last hour

**Severity**: Warning

**Troubleshooting Steps**:

1. Check training logs for errors:
   ```bash
   kubectl logs -f deployment/moe-trainer
   ```

2. Verify learning rate and optimizer settings:
   ```bash
   # Check current configuration
   curl http://trainer:8080/config
   ```

3. Check expert utilization balance:
   ```bash
   # Query expert routing metrics
   curl 'http://prometheus:9090/api/v1/query?query=moe_lab_expert_utilization_variance'
   ```

4. Restart training with checkpoint if issues persist:
   ```bash
   kubectl rollout restart deployment/moe-trainer
   ```

### High GPU Memory Usage

**Alert**: `GPUMemoryUsageHigh`

**Description**: GPU memory usage above 95%

**Severity**: Critical

**Troubleshooting Steps**:

1. Check current GPU memory usage:
   ```bash
   nvidia-smi
   ```

2. Reduce batch size temporarily:
   ```bash
   # Update training config
   kubectl patch configmap training-config --patch '{"data":{"batch_size":"16"}}'
   ```

3. Enable gradient checkpointing:
   ```bash
   kubectl patch configmap training-config --patch '{"data":{"gradient_checkpointing":"true"}}'
   ```

4. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Expert Load Imbalance

**Alert**: `ExpertLoadImbalance`

**Description**: Expert utilization variance too high

**Severity**: Warning

**Troubleshooting Steps**:

1. Check expert routing patterns:
   ```bash
   # View expert utilization dashboard
   open http://grafana:3000/d/expert-routing
   ```

2. Increase load balancing loss coefficient:
   ```yaml
   # training-config.yaml
   load_balance_loss_coef: 0.02  # Increase from 0.01
   ```

3. Review router architecture:
   ```bash
   # Check router configuration
   kubectl get configmap router-config -o yaml
   ```

4. Consider expert capacity adjustments:
   ```yaml
   # Increase expert capacity factor
   expert_capacity_factor: 1.5
   ```

### High Inference Latency

**Alert**: `InferenceLatencyHigh`

**Description**: 99th percentile inference latency above 1 second

**Severity**: Warning

**Troubleshooting Steps**:

1. Check inference service logs:
   ```bash
   kubectl logs -f deployment/moe-inference
   ```

2. Monitor expert cache performance:
   ```bash
   # Check cache hit rate
   curl 'http://prometheus:9090/api/v1/query?query=moe_lab_expert_cache_hit_rate'
   ```

3. Increase cache size:
   ```yaml
   # inference-config.yaml
   expert_cache_size_gb: 16  # Increase cache size
   ```

4. Enable model quantization:
   ```yaml
   # Enable int8 quantization
   quantization:
     enabled: true
     method: "int8"
   ```

5. Scale up inference replicas:
   ```bash
   kubectl scale deployment moe-inference --replicas=3
   ```

## Health Checks

### Application Health

```bash
# Training service health
curl http://trainer:8080/health

# Inference service health
curl http://inference:8000/health

# Dashboard health
curl http://dashboard:8080/health
```

### Database Health

```bash
# PostgreSQL health
psql -h postgres -c "SELECT 1;"

# Redis health
redis-cli -h redis ping
```

### Storage Health

```bash
# Check model storage
ls -la /storage/models/

# Check checkpoint storage
ls -la /storage/checkpoints/

# Check data storage
df -h /storage/data/
```

## Performance Tuning

### Training Optimization

1. **Batch Size Tuning**:
   ```yaml
   # Start with smaller batch size and increase
   batch_size: 32
   gradient_accumulation_steps: 4  # Effective batch size: 128
   ```

2. **Mixed Precision**:
   ```yaml
   # Enable automatic mixed precision
   mixed_precision: true
   amp_dtype: "float16"
   ```

3. **Expert Parallelism**:
   ```yaml
   # Distribute experts across GPUs
   expert_parallel_size: 4
   ```

### Inference Optimization

1. **Expert Caching**:
   ```yaml
   expert_cache:
     enabled: true
     size_gb: 8
     policy: "lru"
     preload_top_k: 4
   ```

2. **Batch Processing**:
   ```yaml
   # Enable dynamic batching
   dynamic_batching:
     enabled: true
     max_batch_size: 32
     batch_timeout_ms: 50
   ```

3. **Model Compilation**:
   ```yaml
   # Enable TorchScript compilation
   compilation:
     enabled: true
     backend: "inductor"
   ```

## Log Analysis

### Common Log Patterns

```bash
# Training convergence issues
grep "loss.*nan\|loss.*inf" training.log

# Expert routing problems
grep "expert.*overflow\|routing.*failed" training.log

# Memory issues
grep "CUDA out of memory\|OOM" training.log

# Distributed training issues
grep "NCCL\|distributed" training.log
```

### Log Aggregation Queries

```bash
# Loki queries for common issues

# Error rate by service
{job="moe-lab"} |= "ERROR" | rate(1m)

# GPU memory warnings
{job="moe-lab"} |= "GPU memory" |= "warning"

# Expert routing statistics
{job="moe-lab"} |= "expert_utilization" | json
```

## Backup and Recovery

### Checkpoint Management

```bash
# List available checkpoints
ls -la /storage/checkpoints/

# Restore from specific checkpoint
kubectl set env deployment/moe-trainer CHECKPOINT_PATH=/storage/checkpoints/step-1000

# Backup critical checkpoints
aws s3 sync /storage/checkpoints/ s3://moe-lab-backups/checkpoints/
```

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h postgres moe_lab > moe_lab_backup.sql

# Redis backup
redis-cli -h redis --rdb dump.rdb
```

## Escalation Procedures

### Severity Levels

- **P0 - Critical**: Service completely down, data loss risk
- **P1 - High**: Major functionality impaired
- **P2 - Medium**: Minor functionality impaired
- **P3 - Low**: Cosmetic issues, optimization opportunities

### Contact Information

- **On-call Engineer**: Slack #moe-lab-alerts
- **ML Team Lead**: @ml-lead
- **Infrastructure Team**: @infra-team
- **Emergency**: phone-number

### Escalation Timeline

- **P0**: Immediate notification
- **P1**: 15 minutes
- **P2**: 1 hour
- **P3**: Next business day

## Dashboard Links

- [Training Overview](http://grafana:3000/d/training)
- [Inference Metrics](http://grafana:3000/d/inference)
- [Expert Routing](http://grafana:3000/d/routing)
- [System Health](http://grafana:3000/d/system)
- [GPU Monitoring](http://grafana:3000/d/gpu)

## Maintenance Windows

### Planned Maintenance

- **Schedule**: Every Sunday 2:00-4:00 UTC
- **Duration**: 2 hours maximum
- **Notification**: 48 hours advance notice

### Maintenance Checklist

- [ ] Backup critical data
- [ ] Update monitoring configuration
- [ ] Restart services in rolling fashion
- [ ] Validate health checks
- [ ] Update documentation
- [ ] Send completion notification