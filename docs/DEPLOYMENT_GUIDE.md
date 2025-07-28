# Deployment Guide

This comprehensive guide covers deployment strategies, configurations, and best practices for the Open MoE Trainer Lab across different environments and platforms.

## Table of Contents

- [Overview](#overview)
- [Environment Configurations](#environment-configurations)
- [Container Deployment](#container-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Distributed Training Setup](#distributed-training-setup)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

The Open MoE Trainer Lab supports multiple deployment scenarios:

- **Development**: Local development with Docker Compose
- **Staging**: Kubernetes cluster for testing and validation
- **Production**: Scalable cloud deployment with monitoring
- **Distributed Training**: Multi-node GPU clusters
- **Edge Inference**: Lightweight inference deployments

### Architecture Components

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer/Ingress]
    end
    
    subgraph "Application Layer"
        API[API Server]
        WEB[Web Interface]
        WORKER[Training Workers]
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        REDIS[(Redis Cache)]
        S3[(Object Storage)]
    end
    
    subgraph "Infrastructure"
        MON[Monitoring]
        LOG[Logging]
        SEC[Security]
    end
    
    subgraph "GPU Cluster"
        GPU1[GPU Node 1]
        GPU2[GPU Node 2]
        GPUN[GPU Node N]
    end
    
    LB --> API
    LB --> WEB
    API --> DB
    API --> REDIS
    API --> S3
    WORKER --> GPU1
    WORKER --> GPU2
    WORKER --> GPUN
    MON --> API
    MON --> WORKER
    LOG --> API
    LOG --> WORKER
```

## Environment Configurations

### Development Environment

#### Docker Compose Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
      - "8888:8888"  # Jupyter
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://moelab:moelab123@postgres:5432/moelab_dev
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    command: ["python", "-m", "moe_lab.serve", "--dev", "--reload"]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: moelab_dev
      POSTGRES_USER: moelab
      POSTGRES_PASSWORD: moelab123
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_dev_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_dev_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_dev_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  postgres_dev_data:
  redis_dev_data:
  prometheus_dev_data:
  grafana_dev_data:
```

#### Environment Variables

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://moelab:moelab123@localhost:5432/moelab_dev
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Security
SECRET_KEY=dev-secret-key-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1

# Training Configuration
DEFAULT_MODEL_PATH=/app/models
DEFAULT_DATA_PATH=/app/data
MAX_TRAINING_JOBS=2

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_EXPORT_INTERVAL=30

# Development Tools
JUPYTER_ENABLED=true
PROFILING_ENABLED=true
```

### Staging Environment

#### Kubernetes Configuration

```yaml
# k8s/staging/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: moe-lab-staging
  labels:
    environment: staging
    app: open-moe-trainer-lab
---
# k8s/staging/configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: moe-lab-config
  namespace: moe-lab-staging
data:
  ENVIRONMENT: "staging"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  PROMETHEUS_ENABLED: "true"
  METRICS_EXPORT_INTERVAL: "30"
---
# k8s/staging/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-lab-api
  namespace: moe-lab-staging
  labels:
    app: moe-lab
    component: api
    environment: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: moe-lab
      component: api
  template:
    metadata:
      labels:
        app: moe-lab
        component: api
    spec:
      containers:
      - name: api
        image: ghcr.io/your-org/open-moe-trainer-lab:staging
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: moe-lab-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: moe-lab-secrets
              key: redis-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: moe-lab-secrets
              key: secret-key
        envFrom:
        - configMapRef:
            name: moe-lab-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# k8s/staging/service.yml
apiVersion: v1
kind: Service
metadata:
  name: moe-lab-api-service
  namespace: moe-lab-staging
spec:
  selector:
    app: moe-lab
    component: api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/staging/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: moe-lab-ingress
  namespace: moe-lab-staging
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - staging.moe-lab.example.com
    secretName: moe-lab-staging-tls
  rules:
  - host: staging.moe-lab.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: moe-lab-api-service
            port:
              number: 80
```

### Production Environment

#### High Availability Setup

```yaml
# k8s/production/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: moe-lab-api-hpa
  namespace: moe-lab-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: moe-lab-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
---
# k8s/production/pdb.yml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: moe-lab-api-pdb
  namespace: moe-lab-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: moe-lab
      component: api
```

## Container Deployment

### Multi-Stage Dockerfile

```dockerfile
# Dockerfile (already exists, but showing key deployment-relevant parts)
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Production stage
FROM base as production
COPY --from=builder /opt/venv /opt/venv
COPY . /app
WORKDIR /app

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash moelab
USER moelab

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "moe_lab.main:app"]

# Training stage for distributed training
FROM production as training
USER root
RUN apt-get update && apt-get install -y \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for multi-node training
RUN mkdir /var/run/sshd && \
    echo 'root:training123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

USER moelab
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# Inference stage - lightweight
FROM python:3.11-slim as inference
COPY --from=builder /opt/venv /opt/venv
COPY moe_lab/ /app/moe_lab/
COPY models/ /app/models/
WORKDIR /app

RUN useradd --create-home --shell /bin/bash inference
USER inference

EXPOSE 8080
CMD ["python", "-m", "moe_lab.serve", "--inference-only", "--port", "8080"]
```

### Container Registry and Tagging Strategy

```bash
# Build and tag images
docker build --target production -t ghcr.io/your-org/open-moe-trainer-lab:latest .
docker build --target production -t ghcr.io/your-org/open-moe-trainer-lab:v1.0.0 .
docker build --target training -t ghcr.io/your-org/open-moe-trainer-lab:v1.0.0-training .
docker build --target inference -t ghcr.io/your-org/open-moe-trainer-lab:v1.0.0-inference .

# Push to registry
docker push ghcr.io/your-org/open-moe-trainer-lab:latest
docker push ghcr.io/your-org/open-moe-trainer-lab:v1.0.0
docker push ghcr.io/your-org/open-moe-trainer-lab:v1.0.0-training
docker push ghcr.io/your-org/open-moe-trainer-lab:v1.0.0-inference
```

## Kubernetes Deployment

### Helm Chart Structure

```
helm/moe-lab/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-production.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── rbac.yaml
│   └── gpu-training-job.yaml
└── charts/
    ├── postgresql/
    ├── redis/
    └── prometheus/
```

### GPU Training Job Template

```yaml
# helm/moe-lab/templates/gpu-training-job.yaml
{{- if .Values.training.enabled }}
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "moe-lab.fullname" . }}-training-{{ .Values.training.jobId }}
  namespace: {{ .Release.Namespace }}
spec:
  parallelism: {{ .Values.training.parallelism }}
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        accelerator: nvidia-tesla-v100  # or appropriate GPU type
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: trainer
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}-training"
        resources:
          requests:
            nvidia.com/gpu: {{ .Values.training.gpuCount }}
            memory: "{{ .Values.training.memory }}"
            cpu: "{{ .Values.training.cpu }}"
          limits:
            nvidia.com/gpu: {{ .Values.training.gpuCount }}
            memory: "{{ .Values.training.memory }}"
            cpu: "{{ .Values.training.cpu }}"
        env:
        - name: WORLD_SIZE
          value: "{{ .Values.training.worldSize }}"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MASTER_ADDR
          value: "{{ include "moe-lab.fullname" . }}-training-master"
        - name: MASTER_PORT
          value: "23456"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: data-storage
          mountPath: /app/data
        command:
        - python
        - -m
        - torch.distributed.launch
        - --nproc_per_node={{ .Values.training.gpuCount }}
        - --nnodes={{ .Values.training.nodes }}
        - --node_rank=$(RANK)
        - --master_addr=$(MASTER_ADDR)
        - --master_port=$(MASTER_PORT)
        - moe_lab/train.py
        - --config={{ .Values.training.configPath }}
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: {{ include "moe-lab.fullname" . }}-models
      - name: data-storage
        persistentVolumeClaim:
          claimName: {{ include "moe-lab.fullname" . }}-data
{{- end }}
```

### Deployment Commands

```bash
# Install or upgrade with Helm
helm upgrade --install moe-lab ./helm/moe-lab \
  --namespace moe-lab-production \
  --create-namespace \
  --values helm/moe-lab/values-production.yaml

# Deploy specific version
helm upgrade --install moe-lab ./helm/moe-lab \
  --namespace moe-lab-production \
  --set image.tag=v1.0.0 \
  --values helm/moe-lab/values-production.yaml

# Deploy training job
helm upgrade --install moe-lab-training ./helm/moe-lab \
  --namespace moe-lab-training \
  --set training.enabled=true \
  --set training.jobId=$(date +%s) \
  --values helm/moe-lab/values-training.yaml
```

## Cloud Platform Deployment

### AWS Deployment

#### EKS Cluster Setup

```bash
# Create EKS cluster
eksctl create cluster \
  --name moe-lab-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Install GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.2/nvidia-device-plugin.yml

# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --set clusterName=moe-lab-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller \
  -n kube-system
```

#### RDS and ElastiCache Setup

```yaml
# terraform/aws/rds.tf
resource "aws_db_instance" "moe_lab_db" {
  identifier = "moe-lab-production"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r5.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  db_name  = "moelab"
  username = "moelab"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "moe-lab-production-final-snapshot"
  
  tags = {
    Name        = "MoE Lab Production DB"
    Environment = "production"
  }
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "moe-lab-cache-subnet"
  subnet_ids = var.private_subnet_ids
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "moe-lab-redis"
  description                  = "Redis cluster for MoE Lab"
  
  node_type                    = "cache.r5.large"
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 2
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  
  snapshot_retention_limit    = 5
  snapshot_window            = "03:00-05:00"
  
  tags = {
    Name        = "MoE Lab Redis"
    Environment = "production"
  }
}
```

### Google Cloud Platform

#### GKE with GPU Nodes

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create moe-lab-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=moe-lab-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=8

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.2/nvidia-device-plugin.yml
```

### Azure Deployment

#### AKS with GPU Support

```bash
# Create AKS cluster
az aks create \
  --resource-group moe-lab-rg \
  --name moe-lab-cluster \
  --node-count 3 \
  --node-vm-size Standard_DS3_v2 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group moe-lab-rg \
  --cluster-name moe-lab-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 5
```

## Distributed Training Setup

### Multi-Node Training Configuration

```yaml
# k8s/training/distributed-training.yml
apiVersion: batch/v1
kind: Job
metadata:
  name: moe-distributed-training
spec:
  parallelism: 4  # Number of training nodes
  template:
    spec:
      subdomain: training
      containers:
      - name: trainer
        image: ghcr.io/your-org/open-moe-trainer-lab:training
        resources:
          requests:
            nvidia.com/gpu: 8  # 8 GPUs per node
            memory: "128Gi"
            cpu: "32"
          limits:
            nvidia.com/gpu: 8
            memory: "128Gi"
            cpu: "32"
        env:
        - name: WORLD_SIZE
          value: "32"  # 4 nodes * 8 GPUs
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: MASTER_ADDR
          value: "training-master-0.training"
        - name: MASTER_PORT
          value: "23456"
        - name: NCCL_DEBUG
          value: "INFO"
        - name: NCCL_IB_DISABLE
          value: "1"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        ports:
        - containerPort: 23456
        volumeMounts:
        - name: shared-storage
          mountPath: /shared
        - name: local-ssd
          mountPath: /tmp
        command:
        - /bin/bash
        - -c
        - |
          torchrun \
            --nproc_per_node=8 \
            --nnodes=4 \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            moe_lab/train_distributed.py \
            --config=/shared/config/training_config.yaml
      volumes:
      - name: shared-storage
        nfs:
          server: nfs-server.example.com
          path: /shared/moe-lab
      - name: local-ssd
        hostPath:
          path: /mnt/local-ssd
          type: Directory
      restartPolicy: Never
  backoffLimit: 3
```

### MPI-based Training (Alternative)

```yaml
# k8s/training/mpi-training.yml
apiVersion: kubeflow.org/v1alpha2
kind: MPIJob
metadata:
  name: moe-mpi-training
spec:
  slotsPerWorker: 8
  cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - image: ghcr.io/your-org/open-moe-trainer-lab:training
            name: mpi-launcher
            command:
            - mpirun
            - --allow-run-as-root
            - -np
            - "32"
            - -bind-to
            - none
            - -map-by
            - slot
            - python
            - moe_lab/train_mpi.py
            resources:
              requests:
                cpu: 1
                memory: 2Gi
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - image: ghcr.io/your-org/open-moe-trainer-lab:training
            name: mpi-worker
            resources:
              requests:
                nvidia.com/gpu: 8
                cpu: 32
                memory: 128Gi
              limits:
                nvidia.com/gpu: 8
                cpu: 32
                memory: 128Gi
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus-config.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "moe_lab_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'moe-lab-api'
    static_configs:
      - targets: ['moe-lab-api:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'moe-lab-training'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - moe-lab-training
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: moe-lab-trainer
```

### Grafana Dashboards

```json
# monitoring/grafana/dashboards/moe-lab-overview.json
{
  "dashboard": {
    "title": "MoE Lab Overview",
    "panels": [
      {
        "title": "Training Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(moe_lab_training_tokens_total[5m])",
            "legendFormat": "Tokens/sec"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Expert Load Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "moe_lab_expert_load_distribution",
            "legendFormat": "Expert {{expert_id}}"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# logging/fluentd-config.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*moe-lab*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.var.log.containers.**moe-lab**.log>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      logstash_format true
      logstash_prefix moe-lab
    </match>
```

## Security Considerations

### Network Policies

```yaml
# security/network-policies.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: moe-lab-network-policy
  namespace: moe-lab-production
spec:
  podSelector:
    matchLabels:
      app: moe-lab
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: moe-lab
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

### Pod Security Standards

```yaml
# security/pod-security.yml
apiVersion: v1
kind: Namespace
metadata:
  name: moe-lab-production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: moe-lab-sa
  namespace: moe-lab-production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: moe-lab-role
  namespace: moe-lab-production
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: moe-lab-rolebinding
  namespace: moe-lab-production
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: moe-lab-role
subjects:
- kind: ServiceAccount
  name: moe-lab-sa
  namespace: moe-lab-production
```

### Secrets Management

```bash
# Create secrets using sealed-secrets or external secrets operator
kubectl create secret generic moe-lab-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://host:6379" \
  --from-literal=secret-key="your-secret-key" \
  --dry-run=client -o yaml | \
  kubeseal -o yaml > moe-lab-sealed-secrets.yml
```

## Performance Optimization

### Resource Allocation

```yaml
# k8s/production/optimized-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-lab-api-optimized
spec:
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/your-org/open-moe-trainer-lab:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            ephemeral-storage: "5Gi"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            ephemeral-storage: "10Gi"
        env:
        - name: WORKER_PROCESSES
          value: "4"
        - name: WORKER_CONNECTIONS
          value: "1000"
        - name: KEEPALIVE_TIMEOUT
          value: "30"
        - name: MAX_REQUESTS
          value: "1000"
        - name: MAX_REQUESTS_JITTER
          value: "100"
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: model-cache
          mountPath: /app/model_cache
      volumes:
      - name: tmp-volume
        emptyDir:
          sizeLimit: 1Gi
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
      nodeSelector:
        node-type: compute-optimized
      tolerations:
      - key: compute-optimized
        operator: Equal
        value: "true"
        effect: NoSchedule
```

### Caching Strategy

```yaml
# k8s/production/redis-cluster.yml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: moe-lab-cache
spec:
  clusterSize: 6
  persistenceEnabled: true
  redisExporter:
    enabled: true
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
        storageClassName: fast-ssd
  resources:
    requests:
      memory: "2Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "1000m"
```

## Troubleshooting

### Common Issues and Solutions

#### Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n moe-lab-production

# Describe pod for events
kubectl describe pod <pod-name> -n moe-lab-production

# Check logs
kubectl logs <pod-name> -n moe-lab-production --previous

# Debug with interactive shell
kubectl exec -it <pod-name> -n moe-lab-production -- /bin/bash
```

#### GPU Training Issues

```bash
# Verify GPU availability
kubectl exec -it <training-pod> -- nvidia-smi

# Check GPU device plugin
kubectl get pods -n kube-system | grep nvidia

# Verify GPU resources
kubectl describe node <gpu-node-name>

# Check NCCL communication
kubectl exec -it <training-pod> -- python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device()}')
"
```

#### Network Connectivity Issues

```bash
# Test service connectivity
kubectl exec -it <pod-name> -- curl http://service-name:port/health

# Check DNS resolution
kubectl exec -it <pod-name> -- nslookup service-name

# Test external connectivity
kubectl exec -it <pod-name> -- curl -I https://api.github.com

# Check network policies
kubectl get networkpolicies -n moe-lab-production
```

#### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n moe-lab-production

# Describe PVC for events
kubectl describe pvc <pvc-name> -n moe-lab-production

# Check storage class
kubectl get storageclass

# Test volume mount
kubectl exec -it <pod-name> -- df -h /mount/path
```

### Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n moe-lab-production

# Check resource usage
kubectl top pods -n moe-lab-production
kubectl top nodes

# View events
kubectl get events -n moe-lab-production --sort-by=.metadata.creationTimestamp

# Port forward for local debugging
kubectl port-forward svc/moe-lab-api-service 8000:80 -n moe-lab-production

# Scale deployment
kubectl scale deployment moe-lab-api --replicas=5 -n moe-lab-production

# Rollback deployment
kubectl rollout undo deployment/moe-lab-api -n moe-lab-production

# Check rollout status
kubectl rollout status deployment/moe-lab-api -n moe-lab-production
```

### Performance Debugging

```bash
# Check resource utilization
kubectl exec -it <pod-name> -- top
kubectl exec -it <pod-name> -- iostat -x 1

# Profile Python application
kubectl exec -it <pod-name> -- python -m cProfile -o profile.pstats -m moe_lab.main

# Memory profiling
kubectl exec -it <pod-name> -- python -m memory_profiler moe_lab/train.py

# Network debugging
kubectl exec -it <pod-name> -- netstat -tuln
kubectl exec -it <pod-name> -- ss -tuln
```

---

This deployment guide provides comprehensive coverage of deployment strategies and configurations. For specific questions or issues not covered here, please refer to the troubleshooting section or create an issue with the appropriate labels.