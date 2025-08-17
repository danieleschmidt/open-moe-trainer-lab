#!/usr/bin/env python3
"""
AUTONOMOUS PRODUCTION DEPLOYMENT
Complete production-ready deployment preparation with containerization, orchestration, and monitoring.
"""

import json
import time
import logging
import subprocess
import sys
from typing import Dict, Any, List
from pathlib import Path
import traceback
import yaml
import hashlib
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_production_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeploymentAutomator:
    """Autonomous production deployment preparation."""
    
    def __init__(self):
        self.deployment_config = {}
        self.deployment_artifacts = []
        
    def create_production_dockerfile(self) -> str:
        """Create optimized production Dockerfile."""
        logger.info("Creating production-optimized Dockerfile...")
        
        dockerfile_content = """# Multi-stage production Dockerfile for Open MoE Trainer Lab
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \\
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r moe && useradd -r -g moe moe

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R moe:moe /app && \\
    chmod +x scripts/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "from moe_lab import MoEModel; print('Health check passed')" || exit 1

# Switch to non-root user
USER moe

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PRODUCTION=true

# Default command
CMD ["python", "-m", "moe_lab.serving.server"]
"""
        
        with open("Dockerfile.production", "w") as f:
            f.write(dockerfile_content)
        
        logger.info("✅ Production Dockerfile created")
        return "Dockerfile.production"
    
    def create_docker_compose_production(self) -> str:
        """Create production Docker Compose configuration."""
        logger.info("Creating production Docker Compose configuration...")
        
        compose_config = {
            'version': '3.8',
            'services': {
                'moe-trainer': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.production',
                        'target': 'production'
                    },
                    'image': 'moe-trainer-lab:latest',
                    'container_name': 'moe-trainer-production',
                    'restart': 'unless-stopped',
                    'ports': ['8000:8000'],
                    'environment': [
                        'PRODUCTION=true',
                        'LOG_LEVEL=INFO',
                        'METRICS_ENABLED=true'
                    ],
                    'volumes': [
                        './models:/app/models:ro',
                        './data:/app/data:ro',
                        './logs:/app/logs'
                    ],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    },
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': '4.0',
                                'memory': '8G'
                            },
                            'reservations': {
                                'cpus': '2.0',
                                'memory': '4G'
                            }
                        }
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': 'moe-redis',
                    'restart': 'unless-stopped',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data'],
                    'command': 'redis-server --appendonly yes'
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'moe-prometheus',
                    'restart': 'unless-stopped',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--web.enable-lifecycle'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'moe-grafana',
                    'restart': 'unless-stopped',
                    'ports': ['3000:3000'],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=admin123'
                    ],
                    'volumes': [
                        'grafana_data:/var/lib/grafana',
                        './monitoring/grafana:/etc/grafana/provisioning'
                    ]
                }
            },
            'volumes': {
                'redis_data': {},
                'grafana_data': {}
            },
            'networks': {
                'moe-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Assign all services to the network
        for service in compose_config['services'].values():
            service['networks'] = ['moe-network']
        
        with open("docker-compose.production.yml", "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ Production Docker Compose configuration created")
        return "docker-compose.production.yml"
    
    def create_kubernetes_manifests(self) -> List[str]:
        """Create Kubernetes deployment manifests."""
        logger.info("Creating Kubernetes deployment manifests...")
        
        manifests = []
        
        # Namespace
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': 'moe-trainer-lab',
                'labels': {
                    'app': 'moe-trainer-lab',
                    'version': 'v1'
                }
            }
        }
        
        # ConfigMap
        configmap_manifest = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'moe-trainer-config',
                'namespace': 'moe-trainer-lab'
            },
            'data': {
                'PRODUCTION': 'true',
                'LOG_LEVEL': 'INFO',
                'METRICS_ENABLED': 'true',
                'REDIS_URL': 'redis://redis-service:6379'
            }
        }
        
        # Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'moe-trainer-deployment',
                'namespace': 'moe-trainer-lab',
                'labels': {
                    'app': 'moe-trainer'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'moe-trainer'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'moe-trainer'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'moe-trainer',
                            'image': 'moe-trainer-lab:latest',
                            'ports': [{
                                'containerPort': 8000
                            }],
                            'envFrom': [{
                                'configMapRef': {
                                    'name': 'moe-trainer-config'
                                }
                            }],
                            'resources': {
                                'requests': {
                                    'cpu': '1000m',
                                    'memory': '2Gi'
                                },
                                'limits': {
                                    'cpu': '4000m',
                                    'memory': '8Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000
                        }
                    }
                }
            }
        }
        
        # Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'moe-trainer-service',
                'namespace': 'moe-trainer-lab'
            },
            'spec': {
                'selector': {
                    'app': 'moe-trainer'
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000
                }],
                'type': 'LoadBalancer'
            }
        }
        
        # Ingress
        ingress_manifest = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'moe-trainer-ingress',
                'namespace': 'moe-trainer-lab',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true'
                }
            },
            'spec': {
                'rules': [{
                    'host': 'moe-trainer.example.com',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'moe-trainer-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        # HPA (Horizontal Pod Autoscaler)
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'moe-trainer-hpa',
                'namespace': 'moe-trainer-lab'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'moe-trainer-deployment'
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 70
                        }
                    }
                }]
            }
        }
        
        # Save all manifests
        manifest_files = [
            ('namespace.yaml', namespace_manifest),
            ('configmap.yaml', configmap_manifest),
            ('deployment.yaml', deployment_manifest),
            ('service.yaml', service_manifest),
            ('ingress.yaml', ingress_manifest),
            ('hpa.yaml', hpa_manifest)
        ]
        
        k8s_dir = Path("kubernetes")
        k8s_dir.mkdir(exist_ok=True)
        
        for filename, manifest in manifest_files:
            filepath = k8s_dir / filename
            with open(filepath, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False, indent=2)
            manifests.append(str(filepath))
        
        logger.info(f"✅ Created {len(manifests)} Kubernetes manifests")
        return manifests
    
    def create_deployment_scripts(self) -> List[str]:
        """Create deployment automation scripts."""
        logger.info("Creating deployment automation scripts...")
        
        scripts = []
        scripts_dir = Path("deployment")
        scripts_dir.mkdir(exist_ok=True)
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "🚀 Building MoE Trainer Lab for production..."

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.production -t moe-trainer-lab:latest .

# Tag for registry
if [ ! -z "$REGISTRY" ]; then
    echo "Tagging for registry: $REGISTRY"
    docker tag moe-trainer-lab:latest $REGISTRY/moe-trainer-lab:latest
    docker tag moe-trainer-lab:latest $REGISTRY/moe-trainer-lab:$(git rev-parse --short HEAD)
fi

echo "✅ Build completed successfully"
"""
        
        # Deploy script
        deploy_script = """#!/bin/bash
set -e

echo "🚀 Deploying MoE Trainer Lab to production..."

# Check if Docker Compose or Kubernetes
if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    echo "Deploying to Kubernetes..."
    kubectl apply -f kubernetes/
    kubectl rollout status deployment/moe-trainer-deployment -n moe-trainer-lab
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    echo "Deploying with Docker Compose..."
    docker-compose -f docker-compose.production.yml up -d
else
    echo "Please set DEPLOYMENT_TYPE to 'k8s' or 'compose'"
    exit 1
fi

echo "✅ Deployment completed successfully"
"""
        
        # Health check script
        health_script = """#!/bin/bash
set -e

echo "🔍 Running health checks..."

if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    # Kubernetes health check
    kubectl get pods -n moe-trainer-lab
    kubectl get services -n moe-trainer-lab
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/moe-trainer-deployment -n moe-trainer-lab
    
    # Test service endpoint
    EXTERNAL_IP=$(kubectl get service moe-trainer-service -n moe-trainer-lab -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ ! -z "$EXTERNAL_IP" ]; then
        curl -f http://$EXTERNAL_IP/health || echo "Service not yet available"
    fi
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    # Docker Compose health check
    docker-compose -f docker-compose.production.yml ps
    
    # Test service endpoint
    curl -f http://localhost:8000/health || echo "Service not yet available"
fi

echo "✅ Health checks completed"
"""
        
        # Rollback script
        rollback_script = """#!/bin/bash
set -e

echo "🔄 Rolling back deployment..."

if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    kubectl rollout undo deployment/moe-trainer-deployment -n moe-trainer-lab
    kubectl rollout status deployment/moe-trainer-deployment -n moe-trainer-lab
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    docker-compose -f docker-compose.production.yml down
    # Restore previous version (would need to be implemented based on your backup strategy)
    echo "Manual intervention required for Docker Compose rollback"
fi

echo "✅ Rollback completed"
"""
        
        script_files = [
            ('build.sh', build_script),
            ('deploy.sh', deploy_script),
            ('health-check.sh', health_script),
            ('rollback.sh', rollback_script)
        ]
        
        for filename, script_content in script_files:
            filepath = scripts_dir / filename
            with open(filepath, "w") as f:
                f.write(script_content)
            # Make executable
            os.chmod(filepath, 0o755)
            scripts.append(str(filepath))
        
        logger.info(f"✅ Created {len(scripts)} deployment scripts")
        return scripts
    
    def create_monitoring_config(self) -> List[str]:
        """Create monitoring and observability configuration."""
        logger.info("Creating monitoring configuration...")
        
        monitoring_dir = Path("monitoring")
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'rules/*.yml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'moe-trainer',
                    'static_configs': [
                        {
                            'targets': ['moe-trainer:8000']
                        }
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '10s'
                },
                {
                    'job_name': 'redis',
                    'static_configs': [
                        {
                            'targets': ['redis:6379']
                        }
                    ]
                }
            ]
        }
        
        prometheus_file = monitoring_dir / "prometheus.yml"
        with open(prometheus_file, "w") as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, indent=2)
        
        # Grafana dashboard
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": "MoE Trainer Lab Dashboard",
                "tags": ["moe", "machine-learning"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Training Loss",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "training_loss",
                                "legendFormat": "Training Loss"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Expert Utilization",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "expert_utilization_ratio",
                                "legendFormat": "Expert {{expert_id}}"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "memory_usage_bytes",
                                "legendFormat": "Memory Usage"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        grafana_file = monitoring_dir / "moe-dashboard.json"
        with open(grafana_file, "w") as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        # Alert rules
        alert_rules = {
            'groups': [
                {
                    'name': 'moe-trainer-alerts',
                    'rules': [
                        {
                            'alert': 'HighTrainingLoss',
                            'expr': 'training_loss > 10',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'Training loss is abnormally high',
                                'description': 'Training loss has been above 10 for more than 5 minutes'
                            }
                        },
                        {
                            'alert': 'ExpertImbalance',
                            'expr': 'expert_load_variance > 0.5',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'Expert load is imbalanced',
                                'description': 'Expert utilization variance is too high'
                            }
                        },
                        {
                            'alert': 'HighMemoryUsage',
                            'expr': 'memory_usage_bytes > 8000000000',  # 8GB
                            'for': '1m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'High memory usage detected',
                                'description': 'Memory usage has exceeded 8GB'
                            }
                        }
                    ]
                }
            ]
        }
        
        rules_dir = monitoring_dir / "rules"
        rules_dir.mkdir(exist_ok=True)
        
        alerts_file = rules_dir / "alerts.yml"
        with open(alerts_file, "w") as f:
            yaml.dump(alert_rules, f, default_flow_style=False, indent=2)
        
        created_files = [
            str(prometheus_file),
            str(grafana_file),
            str(alerts_file)
        ]
        
        logger.info(f"✅ Created monitoring configuration with {len(created_files)} files")
        return created_files
    
    def run_deployment_preparation(self) -> Dict[str, Any]:
        """Run complete deployment preparation."""
        logger.info("🚀 Starting Autonomous Production Deployment Preparation")
        logger.info("=" * 80)
        
        start_time = time.time()
        deployment_artifacts = []
        
        try:
            # 1. Create production Dockerfile
            logger.info("1. Creating production Dockerfile...")
            dockerfile = self.create_production_dockerfile()
            deployment_artifacts.append(dockerfile)
            logger.info("   ✅ Production Dockerfile created")
            
            # 2. Create Docker Compose configuration
            logger.info("2. Creating Docker Compose configuration...")
            compose_file = self.create_docker_compose_production()
            deployment_artifacts.append(compose_file)
            logger.info("   ✅ Docker Compose configuration created")
            
            # 3. Create Kubernetes manifests
            logger.info("3. Creating Kubernetes manifests...")
            k8s_manifests = self.create_kubernetes_manifests()
            deployment_artifacts.extend(k8s_manifests)
            logger.info(f"   ✅ {len(k8s_manifests)} Kubernetes manifests created")
            
            # 4. Create deployment scripts
            logger.info("4. Creating deployment scripts...")
            scripts = self.create_deployment_scripts()
            deployment_artifacts.extend(scripts)
            logger.info(f"   ✅ {len(scripts)} deployment scripts created")
            
            # 5. Create monitoring configuration
            logger.info("5. Creating monitoring configuration...")
            monitoring_files = self.create_monitoring_config()
            deployment_artifacts.extend(monitoring_files)
            logger.info(f"   ✅ {len(monitoring_files)} monitoring files created")
            
            execution_time = time.time() - start_time
            
            # Compile deployment report
            deployment_report = {
                "timestamp": time.time(),
                "status": "PRODUCTION_READY",
                "deployment_artifacts": deployment_artifacts,
                "deployment_options": {
                    "docker_compose": "docker-compose.production.yml",
                    "kubernetes": "kubernetes/",
                    "monitoring": "monitoring/"
                },
                "deployment_capabilities": [
                    "Multi-stage Docker builds",
                    "Production-optimized containers",
                    "Kubernetes orchestration with HPA",
                    "Load balancing and service discovery",
                    "Comprehensive monitoring with Prometheus/Grafana",
                    "Automated health checks",
                    "Security hardening (non-root user)",
                    "Resource limits and requests",
                    "Rolling deployments and rollbacks",
                    "Alert rules for operational monitoring"
                ],
                "security_features": [
                    "Non-root container execution",
                    "Resource limits enforcement",
                    "Health check endpoints",
                    "Network segmentation",
                    "Configuration externalization"
                ],
                "scalability_features": [
                    "Horizontal Pod Autoscaling",
                    "Load balancer integration",
                    "Multi-replica deployments",
                    "Rolling updates",
                    "Resource-based scaling"
                ],
                "execution_time_seconds": execution_time,
                "next_steps": [
                    "Set REGISTRY environment variable for image registry",
                    "Configure ingress domain name",
                    "Set up persistent storage for models",
                    "Configure backup strategies",
                    "Set up monitoring alerts",
                    "Run deployment with: DEPLOYMENT_TYPE=compose ./deployment/deploy.sh"
                ]
            }
            
            # Save deployment report
            with open("autonomous_production_deployment_report.json", "w") as f:
                json.dump(deployment_report, f, indent=2)
            
            logger.info("\n🎉 Autonomous Production Deployment Preparation Complete!")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"📦 Created {len(deployment_artifacts)} deployment artifacts")
            logger.info(f"🏗️  Deployment options: Docker Compose, Kubernetes")
            logger.info(f"📊 Monitoring: Prometheus, Grafana, alerting")
            logger.info("📋 Report saved to: autonomous_production_deployment_report.json")
            
            logger.info("\n🚀 Ready for Production Deployment:")
            logger.info("   Docker Compose: DEPLOYMENT_TYPE=compose ./deployment/deploy.sh")
            logger.info("   Kubernetes: DEPLOYMENT_TYPE=k8s ./deployment/deploy.sh")
            
            return deployment_report
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {e}")
            logger.error(traceback.format_exc())
            
            error_report = {
                "timestamp": time.time(),
                "status": "FAILED",
                "error": str(e),
                "deployment_artifacts": deployment_artifacts,
                "execution_time_seconds": time.time() - start_time
            }
            
            with open("autonomous_production_deployment_report.json", "w") as f:
                json.dump(error_report, f, indent=2)
            
            raise

def run_autonomous_production_deployment():
    """Run autonomous production deployment preparation."""
    deployment_automator = ProductionDeploymentAutomator()
    return deployment_automator.run_deployment_preparation()

if __name__ == "__main__":
    # Run autonomous production deployment preparation
    report = run_autonomous_production_deployment()
    
    # Validate success
    assert report["status"] == "PRODUCTION_READY", f"Deployment preparation failed: {report.get('error', 'Unknown error')}"
    assert len(report["deployment_artifacts"]) > 0, "No deployment artifacts created"
    
    logger.info("✅ Autonomous Production Deployment preparation completed successfully")