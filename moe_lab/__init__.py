"""Open MoE Trainer Lab - End-to-end training toolkit for Mixture of Experts models."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from .models import MoEModel
from .training import MoETrainer
from .inference import OptimizedMoEModel

__all__ = [
    "MoEModel",
    "MoETrainer", 
    "OptimizedMoEModel",
    "__version__",
]