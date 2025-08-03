"""MoE model trainer implementation."""

from typing import Optional, Dict, Any, Union, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import logging
import time
import os
from tqdm import tqdm
import json

from ..models import MoEModel, MoEOutput
from ..utils.logging import get_logger


logger = get_logger(__name__)


class TrainingResult:
    """Container for training results."""
    
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.loss = metrics.get('loss', 0.0)
        self.aux_loss = metrics.get('aux_loss', 0.0)
        self.router_z_loss = metrics.get('router_z_loss', 0.0)
        self.expert_load_variance = metrics.get('expert_load_variance', 0.0)
        self.routing_entropy = metrics.get('routing_entropy', 0.0)


class EvalResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.eval_loss = metrics.get('eval_loss', 0.0)
        self.perplexity = metrics.get('perplexity', 0.0)


class MoETrainer:
    """Trainer for MoE models with load balancing and distributed support."""
    
    def __init__(
        self,
        model: MoEModel,
        load_balancing: str = "auxiliary_loss",
        router_z_loss_coef: float = 0.01,
        aux_loss_coef: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        logging_steps: int = 10,
        save_steps: int = 1000,
        eval_steps: int = 500,
        output_dir: str = "./checkpoints",
        use_mixed_precision: bool = True,
        dataloader_num_workers: int = 4,
        **kwargs
    ):
        self.model = model
        self.load_balancing = load_balancing
        self.router_z_loss_coef = router_z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.dataloader_num_workers = dataloader_num_workers
        
        # Training state
        self.history = {
            'train_loss': [],
            'aux_loss': [],
            'router_z_loss': [],
            'expert_load_variance': [],
            'routing_entropy': [],
            'eval_loss': [],
            'perplexity': []
        }
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup mixed precision training
        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Distributed training setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1
            
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        **training_args
    ) -> TrainingResult:
        """Train the MoE model with comprehensive logging and checkpointing."""
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Setup data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            pin_memory=True,
            collate_fn=getattr(train_dataset, 'collate_fn', None)
        )
        
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
                collate_fn=getattr(eval_dataset, 'collate_fn', None)
            )
        
        # Setup optimizer
        if optimizer is None:
            # Use AdamW with different learning rates for router and experts
            router_params = []
            expert_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if 'router' in name:
                    router_params.append(param)
                elif 'expert' in name:
                    expert_params.append(param)
                else:
                    other_params.append(param)
                    
            param_groups = [
                {'params': other_params, 'lr': learning_rate},
                {'params': expert_params, 'lr': learning_rate * 0.5},  # Lower LR for experts
                {'params': router_params, 'lr': learning_rate * 2.0}   # Higher LR for routers
            ]
            
            optimizer = optim.AdamW(
                param_groups,
                lr=learning_rate,
                weight_decay=0.01,
                eps=1e-8
            )
            
        # Setup scheduler
        if scheduler is None:
            total_steps = len(train_dataloader) * num_epochs
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        
        # Training loop
        self.model.train()
        device = next(self.model.parameters()).device
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_aux_loss = 0.0
            epoch_router_z_loss = 0.0
            
            pbar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=(self.local_rank != 0)
            )
            
            for step, batch in enumerate(pbar):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                    
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    if isinstance(batch, dict):
                        outputs: MoEOutput = self.model(**batch)
                    else:
                        outputs: MoEOutput = self.model(batch)
                    
                    # Compute language modeling loss
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = self.model.lm_head(outputs.last_hidden_state)
                        
                    if isinstance(batch, dict) and 'labels' in batch:
                        labels = batch['labels']
                    else:
                        # Use input_ids shifted by one as labels
                        input_ids = batch if not isinstance(batch, dict) else batch['input_ids']
                        labels = input_ids[:, 1:].contiguous()
                        logits = logits[:, :-1].contiguous()
                    
                    # Compute cross-entropy loss
                    loss_fct = nn.CrossEntropyLoss()
                    lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # Add auxiliary losses
                    total_loss = lm_loss
                    aux_loss = 0.0
                    router_z_loss = 0.0
                    
                    if outputs.load_balancing_loss is not None:
                        aux_loss = outputs.load_balancing_loss
                        total_loss += self.aux_loss_coef * aux_loss
                        
                    if outputs.router_z_loss is not None:
                        router_z_loss = outputs.router_z_loss
                        total_loss += self.router_z_loss_coef * router_z_loss
                        
                    # Scale loss for gradient accumulation
                    total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                    
                # Update statistics
                epoch_loss += lm_loss.item()
                epoch_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                epoch_router_z_loss += router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss
                
                # Gradient accumulation step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_mixed_precision:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        
                        # Get routing statistics
                        routing_stats = self._get_routing_stats(outputs.routing_info)
                        
                        log_dict = {
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'lr': lr,
                            'loss': lm_loss.item(),
                            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                            'router_z_loss': router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss,
                            **routing_stats
                        }
                        
                        # Log to history
                        self.history['train_loss'].append(lm_loss.item())
                        self.history['aux_loss'].append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
                        self.history['router_z_loss'].append(router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss)
                        self.history['expert_load_variance'].append(routing_stats.get('load_variance', 0.0))
                        self.history['routing_entropy'].append(routing_stats.get('entropy', 0.0))
                        
                        if self.local_rank == 0:
                            logger.info(f"Step {self.global_step}: {log_dict}")
                            
                    # Evaluation
                    if eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_result = self.evaluate(eval_dataset)
                        self.history['eval_loss'].append(eval_result.eval_loss)
                        self.history['perplexity'].append(eval_result.perplexity)
                        
                        if self.local_rank == 0:
                            logger.info(f"Eval at step {self.global_step}: {eval_result.metrics}")
                            
                    # Checkpointing
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                        
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{lm_loss.item():.4f}",
                    'aux': f"{aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}",
                    'step': self.global_step
                })
                
        # End of epoch logging
        avg_loss = epoch_loss / len(train_dataloader)
        avg_aux_loss = epoch_aux_loss / len(train_dataloader)
        avg_router_z_loss = epoch_router_z_loss / len(train_dataloader)
        
        if self.local_rank == 0:
            logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}, Aux loss: {avg_aux_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        # Final evaluation
        if eval_dataloader is not None:
            final_eval = self.evaluate(eval_dataset)
        else:
            final_eval = EvalResult({'eval_loss': avg_loss})
            
        return TrainingResult({
            'loss': avg_loss,
            'aux_loss': avg_aux_loss,
            'router_z_loss': avg_router_z_loss,
            'final_eval': final_eval.metrics,
            'total_steps': self.global_step
        })
        
    def evaluate(self, eval_dataset: Dataset, batch_size: int = 16) -> EvalResult:
        """Evaluate the model on validation set."""
        self.model.eval()
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=getattr(eval_dataset, 'collate_fn', None)
        )
        
        total_loss = 0.0
        total_tokens = 0
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=(self.local_rank != 0)):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                    
                # Forward pass
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                else:
                    outputs = self.model(batch)
                    
                # Compute loss
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = self.model.lm_head(outputs.last_hidden_state)
                    
                if isinstance(batch, dict) and 'labels' in batch:
                    labels = batch['labels']
                else:
                    input_ids = batch if not isinstance(batch, dict) else batch['input_ids']
                    labels = input_ids[:, 1:].contiguous()
                    logits = logits[:, :-1].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                total_tokens += labels.numel()
                
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        
        return EvalResult({
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens
        })
        
    def save_model(self, path: str) -> None:
        """Save model checkpoint with training state."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        model_state = self.model.state_dict()
        torch.save(model_state, os.path.join(path, "pytorch_model.bin"))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'history': self.history
        }
        with open(os.path.join(path, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
            
        if self.local_rank == 0:
            logger.info(f"Model saved to {path}")
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        self.save_model(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        # Load model state
        model_state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_state_path):
            state_dict = torch.load(model_state_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
                self.global_step = training_state.get('global_step', 0)
                self.current_epoch = training_state.get('current_epoch', 0)
                self.history = training_state.get('history', self.history)
                
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def _get_routing_stats(self, routing_info) -> Dict[str, float]:
        """Extract routing statistics from routing info."""
        if routing_info is None:
            return {'load_variance': 0.0, 'entropy': 0.0}
            
        return {
            'load_variance': routing_info.load_variance,
            'entropy': routing_info.entropy
        }
        
    def visualize_routing(self, save_to: str = "routing_analysis.html") -> None:
        """Generate routing visualization (placeholder for now)."""
        logger.info(f"Routing visualization will be saved to {save_to}")
        # Implementation will be added in visualization module