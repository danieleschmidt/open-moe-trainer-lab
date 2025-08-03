"""Data collators for MoE training."""

from typing import List, Dict, Any, Optional, Union
import torch
from dataclasses import dataclass


@dataclass
class MoEDataCollator:
    """Data collator for MoE models with proper padding and batching."""
    
    pad_token_id: int = 0
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features into tensors."""
        
        batch = {}
        
        # Handle input_ids
        if 'input_ids' in features[0]:
            input_ids = [f['input_ids'] for f in features]
            batch['input_ids'] = self._pad_sequence(input_ids)
            
        # Handle attention_mask
        if 'attention_mask' in features[0]:
            attention_masks = [f['attention_mask'] for f in features]
            batch['attention_mask'] = self._pad_sequence(attention_masks, pad_value=0)
        elif 'input_ids' in batch:
            # Create attention mask from input_ids
            batch['attention_mask'] = (batch['input_ids'] != self.pad_token_id).long()
            
        # Handle labels
        if 'labels' in features[0]:
            labels = [f['labels'] for f in features]
            batch['labels'] = self._pad_sequence(labels, pad_value=-100)
            
        # Handle task_ids
        if 'task_ids' in features[0]:
            task_ids = [f['task_ids'] for f in features]
            batch['task_ids'] = torch.stack(task_ids)
            
        # Handle domain_ids
        if 'domain_ids' in features[0]:
            domain_ids = [f['domain_ids'] for f in features]
            batch['domain_ids'] = torch.stack(domain_ids)
            
        return batch
        
    def _pad_sequence(
        self, 
        sequences: List[torch.Tensor], 
        pad_value: int = None
    ) -> torch.Tensor:
        """Pad sequences to same length."""
        
        if pad_value is None:
            pad_value = self.pad_token_id
            
        # Get max length
        max_len = max(len(seq) for seq in sequences)
        
        # Apply max_length if specified
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
            
        # Apply padding to multiple
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_len:
                # Truncate if too long
                padded_seq = seq[:max_len]
            else:
                # Pad if too short
                padding = torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
                
            padded_sequences.append(padded_seq)
            
        return torch.stack(padded_sequences)


@dataclass  
class MoEInstructionCollator:
    """Collator for instruction-following datasets."""
    
    tokenizer: Any
    max_length: int = 512
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id or 0
            
    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Collate instruction-response pairs."""
        
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        for feature in features:
            instruction = feature.get('instruction', '')
            response = feature.get('response', '')
            
            # Format as instruction-response pair
            text = f"Instruction: {instruction}\nResponse: {response}"
            
            # Tokenize  
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            batch['input_ids'].append(torch.tensor(encoding['input_ids']))
            batch['attention_mask'].append(torch.tensor(encoding['attention_mask']))
            
            # For instruction tuning, typically mask the instruction part
            labels = encoding['input_ids'].copy()
            
            # Find the response start (simplified)
            response_start = text.find("Response:") + len("Response: ")
            instruction_tokens = len(self.tokenizer.encode(text[:response_start]))
            
            # Mask instruction tokens
            labels[:instruction_tokens] = [-100] * instruction_tokens
            batch['labels'].append(torch.tensor(labels))
            
        # Pad sequences
        collator = MoEDataCollator(pad_token_id=self.pad_token_id)
        return collator(batch)


class DynamicPaddingCollator:
    """Collator with dynamic padding for efficient batching."""
    
    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        bucket_boundaries: Optional[List[int]] = None
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.bucket_boundaries = bucket_boundaries or [64, 128, 256, 512, 1024]
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with dynamic padding based on batch sequence lengths."""
        
        # Find the length bucket for this batch
        lengths = [len(f['input_ids']) for f in features]
        max_batch_length = max(lengths)
        
        # Find appropriate bucket
        target_length = max_batch_length
        for boundary in self.bucket_boundaries:
            if max_batch_length <= boundary:
                target_length = boundary
                break
                
        # Apply max_length limit
        if self.max_length is not None:
            target_length = min(target_length, self.max_length)
        
        # Create temporary collator with target length
        temp_collator = MoEDataCollator(
            pad_token_id=self.pad_token_id,
            max_length=target_length
        )
        
        return temp_collator(features)


class MultiTaskCollator:
    """Collator for multi-task training."""
    
    def __init__(
        self,
        task_collators: Dict[str, Any],
        default_collator: Optional[Any] = None
    ):
        self.task_collators = task_collators
        self.default_collator = default_collator or MoEDataCollator()
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features based on task type."""
        
        # Group features by task
        task_groups = {}
        for i, feature in enumerate(features):
            task = feature.get('task', 'default')
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append((i, feature))
            
        # Process each task group
        batch = {}
        indices = []
        
        for task, task_features in task_groups.items():
            task_indices, task_data = zip(*task_features)
            
            # Get appropriate collator
            collator = self.task_collators.get(task, self.default_collator)
            
            # Collate task data
            task_batch = collator(list(task_data))
            
            # Add task-specific prefix to keys
            for key, value in task_batch.items():
                batch_key = f"{task}_{key}" if len(task_groups) > 1 else key
                if batch_key not in batch:
                    batch[batch_key] = []
                batch[batch_key].append(value)
                
            indices.extend(task_indices)
            
        # Concatenate task batches and reorder
        final_batch = {}
        for key, values in batch.items():
            concatenated = torch.cat(values, dim=0)
            
            # Reorder to original sequence
            reorder_indices = torch.tensor(indices).argsort()
            final_batch[key] = concatenated[reorder_indices]
            
        return final_batch