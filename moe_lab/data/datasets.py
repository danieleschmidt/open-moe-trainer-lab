"""Dataset classes for MoE training."""

from typing import List, Optional, Dict, Any, Union
import torch
from torch.utils.data import Dataset
import random
import json


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 512,
        return_labels: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_labels = return_labels
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer is not None:
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
            
            if self.return_labels:
                # For language modeling, labels are the input_ids shifted by one
                item['labels'] = encoding['input_ids'].squeeze(0).clone()
                
        else:
            # Return raw text if no tokenizer
            item = {'text': text}
            
        return item


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized data."""
    
    def __init__(
        self,
        token_ids: List[List[int]],
        max_length: int = 512,
        pad_token_id: int = 0,
        return_labels: bool = True
    ):
        self.token_ids = token_ids
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.return_labels = return_labels
        
    def __len__(self):
        return len(self.token_ids)
        
    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            
        # Create attention mask
        attention_mask = [1 if token != self.pad_token_id else 0 for token in tokens]
        
        item = {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        if self.return_labels:
            item['labels'] = torch.tensor(tokens, dtype=torch.long)
            
        return item


class MoEDataset(Dataset):
    """Enhanced dataset with MoE-specific features."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 512,
        task_labels: Optional[List[str]] = None,
        domain_labels: Optional[List[str]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        self.task_labels = task_labels or []
        self.domain_labels = domain_labels or []
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        else:
            # Assume plain text file
            with open(data_path, 'r') as f:
                texts = f.read().strip().split('\n')
            return [{'text': text} for text in texts if text.strip()]
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            result = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0).clone()
            }
        else:
            result = {'text': text}
        
        # Add task information if available
        if 'task' in item and self.task_labels:
            task_id = self.task_labels.index(item['task']) if item['task'] in self.task_labels else 0
            result['task_ids'] = torch.tensor([task_id], dtype=torch.long)
            
        # Add domain information if available  
        if 'domain' in item and self.domain_labels:
            domain_id = self.domain_labels.index(item['domain']) if item['domain'] in self.domain_labels else 0
            result['domain_ids'] = torch.tensor([domain_id], dtype=torch.long)
            
        return result


def create_sample_dataset(
    num_samples: int = 1000,
    vocab_size: int = 1000,
    max_length: int = 128,
    min_length: int = 32,
    seed: int = 42
) -> TokenizedDataset:
    """Create a synthetic dataset for testing."""
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random token sequences
    token_sequences = []
    for _ in range(num_samples):
        seq_length = random.randint(min_length, max_length)
        tokens = [random.randint(1, vocab_size - 1) for _ in range(seq_length)]
        token_sequences.append(tokens)
        
    return TokenizedDataset(
        token_ids=token_sequences,
        max_length=max_length,
        pad_token_id=0
    )


def create_domain_dataset(
    domains: List[str] = None,
    samples_per_domain: int = 100,
    vocab_size: int = 1000,
    max_length: int = 128,
    seed: int = 42
) -> MoEDataset:
    """Create a synthetic multi-domain dataset."""
    
    if domains is None:
        domains = ['math', 'science', 'literature', 'history', 'technology']
        
    random.seed(seed)
    
    # Generate domain-specific samples
    data = []
    for domain in domains:
        for i in range(samples_per_domain):
            # Create domain-biased text (simplified)
            text_length = random.randint(50, 200)
            
            # Generate different patterns for different domains
            if domain == 'math':
                tokens = [random.randint(100, 200) for _ in range(text_length)]
            elif domain == 'science':
                tokens = [random.randint(200, 300) for _ in range(text_length)]
            elif domain == 'literature':
                tokens = [random.randint(300, 400) for _ in range(text_length)]
            elif domain == 'history':
                tokens = [random.randint(400, 500) for _ in range(text_length)]
            else:  # technology
                tokens = [random.randint(500, 600) for _ in range(text_length)]
                
            # Add some randomness
            tokens = [max(1, t + random.randint(-50, 50)) for t in tokens]
            tokens = [min(vocab_size - 1, t) for t in tokens]
            
            text = ' '.join(map(str, tokens))  # Simple token-to-text conversion
            
            data.append({
                'text': text,
                'domain': domain,
                'task': 'language_modeling'
            })
    
    # Save to temporary file for MoEDataset
    import tempfile
    import os
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, temp_file, indent=2)
    temp_file.close()
    
    dataset = MoEDataset(
        data_path=temp_file.name,
        task_labels=['language_modeling'],
        domain_labels=domains
    )
    
    # Clean up temp file
    os.unlink(temp_file.name)
    
    return dataset


class WikiTextDataset(Dataset):
    """WikiText-style dataset for language modeling."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Load and process text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Tokenize entire text
        tokens = tokenizer.encode(text)
        
        # Create overlapping chunks
        self.examples = []
        for i in range(0, len(tokens) - max_length + 1, stride):
            chunk = tokens[i:i + max_length]
            self.examples.append(chunk)
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }


def create_instruction_dataset(
    num_samples: int = 500,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Create a synthetic instruction-following dataset."""
    
    random.seed(seed)
    
    # Template instructions for different tasks
    templates = {
        'math': [
            "Calculate {a} + {b}",
            "What is {a} * {b}?",
            "Solve: {a} - {b} = ?",
            "Find the result of {a} / {b}"
        ],
        'translation': [
            "Translate '{text}' to Spanish",
            "Convert '{text}' into French", 
            "Translate the following to German: '{text}'",
            "What is '{text}' in Italian?"
        ],
        'summarization': [
            "Summarize the following text: {text}",
            "Give a brief summary of: {text}",
            "What are the key points in: {text}",
            "Condense this text: {text}"
        ],
        'question_answering': [
            "Answer: {question}",
            "Question: {question}",
            "What is the answer to: {question}",
            "Please answer: {question}"
        ]
    }
    
    # Sample responses
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Climate change affects global weather patterns",
        "The internet has revolutionized communication",
        "Renewable energy sources include solar and wind power"
    ]
    
    questions = [
        "What is the capital of France?",
        "How do computers work?",
        "What causes rain?",
        "Why is the sky blue?",
        "How do plants make food?"
    ]
    
    dataset = []
    
    for _ in range(num_samples):
        task = random.choice(list(templates.keys()))
        template = random.choice(templates[task])
        
        if task == 'math':
            a, b = random.randint(1, 100), random.randint(1, 100)
            instruction = template.format(a=a, b=b)
            
            if '+' in template:
                response = str(a + b)
            elif '*' in template:
                response = str(a * b)
            elif '-' in template:
                response = str(a - b)
            else:  # division
                response = str(round(a / b, 2))
                
        elif task == 'translation':
            text = random.choice(sample_texts)
            instruction = template.format(text=text)
            response = f"[Translation of '{text}']"  # Placeholder
            
        elif task == 'summarization':
            text = random.choice(sample_texts)
            instruction = template.format(text=text)
            response = f"Summary: {text[:50]}..."  # Simplified summary
            
        else:  # question_answering
            question = random.choice(questions)
            instruction = template.format(question=question)
            response = f"[Answer to '{question}']"  # Placeholder
            
        dataset.append({
            'instruction': instruction,
            'response': response,
            'task': task
        })
        
    return dataset