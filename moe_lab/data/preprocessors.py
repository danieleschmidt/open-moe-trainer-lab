"""Text preprocessing utilities for MoE training."""

from typing import List, Dict, Any, Optional, Callable, Union
import re
import html
import unicodedata
from dataclasses import dataclass


@dataclass
class TextPreprocessor:
    """Configurable text preprocessor for MoE training data."""
    
    lowercase: bool = False
    remove_html: bool = True
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = False
    custom_patterns: Optional[List[tuple]] = None  # (pattern, replacement)
    
    def __post_init__(self):
        """Initialize regex patterns."""
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def __call__(self, text: str) -> str:
        """Process a single text string."""
        return self.process_text(text)
        
    def process_text(self, text: str) -> str:
        """Apply all preprocessing steps to text."""
        
        if not isinstance(text, str):
            return ""
            
        # Remove HTML tags and entities
        if self.remove_html:
            text = self._remove_html(text)
            
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
            
        # Remove phone numbers
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub('', text)
            
        # Apply custom patterns
        if self.custom_patterns:
            for pattern, replacement in self.custom_patterns:
                if isinstance(pattern, str):
                    pattern = re.compile(pattern)
                text = pattern.sub(replacement, text)
                
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
            
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
            
        return text
        
    def process_texts(self, texts: List[str]) -> List[str]:
        """Process a list of texts."""
        processed = []
        
        for text in texts:
            processed_text = self.process_text(text)
            
            # Apply length filters
            if self.min_length and len(processed_text) < self.min_length:
                continue
            if self.max_length and len(processed_text) > self.max_length:
                processed_text = processed_text[:self.max_length]
                
            processed.append(processed_text)
            
        return processed
        
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and entities."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text


class DomainAwarePreprocessor:
    """Preprocessor that applies different rules based on text domain."""
    
    def __init__(
        self,
        domain_processors: Dict[str, TextPreprocessor],
        default_processor: Optional[TextPreprocessor] = None,
        domain_detector: Optional[Callable[[str], str]] = None
    ):
        self.domain_processors = domain_processors
        self.default_processor = default_processor or TextPreprocessor()
        self.domain_detector = domain_detector or self._simple_domain_detector
        
    def __call__(self, text: str, domain: Optional[str] = None) -> str:
        """Process text with domain-specific rules."""
        
        if domain is None:
            domain = self.domain_detector(text)
            
        processor = self.domain_processors.get(domain, self.default_processor)
        return processor.process_text(text)
        
    def _simple_domain_detector(self, text: str) -> str:
        """Simple domain detection based on keywords."""
        
        text_lower = text.lower()
        
        # Math/science keywords
        if any(word in text_lower for word in ['equation', 'formula', 'theorem', 'proof', 'calculate']):
            return 'math'
        if any(word in text_lower for word in ['experiment', 'hypothesis', 'research', 'study', 'analysis']):
            return 'science'
        if any(word in text_lower for word in ['def ', 'class ', 'import ', 'function', 'algorithm']):
            return 'code'
        if any(word in text_lower for word in ['novel', 'poetry', 'story', 'author', 'character']):
            return 'literature'
            
        return 'general'


class InstructionPreprocessor:
    """Preprocessor for instruction-following datasets."""
    
    def __init__(
        self,
        instruction_template: str = "Instruction: {instruction}\nResponse: {response}",
        max_instruction_length: int = 256,
        max_response_length: int = 512,
        text_processor: Optional[TextPreprocessor] = None
    ):
        self.instruction_template = instruction_template
        self.max_instruction_length = max_instruction_length
        self.max_response_length = max_response_length
        self.text_processor = text_processor or TextPreprocessor(
            remove_html=True,
            normalize_unicode=True,
            remove_extra_whitespace=True
        )
        
    def __call__(
        self, 
        instruction: str, 
        response: str,
        **kwargs
    ) -> Dict[str, str]:
        """Process instruction-response pair."""
        
        # Clean instruction and response
        clean_instruction = self.text_processor.process_text(instruction)
        clean_response = self.text_processor.process_text(response)
        
        # Apply length limits
        if len(clean_instruction) > self.max_instruction_length:
            clean_instruction = clean_instruction[:self.max_instruction_length]
            
        if len(clean_response) > self.max_response_length:
            clean_response = clean_response[:self.max_response_length]
            
        # Format with template
        formatted_text = self.instruction_template.format(
            instruction=clean_instruction,
            response=clean_response,
            **kwargs
        )
        
        return {
            'text': formatted_text,
            'instruction': clean_instruction,
            'response': clean_response
        }


class TokenizerPreprocessor:
    """Preprocessor that handles tokenization-specific tasks."""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        stride: int = 256,
        add_special_tokens: bool = True,
        return_overflowing_tokens: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_special_tokens = add_special_tokens
        self.return_overflowing_tokens = return_overflowing_tokens
        
    def __call__(self, text: str) -> Dict[str, Any]:
        """Tokenize text with proper handling of long sequences."""
        
        # Basic tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            stride=self.stride if self.return_overflowing_tokens else None,
            truncation=True,
            padding=False,
            add_special_tokens=self.add_special_tokens,
            return_overflowing_tokens=self.return_overflowing_tokens,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        result = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
        
        # Handle overflowing tokens
        if self.return_overflowing_tokens and 'overflowing_tokens' in encoding:
            result['overflowing_tokens'] = encoding['overflowing_tokens']
            result['num_truncated_tokens'] = encoding['num_truncated_tokens']
            
        # Add offset mapping if available
        if 'offset_mapping' in encoding:
            result['offset_mapping'] = encoding['offset_mapping']
            
        return result


class QualityFilter:
    """Filter for text quality assessment."""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        min_words: int = 5,
        max_repetition_ratio: float = 0.3,
        min_unique_words_ratio: float = 0.5,
        banned_words: Optional[List[str]] = None,
        required_patterns: Optional[List[str]] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_unique_words_ratio = min_unique_words_ratio
        self.banned_words = set(banned_words or [])
        self.required_patterns = [re.compile(p) for p in (required_patterns or [])]
        
    def __call__(self, text: str) -> bool:
        """Check if text passes quality filters."""
        
        # Length checks
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
            
        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            return False
            
        # Repetition check
        if self._check_repetition(text):
            return False
            
        # Unique words ratio
        unique_words = set(words)
        if len(unique_words) / len(words) < self.min_unique_words_ratio:
            return False
            
        # Banned words check
        if any(word.lower() in self.banned_words for word in words):
            return False
            
        # Required patterns check
        if self.required_patterns:
            if not all(pattern.search(text) for pattern in self.required_patterns):
                return False
                
        return True
        
    def _check_repetition(self, text: str) -> bool:
        """Check for excessive repetition in text."""
        
        # Simple repetition detection
        lines = text.split('\n')
        if len(lines) > 1:
            # Check for repeated lines
            line_counts = {}
            for line in lines:
                line = line.strip()
                if line:
                    line_counts[line] = line_counts.get(line, 0) + 1
                    
            max_repetitions = max(line_counts.values()) if line_counts else 0
            if max_repetitions / len(lines) > self.max_repetition_ratio:
                return True
                
        # Check for repeated character sequences
        for i in range(len(text) - 10):
            substring = text[i:i+10]
            count = text.count(substring)
            if count > 3:  # Arbitrary threshold
                return True
                
        return False