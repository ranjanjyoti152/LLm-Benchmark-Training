"""
Dataset utilities for LLM benchmarking.
Creates synthetic and real datasets for training benchmarks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)


class SyntheticTextDataset(Dataset):
    """
    Synthetic dataset for benchmarking purposes.
    Generates random text-like sequences for training.
    """
    
    def __init__(
        self, 
        tokenizer, 
        num_samples: int = 10000,
        max_length: int = 512,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            tokenizer: Tokenizer to use
            num_samples: Number of samples to generate
            max_length: Maximum sequence length
            vocab_size: Vocabulary size (uses tokenizer vocab size if None)
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size or len(tokenizer)
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[torch.Tensor]:
        """Generate synthetic token sequences."""
        logger.info(f"Generating {self.num_samples} synthetic samples...")
        
        data = []
        for i in range(self.num_samples):
            # Generate sequences of fixed max_length to avoid collation issues
            seq_len = self.max_length
            
            # Generate random token IDs (avoiding special tokens)
            token_ids = torch.randint(
                10,  # Start after special tokens
                min(self.vocab_size - 10, 50000),  # Reasonable upper bound
                (seq_len,),
                dtype=torch.long
            )
            
            data.append(token_ids)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{self.num_samples} samples")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        input_ids = self.data[idx]
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()  # For causal LM, labels = input_ids
        }


class TextFileDataset(Dataset):
    """
    Dataset that loads text from files for more realistic benchmarking.
    """
    
    def __init__(
        self, 
        file_paths: List[str],
        tokenizer,
        max_length: int = 512,
        overlap: int = 50
    ):
        """
        Initialize text file dataset.
        
        Args:
            file_paths: List of text file paths to load
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            overlap: Overlap between consecutive chunks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        
        # Load and tokenize all text files
        self.sequences = self._load_and_tokenize_files(file_paths)
        
    def _load_and_tokenize_files(self, file_paths: List[str]) -> List[torch.Tensor]:
        """Load text files and create tokenized sequences."""
        sequences = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize the entire text
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                
                # Split into overlapping chunks
                step = self.max_length - self.overlap
                for i in range(0, len(tokens) - self.max_length + 1, step):
                    chunk = tokens[i:i + self.max_length]
                    sequences.append(torch.tensor(chunk, dtype=torch.long))
                
                logger.info(f"Loaded {file_path}: {len(tokens)} tokens, {len(sequences)} sequences")
                
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        input_ids = self.sequences[idx]
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }


class BenchmarkDataLoader:
    """
    Data loader factory for benchmarking datasets.
    """
    
    @staticmethod
    def create_synthetic_dataloader(
        tokenizer,
        batch_size: int = 4,
        num_samples: int = 10000,
        max_length: int = 512,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create dataloader with synthetic data.
        
        Args:
            tokenizer: Tokenizer to use
            batch_size: Batch size
            num_samples: Number of synthetic samples
            max_length: Maximum sequence length
            num_workers: Number of data loading workers
            
        Returns:
            DataLoader instance
        """
        dataset = SyntheticTextDataset(
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_fn
        )
    
    @staticmethod
    def create_file_dataloader(
        file_paths: List[str],
        tokenizer,
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create dataloader from text files.
        
        Args:
            file_paths: List of text file paths
            tokenizer: Tokenizer to use
            batch_size: Batch size
            max_length: Maximum sequence length
            num_workers: Number of data loading workers
            
        Returns:
            DataLoader instance
        """
        dataset = TextFileDataset(
            file_paths=file_paths,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_fn
        )
    
    @staticmethod
    def create_quick_benchmark_data(
        tokenizer,
        model_size: str,
        num_samples: int = 1000
    ) -> List[torch.Tensor]:
        """
        Create a small dataset for quick benchmarking.
        
        Args:
            tokenizer: Tokenizer to use
            model_size: Model size to optimize data for
            num_samples: Number of samples to create
            
        Returns:
            List of tokenized sequences
        """
        # Adjust sequence length based on model size
        length_map = {
            "1B": 256,
            "7B": 512,
            "20B": 1024,
            "120B": 2048
        }
        
        max_length = length_map.get(model_size, 512)
        
        dataset = SyntheticTextDataset(
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length
        )
        
        return [dataset[i]["input_ids"] for i in range(len(dataset))]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable length sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched and padded tensors
    """
    # Extract input_ids and labels
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    
    for seq_input, seq_labels in zip(input_ids, labels):
        # Pad with -100 for labels (ignored in loss calculation)
        padding_length = max_len - len(seq_input)
        
        padded_input = torch.cat([
            seq_input,
            torch.zeros(padding_length, dtype=seq_input.dtype)
        ])
        
        padded_label = torch.cat([
            seq_labels,
            torch.full((padding_length,), -100, dtype=seq_labels.dtype)
        ])
        
        padded_input_ids.append(padded_input)
        padded_labels.append(padded_label)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels)
    }