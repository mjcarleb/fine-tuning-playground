from datasets import load_dataset
from typing import Dict
import torch
from transformers import AutoTokenizer

def prepare_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    dataset_path: str = "lamini/lamini_docs"
) -> torch.utils.data.Dataset:
    """
    Prepare Lamini docs dataset for training.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        dataset_path: Path to the Lamini dataset
    """
    # Load Lamini dataset
    dataset = load_dataset(dataset_path)
    
    def tokenize_function(examples):
        """Tokenize examples using Q&A format."""
        # Combine question and answer in a chat format
        prompts = [
            f"### Question: {question}\n\n### Answer: {answer}"
            for question, answer in zip(examples["question"], examples["answer"])
        ]
        
        # Tokenize the text
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Set labels for training
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset["train"] 