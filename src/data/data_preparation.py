from datasets import load_dataset
from typing import Dict
import torch
from transformers import AutoTokenizer

def prepare_dataset(tokenizer, split="train", test_size=0.2, seed=42):
    """
    Prepare dataset with proper train/test split
    
    Args:
        tokenizer: The tokenizer to use
        split: Either "train" or "test"
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
    """
    # Load dataset
    dataset = load_dataset("lamini/lamini_docs")
    
    # Split into train and test
    splits = dataset["train"].train_test_split(
        test_size=test_size, 
        seed=seed
    )
    
    # Select appropriate split
    if split == "train":
        data = splits["train"]
    else:
        data = splits["test"]
    
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
    tokenized_dataset = data.map(
        tokenize_function,
        batched=True,
        remove_columns=data.column_names
    )
    
    return tokenized_dataset 