import sys
sys.path.append("src")  # Add src directory to Python path

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from datasets import load_dataset
import os
from dotenv import load_dotenv
from data.data_preparation import prepare_dataset

# Load environment variables
load_dotenv()

def convert_model():
    print("Loading merged model...")
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create quantization config
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize to 4-bit
        group_size=128,
        desc_act=True
    )

    # Load tokenizer first
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        token=os.getenv('HF_TOKEN')
    )
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoGPTQForCausalLM.from_pretrained(
        'merged_model',
        quantize_config=quantize_config,
        trust_remote_code=True,
        device_map={"": device},  # Explicitly set device
        torch_dtype=torch.float16
    )
    
    # Prepare example data
    print("Preparing example data...")
    dataset = prepare_dataset(
        tokenizer=tokenizer,
        split="test",
        max_length=512
    )

    # Format examples and ensure they're on the same device as model
    examples = [
        {
            'input_ids': torch.tensor(dataset[0]['input_ids'], device=device),
            'attention_mask': torch.tensor(dataset[0]['attention_mask'], device=device)
        }
    ]

    # Add this after dataset loading to inspect the data
    print("\nDataset inspection:")
    print(f"Dataset type: {type(dataset)}")
    print(f"First item type: {type(dataset[0])}")
    print(f"First item keys: {dataset[0].keys()}")
    print(f"Input IDs shape: {len(dataset[0]['input_ids'])}")

    # Add before quantization to inspect examples
    print("\nExamples inspection:")
    print(f"Examples type: {type(examples)}")
    print(f"Number of examples: {len(examples)}")
    print(f"First example type: {type(examples[0])}")
    print(f"First example keys: {examples[0].keys()}")
    print(f"First example input_ids length: {len(examples[0]['input_ids'])}")

    print("Quantizing model...")
    model.quantize(examples)
    
    print("Converting and saving model...")
    model.save_pretrained(
        'quantized_model',
        use_safetensors=True
    )
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained('quantized_model')
    
    print("Conversion complete! Model saved to: quantized_model/")

    # Add debug print of quantize method signature
    print("\nQuantize method info:")
    print(f"Quantize method: {model.quantize.__doc__}")

if __name__ == "__main__":
    convert_model() 