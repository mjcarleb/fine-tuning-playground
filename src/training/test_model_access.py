from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

def test_model_access():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Load token from .env file
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    try:
        print(f"Checking CUDA availability: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"\nAttempting to load tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        print("✓ Tokenizer loaded successfully!")
        
        print("\nAttempting to verify model access (without downloading full model)...")
        model_info = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto",  # This will handle GPU allocation
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            trust_remote_code=True
        )
        print("✓ Model access verified!")
        return True
        
    except Exception as e:
        print("❌ Error occurred:")
        print(e)
        return False

if __name__ == "__main__":
    test_model_access()