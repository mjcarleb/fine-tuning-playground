from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_access():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        print(f"Attempting to load tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully!")
        
        print("\nAttempting to verify model access (without downloading full model)...")
        model_info = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=True,  # Verify token access
            device_map=None,
            torch_dtype="auto",
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