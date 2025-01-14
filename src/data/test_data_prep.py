from transformers import AutoTokenizer
from data_preparation import prepare_dataset
import torch

def test_data_preparation():
    # Load a small model's tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # Prepare a small subset of the dataset
    dataset = prepare_dataset(
        tokenizer=tokenizer,
        max_length=256
    )
    
    # Print sample to verify format
    print("Dataset size:", len(dataset))
    print("\nFirst example format:")
    for key, value in dataset[0].items():
        print(f"{key}: {len(value)}")
    
    # Decode a sample to verify text formatting
    decoded = tokenizer.decode(dataset[0]['input_ids'])
    print("\nSample decoded text:")
    print(decoded)
    
    # Print actual length of first example (non-padding tokens)
    attention_mask = torch.tensor(dataset[0]['attention_mask'])  # Convert to tensor
    actual_length = attention_mask.sum().item()
    print(f"\nActual content length (non-padding): {actual_length}")

if __name__ == "__main__":
    test_data_preparation()