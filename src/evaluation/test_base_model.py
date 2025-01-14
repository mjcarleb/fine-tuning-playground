from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model and tokenizer from {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto"  # Automatically choose best device (CPU/GPU)
    )
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    # Format the prompt
    prompt = f"### Question: {question}\n\n### Answer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Answer:")[-1].strip()

def evaluate_base_model():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load some test questions from Lamini dataset
    dataset = load_dataset("lamini/lamini_docs")
    test_samples = dataset["train"].select(range(5))  # Test with first 5 questions
    
    print("\nEvaluating base model on sample questions:")
    print("----------------------------------------")
    
    for idx, sample in enumerate(test_samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print(f"\nQuestion {idx + 1}: {question}")
        print("\nModel Response:")
        response = generate_response(model, tokenizer, question)
        print(response)
        print("\nGround Truth:")
        print(ground_truth)
        print("\n----------------------------------------")

if __name__ == "__main__":
    evaluate_base_model()