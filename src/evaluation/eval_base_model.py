from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import time
from rouge_score import rouge_scorer
import numpy as np

def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model and tokenizer from {model_name}...")
    
    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="mps" if torch.backends.mps.is_available() else "auto",
        low_cpu_mem_usage=True,     # Reduce memory usage
        offload_folder="offload"    # Offload to disk if needed
    )
    
    # Enable memory efficient attention if available
    if hasattr(model.config, "use_memory_efficient_attention"):
        model.config.use_memory_efficient_attention = True
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=256):  # Reduced max length
    prompt = f"### Question: {question}\n\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Record generation time
    start_time = time.time()
    
    with torch.inference_mode():  # More efficient than no_grad
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,           # Disable beam search for speed
            early_stopping=True,   # Stop when EOS token is generated
            top_k=50,             # Limit vocabulary choices
            top_p=0.95,           # Nucleus sampling
            repetition_penalty=1.2 # Reduce repetition
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Answer:")[-1].strip()
    
    return response, generation_time

def calculate_metrics(response, ground_truth):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(ground_truth, response)
    
    # Calculate response length
    response_length = len(response.split())
    ground_truth_length = len(ground_truth.split())
    
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
        'response_length': response_length,
        'ground_truth_length': ground_truth_length,
        'length_ratio': response_length / ground_truth_length if ground_truth_length > 0 else 0
    }

def evaluate_base_model(num_samples=3):  # Reduced number of samples for testing
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load test questions
    dataset = load_dataset("lamini/lamini_docs")
    test_samples = dataset["train"].select(range(num_samples))
    
    # Initialize metrics storage
    all_metrics = []
    total_generation_time = 0
    
    print("\nEvaluating base model on sample questions:")
    print("----------------------------------------")
    
    for idx, sample in enumerate(test_samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print(f"\nQuestion {idx + 1}: {question}")
        
        try:
            # Generate response with timeout
            response, generation_time = generate_response(model, tokenizer, question)
            total_generation_time += generation_time
            
            # Calculate metrics
            metrics = calculate_metrics(response, ground_truth)
            all_metrics.append(metrics)
            
            print("\nModel Response:")
            print(response)
            print("\nMetrics:")
            print(f"Generation Time: {generation_time:.2f} seconds")
            print(f"ROUGE-1 F1: {metrics['rouge1_f1']:.3f}")
            print(f"Response Length: {metrics['response_length']} words")
            
        except Exception as e:
            print(f"Error processing question {idx + 1}: {str(e)}")
            continue
    
    if all_metrics:
        # Calculate and display average metrics
        print("\nAverage Metrics:")
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        print(f"Average Generation Time: {total_generation_time/len(test_samples):.2f} seconds")
        print(f"Average ROUGE-1 F1: {avg_metrics['rouge1_f1']:.3f}")
        print(f"Average Response Length: {avg_metrics['response_length']:.1f} words")

if __name__ == "__main__":
    evaluate_base_model()