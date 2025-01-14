from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import time
from rouge_score import rouge_scorer
import numpy as np

def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model and tokenizer from {model_name}...")
    
    # Force CPU usage and enable memory optimizations
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with CPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # Force CPU
        low_cpu_mem_usage=True,
        offload_folder="offload"
    ).to(device)  # Ensure model is on CPU
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=64):  # Further reduced max_length
    prompt = f"### Question: {question}\n\n### Answer:"
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_length
    ).to(model.device)  # Ensure inputs are on CPU
    
    start_time = time.time()
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # Reduced beam search
            early_stopping=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
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

def evaluate_base_model(num_samples=2):  # Reduced to 2 samples
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