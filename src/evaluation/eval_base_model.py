from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import time
from rouge_score import rouge_scorer
import numpy as np
from data.data_preparation import prepare_dataset

def load_model_and_tokenizer(model_path=None):
    model_name = model_path or "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model and tokenizer from {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    prompt = f"### Question: {question}\n\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Record generation time
    start_time = time.time()
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
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

def evaluate_model(model_path=None, num_samples=10):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load test data only
    dataset = prepare_dataset(tokenizer, split="test")
    
    model_name = model_path or "Base Model (Llama-3.2-3B)"
    print(f"\nEvaluating {model_name} on {num_samples} sample questions:")
    
    # Get total dataset size
    total_samples = len(dataset["train"])
    print(f"Total available samples: {total_samples}")
    
    # Select samples with some spacing to get diverse examples
    indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
    test_samples = dataset["train"].select(indices)
    
    print("----------------------------------------")
    
    # Initialize metrics storage
    all_metrics = []
    total_generation_time = 0
    
    for idx, sample in enumerate(test_samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print(f"\nQuestion {idx + 1}/{num_samples}: {question}")
        
        try:
            # Generate response with timeout
            response, generation_time = generate_response(model, tokenizer, question)
            total_generation_time += generation_time
            
            # Calculate metrics
            metrics = calculate_metrics(response, ground_truth)
            all_metrics.append(metrics)
            
            print("\nModel Response:")
            print(response)
            print("\nGround Truth:")
            print(ground_truth)
            print("\nMetrics:")
            print(f"Generation Time: {generation_time:.2f} seconds")
            print(f"ROUGE-1 F1: {metrics['rouge1_f1']:.3f}")
            print(f"ROUGE-2 F1: {metrics['rouge2_f1']:.3f}")
            print(f"ROUGE-L F1: {metrics['rougeL_f1']:.3f}")
            print(f"Response Length: {metrics['response_length']} words")
            print(f"Ground Truth Length: {metrics['ground_truth_length']} words")
            print(f"Length Ratio: {metrics['length_ratio']:.2f}")
            print("----------------------------------------")
            
        except Exception as e:
            print(f"Error processing question {idx + 1}: {str(e)}")
            continue
    
    if all_metrics:
        # Calculate and display average metrics with standard deviations
        print("\nFinal Metrics Across All Samples:")
        print("----------------------------------------")
        metrics_summary = {}
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            metrics_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        print(f"Total Samples Processed: {len(all_metrics)}")
        print(f"Average Generation Time: {total_generation_time/len(all_metrics):.2f} seconds")
        print(f"ROUGE-1 F1: {metrics_summary['rouge1_f1']['mean']:.3f} (±{metrics_summary['rouge1_f1']['std']:.3f})")
        print(f"ROUGE-2 F1: {metrics_summary['rouge2_f1']['mean']:.3f} (±{metrics_summary['rouge2_f1']['std']:.3f})")
        print(f"ROUGE-L F1: {metrics_summary['rougeL_f1']['mean']:.3f} (±{metrics_summary['rougeL_f1']['std']:.3f})")
        print(f"Average Response Length: {metrics_summary['response_length']['mean']:.1f} (±{metrics_summary['response_length']['std']:.1f}) words")
        print(f"Average Ground Truth Length: {metrics_summary['ground_truth_length']['mean']:.1f} (±{metrics_summary['ground_truth_length']['std']:.1f}) words")
        print(f"Average Length Ratio: {metrics_summary['length_ratio']['mean']:.2f} (±{metrics_summary['length_ratio']['std']:.2f})")

if __name__ == "__main__":
    # Evaluate base model
    print("\n=== Evaluating Base Model ===")
    evaluate_model()
    
    # Evaluate fine-tuned model
    print("\n=== Evaluating Fine-tuned Model ===")
    evaluate_model("./fine_tuned_model_eval_loss_0.38830")