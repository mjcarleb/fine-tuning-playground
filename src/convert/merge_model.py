# Save as merge_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import os

# Load token from .env file
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

print("Loading and merging LoRA weights...")
model = PeftModel.from_pretrained(
    base_model,
    "./fine_tuned_model_eval_loss_0.3830"
)
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("./merged_model", safe_serialization=True)

# Save tokenizer files
print("Copying tokenizer files...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv('HF_TOKEN')
)
tokenizer.save_pretrained("merged_model")

print("Merge complete! Model saved to: merged_model/")