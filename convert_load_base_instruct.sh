#!/bin/bash
# Script to download base Instruct model, convert to GGUF and create Ollama model

echo "Step 1: Downloading Llama 3.2B Instruct model from HuggingFace..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
load_dotenv()

print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    torch_dtype='float16',
    device_map='auto',
    token=os.getenv('HF_TOKEN')
)
model.save_pretrained('./base_instruct_model')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    token=os.getenv('HF_TOKEN')
)
tokenizer.save_pretrained('./base_instruct_model')
"

if [ $? -eq 0 ]; then
    echo "Base model download successful!"
    
    echo "Step 2: Converting to GGUF format..."
    cd llama.cpp
    python3 convert_hf_to_gguf.py ../base_instruct_model/ \
        --outfile base_instruct.gguf \
        --outtype f16 \
        --model-name LlamaForCausalLM \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "GGUF conversion successful!"
        
        echo "Step 3: Creating Ollama model..."
        cd ..
        
        ollama rm llama3.2-3b-instruct 2>/dev/null  # Remove if exists
        ollama create llama3.2-3b-instruct -f base_instruct_Modelfile
        
        if [ $? -eq 0 ]; then
            echo "Success! Base Instruct model is ready to use"
        else
            echo "Error: Failed to create Ollama model"
            exit 1
        fi
    else
        echo "Error: GGUF conversion failed"
        exit 1
    fi
else
    echo "Error: Model download failed"
    exit 1
fi 