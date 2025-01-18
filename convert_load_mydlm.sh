#!/bin/bash
# Script to convert merged model to GGUF and create Ollama model

echo "Step 1: Converting merged model to GGUF format..."
cd llama.cpp
python3 convert_hf_to_gguf.py ../merged_model/ \
    --outfile model.gguf \
    --outtype f16 \
    --model-name LlamaForCausalLM \
    --verbose

if [ $? -eq 0 ]; then
    echo "GGUF conversion successful!"
    
    echo "Step 2: Creating Ollama model..."
    cd ..
    ollama rm mydlm 2>/dev/null  # Remove existing model if it exists
    ollama create mydlm -f mydlm_Modelfile
    
    if [ $? -eq 0 ]; then
        echo "Success! Model mydlm is ready to use"
    else
        echo "Error: Failed to create Ollama model"
        exit 1
    fi
else
    echo "Error: GGUF conversion failed"
    exit 1
fi 