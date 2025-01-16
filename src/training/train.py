import sys
sys.path.append("src")  # Add src directory to Python path

from data.data_preparation import prepare_dataset  # Updated import path
from training.model_training import LlamaTrainer
import torch

def main():
    print("Starting training process...")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize trainer
    print("Loading config and initializing trainer...")
    trainer = LlamaTrainer("config/training_config.yaml")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    trainer.setup_model()
    
    # Load training data only
    print("Preparing dataset...")
    train_dataset = prepare_dataset(trainer.tokenizer, split="train")
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Train the model
    print("Starting training...")
    eval_dataset = prepare_dataset(trainer.tokenizer, split="test")
    trainer.train(train_dataset, eval_dataset=eval_dataset)
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    trainer.save_model("./fine_tuned_model")
    
    print("Training complete!")

if __name__ == "__main__":
    main()