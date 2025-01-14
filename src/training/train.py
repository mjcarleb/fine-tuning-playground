from training.model_training import LlamaTrainer
from data_preparation import prepare_dataset  # We'll create this next

def main():
    # Initialize trainer
    trainer = LlamaTrainer("config/training_config.yaml")
    
    # Setup model and tokenizer
    trainer.setup_model()
    
    # Load and prepare dataset
    train_dataset = prepare_dataset()  # We'll implement this next
    
    # Train the model
    trainer.train(train_dataset)
    
    # Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    main() 