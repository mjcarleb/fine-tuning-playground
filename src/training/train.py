from data.data_preparation import prepare_dataset  # Updated import path
from training.model_training import LlamaTrainer

def main():
    # Initialize trainer
    trainer = LlamaTrainer("config/training_config.yaml")
    
    # Setup model and tokenizer
    trainer.setup_model()
    
    # Load training data only
    train_dataset = prepare_dataset(trainer.tokenizer, split="train")
    
    # Train the model
    trainer.train(train_dataset)
    
    # Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    main()