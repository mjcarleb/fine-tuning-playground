 from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import yaml

@dataclass
class TrainingConfig:
    model_name: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    max_length: int
    gradient_accumulation_steps: int

@dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    bias: str
    task_type: str

class LlamaTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup_model(self):
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def train(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            num_train_epochs=self.config['training']['num_epochs'],
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        
    def save_model(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)