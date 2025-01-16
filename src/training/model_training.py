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
from peft import LoraConfig as PeftLoraConfig, get_peft_model
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
    base_model_name_or_path: str
    is_prompt_learning: bool = False
    peft_type: str = "LORA"
    layer_replication: bool = False
    rank_pattern: dict = None

    def __post_init__(self):
        if self.rank_pattern is None:
            self.rank_pattern = {}

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
        print("Loading model configuration...")
        model_config = {}
        
        if self.config['model'].get('quantization', {}).get('enabled', False):
            print("Setting up quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config['model']['quantization']['load_in_4bit'],
                bnb_4bit_compute_dtype=getattr(torch, self.config['model']['quantization']['compute_dtype']),
                bnb_4bit_quant_type=self.config['model']['quantization']['quant_type'],
                bnb_4bit_use_double_quant=self.config['model']['quantization']['use_double_quant'],
            )
            model_config['quantization_config'] = bnb_config
        
        print(f"Loading model: {self.config['model']['name']}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            device_map="auto",
            trust_remote_code=True,
            **model_config
        ).train()
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        print("Configuring LoRA...")
        lora_config = PeftLoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        print("Applying LoRA to model...")
        self.model = get_peft_model(self.model, lora_config)
        print("Model setup complete!")
                
    def train(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=float(self.config['training']['learning_rate']),
            per_device_train_batch_size=int(self.config['training']['batch_size']),
            gradient_accumulation_steps=int(self.config['training']['gradient_accumulation_steps']),
            num_train_epochs=int(self.config['training']['num_epochs']),
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            early_stopping_patience=int(self.config['training'].get('early_stopping_patience', 3)),
            early_stopping_threshold=float(self.config['training'].get('early_stopping_threshold', 0.001))
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