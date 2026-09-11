import os
import sys
import math
import yaml
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Add the project root to the path to import from the 'data' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.loader import load_and_prepare_data
from scripts.preprocess import get_tokenizer, tokenize_dataset
from scripts.metrics import build_compute_metrics_fn, QualitativeEvaluationCallback

def load_config(config_path="configs/base.yaml"):
    """
    Loads the YAML configuration file containing all hyperparameters.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    """
    Main function to execute the end-to-end training pipeline.
    """
    # Load configuration
    config = load_config()
    print(f"Starting experiment: {config['experiment']['name']}")
    
    # 2. Set the random seed for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    
    # Load and preprocess data
    # Using the debug mode by default here to follow the "Start tiny" advice
    train_ds, val_ds, test_ds = load_and_prepare_data(debug=True, debug_size=1000)
    
    tokenizer = get_tokenizer(config['model']['checkpoint'])
    
    tokenized_train = tokenize_dataset(
        train_ds, tokenizer, 
        max_input_len=config['data']['max_input_length'],
        max_target_len=config['data']['max_target_length']
    )
    tokenized_val = tokenize_dataset(
        val_ds, tokenizer, 
        max_input_len=config['data']['max_input_length'],
        max_target_len=config['data']['max_target_length']
    )
    
    # Load the Seq2Seq Model
    print(f"Loading model: {config['model']['checkpoint']}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['checkpoint'])
    
    # A DataCollator automatically pads inputs to the maximum length of the current batch
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['training']['output_dir'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        weight_decay=config['training']['weight_decay'],
        save_total_limit=config['training']['save_total_limit'],
        num_train_epochs=config['training']['num_train_epochs'],
        predict_with_generate=True,     # Crucial for computing BLEU/ROUGE during evaluation
        fp16=config['training']['fp16'], # Use Mixed Precision if a compatible GPU is available
        seed=config['experiment']['seed']
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer)
    )
    
    # Add custom callback to see generated summaries during training
    trainer.add_callback(QualitativeEvaluationCallback(tokenizer, tokenized_val, num_examples=3))
    
    # Start Training
    print("Starting the training process...")
    train_result = trainer.train()
    
    # Save the final model and tokenizer
    print("Saving the final model...")
    trainer.save_model(os.path.join(config['training']['output_dir'], "final_model"))
    
    # Final Evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Compute Perplexity from the Cross-Entropy Loss
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Final Validation Perplexity: {perplexity:.4f}")
    print(f"Final Validation BLEU: {eval_results.get('eval_bleu', 0):.4f}")

def load_config(config_path="configs/base.yaml"):
    """
    Loads the YAML configuration file containing all hyperparameters.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.join(project_root, config_path)
    
    with open(full_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    main()