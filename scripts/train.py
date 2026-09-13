import os
import sys
import math
import yaml
import argparse
import torch
import subprocess
import csv
from datetime import datetime
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


# Add project root directory to sys.path to allow modular imports from data/ and scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.data_loader import load_and_prepare_data
from data.preprocess import get_tokenizer, tokenize_dataset
from scripts.metrics import build_compute_metrics_fn, QualitativeEvaluationCallback

def main():
    """
    Main function to execute the end-to-end training pipeline.
    """
    # Load configuration
    args = parse_args()
    config = load_config(args.config)
    print(f"=== Starting experiment: {config['experiment']['name']} ===")
    print(f"Configuration file in use: {args.config}")
    
    # Set the random seed for reproducibility
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load and preprocess data
    is_debug = config['data'].get('debug', False)
    debug_size = config['data'].get('debug_size', 1000)
    train_ds, val_ds, test_ds = load_and_prepare_data(debug=is_debug, debug_size=debug_size)

    #  Tokenizer initialization and dataset tokenization
    tokenizer = get_tokenizer(config['model']['checkpoint'])

    tokenized_train = tokenize_dataset(
        train_ds,
        tokenizer,
        max_input_len=config['data']['max_input_length'],
        max_target_len=config['data']['max_target_length']
    )
    tokenized_val = tokenize_dataset(
        val_ds,
        tokenizer,
        max_input_len=config['data']['max_input_length'],
        max_target_len=config['data']['max_target_length']
    )
    
    # Model instantiation
    print(f"Loading pre-trained model weights: {config['model']['checkpoint']}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['checkpoint'])
    
    # Dynamic batch-level padding collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Define Training Arguments
    training_kwargs = {
        "output_dir": config['training']['output_dir'],
        "evaluation_strategy": "steps",
        "eval_steps": config['training']['eval_steps'],
        "logging_steps": config['training']['logging_steps'],
        "save_steps": config['training']['save_steps'],
        "learning_rate": float(config['training']['learning_rate']),
        "per_device_train_batch_size": config['training']['batch_size'],
        "per_device_eval_batch_size": config['training']['batch_size'],
        "weight_decay": config['training'].get('weight_decay', 0.01),
        "save_total_limit": config['training'].get('save_total_limit', 3),
        "predict_with_generate": True,
        "fp16": config['training'].get('fp16', torch.cuda.is_available()),
        "seed": seed
    }

    # Reconcile max_steps vs. num_train_epochs
    max_steps = config['training'].get('max_steps', None)
    if max_steps and max_steps > 0:
        training_kwargs["max_steps"] = max_steps
        print(f"Training will be limited to max_steps: {max_steps}")
    else:
        training_kwargs["num_train_epochs"] = config['training'].get('num_train_epochs', 3)
        print(f"Training configured for {training_kwargs['num_train_epochs']} full epochs")

    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer)
    )
    
    # Add callback to see generated summaries during training
    num_examples = config.get('generation', {}).get('num_qualitative_examples', 5)
    trainer.add_callback(QualitativeEvaluationCallback(tokenizer, tokenized_val, num_examples=num_examples))


    # Model execution (supports resuming from checkpoints)
    print("Starting model optimization phase...")
    resume_checkpoint = args.resume
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Persist final model artifacts (weights and tokenizer vocabulary)
    final_output_path = os.path.join(config['training']['output_dir'], "final_model")
    print(f"Saving final model weights and tokenizer to: {final_output_path}...")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    # Final validation evaluation
    print("Running final evaluation pass on validation split...")
    eval_results = trainer.evaluate()

    if "eval_loss" in eval_results:
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Final Validation Perplexity : {perplexity:.4f}")
    print(f"Final Validation SacreBLEU  : {eval_results.get('eval_bleu', 0.0):.4f}")

    print("Logging experiment results to CSV...")
    # Save the evaluation results to a CSV file for experiment tracking
    log_path = config.get('experiment', {}).get('log_path', 'logs/experiments_log.csv')
    log_experiment_to_csv(args.config, eval_results, notes="", output_file=log_path)

def load_config(config_path="configs/debug.yaml"):
    """
    Loads the YAML configuration file, resolving paths relative to the project root.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_git_commit_hash():
    """
    Retrieves the hash of the last Git commit to track which code was running.
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            shell=True,
            cwd=PROJECT_ROOT,
            stderr=subprocess.STDOUT
        )
        return commit.decode('ascii').strip()
    except Exception as e:
        print(f"ERROR GIT: Unable to retrieve Git commit hash: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"OUTPUT ERROR GIT: {e.output.decode('utf-8', errors='ignore')}\n")
        return "No-Git"

def log_experiment_to_csv(config_path, eval_results, notes="", output_file="logs/experiments_log.csv"):    
    """
    Saves the final results to a CSV file acting as a history log.
    """
    git_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Git Hash", "Config", "Notes", "Eval Loss", "SacreBLEU", "ROUGE-L"])
            
        writer.writerow([
            timestamp,
            git_hash,
            config_path,
            notes,
            round(eval_results.get("eval_loss", 0.0), 4),
            round(eval_results.get("eval_bleu", 0.0), 4),
            round(eval_results.get("eval_rougeL", 0.0), 4)
        ])

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train the code summarization model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="Relative or absolute path to the YAML configuration file (default: configs/debug.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a checkpoint directory to resume training from (e.g., checkpoints/checkpoint-500)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
