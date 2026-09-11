import os
import sys
import math
import yaml
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Add the project root to the path to import from custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.data_loader import load_and_prepare_data
from data.preprocess import get_tokenizer, tokenize_dataset
from scripts.metrics import build_compute_metrics_fn

def parse_args():
    """
    Parses command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate the Code Summarization model on the test split.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to trained checkpoint directory (e.g., checkpoints/final_model)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base.yaml", 
        help="Path to configuration YAML file"
    )
    return parser.parse_args()

def load_config(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Starting standalone evaluation for checkpoint: {args.checkpoint}")
    
    # Load dataset splits
    _, _, test_ds = load_and_prepare_data(debug=config['data'].get('debug', False))
    
    # Load artifacts from the target checkpoint
    tokenizer = get_tokenizer(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    
    # Tokenize the test dataset
    tokenized_test = tokenize_dataset(
        test_ds, 
        tokenizer, 
        max_input_len=config['data']['max_input_length'],
        max_target_len=config['data']['max_target_length']
    )
    
    # Set up the Trainer for evaluation
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    use_fp16 = config['training'].get('fp16', False) and torch.cuda.is_available()

    # Define evaluation arguments
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=config['training']['batch_size'],
        predict_with_generate=True,
        generation_max_length=config.get('generation', {}).get('max_length', 64),
        generation_num_beams=config.get('generation', {}).get('num_beams', 4),
        fp16=use_fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer)
    )
    
    # Run evaluation
    print("Running evaluation on the test set...")
    metrics = trainer.evaluate()
    
    # Format and print the final results
    perplexity = math.exp(metrics.get("eval_loss", 0))
    
    print("\n" + "=" * 40)
    print("FINAL TEST EVALUATION METRICS")
    print("=" * 40)
    print(f"Cross-Entropy Loss : {metrics.get('eval_loss', 0):.4f}")
    print(f"Perplexity         : {perplexity:.4f}")
    print(f"SacreBLEU          : {metrics.get('eval_bleu', 0):.4f}")
    print(f"ROUGE-1            : {metrics.get('eval_rouge1', 0):.4f}")
    print(f"ROUGE-2            : {metrics.get('eval_rouge2', 0):.4f}")
    print(f"ROUGE-L            : {metrics.get('eval_rougeL', 0):.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()