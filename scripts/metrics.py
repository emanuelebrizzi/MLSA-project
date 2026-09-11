from xml.parsers.expat import model

import torch
import numpy as np
import torch
import evaluate
from transformers import TrainerCallback

# Load the evaluation metrics
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def build_compute_metrics_fn(tokenizer):
    """
    Returns a compute_metrics function configured with the given tokenizer.
    This function will be called by the Seq2SeqTrainer during evaluation.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Replace -100s in the labels as we can't decode them.
        # -100 is the default PyTorch value to ignore tokens in the loss calculation.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode generated predictions and labels back into strings
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing: strip whitespaces
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        # Compute BLEU score
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute ROUGE score
        # ROUGE expects a list of strings for references, not a list of lists
        flat_labels = [label[0] for label in decoded_labels]
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=flat_labels)
        
        # Extract the specific metrics we care about
        metrics = {
            "bleu": bleu_result["score"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"]
        }
        
        # Round the metrics to 4 decimal places for cleaner logging
        return {k: round(v, 4) for k, v in metrics.items()}
        
    return compute_metrics

class QualitativeEvaluationCallback(TrainerCallback):
    """
    A custom callback to generate and print qualitative examples at the end of each evaluation.
    This satisfies the "Show example summaries and improvements over time" requirement.
    """
    def __init__(self, tokenizer, eval_dataset, num_examples=3):
        self.tokenizer = tokenizer
        # Select a small, fixed subset to observe how the same examples improve over time
        self.eval_dataset = eval_dataset.select(range(num_examples))
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        print(f"\n--- Qualitative Evaluation at Step {state.global_step} ---")
        
        # Set the model to evaluation mode
        model.eval()
        
        for i in range(len(self.eval_dataset)):
            example = self.eval_dataset[i]
            
            # Inside QualitativeEvaluationCallback.on_evaluate:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)
            
            # Generate the summary using the model
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated text and the reference text
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # We must ignore the -100 padding tokens to decode the label properly
            label_ids = [id for id in example["labels"] if id != -100]
            expected_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
            
            print(f"\nExample {i+1}:")
            print(f"EXPECTED : {expected_text}")
            print(f"GENERATED: {generated_text}")
            
        print("-" * 50)