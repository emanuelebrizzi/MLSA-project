import torch
import numpy as np
import evaluate
from transformers import TrainerCallback

# Load evaluation metrics from Hugging Face evaluate library
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def build_compute_metrics_fn(tokenizer):
    """
    Returns a compute_metrics function configured with the given tokenizer.
    Used by Seq2SeqTrainer during evaluation loops.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Replace -100 masking tokens with pad token id so tokenizer can decode
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None:
            preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if vocab_size is not None:
            labels = np.where((labels >= 0) & (labels < vocab_size), labels, tokenizer.pad_token_id)

          
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        flat_labels = [label[0] for label in decoded_labels]
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=flat_labels)
        
        metrics = {
            "bleu": bleu_result["score"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"]
        }
        
        return {k: round(v, 4) for k, v in metrics.items()}
        
    return compute_metrics


class QualitativeEvaluationCallback(TrainerCallback):
    """
    Custom callback to sample and log qualitative summaries during training checks.
    """
    def __init__(self, tokenizer, eval_dataset, num_examples=5):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset.select(range(num_examples))
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        print(f"\n{'='*25} Step {state.global_step:4d} Qualitative Check {'='*25}")
        model.eval()
        
        for i in range(len(self.eval_dataset)):
            example = self.eval_dataset[i]
            
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode and collapse repeated whitespaces/newlines into a single line
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            generated_text = " ".join(generated_text.replace('"""', '').split())
            
            label_ids = [idx for idx in example["labels"] if idx != -100]
            expected_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
            expected_text = " ".join(expected_text.replace('"""', '').split())
            
            print(f"[{i+1}] TARGET : {expected_text}")
            print(f"    PREDICT: {generated_text}\n")
            
        print("=" * 72)