import sacrebleu
from rouge_score import rouge_scorer

def calculate_bleu(predictions, references):
    """
    Calculates the BLEU score for a list of predictions against references.
    sacrebleu expects references as a list of lists (for multiple references per prediction).
    """
    # Wrap references in a list to match sacrebleu's expected format
    refs = [[ref] for ref in references]
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return bleu.score

def calculate_rouge(predictions, references):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_f, rouge2_f, rougel_f = 0.0, 0.0, 0.0
    n = len(predictions)
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_f += scores['rouge1'].fmeasure
        rouge2_f += scores['rouge2'].fmeasure
        rougel_f += scores['rougeL'].fmeasure
        
    # Return average scores
    return {
        'rouge1': (rouge1_f / n) * 100,
        'rouge2': (rouge2_f / n) * 100,
        'rougeL': (rougel_f / n) * 100
    }