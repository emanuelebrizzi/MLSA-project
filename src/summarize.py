import os
import sys
import torch
import pickle

# Add the root directory to the python path to allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenizer import SimpleTokenizer
from src.model import CodeSummarizationTransformer
from src.preprocess import clean_code

def greedy_decode(model, src, max_len, start_symbol_idx, end_symbol_idx, device):
    """
    Generates a sequence token by token using greedy decoding.
    """
    src = src.to(device)
    
    memory = model.transformer.encoder(
        model.pos_encoder(model.src_embedding(src) * (model.d_model ** 0.5))
    )
    
    ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        
        out = model.transformer.decoder(
            model.pos_encoder(model.tgt_embedding(ys) * (model.d_model ** 0.5)), 
            memory, 
            tgt_mask=tgt_mask
        )
        
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol_idx:
            break
            
    return ys

def summarize_code(code_snippet, model, code_tok, sum_tok, device, max_len=50):
    """
    Full pipeline to clean, tokenize, generate, and decode a summary.
    """
    model.eval()
    
    clean_snippet = clean_code(code_snippet)
    src_tokens = code_tok.encode(clean_snippet, add_special_tokens=True)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long)
    
    with torch.no_grad():
        tgt_tokens = greedy_decode(
            model, src_tensor, max_len, 
            start_symbol_idx=sum_tok.SOS_IDX, 
            end_symbol_idx=sum_tok.EOS_IDX, 
            device=device
        )
        
    summary = sum_tok.decode(tgt_tokens[0].cpu().numpy(), skip_special_tokens=True)
    return summary

def run_inference():
    """
    Loads the trained model and tokenizers, then runs a test inference.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {DEVICE}...")

    try:
        with open('models/tokenizers.pkl', 'rb') as f:
            tokenizers = pickle.load(f)
            code_tokenizer = tokenizers['code_tokenizer']
            summary_tokenizer = tokenizers['summary_tokenizer']
    except FileNotFoundError:
        print("Error: tokenizers.pkl not found. Make sure you saved them in train.py!")
        return

    model = CodeSummarizationTransformer(
        src_vocab_size=len(code_tokenizer),
        tgt_vocab_size=len(summary_tokenizer),
        d_model=128,
        nhead=4, 
        num_encoder_layers=2, 
        num_decoder_layers=2
    ).to(DEVICE)

    model.load_state_dict(torch.load('models/transformer_model.pkl', map_location=DEVICE))
    model.eval()

    test_code = "def multiply(x, y):\n    return x * y"
    print(f"\n[Input Code]:\n{test_code}")
    
    predicted_summary = summarize_code(test_code, model, code_tokenizer, summary_tokenizer, DEVICE)
    print(f"\n[Predicted Summary]:\n{predicted_summary}\n")

if __name__ == "__main__":
    run_inference()