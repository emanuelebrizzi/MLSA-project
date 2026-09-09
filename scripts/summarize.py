import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenizer import SimpleTokenizer
from src.model import CodeSummarizationTransformer
from src.preprocess import clean_code

def greedy_decode(model, src, max_len, start_symbol_idx, end_symbol_idx, device):
    """
    Generates a sequence token by token using greedy decoding.
    """
    # Create source mask
    src = src.to(device)
    
    # Encode source
    memory = model.transformer.encoder(
        model.pos_encoder(model.src_embedding(src) * (model.d_model ** 0.5))
    )
    
    # Initialize target sequence with the start symbol <SOS>
    ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        # Create target mask to prevent looking ahead
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        
        # Decode
        out = model.transformer.decoder(
            model.pos_encoder(model.tgt_embedding(ys) * (model.d_model ** 0.5)), 
            memory, 
            tgt_mask=tgt_mask
        )
        
        # Get the next word (projection)
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Append to target sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # Stop if <EOS> is generated
        if next_word == end_symbol_idx:
            break
            
    return ys

def summarize_code(code_snippet, model, code_tok, sum_tok, device, max_len=50):
    """
    Full pipeline to clean, tokenize, generate, and decode a summary.
    """
    model.eval()
    
    # Preprocess and tokenize input
    clean_snippet = clean_code(code_snippet)
    src_tokens = code_tok.encode(clean_snippet, add_special_tokens=True)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long)
    
    # Generate token IDs
    with torch.no_grad():
        tgt_tokens = greedy_decode(
            model, src_tensor, max_len, 
            start_symbol_idx=sum_tok.SOS_IDX, 
            end_symbol_idx=sum_tok.EOS_IDX, 
            device=device
        )
        
    # Decode back to text (skipping special tokens)
    summary = sum_tok.decode(tgt_tokens[0].cpu().numpy(), skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    # Esempio di utilizzo (supponendo di aver salvato i tokenizer e il modello)
    print("Script pronto per l'inferenza. Per utilizzarlo, dovrai caricare il modello")
    print("addestrato da checkpoints/transformer_model.pth e i vocabolari costruiti in fase di training.")