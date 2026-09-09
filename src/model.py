import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the 
    tokens in the sequence. Necessary for Transformer architectures.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CodeSummarizationTransformer(nn.Module):
    """
    A Sequence-to-Sequence model using a Transformer architecture
    for translating code into natural language summaries.
    """
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        d_model=256, 
        nhead=8, 
        num_encoder_layers=4, 
        num_decoder_layers=4, 
        dim_feedforward=512, 
        dropout=0.1
    ):
        super(CodeSummarizationTransformer, self).__init__()
        self.d_model = d_model
        
        # Embedding layers for source (code) and target (summary)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Standard PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Treats inputs as (batch, seq, feature)
        )
        
        # Final linear layer to map to target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a mask for the target sequence to prevent the decoder
        from looking ahead into future tokens during training.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, matrix, pad_token):
        """
        Creates a mask for padding tokens, so the attention mechanism ignores them.
        Returns a boolean mask where True indicates a padding position.
        """
        return (matrix == pad_token)

    def forward(self, src, tgt, src_pad_token=0, tgt_pad_token=0):
        """
        Forward pass of the model.
        src shape: (batch_size, src_seq_len)
        tgt shape: (batch_size, tgt_seq_len)
        """
        # Create masks
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        src_key_padding_mask = self.create_padding_mask(src, src_pad_token).to(src.device)
        tgt_key_padding_mask = self.create_padding_mask(tgt, tgt_pad_token).to(src.device)
        
        # Apply embeddings and positional encoding (multiply by sqrt(d_model) as per original paper)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Pass through Transformer
        outs = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Decoder attends to encoder outputs
        )
        
        # Project to vocabulary size
        return self.fc_out(outs)