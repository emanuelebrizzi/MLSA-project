import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import sys

# Add the root directory to the python path to allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_dataset import load_data
from src.preprocess import preprocess_dataframe
from src.tokenizer import SimpleTokenizer
from src.dataset import get_dataloaders
from src.model import CodeSummarizationTransformer

def train():
    # 1. Configuration (Hyperparameters)
    EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. Load and Preprocess Data
    df_raw = load_data() # Loads dummy data by default
    df_clean = preprocess_dataframe(df_raw)
    
    # Split using scikit-learn
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # 3. Tokenization (Build vocabularies on training data only to avoid data leakage)
    code_tokenizer = SimpleTokenizer(max_vocab_size=5000)
    summary_tokenizer = SimpleTokenizer(max_vocab_size=3000)
    
    code_tokenizer.build_vocab(train_df['code'].tolist())
    summary_tokenizer.build_vocab(train_df['summary'].tolist())

    # 4. Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, 
        code_tokenizer, summary_tokenizer, 
        batch_size=BATCH_SIZE
    )

    # 5. Initialize Model
    model = CodeSummarizationTransformer(
        src_vocab_size=len(code_tokenizer),
        tgt_vocab_size=len(summary_tokenizer),
        d_model=128,      # Reduced for dummy data testing
        nhead=4, 
        num_encoder_layers=2, 
        num_decoder_layers=2
    ).to(DEVICE)

    # 6. Loss and Optimizer
    # Ignore padding index when calculating loss (standard practice)
    PAD_IDX = summary_tokenizer.PAD_IDX
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # For Seq2Seq, we feed all but the last token into the decoder
            # and expect it to predict all but the first token (shifted by 1)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input, src_pad_token=PAD_IDX, tgt_pad_token=PAD_IDX)
            
            # Calculate loss (CrossEntropy expects inputs of shape [N, C] and targets of shape [N])
            # Reshape predictions to (batch_size * seq_len, vocab_size) and targets to (batch_size * seq_len)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            
            # Optional: Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        
    print("Training completed.")
    
    # Save the model (make sure the checkpoints directory exists)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/transformer_model.pth')
    print("Model saved to checkpoints/transformer_model.pth")

if __name__ == "__main__":
    train()