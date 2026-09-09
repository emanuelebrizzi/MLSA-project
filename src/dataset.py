import torch
from torch.utils.data import Dataset, DataLoader

class CodeSummaryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading code and summary pairs.
    It takes a pandas DataFrame and tokenizers to output PyTorch tensors.
    """
    def __init__(self, df, code_tokenizer, summary_tokenizer, max_len_code=100, max_len_summary=50):
        self.data = df
        self.code_tokenizer = code_tokenizer
        self.summary_tokenizer = summary_tokenizer
        self.max_len_code = max_len_code
        self.max_len_summary = max_len_summary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract row data using pandas iloc
        row = self.data.iloc[index]
        code_text = row['code']
        summary_text = row['summary']
        
        # Tokenize and pad/truncate sequences
        code_tokens = self.code_tokenizer.encode(
            code_text, 
            max_len=self.max_len_code, 
            add_special_tokens=True
        )
        summary_tokens = self.summary_tokenizer.encode(
            summary_text, 
            max_len=self.max_len_summary, 
            add_special_tokens=True
        )
        
        # Convert lists to PyTorch long tensors
        return torch.tensor(code_tokens, dtype=torch.long), torch.tensor(summary_tokens, dtype=torch.long)


def get_dataloaders(train_df, val_df, code_tokenizer, summary_tokenizer, batch_size=32, num_workers=0):
    """
    Helper function to quickly generate PyTorch DataLoaders for training and validation.
    """
    train_dataset = CodeSummaryDataset(train_df, code_tokenizer, summary_tokenizer)
    val_dataset = CodeSummaryDataset(val_df, code_tokenizer, summary_tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader