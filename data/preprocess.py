from transformers import AutoTokenizer

def get_tokenizer(checkpoint="Salesforce/codet5-small"):
    """
    Loads the tokenizer from the Hugging Face hub.
    """
    print(f"Loading tokenizer for {checkpoint}...")
    return AutoTokenizer.from_pretrained(checkpoint)

def tokenize_dataset(dataset, tokenizer, max_input_len=256, max_target_len=64):
    """
    Tokenizes the code (input) and the summary (target) for the Seq2Seq model.
    """
    def preprocess_function(examples):
        # Tokenize the input sequence (the Python code)
        model_inputs = tokenizer(
            examples["code"],
            max_length=max_input_len,
            padding="max_length",
            truncation=True
        )

        # Tokenize the target sequence (the natural language summary)
        # Using text_target is the standard approach for Seq2Seq labels in HF
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True
        )

        # Assign the tokenized labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing the dataset...")
    # Apply the tokenization to all splits (train/val/test) using batched mapping
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["code", "summary"]  # Remove raw strings to save memory
    )
    
    return tokenized_dataset

if __name__ == "__main__":
    # Example usage for debugging purposes
    from data.data_loader import load_and_prepare_data
    
    # Load a tiny subset of data
    train_ds, val_ds, test_ds = load_and_prepare_data(debug=True, debug_size=100)
    
    # Initialize tokenizer and process the training set
    tokenizer = get_tokenizer()
    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    
    print("Example of tokenized input keys:", tokenized_train[0].keys())