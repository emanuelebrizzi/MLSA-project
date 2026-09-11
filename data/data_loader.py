from datasets import load_dataset

def clean_batch(batch):
    """
    Cleans and normalizes code and summary fields in batches.
    Strips whitespaces and converts natural language docstrings to lowercase.
    """
    cleaned_codes = [code.strip() if code else "" for code in batch["func_code_string"]]
    cleaned_summaries = [
        summary.strip().lower() if summary else "" for summary in batch["func_documentation_string"]
    ]
    return {"code": cleaned_codes, "summary": cleaned_summaries}

def filter_empty(example):
    """
    Discard examples where the code or summary are empty or too short.
    """
    return len(example["code"]) > 10 and len(example["summary"]) > 5

def load_and_prepare_data(debug=False, debug_size=1000):
    """
    Downloads CodeSearchNet (Python), normalizes text, prunes invalid samples,
    and returns train, validation, and test splits.
    """
    print("Loading CodeSearchNet dataset (language: Python)...")
    dataset = load_dataset("code-search-net/code_search_net", "python")

    print("Cleaning and normalizing text fields across splits...")
    # Retrieve original column names from the train split to strip unused metadata
    columns_to_remove = dataset["train"].column_names
    
    # Process in batches for significantly faster execution
    dataset = dataset.map(
        clean_batch,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Normalizing text"
    )

    print("Filtering invalid or overly short sequences...")
    dataset = dataset.filter(filter_empty, desc="Filtering samples")

    if debug:
        print(f"Debug mode enabled: slicing tiny subsets (debug_size={debug_size})...")
        train_ds = dataset["train"].select(range(min(debug_size, len(dataset["train"]))))
        val_ds = dataset["validation"].select(range(min(debug_size // 10, len(dataset["validation"]))))
        test_ds = dataset["test"].select(range(min(debug_size // 10, len(dataset["test"]))))
    else:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
        test_ds = dataset["test"]

    return train_ds, val_ds, test_ds

    
if __name__ == "__main__":
    # # Quick script test
    print("Executing standalone smoke test for data_loader...")
    train, val, test = load_and_prepare_data(debug=True, debug_size=50)
    
    print(f"\nSubset Sizes -> Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    print("\n--- SAMPLE CODE ---")
    print(train[0]["code"][:250], "\n...")
    print("--- SAMPLE SUMMARY ---")
    print(train[0]["summary"])