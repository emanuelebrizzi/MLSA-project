from datasets import load_dataset

def clean_text(example):
    """
    Normalize the text by removing extra spaces and formatting the summary.
    """
    # Keep the code structure but remove trailing whitespaces
    code = example['func_code_string'].strip()
    
    # Lowercase the summary, as suggested in the "Data Handling Tips"
    summary = example['func_documentation_string'].strip().lower()
    
    return {"code": code, "summary": summary}

def filter_empty(example):
    """
    Discard examples where the code or summary are empty or too short.
    """
    return len(example["code"]) > 10 and len(example["summary"]) > 5

def load_and_prepare_data(debug=False, debug_size=1000):
    """
    Load CodeSearchNet, clean it, and return train/val/test datasets.
    """
    print("Loading CodeSearchNet (Python)...")
    dataset = load_dataset("code-search-net/code_search_net", "python")
    
    # 1. Rename and clean columns
    print("Cleaning data...")
    dataset = dataset.map(clean_text, remove_columns=dataset['train'].column_names)
    
    # 2. Filter empty or invalid rows
    dataset = dataset.filter(filter_empty)
    
    # 3. Create a "Tiny Dataset" to test the pipeline on CPU
    if debug:
        print(f"DEBUG mode active: extracting only {debug_size} examples.")
        # Select a deterministic subset for reproducibility
        train_ds = dataset['train'].select(range(debug_size))
        val_ds = dataset['validation'].select(range(debug_size // 10))
        test_ds = dataset['test'].select(range(debug_size // 10))
    else:
        train_ds = dataset['train']
        val_ds = dataset['validation']
        test_ds = dataset['test']
        
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    # Quick script test
    train, val, test = load_and_prepare_data(debug=True)
    print(f"\nTrain Size: {len(train)}")
    print("Data Example:")
    print("--- CODE ---")
    print(train[0]['code'][:200], "...")
    print("--- SUMMARY ---")
    print(train[0]['summary'])