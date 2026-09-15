import re
import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

# Ensure NLTK tokenizers are available locally
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def remove_docstring_from_code(code_str: str) -> str:
    """
    Strips top-level docstrings enclosed in triple quotes from Python code.
    This prevents target-leakage into the encoder inputs.
    """
    if not code_str:
        return ""
    # Matches triple quotes right after the function definition header
    pattern = r'^\s*(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:\s*\n)(\s*("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))'
    cleaned = re.sub(pattern, r'\1', code_str, count=1, flags=re.MULTILINE)
    return cleaned.strip()


def clean_batch(batch):
    """
    Normalizes code and docstrings in batches:
    1. Removes docstrings from code inputs to avoid direct target leakage.
    2. Extracts the first semantic sentence from docstrings using NLTK.
    """
    cleaned_codes = [remove_docstring_from_code(code) for code in batch["func_code_string"]]
    cleaned_summaries = []
    
    for summary in batch["func_documentation_string"]:
        if summary:
            # Segment into semantic sentences and pick the first non-empty sentence
            sentences = sent_tokenize(summary.strip())
            first_sentence = sentences[0].strip().lower() if sentences else ""
            cleaned_summaries.append(first_sentence)
        else:
            cleaned_summaries.append("")
            
    return {"code": cleaned_codes, "summary": cleaned_summaries}


def filter_empty(example):
    """
    Discards samples where either the code or summary are empty or too short.
    """
    return len(example["code"]) > 10 and len(example["summary"]) > 5


def load_and_prepare_data(debug=False, debug_size=1000):
    """
    Downloads CodeSearchNet (Python), normalizes text, strips docstring leakage,
    shuffles the training split, and returns train, validation, and test subsets.
    """
    print("Loading CodeSearchNet dataset (language: Python)...")
    dataset = load_dataset("code-search-net/code_search_net", "python")

    print("Cleaning and normalizing text fields across splits...")
    columns_to_remove = dataset["train"].column_names
    
    # Process in batches for faster parallel execution
    dataset = dataset.map(
        clean_batch,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Normalizing text and removing docstrings"
    )

    print("Filtering invalid or overly short sequences...")
    dataset = dataset.filter(filter_empty, desc="Filtering samples")

    # Shuffle training set to prevent contiguous batches from the same repository
    print("Shuffling training split...")
    dataset["train"] = dataset["train"].shuffle(seed=42)

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
    print("Executing standalone smoke test for data_loader...")
    train, val, test = load_and_prepare_data(debug=True, debug_size=50)
    
    print(f"\nSubset Sizes -> Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    print("\n--- SAMPLE CODE (WITHOUT DOCSTRING) ---")
    print(train[0]["code"][:250], "\n...")
    print("--- SAMPLE SUMMARY (FIRST SENTENCE) ---")
    print(train[0]["summary"])