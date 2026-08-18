from datasets import DatasetDict

from load_dataset import load_codesearchnet

COLUMNS = ["code", "docstring"]

# Minimum and maximum character lengths
MIN_CODE_LENGTH = 10
MAX_CODE_LENGTH = 10000

MIN_DOCSTRING_LENGTH = 3
MAX_DOCSTRING_LENGTH = 1000

def clean_code(code):
    """
    Basic cleaning of source code.

    We only remove leading/trailing whitespace and preserve
    the internal formatting of the code.
    """
    if code is None:
        return ""

    return code.strip()

def clean_docstring(docstring):
    """
    Clean a docstring by removing leading/trailing whitespace
    and normalizing consecutive whitespace characters.
    """
    if docstring is None:
        return ""

    docstring = docstring.strip()
    docstring = " ".join(docstring.split())

    return docstring

def preprocess_example(example):
    """
    Preprocess a single dataset example.
    """
    code = clean_code(example["code"])
    docstring = clean_docstring(example["docstring"])

    return {
        "code": code,
        "docstring": docstring,
    }

def is_valid_example(example):
    """
    Check whether an example satisfies the length constraints.
    """
    code = example["code"]
    docstring = example["docstring"]

    if not code or not docstring:
        return False

    if not MIN_CODE_LENGTH <= len(code) <= MAX_CODE_LENGTH:
        return False

    if not MIN_DOCSTRING_LENGTH <= len(docstring) <= MAX_DOCSTRING_LENGTH:
        return False

    return True

def preprocess_dataset(dataset):
    """
    Apply preprocessing to all dataset splits.
    """

    # Keep only the columns required for code summarization.
    dataset = dataset.select_columns(COLUMNS)

    # Clean code and docstrings.
    dataset = dataset.map(preprocess_example)

    # Remove missing, empty, or excessively long examples.
    dataset = dataset.filter(is_valid_example)

    return dataset