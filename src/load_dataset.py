from datasets import load_dataset

DATASET_NAME = "code_search_net"
LANGUAGE = "python"

def load_codesearchnet():
    """
    Load the Python portion of the CodeSearchNet dataset.
    """
    dataset = load_dataset(DATASET_NAME, LANGUAGE)
    return dataset
