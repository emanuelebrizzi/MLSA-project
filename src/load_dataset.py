import pandas as pd
import os

def load_data(file_path=None):
    """
    Loads the dataset from a file. If no path is provided,
    it returns a small dummy DataFrame for local testing.
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        # Assuming the data is saved in CSV or Parquet format
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
    
    print("No file found or provided. Generating dummy dataset...")
    return pd.DataFrame({
        'code': [
            'def add(a, b):\n    return a + b',
            'def sub(a, b):\n    return a - b',
            'def multiply(x, y):\n    return x * y'
        ],
        'summary': [
            'adds two numbers together',
            'subtracts the second number from the first',
            'multiplies two values'
        ]
    })