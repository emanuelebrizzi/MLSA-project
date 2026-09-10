import pandas as pd
import os

def load_data(file_path="data/real_dataset.csv"):
    """
    Loads the real dataset downloaded from CodeSearchNet.
    """
    if os.path.exists(file_path):
        print(f"Loading real dataset from {file_path}...")
        df = pd.read_csv(file_path)
        # Drop any missing values just to be safe
        df = df.dropna(subset=['code', 'summary'])
        return df
    else:
        raise FileNotFoundError(f"Dataset not found in {file_path}. Run the download script first.")