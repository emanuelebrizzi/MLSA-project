import re

def clean_text(text):
    """
    Cleans a generic text string (useful for docstrings/summaries).
    """
    if not isinstance(text, str):
        return ""
    
    # Removes multiple spaces and line breaks
    text = re.sub(r'\s+', ' ', text)
    # Removes unnecessary special characters while keeping basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    return text.strip().lower()

def clean_code(code_str):
    """
    Cleans the code snippet. 
    Keeps the basic structure but removes excess whitespace.
    """
    if not isinstance(code_str, str):
        return ""
    
    # Replaces tabs with single spaces
    code_str = code_str.replace('\t', ' ')
    # Removes multiple empty spaces, but keeps a single space to separate tokens
    code_str = re.sub(r'\s+', ' ', code_str)
    return code_str.strip()

def preprocess_dataframe(df):
    """
    Applies the cleaning functions to the entire DataFrame.
    """
    df_clean = df.copy()
    
    # Assuming the columns are named 'code' and 'summary'
    df_clean['code'] = df_clean['code'].apply(clean_code)
    df_clean['summary'] = df_clean['summary'].apply(clean_text)
    
    # Removes any rows left empty after cleaning
    df_clean = df_clean[(df_clean['code'] != '') & (df_clean['summary'] != '')]
    df_clean = df_clean.dropna()
    
    return df_clean