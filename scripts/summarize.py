import os
import sys
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_args():
    """
    Parses command-line arguments for running inference.
    Supports both direct string input and file input.
    """
    parser = argparse.ArgumentParser(description="Generate summaries for Python code snippets.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint"
    )
    
    # Mutually exclusive group: provide either --input or --file, not both
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", 
        type=str, 
        help="A Python code snippet as a string"
    )
    group.add_argument(
        "--file", 
        type=str, 
        help="Path to a Python file to summarize"
    )
    
    return parser.parse_args()

def generate_summary(code_snippet, model, tokenizer, device):
    """
    Takes a string of Python code and generates a natural language summary.
    """
    # 1. Tokenize the input code
    inputs = tokenizer(
        code_snippet, 
        return_tensors="pt", 
        max_length=256, 
        truncation=True
    ).to(device)
    
    # 2. Generate output tokens using Beam Search for better quality
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,            # Explores multiple paths to find the best summary
            length_penalty=1.0,     # Balances the length of the output
            early_stopping=True
        )
        
    # 3. Decode the generated tokens back into a readable string
    summary = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return summary

def main():
    args = parse_args()
    
    # Determine the execution device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.eval() # Set to evaluation mode to disable dropout layers
    
    # Extract the code to summarize
    code_to_summarize = ""
    if args.input:
        code_to_summarize = args.input
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            code_to_summarize = f.read()
            
    print("\n--- INPUT CODE ---")
    print(code_to_summarize)
    print("------------------\n")
    
    # Generate and print the summary
    summary = generate_summary(code_to_summarize, model, tokenizer, device)
    
    print("--- GENERATED SUMMARY ---")
    print(summary)
    print("-------------------------")

if __name__ == "__main__":
    main()