import argparse
import sys
from train import train
from summarize import run_inference  # <-- IMPORT AGGIUNTO QUI

def main():
    parser = argparse.ArgumentParser(description="Code Summarization Project - Transformer Seq2Seq")
    
    # Define command-line arguments
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'summarize', 'evaluate'],
        help="Choose the operation to perform: 'train', 'summarize', or 'evaluate'."
    )
    
    args = parser.parse_args()
    
    # Route execution based on the chosen mode
    if args.mode == 'train':
        print("Starting the training pipeline...")
        train()
        
    elif args.mode == 'summarize':
        print("Starting the inference mode...")
        run_inference()  # <-- CHIAMATA ALLA FUNZIONE INVECE DEL PRINT
        
    elif args.mode == 'evaluate':
        print("Starting model evaluation (BLEU/ROUGE)...")
        # Here you could call the evaluation script
        
    else:
        print("Unrecognized mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()