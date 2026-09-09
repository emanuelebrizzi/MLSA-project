import argparse
import sys
from scripts.train import train
# from scripts.summarize import run_inference (if you decide to create a similar function)

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
        # Here you could call your main inference function
        # e.g.: run_summarization_pipeline()
        print("(Make sure you have implemented the entry function in scripts/summarize.py!)")
        
    elif args.mode == 'evaluate':
        print("Starting model evaluation (BLEU/ROUGE)...")
        # Here you could call the evaluation script
        
    else:
        print("Unrecognized mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()