# MLSA Project: Python Code Summarization

This repository contains an end-to-end Machine Learning pipeline for **Source Code Summarization**. The goal of this project is to automatically generate natural language summaries (docstrings) from Python code snippets using a Sequence-to-Sequence (Seq2Seq) neural network architecture.

The project leverages **CodeT5-small** (an Encoder-Decoder Transformer by Salesforce) and is fine-tuned on the Python subset of the **CodeSearchNet** dataset.

## Project Structure

The repository follows a strict modular architecture to separate data processing, model definition, execution scripts, and configurations:

```text
project/
├── data/
│   ├── __init__.py
│   ├── loader.py          # Downloads, cleans, and splits the CodeSearchNet dataset
│   └── preprocess.py      # Handles tokenization and inputs formatting
├── models/
│   ├── __init__.py
│   └── builder.py         # Defines and initializes the Seq2Seq architecture
├── scripts/
│   ├── __init__.py
│   ├── train.py           # Main training loop with Seq2SeqTrainer
│   ├── eval.py            # Evaluation script computing BLEU, ROUGE, and Perplexity
│   ├── summarize.py       # Inference script for real-time code summarization
│   └── metrics.py         # Custom evaluate metrics and qualitative callbacks
├── configs/
│   └── base.yaml          # Centralized hyperparameters and settings
├── checkpoints/           # (Git-ignored) Saved model weights and final models
├── README.md
└── requirements.txt
