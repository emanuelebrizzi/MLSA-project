# MLSA Project: Python Code Summarization

This repository contains an end-to-end Machine Learning pipeline for **Source Code Summarization**. The goal of this project is to automatically generate natural language summaries (docstrings) from Python code snippets using a Sequence-to-Sequence (Seq2Seq) neural network architecture.

The project leverages **CodeT5-small** (an Encoder-Decoder Transformer by Salesforce) and is fine-tuned on the Python subset of the **CodeSearchNet** dataset.

## Project Structure

The repository follows a strict modular architecture to separate data processing, model definition, execution scripts, and configurations:

```text
project/
├── data/
│   ├── loader.py          # Downloads, cleans, and splits the CodeSearchNet dataset
│   └── preprocess.py      # Handles tokenization and inputs formatting
├── models/
├── scripts/
│   ├── train.py           # Main training loop with Seq2SeqTrainer
│   ├── eval.py            # Evaluation script computing BLEU, ROUGE, and Perplexity
│   ├── summarize.py       # Inference script for real-time code summarization
│   └── metrics.py         # Custom evaluate metrics and qualitative callbacks
├── configs/
│   └── base.yaml          # Centralized hyperparameters and settings
├── checkpoints/           # (Git-ignored) Saved model weights and final models
├── README.md
└── requirements.txt
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/emanuelebrizzi/MLSA-project.git
cd MLSA-project
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

The project is controlled via the `configs/base.yaml` file.

### 1. Training
To train the model, simply run the training script. It will automatically load the data, initialize the model from `models/builder.py`, and save the output in the `checkpoints/` directory.

```bash
python scripts/train.py
```

### 2. Evaluation

To evaluate a trained checkpoint on the test set and compute metrics (Cross-Entropy Loss, Perplexity, BLEU, and ROUGE-1/2/L):

```bash
python scripts/eval.py --checkpoint checkpoints/final_model
```


### 3. Summarization
You can test the model dynamically on unseen code snippets. The script accepts either a direct string or a Python file.

```bash
python scripts/eval.py --checkpoint checkpoints/final_model
```