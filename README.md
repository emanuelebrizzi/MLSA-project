# MLSA Project: Python Code Summarization

This repository contains an end-to-end Machine Learning pipeline for **Source Code Summarization**. The goal of this project is to automatically generate natural language summaries (docstrings) from Python code snippets using a Sequence-to-Sequence (Seq2Seq) neural network architecture.

The project leverages **CodeT5-small** (an Encoder-Decoder Transformer by Salesforce) and is fine-tuned on the Python subset of the **CodeSearchNet** dataset.

## Project Structure

The repository follows a strict modular architecture to separate data processing, model definition, execution scripts, and configurations:

```text
project/
├── data/
│   ├── data_loader.py     # Downloads, cleans, and splits the CodeSearchNet dataset
│   └── preprocess.py      # Handles tokenization and inputs formatting
├── scripts/
│   ├── train.py           # Main training loop with Seq2SeqTrainer and CSV logging
│   ├── evaluation.py      # Evaluation script computing BLEU, ROUGE, and Perplexity
│   ├── summarize.py       # Inference script for real-time code summarization
│   └── metrics.py         # Custom evaluate metrics and qualitative callbacks
├── configs/
│   ├── debug.yaml         # Debugging configuration for local smoke tests
│   ├── colab.yaml         # Standard configuration for Colab training
│   └── ...                # Other YAML configs for ablation studies
├── checkpoints/           # (Git-ignored) Saved model weights and final models
├── logs/                  # CSV logs tracking experiment metrics and Git hashes
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

The project is entirely controlled via YAML configuration files.

### 1. Training
To train the model, run the training script specifying a configuration file. It will automatically load the data, initialize the tokenizer and model from Hugging Face, and save the output in the`checkpoints/` directory. It also automatically `logs/` the run metrics to a CSV file in the logs/ directory.

```bash
python scripts/train.py --config configs/debug.yaml
```

### 2. Evaluation
To evaluate a trained checkpoint on the test set and compute final unbiased metrics (Cross-Entropy Loss, Perplexity, BLEU, and ROUGE-1/2/L):

```bash
python scripts/evaluation.py --checkpoint checkpoints/debug_run/final_model --config configs/debug.yaml
```


### 3. Summarization
You can test the model dynamically on unseen code snippets. The script accepts either a direct string (`--input`) or a Python file (`--file`).

```bash
# Using a string directly:
python scripts/summarize.py --checkpoint checkpoints/debug_run/final_model --input "def add(a, b): return a + b"

# Using a python file:
python scripts/summarize.py --checkpoint checkpoints/debug_run/final_model --file script.py
```

### Note for Google Colab Users
Google Colab comes with a pre-installed version of `peft` that causes dependency conflicts with the `accelerate` library. If running in a Colab notebook, uninstall it before installing the requirements:

```bash
!pip uninstall -y peft
!pip install -r requirements.txt
```