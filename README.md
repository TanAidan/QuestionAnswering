# GPT-2 Question-Answering Model Training

This repository contains code for fine-tuning a GPT-2 model on question-answering tasks using the SQuAD dataset. The implementation uses PyTorch and the Hugging Face Transformers library, with experiment tracking via Weights & Biases.

## Features

- Fine-tune GPT-2 on custom question-answering datasets
- Configurable training parameters via dataclass
- Built-in validation during training
- Experiment tracking with Weights & Biases
- Modular and maintainable code structure
- Type hints for better code understanding
- Automated model checkpointing

## Prerequisites

```bash
# Python version
Python 3.7+

# Required packages
torch>=1.7.0
transformers>=4.0.0
wandb
pandas
numpy
nltk
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd gpt2-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

## Dataset

The code expects a CSV file named `squad_csv.csv` containing the SQuAD dataset with at least the following column:
- `question`: The question text

You can modify the data loading logic in the `prepare_data()` function to accommodate different dataset formats.

## Configuration

Training parameters can be configured by modifying the `TrainingConfig` dataclass in the code. Key parameters include:

```python
learning_rate: float = 5e-4
epochs: int = 5
batch_size: int = 32
epsilon: float = 1e-8
warmup_steps: int = 100
max_length: int = 768
train_split: float = 0.9
sample_every: int = 100
seed: int = 42
model_name: str = 'gpt2'
output_dir: str = './model_save/'
```

## Usage

1. Prepare your dataset in the required format.

2. Run the training script:
```bash
python train.py
```

3. Monitor training progress in the Weights & Biases dashboard.

The model and tokenizer will be saved in the specified `output_dir` after training.

## Code Structure

- `GPT2Dataset`: Custom dataset class for handling text data
- `TrainingConfig`: Configuration dataclass for training parameters
- `prepare_tokenizer()`: Initialize and configure the GPT-2 tokenizer
- `prepare_model()`: Initialize and configure the GPT-2 model
- `prepare_data()`: Prepare training and validation dataloaders
- `train_epoch()`: Training logic for one epoch
- `validate()`: Validation logic
- `save_model()`: Save model, tokenizer, and training arguments
- `main()`: Main training loop and orchestration

## Training Process

The training process includes:

1. Data preparation and splitting into train/validation sets
2. Model and tokenizer initialization
3. Training loop with:
   - Per-epoch training
   - Validation after each epoch
   - Progress logging to Weights & Biases
   - Model checkpointing
4. Final model saving and artifact logging

## Logging and Monitoring

The following metrics are logged to Weights & Biases:
- Training loss (per batch)
- Average training loss (per epoch)
- Validation loss (per epoch)
- Training time
- Model checkpoints as artifacts

## Model Saving

The trained model is saved with:
- Model weights and configuration
- Tokenizer configuration
- Training arguments

## Contributing

Feel free to submit issues and enhancement requests.


## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information if applicable]
```

## Acknowledgments

- Hugging Face for the Transformers library
- OpenAI for the GPT-2 model
- Stanford for the SQuAD dataset
