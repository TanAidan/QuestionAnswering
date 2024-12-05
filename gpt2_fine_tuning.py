import os
import time
import datetime
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup
)
import wandb
import nltk

@dataclass
class TrainingConfig:
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

class GPT2Dataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        max_length: int
    ):
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            encodings_dict = tokenizer(
                '<|startoftext|>' + text + '<|endoftext|>',
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attn_masks[idx]

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def format_time(elapsed: float) -> str:
    """Format elapsed time as string."""
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def prepare_tokenizer(model_name: str) -> GPT2Tokenizer:
    """Initialize and prepare the tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name,
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>'
    )
    return tokenizer

def prepare_model(model_name: str, tokenizer: GPT2Tokenizer) -> GPT2LMHeadModel:
    """Initialize and prepare the model."""
    configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    return model

def prepare_data(
    config: TrainingConfig,
    tokenizer: GPT2Tokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders."""
    # Load and preprocess data
    df = pd.read_csv("squad_csv.csv")
    df.dropna(inplace=True)
    questions = df.question.tolist()

    # Create dataset
    dataset = GPT2Dataset(questions, tokenizer, config.max_length)

    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader

def train_epoch(
    model: GPT2LMHeadModel,
    train_dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    config: TrainingConfig,
    device: torch.device
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    t0 = time.time()

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_masks = [b.to(device) for b in batch]

        model.zero_grad()
        outputs = model(
            b_input_ids,
            labels=b_input_ids,
            attention_mask=b_masks,
            token_type_ids=None
        )
        loss = outputs[0]
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        wandb.log({"loss": loss.item()})

        if step % config.sample_every == 0 and step > 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.'
                  f' Loss: {loss.item():>5,}.   Elapsed: {elapsed}.')

    return total_loss / len(train_dataloader)

def validate(
    model: GPT2LMHeadModel,
    val_dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validate the model and return average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_masks = [b.to(device) for b in batch]
            
            outputs = model(
                b_input_ids,
                attention_mask=b_masks,
                labels=b_input_ids
            )
            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

def save_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    config: TrainingConfig,
    output_dir: str
):
    """Save the model, tokenizer, and training arguments."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training arguments
    torch.save(vars(config), os.path.join(output_dir, 'training_args.bin'))

def main():
    # Initialize wandb
    run = wandb.init(project="Transformer-Question-Answering", entity="aidan-tan")
    
    # Initialize configuration
    config = TrainingConfig()
    wandb.config.update(vars(config))
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare tokenizer and model
    tokenizer = prepare_tokenizer(config.model_name)
    model = prepare_model(config.model_name, tokenizer)
    model.to(device)
    
    # Prepare data
    train_dataloader, val_dataloader = prepare_data(config, tokenizer)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.epsilon)
    total_steps = len(train_dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    wandb.watch(model)
    training_stats = []
    total_t0 = time.time()

    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1} / {config.epochs}')
        
        # Training phase
        avg_train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, config, device
        )
        
        # Validation phase
        avg_val_loss = validate(model, val_dataloader, device)
        
        # Log statistics
        stats = {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': format_time(time.time() - total_t0)
        }
        training_stats.append(stats)
        wandb.log(stats)

    # Save model
    save_model(model, tokenizer, config, config.output_dir)
    
    # Log model artifact to wandb
    artifact = wandb.Artifact('QuestionLMV2', type="model",
                            description='trained baseline for QLM V2')
    artifact.add_dir(config.output_dir)
    run.log_artifact(artifact)

if __name__ == "__main__":
    nltk.download('punkt')
    main()
