"""
Transformer Language Model Training Script.

This script handles the training of a Transformer-based language model
using either Hebrew or Shakespeare datasets.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from gpt2.model.transformer import TransformerLM
import gpt2.data_loader as data
import gpt2.model.lm as lm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file.

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(f"conf/conf.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def train_model(
        model: torch.nn.Module,
        data_iterator: Iterator,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_batches_to_train: int,
        gradient_clipping: float,
        tokenizer: Any
) -> None:
    """Train the language model.

    Args:
        model: The transformer language model
        data_iterator: Iterator providing training data
        optimizer: Optimizer for parameter updates
        batch_size: Number of sequences per batch
        num_batches_to_train: Total number of batches to train on
        gradient_clipping: Maximum gradient norm for clipping
        tokenizer: Tokenizer for text generation during training
    """
    model.train()

    num_batches = 0
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(data_iterator, batch_size):
            if num_batches >= num_batches_to_train:
                break

            num_batches += 1

            # Prepare input and target sequences
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            # Forward pass
            logits, _ = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            # Backward pass and optimization
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            # Logging and sample generation
            if num_batches % 10 == 0:
                logger.info(f"Seen {num_batches} batches. Last loss: {loss.item():.4f}")

                if num_batches % 100 == 0:
                    generate_samples(model, tokenizer, num_samples=1)


def generate_samples(model: torch.nn.Module, tokenizer: Any, num_samples: int = 1) -> None:
    """Generate and display text samples from the model.

    Args:
        model: The trained transformer model
        tokenizer: Tokenizer for text conversion
        num_samples: Number of samples to generate
    """
    model.eval()

    for _ in range(num_samples):
        prompt_tokens = tokenizer.tokenize("Hello")
        sample_sequence, _ = model.better_sample_continuation(
            prompt_tokens,
            max_tokens=500,
            temperature=0.5,
            topK=5
        )

        sampled_text = tokenizer.detokenize(sample_sequence)
        logger.info(f"Model sample: '''{sampled_text}'''")

    print("")  # Extra line for separation
    model.train()


def main() -> None:
    """Main entry point for the training script."""
    # Select dataset type - "heb" for Hebrew or "shake" for Shakespeare

    # Load configuration
    config = load_config()

    # Extract configuration parameters
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4
    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    num_batches_to_train = config["num_batches_to_train"]
    weight_decay = config["weight_decay"]

    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    tokenizer, tokenized_data = data.load_data(data_path)

    # Create data iterator
    # Note: Data items are longer by one than the sequence length
    # They will be shortened by 1 when converted to training examples
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    # Initialize the model
    logger.info("Initializing Transformer Language Model")
    model = TransformerLM(
        n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        max_context_len=seq_len,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=mlp_hidden_size,
        with_residuals=True,
    )

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    # Train the model
    logger.info(f"Starting training for {num_batches_to_train} batches")
    train_model(
        model=model,
        data_iterator=data_iter,
        optimizer=optimizer,
        batch_size=batch_size,
        num_batches_to_train=num_batches_to_train,
        gradient_clipping=gradient_clipping,
        tokenizer=tokenizer
    )

    # Save the model
    model_path = Path('model_weights.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path.absolute()}")


if __name__ == '__main__':
    main()