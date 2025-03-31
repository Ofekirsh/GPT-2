"""
Transformer Attention Visualization Script.

This script loads a trained transformer language model and visualizes
the attention patterns for generated text.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from transformer import TransformerLM
import data_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(file_type: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        file_type: Dataset identifier (e.g., "heb" or "shake")

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = Path(f"conf/conf_{file_type}.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def initialize_model(config: Dict[str, Any], vocab_size: int) -> TransformerLM:
    """
    Initialize the transformer language model based on config.

    Args:
        config: Dictionary containing model configuration
        vocab_size: Size of the vocabulary

    Returns:
        Initialized TransformerLM model
    """
    seq_len = config["seq_len"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4

    logger.info("Initializing model with %d layers, %d heads, %d embedding dimensions",
                n_layers, n_heads, embed_size)

    return TransformerLM(
        n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        max_context_len=seq_len,
        vocab_size=vocab_size,
        mlp_hidden_size=mlp_hidden_size,
        with_residuals=True,
    )


def plot_attention_matrix(
        attention_matrix: np.ndarray,
        layer_num: int,
        head_num: int,
        token_num: int,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot attention matrix as a heatmap.

    Args:
        attention_matrix: 2D numpy array containing attention weights
        layer_num: Index of the transformer layer (0-indexed)
        head_num: Index of the attention head (0-indexed)
        token_num: Index of the token position (0-indexed)
        save_path: Optional path to save the plot instead of displaying
    """
    # Ensure the matrix is 2D
    attention_matrix = np.squeeze(attention_matrix)
    if attention_matrix.ndim == 1:
        attention_matrix = attention_matrix[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attention_matrix, cmap='viridis', annot=False, cbar=True)

    plt.title(f'Layer {layer_num + 1}, Head {head_num + 1}, Token {token_num + 1} Attention')
    plt.xlabel('Attended Token Position')
    plt.ylabel('Querying Token Position')

    # Add grid lines for clarity
    ax.set_xticks(np.arange(0, attention_matrix.shape[1], 1))
    ax.set_yticks(np.arange(0, attention_matrix.shape[0], 1))
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved attention plot to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_model_attention(
        model: TransformerLM,
        tokenizer: Any,
        prompt: str = "Hello",
        max_tokens: int = 50,
        temperature: float = 0.5,
        top_k: int = 5,
        layers_to_plot: List[int] = None,
        heads_to_plot: List[int] = None,
        tokens_to_plot: List[int] = None,
        output_dir: Optional[Path] = None
) -> Tuple[str, List[torch.Tensor]]:
    """
    Generate text from the model and visualize attention patterns.

    Args:
        model: The transformer language model
        tokenizer: Tokenizer used for encoding/decoding text
        prompt: Starting text for generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Number of top tokens to consider in sampling
        layers_to_plot: Specific layers to visualize (0-indexed)
        heads_to_plot: Specific attention heads to visualize (0-indexed)
        tokens_to_plot: Specific token positions to visualize (0-indexed)
        output_dir: Directory to save plots (if None, plots are displayed)

    Returns:
        Tuple containing the generated text and attention matrices
    """
    logger.info(f"Generating text from prompt: '{prompt}'")

    # Default parameters if not specified
    if layers_to_plot is None:
        layers_to_plot = [0]
    if heads_to_plot is None:
        heads_to_plot = [0]
    if tokens_to_plot is None:
        tokens_to_plot = range(5)

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate text and collect attention
    sample_prefix = tokenizer.tokenize(prompt)

    with torch.no_grad():
        sample_sequence, all_attentions = model.better_sample_continuation(
            prefix=sample_prefix,
            max_tokens_to_generate=max_tokens,
            temperature=temperature,
            topK=top_k
        )

    # Decode generated text
    sampled_text = tokenizer.detokenize(sample_sequence)
    logger.info(f"Generated text: '{sampled_text}'")

    # Plot attention matrices for selected layers, heads, and tokens
    for token_idx in tokens_to_plot:
        if token_idx >= len(all_attentions):
            logger.warning(f"Token index {token_idx} out of range. Only {len(all_attentions)} tokens available.")
            continue

        for layer_idx in layers_to_plot:
            if layer_idx >= len(all_attentions[0]):
                logger.warning(f"Layer index {layer_idx} out of range. Model has {len(all_attentions[0])} layers.")
                continue

            for head_idx in heads_to_plot:
                if head_idx >= all_attentions[0][0].shape[0]:
                    logger.warning(
                        f"Head index {head_idx} out of range. Layer has {all_attentions[0][0].shape[0]} heads.")
                    continue

                # Extract attention matrix
                attention_matrix = all_attentions[token_idx][layer_idx][head_idx].cpu().numpy()

                # Ensure matrix is 2D
                if attention_matrix.ndim == 3:
                    attention_matrix = attention_matrix[0]

                # Determine save path if output directory provided
                save_path = None
                if output_dir:
                    save_path = output_dir / f"attention_layer{layer_idx + 1}_head{head_idx + 1}_token{token_idx + 1}.png"

                # Plot the attention matrix
                plot_attention_matrix(attention_matrix, layer_idx, head_idx, token_idx, save_path)

    return sampled_text, all_attentions


def main():
    """Main entry point for the attention visualization script."""
    # Load configuration
    file_type = "heb"  # Change to "shake" for Shakespeare dataset
    config = load_config(file_type)

    # Load data and tokenizer
    logger.info(f"Loading data from {config['data_path']}")
    tokenizer, tokenized_data = data.load_data(config["data_path"])

    # Initialize model
    model = initialize_model(config, tokenizer.vocab_size())

    # Load trained model weights
    model_path = Path('model_weights.pth')
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    logger.info(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Visualization parameters
    visualization_params = {
        'prompt': "Hello",
        'max_tokens': 50,
        'temperature': 0.5,
        'top_k': 5,
        'layers_to_plot': [0, 1],  # Plot first and second layers
        'heads_to_plot': [0, 1],  # Plot first and second heads
        'tokens_to_plot': range(5),  # Plot first 5 tokens
        'output_dir': Path('attention_plots')  # Save plots to this directory
    }

    # Generate text and visualize attention
    visualize_model_attention(model, tokenizer, **visualization_params)

    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()