import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from transformer import TransformerLM
import data


def plot_attention_matrix(attention_matrix, layer_num, head_num, token_num):
    attention_matrix = np.squeeze(attention_matrix)  # Ensure the matrix is 2D
    if attention_matrix.ndim == 1:
        attention_matrix = attention_matrix[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, cmap='viridis')
    plt.title(f'Layer {layer_num + 1}, Head {head_num + 1}, Token {token_num + 1} Attention')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.show()


if __name__ == '__main__':
    # Load configuration
    file_type = "heb"
    with open(f"conf/conf_{file_type}.json", "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4

    # Load tokenizer and data
    tokenizer, tokenized_data = data.load_data(data_path)

    # Load the model
    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    )

    # Load model weights
    model.load_state_dict(torch.load('model_weights.pth'))

    model.eval()

    # Generate a sample sequence and collect attention matrices
    sample_prefix = tokenizer.tokenize("Hello")
    sample_sequence, all_attentions = model.better_sample_continuation(sample_prefix, 500, temperature=0.5, topK=5)
    sampled_text = tokenizer.detokenize(sample_sequence)
    print(f"Sampled text: '''{sampled_text}'''")

    # Example: Plot the attention matrix for the first token in the generated sequence for specific layers and heads
    specific_layers = [0]  # List of specific layers to plot (0-indexed)
    specific_heads = [0]  # List of specific heads to plot (0-indexed)
    tokens_to_plot = range(0, min(10, len(all_attentions)))  # Plot attention for the first few tokens

    for token_num in tokens_to_plot:
        for layer_num in specific_layers:
            for head_num in specific_heads:
                # Ensure the attention matrix is 2D
                attention_matrix = all_attentions[token_num][layer_num][head_num].cpu().numpy()
                if attention_matrix.ndim == 3:
                    attention_matrix = attention_matrix[0]
                elif attention_matrix.ndim == 1:
                    attention_matrix = attention_matrix[:, np.newaxis]
                plot_attention_matrix(attention_matrix, layer_num, head_num, token_num)
