"""
Transformer Language Model Implementation.

This module implements a decoder-only transformer architecture
for language modeling tasks.
"""
from typing import List, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F

import gpt2.model.attention as attention
import gpt2.model.mlp as mlp


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block with self-attention and feed-forward layers.

    Implements a standard transformer decoder block with causal self-attention,
    layer normalization, and a feed-forward network.
    """

    def __init__(
            self,
            n_heads: int,
            embed_size: int,
            mlp_hidden_size: int,
            max_context_len: int,
            with_residuals: bool = False
    ):
        """
        Initialize a transformer decoder block.

        Args:
            n_heads: Number of attention heads
            embed_size: Dimensionality of embeddings
            mlp_hidden_size: Hidden size of the MLP layer
            max_context_len: Maximum sequence length
            with_residuals: Whether to use residual connections
        """
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(
            embed_size, n_heads, max_context_len
        )
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder block.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, embed_size)

        Returns:
            tuple: (output tensor, attention weights)
        """
        if self.with_residuals:
            # Residual connection for the attention layer
            x = inputs
            x = self.layer_norm_1(x)
            attn_output, attn_weights = self.causal_attention(x)
            x = x + self.dropout1(attn_output)  # Add residual connection

            # Residual connection for the MLP layer
            x = self.layer_norm_2(x)
            mlp_output = self.mlp(x)
            x = x + self.dropout2(mlp_output)  # Add residual connection

            return x, attn_weights
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x, attn_weights = self.causal_attention(x)
            x = self.layer_norm_2(self.dropout1(x))
            x = self.dropout2(self.mlp(x))
            return x, attn_weights


class Embed(nn.Module):
    """
    Combined token and positional embedding layer.

    Maps token indices to embeddings and adds positional embeddings.
    """

    def __init__(self, vocab_size: int, embed_size: int, max_context_len: int):
        """
        Initialize the embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            embed_size: Dimensionality of embeddings
            max_context_len: Maximum sequence length
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the combined token and position embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
               Each item is an integer indicating a vocabulary item.

        Returns:
            Combined token and position embeddings of shape (batch_size, seq_len, embed_size)

        Raises:
            ValueError: If the sequence length exceeds max_context_len
        """
        batch_size, seq_len = x.shape

        # Ensure the sequence length does not exceed the maximum context length
        if seq_len > self.max_context_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum context length {self.max_context_len}"
            )

        # Token embeddings
        tok_embeddings = self.token_embeddings(x)  # Shape (batch_size, seq_len, embed_size)

        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.position_embeddings(positions)  # Shape (batch_size, seq_len, embed_size)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    """
    Transformer-based language model.

    A decoder-only transformer architecture for autoregressive language modeling.
    """

    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the transformer language model.

        Args:
            n_layers: Number of transformer decoder layers
            n_heads: Number of attention heads per layer
            embed_size: Dimensionality of embeddings
            max_context_len: Maximum sequence length
            vocab_size: Size of the vocabulary
            mlp_hidden_size: Hidden size of the MLP layers
            with_residuals: Whether to use residual connections
            device: Device to place the model on ('cpu', 'cuda', etc.)
        """
        super().__init__()

        self.embed = Embed(vocab_size, embed_size, max_context_len)

        # Create stack of transformer decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                n_heads,
                embed_size,
                mlp_hidden_size,
                max_context_len,
                with_residuals
            ) for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        # Set device
        self.device = torch.device("cpu" if device is None else device)

        # Initialize model weights
        self._init_weights()

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Parameter count: {n_params / 1e6:.2f}M")

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the transformer model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len)

        Returns:
            tuple: (logits, attention_weights)
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - attention_weights: List of attention weights from each layer
        """
        attentions = []

        # Compute embeddings
        x = self.embed(inputs)

        # Pass through transformer layers
        for layer in self.layers:
            x, attention = layer(x)
            attentions.append(attention)

        # Final layer norm and projection to vocabulary
        x = self.layer_norm(x)
        logits = self.word_prediction(x)

        return logits, attentions

    def _init_weights(self):
        """
        Initialize the model weights with appropriate distributions.

        - LayerNorm: bias=0, weight=1
        - Linear: Normal(0, 0.02) for weights, zeros for bias
        - Embedding: Normal(0, 0.02), with special handling for padding indices
        """
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                # Initialize weights with a normal distribution and bias with zeros
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                # Initialize embeddings with a normal distribution
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.padding_idx is not None:
                    torch.nn.init.constant_(p.weight[p.padding_idx], 0)

    def sample_continuation(self, prefix: List[int], max_tokens_to_generate: int) -> List[int]:
        """
        Generate a simple continuation from a prefix sequence.

        Args:
            prefix: List of token IDs to start generation from
            max_tokens_to_generate: Maximum number of tokens to generate

        Returns:
            List of generated token IDs
        """
        feed_to_lm = prefix[:]
        generated = []

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                # Truncate context if too long
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                # Get model prediction
                input_tensor = torch.tensor([feed_to_lm], dtype=torch.int32)
                logits, _ = self(input_tensor)

                # Sample from the last token's distribution
                logits_for_last_token = logits[0, -1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1).item()

                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)

        return generated

    def better_sample_continuation(
            self,
            prefix: List[int],
            max_tokens_to_generate: int,
            temperature: float,
            topK: int
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generate a continuation with temperature and top-K sampling.

        Args:
            prefix: List of token IDs to start generation from
            max_tokens_to_generate: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            topK: Number of top tokens to consider for sampling

        Returns:
            tuple: (generated_tokens, attention_weights)
        """
        feed_to_lm = prefix[:]
        generated = []
        all_attention_weights = []

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                # Truncate context if too long
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                # Move input to device and get model prediction
                input_tensor = torch.tensor([feed_to_lm], dtype=torch.long).to(self.device)
                logits, att_weights = self(input_tensor)
                all_attention_weights.append(att_weights)

                # Apply temperature to logits
                logits_for_last_token = logits[0, -1] / temperature

                # Sample from top-K tokens
                p = F.softmax(logits_for_last_token, dim=-1)
                p_topk, top_k_indices = torch.topk(p, topK)
                sampled_index = torch.multinomial(p_topk, num_samples=1).item()
                sampled_token = top_k_indices[sampled_index].item()

                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)

        return generated, all_attention_weights