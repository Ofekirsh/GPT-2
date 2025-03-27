from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, inputs):
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
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        """
            Forward pass to compute the combined token and position embeddings.

            Args:
                x (torch.Tensor): Input tensor of shape (b, n), where b is the batch size and n is the sequence length.
                                  Each item in the tensor is an integer indicating a vocabulary item.

            Returns:
                torch.Tensor: Output tensor of shape (b, n, d), where d is the embedding dimension, containing the
                              combined token and position embeddings.

            Example:
                >>> vocab_size = 10
                >>> embed_size = 4
                >>> max_context_len = 6
                >>> embed = Embed(vocab_size, embed_size, max_context_len)
                >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])

                # Assume the following embedding weights for demonstration
                >>> embed.token_embeddings.weight = nn.Parameter(torch.tensor([
                ...     [0.0, 0.0, 0.0, 0.0],  # Token 0
                ...     [0.1, 0.2, 0.3, 0.4],  # Token 1
                ...     [0.4, 0.5, 0.6, 0.7],  # Token 2
                ...     [0.7, 0.8, 0.9, 1.0],  # Token 3
                ...     [1.0, 1.1, 1.2, 1.3],  # Token 4
                ...     [1.3, 1.4, 1.5, 1.6],  # Token 5
                ...     [1.6, 1.7, 1.8, 1.9],  # Token 6
                ...     [1.9, 2.0, 2.1, 2.2],  # Token 7
                ...     [2.2, 2.3, 2.4, 2.5],  # Token 8
                ...     [2.5, 2.6, 2.7, 2.8],  # Token 9
                ... ]))

                >>> embed.position_embeddings.weight = nn.Parameter(torch.tensor([
                ...     [0.01, 0.02, 0.03, 0.04],  # Position 0
                ...     [0.04, 0.05, 0.06, 0.07],  # Position 1
                ...     [0.07, 0.08, 0.09, 0.10],  # Position 2
                ...     [0.10, 0.11, 0.12, 0.13],  # Position 3
                ...     [0.13, 0.14, 0.15, 0.16],  # Position 4
                ...     [0.16, 0.17, 0.18, 0.19],  # Position 5
                ... ]))
                # Token embeddings lookup
                >>> token_embeddings = embed.token_embeddings(x)
                >>> print(token_embeddings)
                tensor([[
                  [0.1, 0.2, 0.3, 0.4],
                  [0.4, 0.5, 0.6, 0.7],
                  [0.7, 0.8, 0.9, 1.0]],
                 [[1.0, 1.1, 1.2, 1.3],
                  [1.3, 1.4, 1.5, 1.6],
                  [1.6, 1.7, 1.8, 1.9]]])

                >>> position_embeddings = embed.position_embeddings(positions)
                >>> print(position_embeddings)
                tensor([[
                  [0.01, 0.02, 0.03, 0.04],
                  [0.04, 0.05, 0.06, 0.07],
                  [0.07, 0.08, 0.09, 0.10]],
                 [[0.01, 0.02, 0.03, 0.04],
                  [0.04, 0.05, 0.06, 0.07],
                  [0.07, 0.08, 0.09, 0.10]]])

                # Combined embeddings
                >>> embeddings = token_embeddings + position_embeddings
                >>> print(embeddings)
                tensor([[
                  [0.11, 0.22, 0.33, 0.44],
                  [0.44, 0.55, 0.66, 0.77],
                  [0.77, 0.88, 0.99, 1.10]],
                 [[1.01, 1.12, 1.23, 1.34],
                  [1.34, 1.45, 1.56, 1.67],
                  [1.67, 1.78, 1.89, 2.00]]])
                >>> print(embeddings.shape)
                torch.Size([2, 3, 4])
            """
        b, n = x.shape

        # Ensure the sequence length does not exceed the maximum context length
        if n > self.max_context_len:
            raise ValueError(f"Sequence length {n} exceeds maximum context length {self.max_context_len}")

        # Token embeddings
        tok_embeddings = self.token_embeddings(x)  # Shape (b, n, d)

        # Position embeddings
        positions = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)  # Shape (b, n)
        pos_embeddings = self.position_embeddings(positions)  # Shape (b, n, d)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            device=None):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        attentions = []
        x = self.embed(inputs)
        for layer in self.layers:
            x, attention = layer(x)
            attentions.append(attention)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits, attentions

    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
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


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                logits, att_weights = self(torch.tensor([feed_to_lm], dtype=torch.long).to(self.device))
                # get the logits divided by the temperature
                logits_for_last_token = logits[0, -1] / temperature

                p = F.softmax(logits_for_last_token, dim=-1)
                p_topk, top_k_indices = torch.topk(p, topK)
                sampled_index = torch.multinomial(p_topk, num_samples=1).item()
                sampled_token = top_k_indices[sampled_index]
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated, att_weights

