from torch import nn
import torch
import torch.nn.functional as F


def create_kqv_matrix(input_vector_dim, n_heads=1):
    """
        Creates the linear layers for computing k, q, and v matrices for each head.

        Args:
        input_vector_dim (int): The dimension of the input vector.
        n_heads (int): Number of attention heads (default is 1).

        Returns:
        nn.Linear: Linear layer for computing k, q, and v.
        """
    return nn.Linear(input_vector_dim, input_vector_dim * 3 // n_heads)


def kqv(x, linear):
    """
        Computes the key (k), query (q), and value (v) matrices from input x using a linear transformation.

        Args:
        x (torch.Tensor): Input tensor of shape (B, N, D).
        linear (torch.nn.Linear): Linear layer to transform input x.

        Returns:
        tuple: k, q, v matrices each of shape (B, N, D/3).

        Formula:
        k, q, v = split(linear(x))
    """
    kqv = linear(x)  # Apply linear transformation to input x to get k, q, and v
    k, q, v = torch.chunk(kqv, 3, dim=-1)  # Split the result into k, q, and v
    return k, q, v


def attention_scores(a, b):
    """
        Computes the scaled dot-product attention scores between queries and keys.

        Args:
        a (torch.Tensor): Query matrix of shape (B, N, D).
        b (torch.Tensor): Key matrix of shape (B, N, D).

        Returns:
        torch.Tensor: Attention scores matrix A of shape (B, N, N).

        Formula:
        A = (QK^T) / sqrt(d_k)
    """
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    scaling_factor = torch.sqrt(torch.tensor(D1, dtype=torch.float32))
    A = torch.matmul(b, a.transpose(-2, -1)) / scaling_factor  # Scaled dot product
    return A


def create_causal_mask(embed_dim, n_heads, max_context_len):
    """
        Creates a causal mask to prevent attention to future positions.

        Args:
        embed_dim (int): The embedding dimension of the input (not used in mask creation).
        n_heads (int): Number of attention heads.
        max_context_len (int): Maximum length of the context.

        Returns:
        torch.Tensor: Causal mask tensor.

        Example:
        >>> embed_dim = 4
        >>> n_heads = 2
        >>> max_context_len = 5
        >>> mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        >>> print(mask)
        tensor([[[[1., 0., 0., 0., 0.],
                  [1., 1., 0., 0., 0.],
                  [1., 1., 1., 0., 0.],
                  [1., 1., 1., 1., 0.],
                  [1., 1., 1., 1., 1.]],

                 [[1., 0., 0., 0., 0.],
                  [1., 1., 0., 0., 0.],
                  [1., 1., 1., 0., 0.],
                  [1., 1., 1., 1., 0.],
                  [1., 1., 1., 1., 1.]]]])
        """
    mask = torch.tril(torch.ones(1, max_context_len, max_context_len), diagonal=0)
    return mask


def self_attention(v, A, mask=None):
    """
        Computes the self-attention output using attention scores and value matrix.

        Args:
        v (torch.Tensor): Value matrix of shape (B, N, D).
        A (torch.Tensor): Attention scores matrix of shape (B, N, N).
        mask (torch.Tensor, optional): Mask tensor of shape (B, N, N) to prevent attention to certain positions.

        Returns:
        torch.Tensor: Self-attention output of shape (B, N, D).

        Formula:
        Ã = softmax(A)
        sa = ÃV
    """
    # Apply mask if provided
    b, n, d = v.size()
    mask = mask[:, :n, :n]
    A = A.masked_fill(mask == 0, float("-inf"))
    # Normalize the attention scores
    A = F.softmax(A, dim=-1)

    # Compute the self-attention by multiplying the attention scores with the values
    sa = torch.matmul(A, v)

    return sa, A


def self_attention_layer(x, kqv_matrix, attention_mask):
    """
      Computes the self-attention layer output from input tensor x.

      Args:
      x (torch.Tensor): Input tensor of shape (B, N, D).
      kqv_matrix (torch.nn.Linear): Linear layer to compute k, q, and v matrices.
      attention_mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions.

      Returns:
      torch.Tensor: Self-attention layer output of shape (B, N, D).

      Formula:
      sa = self_attention(v, attention_scores(k, q), attention_mask)
    """
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa, A = self_attention(v, att, attention_mask)
    return sa, A


def multi_head_attention_layer(x, kqv_matrices, mask):
    # Compute the self-attention for each head and concatenate the results
    head_outputs = []
    all_attn_weights = []

    for kqv_matrix in kqv_matrices:
        head_output, attn_weights = self_attention_layer(x, kqv_matrix, mask)
        head_outputs.append(head_output)
        all_attn_weights.append(attn_weights)

    # Concatenate the head outputs along the last dimension
    sa = torch.cat(head_outputs, dim=-1)

    # Ensure the output size is the same as the input size
    assert sa.size() == x.size()
    return sa, all_attn_weights


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for _ in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa, attn_weights = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa, attn_weights
