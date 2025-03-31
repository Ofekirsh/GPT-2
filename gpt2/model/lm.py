from __future__ import annotations
import torch
import torch.nn.functional as F


def batch_to_labeled_samples(batch: torch.LongTensor) -> [torch.IntTensor, torch.IntTensor]:
    """
        Translates the input data to labeled data for language modeling.

        Args:
            batch (torch.IntTensor): Input tensor of shape (b, n), where b is the batch size and n is the sequence length.

        Returns:
            tuple: Two tensors, each of shape (b, n-1). The first tensor is the input sequence, and the second tensor is the labels.

        example:
        Batch:
        tensor([[ 1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10]], dtype=torch.int32)
        Inputs (batch[:, :-1]):
        tensor([[1, 2, 3, 4],
                [6, 7, 8, 9]], dtype=torch.int32)
        Labels (batch[:, 1:]):
        tensor([[ 2,  3,  4,  5],
                [ 7,  8,  9, 10]], dtype=torch.int32)

    """
    # The input sequence is the batch without the last token
    inputs = batch[:, :-1]

    # The label sequence is the batch without the first token
    labels = batch[:, 1:]

    return inputs, labels


def compute_loss(logits, gold_labels):
    """
        Computes the loss between the predicted logits and the gold labels.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch, seq_len, vocab_size).
            gold_labels (torch.Tensor): Gold labels of shape (batch, seq_len).
        Returns:
            torch.Tensor: Computed loss.
    """
    # Reshape logits to (batch * seq_len, vocab_size) for cross-entropy
    b, n, v = logits.size()
    logits = logits.reshape(-1, v)

    # Reshape gold labels to (batch * seq_len)
    gold_labels = gold_labels.reshape(-1)

    # Compute the loss, ignoring the padding index
    loss = F.cross_entropy(logits, gold_labels, ignore_index=0)

    return loss
